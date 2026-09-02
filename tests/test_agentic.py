import json

import pytest

from djcues import agentic
from djcues.constants import KIND_TO_PAD
from djcues.models import BeatGrid, CueProposal, Phrase, Track, WaveformPoint
from djcues.providers import GenerationResult, estimate_cost as price_estimate
from djcues.strategy import CueStrategy


@pytest.fixture
def beat_grid() -> BeatGrid:
    return BeatGrid(first_beat_ms=77.0, bpm=128.0)


@pytest.fixture
def phrases(beat_grid: BeatGrid) -> list[Phrase]:
    """Same World Gone Wild phrase structure the strategy tests use."""
    raw = [
        (1, 1, "Intro"),
        (33, 2, "Up"),
        (65, 2, "Up"),
        (81, 2, "Up"),
        (145, 5, "Chorus"),
        (209, 3, "Down"),
        (241, 2, "Up"),
        (273, 5, "Chorus"),
        (305, 5, "Chorus"),
        (337, 5, "Chorus"),
        (401, 5, "Chorus"),
        (433, 6, "Outro"),
    ]
    end_beat = 461
    out = []
    for i, (beat, kind, label) in enumerate(raw):
        next_beat = raw[i + 1][0] if i + 1 < len(raw) else end_beat
        pos = beat_grid.beat_to_ms(beat)
        end_pos = beat_grid.beat_to_ms(next_beat)
        out.append(Phrase(
            beat_start=beat, beat_end=next_beat, kind=kind, label=label,
            position_ms=pos, duration_ms=end_pos - pos,
        ))
    return out


@pytest.fixture
def sample_track(beat_grid: BeatGrid, phrases: list[Phrase]) -> Track:
    """Track with populated waveform + vocal_track, so build_track_payload
    has real (non-empty) energy/vocal data to condense."""
    waveform = [WaveformPoint(height=0.5, red=4, green=4, blue=4) for _ in range(200)]
    # One genuine (>=2s) vocal-onset run, plus a short (~139ms) blip that
    # should be filtered out as noise -- see test_build_track_payload_*
    # below, which pins both halves of that behavior.
    vocal_track = [0] * 50 + [3] * 50 + [0] * 80 + [3, 3, 3] + [0] * 100
    return Track(
        id=1, title="Test Track", artist="Test Artist", bpm=128.0,
        duration_ms=218000.0, analysis_path="", cues=[], phrases=phrases,
        beat_grid=beat_grid, waveform=waveform, vocal_track=vocal_track,
    )


def _gen(content: dict, input_tokens: int = 100, output_tokens: int = 20) -> GenerationResult:
    return GenerationResult(content=content, input_tokens=input_tokens, output_tokens=output_tokens)


class AuthenticationError(Exception):
    """Mirrors the name-based check in agentic._looks_like_auth_error --
    only the class name matters, not the import path."""


class _FakeProvider:
    """Routes generate_structured calls by which system prompt was used,
    since agentic.py dispatches specialists in parallel with no other way
    to tell them apart. No real network/SDK involved."""

    name = "fake"

    def __init__(self, routes: dict[str, object], token_counts: dict[str, int] | None = None):
        self._routes = routes
        self._token_counts = token_counts or {}
        self.calls: list[str] = []
        self.count_token_calls: list[tuple[str, str]] = []

    def _key(self, system: str) -> str:
        if system == agentic._STRUCTURE_SYSTEM:
            return "structure"
        if system == agentic._VOCAL_SYSTEM:
            return "vocal"
        if system == agentic._ENERGY_SYSTEM:
            return "energy"
        if system == agentic._CRITIC_SYSTEM:
            return "critic"
        raise AssertionError(f"unrecognized system prompt: {system!r}")

    def generate_structured(self, api_key, model, system, user_content, schema):
        key = self._key(system)
        self.calls.append(key)
        outcome = self._routes[key]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def list_models(self, api_key):
        raise NotImplementedError

    def count_tokens(self, api_key, model, content, system=None):
        key = self._key(system)
        self.count_token_calls.append((key, content))
        return self._token_counts.get(key, 100)


def _all_success_routes() -> dict[str, object]:
    return {
        "structure": _gen({
            "drop": {"phrase_index": 4, "confidence": 0.95, "reasoning": "clear chorus"},
            "breakdown": {"phrase_index": 5, "confidence": 0.9, "reasoning": "down after drop"},
            "outro": {"phrase_index": 11, "confidence": 0.9, "reasoning": "labeled outro"},
        }),
        "vocal": _gen({
            "vocal_buildup": {"phrase_index": 3, "confidence": 0.8, "reasoning": "last up before drop"}
        }),
        "energy": _gen({
            "special": {"phrase_index": 7, "confidence": 0.7, "reasoning": "recovery"}
        }),
        "critic": _gen({
            "adjustments": [{"pad": "D", "confidence": 0.99, "note": "very confident"}]
        }),
    }


# --- build_track_payload -----------------------------------------------


def test_build_track_payload_no_raw_arrays_leak(sample_track: Track):
    heuristic = CueStrategy().propose(sample_track)
    payload = agentic.build_track_payload(sample_track, heuristic)

    assert "waveform" not in payload
    assert "vocal_track" not in payload
    for entry in payload["phrase_energy"]:
        assert set(entry.keys()) == {"index", "mean_energy"}
    for region in payload["vocal_regions"]:
        assert set(region.keys()) == {"start_ms", "end_ms"}
    # Condensed to a handful of onset regions, not the raw 283-frame array.
    assert len(payload["vocal_regions"]) < len(sample_track.vocal_track)
    # Only the >=2s run survives; the ~139ms blip is filtered as noise --
    # see test_summarize_vocal_onsets_filters_short_blips for the isolated
    # version of this assertion.
    assert len(payload["vocal_regions"]) == 1


def test_summarize_vocal_onsets_filters_short_blips(beat_grid, phrases):
    """Direct regression test for the duration filter: the heuristic's own
    C-slot logic (strategy.py) only counts a vocal region as real if it
    sustains for >= 2000ms -- a specialist must never see a shorter blip
    the heuristic itself would discard as noise."""
    frame_ms = agentic._VOCAL_FRAME_MS
    long_enough = int(2000 / frame_ms) + 5  # comfortably over the 2000ms line
    too_short = int(2000 / frame_ms) - 5  # comfortably under it
    vocal_track = (
        [0] * 20
        + [3] * long_enough
        + [0] * 20
        + [3] * too_short
        + [0] * 20
    )
    track = Track(
        id=4, title="Blip Test", artist="Test", bpm=128.0,
        duration_ms=218000.0, analysis_path="", cues=[], phrases=phrases,
        beat_grid=beat_grid, vocal_track=vocal_track,
    )

    regions = agentic._summarize_vocal_onsets(track)

    assert len(regions) == 1
    expected_start = round(20 * frame_ms)
    assert regions[0]["start_ms"] == expected_start


def test_build_track_payload_phrase_shape(sample_track: Track):
    heuristic = CueStrategy().propose(sample_track)
    payload = agentic.build_track_payload(sample_track, heuristic)

    assert len(payload["phrases"]) == len(sample_track.phrases)
    first = payload["phrases"][0]
    assert set(first.keys()) == {"index", "label", "beat_start", "position_ms", "duration_ms"}


def test_build_track_payload_includes_heuristic_confidence(sample_track: Track):
    heuristic = CueStrategy().propose(sample_track)
    payload = agentic.build_track_payload(sample_track, heuristic)

    assert payload["heuristic_confidence"].keys() == heuristic.confidence.keys()


def test_build_track_payload_includes_heuristic_phrase_index_as_a_real_anchor(sample_track: Track):
    """A specialist needs a concrete phrase to agree/disagree with, not
    just a bare confidence number with nothing to compare it against."""
    heuristic = CueStrategy().propose(sample_track)
    payload = agentic.build_track_payload(sample_track, heuristic)

    heuristic_idx = payload["heuristic_phrase_index"]
    assert heuristic_idx.keys() == heuristic.confidence.keys()

    hot_d = next(c for c in heuristic.hot_cues if c.kind == 5)  # Drop = pad D
    resolved_phrase = sample_track.phrases[heuristic_idx["D"]]
    assert resolved_phrase.position_ms == hot_d.position_ms


def test_build_track_payload_is_json_serializable(sample_track: Track):
    heuristic = CueStrategy().propose(sample_track)
    payload = agentic.build_track_payload(sample_track, heuristic)

    json.dumps(payload)  # must not raise


def test_build_track_payload_empty_vocal_track_gives_no_regions(beat_grid, phrases):
    track = Track(
        id=2, title="No Vocal Data", artist="Test", bpm=128.0,
        duration_ms=218000.0, analysis_path="", cues=[], phrases=phrases,
        beat_grid=beat_grid,  # waveform=None, vocal_track=None
    )
    heuristic = CueStrategy().propose(track)
    payload = agentic.build_track_payload(track, heuristic)
    assert payload["vocal_regions"] == []
    assert payload["phrase_energy"] == []


# --- estimate_track_cost --------------------------------------------------
# Real per-track estimate: input tokens come from live count_tokens() calls
# against the actual constructed payload (via the fake provider here),
# not a flat guess -- output tokens still an approximation, since no
# provider can count hypothetical output before generation.

_TOKEN_COUNTS = {"structure": 200, "vocal": 150, "energy": 180, "critic": 220}


def test_estimate_track_cost_with_critic_sums_4_real_input_counts(sample_track: Track):
    provider = _FakeProvider(_all_success_routes(), token_counts=_TOKEN_COUNTS)
    input_tokens, output_tokens, cost = agentic.estimate_track_cost(
        sample_track, provider, api_key="k", model="claude-haiku-4-5"
    )
    assert input_tokens == 200 + 150 + 180 + 220
    assert output_tokens == 150 * 4
    assert cost == pytest.approx(price_estimate("claude-haiku-4-5", input_tokens, output_tokens))
    assert {key for key, _content in provider.count_token_calls} == {
        "structure", "vocal", "energy", "critic"
    }


def test_estimate_track_cost_skip_critic_never_counts_critic(sample_track: Track):
    provider = _FakeProvider(_all_success_routes(), token_counts=_TOKEN_COUNTS)
    input_tokens, output_tokens, cost = agentic.estimate_track_cost(
        sample_track, provider, api_key="k", model="claude-haiku-4-5", skip_critic=True
    )
    assert input_tokens == 200 + 150 + 180  # critic never counted, never charged
    assert output_tokens == 150 * 3
    assert "critic" not in {key for key, _content in provider.count_token_calls}


def test_estimate_track_cost_unpriced_model_is_none(sample_track: Track):
    provider = _FakeProvider(_all_success_routes(), token_counts=_TOKEN_COUNTS)
    _, _, cost = agentic.estimate_track_cost(
        sample_track, provider, api_key="k", model="some-unpriced-model"
    )
    assert cost is None


def test_estimate_track_cost_counts_the_real_payload_not_a_placeholder(sample_track: Track):
    """The whole point of replacing the old flat per-call guess: what
    gets counted must be this track's actual payload, not a static
    string -- so a bigger/smaller track produces a different count."""
    provider = _FakeProvider(_all_success_routes(), token_counts=_TOKEN_COUNTS)
    agentic.estimate_track_cost(sample_track, provider, api_key="k", model="claude-haiku-4-5")

    heuristic = CueStrategy().propose(sample_track)
    expected_payload = agentic.build_track_payload(sample_track, heuristic)

    structure_content = next(
        content for key, content in provider.count_token_calls if key == "structure"
    )
    assert json.loads(structure_content) == expected_payload

    critic_content = next(
        content for key, content in provider.count_token_calls if key == "critic"
    )
    critic_payload = json.loads(critic_content)
    assert critic_payload["proposed_positions_ms"]  # critic payload has the extra fields
    assert critic_payload["phrases"] == expected_payload["phrases"]


# --- propose_with_telemetry: orchestration/fallback ----------------------


def test_full_success_produces_valid_proposal_and_telemetry(sample_track: Track):
    provider = _FakeProvider(_all_success_routes())
    proposal, telemetry = agentic.propose_with_telemetry(
        sample_track, provider, api_key="k", model="claude-haiku-4-5"
    )

    assert isinstance(proposal, CueProposal)
    assert len(proposal.hot_cues) == 8
    assert len(proposal.memory_cues) == 8

    hot_d = next(c for c in proposal.hot_cues if c.kind == 5)  # Drop = pad D
    assert hot_d.position_ms == sample_track.phrases[4].position_ms
    # Critic's override (0.99) should win over the structure specialist's own 0.95.
    assert proposal.confidence["D"] == pytest.approx(0.99)

    assert telemetry.calls_made == 4  # 3 specialists + critic
    assert telemetry.input_tokens == 100 * 4
    assert telemetry.output_tokens == 20 * 4
    assert telemetry.errors == []
    assert telemetry.estimated_cost == pytest.approx(
        price_estimate("claude-haiku-4-5", 400, 80)
    )


def test_skip_critic_never_calls_critic(sample_track: Track):
    routes = _all_success_routes()
    del routes["critic"]  # would KeyError if the code tried to call it anyway
    provider = _FakeProvider(routes)

    proposal, telemetry = agentic.propose_with_telemetry(
        sample_track, provider, api_key="k", model="claude-haiku-4-5", skip_critic=True
    )

    assert "critic" not in provider.calls
    assert telemetry.calls_made == 3
    hot_d = next(c for c in proposal.hot_cues if c.kind == 5)
    assert hot_d.position_ms == sample_track.phrases[4].position_ms
    # No critic ran, so the structure specialist's own confidence stands.
    assert proposal.confidence["D"] == pytest.approx(0.95)


def test_specialist_failure_falls_back_to_heuristic_for_its_pads(sample_track: Track):
    heuristic = CueStrategy().propose(sample_track)
    heuristic_c_position = next(c for c in heuristic.hot_cues if c.kind == 3).position_ms

    routes = _all_success_routes()
    routes["vocal"] = RuntimeError("vocal provider boom")
    provider = _FakeProvider(routes)

    proposal, telemetry = agentic.propose_with_telemetry(
        sample_track, provider, api_key="k", model="claude-haiku-4-5"
    )

    assert len(proposal.hot_cues) == 8  # never crashes the whole run
    hot_c = next(c for c in proposal.hot_cues if c.kind == 3)  # Vocal/Buildup = pad C
    assert hot_c.position_ms == heuristic_c_position
    assert any("vocal" in e for e in telemetry.errors)
    # Other specialists were unaffected.
    hot_d = next(c for c in proposal.hot_cues if c.kind == 5)
    assert hot_d.position_ms == sample_track.phrases[4].position_ms


def test_declined_placement_falls_back_to_heuristic(sample_track: Track):
    heuristic = CueStrategy().propose(sample_track)
    heuristic_d_position = next(c for c in heuristic.hot_cues if c.kind == 5).position_ms

    routes = _all_success_routes()
    routes["structure"] = _gen({
        "drop": {"phrase_index": -1, "confidence": 0.1, "reasoning": "not confident"},
        "breakdown": {"phrase_index": 5, "confidence": 0.9, "reasoning": "down after drop"},
        "outro": {"phrase_index": 11, "confidence": 0.9, "reasoning": "labeled outro"},
    })
    provider = _FakeProvider(routes)

    proposal, telemetry = agentic.propose_with_telemetry(
        sample_track, provider, api_key="k", model="claude-haiku-4-5"
    )

    hot_d = next(c for c in proposal.hot_cues if c.kind == 5)
    assert hot_d.position_ms == heuristic_d_position
    assert any("declined" in n and "D " in n for n in proposal.notes)


def test_auth_error_from_specialist_propagates(sample_track: Track):
    routes = _all_success_routes()
    routes["energy"] = AuthenticationError("invalid API key")
    provider = _FakeProvider(routes)

    with pytest.raises(AuthenticationError):
        agentic.propose_with_telemetry(sample_track, provider, api_key="bad", model="claude-haiku-4-5")


def test_auth_error_from_critic_propagates(sample_track: Track):
    routes = _all_success_routes()
    routes["critic"] = AuthenticationError("invalid API key")
    provider = _FakeProvider(routes)

    with pytest.raises(AuthenticationError):
        agentic.propose_with_telemetry(sample_track, provider, api_key="bad", model="claude-haiku-4-5")


class GeminiStyleClientError(Exception):
    """Mirrors the real google.genai.errors.ClientError shape (confirmed
    by reading the SDK source against a real 403 response): every 4xx
    status -- 401, 403, 429, ... -- raises this exact same class, with
    no distinctly-named subclass to tell an auth failure apart from any
    other client error. Only a `.code` attribute carries the HTTP status."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def test_looks_like_auth_error_detects_gemini_style_permission_denied():
    """Direct regression test for the real bug: a genuine 403 from
    Gemini never matches by class name (Gemini has no
    AuthenticationError/PermissionDeniedError class at all), so
    detection has to fall back to the HTTP code."""
    exc = GeminiStyleClientError(403, "PERMISSION_DENIED")
    assert agentic._looks_like_auth_error(exc) is True


def test_looks_like_auth_error_detects_gemini_style_401():
    exc = GeminiStyleClientError(401, "UNAUTHENTICATED")
    assert agentic._looks_like_auth_error(exc) is True


def test_looks_like_auth_error_false_for_unrelated_client_error():
    exc = GeminiStyleClientError(429, "RESOURCE_EXHAUSTED")  # rate limit, not auth
    assert agentic._looks_like_auth_error(exc) is False


def test_looks_like_auth_error_false_for_plain_exception():
    assert agentic._looks_like_auth_error(RuntimeError("boom")) is False


def test_gemini_style_403_from_specialist_propagates(sample_track: Track):
    """End-to-end version of the class-name-blind detection above --
    this is the exact shape of exception the real Gemini SDK raised
    when it hit a real, live 403 during verification."""
    routes = _all_success_routes()
    routes["structure"] = GeminiStyleClientError(403, "PERMISSION_DENIED")
    provider = _FakeProvider(routes)

    with pytest.raises(GeminiStyleClientError):
        agentic.propose_with_telemetry(sample_track, provider, api_key="bad", model="gemini-2.5-flash-lite")


def test_specialist_failure_logs_error_once_not_once_per_pad(sample_track: Track):
    """Regression test for the inflated-count bug: the structure
    specialist covers 3 pads (D, E, G), so a naive per-pad error log
    triples the count for one real failure. Must log exactly once."""
    routes = _all_success_routes()
    routes["structure"] = RuntimeError("structure boom")
    provider = _FakeProvider(routes)

    _proposal, telemetry = agentic.propose_with_telemetry(
        sample_track, provider, api_key="k", model="claude-haiku-4-5"
    )

    structure_errors = [e for e in telemetry.errors if "structure" in e]
    assert len(structure_errors) == 1


def test_critic_cannot_move_a_cue_position(sample_track: Track):
    """The critic schema has no position field; even if a hallucinating
    provider stuffed one into the response, the code must never read it."""
    routes = _all_success_routes()
    routes["critic"] = _gen({
        "adjustments": [
            {"pad": "D", "confidence": 0.99, "note": "trust me", "position_ms": 999_999},
        ]
    })
    provider = _FakeProvider(routes)

    proposal, _telemetry = agentic.propose_with_telemetry(
        sample_track, provider, api_key="k", model="claude-haiku-4-5"
    )

    hot_d = next(c for c in proposal.hot_cues if c.kind == 5)
    assert hot_d.position_ms == sample_track.phrases[4].position_ms
    assert hot_d.position_ms != 999_999
    assert proposal.confidence["D"] == pytest.approx(0.99)  # confidence still applies


def test_propose_wrapper_returns_bare_cue_proposal(sample_track: Track):
    provider = _FakeProvider(_all_success_routes())
    result = agentic.propose(sample_track, provider, api_key="k", model="claude-haiku-4-5")
    assert isinstance(result, CueProposal)


def test_heuristic_passthrough_pads_never_get_llm_positions(sample_track: Track):
    """A, B, H get no specialist call at all -- their positions must come
    straight from the heuristic (possibly critic-adjusted confidence, but
    never a different position)."""
    heuristic = CueStrategy().propose(sample_track)
    heuristic_positions = {
        KIND_TO_PAD.get(c.kind): c.position_ms for c in heuristic.hot_cues
    }

    provider = _FakeProvider(_all_success_routes())
    proposal, _telemetry = agentic.propose_with_telemetry(
        sample_track, provider, api_key="k", model="claude-haiku-4-5"
    )

    for pad, kind in (("A", 1), ("B", 2), ("H", 9)):
        hot = next(c for c in proposal.hot_cues if c.kind == kind)
        assert hot.position_ms == heuristic_positions[pad]
