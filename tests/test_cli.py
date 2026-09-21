"""Tests for djcues.cli -- the Click command surface (propose, compare,
viz, review, apply, beatgrid, auth, history).

The DB layer (find_playlist/load_playlist_tracks) is always mocked --
these tests never touch a real Rekordbox database. Where a command's
own logic (heuristic cue placement, HTML rendering, session-dict
building) is pure and cheap, it's exercised for real against synthetic
Track fixtures rather than mocked, for stronger coverage; only real I/O
(the local HTTP servers, the browser launch, the blocking wait loops)
is mocked.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from djcues.cli import cli
from djcues.models import BeatGrid, CueProposal, Phrase, Track, TrackSummary, WaveformPoint
from djcues.strategy import CueStrategy


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def beat_grid() -> BeatGrid:
    return BeatGrid(first_beat_ms=77.0, bpm=128.0)


@pytest.fixture
def phrases(beat_grid: BeatGrid) -> list[Phrase]:
    raw = [(1, 1, "Intro"), (33, 2, "Up"), (145, 5, "Chorus"), (401, 6, "Outro")]
    end_beat = 433
    result = []
    for i, (beat, kind, label) in enumerate(raw):
        next_beat = raw[i + 1][0] if i + 1 < len(raw) else end_beat
        pos = beat_grid.beat_to_ms(beat)
        end_pos = beat_grid.beat_to_ms(next_beat)
        result.append(Phrase(
            beat_start=beat, beat_end=next_beat, kind=kind, label=label,
            position_ms=pos, duration_ms=end_pos - pos,
        ))
    return result


@pytest.fixture
def track(beat_grid: BeatGrid, phrases: list[Phrase]) -> Track:
    return Track(
        id=1, title="Test Track", artist="Test Artist", bpm=128.0,
        duration_ms=240_000.0, analysis_path="", cues=[], phrases=phrases,
        beat_grid=beat_grid,
    )


@pytest.fixture
def track_no_phrases(beat_grid: BeatGrid) -> Track:
    return Track(
        id=2, title="No Phrases Track", artist="Test Artist", bpm=128.0,
        duration_ms=240_000.0, analysis_path="", cues=[], phrases=[],
        beat_grid=beat_grid,
    )


@pytest.fixture
def proposal(track: Track) -> CueProposal:
    return CueStrategy().propose(track)


@pytest.fixture(autouse=True)
def no_analysis_cache():
    """Every existing test in this file predates the analysis cache and
    isn't testing it -- without this, a CliRunner invocation of propose/
    compare/review/beatgrid would hit the REAL ~/.djcues/analysis_cache.db
    (by design, analysis_cache has no test-injectable db_path threaded
    through cli.py -- it's meant to be one real, single, user-global
    cache). Neutralizing it here matches test_writer.py's own precedent:
    get_cached/store_result are mocked at the source, the same way that
    file mocks djcues.history.log_session_corrections rather than letting
    apply_session touch real state. Tests that actually exercise caching
    behavior (TestAnalysisCacheIntegration) override this locally with
    their own patch()."""
    with patch("djcues.analysis_cache.get_cached", return_value=None), \
         patch("djcues.analysis_cache.store_result"):
        yield


def _mock_playlist(playlist_id: int = 1, name: str = "Test Playlist") -> MagicMock:
    pl = MagicMock()
    pl.ID = playlist_id
    pl.Name = name
    return pl


# ---------------------------------------------------------------------------
# Helper function unit tests (no CliRunner needed)
# ---------------------------------------------------------------------------


def test_format_time_under_a_minute():
    from djcues.cli import _format_time
    assert _format_time(5_500) == "0:05.5"


def test_format_time_over_a_minute():
    from djcues.cli import _format_time
    assert _format_time(90_250) == "1:30.2"


def test_resolve_agentic_provider_exits_without_key():
    from djcues.cli import _resolve_agentic_provider
    with patch("djcues.auth.load_config", return_value={}), \
         patch("djcues.auth.resolve_api_key", return_value=(None, None)):
        with pytest.raises(SystemExit) as exc_info:
            _resolve_agentic_provider(None, None)
        assert exc_info.value.code == 1


def test_resolve_agentic_provider_returns_resolved_triple():
    from djcues.cli import _resolve_agentic_provider
    with patch("djcues.auth.load_config", return_value={"provider": "anthropic", "model": "saved-model"}), \
         patch("djcues.auth.resolve_api_key", return_value=("sk-real-key", "keyring")):
        provider_name, api_key, model = _resolve_agentic_provider(None, None)
        assert provider_name == "anthropic"
        assert api_key == "sk-real-key"
        assert model == "saved-model"


def test_resolve_agentic_provider_explicit_args_override_config():
    from djcues.cli import _resolve_agentic_provider
    with patch("djcues.auth.load_config", return_value={"provider": "anthropic", "model": "saved-model"}), \
         patch("djcues.auth.resolve_api_key", return_value=("sk-real-key", "env")):
        provider_name, api_key, model = _resolve_agentic_provider("gemini", "explicit-model")
        assert provider_name == "gemini"
        assert model == "explicit-model"


# --- _resolve_device_for_run: mirrors the _resolve_agentic_provider trio
# above exactly -- same arg > config > default precedence shape, applied
# to the device preference instead of provider/model. resolve_device()
# itself is mocked throughout (not relying on this test machine's real
# cpu-only/no-directml environment) so these test precedence and
# argument-passing, independent of what's actually installed. ----------


def test_resolve_device_for_run_returns_configured_value():
    from djcues.cli import _resolve_device_for_run
    from djcues.device import ResolvedDevice

    fake_result = ResolvedDevice(requested="cpu", active="cpu", fell_back=False, reason=None, torch_device=None)
    with patch("djcues.auth.load_config", return_value={"device": "cpu"}), \
         patch("djcues.device.resolve_device", return_value=fake_result) as mock_resolve:
        result = _resolve_device_for_run(None)
        assert result == "cpu"
        mock_resolve.assert_called_once_with("cpu")


def test_resolve_device_for_run_explicit_arg_overrides_config():
    from djcues.cli import _resolve_device_for_run
    from djcues.device import ResolvedDevice

    fake_result = ResolvedDevice(requested="directml", active="directml", fell_back=False, reason=None, torch_device=None)
    with patch("djcues.auth.load_config", return_value={"device": "cpu"}), \
         patch("djcues.device.resolve_device", return_value=fake_result) as mock_resolve:
        result = _resolve_device_for_run("directml")
        assert result == "directml"
        mock_resolve.assert_called_once_with("directml")


def test_resolve_device_for_run_defaults_to_auto_when_unconfigured():
    from djcues.cli import _resolve_device_for_run
    from djcues.device import ResolvedDevice

    fake_result = ResolvedDevice(requested="auto", active="cpu", fell_back=False, reason="no accelerator available", torch_device=None)
    with patch("djcues.auth.load_config", return_value={}), \
         patch("djcues.device.resolve_device", return_value=fake_result) as mock_resolve:
        result = _resolve_device_for_run(None)
        assert result == "cpu"
        mock_resolve.assert_called_once_with("auto")


def test_resolve_device_for_run_warns_on_fallback_and_returns_active_device(capsys):
    """The one-time fallback warning users actually see -- confirms both
    the returned device (cpu, the safe fallback) and that the warning
    text names the reason, not just a generic failure message."""
    from djcues.cli import _resolve_device_for_run
    from djcues.device import ResolvedDevice

    fake_result = ResolvedDevice(requested="cuda", active="cpu", fell_back=True, reason="no GPU found", torch_device=None)
    with patch("djcues.auth.load_config", return_value={}), \
         patch("djcues.device.resolve_device", return_value=fake_result):
        result = _resolve_device_for_run("cuda")

    assert result == "cpu"
    captured = capsys.readouterr()
    assert "Warning" in captured.err
    assert "no GPU found" in captured.err


def test_resolve_device_for_run_silent_on_success(capsys):
    from djcues.cli import _resolve_device_for_run
    from djcues.device import ResolvedDevice

    fake_result = ResolvedDevice(requested="cpu", active="cpu", fell_back=False, reason=None, torch_device=None)
    with patch("djcues.auth.load_config", return_value={}), \
         patch("djcues.device.resolve_device", return_value=fake_result):
        _resolve_device_for_run("cpu")

    captured = capsys.readouterr()
    assert captured.err == ""


def test_resolve_device_for_run_handles_torch_not_installed(capsys):
    """beatgrid calls this unconditionally, with no upfront capability
    gate the way propose/compare/review have via
    _check_refine_drops_available -- must not raise a raw ImportError
    if torch genuinely isn't installed, just degrade to cpu."""
    from djcues.cli import _resolve_device_for_run

    with patch("djcues.auth.load_config", return_value={}), \
         patch("djcues.device.resolve_device", side_effect=ImportError("no torch")):
        result = _resolve_device_for_run(None)

    assert result == "cpu"


# ---------------------------------------------------------------------------
# propose
# ---------------------------------------------------------------------------


class TestPropose:
    def test_playlist_not_found(self, runner: CliRunner):
        with patch("djcues.cli.find_playlist", return_value=None):
            result = runner.invoke(cli, ["propose", "Nope", "--all"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_empty_playlist(self, runner: CliRunner):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[]):
            result = runner.invoke(cli, ["propose", "Empty", "--all"])
        assert result.exit_code == 1
        assert "no tracks found" in result.output

    def test_no_track_name_and_no_all_flag(self, runner: CliRunner, track: Track):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]):
            result = runner.invoke(cli, ["propose", "Test Playlist"])
        assert result.exit_code == 1
        assert "provide a track name or use --all" in result.output

    def test_track_name_not_matched(self, runner: CliRunner, track: Track):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]):
            result = runner.invoke(cli, ["propose", "Test Playlist", "Nonexistent Song"])
        assert result.exit_code == 1
        assert "no track matching" in result.output

    def test_estimate_only_without_agentic_errors(self, runner: CliRunner, track: Track):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]):
            result = runner.invoke(cli, ["propose", "Test Playlist", "--all", "--estimate-only"])
        assert result.exit_code == 1
        assert "--estimate-only only applies with --agentic" in result.output

    def test_deep_without_refine_drops_errors(self, runner: CliRunner, track: Track):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]):
            result = runner.invoke(cli, ["propose", "Test Playlist", "--all", "--deep"])
        assert result.exit_code == 1
        assert "--deep only applies with --refine-drops" in result.output

    def test_heuristic_happy_path_prints_proposal(self, runner: CliRunner, track: Track):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]):
            result = runner.invoke(cli, ["propose", "Test Playlist", "Test Track"])
        assert result.exit_code == 0
        assert "Test Track" in result.output
        assert "Test Artist" in result.output
        # This fixture's 4 phrases don't give the heuristic enough structure
        # to place all 8 slots (no Down/Bridge for E, no later Chorus for F)
        # -- 6 is the real, correct count for this input, not a stand-in for 8.
        assert "Hot Cues (6)" in result.output
        assert "Memory Cues (6)" in result.output

    def test_all_flag_processes_every_track(self, runner: CliRunner, track: Track, phrases):
        second = Track(
            id=3, title="Second Track", artist="Test Artist", bpm=128.0,
            duration_ms=240_000.0, analysis_path="", cues=[], phrases=phrases,
            beat_grid=track.beat_grid,
        )
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track, second]):
            result = runner.invoke(cli, ["propose", "Test Playlist", "--all"])
        assert result.exit_code == 0
        assert "Test Track" in result.output
        assert "Second Track" in result.output

    def test_all_flag_skips_tracks_with_no_phrase_data_instead_of_crashing(
        self, runner: CliRunner, track: Track, track_no_phrases: Track
    ):
        # Regression test: a track with no phrase data also typically has
        # bpm=0 in practice (Rekordbox never finished analyzing it), and
        # CueStrategy.propose() divides by bpm -- this used to crash
        # --all outright instead of skipping the one bad track, found via
        # a real ZeroDivisionError against a real Rekordbox library.
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track, track_no_phrases]):
            result = runner.invoke(cli, ["propose", "Test Playlist", "--all"])
        assert result.exit_code == 0
        assert "Test Track" in result.output
        assert "Skipping No Phrases Track (no phrase data)" in result.output

    def test_refine_drops_composes_and_prints_summary(self, runner: CliRunner, track: Track):
        from djcues.models import DropRefinement

        fake_refinement = DropRefinement(
            pad="D", outcome="refined", original_ms=1000.0, refined_ms=1200.0,
            offset_ms=200.0, strength=2.0, source="full_mix", note="test",
        )

        def fake_enhance(proposal, t, offset, loop_bars, deep=False, device="cpu"):
            return proposal, [fake_refinement]

        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.drop_enhance.enhance_proposal_drops", side_effect=fake_enhance):
            result = runner.invoke(cli, ["propose", "Test Playlist", "Test Track", "--refine-drops"])
        assert result.exit_code == 0
        assert "1 cue(s) checked against real audio" in result.output
        assert "1 refined" in result.output
        assert "1.0s -> 1.2s" in result.output

    def test_deep_with_unavailable_device_warns_and_still_completes(self, runner: CliRunner, track: Track):
        """The end-to-end "no regression" guarantee at the CLI level:
        an unavailable --device must never crash the run -- it warns
        once and completes using the resolved (cpu) fallback, which
        this test confirms actually reaches enhance_proposal_drops."""
        from djcues.device import ResolvedDevice
        from djcues.models import DropRefinement

        fake_refinement = DropRefinement(
            pad="D", outcome="confirmed", original_ms=1000.0, refined_ms=1000.0,
            offset_ms=0.0, strength=1.0, source="stems", note="test",
        )
        seen_devices = []

        def fake_enhance(proposal, t, offset, loop_bars, deep=False, device="cpu"):
            seen_devices.append(device)
            return proposal, [fake_refinement]

        fake_result = ResolvedDevice(requested="cuda", active="cpu", fell_back=True, reason="no GPU found", torch_device=None)
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.drop_enhance.enhance_proposal_drops", side_effect=fake_enhance), \
             patch("djcues.device.resolve_device", return_value=fake_result):
            result = runner.invoke(
                cli, ["propose", "Test Playlist", "Test Track", "--refine-drops", "--deep", "--device", "cuda"]
            )

        assert result.exit_code == 0, result.output
        assert "Warning: device 'cuda' isn't usable right now" in result.output
        assert "no GPU found" in result.output
        assert seen_devices == ["cpu"]  # the resolved fallback, not the requested "cuda"

    def test_refine_drops_without_librosa_errors(self, runner: CliRunner, track: Track):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch.dict("sys.modules", {"librosa": None}):
            result = runner.invoke(cli, ["propose", "Test Playlist", "Test Track", "--refine-drops"])
        assert result.exit_code == 1
        assert "djcues[audio]" in result.output

    def test_agentic_estimate_only_never_calls_proposer(self, runner: CliRunner, track: Track):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.auth.load_config", return_value={"provider": "anthropic", "model": "m"}), \
             patch("djcues.auth.resolve_api_key", return_value=("sk-key", "keyring")), \
             patch("djcues.providers.get_provider", return_value=MagicMock()), \
             patch("djcues.agentic.estimate_track_cost", return_value=(100, 50, 0.001)), \
             patch("djcues.agentic.propose_with_telemetry") as mock_propose:
            result = runner.invoke(
                cli, ["propose", "Test Playlist", "--all", "--agentic", "--estimate-only"]
            )
        assert result.exit_code == 0
        assert "Estimated cost" in result.output
        mock_propose.assert_not_called()

    def test_agentic_happy_path_prints_cost_summary(self, runner: CliRunner, track: Track):
        from djcues.agentic import AgenticTelemetry

        telemetry = AgenticTelemetry(calls_made=4, input_tokens=500, output_tokens=200)

        def fake_propose_with_telemetry(t, provider, api_key, model, **kwargs):
            return CueStrategy().propose(t), telemetry

        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.auth.load_config", return_value={"provider": "anthropic", "model": "claude-x"}), \
             patch("djcues.auth.resolve_api_key", return_value=("sk-key", "keyring")), \
             patch("djcues.providers.get_provider", return_value=MagicMock()), \
             patch("djcues.providers.estimate_cost", return_value=0.0042), \
             patch("djcues.agentic.propose_with_telemetry", side_effect=fake_propose_with_telemetry):
            result = runner.invoke(cli, ["propose", "Test Playlist", "Test Track", "--agentic"])
        assert result.exit_code == 0
        assert "Actual cost: $0.0042" in result.output
        assert "4 calls" in result.output

    def test_agentic_provider_error_aborts_cleanly(self, runner: CliRunner, track: Track):
        def raising_propose(t, provider, api_key, model, **kwargs):
            raise RuntimeError("401 unauthorized")

        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.auth.load_config", return_value={"provider": "anthropic", "model": "claude-x"}), \
             patch("djcues.auth.resolve_api_key", return_value=("sk-bad-key", "keyring")), \
             patch("djcues.providers.get_provider", return_value=MagicMock()), \
             patch("djcues.agentic.propose_with_telemetry", side_effect=raising_propose):
            result = runner.invoke(cli, ["propose", "Test Playlist", "Test Track", "--agentic"])
        assert result.exit_code == 1
        assert "aborted" in result.output
        assert "djcues auth status" in result.output


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


class TestCompare:
    def test_playlist_not_found(self, runner: CliRunner):
        with patch("djcues.cli.find_playlist", return_value=None):
            result = runner.invoke(cli, ["compare", "Nope", "--all"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_no_track_name_and_no_all_flag(self, runner: CliRunner, track: Track):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]):
            result = runner.invoke(cli, ["compare", "Test Playlist"])
        assert result.exit_code == 1
        assert "provide a track name or use --all" in result.output

    def test_single_track_prints_comparison_table(self, runner: CliRunner, track: Track):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]):
            result = runner.invoke(cli, ["compare", "Test Playlist", "Test Track"])
        assert result.exit_code == 0
        assert "Test Track" in result.output
        assert "Existing" in result.output and "Proposed" in result.output

    def test_all_flag_prints_overall_stats(self, runner: CliRunner, track: Track):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]):
            result = runner.invoke(cli, ["compare", "Test Playlist", "--all"])
        assert result.exit_code == 0
        assert "Overall" in result.output

    def test_all_flag_skips_tracks_with_no_phrase_data_instead_of_crashing(
        self, runner: CliRunner, track: Track, track_no_phrases: Track
    ):
        # Same real-library regression as propose's version of this test.
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track, track_no_phrases]):
            result = runner.invoke(cli, ["compare", "Test Playlist", "--all"])
        assert result.exit_code == 0
        assert "Skipping No Phrases Track (no phrase data)" in result.output
        assert "Overall" in result.output


# ---------------------------------------------------------------------------
# viz
# ---------------------------------------------------------------------------


class TestViz:
    def test_playlist_not_found(self, runner: CliRunner):
        with patch("djcues.cli.find_playlist", return_value=None):
            result = runner.invoke(cli, ["viz", "Nope", "--all"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_track_with_no_phrases_errors(self, runner: CliRunner, track_no_phrases: Track):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track_no_phrases]), \
             patch("webbrowser.open"):
            result = runner.invoke(cli, ["viz", "Test Playlist", "No Phrases Track"])
        assert result.exit_code == 1
        assert "no phrase data" in result.output

    def test_single_track_writes_real_html(self, runner: CliRunner, track: Track, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("webbrowser.open") as mock_open:
            result = runner.invoke(cli, ["viz", "Test Playlist", "Test Track"])
        assert result.exit_code == 0
        mock_open.assert_called_once()

        written = list(tmp_path.glob("*.html"))
        assert len(written) == 1
        html_content = written[0].read_text(encoding="utf-8")
        assert "Test Track" in html_content
        assert "<!DOCTYPE html>" in html_content

    def test_all_flag_skips_tracks_without_phrases(
        self, runner: CliRunner, track: Track, track_no_phrases: Track, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track, track_no_phrases]), \
             patch("webbrowser.open"):
            result = runner.invoke(cli, ["viz", "Test Playlist", "--all"])
        assert result.exit_code == 0
        assert "Rendering 1 tracks" in result.output
        assert "Skipping No Phrases Track" in result.output


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------


class TestReview:
    def test_playlist_not_found(self, runner: CliRunner):
        with patch("djcues.cli.find_playlist", return_value=None):
            result = runner.invoke(cli, ["review", "Nope", "--all"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_empty_playlist(self, runner: CliRunner):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[]):
            result = runner.invoke(cli, ["review", "Test Playlist", "--all"])
        assert result.exit_code == 1
        assert "No tracks found" in result.output

    def test_happy_path_starts_server_and_writes_session(
        self, runner: CliRunner, track: Track, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        fake_server = MagicMock()
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.server.start_server", return_value=(fake_server, 54123)) as mock_start, \
             patch("webbrowser.open") as mock_open, \
             patch("time.sleep", side_effect=KeyboardInterrupt):
            result = runner.invoke(cli, ["review", "Test Playlist", "--all"])

        assert result.exit_code == 0
        assert "Server stopped." in result.output
        mock_start.assert_called_once()
        mock_open.assert_called_once_with("http://127.0.0.1:54123")

        sessions = list(tmp_path.glob("*-session.json"))
        htmls = list(tmp_path.glob("*-review.html"))
        assert len(sessions) == 1
        assert len(htmls) == 1
        assert "Test Track" in htmls[0].read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------


class TestDashboard:
    """Mirrors TestReview's/TestAuthWeb's own established pattern for a
    server-backed command: mock start_dashboard_server/webbrowser.open,
    break the blocking wait loop via time.sleep's side_effect."""

    def test_happy_path_starts_server_and_opens_browser(self, runner: CliRunner):
        fake_server = MagicMock()
        with patch("djcues.server.start_dashboard_server", return_value=(fake_server, 54123)) as mock_start, \
             patch("webbrowser.open") as mock_open, \
             patch("time.sleep", side_effect=KeyboardInterrupt):
            result = runner.invoke(cli, ["dashboard"])

        assert result.exit_code == 0
        assert "Dashboard: http://127.0.0.1:54123" in result.output
        assert "Server stopped." in result.output
        mock_start.assert_called_once()
        mock_open.assert_called_once_with("http://127.0.0.1:54123")

    def test_no_playlist_arg_means_no_initial_playlist(self, runner: CliRunner):
        fake_server = MagicMock()
        with patch("djcues.server.start_dashboard_server", return_value=(fake_server, 54123)), \
             patch("djcues.dashboard.render_dashboard_html", return_value="<html></html>") as mock_render, \
             patch("webbrowser.open"), \
             patch("time.sleep", side_effect=KeyboardInterrupt):
            runner.invoke(cli, ["dashboard"])

        _, kwargs = mock_render.call_args
        assert kwargs["initial_playlist"] is None

    def test_playlist_arg_resolves_to_initial_playlist(self, runner: CliRunner):
        fake_server = MagicMock()
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist(playlist_id=42, name="Tech House")), \
             patch("djcues.server.start_dashboard_server", return_value=(fake_server, 54123)), \
             patch("djcues.dashboard.render_dashboard_html", return_value="<html></html>") as mock_render, \
             patch("webbrowser.open"), \
             patch("time.sleep", side_effect=KeyboardInterrupt):
            result = runner.invoke(cli, ["dashboard", "Tech House"])

        assert result.exit_code == 0
        _, kwargs = mock_render.call_args
        assert kwargs["initial_playlist"] == {"id": 42, "name": "Tech House"}

    def test_unresolvable_playlist_arg_degrades_to_no_initial_playlist(self, runner: CliRunner):
        """Never a hard error -- matches the command's whole "you don't
        need to already know exact names" premise: browsing manually is
        always the fallback."""
        fake_server = MagicMock()
        with patch("djcues.cli.find_playlist", return_value=None), \
             patch("djcues.server.start_dashboard_server", return_value=(fake_server, 54123)), \
             patch("djcues.dashboard.render_dashboard_html", return_value="<html></html>") as mock_render, \
             patch("webbrowser.open"), \
             patch("time.sleep", side_effect=KeyboardInterrupt):
            result = runner.invoke(cli, ["dashboard", "Nonexistent Playlist"])

        assert result.exit_code == 0
        _, kwargs = mock_render.call_args
        assert kwargs["initial_playlist"] is None

    def test_offset_and_loop_bars_passed_through(self, runner: CliRunner):
        fake_server = MagicMock()
        with patch("djcues.server.start_dashboard_server", return_value=(fake_server, 54123)), \
             patch("djcues.dashboard.render_dashboard_html", return_value="<html></html>") as mock_render, \
             patch("webbrowser.open"), \
             patch("time.sleep", side_effect=KeyboardInterrupt):
            runner.invoke(cli, ["dashboard", "--offset", "8", "--loop-bars", "2"])

        _, kwargs = mock_render.call_args
        assert kwargs["default_offset"] == 8
        assert kwargs["default_loop_bars"] == 2


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


class TestApply:
    def test_missing_session_file_errors(self, runner: CliRunner):
        result = runner.invoke(cli, ["apply", "does-not-exist.json"])
        assert result.exit_code != 0

    def test_delegates_to_apply_session(self, runner: CliRunner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "session.json").write_text("{}")
        with patch("djcues.writer.apply_session", return_value={}) as mock_apply:
            result = runner.invoke(cli, ["apply", "session.json", "--dry-run"])
        assert result.exit_code == 0
        mock_apply.assert_called_once()
        _, kwargs = mock_apply.call_args
        assert kwargs["dry_run"] is True
        assert kwargs["force"] is False


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------


class TestHistory:
    def test_no_history_yet(self, runner: CliRunner, tmp_path):
        db_path = tmp_path / "history.db"
        with patch("djcues.history.summary", return_value=[]), \
             patch("djcues.history.default_db_path", return_value=db_path):
            result = runner.invoke(cli, ["history"])
        assert result.exit_code == 0
        assert "No correction history yet" in result.output

    def test_prints_per_pad_stats(self, runner: CliRunner, tmp_path):
        db_path = tmp_path / "history.db"
        rows = [
            {"pad": "D", "total": 10, "corrected": 3, "first_seen": "2026-01-01", "last_seen": "2026-02-01"},
        ]
        with patch("djcues.history.summary", return_value=rows), \
             patch("djcues.history.default_db_path", return_value=db_path):
            result = runner.invoke(cli, ["history"])
        assert result.exit_code == 0
        assert "D" in result.output
        assert "10" in result.output and "3" in result.output


# ---------------------------------------------------------------------------
# beatgrid
# ---------------------------------------------------------------------------


class TestBeatgrid:
    def test_playlist_not_found(self, runner: CliRunner):
        with patch("djcues.cli.find_playlist", return_value=None):
            result = runner.invoke(cli, ["beatgrid", "Nope", "--all"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_ok_track_prints_self_consistent(self, runner: CliRunner, track: Track):
        from djcues.models import BeatGridReport, SelfConsistencyResult

        report = BeatGridReport(
            track_id=track.id, title=track.title,
            self_consistency=SelfConsistencyResult(
                is_consistent=True, tempo_varies=False,
                max_pairwise_gap_error_ms=0.0, cumulative_drift_at_end_ms=0.0,
                entry_count=10, notes=[],
            ),
            audio=None, status="ok",
        )
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.db.get_db", return_value=MagicMock()), \
             patch("djcues.db.extract_raw_beat_grid", return_value=[]), \
             patch("djcues.beat_verify.verify_beat_grid", return_value=report):
            result = runner.invoke(cli, ["beatgrid", "Test Playlist", "Test Track"])
        assert result.exit_code == 0
        assert "self-consistent" in result.output

    def test_octave_error_shows_explanation(self, runner: CliRunner, track: Track):
        from djcues.models import AudioBeatVerification, BeatGridReport, SelfConsistencyResult

        report = BeatGridReport(
            track_id=track.id, title=track.title,
            self_consistency=SelfConsistencyResult(
                is_consistent=False, tempo_varies=False,
                max_pairwise_gap_error_ms=500.0, cumulative_drift_at_end_ms=1000.0,
                entry_count=10, notes=["flagged"],
            ),
            audio=AudioBeatVerification(
                matched_beats=100, mean_abs_drift_ms=200.0, max_abs_drift_ms=300.0,
                pct_within_tolerance=10.0, tracker_name="beat_this",
                verdict="octave_error", octave_error="double",
            ),
            status="flagged",
        )
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.db.get_db", return_value=MagicMock()), \
             patch("djcues.db.extract_raw_beat_grid", return_value=[]), \
             patch("djcues.beat_verify.verify_beat_grid", return_value=report):
            result = runner.invoke(cli, ["beatgrid", "Test Playlist", "Test Track"])
        assert result.exit_code == 0
        assert "POSSIBLE OCTAVE ERROR" in result.output
        assert "phantom beats" in result.output

    def test_all_flag_prints_aggregate_summary(self, runner: CliRunner, track: Track, phrases):
        from djcues.models import BeatGridReport, SelfConsistencyResult

        second = Track(
            id=4, title="Second", artist="A", bpm=128.0, duration_ms=1000.0,
            analysis_path="", cues=[], phrases=phrases, beat_grid=track.beat_grid,
        )
        report = BeatGridReport(
            track_id=1, title="x",
            self_consistency=SelfConsistencyResult(
                is_consistent=True, tempo_varies=False,
                max_pairwise_gap_error_ms=0.0, cumulative_drift_at_end_ms=0.0,
                entry_count=1, notes=[],
            ),
            audio=None, status="ok",
        )
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track, second]), \
             patch("djcues.db.get_db", return_value=MagicMock()), \
             patch("djcues.db.extract_raw_beat_grid", return_value=[]), \
             patch("djcues.beat_verify.verify_beat_grid", return_value=report):
            result = runner.invoke(cli, ["beatgrid", "Test Playlist", "--all"])
        assert result.exit_code == 0
        assert "2 tracks checked, 2 OK" in result.output


# ---------------------------------------------------------------------------
# analysis cache (propose/compare/review/beatgrid wiring, cache group)
# ---------------------------------------------------------------------------


class TestAnalysisCacheIntegration:
    """propose/compare/review/beatgrid's own analysis_cache wiring.
    djcues.analysis_cache.get_cached/store_result are mocked at the
    source in each test here (matching test_writer.py's precedent for
    cross-module persistence calls -- that file mocks
    djcues.history.log_session_corrections rather than re-testing it),
    overriding the file's own no_analysis_cache autouse fixture locally
    wherever a specific hit/miss/failure scenario needs asserting on.
    """

    def test_propose_cache_miss_calls_proposer_and_stores_result(
        self, runner: CliRunner, track: Track
    ):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.analysis_cache.get_cached", return_value=None) as mock_get, \
             patch("djcues.analysis_cache.store_result") as mock_store:
            result = runner.invoke(cli, ["propose", "Test Playlist", "Test Track"])

        assert result.exit_code == 0
        mock_get.assert_called_once()
        mock_store.assert_called_once()
        _, kwargs = mock_store.call_args
        assert kwargs["track_title"] == "Test Track"
        assert "Using cached analysis" not in result.output

    def test_propose_cache_hit_skips_proposer_and_prints_indicator(
        self, runner: CliRunner, track: Track
    ):
        from djcues.analysis_cache import CachedResult

        cached = CachedResult(
            result={"positions": {"A": 77.0}, "confidence": {"A": 1.0}, "notes": ["cached"]},
            cost_usd=None, compute_seconds=190.4, source="cli",
            created_at="2026-09-09T12:00:00", updated_at="2026-09-09T12:00:00",
        )
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.analysis_cache.get_cached", return_value=cached), \
             patch("djcues.analysis_cache.store_result") as mock_store, \
             patch("djcues.strategy.CueStrategy.propose") as mock_propose:
            result = runner.invoke(cli, ["propose", "Test Playlist", "Test Track"])

        assert result.exit_code == 0
        mock_propose.assert_not_called()
        mock_store.assert_not_called()
        assert "Using cached analysis for Test Track (analyzed 2026-09-09T12:00:00)" in result.output
        assert "Cache: 1/1 track(s) reused" in result.output

    def test_propose_no_cache_flag_skips_read_but_still_stores(
        self, runner: CliRunner, track: Track
    ):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.analysis_cache.get_cached") as mock_get, \
             patch("djcues.analysis_cache.store_result") as mock_store:
            result = runner.invoke(cli, ["propose", "Test Playlist", "Test Track", "--no-cache"])

        assert result.exit_code == 0
        mock_get.assert_not_called()
        mock_store.assert_called_once()

    def test_compare_all_and_single_track_both_use_cache(
        self, runner: CliRunner, track: Track
    ):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.analysis_cache.get_cached", return_value=None) as mock_get, \
             patch("djcues.analysis_cache.store_result") as mock_store:
            result_all = runner.invoke(cli, ["compare", "Test Playlist", "--all"])
        assert result_all.exit_code == 0
        assert mock_get.call_count == 1
        assert mock_store.call_count == 1

        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.analysis_cache.get_cached", return_value=None) as mock_get2, \
             patch("djcues.analysis_cache.store_result") as mock_store2:
            result_single = runner.invoke(cli, ["compare", "Test Playlist", "Test Track"])
        assert result_single.exit_code == 0
        assert mock_get2.call_count == 1
        assert mock_store2.call_count == 1

    def test_review_session_uses_cached_proposal_transparently(
        self, runner: CliRunner, track: Track, tmp_path, monkeypatch
    ):
        from djcues.analysis_cache import CachedResult

        monkeypatch.chdir(tmp_path)
        cached = CachedResult(
            result={"positions": {"A": 77.0}, "confidence": {"A": 1.0}, "notes": []},
            cost_usd=None, compute_seconds=1.0, source="cli",
            created_at="2026-09-09T12:00:00", updated_at="2026-09-09T12:00:00",
        )
        fake_server = MagicMock()
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.analysis_cache.get_cached", return_value=cached), \
             patch("djcues.analysis_cache.store_result") as mock_store, \
             patch("djcues.strategy.CueStrategy.propose") as mock_propose, \
             patch("djcues.server.start_server", return_value=(fake_server, 54123)), \
             patch("webbrowser.open"), \
             patch("time.sleep", side_effect=KeyboardInterrupt):
            result = runner.invoke(cli, ["review", "Test Playlist", "--all"])

        assert result.exit_code == 0
        mock_propose.assert_not_called()
        mock_store.assert_not_called()
        assert "Using cached analysis" in result.output

    def test_beatgrid_second_invocation_pattern_uses_cache(
        self, runner: CliRunner, track: Track
    ):
        from djcues.analysis_cache import CachedResult

        cached = CachedResult(
            result={
                "status": "ok",
                "self_consistency": {
                    "is_consistent": True, "tempo_varies": False,
                    "max_pairwise_gap_error_ms": 0.0, "cumulative_drift_at_end_ms": 0.0,
                    "entry_count": 10, "notes": [],
                },
                "audio": None,
            },
            cost_usd=None, compute_seconds=0.1, source="cli",
            created_at="2026-09-09T12:00:00", updated_at="2026-09-09T12:00:00",
        )
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.db.get_db", return_value=MagicMock()), \
             patch("djcues.db.extract_raw_beat_grid", return_value=[]), \
             patch("djcues.analysis_cache.get_cached", return_value=cached), \
             patch("djcues.analysis_cache.store_result") as mock_store, \
             patch("djcues.beat_verify.verify_beat_grid") as mock_verify:
            result = runner.invoke(cli, ["beatgrid", "Test Playlist", "Test Track"])

        assert result.exit_code == 0
        mock_verify.assert_not_called()
        mock_store.assert_not_called()
        assert "Using cached beat-grid check" in result.output
        assert "self-consistent" in result.output

    def test_cache_io_failure_degrades_to_recompute_not_crash(
        self, runner: CliRunner, track: Track
    ):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.analysis_cache.get_cached", side_effect=RuntimeError("disk full")), \
             patch("djcues.analysis_cache.store_result", side_effect=RuntimeError("disk full")):
            result = runner.invoke(cli, ["propose", "Test Playlist", "Test Track"])

        assert result.exit_code == 0
        assert "Warning: analysis cache unavailable" in result.output
        # Real proposal output is still printed despite the cache being down.
        assert "Test Track" in result.output

    def test_beatgrid_cache_io_failure_degrades_to_recompute_not_crash(
        self, runner: CliRunner, track: Track
    ):
        """beatgrid wires the cache inline (no _apply_cache composition,
        unlike propose/compare/review -- see cli.py) so its own
        get_cached/store_result failure-degradation branches need their
        own, separate coverage from test_cache_io_failure_degrades_to_recompute_not_crash."""
        from djcues.models import BeatGridReport, SelfConsistencyResult

        report = BeatGridReport(
            track_id=track.id, title=track.title,
            self_consistency=SelfConsistencyResult(
                is_consistent=True, tempo_varies=False,
                max_pairwise_gap_error_ms=0.0, cumulative_drift_at_end_ms=0.0,
                entry_count=10, notes=[],
            ),
            audio=None, status="ok",
        )
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.db.get_db", return_value=MagicMock()), \
             patch("djcues.db.extract_raw_beat_grid", return_value=[]), \
             patch("djcues.beat_verify.verify_beat_grid", return_value=report), \
             patch("djcues.analysis_cache.get_cached", side_effect=RuntimeError("disk full")), \
             patch("djcues.analysis_cache.store_result", side_effect=RuntimeError("disk full")):
            result = runner.invoke(cli, ["beatgrid", "Test Playlist", "Test Track"])

        assert result.exit_code == 0
        assert "Warning: analysis cache unavailable" in result.output
        assert "self-consistent" in result.output

    def test_beatgrid_store_failure_alone_still_warns(self, runner: CliRunner, track: Track):
        """Same as above but with a clean cache read (a real miss, not a
        failure) followed by a store failure in isolation -- the
        previous test's simultaneous read+write failure only ever
        exercises the *first* warn-once branch (the read one), since
        cache_warned is already True by the time the write is attempted;
        this is what actually reaches the write branch's own
        `if not cache_warned` check with it still False."""
        from djcues.models import BeatGridReport, SelfConsistencyResult

        report = BeatGridReport(
            track_id=track.id, title=track.title,
            self_consistency=SelfConsistencyResult(
                is_consistent=True, tempo_varies=False,
                max_pairwise_gap_error_ms=0.0, cumulative_drift_at_end_ms=0.0,
                entry_count=10, notes=[],
            ),
            audio=None, status="ok",
        )
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.db.get_db", return_value=MagicMock()), \
             patch("djcues.db.extract_raw_beat_grid", return_value=[]), \
             patch("djcues.beat_verify.verify_beat_grid", return_value=report), \
             patch("djcues.analysis_cache.get_cached", return_value=None), \
             patch("djcues.analysis_cache.store_result", side_effect=RuntimeError("disk full")):
            result = runner.invoke(cli, ["beatgrid", "Test Playlist", "Test Track"])

        assert result.exit_code == 0
        assert "Warning: analysis cache unavailable" in result.output
        assert "self-consistent" in result.output

    def test_get_proposer_returns_resolved_provider_heuristic_is_none(self):
        from djcues.cli import _get_proposer

        proposer, telemetry_list, resolved_model, resolved_provider = _get_proposer(
            agentic=False, provider_name=None, model=None, offset=16, loop_bars=4, skip_critic=False,
        )
        assert telemetry_list is None
        assert resolved_model is None
        assert resolved_provider is None

    def test_get_proposer_agentic_returns_real_resolved_provider(self):
        from djcues.cli import _get_proposer

        with patch("djcues.auth.load_config", return_value={"provider": "gemini", "model": "gemini-x"}), \
             patch("djcues.auth.resolve_api_key", return_value=("sk-key", "keyring")), \
             patch("djcues.providers.get_provider", return_value=MagicMock()):
            proposer, telemetry_list, resolved_model, resolved_provider = _get_proposer(
                agentic=True, provider_name=None, model=None, offset=16, loop_bars=4, skip_critic=False,
            )
        assert telemetry_list == []
        assert resolved_model == "gemini-x"
        assert resolved_provider == "gemini"


class TestCacheCommand:
    def test_status_empty_cache(self, runner: CliRunner, tmp_path):
        db_path = tmp_path / "analysis_cache.db"
        with patch("djcues.analysis_cache.summary", return_value=[]), \
             patch("djcues.analysis_cache.default_db_path", return_value=db_path):
            result = runner.invoke(cli, ["cache", "status"])
        assert result.exit_code == 0
        assert "No cached analysis yet" in result.output

    def test_status_prints_rows(self, runner: CliRunner, tmp_path):
        db_path = tmp_path / "analysis_cache.db"
        rows = [
            {"analysis_kind": "cue_proposal", "engine": "heuristic", "total": 5,
             "total_cost_usd": 0.0, "first_seen": "2026-01-01", "last_seen": "2026-01-02"},
        ]
        with patch("djcues.analysis_cache.summary", return_value=rows), \
             patch("djcues.analysis_cache.default_db_path", return_value=db_path):
            result = runner.invoke(cli, ["cache", "status"])
        assert result.exit_code == 0
        assert "cue_proposal" in result.output
        assert "heuristic" in result.output

    def test_clear_already_empty(self, runner: CliRunner):
        with patch("djcues.analysis_cache.summary", return_value=[]):
            result = runner.invoke(cli, ["cache", "clear", "--force"])
        assert result.exit_code == 0
        assert "already empty" in result.output

    def test_clear_with_force_skips_confirmation(self, runner: CliRunner):
        rows = [{"analysis_kind": "cue_proposal", "engine": "heuristic", "total": 3,
                 "total_cost_usd": 0.0, "first_seen": "x", "last_seen": "y"}]
        with patch("djcues.analysis_cache.summary", return_value=rows), \
             patch("djcues.analysis_cache.clear_all", return_value=3) as mock_clear:
            result = runner.invoke(cli, ["cache", "clear", "--force"])
        assert result.exit_code == 0
        mock_clear.assert_called_once()
        assert "Deleted 3 cached result(s)" in result.output

    def test_clear_without_force_prompts_and_aborts_on_no(self, runner: CliRunner):
        rows = [{"analysis_kind": "cue_proposal", "engine": "heuristic", "total": 3,
                 "total_cost_usd": 0.0, "first_seen": "x", "last_seen": "y"}]
        with patch("djcues.analysis_cache.summary", return_value=rows), \
             patch("djcues.analysis_cache.clear_all") as mock_clear:
            result = runner.invoke(cli, ["cache", "clear"], input="n\n")
        assert result.exit_code == 0
        assert "Aborted" in result.output
        mock_clear.assert_not_called()


# ---------------------------------------------------------------------------
# auth set / status / clear / models / web
# ---------------------------------------------------------------------------


class TestAuthSet:
    def test_happy_path_saves_config_and_key(self, runner: CliRunner):
        from djcues.providers import ModelInfo

        fake_provider = MagicMock()
        fake_provider.list_models.return_value = [
            ModelInfo(id="claude-cheap", display_name="Cheap", provider="anthropic"),
            ModelInfo(id="claude-good", display_name="Good", provider="anthropic", context_window=200_000),
        ]
        saved_config = {}
        with patch("djcues.providers.get_provider", return_value=fake_provider), \
             patch("djcues.providers.DEFAULT_MODEL", {"anthropic": "claude-cheap"}), \
             patch("djcues.providers.RECOMMENDED_FOR_ACCURACY", {"anthropic": "claude-good"}), \
             patch("djcues.auth.set_api_key") as mock_set_key, \
             patch("djcues.auth.load_config", return_value={}), \
             patch("djcues.auth.save_config", side_effect=saved_config.update):
            result = runner.invoke(
                cli, ["auth", "set", "--provider", "anthropic"],
                input="sk-test-key\n1\n",
            )
        assert result.exit_code == 0, result.output
        mock_set_key.assert_called_once_with("anthropic", "sk-test-key")
        assert saved_config["provider"] == "anthropic"
        # The accuracy-recommended model sorts first in the picker (see
        # models_sorted's key in auth_set) -- choice "1" is claude-good,
        # not claude-cheap, despite claude-cheap being listed first above.
        assert saved_config["model"] == "claude-good"
        assert "Saved" in result.output

    def test_no_models_returned_errors(self, runner: CliRunner):
        fake_provider = MagicMock()
        fake_provider.list_models.return_value = []
        with patch("djcues.providers.get_provider", return_value=fake_provider):
            result = runner.invoke(
                cli, ["auth", "set", "--provider", "anthropic"], input="sk-test-key\n"
            )
        assert result.exit_code == 1
        assert "no models returned" in result.output

    def test_list_models_failure_errors(self, runner: CliRunner):
        fake_provider = MagicMock()
        fake_provider.list_models.side_effect = RuntimeError("invalid key")
        with patch("djcues.providers.get_provider", return_value=fake_provider):
            result = runner.invoke(
                cli, ["auth", "set", "--provider", "anthropic"], input="sk-bad-key\n"
            )
        assert result.exit_code == 1
        assert "could not validate key" in result.output


class TestAuthDevice:
    """Mirrors TestAuthSet's own conventions -- prompt/flag -> live
    probe/validate -> persist, just for the device preference instead
    of provider/model. resolve_device()/list_available_backends() are
    always mocked here, matching test_resolve_device_for_run's own
    reasoning: these tests exercise auth_device's own logic (probe
    display, confirm-on-fallback, persistence), independent of what
    this particular machine actually has installed."""

    @staticmethod
    def _fake_probes():
        from djcues.device import DeviceProbeResult
        return [
            DeviceProbeResult("cpu", ok=True, error=None),
            DeviceProbeResult("cuda", ok=False, error="no GPU found"),
            DeviceProbeResult("directml", ok=False, error="not installed"),
        ]

    def test_happy_path_saves_config_no_confirmation_needed(self, runner: CliRunner):
        from djcues.device import ResolvedDevice

        fake_result = ResolvedDevice(requested="cpu", active="cpu", fell_back=False, reason=None, torch_device=None)
        saved_config = {}
        with patch("djcues.device.list_available_backends", return_value=self._fake_probes()), \
             patch("djcues.device.resolve_device", return_value=fake_result), \
             patch("djcues.auth.load_config", return_value={}), \
             patch("djcues.auth.save_config", side_effect=saved_config.update):
            result = runner.invoke(cli, ["auth", "device", "--device", "cpu"])

        assert result.exit_code == 0, result.output
        assert saved_config["device"] == "cpu"
        assert "passed a live smoke test" in result.output
        assert "Saved. Device preference: cpu." in result.output

    def test_fallback_prompts_and_saves_on_confirm(self, runner: CliRunner):
        from djcues.device import ResolvedDevice

        fake_result = ResolvedDevice(requested="cuda", active="cpu", fell_back=True, reason="no GPU found", torch_device=None)
        saved_config = {}
        with patch("djcues.device.list_available_backends", return_value=self._fake_probes()), \
             patch("djcues.device.resolve_device", return_value=fake_result), \
             patch("djcues.auth.load_config", return_value={}), \
             patch("djcues.auth.save_config", side_effect=saved_config.update):
            result = runner.invoke(cli, ["auth", "device", "--device", "cuda"], input="y\n")

        assert result.exit_code == 0, result.output
        assert saved_config["device"] == "cuda"
        assert "isn't usable right now" in result.output
        assert "no GPU found" in result.output

    def test_fallback_declined_aborts_without_saving(self, runner: CliRunner):
        from djcues.device import ResolvedDevice

        fake_result = ResolvedDevice(requested="cuda", active="cpu", fell_back=True, reason="no GPU found", torch_device=None)
        with patch("djcues.device.list_available_backends", return_value=self._fake_probes()), \
             patch("djcues.device.resolve_device", return_value=fake_result), \
             patch("djcues.auth.load_config", return_value={}), \
             patch("djcues.auth.save_config") as mock_save:
            result = runner.invoke(cli, ["auth", "device", "--device", "cuda"], input="n\n")

        assert result.exit_code == 1
        mock_save.assert_not_called()

    def test_lists_every_probed_backend(self, runner: CliRunner):
        from djcues.device import ResolvedDevice

        fake_result = ResolvedDevice(requested="cpu", active="cpu", fell_back=False, reason=None, torch_device=None)
        with patch("djcues.device.list_available_backends", return_value=self._fake_probes()), \
             patch("djcues.device.resolve_device", return_value=fake_result), \
             patch("djcues.auth.load_config", return_value={}), \
             patch("djcues.auth.save_config"):
            result = runner.invoke(cli, ["auth", "device", "--device", "cpu"])

        assert "cpu: OK" in result.output
        assert "cuda: unavailable (no GPU found)" in result.output
        assert "directml: unavailable (not installed)" in result.output


class TestAuthStatus:
    def test_not_configured(self, runner: CliRunner):
        with patch("djcues.auth.load_config", return_value={}):
            result = runner.invoke(cli, ["auth", "status"])
        assert result.exit_code == 0
        assert "No agentic provider configured" in result.output

    def test_shows_device_default_when_unconfigured(self, runner: CliRunner):
        with patch("djcues.auth.load_config", return_value={}):
            result = runner.invoke(cli, ["auth", "status"])
        assert result.exit_code == 0
        assert "Device: auto" in result.output

    def test_shows_configured_device(self, runner: CliRunner):
        with patch("djcues.auth.load_config", return_value={"device": "cuda"}):
            result = runner.invoke(cli, ["auth", "status"])
        assert result.exit_code == 0
        assert "Device: cuda" in result.output

    def test_configured_with_key(self, runner: CliRunner):
        with patch("djcues.auth.load_config", return_value={"provider": "anthropic", "model": "claude-x"}), \
             patch("djcues.auth.resolve_api_key", return_value=("sk-key", "keyring")):
            result = runner.invoke(cli, ["auth", "status"])
        assert result.exit_code == 0
        assert "anthropic" in result.output
        assert "configured (source: keyring)" in result.output

    def test_configured_without_key(self, runner: CliRunner):
        with patch("djcues.auth.load_config", return_value={"provider": "anthropic", "model": "claude-x"}), \
             patch("djcues.auth.resolve_api_key", return_value=(None, None)):
            result = runner.invoke(cli, ["auth", "status"])
        assert result.exit_code == 0
        assert "not found" in result.output


class TestAuthClear:
    def test_confirmed_removes_key(self, runner: CliRunner):
        with patch("djcues.auth.clear_api_key") as mock_clear:
            result = runner.invoke(
                cli, ["auth", "clear", "--provider", "anthropic"], input="y\n"
            )
        assert result.exit_code == 0
        mock_clear.assert_called_once_with("anthropic")
        assert "Removed" in result.output

    def test_declined_does_not_remove(self, runner: CliRunner):
        with patch("djcues.auth.clear_api_key") as mock_clear:
            result = runner.invoke(
                cli, ["auth", "clear", "--provider", "anthropic"], input="n\n"
            )
        assert result.exit_code == 0
        mock_clear.assert_not_called()


class TestAuthModels:
    def test_no_provider_configured_or_specified_errors(self, runner: CliRunner):
        with patch("djcues.auth.load_config", return_value={}):
            result = runner.invoke(cli, ["auth", "models"])
        assert result.exit_code == 1
        assert "no provider configured" in result.output

    def test_lists_models_for_configured_provider(self, runner: CliRunner):
        from djcues.providers import ModelInfo

        fake_provider = MagicMock()
        fake_provider.list_models.return_value = [
            ModelInfo(id="claude-x", display_name="Claude X", provider="anthropic", context_window=200_000),
        ]
        with patch("djcues.auth.load_config", return_value={"provider": "anthropic"}), \
             patch("djcues.auth.resolve_api_key", return_value=("sk-key", "keyring")), \
             patch("djcues.providers.get_provider", return_value=fake_provider):
            result = runner.invoke(cli, ["auth", "models"])
        assert result.exit_code == 0
        assert "claude-x" in result.output
        assert "Claude X" in result.output


class TestAuthWeb:
    def _fake_server(self, **overrides):
        base = dict(_setup_complete=False, _shutdown_flag=False, _timed_out=False)
        base.update(overrides)
        return types.SimpleNamespace(**base)

    def test_happy_path_prints_saved(self, runner: CliRunner):
        fake_server = self._fake_server(_setup_complete=True)
        with patch("djcues.auth_web.render_auth_setup_html", return_value="<html></html>"), \
             patch("djcues.server.start_auth_server", return_value=(fake_server, 5555)), \
             patch("webbrowser.open") as mock_open, \
             patch("djcues.auth.load_config", return_value={"provider": "anthropic", "model": "claude-x"}):
            result = runner.invoke(cli, ["auth", "web"])
        assert result.exit_code == 0
        mock_open.assert_called_once_with("http://127.0.0.1:5555")
        assert "Saved. Provider: anthropic, model: claude-x." in result.output

    def test_keyboard_interrupt_cancels(self, runner: CliRunner):
        fake_server = self._fake_server()
        with patch("djcues.auth_web.render_auth_setup_html", return_value="<html></html>"), \
             patch("djcues.server.start_auth_server", return_value=(fake_server, 5555)), \
             patch("webbrowser.open"), \
             patch("time.sleep", side_effect=KeyboardInterrupt):
            result = runner.invoke(cli, ["auth", "web"])
        assert result.exit_code == 0
        assert "Cancelled" in result.output
        assert fake_server._shutdown_flag is True

    def test_timeout_errors(self, runner: CliRunner):
        fake_server = self._fake_server(_timed_out=True)
        with patch("djcues.auth_web.render_auth_setup_html", return_value="<html></html>"), \
             patch("djcues.server.start_auth_server", return_value=(fake_server, 5555)), \
             patch("webbrowser.open"):
            result = runner.invoke(cli, ["auth", "web"])
        assert result.exit_code == 1
        assert "Timed out" in result.output


# ---------------------------------------------------------------------------
# playlist add/remove/move -- djcues's first commands that write to the
# database. djcues.writer's actual write functions are always mocked here
# (they have their own real-logic tests in test_writer.py); these tests
# only cover cli.py's own resolution/dispatch/error-mapping layer:
# find_playlist -> load_playlist_tracks -> substring match (shared with
# propose/compare via the same djcues.cli.find_playlist/load_playlist_tracks
# mock points), and mapping djcues.writer's exception hierarchy to exit
# codes and messages.
# ---------------------------------------------------------------------------


class TestPlaylistCommands:
    def test_add_source_playlist_not_found(self, runner: CliRunner):
        with patch("djcues.cli.find_playlist", return_value=None):
            result = runner.invoke(cli, ["playlist", "add", "Nope", "Track", "Dest"])
        assert result.exit_code == 1
        assert "'Nope' not found" in result.output

    def test_add_dest_playlist_not_found(self, runner: CliRunner, track: Track):
        def _find(name):
            return _mock_playlist(name=name) if name == "Source" else None

        with patch("djcues.cli.find_playlist", side_effect=_find), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]):
            result = runner.invoke(cli, ["playlist", "add", "Source", "Test Track", "NopeDest"])
        assert result.exit_code == 1
        assert "'NopeDest' not found" in result.output

    def test_add_no_track_matches(self, runner: CliRunner, track: Track):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]):
            result = runner.invoke(cli, ["playlist", "add", "Source", "Nonexistent", "Dest"])
        assert result.exit_code == 1
        assert "no track matching" in result.output

    def test_add_ambiguous_track_name_lists_candidates(
        self, runner: CliRunner, track: Track, beat_grid: BeatGrid, phrases: list[Phrase]
    ):
        track2 = Track(
            id=3, title="Test Track Two", artist="Other Artist", bpm=128.0,
            duration_ms=200_000.0, analysis_path="", cues=[], phrases=phrases, beat_grid=beat_grid,
        )
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track, track2]):
            result = runner.invoke(cli, ["playlist", "add", "Source", "Test Track", "Dest"])
        assert result.exit_code == 1
        assert "2 tracks match" in result.output
        assert "Test Track -- Test Artist" in result.output
        assert "Test Track Two -- Other Artist" in result.output

    def test_add_happy_path_calls_writer(self, runner: CliRunner, track: Track):
        dest = _mock_playlist(playlist_id=99, name="Dest")
        source = _mock_playlist(playlist_id=1, name="Source")

        def _find(name):
            return dest if name == "Dest" else source

        with patch("djcues.cli.find_playlist", side_effect=_find), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.writer.add_track_to_playlist") as mock_add:
            result = runner.invoke(cli, ["playlist", "add", "Source", "Test Track", "Dest"])
        assert result.exit_code == 0
        assert "Added 'Test Track' to 'Dest'." in result.output
        mock_add.assert_called_once_with(99, 1)

    def test_remove_happy_path_calls_writer_with_position(self, runner: CliRunner, track: Track):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist(name="Source")), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.writer.remove_track_from_playlist") as mock_remove:
            result = runner.invoke(cli, ["playlist", "remove", "Source", "Test Track", "--position", "2"])
        assert result.exit_code == 0
        assert "Removed 'Test Track' from 'Source'." in result.output
        mock_remove.assert_called_once_with(1, 1, position=2)

    def test_remove_writer_track_not_in_playlist_error(self, runner: CliRunner, track: Track):
        from djcues.writer import TrackNotInPlaylistError

        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch(
                 "djcues.writer.remove_track_from_playlist",
                 side_effect=TrackNotInPlaylistError("track 1 is not in playlist 1"),
             ):
            result = runner.invoke(cli, ["playlist", "remove", "Source", "Test Track"])
        assert result.exit_code == 1
        assert "track 1 is not in playlist 1" in result.output

    def test_remove_writer_ambiguous_entry_error_lists_candidates(self, runner: CliRunner, track: Track):
        from djcues.writer import AmbiguousTrackEntryError

        err = AmbiguousTrackEntryError(
            "track 1 appears 2 times in playlist 1", candidates=[(1, "entry-1"), (3, "entry-3")]
        )
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.writer.remove_track_from_playlist", side_effect=err):
            result = runner.invoke(cli, ["playlist", "remove", "Source", "Test Track"])
        assert result.exit_code == 1
        assert "appears 2 times" in result.output
        assert "1: entry-1" in result.output
        assert "3: entry-3" in result.output

    def test_remove_writer_rekordbox_running_error(self, runner: CliRunner, track: Track):
        from djcues.writer import RekordboxRunningError

        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch(
                 "djcues.writer.remove_track_from_playlist",
                 side_effect=RekordboxRunningError("Rekordbox is running. Close it before moving/adding/removing playlist tracks."),
             ):
            result = runner.invoke(cli, ["playlist", "remove", "Source", "Test Track"])
        assert result.exit_code == 1
        assert "Rekordbox is running" in result.output

    def test_move_dest_playlist_not_found(self, runner: CliRunner, track: Track):
        def _find(name):
            return _mock_playlist(name=name) if name == "Source" else None

        with patch("djcues.cli.find_playlist", side_effect=_find), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.writer.move_track_between_playlists") as mock_move:
            result = runner.invoke(cli, ["playlist", "move", "Source", "Test Track", "NopeDest"])
        assert result.exit_code == 1
        assert "'NopeDest' not found" in result.output
        mock_move.assert_not_called()

    def test_move_happy_path_calls_writer(self, runner: CliRunner, track: Track):
        dest = _mock_playlist(playlist_id=99, name="Dest")
        source = _mock_playlist(playlist_id=1, name="Source")

        def _find(name):
            return dest if name == "Dest" else source

        with patch("djcues.cli.find_playlist", side_effect=_find), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.writer.move_track_between_playlists") as mock_move:
            result = runner.invoke(cli, ["playlist", "move", "Source", "Test Track", "Dest", "--position", "3"])
        assert result.exit_code == 0
        assert "Moved 'Test Track' from 'Source' to 'Dest'." in result.output
        mock_move.assert_called_once_with(1, 99, 1, position=3)

    def test_move_source_equals_dest_error_from_writer_surfaces(self, runner: CliRunner, track: Track):
        # writer.move_track_between_playlists is the actual source of this
        # rejection (see test_writer.py) -- this just confirms the CLI
        # surfaces it as a normal Error/exit 1, not an unhandled traceback.
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist(name="Same")), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch(
                 "djcues.writer.move_track_between_playlists",
                 side_effect=ValueError("source and destination playlists are the same"),
             ):
            result = runner.invoke(cli, ["playlist", "move", "Same", "Test Track", "Same"])
        assert result.exit_code == 1
        assert "source and destination playlists are the same" in result.output


# ---------------------------------------------------------------------------
# suggest -- harmonic mixing suggestions. djcues.harmony's own logic
# (Camelot compatibility, BPM tolerance) has its own exhaustive tests in
# test_harmony.py; these only cover cli.py's resolution/dispatch/
# presentation layer, using real TrackSummary fixtures (cheap, no
# ANLZ-reading Track needed) mirroring TestPlaylistCommands's mocking shape.
# ---------------------------------------------------------------------------


def _summary(
    id_: str, title: str, artist: str = "Artist", bpm: float = 128.0,
    key: str | None = "8A", comment: str | None = None,
) -> TrackSummary:
    return TrackSummary(
        id=id_, track_no=None, title=title, artist=artist, bpm=bpm,
        duration_ms=200_000.0, key=key, comment=comment,
    )


class TestSuggest:
    def test_playlist_not_found(self, runner: CliRunner):
        with patch("djcues.cli.find_playlist", return_value=None):
            result = runner.invoke(cli, ["suggest", "Nope", "Track"])
        assert result.exit_code == 1
        assert "'Nope' not found" in result.output

    def test_no_track_matches(self, runner: CliRunner):
        ref = _summary("1", "Some Track")
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=[ref]):
            result = runner.invoke(cli, ["suggest", "Playlist", "Nonexistent"])
        assert result.exit_code == 1
        assert "no track matching" in result.output

    def test_ambiguous_track_matches_lists_candidates(self, runner: CliRunner):
        t1 = _summary("1", "Test Track", artist="Artist A")
        t2 = _summary("2", "Test Track Two", artist="Artist B")
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=[t1, t2]):
            result = runner.invoke(cli, ["suggest", "Playlist", "Test Track"])
        assert result.exit_code == 1
        assert "2 tracks match" in result.output
        assert "Test Track -- Artist A" in result.output
        assert "Test Track Two -- Artist B" in result.output

    def test_reference_with_no_key_errors_clearly(self, runner: CliRunner):
        ref = _summary("1", "Test Track", key=None)
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=[ref]):
            result = runner.invoke(cli, ["suggest", "Playlist", "Test Track"])
        assert result.exit_code == 1
        assert "no usable Camelot key" in result.output

    def test_happy_path_ranked_output(self, runner: CliRunner):
        ref = _summary("1", "Reference Track", bpm=128.0, key="8A")
        same_key = _summary("2", "Same Key Match", bpm=128.0, key="8A")
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=[ref, same_key]):
            result = runner.invoke(cli, ["suggest", "Playlist", "Reference Track"])
        assert result.exit_code == 0
        assert "Reference Track" in result.output
        assert "Same Key Match" in result.output
        assert "Same key" in result.output
        assert "Same tempo" in result.output

    def test_no_matches_found_message(self, runner: CliRunner):
        ref = _summary("1", "Reference Track", bpm=128.0, key="8A")
        far = _summary("2", "Far Off", bpm=145.0, key="3B")
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=[ref, far]):
            result = runner.invoke(cli, ["suggest", "Playlist", "Reference Track"])
        assert result.exit_code == 0
        assert "No compatible tracks found." in result.output

    def test_library_flag_uses_list_all_tracks(self, runner: CliRunner):
        ref = _summary("1", "Reference Track", bpm=128.0, key="8A")
        library_track = _summary("2", "Library Match", bpm=128.0, key="8A")
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=[ref]), \
             patch("djcues.cli.list_all_tracks", return_value=[ref, library_track]) as mock_all:
            result = runner.invoke(cli, ["suggest", "Playlist", "Reference Track", "--library"])
        assert result.exit_code == 0
        mock_all.assert_called_once()
        assert "Library Match" in result.output
        assert "whole library" in result.output

    def test_without_library_flag_never_calls_list_all_tracks(self, runner: CliRunner):
        ref = _summary("1", "Reference Track", bpm=128.0, key="8A")
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=[ref]), \
             patch("djcues.cli.list_all_tracks") as mock_all:
            result = runner.invoke(cli, ["suggest", "Playlist", "Reference Track"])
        assert result.exit_code == 0
        mock_all.assert_not_called()

    def test_bpm_tolerance_and_no_half_double_and_limit_passed_through(self, runner: CliRunner):
        ref = _summary("1", "Reference Track", bpm=128.0, key="8A")
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=[ref]), \
             patch("djcues.cli.suggest_compatible_tracks") as mock_suggest:
            mock_suggest.return_value = MagicMock(reference=ref, suggestions=[], excluded=[])
            result = runner.invoke(cli, [
                "suggest", "Playlist", "Reference Track",
                "--bpm-tolerance", "10", "--no-half-double", "--limit", "3",
            ])
        assert result.exit_code == 0
        mock_suggest.assert_called_once()
        _args, kwargs = mock_suggest.call_args
        assert kwargs["bpm_tolerance_pct"] == 10.0
        assert kwargs["allow_half_double"] is False
        assert kwargs["limit"] == 3

    def test_excluded_tracks_summary_line_printed(self, runner: CliRunner):
        ref = _summary("1", "Reference Track", bpm=128.0, key="8A")
        no_key = _summary("2", "No Key Track", bpm=128.0, key=None)
        non_camelot = _summary("3", "Weird Key Track", bpm=128.0, key="Dm")
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=[ref, no_key, non_camelot]):
            result = runner.invoke(cli, ["suggest", "Playlist", "Reference Track"])
        assert result.exit_code == 0
        assert "excluded: 1 no-key, 1 non-Camelot key" in result.output

    def test_encrypted_metadata_track_excluded_not_shown_as_garbage(self, runner: CliRunner):
        # Real Rekordbox behavior: a streaming-linked (e.g. Spotify) track
        # can have a real Key/BPM but an encrypted Title -- must never be
        # printed as a suggestion, and must be counted separately in the
        # excluded summary rather than folded into "no-key"/"non-Camelot".
        ref = _summary("1", "Reference Track", bpm=128.0, key="8A")
        encrypted = _summary(
            "2", "$A7:v1:cXRGX5Nhvaa9rt8efO7O4g==:9qEcCSYCFHtls1UhP1IHWTd312u5L59/27m9UzFbZs4=",
            bpm=128.0, key="8A",
        )
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=[ref, encrypted]):
            result = runner.invoke(cli, ["suggest", "Playlist", "Reference Track"])
        assert result.exit_code == 0
        assert "$A7:" not in result.output


# ---------------------------------------------------------------------------
# audit -- BPM/Key data-quality report. djcues.audit's own logic (comment-
# hint parsing, the two real confirmed cases) has its own exhaustive tests
# in test_audit.py; these only cover cli.py's own scope-resolution/
# dispatch/presentation layer.
# ---------------------------------------------------------------------------


class TestAudit:
    def test_playlist_not_found(self, runner: CliRunner):
        with patch("djcues.cli.find_playlist", return_value=None):
            result = runner.invoke(cli, ["audit", "Nope"])
        assert result.exit_code == 1
        assert "'Nope' not found" in result.output

    def test_neither_playlist_name_nor_library_is_an_error(self, runner: CliRunner):
        result = runner.invoke(cli, ["audit"])
        assert result.exit_code == 1
        assert "provide a playlist name or use --library" in result.output

    def test_library_flag_uses_list_all_tracks(self, runner: CliRunner):
        track = _summary("1", "Track", bpm=128.0, key="8A")
        with patch("djcues.cli.list_all_tracks", return_value=[track]) as mock_all:
            result = runner.invoke(cli, ["audit", "--library"])
        assert result.exit_code == 0
        mock_all.assert_called_once()
        assert "whole library" in result.output
        assert "1 tracks scanned" in result.output

    def test_without_library_flag_never_calls_list_all_tracks(self, runner: CliRunner):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=[]), \
             patch("djcues.cli.list_all_tracks") as mock_all:
            result = runner.invoke(cli, ["audit", "Playlist"])
        assert result.exit_code == 0
        mock_all.assert_not_called()

    def test_dreamin_shaped_finding_printed_with_both_bpm_values_and_comment(self, runner: CliRunner):
        dreamin = _summary("1", "Dreamin' Original Mix", bpm=146.92, key="2A", comment="2A - 118")
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=[dreamin]):
            result = runner.invoke(cli, ["audit", "Playlist"])
        assert result.exit_code == 0
        assert "Dreamin' Original Mix" in result.output
        assert "146.9" in result.output
        assert "118.0" in result.output
        assert "2A - 118" in result.output

    def test_no_findings_message(self, runner: CliRunner):
        agreeing = _summary("1", "Track", bpm=128.0, key="8A", comment="8A - 128")
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=[agreeing]):
            result = runner.invoke(cli, ["audit", "Playlist"])
        assert result.exit_code == 0
        assert "No comment-hint disagreements found." in result.output

    def test_unusable_keys_summary_line_printed(self, runner: CliRunner):
        no_key = _summary("1", "Track A", bpm=128.0, key=None)
        non_camelot = _summary("2", "Track B", bpm=128.0, key="Dm")
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=[no_key, non_camelot]):
            result = runner.invoke(cli, ["audit", "Playlist"])
        assert result.exit_code == 0
        assert "unusable keys: 1 no-key, 1 non-Camelot key" in result.output

    def test_bpm_tolerance_and_no_half_double_passed_through(self, runner: CliRunner):
        track = _summary("1", "Track", bpm=128.0, key="8A")
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=[track]), \
             patch("djcues.cli.audit_tracks") as mock_audit:
            mock_audit.return_value = MagicMock(scanned=1, comment_hint_count=0, findings=[], unusable_keys=[])
            result = runner.invoke(cli, ["audit", "Playlist", "--bpm-tolerance", "10", "--no-half-double"])
        assert result.exit_code == 0
        mock_audit.assert_called_once()
        _args, kwargs = mock_audit.call_args
        assert kwargs["bpm_tolerance_pct"] == 10.0
        assert kwargs["allow_half_double"] is False


# ---------------------------------------------------------------------------
# flow -- energy-flow set ordering. djcues.flow's own logic (the
# peak-then-cooldown algorithm, the worked example) has its own exhaustive
# tests in test_flow.py; these only cover cli.py's own resolution/loading/
# dispatch/presentation layer.
# ---------------------------------------------------------------------------


def _flow_track(id_: int, title: str, energy: float, duration_ms: float = 200_000.0) -> Track:
    """A Track with a single phrase spanning its whole duration at a
    uniform waveform height -- mirrors test_flow.py's own _uniform_track
    helper, so mean_energy == peak_energy == energy exactly."""
    phrase = Phrase(beat_start=1, beat_end=2, kind=1, label="Intro", position_ms=0.0, duration_ms=duration_ms)
    waveform = [WaveformPoint(height=energy, red=4, green=4, blue=4) for _ in range(10)]
    return Track(
        id=id_, title=title, artist="Artist", bpm=128.0, duration_ms=duration_ms,
        analysis_path="", cues=[], phrases=[phrase], beat_grid=BeatGrid(first_beat_ms=0.0, bpm=128.0),
        waveform=waveform,
    )


class TestFlow:
    def test_playlist_not_found(self, runner: CliRunner):
        with patch("djcues.cli.find_playlist", return_value=None):
            result = runner.invoke(cli, ["flow", "Nope"])
        assert result.exit_code == 1
        assert "'Nope' not found" in result.output

    def test_empty_playlist_errors_before_expensive_load(self, runner: CliRunner):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=[]), \
             patch("djcues.cli.load_playlist_tracks") as mock_load:
            result = runner.invoke(cli, ["flow", "Playlist"])
        assert result.exit_code == 1
        assert "no tracks found" in result.output
        mock_load.assert_not_called()

    def test_loading_message_shows_track_count(self, runner: CliRunner):
        summaries = [_summary("1", "A"), _summary("2", "B")]
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=summaries), \
             patch("djcues.cli.load_playlist_tracks", return_value=[]):
            result = runner.invoke(cli, ["flow", "Playlist"])
        assert result.exit_code == 0
        assert "Loading 2 track(s) from 'Playlist'" in result.output

    def test_happy_path_shows_order_and_positions(self, runner: CliRunner):
        summaries = [
            TrackSummary(id="1", track_no=3, title="Calm", artist="Artist", bpm=128.0, duration_ms=200_000.0, key="8A"),
            TrackSummary(id="2", track_no=1, title="Loud", artist="Artist", bpm=128.0, duration_ms=200_000.0, key="3B"),
        ]
        tracks = [_flow_track(1, "Calm", 0.2), _flow_track(2, "Loud", 0.9)]
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=summaries), \
             patch("djcues.cli.load_playlist_tracks", return_value=tracks):
            result = runner.invoke(cli, ["flow", "Playlist"])
        assert result.exit_code == 0
        assert "Calm" in result.output
        assert "Loud" in result.output
        assert "Key" in result.output  # column header
        # Calm (lower energy) is the opener at new position 1, was position 3;
        # Loud was position 1, now ordered second.
        calm_line = next(line for line in result.output.splitlines() if "Calm" in line)
        loud_line = next(line for line in result.output.splitlines() if "Loud" in line)
        assert calm_line.strip().startswith("1")
        assert "3" in calm_line
        assert "8A" in calm_line
        assert loud_line.strip().startswith("2")
        assert "1" in loud_line
        assert "3B" in loud_line

    def test_cooldown_fraction_passed_through(self, runner: CliRunner):
        summaries = [_summary("1", "A")]
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=summaries), \
             patch("djcues.cli.load_playlist_tracks", return_value=[]), \
             patch("djcues.cli.suggest_energy_flow") as mock_flow:
            mock_flow.return_value = MagicMock(ordered_tracks=[], cooldown_start_index=0, unscored=[])
            result = runner.invoke(cli, ["flow", "Playlist", "--cooldown-fraction", "0.3"])
        assert result.exit_code == 0
        mock_flow.assert_called_once()
        _args, kwargs = mock_flow.call_args
        assert kwargs["cooldown_fraction"] == 0.3

    def test_skipped_tracks_printed_with_reason(self, runner: CliRunner):
        no_phrases = Track(
            id=1, title="No Phrases Track", artist="Artist", bpm=128.0, duration_ms=200_000.0,
            analysis_path="", cues=[], phrases=[], beat_grid=BeatGrid(first_beat_ms=0.0, bpm=128.0),
        )
        summaries = [_summary("1", "No Phrases Track")]
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=summaries), \
             patch("djcues.cli.load_playlist_tracks", return_value=[no_phrases]):
            result = runner.invoke(cli, ["flow", "Playlist"])
        assert result.exit_code == 0
        assert "Skipping No Phrases Track (no phrase data)" in result.output

    def test_no_scorable_tracks_message(self, runner: CliRunner):
        no_phrases = Track(
            id=1, title="No Phrases Track", artist="Artist", bpm=128.0, duration_ms=200_000.0,
            analysis_path="", cues=[], phrases=[], beat_grid=BeatGrid(first_beat_ms=0.0, bpm=128.0),
        )
        summaries = [_summary("1", "No Phrases Track")]
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=summaries), \
             patch("djcues.cli.load_playlist_tracks", return_value=[no_phrases]):
            result = runner.invoke(cli, ["flow", "Playlist"])
        assert result.exit_code == 0
        assert "No scorable tracks to order." in result.output


# ---------------------------------------------------------------------------
# playlist reorder -- writes flow's computed order into the real playlist.
# writer.reorder_playlist's own logic (the shift algorithm, the tracking-
# bug guard, the duplicate-content-id fix) has its own exhaustive tests in
# test_writer.py; these only cover cli.py's own loading/preview/
# confirmation/dispatch layer.
# ---------------------------------------------------------------------------


class TestPlaylistReorder:
    def test_playlist_not_found(self, runner: CliRunner):
        with patch("djcues.cli.find_playlist", return_value=None):
            result = runner.invoke(cli, ["playlist", "reorder", "Nope"])
        assert result.exit_code == 1
        assert "'Nope' not found" in result.output

    def test_empty_playlist_errors_before_expensive_load(self, runner: CliRunner):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=[]), \
             patch("djcues.cli.load_playlist_tracks") as mock_load:
            result = runner.invoke(cli, ["playlist", "reorder", "Playlist"])
        assert result.exit_code == 1
        assert "no tracks found" in result.output
        mock_load.assert_not_called()

    def test_preview_output_shown_before_confirmation(self, runner: CliRunner):
        summaries = [_summary("1", "A"), _summary("2", "B")]
        tracks = [_flow_track(1, "A", 0.2), _flow_track(2, "B", 0.9)]
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=summaries), \
             patch("djcues.cli.load_playlist_tracks", return_value=tracks), \
             patch("click.confirm", return_value=False):
            result = runner.invoke(cli, ["playlist", "reorder", "Playlist"])
        assert "Energy Flow: Playlist" in result.output
        assert "A" in result.output and "B" in result.output

    def test_confirmation_prompt_gates_the_write(self, runner: CliRunner):
        summaries = [_summary("1", "A"), _summary("2", "B")]
        tracks = [_flow_track(1, "A", 0.2), _flow_track(2, "B", 0.9)]
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=summaries), \
             patch("djcues.cli.load_playlist_tracks", return_value=tracks), \
             patch("djcues.writer.reorder_playlist") as mock_reorder, \
             patch("click.confirm", return_value=False):
            result = runner.invoke(cli, ["playlist", "reorder", "Playlist"])
        assert result.exit_code == 0
        assert "Aborted." in result.output
        mock_reorder.assert_not_called()

    def test_confirmed_calls_writer_with_full_target_order(self, runner: CliRunner):
        summaries = [_summary("1", "A"), _summary("2", "B")]
        tracks = [_flow_track(1, "A", 0.2), _flow_track(2, "B", 0.9)]
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist(playlist_id=42)), \
             patch("djcues.cli.list_playlist_tracks", return_value=summaries), \
             patch("djcues.cli.load_playlist_tracks", return_value=tracks), \
             patch("djcues.writer.reorder_playlist") as mock_reorder, \
             patch("click.confirm", return_value=True):
            result = runner.invoke(cli, ["playlist", "reorder", "Playlist"])
        assert result.exit_code == 0
        assert "Wrote new order to 'Playlist'." in result.output
        mock_reorder.assert_called_once_with(42, [1, 2], db=None)

    def test_force_flag_skips_confirmation(self, runner: CliRunner):
        summaries = [_summary("1", "A")]
        tracks = [_flow_track(1, "A", 0.5)]
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=summaries), \
             patch("djcues.cli.load_playlist_tracks", return_value=tracks), \
             patch("djcues.writer.reorder_playlist") as mock_reorder, \
             patch("click.confirm") as mock_confirm:
            result = runner.invoke(cli, ["playlist", "reorder", "Playlist", "--force"])
        assert result.exit_code == 0
        mock_confirm.assert_not_called()
        mock_reorder.assert_called_once()

    def test_unscored_tracks_appended_at_end_of_target_order(self, runner: CliRunner):
        no_phrases = Track(
            id=3, title="No Phrases Track", artist="Artist", bpm=128.0, duration_ms=200_000.0,
            analysis_path="", cues=[], phrases=[], beat_grid=BeatGrid(first_beat_ms=0.0, bpm=128.0),
        )
        summaries = [_summary("1", "A"), _summary("2", "B"), _summary("3", "No Phrases Track")]
        tracks = [_flow_track(1, "A", 0.2), _flow_track(2, "B", 0.9), no_phrases]
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist(playlist_id=42)), \
             patch("djcues.cli.list_playlist_tracks", return_value=summaries), \
             patch("djcues.cli.load_playlist_tracks", return_value=tracks), \
             patch("djcues.writer.reorder_playlist") as mock_reorder, \
             patch("click.confirm", return_value=True):
            result = runner.invoke(cli, ["playlist", "reorder", "Playlist"])
        assert result.exit_code == 0
        assert "unscored track(s) appended at the end" in result.output
        mock_reorder.assert_called_once_with(42, [1, 2, 3], db=None)

    def test_cooldown_fraction_passed_through(self, runner: CliRunner):
        summaries = [_summary("1", "A")]
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=summaries), \
             patch("djcues.cli.load_playlist_tracks", return_value=[]), \
             patch("djcues.cli.suggest_energy_flow") as mock_flow, \
             patch("click.confirm", return_value=False):
            mock_flow.return_value = MagicMock(ordered_tracks=[], cooldown_start_index=0, unscored=[])
            result = runner.invoke(cli, ["playlist", "reorder", "Playlist", "--cooldown-fraction", "0.3"])
        assert result.exit_code == 0
        _args, kwargs = mock_flow.call_args
        assert kwargs["cooldown_fraction"] == 0.3

    def test_writer_rekordbox_running_error_passthrough(self, runner: CliRunner):
        from djcues.writer import RekordboxRunningError

        summaries = [_summary("1", "A")]
        tracks = [_flow_track(1, "A", 0.5)]
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=summaries), \
             patch("djcues.cli.load_playlist_tracks", return_value=tracks), \
             patch(
                 "djcues.writer.reorder_playlist",
                 side_effect=RekordboxRunningError("Rekordbox is running."),
             ), \
             patch("click.confirm", return_value=True):
            result = runner.invoke(cli, ["playlist", "reorder", "Playlist", "--force"])
        assert result.exit_code == 1
        assert "Rekordbox is running" in result.output

    def test_writer_value_error_passthrough(self, runner: CliRunner):
        summaries = [_summary("1", "A")]
        tracks = [_flow_track(1, "A", 0.5)]
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=summaries), \
             patch("djcues.cli.load_playlist_tracks", return_value=tracks), \
             patch(
                 "djcues.writer.reorder_playlist",
                 side_effect=ValueError("the playlist changed since this order was computed"),
             ), \
             patch("click.confirm", return_value=True):
            result = runner.invoke(cli, ["playlist", "reorder", "Playlist", "--force"])
        assert result.exit_code == 1
        assert "changed since this order was computed" in result.output


# ---------------------------------------------------------------------------
# clash -- vocal-clash detection. djcues.clash's own logic (the pairwise
# algorithm, the real anchored pairs) has its own exhaustive tests in
# test_clash.py; these only cover cli.py's own resolution/loading/
# reordering/dispatch/presentation layer.
# ---------------------------------------------------------------------------


_CLASH_PHRASES = [
    Phrase(beat_start=1, beat_end=2, kind=1, label="Intro", position_ms=0.0, duration_ms=20_000.0),
    Phrase(beat_start=2, beat_end=3, kind=5, label="Chorus", position_ms=20_000.0, duration_ms=130_000.0),
    Phrase(beat_start=3, beat_end=4, kind=6, label="Outro", position_ms=150_000.0, duration_ms=50_000.0),
]


def _clash_vocal_track(windows: tuple[tuple[float, float], ...] = ()) -> list[int]:
    frame_ms = 1024 / 22050 * 1000
    n_frames = int(200_000.0 / frame_ms) + 1
    vt = [0] * n_frames
    for start_ms, end_ms in windows:
        i0, i1 = int(start_ms / frame_ms), int(end_ms / frame_ms)
        for i in range(i0, min(i1, n_frames)):
            vt[i] = 4
    return vt


def _clash_track(id_: int, title: str, *, vocal_windows: tuple[tuple[float, float], ...] = ()) -> Track:
    """A 200s track with a standard Intro (0-20s)/Outro (150-200s)
    structure -- mirrors test_clash.py's own _scorable_track helper."""
    return Track(
        id=id_, title=title, artist="Artist", bpm=128.0, duration_ms=200_000.0,
        analysis_path="", cues=[], phrases=list(_CLASH_PHRASES), beat_grid=BeatGrid(first_beat_ms=0.0, bpm=128.0),
        vocal_track=_clash_vocal_track(vocal_windows),
    )


class TestClash:
    def test_playlist_not_found(self, runner: CliRunner):
        with patch("djcues.cli.find_playlist", return_value=None):
            result = runner.invoke(cli, ["clash", "Nope"])
        assert result.exit_code == 1
        assert "'Nope' not found" in result.output

    def test_empty_playlist_errors_before_expensive_load(self, runner: CliRunner):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=[]), \
             patch("djcues.cli.load_playlist_tracks") as mock_load:
            result = runner.invoke(cli, ["clash", "Playlist"])
        assert result.exit_code == 1
        assert "no tracks found" in result.output
        mock_load.assert_not_called()

    def test_loading_message_shows_track_count(self, runner: CliRunner):
        summaries = [_summary("1", "A"), _summary("2", "B")]
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=summaries), \
             patch("djcues.cli.load_playlist_tracks", return_value=[]):
            result = runner.invoke(cli, ["clash", "Playlist"])
        assert result.exit_code == 0
        assert "Loading 2 track(s) from 'Playlist'" in result.output

    def test_tracks_are_reordered_to_match_list_playlist_tracks(self, runner: CliRunner):
        # The real bug this command must work around: load_playlist_tracks()
        # does not guarantee TrackNo order. Here it deliberately returns
        # tracks in the OPPOSITE order from list_playlist_tracks()'s real
        # (TrackNo-sorted) order -- find_vocal_clashes() must still be
        # called with the corrected, summaries-matching order.
        summaries = [_summary("1", "First"), _summary("2", "Second")]
        track_1 = _clash_track(1, "First")
        track_2 = _clash_track(2, "Second")
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=summaries), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track_2, track_1]), \
             patch("djcues.cli.find_vocal_clashes") as mock_clash:
            mock_clash.return_value = MagicMock(scanned_pairs=1, findings=[], unscorable=[])
            result = runner.invoke(cli, ["clash", "Playlist"])
        assert result.exit_code == 0
        mock_clash.assert_called_once()
        args, _kwargs = mock_clash.call_args
        assert [t.id for t in args[0]] == [1, 2]

    def test_min_vocal_region_ms_passthrough(self, runner: CliRunner):
        summaries = [_summary("1", "A")]
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=summaries), \
             patch("djcues.cli.load_playlist_tracks", return_value=[_clash_track(1, "A")]), \
             patch("djcues.cli.find_vocal_clashes") as mock_clash:
            mock_clash.return_value = MagicMock(scanned_pairs=0, findings=[], unscorable=[])
            result = runner.invoke(cli, ["clash", "Playlist", "--min-vocal-region-ms", "500"])
        assert result.exit_code == 0
        mock_clash.assert_called_once()
        _args, kwargs = mock_clash.call_args
        assert kwargs["min_vocal_region_ms"] == 500.0

    def test_no_findings_message(self, runner: CliRunner):
        summaries = [_summary("1", "A"), _summary("2", "B")]
        a, b = _clash_track(1, "A"), _clash_track(2, "B")  # both clean
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=summaries), \
             patch("djcues.cli.load_playlist_tracks", return_value=[a, b]):
            result = runner.invoke(cli, ["clash", "Playlist"])
        assert result.exit_code == 0
        assert "No vocal clashes found." in result.output

    def test_finding_output_formatting(self, runner: CliRunner):
        # Built from a real ClashResult/ClashFinding/VocalRegion (not a
        # MagicMock) so _print_clash_report's actual formatting code path
        # -- including _format_time -- runs for real.
        from djcues.clash import ClashFinding, ClashResult
        from djcues.models import VocalRegion

        summaries = [_summary("1", "Track A"), _summary("2", "Track B")]
        a, b = _clash_track(1, "Track A"), _clash_track(2, "Track B")
        finding = ClashFinding(
            track_a=a, track_b=b, position_a=1, position_b=2,
            regions_a=[VocalRegion(start_ms=160_000.0, end_ms=165_000.0)],
            regions_b=[VocalRegion(start_ms=2_000.0, end_ms=7_000.0)],
        )
        fake_result = ClashResult(scanned_pairs=1, findings=[finding], unscorable=[])
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=summaries), \
             patch("djcues.cli.load_playlist_tracks", return_value=[a, b]), \
             patch("djcues.cli.find_vocal_clashes", return_value=fake_result):
            result = runner.invoke(cli, ["clash", "Playlist"])
        assert result.exit_code == 0
        assert "[1] Track A → [2] Track B" in result.output
        assert "2:40.0-2:45.0" in result.output  # 160_000ms-165_000ms
        assert "0:02.0-0:07.0" in result.output  # 2_000ms-7_000ms

    def test_unscorable_tracks_skipped_with_reason(self, runner: CliRunner):
        no_phrases = Track(
            id=1, title="No Phrases Track", artist="Artist", bpm=128.0, duration_ms=200_000.0,
            analysis_path="", cues=[], phrases=[], beat_grid=BeatGrid(first_beat_ms=0.0, bpm=128.0),
            vocal_track=_clash_vocal_track(),
        )
        summaries = [_summary("1", "No Phrases Track")]
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.list_playlist_tracks", return_value=summaries), \
             patch("djcues.cli.load_playlist_tracks", return_value=[no_phrases]):
            result = runner.invoke(cli, ["clash", "Playlist"])
        assert result.exit_code == 0
        assert "Skipping No Phrases Track (no phrase data)" in result.output
