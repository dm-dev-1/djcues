"""Agentic multi-agent cue analysis -- an LLM-based alternative to
CueStrategy's local heuristic, using the user's own API key (BYOK).

Always runs the heuristic first, as both an anchor/reference value every
specialist sees and the fallback for any pad whose specialist call fails.
Produces a standard CueProposal so it's a drop-in replacement for
CueStrategy.propose() everywhere downstream -- review/apply/compare/
metrics/history all work unmodified.

Four LLM calls per track: three parallel specialists (Structure: D/E/G,
Vocal: C, Energy: F) plus a Critic pass that may only adjust confidence
and notes, never reposition a cue -- the same discipline the heuristic
itself learned the hard way (see strategy.py's _spectral_similarity
docstring). Slots A, B, H never get an LLM call: A is closed-form
beat-grid math, B is defined as "= A", and H already has a dedicated
confidence signal (spectral similarity) with no new information an LLM
could add from the same inputs.

No raw audio and no raw per-frame arrays ever leave this machine --
only the same compact structured summary (phrases, a condensed vocal-
onset region list, per-phrase energy) the heuristic itself computes
from Rekordbox's analysis data.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field

from djcues.models import CueProposal, Track
from djcues.providers import GenerationResult, ModelProvider, estimate_cost as price_estimate
from djcues.strategy import (
    CueStrategy,
    build_cue_points,
    compute_phrase_energy,
    find_drop_candidates,
    find_energy_recoveries,
)

_VOCAL_FRAME_MS = 1024 / 22050 * 1000  # ~46.4ms per PVDI frame, matches strategy.py
_MIN_VOCAL_REGION_MS = 2000  # matches strategy.py's C-slot min_frames threshold

# No provider offers a way to count hypothetical *output* tokens before
# generation -- only input. Output tokens for --estimate-only stay a
# static per-call approximation; the structured-output schemas are small
# and bounded (a phrase index, a confidence float, a short reasoning
# string), so this is a reasonable one, unlike guessing input size too.
_ESTIMATED_OUTPUT_TOKENS_PER_CALL = 150
_CALLS_PER_TRACK_WITH_CRITIC = 4
_CALLS_PER_TRACK_NO_CRITIC = 3


@dataclass
class AgenticTelemetry:
    """Cost/error accounting for one propose_with_telemetry() call."""

    calls_made: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float | None = None
    errors: list[str] = field(default_factory=list)


def _summarize_vocal_onsets(track: Track) -> list[dict]:
    """Condense the PVDI per-frame vocal-confidence array into a compact
    list of strong-vocal-onset regions, instead of shipping the raw
    per-frame array (which can be thousands of points). Mirrors the
    onset-detection thresholds strategy.py's slot C already uses,
    including its minimum-duration filter -- a region has to sustain for
    at least _MIN_VOCAL_REGION_MS to count as real, so a specialist never
    sees a short blip the heuristic itself would discard as noise."""
    if not track.vocal_track:
        return []
    vt = track.vocal_track
    regions: list[dict] = []
    i = 0
    while i < len(vt):
        if vt[i] >= 3:
            start = i
            while i < len(vt) and vt[i] > 0:
                i += 1
            duration_ms = (i - start) * _VOCAL_FRAME_MS
            if duration_ms >= _MIN_VOCAL_REGION_MS:
                regions.append({
                    "start_ms": round(start * _VOCAL_FRAME_MS),
                    "end_ms": round(i * _VOCAL_FRAME_MS),
                })
        else:
            i += 1
    return regions


def build_track_payload(track: Track, heuristic_proposal: CueProposal) -> dict:
    """Compact, structured summary of a track for LLM specialists.

    No raw audio, no raw per-frame waveform/vocal arrays -- phrases, a
    condensed vocal-onset region list, and per-phrase energy (via the
    same compute_phrase_energy() the heuristic itself uses).
    """
    from djcues.constants import KIND_TO_PAD

    phrases = [
        {
            "index": i,
            "label": p.label,
            "beat_start": p.beat_start,
            "position_ms": round(p.position_ms),
            "duration_ms": round(p.duration_ms),
        }
        for i, p in enumerate(track.phrases)
    ]

    phrase_energy = compute_phrase_energy(track)
    energy_by_index = [
        {"index": i, "mean_energy": round(e, 3)} for i, (_p, e) in enumerate(phrase_energy)
    ]

    heuristic_by_pad = {
        pad: round(conf, 3) for pad, conf in heuristic_proposal.confidence.items()
    }

    # Which phrase (by index into the list above) the heuristic itself
    # picked per pad -- a concrete anchor for a specialist to agree or
    # disagree with, not just a bare confidence number with nothing to
    # compare it against. Heuristic positions always land exactly on a
    # phrase boundary, so matching by position_ms is safe.
    heuristic_phrase_index: dict[str, int | None] = {}
    for cue in heuristic_proposal.hot_cues:
        pad = KIND_TO_PAD.get(cue.kind)
        if pad is None:
            continue
        heuristic_phrase_index[pad] = next(
            (i for i, p in enumerate(track.phrases) if p.position_ms == cue.position_ms), None
        )

    # Pre-computed dip-then-recovery candidates for the Energy/Special
    # specialist to judge, instead of asking it to derive the same
    # multi-step numeric procedure itself from raw energy values -- real
    # testing showed that unreliable. Uses the exact same detection
    # find_energy_recoveries() gives the heuristic, so the LLM's
    # candidate set matches what the heuristic itself would have found.
    drop_cue = next(
        (c for c in heuristic_proposal.hot_cues if KIND_TO_PAD.get(c.kind) == "D"), None
    )
    recovery_candidates: list[dict] = []
    if drop_cue is not None:
        recoveries, _peak, _threshold = find_energy_recoveries(track, drop_cue.position_ms)
        recovery_candidates = [
            {"phrase_index": i, "mean_energy": round(e, 3), "cycle_number": n}
            for n, (i, _p, e) in enumerate(recoveries, start=1)
        ]

    # Pre-computed Drop candidates (already passing the empirically-tuned
    # 20%-of-duration rule, or the correct fallback if none do), for the
    # same reason: real testing on tracks with many (15+) Chorus/Up
    # phrases showed the model unreliably computing/checking that
    # percentage itself when scanning the full phrase list.
    drop_phrase_candidates, _drop_rule = find_drop_candidates(track)
    drop_candidate_ids = {id(p) for p in drop_phrase_candidates}
    drop_candidates_payload = [
        {"phrase_index": i, "label": p.label}
        for i, p in enumerate(track.phrases)
        if id(p) in drop_candidate_ids
    ]

    return {
        "bpm": track.bpm,
        "duration_ms": round(track.duration_ms),
        "phrases": phrases,
        "vocal_regions": _summarize_vocal_onsets(track),
        "phrase_energy": energy_by_index,
        "heuristic_confidence": heuristic_by_pad,
        "heuristic_phrase_index": heuristic_phrase_index,
        "energy_recovery_candidates": recovery_candidates,
        "drop_candidates": drop_candidates_payload,
    }


# --- Specialist schemas -----------------------------------------------
# Positions are always a phrase-list index (or -1 for "no confident
# placement"), never a freeform millisecond float -- mirrors how the
# heuristic always snaps to a real phrase boundary, and prevents an LLM
# from hallucinating an off-grid timestamp.

_PAD_PLACEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "phrase_index": {
            "type": "integer",
            "description": "Index into the provided phrase list, or -1 if no confident placement can be made.",
        },
        "confidence": {"type": "number", "description": "0.0 to 1.0"},
        "reasoning": {"type": "string"},
    },
    "required": ["phrase_index", "confidence", "reasoning"],
    "additionalProperties": False,
}

_STRUCTURE_SCHEMA = {
    "type": "object",
    "properties": {
        "drop": _PAD_PLACEMENT_SCHEMA,
        "breakdown": _PAD_PLACEMENT_SCHEMA,
        "outro": _PAD_PLACEMENT_SCHEMA,
    },
    "required": ["drop", "breakdown", "outro"],
    "additionalProperties": False,
}

_VOCAL_SCHEMA = {
    "type": "object",
    "properties": {"vocal_buildup": _PAD_PLACEMENT_SCHEMA},
    "required": ["vocal_buildup"],
    "additionalProperties": False,
}

_ENERGY_SCHEMA = {
    "type": "object",
    "properties": {"special": _PAD_PLACEMENT_SCHEMA},
    "required": ["special"],
    "additionalProperties": False,
}

_CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "adjustments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pad": {"type": "string", "enum": ["A", "B", "C", "D", "E", "F", "G", "H"]},
                    "confidence": {"type": "number", "description": "Replacement confidence, 0.0-1.0"},
                    "note": {"type": "string", "description": "Why the confidence changed"},
                },
                "required": ["pad", "confidence", "note"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["adjustments"],
    "additionalProperties": False,
}

_STRUCTURE_SYSTEM = (
    "You place 3 DJ cue points on a dance track: Drop (the main energy peak "
    "after the intro), Breakdown (the drop in energy after the Drop), and "
    "Outro (where the track's ending section begins). You are given the "
    "track's phrase structure (intro/up/chorus/down/outro-style labels with "
    "beat positions and position_ms), its total duration_ms, drop_candidates "
    "(phrases already filtered to the ones that qualify for the Drop -- see "
    "below), and the local heuristic's own choice (heuristic_phrase_index) "
    "and confidence (heuristic_confidence) per pad for reference. Pick a "
    "phrase_index for each from the provided list -- never invent a "
    "position outside it.\n\n"
    "Rule for Drop: drop_candidates is already filtered to phrases "
    "satisfying an empirically-measured rule (at least 20% into the track, "
    "or the correct fallback if none qualify) -- pick the FIRST entry in "
    "drop_candidates by default; you do not need to compute or check the "
    "percentage yourself, it's already been done. Only pick a different "
    "phrase_index if you have a clear, specific reason from the phrase "
    "structure, and say why -- don't second-guess the filtering by "
    "re-deriving it from position_ms/duration_ms.\n\n"
    "Rule for Breakdown: it must be a Down- or Bridge-labeled phrase whose "
    "position_ms is strictly after the Drop's position_ms. Among phrases "
    "that qualify, pick the EARLIEST one, not a later one that might look "
    "like a bigger energy drop. If no Down/Bridge phrase exists after the "
    "Drop, decline (phrase_index -1) rather than guessing at an unrelated "
    "phrase."
)

_VOCAL_SYSTEM = (
    "You place a single DJ cue point, Vocal/Buildup, marking where the "
    "pre-drop buildup or first strong vocal begins. You are given the "
    "track's phrase structure, a list of detected vocal regions (each "
    "already filtered to genuine, sustained vocal presence -- at least 2 "
    "seconds long, not brief noise), and the local heuristic's own choice "
    "(heuristic_phrase_index) per pad for reference -- "
    "heuristic_phrase_index.D is the Drop's phrase index.\n\n"
    "Follow this procedure in order:\n"
    "1. If heuristic_phrase_index.D is not null, discard any vocal_regions "
    "entry whose start_ms is at or after the Drop phrase's position_ms -- "
    "Vocal/Buildup must come before the Drop, never after or during it, "
    "no matter how prominent a later vocal moment looks.\n"
    "2. Of what remains, take the EARLIEST vocal_regions entry (by "
    "start_ms) -- not the longest, not the one that best matches a phrase "
    "label, the earliest. Pick the phrase whose position_ms is closest to "
    "that region's start_ms.\n"
    "3. Only if no vocal_regions entries remain after step 1 (or none "
    "exist at all), fall back to the last Up- or Verse-labeled phrase "
    "before the Drop.\n"
    "4. If heuristic_phrase_index.D is null (no Drop identified), use "
    "your own judgment for where a pre-buildup moment would be."
)

_ENERGY_SYSTEM = (
    "You place a single DJ cue point, Special (also called the 'second "
    "drop'): the point where energy returns to near-peak after dipping "
    "following the main Drop. You are given energy_recovery_candidates: "
    "a list of phrases already detected as genuine dip-then-recovery "
    "moments after the Drop (computed from the track's real energy data "
    "-- you do not need to compute this yourself), each with a "
    "cycle_number (1 = first recovery after the Drop, 2 = second, in "
    "track order). You are also given the phrase structure and the "
    "local heuristic's own choice (heuristic_phrase_index) for "
    "reference.\n\n"
    "Rule: if a candidate with cycle_number 2 exists, pick it -- that's "
    "the genuine second drop, not the first energy bump after the main "
    "Drop. If only cycle_number 1 exists, pick that one. Trust these "
    "pre-computed candidates by default; only override the rule if the "
    "phrase structure gives you a clear, specific reason to prefer a "
    "different one, and say why. If energy_recovery_candidates is "
    "empty, fall back to the first Chorus-labeled phrase after the "
    "Breakdown (heuristic_phrase_index.E, if not null -- otherwise "
    "after the Drop instead); if none of those exist either, decline "
    "(phrase_index -1)."
)

_CRITIC_SYSTEM = (
    "You review a complete 8-pad DJ cue proposal for a track (First Beat, "
    "Loop In, Vocal/Buildup, Drop, Breakdown, Special, Outro, Loop Out) for "
    "cross-cue coherence: correct ordering, sensible spacing, and whether "
    "the overall pacing matches the track's phrase structure. You may ONLY "
    "adjust confidence values and add explanatory notes -- you cannot move "
    "any cue's position. Only include a pad in `adjustments` if you're "
    "revising its confidence; omit pads you have nothing to add for."
)


def estimate_track_cost(
    track: Track,
    provider: ModelProvider,
    api_key: str,
    model: str,
    skip_critic: bool = False,
    memory_offset_bars: int = 16,
    loop_length_bars: int = 4,
) -> tuple[int, int, float | None]:
    """Real pre-flight estimate for one track: (input_tokens, output_tokens, usd_cost).

    Input tokens are genuine count_tokens() calls against this track's
    actual constructed payload and each real system prompt -- one call per
    specialist (plus the critic, unless skip_critic), so the number varies
    by track (phrase count, vocal regions, ...) instead of being a flat
    guess. Output tokens stay an approximation (see
    _ESTIMATED_OUTPUT_TOKENS_PER_CALL) since no provider can count
    hypothetical output before generation. This makes 3-4 real API calls,
    so it needs a valid key -- "cheap and safe" (free on Anthropic; a
    normal metered call on Gemini), not "fully offline."
    """
    import json

    from djcues.constants import KIND_TO_PAD

    heuristic = CueStrategy(memory_offset_bars, loop_length_bars).propose(track)
    payload = build_track_payload(track, heuristic)
    payload_json = json.dumps(payload)

    input_tokens = sum(
        provider.count_tokens(api_key, model, payload_json, system=system)
        for system in (_STRUCTURE_SYSTEM, _VOCAL_SYSTEM, _ENERGY_SYSTEM)
    )
    calls = _CALLS_PER_TRACK_NO_CRITIC

    if not skip_critic:
        heuristic_positions = {
            KIND_TO_PAD.get(c.kind): c.position_ms for c in heuristic.hot_cues
        }
        critic_payload = {
            **payload,
            "proposed_positions_ms": {pad: round(pos) for pad, pos in heuristic_positions.items()},
            "proposed_confidence": {pad: round(c, 3) for pad, c in heuristic.confidence.items()},
        }
        input_tokens += provider.count_tokens(
            api_key, model, json.dumps(critic_payload), system=_CRITIC_SYSTEM
        )
        calls = _CALLS_PER_TRACK_WITH_CRITIC

    output_tokens = _ESTIMATED_OUTPUT_TOKENS_PER_CALL * calls
    cost = price_estimate(model, input_tokens, output_tokens)
    return input_tokens, output_tokens, cost


def _call_specialist(
    provider: ModelProvider, api_key: str, model: str, system: str, payload: dict, schema: dict
) -> GenerationResult:
    import json

    return provider.generate_structured(
        api_key=api_key,
        model=model,
        system=system,
        user_content=json.dumps(payload),
        schema=schema,
    )


def _apply_placement(
    result: dict | None,
    key: str,
    pad: str,
    phrases: list,
    positions: dict[str, float],
    confidence: dict[str, float],
    notes: list[str],
    specialist_name: str,
) -> None:
    """Apply one specialist's placement for one pad, falling back to
    whatever the heuristic already put in positions/confidence (passed
    in pre-populated) if the specialist declined (-1) or errored."""
    if result is None:
        # The real failure reason is already in telemetry.errors, added
        # once by the caller when the specialist call itself failed --
        # don't duplicate it here for every pad that specialist covers
        # (structure alone covers 3), just note which pad kept the
        # heuristic value.
        notes.append(f"{pad} ({specialist_name}): kept heuristic (specialist call failed)")
        return

    placement = result.get(key)
    if not placement:
        return
    idx = placement.get("phrase_index", -1)
    if idx is None or idx < 0 or idx >= len(phrases):
        notes.append(f"{pad} ({specialist_name}): declined to place, kept heuristic")
        return

    positions[pad] = phrases[idx].position_ms
    confidence[pad] = max(0.0, min(1.0, float(placement.get("confidence", 0.5))))
    reasoning = placement.get("reasoning", "")
    notes.append(f"{pad} ({specialist_name}): {reasoning}".strip())


def propose_with_telemetry(
    track: Track,
    provider: ModelProvider,
    api_key: str,
    model: str,
    memory_offset_bars: int = 16,
    loop_length_bars: int = 4,
    skip_critic: bool = False,
) -> tuple[CueProposal, AgenticTelemetry]:
    """Full agentic proposal plus cost/error telemetry.

    Never raises for an individual specialist failure -- that pad falls
    back to the heuristic value and the failure is recorded in
    telemetry.errors. Authentication errors from the provider SDK are
    not caught here and propagate to the caller, since a bad/expired key
    should abort the whole run rather than silently degrading to the
    heuristic for every pad.
    """
    telemetry = AgenticTelemetry()

    heuristic = CueStrategy(memory_offset_bars, loop_length_bars).propose(track)
    payload = build_track_payload(track, heuristic)
    phrases = track.phrases

    # Seed positions/confidence from the heuristic for every pad it placed.
    # A, B, H stay exactly this (heuristic passthrough, no LLM call -- see
    # module docstring); C, D, E, F, G are the fallback if their specialist
    # fails or declines, and get overwritten below on success.
    from djcues.constants import KIND_TO_PAD

    heuristic_positions = {
        KIND_TO_PAD.get(c.kind): c.position_ms for c in heuristic.hot_cues
    }
    positions: dict[str, float] = dict(heuristic_positions)
    confidence: dict[str, float] = dict(heuristic.confidence)
    notes: list[str] = list(heuristic.notes)

    def run_structure():
        return _call_specialist(
            provider, api_key, model, _STRUCTURE_SYSTEM, payload, _STRUCTURE_SCHEMA
        )

    def run_vocal():
        return _call_specialist(provider, api_key, model, _VOCAL_SYSTEM, payload, _VOCAL_SCHEMA)

    def run_energy():
        return _call_specialist(provider, api_key, model, _ENERGY_SYSTEM, payload, _ENERGY_SCHEMA)

    specialists = {"structure": run_structure, "vocal": run_vocal, "energy": run_energy}
    results: dict[str, dict | None] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        future_to_name = {pool.submit(fn): name for name, fn in specialists.items()}
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                gen_result = future.result()
                results[name] = gen_result.content
                telemetry.calls_made += 1
                telemetry.input_tokens += gen_result.input_tokens
                telemetry.output_tokens += gen_result.output_tokens
            except Exception as e:  # noqa: BLE001 -- provider errors vary by SDK
                if _looks_like_auth_error(e):
                    raise
                results[name] = None
                telemetry.errors.append(f"{name} specialist failed: {e}")

    _apply_placement(
        results.get("structure"), "drop", "D", phrases, positions, confidence, notes,
        "structure",
    )
    _apply_placement(
        results.get("structure"), "breakdown", "E", phrases, positions, confidence, notes,
        "structure",
    )
    _apply_placement(
        results.get("structure"), "outro", "G", phrases, positions, confidence, notes,
        "structure",
    )
    _apply_placement(
        results.get("vocal"), "vocal_buildup", "C", phrases, positions, confidence, notes,
        "vocal",
    )
    _apply_placement(
        results.get("energy"), "special", "F", phrases, positions, confidence, notes,
        "energy",
    )

    if not skip_critic:
        try:
            import json

            critic_payload = {
                **payload,
                "proposed_positions_ms": {pad: round(pos) for pad, pos in positions.items()},
                "proposed_confidence": {pad: round(c, 3) for pad, c in confidence.items()},
            }
            critic_gen_result = provider.generate_structured(
                api_key=api_key,
                model=model,
                system=_CRITIC_SYSTEM,
                user_content=json.dumps(critic_payload),
                schema=_CRITIC_SCHEMA,
            )
            telemetry.calls_made += 1
            telemetry.input_tokens += critic_gen_result.input_tokens
            telemetry.output_tokens += critic_gen_result.output_tokens
            critic_result = critic_gen_result.content
            for adj in critic_result.get("adjustments", []):
                pad = adj.get("pad")
                if pad in confidence:
                    confidence[pad] = max(0.0, min(1.0, float(adj.get("confidence", confidence[pad]))))
                    note = adj.get("note", "")
                    notes.append(f"{pad} (critic): {note}".strip())
        except Exception as e:  # noqa: BLE001
            if _looks_like_auth_error(e):
                raise
            telemetry.errors.append(f"critic failed: {e}")

    hot_cues, memory_cues = build_cue_points(
        positions, confidence, track, memory_offset_bars, loop_length_bars
    )
    proposal = CueProposal(
        track=track, hot_cues=hot_cues, memory_cues=memory_cues, confidence=confidence, notes=notes
    )

    telemetry.estimated_cost = price_estimate(
        model, telemetry.input_tokens, telemetry.output_tokens
    )
    return proposal, telemetry


def propose(
    track: Track,
    provider: ModelProvider,
    api_key: str,
    model: str,
    memory_offset_bars: int = 16,
    loop_length_bars: int = 4,
    skip_critic: bool = False,
) -> CueProposal:
    """Drop-in equivalent to CueStrategy.propose(track) -- same return
    type, so callers that don't need cost telemetry (e.g. review/apply's
    existing per-track loops) can swap this in directly."""
    proposal, _telemetry = propose_with_telemetry(
        track, provider, api_key, model, memory_offset_bars, loop_length_bars, skip_critic
    )
    return proposal


_AUTH_ERROR_TYPE_NAMES = ("AuthenticationError", "PermissionDeniedError")
_AUTH_ERROR_HTTP_CODES = (401, 403)


def _looks_like_auth_error(exc: Exception) -> bool:
    """True for an authentication/permission failure that should abort
    the whole run rather than degrade one pad to its heuristic value.

    Checked two ways, neither importing any provider SDK (they're
    optional dependencies): the exception's type name -- works for
    Anthropic, whose SDK has distinctly-named AuthenticationError/
    PermissionDeniedError classes -- and an HTTP status code exposed via
    a `.code` or `.status_code` attribute, whichever the SDK sets. The
    second check exists because Gemini's SDK does NOT have a distinctly
    named class for this: confirmed against a real 403 response, its
    google.genai.errors module raises the exact same ClientError for
    every 4xx status (401, 403, 429, ...), so type-name alone silently
    misses even a flatly rejected API key on that provider.
    """
    if type(exc).__name__ in _AUTH_ERROR_TYPE_NAMES:
        return True
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code in _AUTH_ERROR_HTTP_CODES
