"""Drop enhancement -- refines an already-placed Drop (D), Breakdown
(E), or Special (F) cue position using real audio, in two tiers.

Tier 1 (default, needs the `audio` extra): refine_drop_position() and
refine_breakdown_position() look for the dominant acoustic transition
strictly within a bounded window around the position the
heuristic/agentic engine already placed -- never an independent
full-track redetection. This constraint isn't arbitrary: strategy.py's
own git history (_spectral_similarity's docstring, commit 9a70f78)
shows a secondary signal overriding a phrase-anchored position outright
measurably hurt real accuracy (B: 88%->56%, H: 42%->27%) before being
reverted.

D and F are both "rise" cues (a transient -- kick re-entry, sub-bass
hit, a filtered buildup opening up); E is a "dip" cue (a breakdown is
defined by energy falling away, the opposite signal shape) -- both
share the same windowed-RMS scoring core (_score_transition()), just
looking for the opposite sign of the same "energy after minus energy
before" curve, since a breakdown's onset genuinely is the mirror image
of a drop's.

Tier 2 (--deep, needs the `ml` extra): separate_stems() runs real
Demucs source separation so the tier-1 analysis can run against an
isolated bass+drums signal instead of the full mix -- a cleaner
drop-energy signal, at the cost of ~7-12 minutes/track of CPU time
(confirmed by direct measurement, see the plan's Grounded Research).

enhance_proposal_drops() orchestrates both: resolves and loads the
track's real audio file, refines D, E, and F, and rebuilds cues via
strategy.build_cue_points() so memory-cue offset/loop math stays
consistent instead of being hand-rolled a third time.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from djcues.audio import AudioExtraUnavailableError, load_audio, resolve_audio_path
from djcues.constants import KIND_TO_PAD
from djcues.models import CueProposal, DropRefinement, Track
from djcues.strategy import build_cue_points

# None of these are tuned against real data yet -- a documented sketch
# per the plan, not measured thresholds. There's no automated ground
# truth for "where the drop really is"; expect to revisit once this
# has actually been judged by ear against real tracks.
_DEFAULT_SEARCH_WINDOW_MS = 4000.0
_RISE_SIDE_MS = 300.0  # width of the before/after windows used to compute each frame's energy rise
_MIN_REFINE_DISTANCE_MS = 500.0  # an alternative closer than this to the candidate isn't a genuinely different event
_MIN_REFINE_RATIO = 1.5  # the alternative must be at least this many times stronger than the candidate's own rise
_MIN_RISE_STRENGTH = 0.01  # floor below which "the strongest rise in the window" still isn't a real transient
_MIN_WINDOW_FRAMES = 4  # fewer than this and there's not enough resolution to say anything

_RISE_PADS = ("D", "F")
_DIP_PADS = ("E",)
# Iteration order here becomes the order refinement notes are appended
# in -- D, E, F matches the cue system's own A-H ordering.
_REFINABLE_DIRECTIONS = {"D": "rise", "E": "dip", "F": "rise"}

# Cached across calls so a whole playlist run with --deep doesn't
# reload Demucs weights per track -- mirrors beat_verify.py's
# _get_beat_tracker() pattern.
_stem_separator = None


def _score_transition(
    samples: np.ndarray,
    sr: int,
    candidate_ms: float,
    pad: str,
    direction: str,
    search_window_ms: float,
    source: str,
) -> DropRefinement:
    """Shared scoring core for refine_drop_position (direction="rise",
    D/F) and refine_breakdown_position (direction="dip", E) -- both
    look for the dominant energy transition strictly within
    [candidate_ms - search_window_ms/2, candidate_ms + search_window_ms/2],
    clamped to the track's actual bounds, and only act on it if it's
    both far enough away to be a genuinely different event
    (> _MIN_REFINE_DISTANCE_MS) and clearly dominant over whatever
    transition already exists at the candidate (>= _MIN_REFINE_RATIO
    times stronger) -- otherwise the existing position is confirmed,
    or left inconclusive if no real transition was found anywhere in
    the window at all.

    Computes an RMS energy envelope over the window, then a signed
    "rise" value at each frame (mean energy in a short window after
    minus mean energy in a short window before) -- a rise cue looks
    for the frame where this is most positive (a transient hitting), a
    dip cue looks for the frame where this is most negative (energy
    falling away, a breakdown beginning), by simply negating the same
    curve before searching it.
    """
    import librosa

    total_ms = len(samples) / sr * 1000.0
    half_window = search_window_ms / 2.0
    start_ms = max(0.0, candidate_ms - half_window)
    end_ms = min(total_ms, candidate_ms + half_window)
    word = "rise" if direction == "rise" else "dip"

    def _inconclusive(note: str, strength: float = 0.0) -> DropRefinement:
        return DropRefinement(
            pad=pad, outcome="inconclusive", original_ms=candidate_ms,
            refined_ms=candidate_ms, offset_ms=0.0, strength=strength,
            source=source, note=note,
        )

    if end_ms - start_ms < _RISE_SIDE_MS * 2:
        return _inconclusive("search window too close to a track boundary to analyze")

    # Extend the analysis buffer by a real-audio margin on each side so
    # librosa's edge (reflection) padding lands outside the window we
    # actually score -- without this, a frame right at start_ms/end_ms
    # has no genuine "before"/"after" audio to compare and librosa
    # fills that gap with a reflected copy of the segment, which can
    # look like a spurious transition on sustained, non-silent audio
    # (confirmed live: a perfectly constant tone was misread as having
    # a dominant "dip" right at the window's edge before this fix).
    margin_ms = _RISE_SIDE_MS
    buf_start_ms = max(0.0, start_ms - margin_ms)
    buf_end_ms = min(total_ms, end_ms + margin_ms)

    buf_start_sample = int(buf_start_ms / 1000.0 * sr)
    buf_end_sample = int(buf_end_ms / 1000.0 * sr)
    segment = samples[buf_start_sample:buf_end_sample]

    hop_length = max(1, int(sr * 0.01))  # ~10ms hop
    frame_length = max(hop_length * 2, int(sr * 0.05))  # ~50ms analysis frame
    rms = librosa.feature.rms(y=segment, frame_length=frame_length, hop_length=hop_length)[0]

    if len(rms) < _MIN_WINDOW_FRAMES:
        return _inconclusive("not enough audio in the search window to analyze")

    frame_times_ms = buf_start_ms + (np.arange(len(rms)) * hop_length / sr * 1000.0)
    side_frames = max(1, int(_RISE_SIDE_MS / 1000.0 * sr / hop_length))

    signed_rise = np.zeros(len(rms), dtype=np.float64)
    for i in range(len(rms)):
        before = rms[max(0, i - side_frames):i]
        after = rms[i:min(len(rms), i + side_frames)]
        if len(before) == 0 or len(after) == 0:
            continue
        signed_rise[i] = float(after.mean()) - float(before.mean())

    transition = signed_rise if direction == "rise" else -signed_rise

    # Only frames inside the true (non-margin) window are eligible to
    # win -- the margin exists purely to give edge frames real
    # before/after context, never to be selected itself.
    in_window = np.where((frame_times_ms >= start_ms) & (frame_times_ms <= end_ms))[0]
    if len(in_window) == 0:
        return _inconclusive("not enough audio in the search window to analyze")

    best_idx = int(in_window[np.argmax(transition[in_window])])
    best_ms = float(frame_times_ms[best_idx])
    best_strength = float(transition[best_idx])

    if best_strength < _MIN_RISE_STRENGTH:
        return _inconclusive(
            f"no clear energy {word} found anywhere in the search window", best_strength
        )

    candidate_idx = int(in_window[np.argmin(np.abs(frame_times_ms[in_window] - candidate_ms))])
    candidate_strength = float(transition[candidate_idx])
    ratio = best_strength / max(candidate_strength, _MIN_RISE_STRENGTH)
    distance_ms = abs(best_ms - candidate_ms)

    if distance_ms > _MIN_REFINE_DISTANCE_MS and ratio >= _MIN_REFINE_RATIO:
        return DropRefinement(
            pad=pad, outcome="refined", original_ms=candidate_ms, refined_ms=best_ms,
            offset_ms=best_ms - candidate_ms, strength=ratio, source=source,
            note=(
                f"stronger energy {word} found {best_ms - candidate_ms:+.0f}ms away "
                f"({ratio:.1f}x the {word} at the original position)"
            ),
        )

    return DropRefinement(
        pad=pad, outcome="confirmed", original_ms=candidate_ms, refined_ms=candidate_ms,
        offset_ms=0.0, strength=ratio, source=source,
        note="audio analysis agrees with the existing position",
    )


def refine_drop_position(
    samples: np.ndarray,
    sr: int,
    candidate_ms: float,
    pad: str,
    search_window_ms: float = _DEFAULT_SEARCH_WINDOW_MS,
    source: str = "full_mix",
) -> DropRefinement:
    """Look for the dominant energy rise (a transient -- kick re-entry,
    sub-bass hit, a filtered buildup opening up) near candidate_ms. See
    _score_transition() for the shared scoring mechanics."""
    return _score_transition(samples, sr, candidate_ms, pad, "rise", search_window_ms, source)


def refine_breakdown_position(
    samples: np.ndarray,
    sr: int,
    candidate_ms: float,
    pad: str = "E",
    search_window_ms: float = _DEFAULT_SEARCH_WINDOW_MS,
    source: str = "full_mix",
) -> DropRefinement:
    """Mirror of refine_drop_position for the Breakdown (E) cue -- looks
    for the dominant energy DIP rather than a rise, since a breakdown
    is defined by energy falling away, the opposite signal shape from
    D/F. Same conservative refine/confirm/inconclusive design and
    thresholds; see _score_transition() for the shared mechanics."""
    return _score_transition(samples, sr, candidate_ms, pad, "dip", search_window_ms, source)


def _get_stem_separator() -> Any:
    """Lazily construct and cache the real Demucs model. Live-only,
    covered only by a requires_ml smoke test -- loads real model
    weights (auto-downloaded from HuggingFace Hub on first use)."""
    global _stem_separator
    if _stem_separator is None:
        from demucs.pretrained import get_model

        model = get_model("htdemucs")
        model.eval()
        _stem_separator = model
    return _stem_separator


def separate_stems(samples: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Real Demucs source separation -- isolates drums/bass/other/vocals
    from a mono mix. Live-only, covered only by a requires_ml smoke
    test; the underlying get_model()/apply_model() call was confirmed
    working end-to-end against real synthetic audio before this was
    written (see the plan's Grounded Research -- correct isolation of
    a low-frequency test tone into the bass stem, near-silence
    elsewhere).

    Demucs' htdemucs checkpoint needs stereo input at its own 44.1kHz
    samplerate -- the mono samples this codebase otherwise works with
    are duplicated to fake stereo, resampled in and back out via
    librosa so the returned stems stay time-aligned with the caller's
    original `samples`/`sr`, and averaged back down to mono per stem
    (matching every other function in this module's mono-samples
    contract). ~1.6-1.7s of CPU wall-clock per second of audio on the
    machine this was measured on -- not fast, only called from
    enhance_proposal_drops when --deep is explicitly passed.
    """
    import librosa
    import torch
    from demucs.apply import apply_model

    model = _get_stem_separator()
    model_sr = model.samplerate

    mix_samples = (
        librosa.resample(samples, orig_sr=sr, target_sr=model_sr)
        if sr != model_sr
        else samples
    )
    stereo = np.stack([mix_samples, mix_samples], axis=0).astype(np.float32)
    mix = torch.from_numpy(stereo).unsqueeze(0)

    with torch.no_grad():
        out = apply_model(model, mix, device="cpu", progress=False, shifts=0)

    stems: dict[str, np.ndarray] = {}
    for i, name in enumerate(model.sources):
        stem_mono = out[0, i].mean(dim=0).numpy()
        stems[name] = (
            librosa.resample(stem_mono, orig_sr=model_sr, target_sr=sr)
            if sr != model_sr
            else stem_mono
        )
    return stems


def enhance_proposal_drops(
    proposal: CueProposal,
    track: Track,
    memory_offset_bars: int,
    loop_length_bars: int,
    deep: bool = False,
) -> tuple[CueProposal, list[DropRefinement]]:
    """Refine the D (Drop), E (Breakdown), and F (Special) cues of an
    already-built proposal against the track's real audio -- D/F look
    for a dominant energy rise, E for a dominant energy dip, via
    refine_drop_position()/refine_breakdown_position() respectively.

    Degrades to a no-op (the original proposal, empty refinement list)
    whenever real audio isn't actually available for this specific
    track -- missing/moved file, unsupported format, whatever else --
    rather than failing the whole run. The CLI's upfront extra-install
    check is what stops a run early when the *capability* itself isn't
    installed; this function only ever handles per-track failures.

    Rebuilds hot_cues/memory_cues via strategy.build_cue_points() so
    memory-cue offset/loop math is computed identically to every other
    engine in this codebase, instead of being hand-rolled a third time.
    """
    audio_path = resolve_audio_path(track)
    if audio_path is None:
        return proposal, []

    try:
        loaded = load_audio(audio_path)
    except AudioExtraUnavailableError:
        return proposal, []
    except Exception:
        return proposal, []

    analysis_samples = loaded.samples
    source = "full_mix"
    if deep:
        try:
            stems = separate_stems(loaded.samples, loaded.sr)
            analysis_samples = stems["bass"] + stems["drums"]
            source = "stems"
        except Exception:
            analysis_samples = loaded.samples
            source = "full_mix"

    positions = {KIND_TO_PAD.get(c.kind): c.position_ms for c in proposal.hot_cues}
    confidence = dict(proposal.confidence)
    notes = list(proposal.notes)
    refinements: list[DropRefinement] = []

    for pad, direction in _REFINABLE_DIRECTIONS.items():
        if pad not in positions:
            continue
        refine_fn = refine_drop_position if direction == "rise" else refine_breakdown_position
        refinement = refine_fn(
            analysis_samples, loaded.sr, positions[pad], pad, source=source
        )
        refinements.append(refinement)
        if refinement.outcome == "refined":
            positions[pad] = refinement.refined_ms
        notes.append(
            f"{pad} (drop-enhance, {source}): {refinement.outcome} -- {refinement.note}"
        )

    hot_cues, memory_cues = build_cue_points(
        positions, confidence, track, memory_offset_bars, loop_length_bars
    )
    refined_proposal = CueProposal(
        track=track, hot_cues=hot_cues, memory_cues=memory_cues,
        confidence=confidence, notes=notes,
    )
    return refined_proposal, refinements
