"""Beat-grid verification -- confirms Rekordbox's stored beat grid is
still trustworthy, in two tiers.

Tier 1 (this file, fully implemented): check_grid_self_consistency()
is pure arithmetic over Rekordbox's own full per-beat PQTZ array
(db.py:extract_raw_beat_grid()) -- no audio file, no new dependency.
djcues's BeatGrid model collapses that array down to one constant
(first_beat_ms, bpm); this checks whether that simplification was
actually safe for a given track, using data Rekordbox already computed
and djcues already had access to but discarded.

Tier 2 (real audio, needs the optional `ml` extra): live beat-tracker
inference via `beat_this` (github.com/CPJKU/beat_this), compared against
the stored grid via score_beat_alignment() -- pure comparison math,
fully testable with a synthetic detected-beat-times list, no model
involved. verify_beat_grid() orchestrates both tiers: always runs the
free check first, only escalates to real audio (needing the `ml`
extra) when the free check looks suspicious or force_deep is set --
so a track whose grid is already self-consistent costs nothing beyond
the free check.
"""

from __future__ import annotations

import statistics
from typing import Any

from djcues.audio import AudioExtraUnavailableError, load_audio, resolve_audio_path
from djcues.models import (
    AudioBeatVerification,
    BeatGrid,
    BeatGridReport,
    RawBeatGridEntry,
    SelfConsistencyResult,
    Track,
)

# Neither tolerance has been tuned against real data yet -- these are
# reasonable starting points, not measured thresholds. Expect to revisit
# once this has run against a real library (see the plan's Verification
# section).
_DEFAULT_TEMPO_EPSILON_BPM = 0.5
_DEFAULT_GAP_ERROR_TOLERANCE_MS = 5.0
_DEFAULT_DRIFT_TOLERANCE_MS = 30.0
_DEFAULT_AUDIO_TOLERANCE_MS = 30.0
_TRACKER_NAME = "beat_this"

# How close a detected/grid tempo ratio needs to be to exactly 2.0 or
# 0.5 to count as an octave error rather than coincidence -- e.g. 0.15
# accepts a ratio of 1.7-2.3 as "double". Untuned against real data,
# same caveat as every other threshold in this file.
_OCTAVE_RATIO_TOLERANCE = 0.15
# Fewer detected beats than this and the median inter-beat interval is
# too noisy to trust for octave-ratio comparison.
_MIN_BEATS_FOR_OCTAVE_CHECK = 8

# Cached across calls so processing a whole playlist doesn't reload
# model weights per track -- mirrors db.py's get_db() pattern.
_beat_tracker = None


def check_grid_self_consistency(
    entries: list[RawBeatGridEntry],
    tempo_epsilon_bpm: float = _DEFAULT_TEMPO_EPSILON_BPM,
    gap_error_tolerance_ms: float = _DEFAULT_GAP_ERROR_TOLERANCE_MS,
    drift_tolerance_ms: float = _DEFAULT_DRIFT_TOLERANCE_MS,
) -> SelfConsistencyResult:
    """Check whether Rekordbox's own full beat grid is internally
    consistent with a constant-tempo model.

    Two independent, deliberately separate signals, since they catch
    different failure modes:
    - max_pairwise_gap_error_ms: the single worst |actual gap - expected
      gap| between consecutive entries, using each entry's own recorded
      bpm. Catches local jitter even if it cancels out over the track.
    - cumulative_drift_at_end_ms: the *signed* sum of (actual - expected)
      gap across every consecutive pair. Catches gradual, one-directional
      drift that a per-pair check alone would miss if each individual
      step's error is small -- this is the failure mode that actually
      matters for "is beat 500 still on the beat."

    Both use only consecutive-pair math -- deliberately does not assume
    entries[i] corresponds to any particular absolute beat number the
    rest of djcues uses (e.g. BeatGrid.beat_to_ms's 1-indexed beats).
    That cross-tag mapping is plausible but wasn't independently
    confirmed against a real ANLZ file, so this check is self-contained
    and doesn't depend on it.

    tempo_varies just reports whether the entries' own recorded BPMs
    differ by more than tempo_epsilon_bpm -- informational, not itself a
    failure. A track with real tempo changes that Rekordbox tracked
    correctly (each gap matching its own entry's bpm) is still fully
    self-consistent.
    """
    if len(entries) < 2:
        return SelfConsistencyResult(
            is_consistent=True,
            tempo_varies=False,
            max_pairwise_gap_error_ms=0.0,
            cumulative_drift_at_end_ms=0.0,
            entry_count=len(entries),
            notes=["fewer than 2 beat-grid entries -- nothing to check"],
        )

    bpms = [e.bpm for e in entries]
    tempo_varies = (max(bpms) - min(bpms)) > tempo_epsilon_bpm

    max_gap_error = 0.0
    cumulative_drift = 0.0
    for prev, curr in zip(entries, entries[1:]):
        actual_gap = curr.time_ms - prev.time_ms
        expected_gap = 60_000.0 / prev.bpm if prev.bpm > 0 else actual_gap
        gap_error = actual_gap - expected_gap
        max_gap_error = max(max_gap_error, abs(gap_error))
        cumulative_drift += gap_error

    notes: list[str] = []
    if tempo_varies:
        notes.append(
            f"Rekordbox's own grid records tempo variation "
            f"({min(bpms):.2f}-{max(bpms):.2f} BPM) that djcues's "
            f"constant-BPM model currently ignores."
        )

    is_consistent = (
        max_gap_error <= gap_error_tolerance_ms
        and abs(cumulative_drift) <= drift_tolerance_ms
    )
    if not is_consistent:
        notes.append(
            f"grid inconsistency: max pairwise gap error {max_gap_error:.1f}ms "
            f"(tolerance {gap_error_tolerance_ms}ms), cumulative drift "
            f"{cumulative_drift:+.1f}ms (tolerance ±{drift_tolerance_ms}ms)"
        )

    return SelfConsistencyResult(
        is_consistent=is_consistent,
        tempo_varies=tempo_varies,
        max_pairwise_gap_error_ms=max_gap_error,
        cumulative_drift_at_end_ms=cumulative_drift,
        entry_count=len(entries),
        notes=notes,
    )


def _get_beat_tracker() -> Any:
    """Lazily construct and cache the real beat_this model. Live-only,
    not unit-tested -- loads real model weights on first call."""
    global _beat_tracker
    if _beat_tracker is None:
        from beat_this.inference import Audio2Beats

        _beat_tracker = Audio2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    return _beat_tracker


def _run_beat_tracker(samples: Any, sr: int) -> list[float]:
    """Real beat_this inference against real audio samples. Live-only,
    not unit-tested -- calls a real model (verified working this
    session via direct testing, not assumed). Returns detected beat
    times in milliseconds (beat_this's own native unit is seconds)."""
    tracker = _get_beat_tracker()
    beats, _downbeats = tracker(samples, sr)
    return [float(b) * 1000.0 for b in beats]


def _detect_octave_error(detected_beat_times_ms: list[float], ms_per_beat: float) -> str | None:
    """Check whether the detected beats' own median spacing is roughly
    double or half the grid's expected ms_per_beat -- the signature of
    a beat tracker locking onto the wrong tempo octave, a well-known
    MIR failure mode, distinct from genuine drift (which doesn't
    change the *spacing* between detected beats, just their alignment
    to the grid). Uses the median (not mean) interval so a handful of
    missed detections in a quiet section don't skew the estimate.

    Returns None (not enough beats, or no clean 2x/0.5x ratio), "half"
    (detected beats roughly twice as far apart as expected -- the
    tracker likely only caught every other real beat), or "double"
    (detected beats roughly half as far apart -- likely inserting a
    phantom beat between each real pair).
    """
    if len(detected_beat_times_ms) < _MIN_BEATS_FOR_OCTAVE_CHECK or ms_per_beat <= 0:
        return None

    sorted_times = sorted(detected_beat_times_ms)
    intervals = [b - a for a, b in zip(sorted_times, sorted_times[1:])]
    if not intervals:
        return None

    ratio = statistics.median(intervals) / ms_per_beat

    if abs(ratio - 2.0) <= 2.0 * _OCTAVE_RATIO_TOLERANCE:
        return "half"
    if abs(ratio - 0.5) <= 0.5 * _OCTAVE_RATIO_TOLERANCE:
        return "double"
    return None


def score_beat_alignment(
    detected_beat_times_ms: list[float],
    beat_grid: BeatGrid,
    tolerance_ms: float = _DEFAULT_AUDIO_TOLERANCE_MS,
) -> AudioBeatVerification:
    """Compare real, audio-detected beat times against the stored
    BeatGrid. Pure comparison math -- fully testable with a synthetic
    detected_beat_times_ms list, no model involved.

    Uses BeatGrid's own ms_to_beat()/beat_to_ms() to find each detected
    beat's nearest grid beat rather than reinventing that lookup, so
    this stays consistent with how the rest of djcues already reasons
    about beat positions.

    Also checks for an octave error (see _detect_octave_error()) --
    when detected, a "double" error reports verdict="octave_error"
    instead of the misleadingly-large "drift_detected" that raw
    nearest-beat distance would otherwise produce.
    """
    if not detected_beat_times_ms:
        return AudioBeatVerification(
            matched_beats=0,
            mean_abs_drift_ms=0.0,
            max_abs_drift_ms=0.0,
            pct_within_tolerance=0.0,
            tracker_name=_TRACKER_NAME,
            verdict="no_beats_detected",
        )

    abs_drifts = []
    for t in detected_beat_times_ms:
        nearest_beat_num = beat_grid.ms_to_beat(t)
        expected_ms = beat_grid.beat_to_ms(nearest_beat_num)
        abs_drifts.append(abs(t - expected_ms))

    mean_abs_drift = sum(abs_drifts) / len(abs_drifts)
    max_abs_drift = max(abs_drifts)
    within_tolerance = sum(1 for d in abs_drifts if d <= tolerance_ms)
    pct_within = within_tolerance / len(abs_drifts) * 100.0

    octave_error = _detect_octave_error(detected_beat_times_ms, beat_grid.ms_per_beat)
    if octave_error == "double":
        # A "double" error is what would otherwise misleadingly report
        # as a huge, scattered-looking drift -- the tracker's phantom
        # extra beats land almost exactly halfway between real grid
        # beats, which reads as "half the beats are wrong" rather than
        # what it actually is. "half" doesn't have this problem (every
        # detected beat still lines up fine), so it doesn't override
        # a verdict that's already correctly "consistent" below.
        verdict = "octave_error"
    elif pct_within >= 90.0:
        verdict = "consistent"
    else:
        verdict = "drift_detected"

    return AudioBeatVerification(
        matched_beats=len(detected_beat_times_ms),
        mean_abs_drift_ms=mean_abs_drift,
        max_abs_drift_ms=max_abs_drift,
        pct_within_tolerance=pct_within,
        tracker_name=_TRACKER_NAME,
        verdict=verdict,
        octave_error=octave_error,
    )


def verify_beat_grid_against_audio(
    track: Track,
    samples: Any,
    sr: int,
    tolerance_ms: float = _DEFAULT_AUDIO_TOLERANCE_MS,
) -> AudioBeatVerification:
    """Real beat_this inference against real audio, compared to the
    track's stored BeatGrid. Live-only -- not unit-tested, calls a real
    model; see score_beat_alignment() for the testable comparison math
    this wraps.
    """
    detected_ms = _run_beat_tracker(samples, sr)
    return score_beat_alignment(detected_ms, track.beat_grid, tolerance_ms)


def verify_beat_grid(
    track: Track,
    raw_entries: list[RawBeatGridEntry] | None,
    force_deep: bool = False,
    tempo_epsilon_bpm: float = _DEFAULT_TEMPO_EPSILON_BPM,
    gap_error_tolerance_ms: float = _DEFAULT_GAP_ERROR_TOLERANCE_MS,
    drift_tolerance_ms: float = _DEFAULT_DRIFT_TOLERANCE_MS,
    audio_tolerance_ms: float = _DEFAULT_AUDIO_TOLERANCE_MS,
) -> BeatGridReport:
    """Top-level orchestration: always run the free self-consistency
    check first; only escalate to real audio analysis (needing the
    `ml` extra) when the free check looks suspicious or force_deep is
    set. A track whose grid is already self-consistent costs nothing
    beyond the free check -- no audio file is even resolved for it.
    """
    if raw_entries is None:
        return BeatGridReport(
            track_id=track.id,
            title=track.title,
            self_consistency=SelfConsistencyResult(
                is_consistent=True,
                tempo_varies=False,
                max_pairwise_gap_error_ms=0.0,
                cumulative_drift_at_end_ms=0.0,
                entry_count=0,
                notes=["no beat-grid data available"],
            ),
            audio=None,
            status="no_grid_data",
        )

    self_consistency = check_grid_self_consistency(
        raw_entries, tempo_epsilon_bpm, gap_error_tolerance_ms, drift_tolerance_ms
    )

    if not force_deep and self_consistency.is_consistent:
        return BeatGridReport(
            track_id=track.id,
            title=track.title,
            self_consistency=self_consistency,
            audio=None,
            status="ok",
        )

    audio_path = resolve_audio_path(track)
    if audio_path is None:
        return BeatGridReport(
            track_id=track.id,
            title=track.title,
            self_consistency=self_consistency,
            audio=None,
            status="audio_unavailable",
        )

    try:
        loaded = load_audio(audio_path)
    except AudioExtraUnavailableError:
        return BeatGridReport(
            track_id=track.id,
            title=track.title,
            self_consistency=self_consistency,
            audio=None,
            status="audio_extra_missing",
        )
    except Exception:
        return BeatGridReport(
            track_id=track.id,
            title=track.title,
            self_consistency=self_consistency,
            audio=None,
            status="decode_failed",
        )

    audio_result = verify_beat_grid_against_audio(
        track, loaded.samples, loaded.sr, audio_tolerance_ms
    )
    status = "ok" if audio_result.verdict == "consistent" else "flagged"
    return BeatGridReport(
        track_id=track.id,
        title=track.title,
        self_consistency=self_consistency,
        audio=audio_result,
        status=status,
    )
