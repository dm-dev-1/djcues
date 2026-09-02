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
the stored grid. Not yet implemented in this file.
"""

from __future__ import annotations

from djcues.models import RawBeatGridEntry, SelfConsistencyResult

# Neither tolerance has been tuned against real data yet -- these are
# reasonable starting points, not measured thresholds. Expect to revisit
# once this has run against a real library (see the plan's Verification
# section).
_DEFAULT_TEMPO_EPSILON_BPM = 0.5
_DEFAULT_GAP_ERROR_TOLERANCE_MS = 5.0
_DEFAULT_DRIFT_TOLERANCE_MS = 30.0


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
