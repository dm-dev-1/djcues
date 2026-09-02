import pytest

from djcues.beat_verify import check_grid_self_consistency
from djcues.models import RawBeatGridEntry


def _constant_tempo_entries(bpm: float, count: int) -> list[RawBeatGridEntry]:
    gap = 60_000.0 / bpm
    return [
        RawBeatGridEntry(beat_in_bar=(i % 4) + 1, bpm=bpm, time_ms=i * gap)
        for i in range(count)
    ]


def test_constant_tempo_is_fully_consistent():
    entries = _constant_tempo_entries(bpm=128.0, count=20)
    result = check_grid_self_consistency(entries)

    assert result.is_consistent is True
    assert result.tempo_varies is False
    assert result.max_pairwise_gap_error_ms == pytest.approx(0.0, abs=1e-6)
    assert result.cumulative_drift_at_end_ms == pytest.approx(0.0, abs=1e-6)
    assert result.entry_count == 20
    assert result.notes == []


def test_real_tempo_change_flags_tempo_varies_but_stays_consistent():
    """A track with a genuine, correctly-tracked tempo change: each gap
    matches its own entry's recorded bpm at every step, including the
    transition -- Rekordbox got it right, so this must NOT be flagged as
    an inconsistency, only as informational tempo variation."""
    bpm_schedule = [128.0] * 10 + [140.0] * 10
    entries = []
    t = 0.0
    for i, bpm in enumerate(bpm_schedule):
        entries.append(RawBeatGridEntry(beat_in_bar=(i % 4) + 1, bpm=bpm, time_ms=t))
        t += 60_000.0 / bpm

    result = check_grid_self_consistency(entries)

    assert result.tempo_varies is True
    assert result.is_consistent is True
    assert result.max_pairwise_gap_error_ms == pytest.approx(0.0, abs=1e-6)
    assert result.cumulative_drift_at_end_ms == pytest.approx(0.0, abs=1e-6)


def test_jitter_that_cancels_flags_gap_error_not_cumulative_drift():
    """Pure local jitter, alternating +/-, summing to zero over the
    track -- this is exactly the distinction the two metrics exist for:
    the worst single gap is bad, but nothing drifts overall."""
    bpm = 128.0
    base_gap = 60_000.0 / bpm
    jitters = [8.0, -8.0, 8.0, -8.0]  # 4 gaps, alternating, sums to 0 exactly
    times = [0.0]
    for j in jitters:
        times.append(times[-1] + base_gap + j)
    entries = [
        RawBeatGridEntry(beat_in_bar=(i % 4) + 1, bpm=bpm, time_ms=t)
        for i, t in enumerate(times)
    ]

    result = check_grid_self_consistency(entries)

    assert result.max_pairwise_gap_error_ms == pytest.approx(8.0, abs=1e-6)
    assert result.cumulative_drift_at_end_ms == pytest.approx(0.0, abs=1e-6)
    assert result.is_consistent is False  # 8ms exceeds the default 5ms gap tolerance


def test_one_directional_drift_flags_cumulative_drift_even_with_small_gap_error():
    """Each individual gap is only 2ms off -- comfortably inside the
    default 5ms gap tolerance on its own -- but the error is the same
    sign every time, so it accumulates. This is the failure mode
    max_pairwise_gap_error_ms alone would miss."""
    bpm = 128.0
    base_gap = 60_000.0 / bpm
    per_gap_error = 2.0
    n = 20
    times = [0.0]
    for _ in range(n - 1):
        times.append(times[-1] + base_gap + per_gap_error)
    entries = [
        RawBeatGridEntry(beat_in_bar=(i % 4) + 1, bpm=bpm, time_ms=t)
        for i, t in enumerate(times)
    ]

    result = check_grid_self_consistency(entries)

    assert result.max_pairwise_gap_error_ms == pytest.approx(2.0, abs=1e-6)
    assert result.cumulative_drift_at_end_ms == pytest.approx(2.0 * (n - 1), abs=1e-6)
    assert result.is_consistent is False  # 38ms cumulative exceeds the 30ms drift tolerance


def test_empty_entries_is_consistent_with_nothing_to_check():
    result = check_grid_self_consistency([])
    assert result.is_consistent is True
    assert result.entry_count == 0
    assert "nothing to check" in result.notes[0]


def test_single_entry_is_consistent_with_nothing_to_check():
    result = check_grid_self_consistency([RawBeatGridEntry(beat_in_bar=1, bpm=128.0, time_ms=0.0)])
    assert result.is_consistent is True
    assert result.entry_count == 1


def test_two_entries_exact_spacing_is_consistent():
    bpm = 128.0
    gap = 60_000.0 / bpm
    entries = [
        RawBeatGridEntry(beat_in_bar=1, bpm=bpm, time_ms=0.0),
        RawBeatGridEntry(beat_in_bar=2, bpm=bpm, time_ms=gap),
    ]
    result = check_grid_self_consistency(entries)
    assert result.is_consistent is True
    assert result.max_pairwise_gap_error_ms == pytest.approx(0.0, abs=1e-6)


def test_custom_tolerances_are_respected():
    """A gap error that passes the default tolerance should fail a
    stricter one, and vice versa -- confirms the parameters aren't
    silently ignored."""
    bpm = 128.0
    base_gap = 60_000.0 / bpm
    entries = [
        RawBeatGridEntry(beat_in_bar=1, bpm=bpm, time_ms=0.0),
        RawBeatGridEntry(beat_in_bar=2, bpm=bpm, time_ms=base_gap + 3.0),
    ]
    assert check_grid_self_consistency(entries, gap_error_tolerance_ms=5.0).is_consistent is True
    assert check_grid_self_consistency(entries, gap_error_tolerance_ms=1.0).is_consistent is False
