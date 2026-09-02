import pytest

from djcues.beat_verify import (
    check_grid_self_consistency,
    score_beat_alignment,
    verify_beat_grid,
    verify_beat_grid_against_audio,
)
from djcues.models import BeatGrid, RawBeatGridEntry, Track
from tests.conftest import requires_audio, requires_ml


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


# --- score_beat_alignment (tier 2's pure, testable comparison math) ------


@pytest.fixture
def grid() -> BeatGrid:
    return BeatGrid(first_beat_ms=77.0, bpm=128.0)


def test_perfectly_aligned_beats_are_consistent(grid: BeatGrid):
    detected = [grid.beat_to_ms(b) for b in range(1, 21)]
    result = score_beat_alignment(detected, grid)

    assert result.verdict == "consistent"
    assert result.matched_beats == 20
    assert result.mean_abs_drift_ms == pytest.approx(0.0, abs=1e-6)
    assert result.max_abs_drift_ms == pytest.approx(0.0, abs=1e-6)
    assert result.pct_within_tolerance == pytest.approx(100.0)
    assert result.tracker_name == "beat_this"


def test_small_consistent_offset_within_tolerance_is_consistent(grid: BeatGrid):
    detected = [grid.beat_to_ms(b) + 15.0 for b in range(1, 21)]  # within default 30ms tolerance
    result = score_beat_alignment(detected, grid)

    assert result.verdict == "consistent"
    assert result.mean_abs_drift_ms == pytest.approx(15.0, abs=1e-6)
    assert result.pct_within_tolerance == pytest.approx(100.0)


def test_large_drift_flags_drift_detected(grid: BeatGrid):
    detected = [grid.beat_to_ms(b) + 100.0 for b in range(1, 21)]  # beyond default 30ms tolerance
    result = score_beat_alignment(detected, grid)

    assert result.verdict == "drift_detected"
    assert result.pct_within_tolerance == pytest.approx(0.0)


def test_empty_detected_beats_is_no_beats_detected(grid: BeatGrid):
    result = score_beat_alignment([], grid)
    assert result.verdict == "no_beats_detected"
    assert result.matched_beats == 0


def test_mixed_alignment_computes_correct_percentage(grid: BeatGrid):
    # 8 beats aligned exactly, 2 beats way off
    detected = [grid.beat_to_ms(b) for b in range(1, 9)] + [
        grid.beat_to_ms(9) + 200.0,
        grid.beat_to_ms(10) + 200.0,
    ]
    result = score_beat_alignment(detected, grid)

    assert result.matched_beats == 10
    assert result.pct_within_tolerance == pytest.approx(80.0)
    assert result.verdict == "drift_detected"  # 80% is below the 90% consistency threshold


def test_custom_tolerance_changes_verdict(grid: BeatGrid):
    detected = [grid.beat_to_ms(b) + 50.0 for b in range(1, 21)]
    assert score_beat_alignment(detected, grid, tolerance_ms=100.0).verdict == "consistent"
    assert score_beat_alignment(detected, grid, tolerance_ms=10.0).verdict == "drift_detected"


# --- verify_beat_grid orchestration ---------------------------------------


def _track(beat_grid: BeatGrid) -> Track:
    return Track(
        id=42, title="Test Track", artist="Test", bpm=beat_grid.bpm, duration_ms=200_000.0,
        analysis_path="", cues=[], phrases=[], beat_grid=beat_grid,
    )


def test_verify_beat_grid_no_data_reports_no_grid_data(grid: BeatGrid):
    report = verify_beat_grid(_track(grid), raw_entries=None)
    assert report.status == "no_grid_data"
    assert report.audio is None


def test_verify_beat_grid_consistent_stays_at_tier_1(grid: BeatGrid):
    """The whole point of the tiering: a consistent grid must never
    escalate to audio (no audio_path set on this track, so if it tried,
    resolve_audio_path would return None and change the status)."""
    entries = _constant_tempo_entries(bpm=128.0, count=20)
    report = verify_beat_grid(_track(grid), raw_entries=entries)

    assert report.status == "ok"
    assert report.audio is None
    assert report.self_consistency.is_consistent is True


def test_verify_beat_grid_inconsistent_escalates_but_finds_no_audio(grid: BeatGrid):
    """An inconsistent self-check DOES try to escalate -- but this test
    track has no audio_path, so it lands on audio_unavailable rather
    than actually running a model."""
    bpm = 128.0
    base_gap = 60_000.0 / bpm
    entries = [
        RawBeatGridEntry(beat_in_bar=1, bpm=bpm, time_ms=0.0),
        RawBeatGridEntry(beat_in_bar=2, bpm=bpm, time_ms=base_gap + 100.0),
    ]
    report = verify_beat_grid(_track(grid), raw_entries=entries)

    assert report.self_consistency.is_consistent is False
    assert report.status == "audio_unavailable"
    assert report.audio is None


def test_verify_beat_grid_force_deep_escalates_even_when_consistent(grid: BeatGrid):
    entries = _constant_tempo_entries(bpm=128.0, count=20)
    report = verify_beat_grid(_track(grid), raw_entries=entries, force_deep=True)

    assert report.self_consistency.is_consistent is True
    assert report.status == "audio_unavailable"  # still no audio_path set on this track


@requires_audio
def test_verify_beat_grid_decode_failure_reports_decode_failed(grid: BeatGrid, tmp_path):
    bad_file = tmp_path / "not-really-audio.wav"
    bad_file.write_bytes(b"this is not a real wav file")
    track = Track(
        id=42, title="Bad File", artist="Test", bpm=grid.bpm, duration_ms=200_000.0,
        analysis_path="", cues=[], phrases=[], beat_grid=grid, audio_path=str(bad_file),
    )
    entries = _constant_tempo_entries(bpm=128.0, count=20)
    report = verify_beat_grid(track, raw_entries=entries, force_deep=True)

    assert report.status == "decode_failed"
    assert report.audio is None


# --- real model inference (live-only) -------------------------------------


@requires_ml
def test_verify_beat_grid_against_audio_runs_a_real_model(grid: BeatGrid):
    """Smoke test only, per the plan's own testing strategy -- confirms
    the real beat_this call completes and returns a sensible shape, not
    exact positions from a real model."""
    import numpy as np

    sr = 22050
    duration_s = 10
    t = np.arange(0, duration_s, 1 / sr)
    samples = np.zeros_like(t, dtype=np.float32)
    for beat_time in np.arange(0, duration_s, 0.5):
        idx = int(beat_time * sr)
        click_len = int(0.01 * sr)
        if idx + click_len < len(samples):
            samples[idx:idx + click_len] += (
                np.sin(2 * np.pi * 1000 * np.arange(click_len) / sr) * 0.8
            )

    result = verify_beat_grid_against_audio(_track(grid), samples, sr)

    assert result.tracker_name == "beat_this"
    assert result.matched_beats > 0
    assert result.verdict in ("consistent", "drift_detected", "no_beats_detected")
