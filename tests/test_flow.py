"""Tests for djcues.flow -- energy-flow set ordering."""

from __future__ import annotations

import pytest

from djcues.flow import (
    DEFAULT_COOLDOWN_FRACTION,
    compute_track_energy,
    suggest_energy_flow,
)
from djcues.models import BeatGrid, Phrase, Track, WaveformPoint
from tests.conftest import requires_rekordbox


def _uniform_track(id_: int, title: str, energy: float, duration_ms: float = 200_000.0) -> Track:
    """A track with a single phrase spanning its whole duration, at a
    uniform waveform height -- compute_phrase_energy() then trivially
    returns exactly [(phrase, energy)], so mean_energy == peak_energy
    == energy with no indexing-precision concerns."""
    phrase = Phrase(beat_start=1, beat_end=2, kind=1, label="Intro", position_ms=0.0, duration_ms=duration_ms)
    waveform = [WaveformPoint(height=energy, red=4, green=4, blue=4) for _ in range(10)]
    return Track(
        id=id_, title=title, artist="Test", bpm=128.0, duration_ms=duration_ms,
        analysis_path="", cues=[], phrases=[phrase], beat_grid=BeatGrid(first_beat_ms=0.0, bpm=128.0),
        waveform=waveform,
    )


def _no_phrase_track(id_: int, title: str) -> Track:
    return Track(
        id=id_, title=title, artist="Test", bpm=128.0, duration_ms=200_000.0,
        analysis_path="", cues=[], phrases=[], beat_grid=BeatGrid(first_beat_ms=0.0, bpm=128.0),
        waveform=[WaveformPoint(height=0.5, red=4, green=4, blue=4)],
    )


def _no_waveform_track(id_: int, title: str) -> Track:
    phrase = Phrase(beat_start=1, beat_end=2, kind=1, label="Intro", position_ms=0.0, duration_ms=200_000.0)
    return Track(
        id=id_, title=title, artist="Test", bpm=128.0, duration_ms=200_000.0,
        analysis_path="", cues=[], phrases=[phrase], beat_grid=BeatGrid(first_beat_ms=0.0, bpm=128.0),
        waveform=None,
    )


# ---------------------------------------------------------------------------
# compute_track_energy
# ---------------------------------------------------------------------------


def test_mean_energy_is_duration_weighted_not_naive():
    # Phrase 1: 10% of the track (10s), energy 0.2. Phrase 2: 90% (90s),
    # energy 0.8. A naive unweighted mean would give 0.5; the correct
    # duration-weighted mean is 0.74 -- distinct enough to prove the
    # weighting is real, not accidentally equivalent to a naive average.
    phrases = [
        Phrase(beat_start=1, beat_end=2, kind=1, label="Intro", position_ms=0.0, duration_ms=10_000.0),
        Phrase(beat_start=2, beat_end=3, kind=5, label="Chorus", position_ms=10_000.0, duration_ms=90_000.0),
    ]
    heights = [0.2] * 10 + [0.8] * 90  # 100 points, each 1% == 1000ms of the 100_000ms track
    waveform = [WaveformPoint(height=h, red=4, green=4, blue=4) for h in heights]
    track = Track(
        id=1, title="Weighted", artist="Test", bpm=128.0, duration_ms=100_000.0,
        analysis_path="", cues=[], phrases=phrases, beat_grid=BeatGrid(first_beat_ms=0.0, bpm=128.0),
        waveform=waveform,
    )
    energy = compute_track_energy(track)
    assert energy is not None
    assert energy.mean_energy == pytest.approx(0.74)
    assert energy.peak_energy == pytest.approx(0.8)


def test_none_when_no_phrase_data():
    assert compute_track_energy(_no_phrase_track(1, "No Phrases")) is None


def test_none_when_no_waveform_data():
    assert compute_track_energy(_no_waveform_track(1, "No Waveform")) is None


def test_none_when_total_phrase_duration_non_positive():
    # A single zero-duration phrase: compute_phrase_energy still returns
    # a (non-empty) [(phrase, 0.0)] for it (its own i1 > i0 guard takes
    # the else branch), so this exercises compute_track_energy's OWN
    # defensive total_duration_ms <= 0 guard, not compute_phrase_energy's.
    phrase = Phrase(beat_start=1, beat_end=1, kind=1, label="Intro", position_ms=0.0, duration_ms=0.0)
    track = Track(
        id=1, title="Zero Duration Phrase", artist="Test", bpm=128.0, duration_ms=1000.0,
        analysis_path="", cues=[], phrases=[phrase], beat_grid=BeatGrid(first_beat_ms=0.0, bpm=128.0),
        waveform=[WaveformPoint(height=0.5, red=4, green=4, blue=4)],
    )
    assert compute_track_energy(track) is None


# ---------------------------------------------------------------------------
# suggest_energy_flow -- the worked example from the approved plan is the
# centerpiece regression test.
# ---------------------------------------------------------------------------


def test_worked_example_exact_order():
    tracks = [
        _uniform_track(1, "A", 0.1),
        _uniform_track(2, "B", 0.3),
        _uniform_track(3, "C", 0.5),
        _uniform_track(4, "D", 0.6),
        _uniform_track(5, "E", 0.7),
        _uniform_track(6, "F", 0.8),
        _uniform_track(7, "G", 0.9),
        _uniform_track(8, "H", 0.95),
    ]
    result = suggest_energy_flow(tracks, cooldown_fraction=0.25)
    assert [te.track.title for te in result.ordered_tracks] == ["A", "D", "E", "F", "G", "H", "C", "B"]
    assert result.cooldown_start_index == 6
    assert result.unscored == []


def test_default_cooldown_fraction_is_0_15():
    assert DEFAULT_COOLDOWN_FRACTION == 0.15
    tracks = [_uniform_track(i, str(i), i / 10) for i in range(1, 9)]
    assert suggest_energy_flow(tracks) == suggest_energy_flow(tracks, cooldown_fraction=0.15)


def test_zero_tracks_returns_empty_result():
    result = suggest_energy_flow([])
    assert result.ordered_tracks == []
    assert result.cooldown_start_index == 0
    assert result.unscored == []


def test_one_track_has_no_cooldown_phase():
    result = suggest_energy_flow([_uniform_track(1, "Solo", 0.5)])
    assert [te.track.title for te in result.ordered_tracks] == ["Solo"]
    assert result.cooldown_start_index == 1


def test_two_tracks_returns_ascending_no_cooldown():
    tracks = [_uniform_track(1, "Loud", 0.9), _uniform_track(2, "Quiet", 0.2)]
    result = suggest_energy_flow(tracks)
    assert [te.track.title for te in result.ordered_tracks] == ["Quiet", "Loud"]
    assert result.cooldown_start_index == 2


def test_opener_never_enters_cooldown_pool_regardless_of_fraction():
    tracks = [_uniform_track(i, str(i), i / 10) for i in range(1, 11)]  # energies 0.1..1.0
    for fraction in (0.1, 0.5, 0.9, 1.0):
        result = suggest_energy_flow(tracks, cooldown_fraction=fraction)
        assert result.ordered_tracks[0].track.title == "1", f"fraction={fraction}"


def test_build_phase_ascending_cooldown_phase_strictly_descending():
    tracks = [_uniform_track(i, str(i), i / 10) for i in range(1, 11)]
    result = suggest_energy_flow(tracks, cooldown_fraction=0.3)
    build = result.ordered_tracks[: result.cooldown_start_index]
    cooldown = result.ordered_tracks[result.cooldown_start_index :]
    build_energies = [te.mean_energy for te in build]
    cooldown_energies = [te.mean_energy for te in cooldown]
    assert build_energies == sorted(build_energies)
    assert cooldown_energies == sorted(cooldown_energies, reverse=True)


def test_mixed_scorable_and_unscorable_tracks_partition_correctly():
    scorable = _uniform_track(1, "Scorable", 0.5)
    no_phrases = _no_phrase_track(2, "No Phrases")
    no_waveform = _no_waveform_track(3, "No Waveform")

    result = suggest_energy_flow([scorable, no_phrases, no_waveform])

    assert [te.track.id for te in result.ordered_tracks] == [1]
    assert {u.track.id: u.reason for u in result.unscored} == {
        2: "no_phrase_data",
        3: "no_waveform_data",
    }


def test_all_unscored_input_returns_empty_ordered_list():
    result = suggest_energy_flow([_no_waveform_track(1, "A"), _no_waveform_track(2, "B")])
    assert result.ordered_tracks == []
    assert result.cooldown_start_index == 0
    assert len(result.unscored) == 2


# ---------------------------------------------------------------------------
# Real-library test. Structural properties only, never hardcoded counts --
# this project's own established lesson (learned twice already this session
# from a stale Tech House track-count assertion). Scoped to one playlist,
# not the whole library -- load_playlist_tracks() does real per-track ANLZ
# I/O (~266ms/track measured live), and --library was ruled out for this
# feature for exactly that reason.
# ---------------------------------------------------------------------------


@requires_rekordbox
def test_flow_real_library_structural_properties():
    from djcues.db import find_playlist, load_playlist_tracks

    playlist = find_playlist("Tech House")
    assert playlist is not None
    tracks = load_playlist_tracks(playlist.ID)
    assert len(tracks) > 0

    result = suggest_energy_flow(tracks)

    assert len(result.ordered_tracks) + len(result.unscored) == len(tracks)
    assert 0 <= result.cooldown_start_index <= len(result.ordered_tracks)
    build = result.ordered_tracks[: result.cooldown_start_index]
    cooldown = result.ordered_tracks[result.cooldown_start_index :]
    assert [te.mean_energy for te in build] == sorted(te.mean_energy for te in build)
    assert [te.mean_energy for te in cooldown] == sorted((te.mean_energy for te in cooldown), reverse=True)
    assert all(u.reason in ("no_phrase_data", "no_waveform_data") for u in result.unscored)
