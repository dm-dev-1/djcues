"""Tests for Rekordbox's bar 1.1 (the first downbeat) as the origin for
pad A and every bar-level calculation.

Rekordbox grids often start mid-bar -- the grid's first beat is beat 2, 3
or 4 of a bar (169 of 624 gridded tracks in the real library) -- while
Rekordbox's own bar counter starts at 1.1, the first beat whose
beat-in-bar is 1. These tests use a grid whose first beat is beat 3 of a
bar, so 1.1 is grid beat 3 (two beats after the first grid beat).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from djcues import analysis_cache
from djcues.models import BeatGrid, Phrase, Track
from djcues.review import create_session, render_review_html
from djcues.strategy import CueStrategy, build_cue_points
from djcues.transition import suggest_transitions

# 120 BPM -> 500ms/beat. First grid beat at 500ms; 1.1 is grid beat 3 = 1500ms.
_MS_PER_BEAT = 500.0


def _grid(first_downbeat_beat: int = 3) -> BeatGrid:
    return BeatGrid(first_beat_ms=500.0, bpm=120.0, first_downbeat_beat=first_downbeat_beat)


def _phrase(label: str, beat_start: int, beats: int, bg: BeatGrid) -> Phrase:
    pos = bg.beat_to_ms(beat_start)
    return Phrase(
        beat_start=beat_start, beat_end=beat_start + beats, kind=1, label=label,
        position_ms=pos, duration_ms=beats * bg.ms_per_beat,
    )


def _track(bg: BeatGrid | None = None, track_id: int = 1) -> Track:
    """Phrases aligned to the 1.1 phase (beats 3, 67, 131, ...), as real
    PSSI phrases are (94% of real phrase starts sit on the 1.1 phase)."""
    bg = bg or _grid()
    phrases = [
        _phrase("Intro", 3, 64, bg),     # 16 bars
        _phrase("Up", 67, 64, bg),
        _phrase("Chorus", 131, 64, bg),  # the Drop (well past 20% of the track)
        _phrase("Down", 195, 64, bg),
        _phrase("Outro", 259, 64, bg),
    ]
    return Track(
        id=track_id, title="Mid-bar grid", artist="Test", bpm=bg.bpm,
        duration_ms=bg.beat_to_ms(323), analysis_path="", cues=[], phrases=phrases, beat_grid=bg,
    )


# --- BeatGrid ---------------------------------------------------------------


def test_bar_one_is_the_first_downbeat_not_the_first_grid_beat():
    bg = _grid()
    assert bg.beat_to_ms(1) == 500.0
    assert bg.bar_one_ms == 1500.0


def test_default_grid_keeps_bar_one_on_the_first_beat():
    bg = BeatGrid(first_beat_ms=77.0, bpm=128.0)
    assert bg.first_downbeat_beat == 1
    assert bg.bar_one_ms == 77.0


@pytest.mark.parametrize("beat,expected", [
    (3, 3), (4, 3), (6, 3), (7, 7), (66, 63), (67, 67),
    (1, -1), (2, -1),  # lead-in before 1.1 belongs to the bar before it
])
def test_bar_start_beat_counts_bars_from_bar_one(beat, expected):
    assert _grid().bar_start_beat(beat) == expected


def test_bar_start_beat_matches_the_old_formula_when_grid_starts_on_a_bar():
    bg = BeatGrid(first_beat_ms=0.0, bpm=128.0)
    for beat in range(1, 40):
        assert bg.bar_start_beat(beat) == ((beat - 1) // 4) * 4 + 1


# --- db extraction ----------------------------------------------------------


class PQTZAnlzTag:  # name matters: db.py matches on type(tag).__name__
    def __init__(self, beats, times):
        self._beats, self._times = beats, times

    def get_beats(self):
        return self._beats

    def get_times(self):
        return self._times


def _content():
    return SimpleNamespace(BPM=12000, Title="t")


def _db_with_grid(beats, times):
    af = SimpleNamespace(tags=[PQTZAnlzTag(beats, times)])
    return SimpleNamespace(read_anlz_files=lambda content: {Path("ANLZ0000.DAT"): af})


def test_extract_beat_grid_finds_the_first_downbeat():
    from djcues.db import _extract_beat_grid

    bg = _extract_beat_grid(_content(), db=_db_with_grid([3, 4, 1, 2, 3, 4, 1], [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]))
    assert bg.first_beat_ms == 500.0
    assert bg.first_downbeat_beat == 3
    assert bg.bar_one_ms == 1500.0


def test_extract_beat_grid_starting_on_a_downbeat():
    from djcues.db import _extract_beat_grid

    bg = _extract_beat_grid(_content(), db=_db_with_grid([1, 2, 3, 4, 1], [0.1, 0.6, 1.1, 1.6, 2.1]))
    assert bg.first_downbeat_beat == 1


def test_extract_beat_grid_without_any_downbeat_falls_back_to_beat_one():
    from djcues.db import _extract_beat_grid

    bg = _extract_beat_grid(_content(), db=_db_with_grid([2, 3, 4], [0.1, 0.6, 1.1]))
    assert bg.first_downbeat_beat == 1


# --- cue placement ----------------------------------------------------------


def test_pad_a_and_b_land_on_bar_one():
    proposal = CueStrategy().propose(_track())
    by_kind = {c.kind: c for c in proposal.hot_cues}
    assert by_kind[1].position_ms == 1500.0  # A: First Beat at 1.1, not 500ms
    assert by_kind[2].position_ms == 1500.0  # B: Loop In follows A
    assert any("bar 1.1" in n and "grid beat 3" in n for n in proposal.notes)


def test_pad_a_note_for_a_grid_that_starts_on_a_bar():
    proposal = CueStrategy().propose(_track(BeatGrid(first_beat_ms=500.0, bpm=120.0)))
    assert "A (First Beat): bar 1.1" in proposal.notes


def test_memory_cues_snap_to_bars_counted_from_bar_one():
    # Regression: snapping memory cues to "bar starts" counted from grid
    # beat 1 moved them 2 beats early on this grid. The Drop memory cue is
    # 16 bars before the Drop (beat 131 -> beat 67), which is already a
    # real bar start and must stay there.
    track = _track()
    bg = track.beat_grid
    hot, memory = build_cue_points({"D": bg.beat_to_ms(131)}, {"D": 0.85}, track, 16, 4)
    drop_memory = next(c for c in memory if c.comment == "Drop")
    assert drop_memory.position_ms == bg.beat_to_ms(67)


def test_memory_cue_off_the_bar_snaps_back_to_a_real_bar_start():
    track = _track()
    bg = track.beat_grid
    # A hot cue one beat after a bar start: its memory cue lands one beat
    # into a bar, and must snap back to that bar's start (beat 67), not to
    # a grid-beat-1-relative "bar" (beat 65).
    hot, memory = build_cue_points({"D": bg.beat_to_ms(132)}, {"D": 0.85}, track, 16, 4)
    drop_memory = next(c for c in memory if c.comment == "Drop")
    assert drop_memory.position_ms == bg.beat_to_ms(67)


def test_memory_cue_clamps_to_bar_one_not_the_first_grid_beat():
    track = _track()
    bg = track.beat_grid
    hot, memory = build_cue_points({"C": bg.beat_to_ms(35)}, {"C": 0.85}, track, 16, 4)
    buildup_memory = next(c for c in memory if c.comment == "Buildup")
    assert buildup_memory.position_ms == 1500.0


# --- transitions --------------------------------------------------------------


def test_transition_mix_in_is_bar_one():
    a = _track(BeatGrid(first_beat_ms=0.0, bpm=120.0), track_id=1)
    b = _track(_grid(), track_id=2)
    result = suggest_transitions([a, b])
    assert result.suggestions[0].mix_in_ms == 1500.0


# --- cache fingerprint ----------------------------------------------------------


def test_fingerprint_unchanged_for_grids_that_start_on_a_bar():
    # Cached results (including expensive --deep/--agentic ones) for the
    # ~73% of tracks unaffected by 1.1 must stay valid.
    t1 = _track(BeatGrid(first_beat_ms=500.0, bpm=120.0))
    t2 = _track(BeatGrid(first_beat_ms=500.0, bpm=120.0, first_downbeat_beat=1))
    assert analysis_cache.fingerprint_track_analysis(t1) == analysis_cache.fingerprint_track_analysis(t2)


def test_fingerprint_changes_when_bar_one_moves():
    on_bar = _track(BeatGrid(first_beat_ms=500.0, bpm=120.0))
    mid_bar = _track(_grid())
    # Same phrases/waveform, so only first_downbeat_beat can explain a difference.
    mid_bar.phrases = on_bar.phrases
    assert analysis_cache.fingerprint_track_analysis(on_bar) != analysis_cache.fingerprint_track_analysis(mid_bar)


# --- review UI ------------------------------------------------------------------


def test_review_session_and_html_carry_bar_one():
    track = _track()
    proposal = CueStrategy().propose(track)
    session = create_session("P", 1, [(track, proposal)])
    assert session["tracks"]["1"]["bar_one_ms"] == 1500.0
    page = render_review_html("P", [(track, proposal)], "s.json", "http://127.0.0.1:1")
    assert 'data-bar-one-ms="1500.0"' in page
    # Bar numbering in the editor counts from 1.1
    assert "function msToBar(ms, barOneMs, msPerBeat)" in page
