"""Tests for the Loop Out (pad H) search: the best drums-no-vocals stretch
of the outro, bar-aligned from Rekordbox's 1.1."""

from __future__ import annotations

import pytest

from djcues.loop_out import LoopOutThresholds, find_loop_out
from djcues.models import BeatGrid, Phrase, Track, WaveformPoint
from djcues.strategy import CueStrategy, build_cue_points, loop_bars_by_pad

# 120 BPM -> 500ms/beat, 2000ms/bar. 60 bars = 120s; 20 waveform points per bar.
BAR_MS = 2000.0
N_BARS = 60
PTS_PER_BAR = 20
FRAME_MS = 1024 / 22050 * 1000

DRUMS = WaveformPoint(height=0.8, red=6, green=1, blue=6)
MELODY = WaveformPoint(height=0.8, red=2, green=6, blue=2)
SILENT = WaveformPoint(height=0.0, red=0, green=0, blue=0)


def _waveform(layout: dict[tuple[int, int], WaveformPoint], default=MELODY) -> list[WaveformPoint]:
    """layout maps (first_bar, end_bar) 0-indexed half-open ranges to a point."""
    wf = [default] * (N_BARS * PTS_PER_BAR)
    for (a, b), pt in layout.items():
        for i in range(a * PTS_PER_BAR, b * PTS_PER_BAR):
            wf[i] = pt
    return wf


def _track(waveform, *, vocal_bars=None, grid=None, duration_ms=N_BARS * BAR_MS) -> Track:
    vocal = None
    if vocal_bars is not None:
        n = int(duration_ms / FRAME_MS) + 1
        vocal = [0] * n
        for a, b in vocal_bars:
            for i in range(int(a * BAR_MS / FRAME_MS), int(b * BAR_MS / FRAME_MS)):
                vocal[i] = 4
    return Track(
        id=1, title="t", artist="a", bpm=120.0, duration_ms=duration_ms, analysis_path="",
        cues=[], phrases=[Phrase(beat_start=1, beat_end=241, kind=6, label="Outro",
                                 position_ms=30 * BAR_MS, duration_ms=30 * BAR_MS)],
        beat_grid=grid or BeatGrid(first_beat_ms=0.0, bpm=120.0),
        waveform=waveform, vocal_track=vocal,
    )


def test_picks_the_drum_stretch_not_the_outro_start():
    wf = _waveform({(38, 46): DRUMS})  # bars 38-45 are drums, the rest melodic
    choice = find_loop_out(_track(wf), 30 * BAR_MS)
    assert choice.bars == 8
    assert choice.start_ms == 38 * BAR_MS
    assert choice.clean


def test_rejects_a_drum_stretch_with_vocals_over_it():
    wf = _waveform({(34, 42): DRUMS, (50, 58): DRUMS})
    track = _track(wf, vocal_bars=[(34, 42)])
    choice = find_loop_out(track, 30 * BAR_MS)
    assert choice.start_ms == 50 * BAR_MS
    assert choice.vocal_overlap == 0


def test_falls_back_to_four_bars_when_eight_do_not_fit():
    wf = _waveform({(56, 60): DRUMS})
    choice = find_loop_out(_track(wf), 56 * BAR_MS)
    assert choice.bars == 4
    assert choice.start_ms == 56 * BAR_MS


def test_best_available_is_flagged_not_clean_when_nothing_has_drums():
    choice = find_loop_out(_track(_waveform({}, default=MELODY)), 30 * BAR_MS)
    assert choice is not None
    assert not choice.clean


def test_silence_is_never_chosen_over_drums():
    wf = _waveform({(30, 40): SILENT, (44, 52): DRUMS}, default=SILENT)
    assert find_loop_out(_track(wf), 30 * BAR_MS).start_ms == 44 * BAR_MS


def test_starts_are_bars_counted_from_bar_one_on_a_mid_bar_grid():
    grid = BeatGrid(first_beat_ms=500.0, bpm=120.0, first_downbeat_beat=3)  # 1.1 = 1500ms
    wf = _waveform({(38, 46): DRUMS})
    choice = find_loop_out(_track(wf, grid=grid), 30 * BAR_MS)
    assert (choice.start_ms - 1500.0) % BAR_MS == pytest.approx(0.0)


def test_no_waveform_means_no_opinion():
    assert find_loop_out(_track(None), 30 * BAR_MS) is None


def test_preferred_length_is_configurable():
    wf = _waveform({(38, 46): DRUMS})
    th = LoopOutThresholds(preferred_bars=4, fallback_bars=4)
    assert find_loop_out(_track(wf), 30 * BAR_MS, th).bars == 4


def test_propose_gives_loop_out_its_own_length_and_leaves_loop_in_at_four():
    wf = _waveform({(38, 46): DRUMS})
    track = _track(wf)
    proposal = CueStrategy(16, 4).propose(track)
    h = next(c for c in proposal.hot_cues if c.kind == 9)
    assert h.position_ms == 38 * BAR_MS
    assert h.loop_end_ms - h.position_ms == 8 * BAR_MS
    mem_h = next(c for c in proposal.memory_cues if c.comment == "Loop Out")
    assert mem_h.position_ms == h.position_ms and mem_h.loop_end_ms == h.loop_end_ms
    b = next(c for c in proposal.hot_cues if c.kind == 2)
    assert b.loop_end_ms - b.position_ms == 4 * BAR_MS
    assert any(n.startswith("H (Loop Out): 8 bars") for n in proposal.notes)


def test_propose_without_a_waveform_keeps_the_old_outro_marker_loop():
    track = _track(None)
    proposal = CueStrategy(16, 4).propose(track)
    h = next(c for c in proposal.hot_cues if c.kind == 9)
    assert h.position_ms == 30 * BAR_MS
    assert h.loop_end_ms - h.position_ms == 4 * BAR_MS
    assert proposal.confidence["H"] == proposal.confidence["G"]


def test_a_rebuild_keeps_loop_outs_own_length():
    track = _track(_waveform({(38, 46): DRUMS}))
    proposal = CueStrategy(16, 4).propose(track)
    assert loop_bars_by_pad(proposal.hot_cues, track)["H"] == 8
    positions = {"H": 38 * BAR_MS, "B": 0.0}
    hot, _ = build_cue_points(positions, {"H": 1.0, "B": 1.0}, track, 16, 4,
                              loop_bars_by_pad=loop_bars_by_pad(proposal.hot_cues, track))
    h = next(c for c in hot if c.kind == 9)
    assert h.loop_end_ms - h.position_ms == 8 * BAR_MS


def test_the_heuristic_cache_key_carries_an_algorithm_version():
    from djcues.analysis_cache import HEURISTIC_VERSION, cue_proposal_key

    key = cue_proposal_key(agentic=False, provider=None, model=None, skip_critic=False,
                           refine_drops=False, deep=False, offset_bars=16, loop_bars=4)
    assert key.model == HEURISTIC_VERSION


# --- moving the Loop Out cues djcues already wrote --------------------------


def _loop_cue(kind, in_ms, out_ms, content_id="1"):
    from types import SimpleNamespace

    return SimpleNamespace(ContentID=content_id, Kind=kind, Comment="Loop Out",
                           InMsec=in_ms, OutMsec=out_ms)


def _plan(monkeypatch, cues, track):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from djcues import db as db_module
    from djcues.writer import plan_loop_out_realignment

    db = MagicMock()
    db.get_cue.return_value = cues
    db.get_content.return_value = SimpleNamespace(Title="t", FolderPath="C:/x/a.mp3")
    monkeypatch.setattr(db_module, "load_track", lambda content, db=None: track)
    return plan_loop_out_realignment(db)


def test_realign_moves_a_pristine_old_style_loop_out(monkeypatch):
    track = _track(_waveform({(38, 46): DRUMS}))
    old = 30 * BAR_MS
    plan = _plan(monkeypatch, [_loop_cue(9, old, old + 4 * BAR_MS), _loop_cue(0, old, old + 4 * BAR_MS)], track)
    assert len(plan["changes"]) == 2
    assert {(c["new_in"], c["new_out"]) for c in plan["changes"]} == {(38 * 2000, 46 * 2000)}


def test_realign_leaves_a_loop_you_moved_or_resized(monkeypatch):
    track = _track(_waveform({(38, 46): DRUMS}))
    moved = _plan(monkeypatch, [_loop_cue(9, 50 * BAR_MS, 54 * BAR_MS)], track)
    assert moved["changes"] == [] and moved["skipped"]["hand-edited or already re-placed"]
    resized = _plan(monkeypatch, [_loop_cue(9, 30 * BAR_MS, 38 * BAR_MS)], track)
    assert resized["changes"] == []


def test_realign_leaves_tracks_without_a_clean_stretch(monkeypatch):
    track = _track(_waveform({}, default=MELODY))
    plan = _plan(monkeypatch, [_loop_cue(9, 30 * BAR_MS, 34 * BAR_MS)], track)
    assert plan["changes"] == [] and plan["skipped"]["no clean drums-only stretch"]
