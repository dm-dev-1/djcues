"""Tests for djcues.tag_analysis -- the I/O orchestration around the pure
tagging rules. Rekordbox is faked: only the db_module functions the
module actually calls are patched, so nothing here touches a real
database or ANLZ file."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from djcues import tag_analysis
from djcues.models import BeatGrid, Phrase, Track, WaveformPoint
from djcues.tag_store import TagStore


def _content(cid, title="T", key="8A", bpm=12800, length=200):
    return SimpleNamespace(
        ID=str(cid), Title=title, Artist=SimpleNamespace(Name="A"), Key=SimpleNamespace(ScaleName=key),
        BPM=bpm, Length=length, Commnt=None,
    )


def _track(cid) -> Track:
    # Intro 8 bars + Outro 8 bars, uniform energy 0.6.
    phrases = [
        Phrase(beat_start=1, beat_end=33, kind=1, label="Intro", position_ms=0.0, duration_ms=15000.0),
        Phrase(beat_start=33, beat_end=65, kind=1, label="Outro", position_ms=15000.0, duration_ms=15000.0),
    ]
    return Track(
        id=str(cid), title="T", artist="A", bpm=128.0, duration_ms=200_000.0, analysis_path="", cues=[],
        phrases=phrases, beat_grid=BeatGrid(first_beat_ms=0.0, bpm=128.0),
        waveform=[WaveformPoint(height=0.6, red=1, green=1, blue=1) for _ in range(20)],
    )


@pytest.fixture
def fakes(monkeypatch):
    """Patch the three Rekordbox-touching functions; record calls."""
    calls = {"load": [], "raw": [], "fp": {}}

    def fingerprint(content, db=None):
        return calls["fp"].get(content.ID, f"fp-{content.ID}")

    def load_track(content, db=None):
        calls["load"].append(content.ID)
        if content.Title == "BROKEN":
            raise RuntimeError("corrupt analysis")
        return _track(content.ID)

    def raw_grid(content, db=None):
        calls["raw"].append(content.ID)
        return None

    monkeypatch.setattr(tag_analysis.db_module, "fingerprint_anlz", fingerprint)
    monkeypatch.setattr(tag_analysis.db_module, "load_track", load_track)
    monkeypatch.setattr(tag_analysis.db_module, "extract_raw_beat_grid", raw_grid)
    return calls


def test_cold_run_loads_every_track_and_caches(tmp_path, fakes):
    contents = [_content(1), _content(2)]
    with TagStore(tmp_path / "t.db") as store:
        run = tag_analysis.collect_features(contents, None, store)
        assert fakes["load"] == ["1", "2"]
        assert run.cache_hits == 0
        assert set(run.features) == {"1", "2"}
        assert store.count_features() == 2


def test_warm_run_parses_nothing(tmp_path, fakes):
    contents = [_content(1), _content(2)]
    with TagStore(tmp_path / "t.db") as store:
        tag_analysis.collect_features(contents, None, store)
        fakes["load"].clear()
        run = tag_analysis.collect_features(contents, None, store)
    assert fakes["load"] == []
    assert run.cache_hits == 2


def test_a_changed_fingerprint_reparses_only_that_track(tmp_path, fakes):
    contents = [_content(1), _content(2)]
    with TagStore(tmp_path / "t.db") as store:
        tag_analysis.collect_features(contents, None, store)
        fakes["load"].clear()
        fakes["fp"]["2"] = "fp-2-reanalysed"
        run = tag_analysis.collect_features(contents, None, store)
    assert fakes["load"] == ["2"]
    assert run.cache_hits == 1


def test_use_cache_false_reparses_everything(tmp_path, fakes):
    contents = [_content(1)]
    with TagStore(tmp_path / "t.db") as store:
        tag_analysis.collect_features(contents, None, store)
        fakes["load"].clear()
        tag_analysis.collect_features(contents, None, store, use_cache=False)
    assert fakes["load"] == ["1"]


def test_only_cached_never_parses_and_counts_misses(tmp_path, fakes):
    contents = [_content(1), _content(2)]
    with TagStore(tmp_path / "t.db") as store:
        tag_analysis.collect_features([contents[0]], None, store)  # cache track 1 only
        fakes["load"].clear()
        run = tag_analysis.collect_features(contents, None, store, only_cached=True)
    assert fakes["load"] == []
    assert set(run.features) == {"1"}
    assert run.uncached == 1


def test_one_bad_track_is_recorded_and_the_run_continues(tmp_path, fakes):
    contents = [_content(1), _content(2, title="BROKEN"), _content(3)]
    with TagStore(tmp_path / "t.db") as store:
        run = tag_analysis.collect_features(contents, None, store)
    assert set(run.features) == {"1", "3"}
    assert run.failures == [("BROKEN", "corrupt analysis")]


def test_progress_callback_fires_once_per_track(tmp_path, fakes):
    seen = []
    with TagStore(tmp_path / "t.db") as store:
        tag_analysis.collect_features(
            [_content(1), _content(2)], None, store, progress=lambda d, t, c: seen.append((d, t, c)),
        )
        seen_warm = []
        tag_analysis.collect_features(
            [_content(1), _content(2)], None, store, progress=lambda d, t, c: seen_warm.append((d, t, c)),
        )
    assert seen == [(1, 2, False), (2, 2, False)]
    assert seen_warm == [(1, 2, True), (2, 2, True)]


def test_calibrate_library_stores_cutpoints(tmp_path, fakes, monkeypatch):
    # 10 tracks with distinct energies so quintiles are well-defined.
    energies = {str(i): 0.1 * i for i in range(1, 11)}

    def load_track(content, db=None):
        t = _track(content.ID)
        t.waveform = [WaveformPoint(height=energies[content.ID], red=1, green=1, blue=1) for _ in range(20)]
        return t

    monkeypatch.setattr(tag_analysis.db_module, "load_track", load_track)
    db = SimpleNamespace(get_content=lambda: [_content(i) for i in range(1, 11)])
    with TagStore(tmp_path / "t.db") as store:
        calibration, run = tag_analysis.calibrate_library(db, store)
    assert len(calibration.cutpoints) == 4
    assert calibration.cutpoints == sorted(calibration.cutpoints)
    assert calibration.n_tracks == 10 and calibration.n_excluded == 0


def test_calibrate_excludes_loops_and_samples(tmp_path, fakes, monkeypatch):
    def load_track(content, db=None):
        t = _track(content.ID)
        if content.ID == "1":
            t.duration_ms = 20_000.0  # a loop
        return t

    monkeypatch.setattr(tag_analysis.db_module, "load_track", load_track)
    db = SimpleNamespace(get_content=lambda: [_content(i) for i in range(1, 9)])
    with TagStore(tmp_path / "t.db") as store:
        calibration, _run = tag_analysis.calibrate_library(db, store)
    assert calibration.n_tracks == 7 and calibration.n_excluded == 1


def test_calibrate_refuses_a_tiny_library(tmp_path, fakes):
    db = SimpleNamespace(get_content=lambda: [_content(1), _content(2)])
    with TagStore(tmp_path / "t.db") as store:
        with pytest.raises(ValueError, match="at least"):
            tag_analysis.calibrate_library(db, store)
        assert store.latest_calibration() is None  # nothing stored on failure


def test_propose_tags_uses_fresh_key_status_even_from_cache(tmp_path, fakes):
    # Features come from the cache, but the key is read from the content
    # row each time, so changing it in Rekordbox is reflected immediately.
    with TagStore(tmp_path / "t.db") as store:
        tag_analysis.collect_features([_content(1, key="8A")], None, store)
        proposals, run = tag_analysis.propose_tags(
            [_content(1, key=None)], None, store, calibration=None,
        )
    assert run.cache_hits == 1
    decisions = {d.tag for d in proposals[0].tags.decisions}
    assert "No Key" in decisions


def test_propose_tags_keeps_input_order_and_identity(tmp_path, fakes):
    contents = [_content(3, title="Third"), _content(1, title="First")]
    with TagStore(tmp_path / "t.db") as store:
        proposals, _run = tag_analysis.propose_tags(contents, None, store, calibration=None)
    assert [(p.track_id, p.title, p.artist) for p in proposals] == [("3", "Third", "A"), ("1", "First", "A")]


def test_propose_tags_skips_tracks_that_failed(tmp_path, fakes):
    contents = [_content(1), _content(2, title="BROKEN")]
    with TagStore(tmp_path / "t.db") as store:
        proposals, run = tag_analysis.propose_tags(contents, None, store, calibration=None)
    assert [p.track_id for p in proposals] == ["1"]
    assert len(run.failures) == 1


def test_playlist_contents_is_trackno_ordered_and_drops_orphans():
    songs = [
        SimpleNamespace(TrackNo=3, Content="c"),
        SimpleNamespace(TrackNo=1, Content="a"),
        SimpleNamespace(TrackNo=2, Content=None),  # orphaned row
        SimpleNamespace(TrackNo=None, Content="z"),  # None sorts first, like db.list_playlist_tracks
    ]
    db = SimpleNamespace(get_playlist_songs=lambda PlaylistID: songs)
    assert tag_analysis.playlist_contents(db, "p") == ["z", "a", "c"]
