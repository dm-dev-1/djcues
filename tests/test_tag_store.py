"""Tests for djcues.tag_store -- calibrations and the feature cache."""

from __future__ import annotations

import sqlite3

from djcues.tag_store import TagStore
from djcues.tagging import AnalysisFeatures


def _features(track_id="1", mean_energy=0.5) -> AnalysisFeatures:
    return AnalysisFeatures(
        track_id=track_id, duration_ms=200_000.0, bpm=128.0, mean_energy=mean_energy, peak_energy=0.9,
        intro_bars=8.0, outro_bars=8.0, vocal_pct=10.0, vocal_intro=[(1.0, 2.0)], vocal_outro=[],
        tempo_varies=False, grid_consistent=True,
    )


def test_no_calibration_before_one_is_saved(tmp_path):
    with TagStore(tmp_path / "tags.db") as store:
        assert store.latest_calibration() is None


def test_calibration_round_trip(tmp_path):
    with TagStore(tmp_path / "tags.db") as store:
        store.save_calibration([0.2, 0.4, 0.6, 0.8], n_tracks=100, n_excluded=7, thresholds={"long_intro_bars": 16.0})
        cal = store.latest_calibration()
    assert cal.cutpoints == [0.2, 0.4, 0.6, 0.8]
    assert (cal.n_tracks, cal.n_excluded) == (100, 7)
    assert cal.thresholds == {"long_intro_bars": 16.0}
    assert cal.created_at  # an ISO timestamp


def test_latest_calibration_is_the_most_recent(tmp_path):
    with TagStore(tmp_path / "tags.db") as store:
        first = store.save_calibration([0.1], 10, 0, {})
        second = store.save_calibration([0.9], 20, 0, {})
        cal = store.latest_calibration()
    assert second > first
    assert cal.id == second and cal.cutpoints == [0.9]


def test_calibration_survives_reopening_the_store(tmp_path):
    path = tmp_path / "tags.db"
    with TagStore(path) as store:
        store.save_calibration([0.5], 5, 0, {})
    with TagStore(path) as store:
        assert store.latest_calibration().cutpoints == [0.5]


def test_feature_cache_hit_requires_the_same_fingerprint(tmp_path):
    with TagStore(tmp_path / "tags.db") as store:
        store.put_features(_features("1"), "fp-a")
        assert store.get_features("1", "fp-a") == _features("1")
        assert store.get_features("1", "fp-b") is None  # track changed -> miss
        assert store.get_features("2", "fp-a") is None  # unknown track -> miss


def test_put_features_replaces_the_previous_entry(tmp_path):
    with TagStore(tmp_path / "tags.db") as store:
        store.put_features(_features("1", mean_energy=0.5), "fp-a")
        store.put_features(_features("1", mean_energy=0.9), "fp-b")
        assert store.count_features() == 1
        assert store.get_features("1", "fp-a") is None
        assert store.get_features("1", "fp-b").mean_energy == 0.9


def test_track_ids_are_compared_as_strings(tmp_path):
    with TagStore(tmp_path / "tags.db") as store:
        store.put_features(_features("123"), "fp")
        assert store.get_features(123, "fp") is not None


def test_uncommitted_puts_are_not_visible_until_commit(tmp_path):
    path = tmp_path / "tags.db"
    with TagStore(path) as store:
        store.put_features(_features("1"), "fp", commit=False)
        other = sqlite3.connect(path)
        try:
            assert other.execute("SELECT COUNT(*) FROM track_features").fetchone()[0] == 0
        finally:
            other.close()
        store.commit()
        other = sqlite3.connect(path)
        try:
            assert other.execute("SELECT COUNT(*) FROM track_features").fetchone()[0] == 1
        finally:
            other.close()


def test_clear_features_returns_how_many_were_removed(tmp_path):
    with TagStore(tmp_path / "tags.db") as store:
        store.put_features(_features("1"), "fp")
        store.put_features(_features("2"), "fp")
        assert store.clear_features() == 2
        assert store.count_features() == 0
        assert store.clear_features() == 0


def test_clearing_features_keeps_the_calibration(tmp_path):
    with TagStore(tmp_path / "tags.db") as store:
        store.save_calibration([0.5], 5, 0, {})
        store.put_features(_features("1"), "fp")
        store.clear_features()
        assert store.latest_calibration() is not None
