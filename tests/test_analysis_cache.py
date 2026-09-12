"""Tests for djcues.analysis_cache -- the persistent cache of completed
analysis runs. Real sqlite3 I/O throughout via an explicit db_path=tmp_path
param, matching test_history.py's own established pattern for this
codebase's ~/.djcues/-backed modules (no monkeypatching Path.home()).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from djcues.analysis_cache import (
    beatgrid_key,
    clear_all,
    cue_proposal_key,
    estimate,
    fingerprint_beat_grid,
    fingerprint_track_analysis,
    get_cached,
    store_result,
    summary,
)
from djcues.models import BeatGrid, Phrase, RawBeatGridEntry, Track, WaveformPoint

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "analysis_cache.db"


@pytest.fixture
def beat_grid() -> BeatGrid:
    return BeatGrid(first_beat_ms=77.0, bpm=128.0)


@pytest.fixture
def phrases(beat_grid: BeatGrid) -> list[Phrase]:
    raw = [(1, 1, "Intro"), (33, 2, "Up"), (145, 5, "Chorus"), (401, 6, "Outro")]
    end_beat = 433
    result = []
    for i, (beat, kind, label) in enumerate(raw):
        next_beat = raw[i + 1][0] if i + 1 < len(raw) else end_beat
        pos = beat_grid.beat_to_ms(beat)
        end_pos = beat_grid.beat_to_ms(next_beat)
        result.append(Phrase(
            beat_start=beat, beat_end=next_beat, kind=kind, label=label,
            position_ms=pos, duration_ms=end_pos - pos,
        ))
    return result


@pytest.fixture
def track(beat_grid: BeatGrid, phrases: list[Phrase]) -> Track:
    return Track(
        id=1, title="Test Track", artist="Test Artist", bpm=128.0,
        duration_ms=240_000.0, analysis_path="", cues=[], phrases=phrases,
        beat_grid=beat_grid,
    )


@pytest.fixture
def heuristic_key():
    return cue_proposal_key(
        agentic=False, provider=None, model=None, skip_critic=False,
        refine_drops=True, deep=True, offset_bars=16, loop_bars=4,
    )


# ---------------------------------------------------------------------------
# get_cached / store_result roundtrip
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_get_cached_returns_none_when_never_stored(self, db_path, heuristic_key, track):
        fp = fingerprint_track_analysis(track)
        assert get_cached(track.id, heuristic_key, fp, db_path=db_path) is None

    def test_store_then_get_cached_roundtrips_result(self, db_path, heuristic_key, track):
        fp = fingerprint_track_analysis(track)
        result = {"positions": {"A": 77.0}, "confidence": {"A": 1.0}, "notes": ["ok"]}
        store_result(track.id, heuristic_key, fp, result, track_title=track.title,
                     track_artist=track.artist, cost_usd=None, compute_seconds=190.4,
                     db_path=db_path)

        cached = get_cached(track.id, heuristic_key, fp, db_path=db_path)

        assert cached is not None
        assert cached.result == result
        assert cached.compute_seconds == 190.4
        assert cached.source == "cli"
        assert cached.created_at and cached.updated_at

    def test_store_result_upsert_preserves_created_at_bumps_updated_at(
        self, db_path, heuristic_key, track
    ):
        fp = fingerprint_track_analysis(track)
        store_result(track.id, heuristic_key, fp, {"positions": {}, "confidence": {}, "notes": []},
                     db_path=db_path)
        first = get_cached(track.id, heuristic_key, fp, db_path=db_path)

        store_result(track.id, heuristic_key, fp, {"positions": {"A": 1.0}, "confidence": {}, "notes": []},
                     db_path=db_path)
        second = get_cached(track.id, heuristic_key, fp, db_path=db_path)

        assert second.result == {"positions": {"A": 1.0}, "confidence": {}, "notes": []}
        assert second.created_at == first.created_at
        # Real clock resolution here is seconds; the important, always-true
        # property is that a second store never regresses created_at, not
        # that updated_at strictly advances within the same test process.
        assert second.updated_at >= first.updated_at

    def test_store_result_does_not_duplicate_rows_on_repeat_calls(
        self, db_path, heuristic_key, track
    ):
        fp = fingerprint_track_analysis(track)
        for _ in range(3):
            store_result(track.id, heuristic_key, fp, {"positions": {}, "confidence": {}, "notes": []},
                         db_path=db_path)

        rows = summary(db_path=db_path)
        assert len(rows) == 1
        assert rows[0]["total"] == 1

    def test_cache_hit_occurs_across_different_devices_for_same_key(
        self, db_path, heuristic_key, track
    ):
        """The regression guard for this feature's own design decision:
        device is informational only (see store_result()'s own
        docstring) -- a track cached while computed on cpu must still
        hit on a later cuda run of the identical analysis, and a cuda-
        computed row must still hit from a plain cpu-default caller.
        Device is deliberately NOT part of AnalysisKey/the UNIQUE
        constraint, so this is really just confirming that stays true."""
        fp = fingerprint_track_analysis(track)
        store_result(track.id, heuristic_key, fp, {"positions": {}, "confidence": {}, "notes": []},
                     device="cpu", db_path=db_path)

        # A caller resolving to "cuda" still hits the row cpu wrote.
        cached = get_cached(track.id, heuristic_key, fp, db_path=db_path)
        assert cached is not None

        # Overwriting with a "cuda"-computed result is still one row
        # (upsert, not a second cache entry keyed by device).
        store_result(track.id, heuristic_key, fp, {"positions": {"A": 1.0}, "confidence": {}, "notes": []},
                     device="cuda", db_path=db_path)
        rows = summary(db_path=db_path)
        assert len(rows) == 1
        assert rows[0]["total"] == 1

        # And the plain cpu-default caller still hits the now-cuda row.
        cached2 = get_cached(track.id, heuristic_key, fp, db_path=db_path)
        assert cached2.result == {"positions": {"A": 1.0}, "confidence": {}, "notes": []}


# ---------------------------------------------------------------------------
# Fingerprint-mismatch invalidation -- the core safety property behind
# making the cache default-on rather than opt-in.
# ---------------------------------------------------------------------------


class TestFingerprintInvalidation:
    def test_get_cached_returns_none_on_fingerprint_mismatch(
        self, db_path, heuristic_key, track
    ):
        stale_fp = fingerprint_track_analysis(track)
        store_result(track.id, heuristic_key, stale_fp, {"positions": {}, "confidence": {}, "notes": []},
                     db_path=db_path)

        # Simulate rekordbox having re-analyzed the track: bpm changed.
        changed_track = Track(
            id=track.id, title=track.title, artist=track.artist, bpm=140.0,
            duration_ms=track.duration_ms, analysis_path="", cues=[],
            phrases=track.phrases, beat_grid=track.beat_grid,
        )
        fresh_fp = fingerprint_track_analysis(changed_track)

        assert get_cached(track.id, heuristic_key, fresh_fp, db_path=db_path) is None
        # The stale entry is still retrievable under its own real fingerprint --
        # a mismatch means "don't serve this," not "the row was deleted."
        assert get_cached(track.id, heuristic_key, stale_fp, db_path=db_path) is not None


class TestTrackFingerprint:
    def test_stable_for_identical_track(self, track):
        assert fingerprint_track_analysis(track) == fingerprint_track_analysis(track)

    def test_stable_when_title_or_artist_changes(self, track):
        retagged = Track(
            id=track.id, title="Renamed", artist="Someone Else", bpm=track.bpm,
            duration_ms=track.duration_ms, analysis_path="", cues=[],
            phrases=track.phrases, beat_grid=track.beat_grid,
            waveform=track.waveform, vocal_track=track.vocal_track,
        )
        assert fingerprint_track_analysis(track) == fingerprint_track_analysis(retagged)

    def test_changes_when_phrases_change(self, track, beat_grid):
        different_phrases = [Phrase(beat_start=1, beat_end=433, kind=1, label="Intro",
                                     position_ms=77.0, duration_ms=1000.0)]
        reanalyzed = Track(
            id=track.id, title=track.title, artist=track.artist, bpm=track.bpm,
            duration_ms=track.duration_ms, analysis_path="", cues=[],
            phrases=different_phrases, beat_grid=beat_grid,
        )
        assert fingerprint_track_analysis(track) != fingerprint_track_analysis(reanalyzed)

    def test_changes_when_vocal_track_changes(self, track):
        with_vocals = Track(
            id=track.id, title=track.title, artist=track.artist, bpm=track.bpm,
            duration_ms=track.duration_ms, analysis_path="", cues=[],
            phrases=track.phrases, beat_grid=track.beat_grid,
            vocal_track=[0, 1, 2, 3],
        )
        assert fingerprint_track_analysis(track) != fingerprint_track_analysis(with_vocals)

    def test_changes_when_waveform_changes(self, track):
        with_waveform = Track(
            id=track.id, title=track.title, artist=track.artist, bpm=track.bpm,
            duration_ms=track.duration_ms, analysis_path="", cues=[],
            phrases=track.phrases, beat_grid=track.beat_grid,
            waveform=[WaveformPoint(height=0.5, red=1, green=2, blue=3)],
        )
        assert fingerprint_track_analysis(track) != fingerprint_track_analysis(with_waveform)


class TestBeatGridFingerprint:
    def test_stable_for_same_entries(self):
        entries = [RawBeatGridEntry(beat_in_bar=1, bpm=128.0, time_ms=77.0)]
        assert fingerprint_beat_grid(entries) == fingerprint_beat_grid(list(entries))

    def test_changes_when_entries_change(self):
        a = [RawBeatGridEntry(beat_in_bar=1, bpm=128.0, time_ms=77.0)]
        b = [RawBeatGridEntry(beat_in_bar=1, bpm=140.0, time_ms=77.0)]
        assert fingerprint_beat_grid(a) != fingerprint_beat_grid(b)

    def test_none_entries_has_stable_sentinel(self):
        assert fingerprint_beat_grid(None) == fingerprint_beat_grid(None)
        assert fingerprint_beat_grid(None) != fingerprint_beat_grid([])


# ---------------------------------------------------------------------------
# Key-dimension independence -- every dimension that can change the actual
# computed result must produce a genuinely separate cache entry.
# ---------------------------------------------------------------------------


class TestKeyDimensions:
    def test_different_offset_bars_are_different_cache_entries(self, db_path, track):
        fp = fingerprint_track_analysis(track)
        key_16 = cue_proposal_key(agentic=False, provider=None, model=None, skip_critic=False,
                                   refine_drops=False, deep=False, offset_bars=16, loop_bars=4)
        key_8 = cue_proposal_key(agentic=False, provider=None, model=None, skip_critic=False,
                                  refine_drops=False, deep=False, offset_bars=8, loop_bars=4)
        store_result(track.id, key_16, fp, {"positions": {"offset": 16}, "confidence": {}, "notes": []},
                     db_path=db_path)

        assert get_cached(track.id, key_8, fp, db_path=db_path) is None
        assert get_cached(track.id, key_16, fp, db_path=db_path) is not None

    def test_different_engine_heuristic_vs_agentic_are_different_entries(self, db_path, track):
        fp = fingerprint_track_analysis(track)
        heuristic = cue_proposal_key(agentic=False, provider=None, model=None, skip_critic=False,
                                      refine_drops=False, deep=False, offset_bars=16, loop_bars=4)
        agentic = cue_proposal_key(agentic=True, provider="gemini", model="gemini-flash-lite-latest",
                                    skip_critic=False, refine_drops=False, deep=False,
                                    offset_bars=16, loop_bars=4)
        store_result(track.id, heuristic, fp, {"positions": {"e": "heuristic"}, "confidence": {}, "notes": []},
                     db_path=db_path)

        assert get_cached(track.id, agentic, fp, db_path=db_path) is None
        assert get_cached(track.id, heuristic, fp, db_path=db_path) is not None

    def test_different_provider_or_model_are_different_entries(self, db_path, track):
        fp = fingerprint_track_analysis(track)
        gemini = cue_proposal_key(agentic=True, provider="gemini", model="gemini-flash-lite-latest",
                                   skip_critic=False, refine_drops=False, deep=False,
                                   offset_bars=16, loop_bars=4)
        anthropic = cue_proposal_key(agentic=True, provider="anthropic", model="claude-sonnet-5",
                                      skip_critic=False, refine_drops=False, deep=False,
                                      offset_bars=16, loop_bars=4)
        store_result(track.id, gemini, fp, {"positions": {}, "confidence": {}, "notes": []}, db_path=db_path)

        assert get_cached(track.id, anthropic, fp, db_path=db_path) is None
        assert get_cached(track.id, gemini, fp, db_path=db_path) is not None

    def test_refine_drops_and_deep_are_independent_dimensions(self, db_path, track):
        fp = fingerprint_track_analysis(track)
        plain = cue_proposal_key(agentic=False, provider=None, model=None, skip_critic=False,
                                  refine_drops=False, deep=False, offset_bars=16, loop_bars=4)
        refined_only = cue_proposal_key(agentic=False, provider=None, model=None, skip_critic=False,
                                         refine_drops=True, deep=False, offset_bars=16, loop_bars=4)
        refined_deep = cue_proposal_key(agentic=False, provider=None, model=None, skip_critic=False,
                                         refine_drops=True, deep=True, offset_bars=16, loop_bars=4)
        store_result(track.id, refined_only, fp, {"positions": {}, "confidence": {}, "notes": []},
                     db_path=db_path)

        assert get_cached(track.id, plain, fp, db_path=db_path) is None
        assert get_cached(track.id, refined_deep, fp, db_path=db_path) is None
        assert get_cached(track.id, refined_only, fp, db_path=db_path) is not None

    def test_beatgrid_key_distinct_from_cue_proposal_key_for_same_track(self, db_path, track):
        entries = [RawBeatGridEntry(beat_in_bar=1, bpm=128.0, time_ms=77.0)]
        cue_fp = fingerprint_track_analysis(track)
        grid_fp = fingerprint_beat_grid(entries)

        cue_key = cue_proposal_key(agentic=False, provider=None, model=None, skip_critic=False,
                                    refine_drops=False, deep=False, offset_bars=0, loop_bars=0)
        grid_key = beatgrid_key(deep=False, tolerance_ms=30.0)
        store_result(track.id, cue_key, cue_fp, {"positions": {}, "confidence": {}, "notes": []},
                     db_path=db_path)

        assert get_cached(track.id, grid_key, grid_fp, db_path=db_path) is None
        assert get_cached(track.id, cue_key, cue_fp, db_path=db_path) is not None

    def test_beatgrid_different_tolerance_ms_are_different_entries(self, db_path, track):
        entries = [RawBeatGridEntry(beat_in_bar=1, bpm=128.0, time_ms=77.0)]
        fp = fingerprint_beat_grid(entries)
        tol_30 = beatgrid_key(deep=False, tolerance_ms=30.0)
        tol_50 = beatgrid_key(deep=False, tolerance_ms=50.0)
        store_result(track.id, tol_30, fp, {"status": "ok"}, db_path=db_path)

        assert get_cached(track.id, tol_50, fp, db_path=db_path) is None
        assert get_cached(track.id, tol_30, fp, db_path=db_path) is not None


# ---------------------------------------------------------------------------
# summary() / clear_all()
# ---------------------------------------------------------------------------


class TestSummaryAndClear:
    def test_summary_empty_db_returns_empty_list(self, db_path):
        assert summary(db_path=db_path) == []

    def test_summary_nonexistent_db_returns_empty_list(self, tmp_path):
        assert summary(db_path=tmp_path / "never_created.db") == []

    def test_summary_groups_by_kind_and_engine(self, db_path, track):
        fp = fingerprint_track_analysis(track)
        heuristic = cue_proposal_key(agentic=False, provider=None, model=None, skip_critic=False,
                                      refine_drops=False, deep=False, offset_bars=16, loop_bars=4)
        agentic = cue_proposal_key(agentic=True, provider="gemini", model="gemini-flash-lite-latest",
                                    skip_critic=False, refine_drops=False, deep=False,
                                    offset_bars=16, loop_bars=4)
        store_result(track.id, heuristic, fp, {"positions": {}, "confidence": {}, "notes": []},
                     db_path=db_path)
        store_result(track.id, agentic, fp, {"positions": {}, "confidence": {}, "notes": []},
                     cost_usd=0.002, db_path=db_path)

        rows = summary(db_path=db_path)

        assert len(rows) == 2
        by_engine = {r["engine"]: r for r in rows}
        assert by_engine["heuristic"]["total"] == 1
        assert by_engine["agentic"]["total"] == 1
        assert by_engine["agentic"]["total_cost_usd"] == pytest.approx(0.002)
        assert by_engine["heuristic"]["total_cost_usd"] == 0.0

    def test_clear_all_removes_rows_and_returns_count(self, db_path, heuristic_key, track):
        fp = fingerprint_track_analysis(track)
        store_result(track.id, heuristic_key, fp, {"positions": {}, "confidence": {}, "notes": []},
                     db_path=db_path)
        store_result(2, heuristic_key, fp, {"positions": {}, "confidence": {}, "notes": []},
                     db_path=db_path)

        removed = clear_all(db_path=db_path)

        assert removed == 2
        assert summary(db_path=db_path) == []

    def test_clear_all_nonexistent_db_returns_zero(self, tmp_path):
        assert clear_all(db_path=tmp_path / "never_created.db") == 0


class TestEstimate:
    """The dashboard's preset picker reads this for a real "here's what
    this has actually cost/taken so far" instead of a hardcoded guess --
    see server.py's _handle_estimate_get."""

    def test_no_data_yet_is_honest_zero_not_a_fabricated_number(self, tmp_path):
        result = estimate("cue_proposal", "heuristic", refine_drops=True, deep=True,
                           db_path=tmp_path / "never_created.db")
        assert result == {"sample_count": 0, "avg_compute_seconds": None, "avg_cost_usd": None}

    def test_averages_across_matching_entries(self, db_path, heuristic_key, track):
        fp = fingerprint_track_analysis(track)
        payload = {"positions": {}, "confidence": {}, "notes": []}
        store_result(1, heuristic_key, fp, payload, compute_seconds=100.0, cost_usd=0.01, db_path=db_path)
        store_result(2, heuristic_key, fp, payload, compute_seconds=200.0, cost_usd=0.03, db_path=db_path)

        result = estimate("cue_proposal", "heuristic", refine_drops=True, deep=True, db_path=db_path)

        assert result["sample_count"] == 2
        assert result["avg_compute_seconds"] == pytest.approx(150.0)
        assert result["avg_cost_usd"] == pytest.approx(0.02)

    def test_different_refine_drops_or_deep_combination_not_averaged_together(
        self, db_path, track
    ):
        fp = fingerprint_track_analysis(track)
        payload = {"positions": {}, "confidence": {}, "notes": []}
        deep_key = cue_proposal_key(agentic=False, provider=None, model=None, skip_critic=False,
                                     refine_drops=True, deep=True, offset_bars=16, loop_bars=4)
        quick_key = cue_proposal_key(agentic=False, provider=None, model=None, skip_critic=False,
                                      refine_drops=False, deep=False, offset_bars=16, loop_bars=4)
        store_result(1, deep_key, fp, payload, compute_seconds=200.0, db_path=db_path)
        store_result(2, quick_key, fp, payload, compute_seconds=1.0, db_path=db_path)

        deep_est = estimate("cue_proposal", "heuristic", refine_drops=True, deep=True, db_path=db_path)
        quick_est = estimate("cue_proposal", "heuristic", refine_drops=False, deep=False, db_path=db_path)

        assert deep_est["sample_count"] == 1 and deep_est["avg_compute_seconds"] == pytest.approx(200.0)
        assert quick_est["sample_count"] == 1 and quick_est["avg_compute_seconds"] == pytest.approx(1.0)

    def test_entries_missing_compute_seconds_are_excluded_not_averaged_as_zero(
        self, db_path, heuristic_key, track
    ):
        """A real gap this caught: fill_cache.py originally never passed
        compute_seconds, and _run_analysis_job's beatgrid branch didn't
        either -- both fixed, but the estimate query itself must also
        never silently treat a missing value as 0s, which would make an
        untimed batch look deceptively fast instead of just not counting."""
        fp = fingerprint_track_analysis(track)
        payload = {"positions": {}, "confidence": {}, "notes": []}
        store_result(1, heuristic_key, fp, payload, db_path=db_path)  # no compute_seconds
        store_result(2, heuristic_key, fp, payload, compute_seconds=50.0, db_path=db_path)

        result = estimate("cue_proposal", "heuristic", refine_drops=True, deep=True, db_path=db_path)

        assert result["sample_count"] == 1
        assert result["avg_compute_seconds"] == pytest.approx(50.0)

    def test_beatgrid_kind_uses_engine_n_a(self, db_path, track):
        key = beatgrid_key(deep=True, tolerance_ms=30.0)
        fp = fingerprint_beat_grid(None)
        store_result(1, key, fp, {"status": "ok", "self_consistency": {}, "audio": None},
                     compute_seconds=5.0, db_path=db_path)

        result = estimate("beatgrid", "n/a", refine_drops=False, deep=True, db_path=db_path)

        assert result["sample_count"] == 1
        assert result["avg_compute_seconds"] == pytest.approx(5.0)

    def test_device_omitted_averages_across_every_device(self, db_path, heuristic_key, track):
        """The default (no device= given) -- matches most callers, who
        just want "how long does this take" regardless of hardware."""
        fp = fingerprint_track_analysis(track)
        payload = {"positions": {}, "confidence": {}, "notes": []}
        store_result(1, heuristic_key, fp, payload, compute_seconds=100.0, device="cpu", db_path=db_path)
        store_result(2, heuristic_key, fp, payload, compute_seconds=20.0, device="cuda", db_path=db_path)

        result = estimate("cue_proposal", "heuristic", refine_drops=True, deep=True, db_path=db_path)

        assert result["sample_count"] == 2
        assert result["avg_compute_seconds"] == pytest.approx(60.0)

    def test_device_given_narrows_to_just_that_device(self, db_path, heuristic_key, track):
        """The real "cpu: ~150s avg" vs "cuda: ~20s avg" comparison this
        was actually built for -- each device's own average kept
        separate when explicitly asked for."""
        fp = fingerprint_track_analysis(track)
        payload = {"positions": {}, "confidence": {}, "notes": []}
        store_result(1, heuristic_key, fp, payload, compute_seconds=100.0, device="cpu", db_path=db_path)
        store_result(2, heuristic_key, fp, payload, compute_seconds=200.0, device="cpu", db_path=db_path)
        store_result(3, heuristic_key, fp, payload, compute_seconds=20.0, device="cuda", db_path=db_path)

        cpu_est = estimate("cue_proposal", "heuristic", refine_drops=True, deep=True, device="cpu", db_path=db_path)
        cuda_est = estimate("cue_proposal", "heuristic", refine_drops=True, deep=True, device="cuda", db_path=db_path)

        assert cpu_est["sample_count"] == 2
        assert cpu_est["avg_compute_seconds"] == pytest.approx(150.0)
        assert cuda_est["sample_count"] == 1
        assert cuda_est["avg_compute_seconds"] == pytest.approx(20.0)

    def test_device_defaults_to_cpu_when_not_passed_to_store_result(self, db_path, heuristic_key, track):
        """store_result()'s own device="cpu" default -- a caller that
        predates this feature (or just doesn't care) still gets a row
        estimate(device="cpu") can find, not a NULL that falls through
        every device-specific filter."""
        fp = fingerprint_track_analysis(track)
        store_result(1, heuristic_key, fp, {"positions": {}, "confidence": {}, "notes": []},
                     compute_seconds=42.0, db_path=db_path)  # no device= passed

        result = estimate("cue_proposal", "heuristic", refine_drops=True, deep=True, device="cpu", db_path=db_path)

        assert result["sample_count"] == 1
        assert result["avg_compute_seconds"] == pytest.approx(42.0)


class TestDeviceColumnMigration:
    """The additive ALTER TABLE in _connect() -- for a database created
    before "device" existed. A brand-new db already gets the column
    straight from CREATE TABLE (see TestEstimate's own tests, all
    passing already); this specifically simulates the pre-existing-db
    case the migration exists for."""

    def test_pre_existing_db_without_device_column_gets_it_added(self, tmp_path):
        import sqlite3

        db_path = tmp_path / "legacy.db"
        # Build the table as it looked before "device" existed -- no
        # _SCHEMA/_connect() involved yet, so this genuinely has no
        # device column, not just a fresh db that happens to already
        # match the current schema.
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE analysis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER NOT NULL,
                track_title TEXT NOT NULL DEFAULT '',
                track_artist TEXT NOT NULL DEFAULT '',
                analysis_kind TEXT NOT NULL,
                engine TEXT NOT NULL DEFAULT 'n/a',
                provider TEXT NOT NULL DEFAULT 'n/a',
                model TEXT NOT NULL DEFAULT 'n/a',
                skip_critic INTEGER NOT NULL DEFAULT 0,
                refine_drops INTEGER NOT NULL DEFAULT 0,
                deep INTEGER NOT NULL DEFAULT 0,
                offset_bars INTEGER NOT NULL DEFAULT 0,
                loop_bars INTEGER NOT NULL DEFAULT 0,
                tolerance_ms REAL NOT NULL DEFAULT 0.0,
                input_fingerprint TEXT NOT NULL,
                result_json TEXT NOT NULL,
                cost_usd REAL,
                compute_seconds REAL,
                source TEXT NOT NULL DEFAULT 'cli',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(track_id, analysis_kind, engine, provider, model, skip_critic,
                       refine_drops, deep, offset_bars, loop_bars, tolerance_ms)
            );
        """)
        conn.execute(
            "INSERT INTO analysis_runs (track_id, analysis_kind, input_fingerprint, result_json, created_at, updated_at) "
            "VALUES (1, 'cue_proposal', 'fp', '{}', 'x', 'x')"
        )
        conn.commit()
        conn.close()

        # Any real entry point that calls _connect() should transparently
        # migrate this on first touch -- store_result is as good as any.
        heuristic_key = cue_proposal_key(agentic=False, provider=None, model=None, skip_critic=False,
                                          refine_drops=True, deep=True, offset_bars=16, loop_bars=4)
        store_result(2, heuristic_key, "fp2", {"positions": {}, "confidence": {}, "notes": []},
                     compute_seconds=10.0, device="cuda", db_path=db_path)

        # The pre-existing row (inserted with no device= at all, before
        # the column existed) backfilled to the column default, "cpu" --
        # confirming ALTER TABLE ... DEFAULT applies retroactively, not
        # just to rows inserted after the migration.
        conn = sqlite3.connect(db_path)
        pre_existing_device = conn.execute(
            "SELECT device FROM analysis_runs WHERE track_id = 1"
        ).fetchone()[0]
        conn.close()
        assert pre_existing_device == "cpu"

        # And the new row's real device is queryable normally.
        result = estimate("cue_proposal", "heuristic", refine_drops=True, deep=True, device="cuda", db_path=db_path)
        assert result["sample_count"] == 1
        assert result["avg_compute_seconds"] == pytest.approx(10.0)
