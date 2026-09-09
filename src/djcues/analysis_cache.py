"""Persistent cache of completed analysis runs (propose/compare/review/
beatgrid) so re-analyzing a track that hasn't changed doesn't re-pay real
compute or API cost -- most notably --refine-drops --deep's ~3-12 min/track
Demucs pass and --agentic's real per-call LLM spend.

Stored locally at ~/.djcues/analysis_cache.db using plain sqlite3,
mirroring history.py's storage convention exactly (deliberately outside
pyrekordbox's own SQLAlchemy engine, and -- unlike this repo's own
OneDrive-synced checkout -- outside any synced directory).

A cache entry is keyed on (track_id, analysis_kind, engine, provider,
model, skip_critic, refine_drops, deep, offset_bars, loop_bars,
tolerance_ms) -- every dimension that can change what an analysis run
actually computes -- plus an input_fingerprint of the track's own
analysis-relevant Rekordbox data (bpm, beat grid, phrases, vocal track,
waveform, or the raw per-beat grid for beatgrid) so a cache hit is only
ever served when that data hasn't changed since the cached result was
computed. This module is a plain, honest store: I/O errors propagate
normally here: it's the caller's job (cli.py's _apply_cache) to decide a
cache failure shouldn't abort an otherwise read-only command.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from djcues.models import RawBeatGridEntry, Track

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_runs (
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
CREATE INDEX IF NOT EXISTS idx_analysis_runs_track ON analysis_runs(track_id);
"""


def default_db_path() -> Path:
    """Return ~/.djcues/analysis_cache.db, creating the parent directory if needed."""
    db_dir = Path.home() / ".djcues"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "analysis_cache.db"


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    return conn


@dataclass(frozen=True)
class AnalysisKey:
    """Identifies *which* analysis was run and with what parameters --
    every dimension that can change the actual computed result. Two
    tracks analyzed under different keys are never the same cached
    analysis, even for the same track_id. Built via cue_proposal_key()/
    beatgrid_key(), not directly -- those fill in the right "n/a"/0
    sentinels for whichever dimensions don't apply to that kind."""

    kind: str  # "cue_proposal" | "beatgrid"
    engine: str = "n/a"
    provider: str = "n/a"
    model: str = "n/a"
    skip_critic: bool = False
    refine_drops: bool = False
    deep: bool = False
    offset_bars: int = 0
    loop_bars: int = 0
    tolerance_ms: float = 0.0


def cue_proposal_key(
    *,
    agentic: bool,
    provider: str | None,
    model: str | None,
    skip_critic: bool,
    refine_drops: bool,
    deep: bool,
    offset_bars: int,
    loop_bars: int,
) -> AnalysisKey:
    """Key for a propose/compare/review-style cue proposal. provider/
    model must be the *resolved* values (cli.py's _get_proposer returns
    the resolved provider alongside the proposer for exactly this) --
    never the raw --provider/--model CLI args, which default to None
    and get resolved from saved config; keying on the unresolved None
    would wrongly collide across different configured defaults."""
    return AnalysisKey(
        kind="cue_proposal",
        engine="agentic" if agentic else "heuristic",
        provider=(provider or "n/a") if agentic else "n/a",
        model=(model or "n/a") if agentic else "n/a",
        skip_critic=bool(skip_critic) if agentic else False,
        refine_drops=bool(refine_drops),
        deep=bool(deep),
        offset_bars=offset_bars,
        loop_bars=loop_bars,
    )


def beatgrid_key(*, deep: bool, tolerance_ms: float) -> AnalysisKey:
    return AnalysisKey(kind="beatgrid", deep=bool(deep), tolerance_ms=tolerance_ms)


def _hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def fingerprint_track_analysis(track: Track) -> str:
    """Hashes exactly the Track fields the heuristic/agentic engines
    read as analysis input: bpm, beat_grid.first_beat_ms, phrases,
    vocal_track, waveform (confirmed via strategy.py --
    compute_phrase_energy/_spectral_similarity both read
    track.waveform -- and agentic.py). Deliberately excludes
    title/artist (retagging a file shouldn't invalidate a cached
    analysis) and existing cues (track.cues is never read as an input
    by either engine, only used downstream for accuracy comparison
    against a proposal already computed without it) -- so a hand-edited
    cue in Rekordbox doesn't wrongly bust the cache either.
    """
    payload = {
        "bpm": track.bpm,
        "first_beat_ms": track.beat_grid.first_beat_ms,
        "phrases": [(p.beat_start, p.beat_end, p.kind, p.label) for p in track.phrases],
        "vocal_track": track.vocal_track,
        "waveform": (
            [(w.height, w.red, w.green, w.blue) for w in track.waveform]
            if track.waveform
            else None
        ),
    }
    return _hash(payload)


def fingerprint_beat_grid(entries: list[RawBeatGridEntry] | None) -> str:
    """Hashes rekordbox's raw per-beat PQTZ array -- beatgrid
    verification's real input, independent of phrase/vocal/waveform
    data. None entries (no grid data) hash to a stable sentinel so
    repeated no-grid-data results still correctly hit each other."""
    if entries is None:
        return _hash("no_grid_data")
    return _hash([(e.beat_in_bar, e.bpm, e.time_ms) for e in entries])


@dataclass(frozen=True)
class CachedResult:
    result: dict
    cost_usd: float | None
    compute_seconds: float | None
    source: str
    created_at: str
    updated_at: str


def get_cached(
    track_id: int,
    key: AnalysisKey,
    input_fingerprint: str,
    db_path: Path | None = None,
) -> CachedResult | None:
    """Return the cached result for (track_id, key) if one exists AND
    its stored input_fingerprint still matches -- a mismatch means
    rekordbox re-analyzed this track since the cached result was
    computed, so it's treated exactly like a miss (never returned)."""
    db_path = Path(db_path) if db_path else default_db_path()
    conn = _connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT input_fingerprint, result_json, cost_usd, compute_seconds,
                   source, created_at, updated_at
            FROM analysis_runs
            WHERE track_id = ? AND analysis_kind = ? AND engine = ? AND provider = ?
              AND model = ? AND skip_critic = ? AND refine_drops = ? AND deep = ?
              AND offset_bars = ? AND loop_bars = ? AND tolerance_ms = ?
            """,
            (
                track_id, key.kind, key.engine, key.provider, key.model,
                int(key.skip_critic), int(key.refine_drops), int(key.deep),
                key.offset_bars, key.loop_bars, key.tolerance_ms,
            ),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None
    stored_fingerprint, result_json, cost_usd, compute_seconds, source, created_at, updated_at = row
    if stored_fingerprint != input_fingerprint:
        return None
    return CachedResult(
        result=json.loads(result_json),
        cost_usd=cost_usd,
        compute_seconds=compute_seconds,
        source=source,
        created_at=created_at,
        updated_at=updated_at,
    )


def store_result(
    track_id: int,
    key: AnalysisKey,
    input_fingerprint: str,
    result: dict,
    *,
    track_title: str = "",
    track_artist: str = "",
    cost_usd: float | None = None,
    compute_seconds: float | None = None,
    source: str = "cli",
    db_path: Path | None = None,
) -> None:
    """Upsert one cache entry. Uses ON CONFLICT ... DO UPDATE rather
    than history.py's INSERT OR REPLACE, which deletes+reinserts and
    would reset created_at on every refresh instead of preserving it."""
    db_path = Path(db_path) if db_path else default_db_path()
    conn = _connect(db_path)
    now = datetime.now().replace(microsecond=0).isoformat()
    try:
        conn.execute(
            """
            INSERT INTO analysis_runs (
                track_id, track_title, track_artist, analysis_kind, engine,
                provider, model, skip_critic, refine_drops, deep,
                offset_bars, loop_bars, tolerance_ms, input_fingerprint,
                result_json, cost_usd, compute_seconds, source,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(track_id, analysis_kind, engine, provider, model,
                        skip_critic, refine_drops, deep, offset_bars,
                        loop_bars, tolerance_ms)
            DO UPDATE SET
                track_title = excluded.track_title,
                track_artist = excluded.track_artist,
                input_fingerprint = excluded.input_fingerprint,
                result_json = excluded.result_json,
                cost_usd = excluded.cost_usd,
                compute_seconds = excluded.compute_seconds,
                source = excluded.source,
                updated_at = excluded.updated_at
            """,
            (
                track_id, track_title, track_artist, key.kind, key.engine,
                key.provider, key.model, int(key.skip_critic), int(key.refine_drops),
                int(key.deep), key.offset_bars, key.loop_bars, key.tolerance_ms,
                input_fingerprint, json.dumps(result), cost_usd, compute_seconds,
                source, now, now,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def summary(db_path: Path | None = None) -> list[dict]:
    """Per-(kind, engine) row counts and total spend, for `djcues cache status`."""
    db_path = Path(db_path) if db_path else default_db_path()
    if not db_path.exists():
        return []
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT analysis_kind, engine,
                   COUNT(*) AS total,
                   COALESCE(SUM(cost_usd), 0.0) AS total_cost_usd,
                   MIN(created_at) AS first_seen,
                   MAX(updated_at) AS last_seen
            FROM analysis_runs
            GROUP BY analysis_kind, engine
            ORDER BY analysis_kind, engine
            """
        ).fetchall()
        cols = ["analysis_kind", "engine", "total", "total_cost_usd", "first_seen", "last_seen"]
        return [dict(zip(cols, row)) for row in rows]
    finally:
        conn.close()


def clear_all(db_path: Path | None = None) -> int:
    """Delete every cached entry. Returns the number of rows removed."""
    db_path = Path(db_path) if db_path else default_db_path()
    if not db_path.exists():
        return 0
    conn = _connect(db_path)
    try:
        cursor = conn.execute("DELETE FROM analysis_runs")
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()
