"""Correction-history logging.

Captures (algorithm-proposed position -> human-corrected/accepted/skipped
outcome) for every hot cue in every applied review session, so a future
calibration model has real usage data to learn from instead of the session
JSON's signal evaporating the moment ``apply`` finishes.

Stored locally at ``~/.djcues/history.db`` using plain ``sqlite3`` — this
table is deliberately not routed through pyrekordbox's SQLAlchemy engine,
keeping its lifecycle independent of the Rekordbox database connection.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from djcues.constants import CUE_SYSTEM_BY_PAD

_SCHEMA = """
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    applied_at TEXT NOT NULL,
    session_path TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    track_title TEXT,
    track_artist TEXT,
    pad TEXT NOT NULL,
    cue_kind INTEGER NOT NULL,
    bpm REAL,
    duration_ms REAL,
    phrase_count INTEGER,
    has_vocal_data INTEGER,
    has_waveform_data INTEGER,
    proposed_position_ms REAL,
    proposed_confidence REAL,
    final_position_ms REAL,
    final_status TEXT NOT NULL,
    UNIQUE(session_path, track_id, pad)
);
CREATE INDEX IF NOT EXISTS idx_corrections_pad ON corrections(pad);
"""


def default_db_path() -> Path:
    """Return ``~/.djcues/history.db``, creating the parent directory if needed."""
    db_dir = Path.home() / ".djcues"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "history.db"


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    return conn


def _resolve_cue_outcome(cue_entry: dict) -> tuple[str, float | None, float | None]:
    """Resolve a cue's logged (status, final_position_ms, proposed_position_ms).

    Cue-level "pending" within an applied track counts as an implicit
    accept: it still ships to Rekordbox even though no Accept button was
    ever explicitly clicked for that individual cue (writer.py's
    build_cue_rows only excludes status == "skipped", not "pending").
    """
    status = cue_entry.get("status", "pending")
    proposed_position_ms = cue_entry.get("original_ms", cue_entry.get("position_ms"))
    if status == "skipped":
        return "skipped", None, proposed_position_ms
    if status == "adjusted":
        return "adjusted", cue_entry.get("position_ms"), proposed_position_ms
    # "accepted" or "pending" (implicit accept)
    return "accepted", cue_entry.get("position_ms"), proposed_position_ms


def log_session_corrections(
    session: dict, session_path: str, db_path: Path | None = None
) -> int:
    """Log one row per hot-cue pad for every accepted/adjusted/skipped track
    in a review session.

    Tracks left at track-level "pending" (never touched at all) are
    skipped — nothing was actually applied for them. Idempotent: logging
    the same ``session_path`` again upserts existing rows rather than
    duplicating them, so re-running ``apply`` on the same session is safe.

    Returns the number of rows written.
    """
    db_path = Path(db_path) if db_path else default_db_path()
    conn = _connect(db_path)
    applied_at = datetime.now().replace(microsecond=0).isoformat()
    written = 0
    try:
        for track_id, track_data in session.get("tracks", {}).items():
            if track_data.get("status") not in ("accepted", "adjusted", "skipped"):
                continue  # never touched -- nothing was applied

            for pad, cue_entry in track_data.get("cues", {}).items():
                slot = CUE_SYSTEM_BY_PAD.get(pad)
                if slot is None:
                    continue
                final_status, final_position_ms, proposed_position_ms = (
                    _resolve_cue_outcome(cue_entry)
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO corrections (
                        applied_at, session_path, track_id, track_title, track_artist,
                        pad, cue_kind, bpm, duration_ms, phrase_count,
                        has_vocal_data, has_waveform_data,
                        proposed_position_ms, proposed_confidence,
                        final_position_ms, final_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        applied_at,
                        session_path,
                        int(track_id),
                        track_data.get("title"),
                        track_data.get("artist"),
                        pad,
                        slot.kind,
                        track_data.get("bpm"),
                        track_data.get("duration_ms"),
                        track_data.get("phrase_count"),
                        int(bool(track_data.get("has_vocal_data"))),
                        int(bool(track_data.get("has_waveform_data"))),
                        proposed_position_ms,
                        cue_entry.get("confidence"),
                        final_position_ms,
                        final_status,
                    ),
                )
                written += 1
        conn.commit()
    finally:
        conn.close()
    return written


def summary(db_path: Path | None = None) -> list[dict]:
    """Per-pad row counts, for the ``djcues history`` CLI command."""
    db_path = Path(db_path) if db_path else default_db_path()
    if not db_path.exists():
        return []
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT pad,
                   COUNT(*) AS total,
                   SUM(CASE WHEN final_status IN ('adjusted', 'skipped')
                            THEN 1 ELSE 0 END) AS corrected,
                   MIN(applied_at) AS first_seen,
                   MAX(applied_at) AS last_seen
            FROM corrections
            GROUP BY pad
            ORDER BY pad
            """
        ).fetchall()
        cols = ["pad", "total", "corrected", "first_seen", "last_seen"]
        return [dict(zip(cols, row)) for row in rows]
    finally:
        conn.close()
