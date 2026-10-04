"""Local store for smart tagging -- the library-wide energy calibration
and a per-track cache of ANLZ-derived features.

Stored at ~/.djcues/tags.db using plain sqlite3, mirroring history.py's
and analysis_cache.py's convention exactly (deliberately outside
pyrekordbox's own SQLAlchemy engine, and outside this repo's OneDrive-
synced checkout), plus the tag ledger: what djcues last wrote to
Rekordbox for each (track, category), and every time a later apply found
the user had changed it by hand.

A plain, honest store: I/O errors propagate. Callers (the `tags` CLI
commands) decide what a cache failure means.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from djcues.tagging import AnalysisFeatures

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calibrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    n_tracks INTEGER NOT NULL,
    n_excluded INTEGER NOT NULL DEFAULT 0,
    cutpoints_json TEXT NOT NULL,
    thresholds_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS track_features (
    track_id TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    features_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tag_ledger (
    track_id TEXT NOT NULL,
    category TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    written_at TEXT NOT NULL,
    PRIMARY KEY (track_id, category)
);
CREATE TABLE IF NOT EXISTS tag_overrides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id TEXT NOT NULL,
    category TEXT NOT NULL,
    written_json TEXT NOT NULL,
    found_json TEXT NOT NULL,
    proposed_json TEXT NOT NULL,
    seen_at TEXT NOT NULL
);
"""


def default_db_path() -> Path:
    """Return ~/.djcues/tags.db, creating the parent directory if needed."""
    db_dir = Path.home() / ".djcues"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "tags.db"


@dataclass(frozen=True)
class Calibration:
    """One stored library-wide energy calibration. n_tracks is how many
    tracks fed the quintiles; n_excluded how many were left out as
    loops/samples or for lacking energy data."""

    id: int
    created_at: str
    n_tracks: int
    n_excluded: int
    cutpoints: list[float]
    thresholds: dict


class TagStore:
    """Thin wrapper over one sqlite connection. Use as a context manager
    so the connection is closed; every write method commits itself except
    put_features(commit=False), which exists so a whole-library run can
    batch into one transaction."""

    def __init__(self, db_path: Path | None = None):
        self._conn = sqlite3.connect(db_path if db_path is not None else default_db_path())
        self._conn.executescript(_SCHEMA)

    def __enter__(self) -> "TagStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def commit(self) -> None:
        self._conn.commit()

    # --- calibration -------------------------------------------------------

    def save_calibration(
        self, cutpoints: list[float], n_tracks: int, n_excluded: int, thresholds: dict
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO calibrations (created_at, n_tracks, n_excluded, cutpoints_json, thresholds_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now().replace(microsecond=0).isoformat(),
                n_tracks,
                n_excluded,
                json.dumps(cutpoints),
                json.dumps(thresholds, sort_keys=True),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def latest_calibration(self) -> Calibration | None:
        row = self._conn.execute(
            "SELECT id, created_at, n_tracks, n_excluded, cutpoints_json, thresholds_json "
            "FROM calibrations ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return Calibration(
            id=row[0], created_at=row[1], n_tracks=row[2], n_excluded=row[3],
            cutpoints=[float(c) for c in json.loads(row[4])], thresholds=json.loads(row[5]),
        )

    # --- feature cache -----------------------------------------------------

    def get_features(self, track_id: str, fingerprint: str) -> AnalysisFeatures | None:
        """Cached features, but only if stored under this exact
        fingerprint -- a changed track (re-analysed, re-gridded) misses."""
        row = self._conn.execute(
            "SELECT fingerprint, features_json FROM track_features WHERE track_id = ?",
            (str(track_id),),
        ).fetchone()
        if row is None or row[0] != fingerprint:
            return None
        return AnalysisFeatures.from_dict(json.loads(row[1]))

    def put_features(
        self, features: AnalysisFeatures, fingerprint: str, *, commit: bool = True
    ) -> None:
        self._conn.execute(
            "INSERT INTO track_features (track_id, fingerprint, features_json, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(track_id) DO UPDATE SET fingerprint = excluded.fingerprint, "
            "features_json = excluded.features_json, updated_at = excluded.updated_at",
            (
                str(features.track_id),
                fingerprint,
                json.dumps(features.to_dict()),
                datetime.now().replace(microsecond=0).isoformat(),
            ),
        )
        if commit:
            self._conn.commit()

    def count_features(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM track_features").fetchone()[0])

    def clear_features(self) -> int:
        n = self.count_features()
        self._conn.execute("DELETE FROM track_features")
        self._conn.commit()
        return n

    # --- tag ledger --------------------------------------------------------
    #
    # One row per (track, category): the exact set of tag names djcues
    # last wrote there. An empty set is a real, meaningful entry ("djcues
    # wrote: nothing"), distinct from no row at all ("djcues has never
    # touched this"). See tag_apply.decide_category() for how it's used.

    def last_written(self, track_id: str, category: str) -> set[str] | None:
        row = self._conn.execute(
            "SELECT tags_json FROM tag_ledger WHERE track_id = ? AND category = ?",
            (str(track_id), category),
        ).fetchone()
        return None if row is None else set(json.loads(row[0]))

    def record_written(self, track_id: str, category: str, tags: set[str], *, commit: bool = True) -> None:
        self._conn.execute(
            "INSERT INTO tag_ledger (track_id, category, tags_json, written_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(track_id, category) DO UPDATE SET tags_json = excluded.tags_json, "
            "written_at = excluded.written_at",
            (str(track_id), category, json.dumps(sorted(tags)), _now()),
        )
        if commit:
            self._conn.commit()

    def forget(self, track_id: str, category: str, *, commit: bool = True) -> None:
        self._conn.execute(
            "DELETE FROM tag_ledger WHERE track_id = ? AND category = ?", (str(track_id), category)
        )
        if commit:
            self._conn.commit()

    def ledger_entries(self) -> list[tuple[str, str, set[str]]]:
        """Every (track_id, category, tags) djcues has written."""
        rows = self._conn.execute(
            "SELECT track_id, category, tags_json FROM tag_ledger ORDER BY track_id, category"
        ).fetchall()
        return [(r[0], r[1], set(json.loads(r[2]))) for r in rows]

    def record_override(
        self, track_id: str, category: str, *, written: set[str], found: set[str],
        proposed: set[str], commit: bool = True,
    ) -> None:
        """The user changed what djcues wrote: djcues wrote `written`,
        Rekordbox now has `found`, and djcues wanted `proposed` -- kept
        as the feedback signal for tuning thresholds later."""
        self._conn.execute(
            "INSERT INTO tag_overrides (track_id, category, written_json, found_json, proposed_json, seen_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (str(track_id), category, json.dumps(sorted(written)), json.dumps(sorted(found)),
             json.dumps(sorted(proposed)), _now()),
        )
        if commit:
            self._conn.commit()

    def overrides(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT track_id, category, written_json, found_json, proposed_json, seen_at "
            "FROM tag_overrides ORDER BY id"
        ).fetchall()
        return [
            {"track_id": r[0], "category": r[1], "written": json.loads(r[2]), "found": json.loads(r[3]),
             "proposed": json.loads(r[4]), "seen_at": r[5]}
            for r in rows
        ]


def _now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()
