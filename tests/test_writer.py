"""Tests for djcues.writer — backup, cue row building, DB write."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import click
import pytest

from djcues.constants import CUE_SYSTEM, CUE_SYSTEM_BY_PAD


# ---------------------------------------------------------------------------
# Test 1: backup_database creates a timestamped copy
# ---------------------------------------------------------------------------


def test_backup_creates_timestamped_copy(tmp_path):
    fake_db = tmp_path / "master.db"
    fake_db.write_bytes(b"sqlite-fake-content")

    from djcues.writer import backup_database

    backup_path = backup_database(fake_db)

    assert backup_path.exists()
    assert backup_path.parent == tmp_path
    assert backup_path.name.startswith("master-backup-")
    assert backup_path.name.endswith(".db")
    assert backup_path.read_bytes() == b"sqlite-fake-content"


# ---------------------------------------------------------------------------
# Helpers for build_cue_rows tests
# ---------------------------------------------------------------------------


def _make_hot_cues(statuses: dict[str, str] | None = None) -> dict:
    """Build a hot cues dict keyed by pad letter with position data.

    *statuses* overrides the default 'accepted' status per pad.
    """
    if statuses is None:
        statuses = {}
    cues: dict[str, dict] = {}
    for pad in "ABCDEFGH":
        slot = CUE_SYSTEM_BY_PAD[pad]
        entry: dict = {
            "status": statuses.get(pad, "accepted"),
            "position_ms": 1000.0 * (ord(pad) - ord("A") + 1),
        }
        if slot.is_loop:
            entry["loop_end_ms"] = entry["position_ms"] + 7500.0
        else:
            entry["loop_end_ms"] = None
        cues[pad] = entry
    return cues


def _make_memory_cues(statuses: dict[str, str] | None = None) -> dict:
    """Build a memory cues dict keyed by slot number string '1'-'8'."""
    if statuses is None:
        statuses = {}
    cues: dict[str, dict] = {}
    for i in range(8):
        slot_num = str(i + 1)
        slot = CUE_SYSTEM[i]
        entry: dict = {
            "status": statuses.get(slot_num, "accepted"),
            "position_ms": 500.0 * (i + 1),
        }
        if slot.is_loop:
            entry["loop_end_ms"] = entry["position_ms"] + 7500.0
        else:
            entry["loop_end_ms"] = None
        cues[slot_num] = entry
    return cues


# ---------------------------------------------------------------------------
# Test 2: build_cue_rows creates hot and memory rows
# ---------------------------------------------------------------------------


def test_build_cue_rows_creates_hot_and_memory():
    from djcues.writer import build_cue_rows

    hot_cues = _make_hot_cues()
    memory_cues = _make_memory_cues()

    hot_rows, mem_rows = build_cue_rows(hot_cues, memory_cues)

    assert len(hot_rows) == 8
    assert len(mem_rows) == 8

    # Verify hot cue fields
    for row in hot_rows:
        assert "Kind" in row
        assert "InMsec" in row
        assert "OutMsec" in row
        assert "ColorTableIndex" in row
        assert "Comment" in row
        assert row["InFrame"] == 0
        assert row["InMpegFrame"] == 0
        assert row["InMpegAbs"] == 0
        assert row["BeatLoopSize"] == 0
        assert row["CueMicrosec"] == 0

    # Check specific hot cue: pad A (kind=1, not a loop)
    hot_a = next(r for r in hot_rows if r["Kind"] == 1)
    assert hot_a["InMsec"] == 1000
    assert hot_a["OutMsec"] == -1
    assert hot_a["ColorTableIndex"] == 18
    assert hot_a["Color"] == -1
    assert hot_a["Comment"] == "First Beat"
    assert hot_a["ActiveLoop"] == -1
    assert hot_a["OutFrame"] == -1
    assert hot_a["OutMpegFrame"] == -1
    assert hot_a["OutMpegAbs"] == -1

    # Check loop cue: pad B (kind=2, is_loop)
    hot_b = next(r for r in hot_rows if r["Kind"] == 2)
    assert hot_b["InMsec"] == 2000
    assert hot_b["OutMsec"] == int(2000.0 + 7500.0)  # 9500
    assert hot_b["ActiveLoop"] == 0
    assert hot_b["OutFrame"] == 0
    assert hot_b["OutMpegFrame"] == 0
    assert hot_b["OutMpegAbs"] == 0

    # All memory cues should have Kind=0
    for row in mem_rows:
        assert row["Kind"] == 0

    # Check memory cue for slot 1 (pad A)
    mem_1 = mem_rows[0]  # first in iteration order
    assert mem_1["Comment"] == "First Beat"
    assert mem_1["Color"] == 4
    assert mem_1["InMsec"] == 500


# ---------------------------------------------------------------------------
# Test 3: build_cue_rows skips skipped cues
# ---------------------------------------------------------------------------


def test_build_cue_rows_skips_skipped_cues():
    from djcues.writer import build_cue_rows

    hot_cues = _make_hot_cues(statuses={"B": "skipped", "H": "skipped"})
    memory_cues = _make_memory_cues(statuses={"2": "skipped", "8": "skipped"})

    hot_rows, mem_rows = build_cue_rows(hot_cues, memory_cues)

    assert len(hot_rows) == 6
    assert len(mem_rows) == 6

    # Kind 2 (B) and Kind 9 (H) should not be present
    hot_kinds = {r["Kind"] for r in hot_rows}
    assert 2 not in hot_kinds
    assert 9 not in hot_kinds


# ---------------------------------------------------------------------------
# Test 4: build_cue_rows handles auto status
# ---------------------------------------------------------------------------


def test_build_cue_rows_handles_auto_status():
    from djcues.writer import build_cue_rows

    hot_cues = _make_hot_cues()
    memory_cues = _make_memory_cues(statuses={"3": "auto"})

    hot_rows, mem_rows = build_cue_rows(hot_cues, memory_cues)

    # "auto" should be treated as accepted — all 8 memory rows present
    assert len(mem_rows) == 8


# ---------------------------------------------------------------------------
# write_cues_for_track -- never touches a real DB: `db` is always a
# MagicMock here, standing in for a Rekordbox6Database connection.
# DjmdCue.create() itself is real and safe to call directly -- confirmed
# live (reading pyrekordbox's own source) that it's pure in-memory
# SQLAlchemy model construction with no DB I/O, so the cue objects
# write_cues_for_track builds are real, just never persisted anywhere.
# ---------------------------------------------------------------------------


def _fake_content(content_id: int = 1) -> MagicMock:
    content = MagicMock()
    content.ID = content_id
    content.UUID = f"content-uuid-{content_id}"
    return content


def test_write_cues_for_track_adds_every_row_and_returns_count():
    from djcues.writer import build_cue_rows, write_cues_for_track

    hot_rows, mem_rows = build_cue_rows(_make_hot_cues(), _make_memory_cues())
    db = MagicMock()
    content = _fake_content()

    count = write_cues_for_track(db, content, hot_rows, mem_rows, overwrite=False)

    assert count == len(hot_rows) + len(mem_rows) == 16
    assert db.add.call_count == 16
    db.get_cue.assert_not_called()
    db.delete.assert_not_called()


def test_write_cues_for_track_sets_content_and_generated_ids():
    from djcues.writer import build_cue_rows, write_cues_for_track

    hot_rows, mem_rows = build_cue_rows(_make_hot_cues(), _make_memory_cues())
    db = MagicMock()
    db.generate_unused_id.return_value = 999
    content = _fake_content(content_id=42)

    write_cues_for_track(db, content, hot_rows, mem_rows, overwrite=False)

    written_cues = [call.args[0] for call in db.add.call_args_list]
    for cue in written_cues:
        assert cue.ContentID == "42"
        assert cue.ContentUUID == "content-uuid-42"
        assert cue.ID == "999"
        assert cue.UUID  # a real uuid4 string, different per cue
    # Every generated UUID is unique, not reused across cues
    assert len({cue.UUID for cue in written_cues}) == len(written_cues)


def test_write_cues_for_track_overwrite_deletes_existing_first():
    from djcues.writer import build_cue_rows, write_cues_for_track

    hot_rows, mem_rows = build_cue_rows(_make_hot_cues(), _make_memory_cues())
    db = MagicMock()
    existing = [MagicMock(name="old_cue_1"), MagicMock(name="old_cue_2")]
    db.get_cue.return_value = existing
    content = _fake_content()

    write_cues_for_track(db, content, hot_rows, mem_rows, overwrite=True)

    db.get_cue.assert_called_once_with(ContentID=content.ID)
    assert db.delete.call_count == 2
    for cue in existing:
        db.delete.assert_any_call(cue)


def test_write_cues_for_track_no_overwrite_skips_delete_even_with_existing():
    from djcues.writer import build_cue_rows, write_cues_for_track

    hot_rows, mem_rows = build_cue_rows(_make_hot_cues(), _make_memory_cues())
    db = MagicMock()
    content = _fake_content()

    write_cues_for_track(db, content, hot_rows, mem_rows, overwrite=False)

    db.get_cue.assert_not_called()
    db.delete.assert_not_called()


# ---------------------------------------------------------------------------
# apply_session -- the highest-stakes function in the project (it's what
# actually writes to a real Rekordbox database). db is always a MagicMock;
# djcues.db.get_db is always patched so a real Rekordbox6Database is never
# constructed; djcues.history.log_session_corrections is always patched so
# no test ever writes to the real correction-history DB either -- that
# function has its own tests in test_history.py, this file only checks
# apply_session calls it correctly. backup_database is NOT mocked -- it's
# a real shutil.copy2 against a real (fake, tmp_path-only) master.db file
# created per test, for genuine end-to-end coverage of that step without
# any risk to real data.
# ---------------------------------------------------------------------------


def _session_track(
    status: str = "accepted",
    has_existing_cues: bool = False,
    cues: dict | None = None,
    memory_cues: dict | None = None,
    title: str = "Track",
) -> dict:
    if cues is None:
        cues = {"A": {"position_ms": 100.0, "loop_end_ms": None, "status": "pending", "confidence": 0.9}}
    if memory_cues is None:
        memory_cues = {"1": {"position_ms": 50.0, "loop_end_ms": None, "status": "pending"}}
    return {
        "title": title,
        "artist": "Artist",
        "bpm": 128.0,
        "duration_ms": 240_000.0,
        "status": status,
        "has_existing_cues": has_existing_cues,
        "cues": cues,
        "memory_cues": memory_cues,
    }


def _write_session_file(tmp_path, tracks: dict) -> "pathlib.Path":
    import pathlib

    session = {
        "playlist": "Test Playlist",
        "playlist_id": 1,
        "settings": {"memory_offset_bars": 16, "loop_length_bars": 4},
        "tracks": tracks,
    }
    path = tmp_path / "session.json"
    path.write_text(json.dumps(session), encoding="utf-8")
    return path


def _fake_db_with_master(tmp_path) -> MagicMock:
    (tmp_path / "master.db").write_bytes(b"fake-sqlite-content")
    db = MagicMock()
    db.db_directory = tmp_path
    db.get_cue.return_value = []  # no existing cues by default
    db.get_content.side_effect = lambda ID: _fake_content(content_id=ID)
    return db


def test_apply_session_dry_run_never_touches_the_db(tmp_path):
    from djcues.writer import apply_session

    session_path = _write_session_file(tmp_path, {"1": _session_track(status="accepted")})

    with patch("djcues.db.get_db") as mock_get_db, \
         patch("djcues.history.log_session_corrections") as mock_log:
        result = apply_session(session_path, dry_run=True)

    mock_get_db.assert_not_called()
    mock_log.assert_not_called()
    assert result == {"accepted": 1, "adjusted": 0, "skipped": 0, "written": 0, "cues_written": 0}


def test_apply_session_counts_and_summary(tmp_path):
    from djcues.writer import apply_session

    tracks = {
        "1": _session_track(status="accepted"),
        "2": _session_track(status="adjusted"),
        "3": _session_track(status="skipped"),
        "4": {**_session_track(), "status": "pending"},
    }
    session_path = _write_session_file(tmp_path, tracks)
    db = _fake_db_with_master(tmp_path)

    with patch("djcues.db.get_db", return_value=db), \
         patch("djcues.history.log_session_corrections") as mock_log:
        result = apply_session(session_path)

    assert result["accepted"] == 1
    assert result["adjusted"] == 1
    assert result["skipped"] == 1
    # Only accepted/adjusted tracks get written -- pending and skipped don't
    assert result["written"] == 2
    assert result["cues_written"] == 2 * (1 + 1)  # 1 hot + 1 memory cue each, from _session_track's defaults


def test_apply_session_writes_backup_before_any_cue_write(tmp_path):
    from djcues.writer import apply_session

    session_path = _write_session_file(tmp_path, {"1": _session_track(status="accepted")})
    db = _fake_db_with_master(tmp_path)

    with patch("djcues.db.get_db", return_value=db), \
         patch("djcues.history.log_session_corrections"):
        apply_session(session_path)

    backups = list(tmp_path.glob("master-backup-*.db"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"fake-sqlite-content"


def test_apply_session_commits_once_per_written_track(tmp_path):
    from djcues.writer import apply_session

    tracks = {"1": _session_track(status="accepted"), "2": _session_track(status="adjusted")}
    session_path = _write_session_file(tmp_path, tracks)
    db = _fake_db_with_master(tmp_path)

    with patch("djcues.db.get_db", return_value=db), \
         patch("djcues.history.log_session_corrections"):
        apply_session(session_path)

    assert db.commit.call_count == 2


def test_apply_session_logs_corrections_for_accepted_adjusted_and_skipped_not_pending(tmp_path):
    from djcues.writer import apply_session

    tracks = {
        "1": _session_track(status="accepted"),
        "2": _session_track(status="adjusted"),
        "3": _session_track(status="skipped"),
        "4": {**_session_track(), "status": "pending"},
    }
    session_path = _write_session_file(tmp_path, tracks)
    db = _fake_db_with_master(tmp_path)

    with patch("djcues.db.get_db", return_value=db), \
         patch("djcues.history.log_session_corrections") as mock_log:
        apply_session(session_path)

    logged_track_ids = set()
    for call in mock_log.call_args_list:
        session_arg = call.args[0]
        logged_track_ids.update(session_arg["tracks"].keys())
    assert logged_track_ids == {"1", "2", "3"}  # not "4" (pending)


def test_apply_session_checks_live_db_not_just_session_flag_for_existing_cues(tmp_path):
    # A track's session data says has_existing_cues=True, but the live DB
    # (checked "at apply time, not just session flag" per the function's
    # own comment) has none -- must not trigger the overwrite-confirmation
    # path at all, since nothing is actually being overwritten.
    from djcues.writer import apply_session

    session_path = _write_session_file(
        tmp_path, {"1": _session_track(status="accepted", has_existing_cues=True)}
    )
    db = _fake_db_with_master(tmp_path)
    db.get_cue.return_value = []  # live DB: nothing there despite the session's stale flag

    with patch("djcues.db.get_db", return_value=db), \
         patch("djcues.history.log_session_corrections"), \
         patch("click.confirm") as mock_confirm:
        apply_session(session_path)

    mock_confirm.assert_not_called()


def test_apply_session_existing_cues_prompts_and_proceeds_on_confirm(tmp_path):
    from djcues.writer import apply_session

    session_path = _write_session_file(
        tmp_path, {"1": _session_track(status="accepted", has_existing_cues=True)}
    )
    db = _fake_db_with_master(tmp_path)
    existing_cue = MagicMock(Kind=1, Comment="First Beat", InMsec=100)
    db.get_cue.return_value = [existing_cue]

    with patch("djcues.db.get_db", return_value=db), \
         patch("djcues.history.log_session_corrections"), \
         patch("click.confirm", return_value=True) as mock_confirm:
        result = apply_session(session_path)

    mock_confirm.assert_called_once()
    assert result["written"] == 1
    # overwrite=True must actually have reached write_cues_for_track --
    # otherwise this would leave the old cue in place *and* add a new
    # one, silently duplicating cues in the real database instead of
    # replacing them.
    db.delete.assert_called_once_with(existing_cue)


def test_apply_session_existing_cues_declined_aborts_without_writing(tmp_path):
    from djcues.writer import apply_session

    session_path = _write_session_file(
        tmp_path, {"1": _session_track(status="accepted", has_existing_cues=True)}
    )
    db = _fake_db_with_master(tmp_path)
    db.get_cue.return_value = [MagicMock(Kind=1, Comment="First Beat", InMsec=100)]

    with patch("djcues.db.get_db", return_value=db), \
         patch("djcues.history.log_session_corrections"), \
         patch("click.confirm", side_effect=click.Abort):
        with pytest.raises(click.Abort):
            apply_session(session_path)

    db.add.assert_not_called()
    db.commit.assert_not_called()


def test_apply_session_force_skips_confirmation_even_with_existing_cues(tmp_path):
    from djcues.writer import apply_session

    session_path = _write_session_file(
        tmp_path, {"1": _session_track(status="accepted", has_existing_cues=True)}
    )
    db = _fake_db_with_master(tmp_path)
    existing_cue = MagicMock(Kind=1, Comment="First Beat", InMsec=100)
    db.get_cue.return_value = [existing_cue]

    with patch("djcues.db.get_db", return_value=db), \
         patch("djcues.history.log_session_corrections"), \
         patch("click.confirm") as mock_confirm:
        result = apply_session(session_path, force=True)

    mock_confirm.assert_not_called()
    assert result["written"] == 1
    # force=True skips the *prompt*, but overwrite itself must still
    # happen -- same reasoning as the confirmed-prompt test above.
    db.delete.assert_called_once_with(existing_cue)
