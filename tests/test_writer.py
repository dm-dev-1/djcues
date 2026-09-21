"""Tests for djcues.writer — backup, cue row building, DB write."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import click
import pytest

from djcues.constants import CUE_SYSTEM, CUE_SYSTEM_BY_PAD
from tests.conftest import requires_rekordbox, requires_rekordbox_closed


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


# ---------------------------------------------------------------------------
# Playlist membership (move/add/remove) -- djcues's first write path that
# isn't cue data. db is always a MagicMock; djcues.db.find_playlist_song
# _entries and pyrekordbox.utils.get_rekordbox_pid are patched at their
# real origin (both are imported locally inside writer.py's functions, the
# same reason djcues.db.get_db is patched at its origin above, not as
# djcues.writer.get_db). backup_database is NOT mocked, same reasoning as
# apply_session's tests -- a real shutil.copy2 against a real (fake,
# tmp_path-only) master.db, for genuine coverage with no risk to real data.
# ---------------------------------------------------------------------------


def _entry(track_no: int, entry_id: str = None) -> MagicMock:
    return MagicMock(TrackNo=track_no, ID=entry_id or f"entry-{track_no}")


def test_ensure_rekordbox_closed_raises_when_running():
    from djcues.writer import RekordboxRunningError, ensure_rekordbox_closed

    with patch("pyrekordbox.utils.get_rekordbox_pid", return_value=12345):
        with pytest.raises(RekordboxRunningError, match="Rekordbox is running"):
            ensure_rekordbox_closed()


def test_ensure_rekordbox_closed_no_error_when_not_running():
    from djcues.writer import ensure_rekordbox_closed

    with patch("pyrekordbox.utils.get_rekordbox_pid", return_value=None):
        ensure_rekordbox_closed()  # must not raise


def test_resolve_playlist_entry_no_match_raises_not_in_playlist():
    from djcues.writer import TrackNotInPlaylistError, resolve_playlist_entry

    with patch("djcues.db.find_playlist_song_entries", return_value=[]):
        with pytest.raises(TrackNotInPlaylistError):
            resolve_playlist_entry("pl1", "c1")


def test_resolve_playlist_entry_single_match_no_position_returns_it():
    from djcues.writer import resolve_playlist_entry

    entry = _entry(3)
    with patch("djcues.db.find_playlist_song_entries", return_value=[entry]):
        result = resolve_playlist_entry("pl1", "c1")

    assert result is entry


def test_resolve_playlist_entry_single_match_wrong_position_still_raises():
    # A stale/wrong --position must be caught, not silently ignored just
    # because there was only one entry to begin with.
    from djcues.writer import TrackNotInPlaylistError, resolve_playlist_entry

    entry = _entry(3)
    with patch("djcues.db.find_playlist_song_entries", return_value=[entry]):
        with pytest.raises(TrackNotInPlaylistError):
            resolve_playlist_entry("pl1", "c1", position=5)


def test_resolve_playlist_entry_multiple_matches_no_position_raises_ambiguous():
    from djcues.writer import AmbiguousTrackEntryError, resolve_playlist_entry

    e1, e2 = _entry(1), _entry(4)
    with patch("djcues.db.find_playlist_song_entries", return_value=[e1, e2]):
        with pytest.raises(AmbiguousTrackEntryError) as exc_info:
            resolve_playlist_entry("pl1", "c1")

    assert set(exc_info.value.candidates) == {(1, "entry-1"), (4, "entry-4")}


def test_resolve_playlist_entry_multiple_matches_position_picks_one():
    from djcues.writer import resolve_playlist_entry

    e1, e2 = _entry(1), _entry(4)
    with patch("djcues.db.find_playlist_song_entries", return_value=[e1, e2]):
        result = resolve_playlist_entry("pl1", "c1", position=4)

    assert result is e2


def _fake_playlist_db(tmp_path, *, source=None, dest=None, content=None) -> MagicMock:
    (tmp_path / "master.db").write_bytes(b"fake-sqlite-content")
    db = MagicMock()
    db.db_directory = tmp_path

    playlists = {}
    if source is not None:
        playlists[source.ID] = source
    if dest is not None:
        playlists[dest.ID] = dest
    db.get_playlist.side_effect = lambda ID: playlists.get(ID)
    db.get_content.return_value = content if content is not None else MagicMock(ID="c1")
    return db


def test_add_track_to_playlist_happy_path(tmp_path):
    from djcues.writer import add_track_to_playlist

    dest = MagicMock(ID="dest1")
    db = _fake_playlist_db(tmp_path, dest=dest)

    with patch("pyrekordbox.utils.get_rekordbox_pid", return_value=None):
        add_track_to_playlist("dest1", "c1", db=db)

    db.add_to_playlist.assert_called_once_with(dest, db.get_content.return_value, track_no=None)
    db.commit.assert_called_once()
    assert list(tmp_path.glob("master-backup-*.db"))


def test_add_track_to_playlist_unknown_playlist_raises_before_backup(tmp_path):
    from djcues.writer import add_track_to_playlist

    db = _fake_playlist_db(tmp_path)  # no playlists registered

    with patch("pyrekordbox.utils.get_rekordbox_pid", return_value=None):
        with pytest.raises(ValueError, match="not found"):
            add_track_to_playlist("dest1", "c1", db=db)

    assert not list(tmp_path.glob("master-backup-*.db"))
    db.add_to_playlist.assert_not_called()


def test_add_track_to_playlist_rekordbox_running_raises_before_backup(tmp_path):
    from djcues.writer import RekordboxRunningError, add_track_to_playlist

    dest = MagicMock(ID="dest1")
    db = _fake_playlist_db(tmp_path, dest=dest)

    with patch("pyrekordbox.utils.get_rekordbox_pid", return_value=99999):
        with pytest.raises(RekordboxRunningError):
            add_track_to_playlist("dest1", "c1", db=db)

    assert not list(tmp_path.glob("master-backup-*.db"))
    db.add_to_playlist.assert_not_called()


def test_add_track_to_playlist_commit_runtime_error_rolls_back_and_reraises(tmp_path):
    from djcues.writer import RekordboxRunningError, add_track_to_playlist

    dest = MagicMock(ID="dest1")
    db = _fake_playlist_db(tmp_path, dest=dest)
    db.commit.side_effect = RuntimeError("Rekordbox is running.")

    with patch("pyrekordbox.utils.get_rekordbox_pid", return_value=None):
        with pytest.raises(RekordboxRunningError):
            add_track_to_playlist("dest1", "c1", db=db)

    db.rollback.assert_called_once()


def test_add_track_to_playlist_unexpected_error_rolls_back_and_reraises(tmp_path):
    from djcues.writer import add_track_to_playlist

    dest = MagicMock(ID="dest1")
    db = _fake_playlist_db(tmp_path, dest=dest)
    db.add_to_playlist.side_effect = ValueError("Playlist must be a normal playlist")

    with patch("pyrekordbox.utils.get_rekordbox_pid", return_value=None):
        with pytest.raises(ValueError, match="normal playlist"):
            add_track_to_playlist("dest1", "c1", db=db)

    db.rollback.assert_called_once()


def test_remove_track_from_playlist_happy_path(tmp_path):
    from djcues.writer import remove_track_from_playlist

    source = MagicMock(ID="src1")
    db = _fake_playlist_db(tmp_path, source=source)
    entry = _entry(1)

    with patch("pyrekordbox.utils.get_rekordbox_pid", return_value=None), \
         patch("djcues.db.find_playlist_song_entries", return_value=[entry]):
        remove_track_from_playlist("src1", "c1", db=db)

    # Passed the already-resolved entry object, not a raw ID -- pyrekordbox's
    # own internal re-lookup (a possible NoResultFound source) never runs.
    db.remove_from_playlist.assert_called_once_with(source, entry)
    db.commit.assert_called_once()  # flushes the trailing TrackNo renumbering
    assert list(tmp_path.glob("master-backup-*.db"))


def test_remove_track_from_playlist_not_in_playlist_raises_before_backup(tmp_path):
    from djcues.writer import TrackNotInPlaylistError, remove_track_from_playlist

    source = MagicMock(ID="src1")
    db = _fake_playlist_db(tmp_path, source=source)

    with patch("pyrekordbox.utils.get_rekordbox_pid", return_value=None), \
         patch("djcues.db.find_playlist_song_entries", return_value=[]):
        with pytest.raises(TrackNotInPlaylistError):
            remove_track_from_playlist("src1", "c1", db=db)

    assert not list(tmp_path.glob("master-backup-*.db"))
    db.remove_from_playlist.assert_not_called()


def test_remove_track_from_playlist_unexpected_error_rolls_back_and_reraises(tmp_path):
    from djcues.writer import remove_track_from_playlist

    source = MagicMock(ID="src1")
    db = _fake_playlist_db(tmp_path, source=source)
    entry = _entry(1)
    db.remove_from_playlist.side_effect = RuntimeError("boom")

    with patch("pyrekordbox.utils.get_rekordbox_pid", return_value=None), \
         patch("djcues.db.find_playlist_song_entries", return_value=[entry]):
        # A generic RuntimeError not from commit() itself would be an odd
        # real-world case (remove_from_playlist has no other RuntimeError
        # source today), but the handler can't tell the difference --
        # confirm it's still mapped to RekordboxRunningError, matching
        # add's handling exactly, and that rollback still happens.
        from djcues.writer import RekordboxRunningError
        with pytest.raises(RekordboxRunningError):
            remove_track_from_playlist("src1", "c1", db=db)

    db.rollback.assert_called_once()


def test_move_track_between_playlists_same_playlist_raises():
    from djcues.writer import move_track_between_playlists

    db = MagicMock()
    with patch("pyrekordbox.utils.get_rekordbox_pid", return_value=None):
        with pytest.raises(ValueError, match="same"):
            move_track_between_playlists("pl1", "pl1", "c1", db=db)

    db.get_playlist.assert_not_called()  # rejected before any lookup


def test_move_track_between_playlists_calls_add_before_remove_with_no_commit_between(tmp_path):
    """The single most important test in this feature: move's atomicity
    guarantee (see writer.py's own comment on move_track_between_playlists)
    depends entirely on add_to_playlist (stage only) being called before
    remove_from_playlist (which commits immediately, flushing both halves
    together) -- with nothing committing in between. If a future change
    reorders these, or routes through the add_track_to_playlist/
    remove_track_from_playlist wrappers instead (each of which commits on
    its own), this test must fail.
    """
    from djcues.writer import move_track_between_playlists

    source = MagicMock(ID="src1")
    dest = MagicMock(ID="dest1")
    content = MagicMock(ID="c1")
    db = _fake_playlist_db(tmp_path, source=source, dest=dest, content=content)
    entry = _entry(1)

    calls: list[str] = []
    db.add_to_playlist.side_effect = lambda *a, **k: calls.append("add_to_playlist")
    db.remove_from_playlist.side_effect = lambda *a, **k: calls.append("remove_from_playlist")
    db.commit.side_effect = lambda: calls.append("commit")

    with patch("pyrekordbox.utils.get_rekordbox_pid", return_value=None), \
         patch("djcues.db.find_playlist_song_entries", return_value=[entry]):
        move_track_between_playlists("src1", "dest1", "c1", db=db)

    # add, then remove (which internally commits both), then exactly one
    # more explicit commit for the trailing renumbering -- never a commit
    # between add and remove.
    assert calls == ["add_to_playlist", "remove_from_playlist", "commit"]
    db.add_to_playlist.assert_called_once_with(dest, content)
    db.remove_from_playlist.assert_called_once_with(source, entry)


def test_move_track_between_playlists_rekordbox_running_raises_before_backup(tmp_path):
    from djcues.writer import RekordboxRunningError, move_track_between_playlists

    source = MagicMock(ID="src1")
    dest = MagicMock(ID="dest1")
    db = _fake_playlist_db(tmp_path, source=source, dest=dest)

    with patch("pyrekordbox.utils.get_rekordbox_pid", return_value=99999):
        with pytest.raises(RekordboxRunningError):
            move_track_between_playlists("src1", "dest1", "c1", db=db)

    assert not list(tmp_path.glob("master-backup-*.db"))
    db.add_to_playlist.assert_not_called()


def test_move_track_between_playlists_failure_rolls_back_and_reraises(tmp_path):
    from djcues.writer import move_track_between_playlists

    source = MagicMock(ID="src1")
    dest = MagicMock(ID="dest1")
    db = _fake_playlist_db(tmp_path, source=source, dest=dest)
    entry = _entry(1)
    db.remove_from_playlist.side_effect = ValueError("something went wrong")

    with patch("pyrekordbox.utils.get_rekordbox_pid", return_value=None), \
         patch("djcues.db.find_playlist_song_entries", return_value=[entry]):
        with pytest.raises(ValueError, match="something went wrong"):
            move_track_between_playlists("src1", "dest1", "c1", db=db)

    db.rollback.assert_called_once()


# ---------------------------------------------------------------------------
# reorder_playlist -- djcues's first write that reorders tracks WITHIN a
# playlist. db is always a MagicMock; backup_database is real (not mocked),
# same reasoning as every other write test above.
# ---------------------------------------------------------------------------


def _song_row(content_id: str, track_no: int) -> MagicMock:
    row = MagicMock()
    row.ContentID = content_id
    row.TrackNo = track_no
    return row


def _make_fake_move_song_in_playlist(rows: list):
    """A side_effect for a mocked db.move_song_in_playlist mimicking
    pyrekordbox's real TrackNo-shifting behavior (db6/database.py:1051-
    1074) closely enough for reorder_playlist's own guard
    (`row.TrackNo != target_track_no`) to see a realistic in-between
    state across the loop -- later positions can become correct "for
    free" from an earlier move's shift, which is exactly the behavior
    the guard is designed to skip."""
    def _move(playlist, song, new_track_no):
        old_track_no = song.TrackNo
        if new_track_no > old_track_no:
            for other in rows:
                if old_track_no < other.TrackNo <= new_track_no:
                    other.TrackNo -= 1
        elif new_track_no < old_track_no:
            for other in rows:
                if new_track_no <= other.TrackNo < old_track_no:
                    other.TrackNo += 1
        song.TrackNo = new_track_no
    return _move


def test_reorder_playlist_rekordbox_running_raises_before_backup(tmp_path):
    from djcues.writer import RekordboxRunningError, reorder_playlist

    playlist = MagicMock(ID="pl1")
    db = _fake_playlist_db(tmp_path, source=playlist)

    with patch("pyrekordbox.utils.get_rekordbox_pid", return_value=99999):
        with pytest.raises(RekordboxRunningError):
            reorder_playlist("pl1", ["A"], db=db)

    assert not list(tmp_path.glob("master-backup-*.db"))
    db.get_playlist_songs.assert_not_called()


def test_reorder_playlist_unknown_playlist_raises_before_backup(tmp_path):
    from djcues.writer import reorder_playlist

    db = _fake_playlist_db(tmp_path)  # no playlists registered

    with patch("pyrekordbox.utils.get_rekordbox_pid", return_value=None):
        with pytest.raises(ValueError, match="not found"):
            reorder_playlist("pl1", ["A"], db=db)

    assert not list(tmp_path.glob("master-backup-*.db"))


def test_reorder_playlist_permutation_mismatch_raises_before_backup(tmp_path):
    from djcues.writer import reorder_playlist

    playlist = MagicMock(ID="pl1")
    db = _fake_playlist_db(tmp_path, source=playlist)
    db.get_playlist_songs.return_value = [_song_row("A", 1), _song_row("B", 2)]

    with patch("pyrekordbox.utils.get_rekordbox_pid", return_value=None):
        with pytest.raises(ValueError, match="changed since this order was computed"):
            reorder_playlist("pl1", ["A", "C"], db=db)  # C isn't really in the playlist

    assert not list(tmp_path.glob("master-backup-*.db"))
    db.move_song_in_playlist.assert_not_called()


def test_reorder_playlist_correct_sequence_and_positions_skipping_free_matches(tmp_path):
    """Hand-verified scenario: current [D,B,A,C] at TrackNo 1-4, target
    [A,B,C,D] -- exactly 3 real moves land on [A,B,C,D]; the 4th (D,
    already at position 4 after the third move's own shift) must be
    skipped by the TrackNo != i guard, not called "just to be safe"."""
    from djcues.writer import reorder_playlist

    playlist = MagicMock(ID="pl1")
    d, b, a, c = _song_row("D", 1), _song_row("B", 2), _song_row("A", 3), _song_row("C", 4)
    rows = [d, b, a, c]
    db = _fake_playlist_db(tmp_path, source=playlist)
    db.get_playlist_songs.return_value = rows
    db.move_song_in_playlist.side_effect = _make_fake_move_song_in_playlist(rows)

    with patch("pyrekordbox.utils.get_rekordbox_pid", return_value=None):
        reorder_playlist("pl1", ["A", "B", "C", "D"], db=db)

    assert db.move_song_in_playlist.call_args_list == [
        call(playlist, a, 1), call(playlist, b, 2), call(playlist, c, 3),
    ]
    assert [row.TrackNo for row in (a, b, c, d)] == [1, 2, 3, 4]
    assert len(list(tmp_path.glob("master-backup-*.db"))) == 1
    db.commit.assert_called_once()


def test_reorder_playlist_already_correct_order_makes_zero_move_calls(tmp_path):
    """Directly enforces no "helpful" special-casing of the last
    position: everything already correct must mean move_song_in_playlist
    is called ZERO times, not once "just to be sure" -- calling it even
    once when new_track_no == old_track_no hits the confirmed
    pyrekordbox tracking-disable bug."""
    from djcues.writer import reorder_playlist

    playlist = MagicMock(ID="pl1")
    db = _fake_playlist_db(tmp_path, source=playlist)
    db.get_playlist_songs.return_value = [_song_row("A", 1), _song_row("B", 2), _song_row("C", 3)]

    with patch("pyrekordbox.utils.get_rekordbox_pid", return_value=None):
        reorder_playlist("pl1", ["A", "B", "C"], db=db)

    db.move_song_in_playlist.assert_not_called()
    db.commit.assert_called_once()
    assert list(tmp_path.glob("master-backup-*.db"))


def test_reorder_playlist_handles_track_appearing_twice_in_one_playlist(tmp_path):
    """A track can legally appear more than once in one playlist (see
    resolve_playlist_entry's own docstring). Confirms each occurrence of
    a repeated content ID resolves to a DISTINCT real row, in original
    TrackNo order -- not silently collapsed onto the same row (a real
    bug a naive content_id -> single row dict would have)."""
    from djcues.writer import reorder_playlist

    playlist = MagicMock(ID="pl1")
    a1 = _song_row("A", 1)  # first "A" entry
    b = _song_row("B", 2)
    a2 = _song_row("A", 3)  # second "A" entry -- same content ID as a1
    rows = [a1, b, a2]
    db = _fake_playlist_db(tmp_path, source=playlist)
    db.get_playlist_songs.return_value = rows
    db.move_song_in_playlist.side_effect = _make_fake_move_song_in_playlist(rows)

    with patch("pyrekordbox.utils.get_rekordbox_pid", return_value=None):
        # position1=A(1st occurrence->a1), position2=A(2nd occurrence->a2), position3=B
        reorder_playlist("pl1", ["A", "A", "B"], db=db)

    # Only a2 should move (to position 2) -- a1 stays at 1, b becomes
    # correct "for free" via a2's shift. A buggy flat dict would instead
    # try to move a1 twice and never touch a2 at all.
    assert db.move_song_in_playlist.call_args_list == [call(playlist, a2, 2)]
    assert a1.TrackNo == 1 and a2.TrackNo == 2 and b.TrackNo == 3


def test_reorder_playlist_commit_runtime_error_rolls_back_and_reraises(tmp_path):
    from djcues.writer import RekordboxRunningError, reorder_playlist

    playlist = MagicMock(ID="pl1")
    db = _fake_playlist_db(tmp_path, source=playlist)
    db.get_playlist_songs.return_value = [_song_row("A", 1), _song_row("B", 2)]
    db.commit.side_effect = RuntimeError("Rekordbox is running.")

    with patch("pyrekordbox.utils.get_rekordbox_pid", return_value=None):
        with pytest.raises(RekordboxRunningError):
            reorder_playlist("pl1", ["B", "A"], db=db)

    db.rollback.assert_called_once()


def test_reorder_playlist_unexpected_error_rolls_back_and_reraises(tmp_path):
    from djcues.writer import reorder_playlist

    playlist = MagicMock(ID="pl1")
    db = _fake_playlist_db(tmp_path, source=playlist)
    db.get_playlist_songs.return_value = [_song_row("A", 1), _song_row("B", 2)]
    db.move_song_in_playlist.side_effect = ValueError("boom")

    with patch("pyrekordbox.utils.get_rekordbox_pid", return_value=None):
        with pytest.raises(ValueError, match="boom"):
            reorder_playlist("pl1", ["B", "A"], db=db)

    db.rollback.assert_called_once()


# ---------------------------------------------------------------------------
# Real, opt-in integration tests -- genuinely write to the real rekordbox
# database (`pytest -m destructive` to run; excluded from a normal
# `pytest` run by pyproject.toml's addopts). Everything above this line
# uses a MagicMock db and never touches real data.
#
# Confined entirely to "CUE Analysis Playlist" -- a real, currently-empty
# (0 tracks) playlist in this library, confirmed live immediately before
# writing this test (not assumed from an old session note -- see
# writer.py's plan file for why that distinction matters here). Every
# test below re-confirms it's still empty at the start, for the same
# reason: a stale assumption about real, live data is exactly the kind
# of mistake this project's own discipline exists to catch. Only ever
# adds to / removes from this one playlist -- a real track from Tech
# House is used as the content being added, but Tech House itself is
# never written to (adding a track to one playlist doesn't remove it
# from any other), so it's never at risk. move_track_between_playlists
# is deliberately NOT exercised live against two real curated playlists
# here -- its exact ordering/atomicity/rollback behavior is already
# covered exhaustively above with mocks (more precisely than a live test
# could observe from end-state alone), and it internally calls the same
# two pyrekordbox primitives add/remove's own live tests below already
# exercise for real, so a live move test would add materially more real-
# data risk for comparatively little additional real-API coverage.
# ---------------------------------------------------------------------------


SCRATCH_PLAYLIST_NAME = "CUE Analysis Playlist"


def _real_scratch_playlist_and_source_track():
    from djcues.db import find_playlist, list_playlist_tracks

    scratch = find_playlist(SCRATCH_PLAYLIST_NAME)
    assert scratch is not None, f"expected a real '{SCRATCH_PLAYLIST_NAME}' playlist in this library"
    assert list_playlist_tracks(scratch.ID) == [], (
        f"'{SCRATCH_PLAYLIST_NAME}' is expected to stay empty between test runs -- "
        "it has tracks in it right now, so this test won't touch it (don't assume "
        "it's still safe to use without checking; see this file's own comment above)"
    )

    source = find_playlist("Tech House")
    assert source is not None, "expected a real 'Tech House' playlist in this library"
    tech_house_tracks = list_playlist_tracks(source.ID)
    assert tech_house_tracks, "expected 'Tech House' to have at least one real track"
    return scratch, tech_house_tracks[0]


@requires_rekordbox
@requires_rekordbox_closed
@pytest.mark.destructive
def test_add_and_remove_real_track_roundtrip_against_scratch_playlist():
    """The one real, live confirmation this feature's whole write path
    (backup -> pyrekordbox add/remove -> commit) works end to end against
    the real installed pyrekordbox package and this real library -- not
    just against mocks. Backup verification is folded into this same
    test rather than a separate one: backup_database()'s timestamp has
    only second-level granularity, so two separate real writes from two
    separate destructive tests running less than a second apart can
    legitimately produce the *same* backup filename (confirmed live
    while writing this) -- not a bug in backup_database (apply_session
    has relied on this exact behavior for a while), just a reason to
    keep this to one real write instead of two.
    """
    from djcues.db import get_db, list_playlist_tracks
    from djcues.writer import add_track_to_playlist, remove_track_from_playlist

    scratch, track = _real_scratch_playlist_and_source_track()
    db = get_db()
    master_db_path = db.db_directory / "master.db"
    backups_before = set(master_db_path.parent.glob("master-backup-*.db"))

    try:
        add_track_to_playlist(scratch.ID, track.id)

        after_add = list_playlist_tracks(scratch.ID)
        assert len(after_add) == 1
        assert after_add[0].id == track.id

        backups_after = set(master_db_path.parent.glob("master-backup-*.db"))
        assert backups_after >= backups_before  # never fewer; a real backup should exist either way
        assert backups_after, "expected at least one real master-backup-*.db file to exist after a real write"
    finally:
        # Runs even if the assertions above failed, so a real assertion
        # failure can't leave the scratch playlist non-empty for the
        # next run of this same test.
        remaining = list_playlist_tracks(scratch.ID)
        for t in remaining:
            remove_track_from_playlist(scratch.ID, t.id)

    assert list_playlist_tracks(scratch.ID) == []


@requires_rekordbox
@requires_rekordbox_closed
@pytest.mark.destructive
def test_reorder_real_track_roundtrip_against_scratch_playlist():
    """The one real, live confirmation reorder_playlist's whole write
    path (backup -> pyrekordbox move_song_in_playlist calls -> commit)
    works end to end against the real installed pyrekordbox package and
    this real library -- not just mocks. A genuinely new test precedent:
    no prior djcues write reorders tracks WITHIN a playlist.

    Populates the scratch playlist with 3 real tracks from 'Tech House'
    (via the already-covered add_track_to_playlist), captures the real
    resulting order, applies a fully reversed target order via
    reorder_playlist, and asserts the real resulting order via a fresh
    list_playlist_tracks() read matches exactly. Cleans up in a finally
    block so a failed assertion can't leave the scratch playlist dirty.
    """
    from djcues.db import find_playlist, list_playlist_tracks
    from djcues.writer import add_track_to_playlist, reorder_playlist, remove_track_from_playlist

    scratch = find_playlist(SCRATCH_PLAYLIST_NAME)
    assert scratch is not None, f"expected a real '{SCRATCH_PLAYLIST_NAME}' playlist in this library"
    assert list_playlist_tracks(scratch.ID) == [], (
        f"'{SCRATCH_PLAYLIST_NAME}' is expected to stay empty between test runs -- "
        "it has tracks in it right now, so this test won't touch it"
    )

    source = find_playlist("Tech House")
    assert source is not None, "expected a real 'Tech House' playlist in this library"
    tech_house_tracks = list_playlist_tracks(source.ID)
    assert len(tech_house_tracks) >= 3, "expected 'Tech House' to have at least 3 real tracks"
    chosen = tech_house_tracks[:3]

    try:
        for t in chosen:
            add_track_to_playlist(scratch.ID, t.id)

        before = list_playlist_tracks(scratch.ID)
        assert len(before) == 3
        assert {t.id for t in before} == {t.id for t in chosen}

        reversed_order = [t.id for t in reversed(before)]
        reorder_playlist(scratch.ID, reversed_order)

        after = list_playlist_tracks(scratch.ID)
        assert [t.id for t in after] == reversed_order
    finally:
        remaining = list_playlist_tracks(scratch.ID)
        for t in remaining:
            remove_track_from_playlist(scratch.ID, t.id)

    assert list_playlist_tracks(scratch.ID) == []
