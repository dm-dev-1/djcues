"""Write cue data to the rekordbox database."""

from __future__ import annotations

import json
import pathlib
import shutil
from datetime import datetime
from uuid import uuid4

from djcues.constants import CUE_SYSTEM, CUE_SYSTEM_BY_PAD


class PlaylistWriteError(Exception):
    """Base class for playlist add/remove/move failures."""


class RekordboxRunningError(PlaylistWriteError):
    """Rekordbox is currently running -- pyrekordbox refuses to commit
    while it's open (no bypass), so this is raised before attempting
    any write rather than letting a raw RuntimeError surface."""


class AmbiguousTrackEntryError(PlaylistWriteError):
    """The track appears more than once in the playlist (legal, if
    unusual), and no --position was given to say which entry to act on.

    candidates: (track_no, entry_id) for every matching entry, sorted
    by track_no -- enough for a caller to print a disambiguation list.
    """

    def __init__(self, message: str, candidates: list[tuple[int, str]]):
        super().__init__(message)
        self.candidates = candidates


class TrackNotInPlaylistError(PlaylistWriteError):
    """The track isn't in the given playlist, or --position didn't
    match any entry that is."""


def _format_ms(ms: float) -> str:
    """Format milliseconds as M:SS.s"""
    total_seconds = ms / 1000
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:04.1f}"


def backup_database(db_path: pathlib.Path) -> pathlib.Path:
    """Copy master.db to a timestamped backup. Returns the backup path."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = db_path.stem
    backup_name = f"{stem}-backup-{timestamp}.db"
    backup_path = db_path.parent / backup_name
    shutil.copy2(db_path, backup_path)
    return backup_path


def build_cue_rows(
    cues: dict, memory_cues: dict
) -> tuple[list[dict], list[dict]]:
    """Convert session cue data into lists of field dicts for DjmdCue creation.

    Hot cues are keyed by pad letter A-H; memory cues by slot number "1"-"8".
    Skips entries with status "skipped". Treats "auto" and "adjusted" as accepted.
    """
    hot_rows: list[dict] = []
    mem_rows: list[dict] = []

    # --- Hot cues ---
    for pad, slot in CUE_SYSTEM_BY_PAD.items():
        entry = cues.get(pad)
        if entry is None:
            continue
        if entry.get("status") == "skipped":
            continue

        is_loop = slot.is_loop
        in_msec = int(entry["position_ms"])
        if is_loop and entry.get("loop_end_ms") is not None:
            out_msec = int(entry["loop_end_ms"])
        else:
            out_msec = -1

        hot_rows.append({
            "Kind": slot.kind,
            "InMsec": in_msec,
            "InFrame": 0,
            "InMpegFrame": 0,
            "InMpegAbs": 0,
            "OutMsec": out_msec,
            "OutFrame": 0 if is_loop else -1,
            "OutMpegFrame": 0 if is_loop else -1,
            "OutMpegAbs": 0 if is_loop else -1,
            "Color": slot.hot_cue_color,
            "ColorTableIndex": slot.hot_cue_color_table_index,
            "ActiveLoop": 0 if is_loop else -1,
            "Comment": slot.hot_cue_label,
            "BeatLoopSize": 0,
            "CueMicrosec": 0,
        })

    # --- Memory cues ---
    for slot_num_str, entry in memory_cues.items():
        if entry.get("status") == "skipped":
            continue

        slot_idx = int(slot_num_str) - 1
        slot = CUE_SYSTEM[slot_idx]

        is_loop = slot.is_loop
        in_msec = int(entry["position_ms"])
        if is_loop and entry.get("loop_end_ms") is not None:
            out_msec = int(entry["loop_end_ms"])
        else:
            out_msec = -1

        mem_rows.append({
            "Kind": 0,
            "InMsec": in_msec,
            "InFrame": 0,
            "InMpegFrame": 0,
            "InMpegAbs": 0,
            "OutMsec": out_msec,
            "OutFrame": 0 if is_loop else -1,
            "OutMpegFrame": 0 if is_loop else -1,
            "OutMpegAbs": 0 if is_loop else -1,
            "Color": slot.memory_cue_color,
            "ColorTableIndex": slot.memory_cue_color_table_index,
            "ActiveLoop": 0 if is_loop else -1,
            "Comment": slot.memory_cue_label,
            "BeatLoopSize": 0,
            "CueMicrosec": 0,
        })

    return hot_rows, mem_rows


def write_cues_for_track(db, content, hot_rows, mem_rows, overwrite=False) -> int:
    """Write cue rows to the DB.

    If overwrite is True, deletes existing cues first.
    Returns the count of cues written.
    """
    from pyrekordbox.db6 import tables

    if overwrite:
        existing = db.get_cue(ContentID=content.ID)
        for cue in existing:
            db.delete(cue)

    count = 0
    for row in hot_rows + mem_rows:
        cue_id = db.generate_unused_id(tables.DjmdCue)
        cue_uuid = str(uuid4())
        cue = tables.DjmdCue.create(
            ID=str(cue_id),
            ContentID=str(content.ID),
            ContentUUID=content.UUID,
            UUID=cue_uuid,
            **row,
        )
        db.add(cue)
        count += 1

    return count


def apply_session(session_path, dry_run=False, force=False) -> dict:
    """Read a session JSON file, show summary, back up DB, and write cues.

    One commit per track. Returns a summary dict.
    """
    import click
    from djcues.db import get_db
    from djcues.history import log_session_corrections

    session_path = pathlib.Path(session_path)
    with open(session_path) as f:
        session = json.load(f)

    tracks = session.get("tracks", {})

    # Count statuses
    accepted = 0
    adjusted = 0
    skipped = 0
    overwrite_ids: list[str] = []

    for track_id, track_data in tracks.items():
        status = track_data.get("status", "")
        if status == "accepted":
            accepted += 1
        elif status == "adjusted":
            adjusted += 1
        elif status == "skipped":
            skipped += 1

        if status in ("accepted", "adjusted"):
            if track_data.get("has_existing_cues", False):
                overwrite_ids.append(track_id)

    total_write = accepted + adjusted

    click.echo(f"Session: {session_path.name}")
    click.echo(f"  Accepted: {accepted}  Adjusted: {adjusted}  Skipped: {skipped}")
    click.echo(f"  Tracks to write: {total_write}")
    if overwrite_ids:
        click.echo(f"  Tracks with existing cues (overwrite): {len(overwrite_ids)}")

    result = {
        "accepted": accepted,
        "adjusted": adjusted,
        "skipped": skipped,
        "written": 0,
        "cues_written": 0,
    }

    if dry_run:
        click.echo("Dry run — no changes written.")
        return result

    db = get_db()
    backup_path = backup_database(db.db_directory / "master.db")
    click.echo(f"Backup: {backup_path}")

    written = 0
    cues_written = 0

    # Check DB for existing cues at apply time (not just session flag)
    tracks_with_existing: dict[str, list] = {}  # track_id -> existing cue list
    for track_id, track_data in tracks.items():
        status = track_data.get("status", "")
        if status not in ("accepted", "adjusted"):
            continue
        existing = list(db.get_cue(ContentID=int(track_id)))
        if existing:
            tracks_with_existing[track_id] = existing

    if tracks_with_existing and not force:
        from djcues.constants import KIND_TO_PAD

        click.echo(f"\n{len(tracks_with_existing)} track(s) have existing cues that will be replaced:\n")
        for track_id, existing in tracks_with_existing.items():
            track_data = tracks[track_id]
            title = track_data.get("title", track_id)
            click.echo(f"  {title}")

            # Show existing cues
            existing_hot = sorted([c for c in existing if c.Kind > 0], key=lambda c: c.Kind)
            existing_mem = [c for c in existing if c.Kind == 0]
            click.echo(f"    Existing: {len(existing_hot)} hot, {len(existing_mem)} memory")
            for c in existing_hot:
                pad = KIND_TO_PAD.get(c.Kind, "?")
                click.echo(f"      [{pad}] {c.Comment or '?':20s} {_format_ms(c.InMsec)}")

            # Show proposed cues
            hot_data = track_data.get("cues", {})
            new_hot = [(p, d) for p, d in hot_data.items() if d.get("status") != "skipped"]
            new_mem = [(n, d) for n, d in track_data.get("memory_cues", {}).items() if d.get("status") != "skipped"]
            click.echo(f"    Proposed: {len(new_hot)} hot, {len(new_mem)} memory")
            for pad, d in sorted(new_hot):
                from djcues.constants import CUE_SYSTEM_BY_PAD
                slot = CUE_SYSTEM_BY_PAD.get(pad)
                label = slot.hot_cue_label if slot else pad
                click.echo(f"      [{pad}] {label:20s} {_format_ms(d['position_ms'])}")
            click.echo()

        click.confirm("Continue?", abort=True)

    for track_id, track_data in tracks.items():
        status = track_data.get("status", "")
        if status not in ("accepted", "adjusted"):
            continue

        hot_cues_data = track_data.get("cues", {})
        mem_cues_data = track_data.get("memory_cues", {})
        hot_rows, mem_rows = build_cue_rows(hot_cues_data, mem_cues_data)

        content = db.get_content(ID=int(track_id))
        overwrite = track_id in tracks_with_existing.keys()
        count = write_cues_for_track(db, content, hot_rows, mem_rows, overwrite=overwrite)
        db.commit()

        # Log immediately after this track's own commit succeeds, so a
        # failure partway through the loop only logs what was actually
        # written to Rekordbox, not tracks that never got there.
        log_session_corrections({"tracks": {track_id: track_data}}, str(session_path))

        title = track_data.get("title", f"ID {track_id}")
        click.echo(f"  Wrote {count} cues for {title}")
        written += 1
        cues_written += count

    # Whole-track skips never reach the loop above (they're excluded by the
    # accepted/adjusted status filter), but they're still a real correction
    # signal — the algorithm's proposals were rejected outright.
    for track_id, track_data in tracks.items():
        if track_data.get("status") == "skipped":
            log_session_corrections({"tracks": {track_id: track_data}}, str(session_path))

    result["written"] = written
    result["cues_written"] = cues_written
    click.echo(f"Done: {written} tracks, {cues_written} cues written.")
    return result


# --- Playlist membership (move/add/remove tracks between playlists) -------
#
# djcues's first write path that isn't cue data -- modeled directly on
# apply_session/backup_database above rather than inventing new safety
# machinery. Every function here takes already-resolved playlist/content
# IDs; name resolution and substring-match ambiguity is the CLI/
# dashboard layer's job (see cli.py's `playlist` group), matching how
# apply_session above already expects a resolved `content` object, not
# a raw title string.


def ensure_rekordbox_closed() -> None:
    """Fail fast if Rekordbox is currently running, before attempting
    any backup or write. pyrekordbox's own db.commit() already refuses
    to commit while Rekordbox is open (no bypass), but nothing in cli.py
    catches that raw RuntimeError today -- letting it surface here would
    print an unhandled traceback instead of a clean message. This is the
    single centralized check; callers below call it first, and CLI/
    dashboard just catch RekordboxRunningError rather than each
    duplicating this call. Doesn't make commit()'s own internal check
    redundant -- that's still the backstop for the race window between
    this check and the actual commit.
    """
    from pyrekordbox.utils import get_rekordbox_pid

    if get_rekordbox_pid():
        raise RekordboxRunningError(
            "Rekordbox is running. Close it before moving/adding/removing playlist tracks."
        )


def resolve_playlist_entry(
    playlist_id, content_id, position: int | None = None, db=None
):
    """Find the exact DjmdSongPlaylist entry for this track in this
    playlist -- needed because pyrekordbox's remove_from_playlist()
    takes the playlist-entry ID, not the track/content ID, and a track
    can legally appear in one playlist more than once.

    position, when given, is the 1-based TrackNo to disambiguate --
    matches even when there's only one entry (so a stale/wrong
    --position is caught rather than silently ignored).
    """
    from djcues.db import find_playlist_song_entries

    entries = find_playlist_song_entries(playlist_id, content_id, db=db)

    if not entries:
        raise TrackNotInPlaylistError(f"track {content_id} is not in playlist {playlist_id}")

    if position is not None:
        for entry in entries:
            if entry.TrackNo == position:
                return entry
        positions = [e.TrackNo for e in entries]
        raise TrackNotInPlaylistError(
            f"track {content_id} is in playlist {playlist_id}, but not at position "
            f"{position} -- it's at position(s) {positions}"
        )

    if len(entries) > 1:
        candidates = [(e.TrackNo, e.ID) for e in entries]
        raise AmbiguousTrackEntryError(
            f"track {content_id} appears {len(entries)} times in playlist {playlist_id} "
            f"-- pass --position to pick one ({[c[0] for c in candidates]})",
            candidates=candidates,
        )

    return entries[0]


def add_track_to_playlist(dest_playlist_id, content_id, track_no: int | None = None, db=None) -> None:
    """Add a track to a playlist (appended at the end unless track_no
    is given). Backs up master.db first. Does not check whether the
    track is already in the destination -- adding it again is a
    legitimate, if unusual, thing to want.
    """
    from djcues.db import get_db

    ensure_rekordbox_closed()
    db = db if db is not None else get_db()

    playlist = db.get_playlist(ID=dest_playlist_id)
    if playlist is None:
        raise ValueError(f"playlist {dest_playlist_id} not found")
    content = db.get_content(ID=content_id)
    if content is None:
        raise ValueError(f"track {content_id} not found")

    backup_database(db.db_directory / "master.db")

    try:
        db.add_to_playlist(playlist, content, track_no=track_no)
        db.commit()
    except RuntimeError as e:
        db.rollback()
        raise RekordboxRunningError(str(e)) from e
    except Exception:
        db.rollback()
        raise


def remove_track_from_playlist(source_playlist_id, content_id, position: int | None = None, db=None) -> None:
    """Remove a track from a playlist. Backs up master.db first."""
    from djcues.db import get_db

    ensure_rekordbox_closed()
    db = db if db is not None else get_db()

    playlist = db.get_playlist(ID=source_playlist_id)
    if playlist is None:
        raise ValueError(f"playlist {source_playlist_id} not found")

    # Resolve before backup -- nothing to undo yet if this raises.
    entry = resolve_playlist_entry(source_playlist_id, content_id, position=position, db=db)

    backup_database(db.db_directory / "master.db")

    try:
        # Pass the already-resolved entry object, not its ID, so
        # pyrekordbox's own internal re-lookup can't raise NoResultFound.
        db.remove_from_playlist(playlist, entry)
        db.commit()  # flush the trailing TrackNo renumbering
    except RuntimeError as e:
        db.rollback()
        raise RekordboxRunningError(str(e)) from e
    except Exception:
        db.rollback()
        raise


def move_track_between_playlists(
    source_playlist_id, dest_playlist_id, content_id, position: int | None = None, db=None
) -> None:
    """Move a track from one playlist to another. Backs up master.db
    once, covering both halves.

    CRITICAL: calls pyrekordbox's raw db.add_to_playlist()/
    db.remove_from_playlist() directly below, not the
    add_track_to_playlist()/remove_track_from_playlist() wrappers above.
    add_to_playlist() only stages (no commit); remove_from_playlist()
    commits immediately, and that commit flushes *everything* currently
    staged on the session -- so calling them back to back like this
    makes one commit persist both halves atomically: if anything fails
    first (including Rekordbox being open), neither half is written and
    the track stays exactly where it started. Using the wrapper
    functions instead would split this into two independent commits and
    reintroduce the exact "track lost from both playlists" hazard this
    function exists to avoid -- do not "simplify" this by calling them.
    """
    from djcues.db import get_db

    ensure_rekordbox_closed()
    db = db if db is not None else get_db()

    if str(source_playlist_id) == str(dest_playlist_id):
        raise ValueError("source and destination playlists are the same")

    source = db.get_playlist(ID=source_playlist_id)
    if source is None:
        raise ValueError(f"playlist {source_playlist_id} not found")
    dest = db.get_playlist(ID=dest_playlist_id)
    if dest is None:
        raise ValueError(f"playlist {dest_playlist_id} not found")
    content = db.get_content(ID=content_id)
    if content is None:
        raise ValueError(f"track {content_id} not found")

    entry = resolve_playlist_entry(source_playlist_id, content_id, position=position, db=db)

    backup_database(db.db_directory / "master.db")

    try:
        db.add_to_playlist(dest, content)  # stage only -- do NOT commit here
        db.remove_from_playlist(source, entry)  # commits both halves atomically
        db.commit()  # flush the trailing TrackNo renumbering only
    except RuntimeError as e:
        db.rollback()
        raise RekordboxRunningError(str(e)) from e
    except Exception:
        db.rollback()
        raise
