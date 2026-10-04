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


# --- Playlist order (write the energy-flow-computed track order) --------
#
# djcues's first write that reorders tracks WITHIN a playlist, rather
# than membership between playlists (the three functions above). Built
# on pyrekordbox's own Rekordbox6Database.move_song_in_playlist(), which
# already existed but was completely unused anywhere in djcues before
# this. Deliberately flow-agnostic -- this file has no existing
# dependency on flow.py/harmony.py/audit.py/clash.py and shouldn't gain
# one; callers (cli.py's `playlist reorder`, server.py's
# _handle_playlist_reorder_post) are responsible for computing the
# target order and handing this a plain list of content IDs.


def reorder_playlist(playlist_id, ordered_content_ids: list, db=None) -> None:
    """Write ordered_content_ids as playlist_id's real DjmdSongPlaylist
    TrackNo order.

    ordered_content_ids must be an exact permutation of the playlist's
    real, current track content IDs (same multiset -- same content IDs,
    same count, including matching any content ID that legitimately
    appears more than once in one playlist, see resolve_playlist_entry's
    own docstring above) -- checked against a fresh read, before
    backup_database(), and rejected with ValueError otherwise. This is
    the guard against writing a corrupted order if the playlist's real
    membership changed between when the order was computed (e.g. an
    earlier flow job in the dashboard) and when this is actually called.
    """
    from collections import defaultdict, deque

    from djcues.db import get_db

    ensure_rekordbox_closed()
    db = db if db is not None else get_db()

    playlist = db.get_playlist(ID=playlist_id)
    if playlist is None:
        raise ValueError(f"playlist {playlist_id} not found")

    # Live DjmdSongPlaylist rows for this playlist, fetched ONCE and
    # reused for both the permutation check below and the move loop --
    # NOT re-queried per-iteration. `.TrackNo or 0` matches
    # list_playlist_tracks()'s own defensive sort key in db.py (a bare
    # `.TrackNo` sort would raise TypeError if any row's TrackNo is
    # None, comparing None against int).
    rows = list(db.get_playlist_songs(PlaylistID=playlist_id))
    rows.sort(key=lambda r: r.TrackNo or 0)

    # Grouped by content ID, not a flat id -> row dict: a track can
    # legally appear more than once in one playlist (see
    # resolve_playlist_entry above). A flat dict would silently collapse
    # two rows sharing a content ID into one, permanently losing track
    # of the other -- a real corrupted-order bug, not just a style
    # choice. Each queue is consumed in original TrackNo order as that
    # content ID's occurrences are encountered in ordered_content_ids
    # below, so N occurrences of the same content ID map 1:1 onto that
    # content ID's N real rows.
    rows_by_content_id: dict = defaultdict(deque)
    for row in rows:
        rows_by_content_id[str(row.ContentID)].append(row)

    target_ids = [str(cid) for cid in ordered_content_ids]
    current_ids = [str(row.ContentID) for row in rows]
    if sorted(target_ids) != sorted(current_ids):
        raise ValueError(
            f"ordered_content_ids does not match playlist {playlist_id}'s current tracks "
            "exactly -- the playlist changed since this order was computed"
        )

    backup_database(db.db_directory / "master.db")

    try:
        for target_track_no, content_id in enumerate(target_ids, start=1):
            row = rows_by_content_id[content_id].popleft()
            # CRITICAL: never call move_song_in_playlist when the track
            # is already at its target position. pyrekordbox's own
            # move_song_in_playlist (db6/database.py) has a confirmed
            # bug on exactly this no-op path: it unconditionally calls
            # self.registry.disable_tracking(), then for
            # new_track_no == old_track_no hits a bare `return` BEFORE
            # its own self.registry.enable_tracking() call ever runs --
            # and RekordboxAgentRegistry.__enabled__ is a CLASS-level
            # attribute (db6/registry.py), not per-call or per-instance.
            # Triggering this even once silently disables change-
            # tracking for every subsequent write on this process's db
            # connection (rows keep getting written, but stop getting
            # their rb_local_usn bumped by commit()'s own
            # autoincrement_local_update_count() step). Do not
            # "simplify" this guard away -- it's what prevents a
            # totally silent, hard-to-diagnose USN corruption later in
            # this same process.
            if row.TrackNo != target_track_no:
                db.move_song_in_playlist(playlist, row, target_track_no)
        db.commit()
    except RuntimeError as e:
        db.rollback()
        raise RekordboxRunningError(str(e)) from e
    except Exception:
        db.rollback()
        raise


# --- My Tags (smart tagging) ------------------------------------------------
#
# Row formats copied from rows Rekordbox itself wrote (the plan's Phase 0
# snapshot diff), not guessed:
# - djmdSongMyTag link: ID and UUID are two different uuid4 strings,
#   TrackNo is NULL, and rb_local_usn comes from the shared
#   localUpdateCount counter -- which pyrekordbox's commit() assigns to
#   every row passed through db.add() (same mechanism as the cue writes
#   above). Tagging a track left its djmdContent row untouched.
# - djmdMyTag tag: ID is a random integer string above 2^28 (Rekordbox's
#   own were e.g. '2938978579'), UUID a uuid4, Attribute=0, ParentID the
#   column ('1'..'4'), Seq the next position in that column.
# Rekordbox allows exactly 4 columns; djcues repurposes them (see
# tagging.REKORDBOX_COLUMNS) rather than adding any.


class TagWriteError(PlaylistWriteError):
    """A My Tag write couldn't proceed (missing column, unknown tag...)."""


def plan_tag_columns(db) -> list[dict]:
    """What ensure_tag_columns() would do, without staging anything --
    so a dry run can show it, from the very same logic the real write
    uses (ensure_tag_columns() executes this plan; the two can't drift).

    One dict per column: {"column": row, "catalog": [djcues tag names
    for this column], "rename_to": str | None, "remove": [default tag
    rows with no tracks], "create": [catalog tag names missing],
    "kept": [rows staying, in Seq order]}.

    Removal is deliberately narrow: only names in
    tagging.REKORDBOX_DEFAULT_TAGS for that column, and only while they
    have zero links. A default tag on even one track, and every tag the
    user created, is left exactly as it is.
    """
    from djcues.tagging import REKORDBOX_COLUMNS, REKORDBOX_DEFAULT_TAGS, TAG_CATALOG

    rows = list(db.get_my_tag())
    by_id = {str(r.ID): r for r in rows}
    linked = {str(link.MyTagID) for link in db.get_my_tag_songs()}
    tags_by_name = {cat.name: cat.tags for cat in TAG_CATALOG}

    plan = []
    for col in REKORDBOX_COLUMNS:
        column = by_id.get(col.category_id)
        if column is None or column.Attribute != 1:
            raise TagWriteError(
                f"My Tag column {col.category_id} ({col.default_name}) not found in this library"
            )
        children = sorted(
            (r for r in rows if str(r.ParentID) == col.category_id and r.Attribute == 0),
            key=lambda r: (r.Seq or 0),
        )
        defaults = set(REKORDBOX_DEFAULT_TAGS.get(col.category_id, ()))
        remove = [r for r in children if r.Name in defaults and str(r.ID) not in linked]
        kept = [r for r in children if r not in remove]
        present = {r.Name for r in kept}
        catalog = [t for c in col.categories for t in tags_by_name[c]]
        create = [t for t in catalog if t not in present]
        plan.append({
            "column": column,
            "catalog": catalog,
            "rename_to": col.name if column.Name != col.name else None,
            "remove": remove,
            "create": create,
            "kept": kept,
        })
    return plan


def ensure_tag_columns(db) -> dict[str, str]:
    """Set up Rekordbox's 4 My Tag columns for djcues (see
    plan_tag_columns() for exactly what changes). Stages only -- the
    caller commits. Returns {tag name: My Tag ID}.

    Per column: rename it (Genre -> Energy, ...), remove the unused
    Rekordbox default tags in it (the user agreed to this when choosing
    to repurpose the columns), create missing catalog tags, then
    renumber Seq gap-free. Rekordbox itself hard-deletes a tag row (no
    rb_local_deleted flag) -- confirmed by a before/after diff of the
    user deleting one in the app -- which is what db.delete() does, and
    renumbered the remaining playlists' Seq gap-free after a playlist
    delete in that same diff.
    """
    from pyrekordbox.db6 import tables

    ids: dict[str, str] = {}
    for col in plan_tag_columns(db):
        column = col["column"]
        if col["rename_to"]:
            column.Name = col["rename_to"]
        for row in col["remove"]:
            db.delete(row)
        kept = list(col["kept"])
        for tag in col["create"]:
            row = tables.DjmdMyTag.create(
                ID=str(db.generate_unused_id(tables.DjmdMyTag, is_28_bit=False)),
                Seq=len(kept) + 1,
                Name=tag,
                Attribute=0,
                ParentID=str(column.ID),
                UUID=str(uuid4()),
            )
            db.add(row)
            db.flush()  # so the next generate_unused_id() sees this ID as taken
            kept.append(row)
        catalog = set(col["catalog"])
        for seq, row in enumerate(kept, start=1):
            if row.Seq != seq:
                row.Seq = seq
            # Only djcues's own tags -- a user tag that happens to share a
            # catalog name in another column must never be mistaken for it.
            if row.Name in catalog:
                ids[row.Name] = str(row.ID)
    return ids


def apply_tag_changes(changes: list[dict], *, db=None) -> dict:
    """Write My Tag changes in ONE transaction: all tracks or none.

    changes: [{"content_id": str, "add": [tag names], "remove_link_ids":
    [djmdSongMyTag IDs]}]. Checks Rekordbox is closed, backs up
    master.db, ensures the 4 columns are set up, then adds/removes links.
    Returns {"added": n, "removed": n, "backup": path}.
    """
    from pyrekordbox.db6 import tables

    from djcues.db import get_db

    ensure_rekordbox_closed()
    db = db if db is not None else get_db()

    # Validate before the backup -- nothing to undo yet if this raises.
    for change in changes:
        if db.get_content(ID=change["content_id"]) is None:
            raise ValueError(f"track {change['content_id']} not found")

    backup_path = backup_database(db.db_directory / "master.db")
    added = removed = 0
    try:
        ids = ensure_tag_columns(db)
        for change in changes:
            for link_id in change.get("remove_link_ids", []):
                link = db.query(tables.DjmdSongMyTag).filter_by(ID=link_id).one_or_none()
                if link is not None:
                    db.delete(link)
                    removed += 1
            for tag in change.get("add", []):
                if tag not in ids:
                    raise TagWriteError(f"unknown tag {tag!r}")
                db.add(tables.DjmdSongMyTag.create(
                    ID=str(uuid4()),
                    MyTagID=ids[tag],
                    ContentID=str(change["content_id"]),
                    TrackNo=None,
                    UUID=str(uuid4()),
                ))
                added += 1
        db.commit()
    except RuntimeError as e:
        db.rollback()
        raise RekordboxRunningError(str(e)) from e
    except Exception:
        db.rollback()
        raise
    return {"added": added, "removed": removed, "backup": backup_path}

# --- Intelligent Playlists from djcues tags ----------------------------------


def create_tag_smart_playlist(
    name: str, conditions: list, match: int, *, folder_id=None, db=None,
):
    """Create a Rekordbox Intelligent Playlist selecting tracks by djcues
    My Tags (tag_smartlist.TagCondition list, match ALL/ANY).

    Same envelope as the other playlist writes: Rekordbox must be closed,
    master.db is backed up first, one commit, rollback on any failure.
    pyrekordbox creates the playlist row and registers it in
    masterPlaylists6.xml (which Rekordbox 7 keeps alongside master.db and
    which this backs up too, since creating a playlist changes it); the
    SmartList XML is then replaced with tag_smartlist.build_smartlist_xml()
    because pyrekordbox's own XML doesn't match what Rekordbox writes
    (see that module). Appended at the end of its folder, so no other
    playlist is renumbered. Refuses a name already used in that folder.
    Returns the new playlist's ID.
    """
    from pyrekordbox.db6.smartlist import SmartList

    from djcues.db import get_db
    from djcues.tag_smartlist import build_smartlist_xml

    ensure_rekordbox_closed()
    db = db if db is not None else get_db()

    parent_id = "root" if folder_id is None else str(folder_id)
    if folder_id is not None:
        folder = db.get_playlist(ID=parent_id)
        if folder is None or folder.Attribute != 1:
            raise ValueError(f"{folder_id} is not a playlist folder")
    if any(p.Name == name for p in db.get_playlist(ParentID=parent_id)):
        raise ValueError(f"a playlist named {name!r} already exists there")
    build_smartlist_xml("1", conditions, match)  # raises on bad input before any backup

    backup_database(db.db_directory / "master.db")
    playlists_xml = db.db_directory / "masterPlaylists6.xml"
    if playlists_xml.exists():
        shutil.copy2(playlists_xml, playlists_xml.with_name(
            f"masterPlaylists6-backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}.xml"
        ))

    try:
        smart = SmartList(logical_operator=match)
        for c in conditions:
            smart.add_condition("myTag", 8 if c.present else 9, value_left=str(c.tag_id))
        playlist = db.create_smart_playlist(
            name, smart, parent=None if folder_id is None else parent_id
        )
        playlist.SmartList = build_smartlist_xml(playlist.ID, conditions, match)
        playlist_id = str(playlist.ID)
        db.commit()
    except RuntimeError as e:
        db.rollback()
        raise RekordboxRunningError(str(e)) from e
    except Exception:
        db.rollback()
        raise
    return playlist_id


# --- Realign First Beat / Loop In cues to bar 1.1 ------------------------
#
# Before the bar-1.1 change (see models.BeatGrid), djcues put the First
# Beat (A) and Loop In (B) cues on the grid's first beat -- 1-3 beats off
# Rekordbox's own 1.1 on tracks whose grid starts mid-bar. This moves
# those already-written cues; new proposals are already correct.


def plan_bar_one_realignment(db, tolerance_ms: int = 2) -> list[dict]:
    """Existing djcues cues that sit off bar 1.1.

    Targets every "First Beat" cue (hot A and memory), plus any "Loop In"
    cue that sits exactly with a First Beat cue of the same track (djcues
    always places them together; a Loop In you moved yourself is left
    alone). A loop keeps its length. Read-only.
    """
    from collections import defaultdict

    from djcues.db import _extract_beat_grid

    by_track: dict[str, list] = defaultdict(list)
    for cue in db.get_cue():
        by_track[str(cue.ContentID)].append(cue)

    changes: list[dict] = []
    for content_id, cues in by_track.items():
        first_beats = [c for c in cues if c.Comment == "First Beat"]
        if not first_beats:
            continue
        content = db.get_content(ID=int(content_id))
        try:
            grid = _extract_beat_grid(content, db)
        except Exception:
            continue
        if grid.bpm <= 0 or grid.first_downbeat_beat == 1:
            continue
        bar_one = int(round(grid.bar_one_ms))
        anchors = [c.InMsec for c in first_beats]
        targets = list(first_beats) + [
            c for c in cues
            if c.Comment == "Loop In" and any(abs(c.InMsec - a) <= tolerance_ms for a in anchors)
        ]
        for cue in targets:
            if abs(cue.InMsec - bar_one) <= tolerance_ms:
                continue
            new_out = cue.OutMsec
            if cue.OutMsec is not None and cue.OutMsec > 0:
                new_out = bar_one + (cue.OutMsec - cue.InMsec)
            changes.append({
                "cue": cue, "title": content.Title, "label": cue.Comment,
                "hot": cue.Kind > 0, "old_in": cue.InMsec, "new_in": bar_one,
                "old_out": cue.OutMsec, "new_out": new_out,
            })
    return changes


def apply_bar_one_realignment(changes: list[dict], *, db) -> int:
    """Write plan_bar_one_realignment's changes in place (cue IDs kept):
    closed check -> backup -> one transaction -> rollback on any failure.
    Returns the number of cues moved."""
    if not changes:
        return 0
    ensure_rekordbox_closed()
    backup_path = backup_database(db.db_directory / "master.db")
    try:
        for c in changes:
            c["cue"].InMsec = c["new_in"]
            c["cue"].OutMsec = c["new_out"]
        db.commit()
    except RuntimeError as e:
        db.rollback()
        raise RekordboxRunningError(str(e)) from e
    except Exception:
        db.rollback()
        raise
    return len(changes)


# --- Re-place Loop Out cues with the drums-no-vocals search --------------
#
# Moves Loop Out cues djcues wrote under the old rule (Outro marker, 4
# bars) to loop_out.find_loop_out()'s choice. Conservative on purpose: a
# loop you moved or resized yourself, and tracks where the search found no
# clean drums-only stretch, are left exactly as they are.


def plan_loop_out_realignment(db, under: str | None = None, tolerance_ms: int = 2) -> dict:
    """{"changes": [...], "skipped": {reason: [title, ...]}}. Read-only.

    A track qualifies when its hot and memory "Loop Out" cues still sit
    exactly where the old rule put them (the first Outro phrase's start,
    or the last phrase's if there is none; 4 bars long).
    """
    from collections import defaultdict

    from djcues.db import load_track
    from djcues.loop_out import find_loop_out

    by_track: dict[str, list] = defaultdict(list)
    for cue in db.get_cue():
        if cue.Comment == "Loop Out":
            by_track[str(cue.ContentID)].append(cue)

    changes: list[dict] = []
    skipped: dict[str, list[str]] = defaultdict(list)
    for content_id, cues in by_track.items():
        content = db.get_content(ID=int(content_id))
        if under is not None and not _is_under(content.FolderPath, under):
            continue
        title = content.Title
        try:
            track = load_track(content, db)
        except Exception:
            skipped["unreadable analysis"].append(title)
            continue
        if not track.phrases or track.beat_grid.bpm <= 0:
            skipped["no phrase/tempo data"].append(title)
            continue
        outros = [p for p in track.phrases if p.label == "Outro"]
        old_start = (outros[0] if outros else track.phrases[-1]).position_ms
        old_len = track.beat_grid.bars_to_ms(4)
        pristine = all(
            abs(c.InMsec - old_start) <= tolerance_ms
            and c.OutMsec is not None and abs((c.OutMsec - c.InMsec) - old_len) <= tolerance_ms + 1
            for c in cues
        )
        if not pristine:
            skipped["hand-edited or already re-placed"].append(title)
            continue
        choice = find_loop_out(track, old_start)
        if choice is None:
            skipped["no waveform to search"].append(title)
            continue
        if not choice.clean:
            skipped["no clean drums-only stretch"].append(title)
            continue
        new_in = int(round(choice.start_ms))
        new_out = int(round(choice.start_ms + track.beat_grid.bars_to_ms(choice.bars)))
        if abs(new_in - cues[0].InMsec) <= tolerance_ms and choice.bars == 4:
            skipped["already the best loop"].append(title)
            continue
        for cue in cues:
            changes.append({
                "cue": cue, "title": title, "hot": cue.Kind > 0, "bars": choice.bars,
                "old_in": cue.InMsec, "new_in": new_in,
                "old_out": cue.OutMsec, "new_out": new_out, "note": choice.note,
            })
    return {"changes": changes, "skipped": dict(skipped)}


def _is_under(path: str | None, folder: str) -> bool:
    import os

    if not path:
        return False
    base = os.path.normcase(os.path.normpath(folder)).rstrip(os.sep) + os.sep
    return os.path.normcase(os.path.normpath(path)).startswith(base)


def apply_loop_out_realignment(changes: list[dict], *, db) -> int:
    """Write plan_loop_out_realignment's changes in place (cue IDs kept):
    closed check -> backup -> one transaction -> rollback on any failure.
    Returns the number of cues moved."""
    if not changes:
        return 0
    ensure_rekordbox_closed()
    backup_database(db.db_directory / "master.db")
    try:
        for c in changes:
            c["cue"].InMsec = c["new_in"]
            c["cue"].OutMsec = c["new_out"]
        db.commit()
    except RuntimeError as e:
        db.rollback()
        raise RekordboxRunningError(str(e)) from e
    except Exception:
        db.rollback()
        raise
    return len(changes)
