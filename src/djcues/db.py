"""Read tracks, cues, and phrase data from the rekordbox database."""

from __future__ import annotations

import logging
from typing import Any

from pyrekordbox import Rekordbox6Database

from djcues.constants import resolve_phrase_label
from djcues.models import (
    BeatGrid,
    CuePoint,
    Phrase,
    PlaylistNode,
    RawBeatGridEntry,
    Track,
    TrackSummary,
    WaveformPoint,
)

logger = logging.getLogger(__name__)

_db: Rekordbox6Database | None = None

# Real Attribute values seen on DjmdPlaylist rows for a regular (non-folder,
# non-smart) playlist -- 0 is pyrekordbox's own PlaylistType.PLAYLIST; -128
# is an empirically-observed value this project has seen in practice that
# isn't in that enum. 1=folder, 4=smart playlist (pyrekordbox's PlaylistType).
_FOLDER_ATTRIBUTE = 1
_SMART_PLAYLIST_ATTRIBUTE = 4
# Real root-level ParentID value confirmed against a live library (not None).
_ROOT_PARENT_ID = "root"


def get_db() -> Rekordbox6Database:
    """Get or create the shared database connection.

    This is the single connection every CLI command and existing test
    uses -- safe there because the CLI is single-threaded end to end.
    It is NOT safe to share across threads (see dashboard.py/server.py's
    _DbWorker, which is exactly why every function below now accepts an
    explicit `db=` override instead of always reaching for this global).
    """
    global _db
    if _db is None:
        _db = Rekordbox6Database()
    return _db


def find_playlist(name: str, db: Rekordbox6Database | None = None) -> Any | None:
    """Find a playlist by exact name. Returns the pyrekordbox playlist object or None.

    Prefers actual playlists over folders when names collide.
    Attribute: 0 or -128 = playlist, 1 = folder.
    """
    db = db if db is not None else get_db()
    folder_match = None
    for pl in db.get_playlist():
        if pl.Name == name:
            if pl.Attribute != _FOLDER_ATTRIBUTE:
                return pl  # actual playlist — return immediately
            folder_match = pl  # folder — keep as fallback
    return folder_match


def build_playlist_tree(db: Rekordbox6Database | None = None) -> list[PlaylistNode]:
    """Build the real playlist folder tree via ParentID/Attribute -- nothing
    else in djcues does this (find_playlist() above is a flat name lookup
    only). Cheap: one full DjmdPlaylist table scan, zero ANLZ I/O, zero
    per-track queries (track_count comes from len(Songs), the relationship
    pyrekordbox already eager-loads no more expensively than counting rows
    already in the row object -- confirmed no extra query fires per node
    for a library of this project's real size).

    Folders (Attribute == 1) get children and a None track_count. Smart
    playlists (Attribute == 4) are listed (so they're not silently
    invisible) but djcues can't browse into them the same way -- their
    DjmdSongPlaylist rows generally aren't populated, a pre-existing
    limitation load_playlist_tracks() already has today, not something
    introduced or fixed here. Everything else (0, or the empirically-
    observed -128) is a regular playlist leaf.

    Root-level rows have ParentID == "root" (confirmed against a real
    library, not None) -- not a value pyrekordbox exposes as a constant.
    """
    db = db if db is not None else get_db()
    rows = list(db.get_playlist())

    def kind_of(attribute: int) -> str:
        if attribute == _FOLDER_ATTRIBUTE:
            return "folder"
        if attribute == _SMART_PLAYLIST_ATTRIBUTE:
            return "smart_playlist"
        return "playlist"

    children_by_parent: dict[str, list[Any]] = {}
    for row in rows:
        children_by_parent.setdefault(row.ParentID, []).append(row)

    def build(row: Any) -> PlaylistNode:
        kind = kind_of(row.Attribute)
        child_rows = sorted(children_by_parent.get(row.ID, []), key=lambda r: (r.Seq or 0, r.Name or ""))
        return PlaylistNode(
            id=row.ID,
            name=row.Name or "",
            kind=kind,
            seq=row.Seq or 0,
            track_count=None if kind == "folder" else len(row.Songs),
            children=[build(child) for child in child_rows],
        )

    roots = sorted(children_by_parent.get(_ROOT_PARENT_ID, []), key=lambda r: (r.Seq or 0, r.Name or ""))
    return [build(row) for row in roots]


def list_playlist_tracks(playlist_id: str, db: Rekordbox6Database | None = None) -> list[TrackSummary]:
    """Cheap track listing for one playlist -- title/artist/bpm/duration
    straight from DjmdContent via the DjmdSongPlaylist relationship, zero
    ANLZ I/O (unlike load_playlist_tracks() below, which is what makes a
    full-library enumeration take ~90-100s in practice). Use this for
    browsing; only call load_track()/load_playlist_tracks() once the user
    has picked one specific track to actually analyze.
    """
    db = db if db is not None else get_db()
    songs = list(db.get_playlist_songs(PlaylistID=playlist_id))
    songs.sort(key=lambda s: (s.TrackNo or 0))
    summaries: list[TrackSummary] = []
    for song in songs:
        content = song.Content
        if content is None:
            continue  # a stale/orphaned song row with no matching content
        artist_name = ""
        if content.Artist:
            artist_name = content.Artist.Name or ""
        summaries.append(TrackSummary(
            id=content.ID,
            track_no=song.TrackNo,
            title=content.Title or "",
            artist=artist_name,
            bpm=(content.BPM or 0) / 100,
            duration_ms=float(content.Length or 0) * 1000,
        ))
    return summaries


def _extract_beat_grid(track_content: Any, db: Rekordbox6Database | None = None) -> BeatGrid:
    """Extract BPM and first beat position from a track's analysis files."""
    db = db if db is not None else get_db()
    bpm = track_content.BPM / 100

    # Try to get first beat from ANLZ beat grid
    first_beat_ms = 0.0
    try:
        anlz_files = db.read_anlz_files(track_content)
        for path, af in anlz_files.items():
            if path.suffix == ".DAT":
                for tag in af.tags:
                    if type(tag).__name__ == "PQTZAnlzTag":
                        times = tag.get_times()
                        if len(times) > 0:
                            # times are in seconds, convert to ms
                            first_beat_ms = float(times[0]) * 1000
                        break
    except Exception as e:
        logger.warning("Could not read beat grid for %s: %s", track_content.Title, e)

    return BeatGrid(first_beat_ms=first_beat_ms, bpm=bpm)


def extract_raw_beat_grid(
    track_content: Any, db: Rekordbox6Database | None = None
) -> list[RawBeatGridEntry] | None:
    """Full per-beat array from the PQTZ tag -- beat-in-bar, tempo, and
    timestamp for every beat Rekordbox's own analysis recorded, not just
    the first one _extract_beat_grid() collapses everything down to.
    Same tag, same file, same failure handling as _extract_beat_grid().

    Returns None (not an empty list) if the grid couldn't be read at
    all, so callers can tell "no data available" apart from "a real,
    empty grid" -- not called from load_track()'s hot path, so this
    costs nothing unless a caller actually asks for it.
    """
    db = db if db is not None else get_db()
    try:
        anlz_files = db.read_anlz_files(track_content)
        for path, af in anlz_files.items():
            if path.suffix == ".DAT":
                for tag in af.tags:
                    if type(tag).__name__ == "PQTZAnlzTag":
                        beats = tag.get_beats()
                        bpms = tag.get_bpms()
                        times = tag.get_times()
                        return [
                            RawBeatGridEntry(
                                beat_in_bar=int(b), bpm=float(p), time_ms=float(t) * 1000
                            )
                            for b, p, t in zip(beats, bpms, times)
                        ]
    except Exception as e:
        logger.warning("Could not read raw beat grid for %s: %s", track_content.Title, e)
    return None


def _extract_phrases(
    track_content: Any, beat_grid: BeatGrid, db: Rekordbox6Database | None = None
) -> list[Phrase]:
    """Extract PSSI phrase structure from a track's ANLZ EXT file."""
    db = db if db is not None else get_db()
    phrases: list[Phrase] = []

    try:
        anlz_files = db.read_anlz_files(track_content)
        for path, af in anlz_files.items():
            if path.suffix == ".EXT":
                for tag in af.tags:
                    if type(tag).__name__ == "PSSIAnlzTag":
                        content = tag.content
                        mood = content.mood
                        entries = list(content.entries)

                        for i, entry in enumerate(entries):
                            next_beat = (
                                entries[i + 1].beat
                                if i + 1 < len(entries)
                                else content.end_beat
                            )
                            label = resolve_phrase_label(mood, entry.kind)
                            pos_ms = beat_grid.beat_to_ms(entry.beat)
                            end_ms = beat_grid.beat_to_ms(next_beat)

                            phrases.append(Phrase(
                                beat_start=entry.beat,
                                beat_end=next_beat,
                                kind=entry.kind,
                                label=label,
                                position_ms=pos_ms,
                                duration_ms=end_ms - pos_ms,
                            ))
                        break  # only process first PSSI tag
    except Exception as e:
        logger.warning("Could not read phrases for %s: %s", track_content.Title, e)

    return phrases


def _extract_cues(track_content: Any, db: Rekordbox6Database | None = None) -> list[CuePoint]:
    """Extract cue points from the database."""
    db = db if db is not None else get_db()
    cues: list[CuePoint] = []

    for c in db.get_cue(ContentID=track_content.ID):
        loop_end = None
        if c.OutMsec is not None and c.OutMsec > 0:
            loop_end = float(c.OutMsec)

        cues.append(CuePoint(
            kind=c.Kind,
            position_ms=float(c.InMsec),
            loop_end_ms=loop_end,
            color_table_index=c.ColorTableIndex,
            color=c.Color if c.Color is not None else -1,
            comment=c.Comment or "",
        ))

    return cues


def _extract_waveform(
    track_content: Any, max_points: int = 1200, db: Rekordbox6Database | None = None
) -> list[WaveformPoint] | None:
    """Extract color waveform from PWV5 tag, downsampled for display.

    max_points was 800 -- raised modestly (not aggressively) after a
    real review-UI readability pass: the raw PWV5 source is far denser
    than 800 points for any real track, so this surfaces more real
    signal rather than fabricating detail, but a real generated review
    HTML for a 115-track playlist already runs ~6.9MB at 800 points, so
    the increase is kept conservative rather than maximized.
    """
    db = db if db is not None else get_db()
    try:
        anlz_files = db.read_anlz_files(track_content)
        for path, af in anlz_files.items():
            if path.suffix == ".EXT":
                for tag in af.tags:
                    if type(tag).__name__ == "PWV5AnlzTag":
                        heights, colors = tag.get()
                        n = len(heights)
                        step = max(1, n // max_points)
                        points: list[WaveformPoint] = []
                        for i in range(0, n, step):
                            # Take max height in each chunk for peak representation
                            chunk_end = min(i + step, n)
                            peak_idx = i
                            for j in range(i, chunk_end):
                                if heights[j] > heights[peak_idx]:
                                    peak_idx = j
                            points.append(WaveformPoint(
                                height=float(heights[peak_idx]),
                                red=int(colors[peak_idx, 0]),
                                green=int(colors[peak_idx, 1]),
                                blue=int(colors[peak_idx, 2]),
                            ))
                        return points
    except Exception as e:
        logger.warning("Could not read waveform for %s: %s", track_content.Title, e)
    return None


def _extract_vocal_track(
    track_content: Any, db: Rekordbox6Database | None = None
) -> list[int] | None:
    """Extract PVDI vocal detection data from the .2EX ANLZ file.

    Returns a list of per-frame vocal confidence values (0-4),
    where each frame covers 1024/22050 ≈ 46.4ms.
    """
    import struct
    db = db if db is not None else get_db()
    ex2_path = db.get_anlz_paths(track_content).get("2EX")
    if not ex2_path or not ex2_path.exists():
        return None
    try:
        with open(ex2_path, "rb") as f:
            data = f.read()
        pos = data.find(b"PVDI")
        if pos < 0:
            return None
        header_len = struct.unpack(">I", data[pos + 4 : pos + 8])[0]
        tag_len = struct.unpack(">I", data[pos + 8 : pos + 12])[0]
        body = data[pos + header_len : pos + tag_len]
        return list(body)
    except Exception as e:
        logger.warning("Could not read vocal track for %s: %s", track_content.Title, e)
        return None


def load_track(track_content: Any, db: Rekordbox6Database | None = None) -> Track:
    """Load a single track with all analysis data."""
    db = db if db is not None else get_db()
    beat_grid = _extract_beat_grid(track_content, db=db)
    phrases = _extract_phrases(track_content, beat_grid, db=db)
    cues = _extract_cues(track_content, db=db)
    waveform = _extract_waveform(track_content, db=db)
    vocal_track = _extract_vocal_track(track_content, db=db)

    artist_name = ""
    if track_content.Artist:
        artist_name = track_content.Artist.Name or ""

    return Track(
        id=track_content.ID,
        title=track_content.Title or "",
        artist=artist_name,
        bpm=track_content.BPM / 100,
        duration_ms=float(track_content.Length or 0) * 1000,
        analysis_path=track_content.AnalysisDataPath or "",
        cues=cues,
        phrases=phrases,
        beat_grid=beat_grid,
        waveform=waveform,
        vocal_track=vocal_track,
        audio_path=track_content.FolderPath or None,
    )


def load_playlist_tracks(playlist_id: int, db: Rekordbox6Database | None = None) -> list[Track]:
    """Load all tracks from a playlist with full analysis data."""
    db = db if db is not None else get_db()
    songs = list(db.get_playlist_songs(PlaylistID=playlist_id))
    tracks: list[Track] = []
    for song in songs:
        try:
            tracks.append(load_track(song.Content, db=db))
        except Exception as e:
            logger.warning("Skipping track %s: %s", song.Content.Title, e)
    return tracks
