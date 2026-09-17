from unittest.mock import MagicMock, patch

import pytest
from tests.conftest import requires_rekordbox


@requires_rekordbox
def test_find_playlist():
    from djcues.db import find_playlist
    pl = find_playlist("Tech House")
    assert pl is not None
    assert pl.Name == "Tech House"


@requires_rekordbox
def test_load_playlist_tracks():
    from djcues.db import find_playlist, load_playlist_tracks
    pl = find_playlist("Tech House")
    tracks = load_playlist_tracks(pl.ID)
    assert len(tracks) > 0
    first = tracks[0]
    assert first.title is not None
    assert first.bpm > 0


@requires_rekordbox
def test_track_has_phrases():
    from djcues.db import find_playlist, load_playlist_tracks
    pl = find_playlist("Tech House")
    tracks = load_playlist_tracks(pl.ID)
    # At least some tracks should have phrases
    tracks_with_phrases = [t for t in tracks if len(t.phrases) > 0]
    assert len(tracks_with_phrases) > 0


@requires_rekordbox
def test_track_has_cues():
    from djcues.db import find_playlist, load_playlist_tracks
    pl = find_playlist("Tech House")
    tracks = load_playlist_tracks(pl.ID)
    # Tech House tracks should all have cues
    for t in tracks:
        assert len(t.cues) > 0, f"{t.title} has no cues"


@requires_rekordbox
def test_track_beat_grid():
    from djcues.db import find_playlist, load_playlist_tracks
    pl = find_playlist("Tech House")
    tracks = load_playlist_tracks(pl.ID)
    first = tracks[0]
    assert first.beat_grid.bpm > 0
    assert first.beat_grid.first_beat_ms >= 0


# ---------------------------------------------------------------------------
# Explicit db= param -- the dashboard's whole thread-safety design leans on
# every DB-touching function accepting its own connection instead of always
# reaching for the shared get_db() singleton (unsafe to share across
# threads -- see server.py's _DbWorker docstring for why). These are direct,
# mockable regression guards for that specific contract.
# ---------------------------------------------------------------------------


class TestExplicitDbParam:
    def test_find_playlist_uses_shared_db_when_omitted(self):
        from djcues.db import find_playlist

        fake_db = MagicMock()
        fake_db.get_playlist.return_value = []
        with patch("djcues.db.get_db", return_value=fake_db) as mock_get_db:
            find_playlist("Nonexistent")
        mock_get_db.assert_called_once()

    def test_find_playlist_skips_shared_db_when_given(self):
        from djcues.db import find_playlist

        fake_db = MagicMock()
        fake_db.get_playlist.return_value = []
        with patch("djcues.db.get_db") as mock_get_db:
            find_playlist("Nonexistent", db=fake_db)
        mock_get_db.assert_not_called()
        fake_db.get_playlist.assert_called_once()

    def test_load_playlist_tracks_skips_shared_db_when_given(self):
        from djcues.db import load_playlist_tracks

        fake_db = MagicMock()
        fake_db.get_playlist_songs.return_value = []
        with patch("djcues.db.get_db") as mock_get_db:
            load_playlist_tracks("123", db=fake_db)
        mock_get_db.assert_not_called()
        fake_db.get_playlist_songs.assert_called_once_with(PlaylistID="123")

    def test_build_playlist_tree_skips_shared_db_when_given(self):
        from djcues.db import build_playlist_tree

        fake_db = MagicMock()
        fake_db.get_playlist.return_value = []
        with patch("djcues.db.get_db") as mock_get_db:
            build_playlist_tree(db=fake_db)
        mock_get_db.assert_not_called()

    def test_list_playlist_tracks_skips_shared_db_when_given(self):
        from djcues.db import list_playlist_tracks

        fake_db = MagicMock()
        fake_db.get_playlist_songs.return_value = []
        with patch("djcues.db.get_db") as mock_get_db:
            list_playlist_tracks("123", db=fake_db)
        mock_get_db.assert_not_called()

    def test_find_playlist_song_entries_skips_shared_db_when_given(self):
        from djcues.db import find_playlist_song_entries

        fake_db = MagicMock()
        fake_db.get_playlist_songs.return_value = []
        with patch("djcues.db.get_db") as mock_get_db:
            find_playlist_song_entries("pl1", "c1", db=fake_db)
        mock_get_db.assert_not_called()
        fake_db.get_playlist_songs.assert_called_once_with(PlaylistID="pl1", ContentID="c1")


# ---------------------------------------------------------------------------
# build_playlist_tree / list_playlist_tracks -- the cheap, ANLZ-free
# browsing path the dashboard uses. Synthetic MagicMock playlist/song rows,
# since real pyrekordbox ORM objects aren't easy to construct directly.
# ---------------------------------------------------------------------------


def _mock_playlist_row(id_, name, attribute, parent_id, seq=1, song_count=0):
    row = MagicMock()
    row.ID = id_
    row.Name = name
    row.Attribute = attribute
    row.ParentID = parent_id
    row.Seq = seq
    row.Songs = [MagicMock() for _ in range(song_count)]
    return row


class TestBuildPlaylistTree:
    def test_flat_library_no_folders(self):
        from djcues.db import build_playlist_tree

        rows = [
            _mock_playlist_row("1", "Tech House", 0, "root", seq=2, song_count=10),
            _mock_playlist_row("2", "House", 0, "root", seq=1, song_count=50),
        ]
        fake_db = MagicMock()
        fake_db.get_playlist.return_value = rows

        tree = build_playlist_tree(db=fake_db)

        assert [n.name for n in tree] == ["House", "Tech House"]  # sorted by Seq
        assert tree[0].kind == "playlist"
        assert tree[0].track_count == 50
        assert tree[1].track_count == 10
        assert tree[0].children == []

    def test_real_folder_hierarchy_via_parent_id(self):
        from djcues.db import build_playlist_tree

        rows = [
            _mock_playlist_row("folder1", "2024 Sets", 1, "root", seq=1, song_count=0),
            _mock_playlist_row("child1", "Set A", 0, "folder1", seq=1, song_count=5),
            _mock_playlist_row("child2", "Set B", 0, "folder1", seq=2, song_count=8),
        ]
        fake_db = MagicMock()
        fake_db.get_playlist.return_value = rows

        tree = build_playlist_tree(db=fake_db)

        assert len(tree) == 1
        folder = tree[0]
        assert folder.kind == "folder"
        assert folder.track_count is None  # folders never report a track count
        assert [c.name for c in folder.children] == ["Set A", "Set B"]
        assert folder.children[0].track_count == 5

    def test_smart_playlist_classified_but_not_browsable(self):
        from djcues.db import build_playlist_tree

        rows = [_mock_playlist_row("1", "Recently Added", 4, "root", song_count=0)]
        fake_db = MagicMock()
        fake_db.get_playlist.return_value = rows

        tree = build_playlist_tree(db=fake_db)

        assert tree[0].kind == "smart_playlist"

    def test_empty_library_returns_empty_tree(self):
        from djcues.db import build_playlist_tree

        fake_db = MagicMock()
        fake_db.get_playlist.return_value = []

        assert build_playlist_tree(db=fake_db) == []


class TestListPlaylistTracks:
    def test_maps_songs_to_track_summaries_in_order(self):
        from djcues.db import list_playlist_tracks

        def _mock_song(track_no, title, artist_name, bpm, length):
            song = MagicMock()
            song.TrackNo = track_no
            content = MagicMock()
            content.ID = f"content-{track_no}"
            content.Title = title
            content.Artist.Name = artist_name
            content.BPM = bpm * 100
            content.Length = length
            song.Content = content
            return song

        songs = [_mock_song(2, "Second", "Artist B", 128.0, 200), _mock_song(1, "First", "Artist A", 126.0, 180)]
        fake_db = MagicMock()
        fake_db.get_playlist_songs.return_value = songs

        tracks = list_playlist_tracks("playlist1", db=fake_db)

        assert [t.title for t in tracks] == ["First", "Second"]  # sorted by TrackNo
        assert tracks[0].bpm == 126.0
        assert tracks[0].duration_ms == 180_000.0
        assert tracks[0].artist == "Artist A"

    def test_skips_orphaned_song_rows_with_no_content(self):
        from djcues.db import list_playlist_tracks

        orphan = MagicMock()
        orphan.TrackNo = 1
        orphan.Content = None
        fake_db = MagicMock()
        fake_db.get_playlist_songs.return_value = [orphan]

        assert list_playlist_tracks("playlist1", db=fake_db) == []

    def test_handles_missing_artist(self):
        from djcues.db import list_playlist_tracks

        song = MagicMock()
        song.TrackNo = 1
        song.Content.ID = "c1"
        song.Content.Title = "No Artist Track"
        song.Content.Artist = None
        song.Content.BPM = 12800
        song.Content.Length = 100
        fake_db = MagicMock()
        fake_db.get_playlist_songs.return_value = [song]

        tracks = list_playlist_tracks("playlist1", db=fake_db)

        assert tracks[0].artist == ""

    def test_key_scale_name_flows_into_key_field(self):
        from djcues.db import list_playlist_tracks

        song = MagicMock()
        song.TrackNo = 1
        song.Content.ID = "c1"
        song.Content.Title = "Track"
        song.Content.Artist = None
        song.Content.BPM = 12800
        song.Content.Length = 100
        song.Content.Key.ScaleName = "9A"
        fake_db = MagicMock()
        fake_db.get_playlist_songs.return_value = [song]

        tracks = list_playlist_tracks("playlist1", db=fake_db)

        assert tracks[0].key == "9A"

    def test_missing_key_relationship_is_none(self):
        from djcues.db import list_playlist_tracks

        song = MagicMock()
        song.TrackNo = 1
        song.Content.ID = "c1"
        song.Content.Title = "Track"
        song.Content.Artist = None
        song.Content.BPM = 12800
        song.Content.Length = 100
        song.Content.Key = None
        fake_db = MagicMock()
        fake_db.get_playlist_songs.return_value = [song]

        tracks = list_playlist_tracks("playlist1", db=fake_db)

        assert tracks[0].key is None

    def test_commnt_flows_into_comment_field(self):
        from djcues.db import list_playlist_tracks

        song = MagicMock()
        song.TrackNo = 1
        song.Content.ID = "c1"
        song.Content.Title = "Track"
        song.Content.Artist = None
        song.Content.BPM = 12800
        song.Content.Length = 100
        song.Content.Key = None
        song.Content.Commnt = "2A - 118"
        fake_db = MagicMock()
        fake_db.get_playlist_songs.return_value = [song]

        tracks = list_playlist_tracks("playlist1", db=fake_db)

        assert tracks[0].comment == "2A - 118"

    def test_missing_or_empty_commnt_is_none(self):
        from djcues.db import list_playlist_tracks

        def _song(commnt):
            song = MagicMock()
            song.TrackNo = 1
            song.Content.ID = "c1"
            song.Content.Title = "Track"
            song.Content.Artist = None
            song.Content.BPM = 12800
            song.Content.Length = 100
            song.Content.Key = None
            song.Content.Commnt = commnt
            return song

        fake_db = MagicMock()
        fake_db.get_playlist_songs.return_value = [_song(None)]
        assert list_playlist_tracks("playlist1", db=fake_db)[0].comment is None

        fake_db.get_playlist_songs.return_value = [_song("")]
        assert list_playlist_tracks("playlist1", db=fake_db)[0].comment is None


class TestListAllTracks:
    def test_maps_content_rows_to_track_summaries_with_track_no_always_none(self):
        from djcues.db import list_all_tracks

        def _mock_content(id_, title, bpm, key_scale, comment=None):
            content = MagicMock()
            content.ID = id_
            content.Title = title
            content.Artist.Name = "Some Artist"
            content.BPM = bpm
            content.Length = 200
            content.Key.ScaleName = key_scale
            content.Commnt = comment
            return content

        rows = [
            _mock_content("1", "Track A", 12800, "9A", comment="9A - 128"),
            _mock_content("2", "Track B", 17400, "8B"),
        ]
        fake_db = MagicMock()
        fake_db.get_content.return_value = rows

        tracks = list_all_tracks(db=fake_db)

        assert [t.title for t in tracks] == ["Track A", "Track B"]
        assert [t.key for t in tracks] == ["9A", "8B"]
        assert [t.comment for t in tracks] == ["9A - 128", None]
        assert all(t.track_no is None for t in tracks)
        fake_db.get_content.assert_called_once_with()

    def test_skips_nothing_extra_missing_key_is_none(self):
        from djcues.db import list_all_tracks

        content = MagicMock()
        content.ID = "1"
        content.Title = "Track"
        content.Artist = None
        content.BPM = 12800
        content.Length = 100
        content.Key = None
        fake_db = MagicMock()
        fake_db.get_content.return_value = [content]

        tracks = list_all_tracks(db=fake_db)

        assert tracks[0].key is None

    def test_skips_shared_db_when_given(self):
        from djcues.db import list_all_tracks

        fake_db = MagicMock()
        fake_db.get_content.return_value = []
        with patch("djcues.db.get_db") as mock_get_db:
            list_all_tracks(db=fake_db)
        mock_get_db.assert_not_called()


class TestFindPlaylistSongEntries:
    def test_sorts_by_track_no(self):
        from djcues.db import find_playlist_song_entries

        e1, e2 = MagicMock(TrackNo=5), MagicMock(TrackNo=2)
        fake_db = MagicMock()
        fake_db.get_playlist_songs.return_value = [e1, e2]

        entries = find_playlist_song_entries("pl1", "c1", db=fake_db)

        assert entries == [e2, e1]
        fake_db.get_playlist_songs.assert_called_once_with(PlaylistID="pl1", ContentID="c1")

    def test_no_match_returns_empty_list(self):
        from djcues.db import find_playlist_song_entries

        fake_db = MagicMock()
        fake_db.get_playlist_songs.return_value = []

        assert find_playlist_song_entries("pl1", "c1", db=fake_db) == []

    def test_more_than_one_match_is_legal_and_returns_all(self):
        # The same track added to one playlist twice -- unusual but real,
        # and the reason writer.py's resolve_playlist_entry() needs a
        # --position to disambiguate rather than this function picking one.
        from djcues.db import find_playlist_song_entries

        e1, e2 = MagicMock(TrackNo=1), MagicMock(TrackNo=3)
        fake_db = MagicMock()
        fake_db.get_playlist_songs.return_value = [e1, e2]

        assert find_playlist_song_entries("pl1", "c1", db=fake_db) == [e1, e2]


@requires_rekordbox
def test_build_playlist_tree_real_library_contains_tech_house():
    from djcues.db import build_playlist_tree

    tree = build_playlist_tree()
    names = [n.name for n in tree]
    assert "Tech House" in names
    tech_house = next(n for n in tree if n.name == "Tech House")
    assert tech_house.kind == "playlist"
    # Not a hardcoded count -- this real library's playlists change over
    # time (e.g. a real djcues playlist move/add/remove), confirmed live
    # when this exact assertion broke after moving a real track into
    # Tech House earlier in this project's history.
    assert tech_house.track_count > 0


@requires_rekordbox
def test_list_playlist_tracks_real_library_matches_load_playlist_tracks():
    from djcues.db import find_playlist, list_playlist_tracks, load_playlist_tracks

    pl = find_playlist("Tech House")
    summaries = list_playlist_tracks(pl.ID)
    full_tracks = load_playlist_tracks(pl.ID)

    assert len(summaries) == len(full_tracks)
    assert {s.title for s in summaries} == {t.title for t in full_tracks}


@requires_rekordbox
def test_list_all_tracks_real_library_has_camelot_parseable_keys():
    from djcues.db import list_all_tracks
    from djcues.harmony import parse_camelot_key

    tracks = list_all_tracks()

    # Structural properties, not hardcoded counts -- the real library's
    # size/coverage can change over time (confirmed non-theoretical
    # above, in test_build_playlist_tree_real_library_contains_tech_house).
    assert len(tracks) > 0
    assert all(t.track_no is None for t in tracks)
    assert any(parse_camelot_key(t.key) is not None for t in tracks)
    assert any(t.comment for t in tracks)
