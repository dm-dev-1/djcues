"""Tests for Intelligent Playlists built from djcues tags
(djcues.tag_smartlist, writer.create_tag_smart_playlist, `tags smartlist`).

The XML is checked byte-for-byte against an Intelligent Playlist the user
created in Rekordbox 7.2.9 itself. The integration test writes to a copy
of the real library in tmp_path -- never the live one."""

from __future__ import annotations

import os
import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from djcues import tags_cli, writer
from djcues.cli import cli
from djcues.models import TagLink
from djcues.tag_smartlist import (
    MATCH_ALL,
    MATCH_ANY,
    TagCondition,
    build_smartlist_xml,
    matching_tracks,
    signed_int32,
)
from tests.conftest import requires_rekordbox, requires_rekordbox_closed

# Copied verbatim from master.db after the user created this playlist in Rekordbox.
REKORDBOX_SAMPLE = (
    '<NODE Id="571261727" LogicalOperator="1" AutomaticUpdate="0">'
    '<CONDITION PropertyName="myTag" Operator="8" ValueUnit="" ValueLeft="-1355988717" ValueRight=""/>'
    "</NODE>"
)


# --- XML --------------------------------------------------------------------


def test_xml_matches_what_rekordbox_wrote_byte_for_byte():
    xml = build_smartlist_xml(571261727, [TagCondition("2938978579", "zz test", True)], MATCH_ALL)
    assert xml == REKORDBOX_SAMPLE


@pytest.mark.parametrize("value,expected", [
    (0, 0), (571261727, 571261727), (2**31 - 1, 2**31 - 1),
    (2**31, -(2**31)), (2938978579, -1355988717), (2**32 - 1, -1),
])
def test_signed_int32(value, expected):
    assert signed_int32(value) == expected
    assert signed_int32(str(value)) == expected


@pytest.mark.parametrize("bad", [-1, 2**32])
def test_signed_int32_rejects_non_32_bit_ids(bad):
    with pytest.raises(ValueError):
        signed_int32(bad)


def test_without_condition_uses_not_contains_and_any_uses_operator_2():
    xml = build_smartlist_xml(5, [
        TagCondition("100", "a", True), TagCondition("200", "b", False),
    ], MATCH_ANY)
    assert 'LogicalOperator="2"' in xml
    assert '<CONDITION PropertyName="myTag" Operator="8" ValueUnit="" ValueLeft="100" ValueRight=""/>' in xml
    assert '<CONDITION PropertyName="myTag" Operator="9" ValueUnit="" ValueLeft="200" ValueRight=""/>' in xml


def test_xml_needs_conditions_and_a_valid_match():
    with pytest.raises(ValueError, match="at least one condition"):
        build_smartlist_xml(1, [], MATCH_ALL)
    with pytest.raises(ValueError, match="match"):
        build_smartlist_xml(1, [TagCondition("1", "a", True)], 3)


# --- matching preview -----------------------------------------------------------


def _link(track, tag_id):
    return TagLink(f"L{track}{tag_id}", str(track), str(tag_id), "n", "1")


_LINKS = [_link(1, "E5"), _link(1, "LI"), _link(2, "E5"), _link(2, "VI"), _link(3, "LI")]
_TRACKS = ["1", "2", "3", "4"]


def test_matching_all():
    conds = [TagCondition("E5", "Energy 5", True), TagCondition("VI", "Vocal Intro", False)]
    assert matching_tracks(_LINKS, conds, MATCH_ALL, _TRACKS) == ["1"]


def test_matching_any():
    conds = [TagCondition("VI", "Vocal Intro", True), TagCondition("LI", "Long Intro", True)]
    assert matching_tracks(_LINKS, conds, MATCH_ANY, _TRACKS) == ["1", "2", "3"]


def test_a_without_condition_matches_untagged_tracks_too():
    conds = [TagCondition("E5", "Energy 5", False)]
    assert matching_tracks(_LINKS, conds, MATCH_ALL, _TRACKS) == ["3", "4"]


# --- writer envelope (MagicMock db) ------------------------------------------------


def _fake_db(tmp_path, existing_names=()):
    db = MagicMock()
    db.db_directory = tmp_path
    (tmp_path / "master.db").write_bytes(b"x")
    db.get_playlist.side_effect = lambda **kw: (
        [SimpleNamespace(Name=n) for n in existing_names] if "ParentID" in kw else None
    )
    return db


_COND = [TagCondition("100", "Energy 5", True)]


def test_create_checks_rekordbox_closed_first(tmp_path):
    db = _fake_db(tmp_path)
    with patch.object(writer, "ensure_rekordbox_closed", side_effect=writer.RekordboxRunningError("running")):
        with pytest.raises(writer.RekordboxRunningError):
            writer.create_tag_smart_playlist("P", _COND, MATCH_ALL, db=db)
    db.create_smart_playlist.assert_not_called()
    assert not list(tmp_path.glob("master-backup-*"))


def test_create_refuses_a_duplicate_name_before_backing_up(tmp_path):
    db = _fake_db(tmp_path, existing_names=["Peak"])
    with patch.object(writer, "ensure_rekordbox_closed"):
        with pytest.raises(ValueError, match="already exists"):
            writer.create_tag_smart_playlist("Peak", _COND, MATCH_ALL, db=db)
    assert not list(tmp_path.glob("master-backup-*"))


def test_create_rejects_a_non_folder_parent(tmp_path):
    db = _fake_db(tmp_path)
    db.get_playlist.side_effect = lambda **kw: SimpleNamespace(Attribute=0) if "ID" in kw else []
    with patch.object(writer, "ensure_rekordbox_closed"):
        with pytest.raises(ValueError, match="not a playlist folder"):
            writer.create_tag_smart_playlist("P", _COND, MATCH_ALL, folder_id="5", db=db)


def test_create_backs_up_both_files_and_rolls_back_on_refused_commit(tmp_path):
    db = _fake_db(tmp_path)
    (tmp_path / "masterPlaylists6.xml").write_text("<x/>", encoding="utf-8")
    db.create_smart_playlist.return_value = SimpleNamespace(ID="42", SmartList=None)
    db.commit.side_effect = RuntimeError("Rekordbox is running")
    with patch.object(writer, "ensure_rekordbox_closed"):
        with pytest.raises(writer.RekordboxRunningError):
            writer.create_tag_smart_playlist("P", _COND, MATCH_ALL, db=db)
    db.rollback.assert_called_once()
    assert list(tmp_path.glob("master-backup-*"))
    assert list(tmp_path.glob("masterPlaylists6-backup-*.xml"))


def test_create_replaces_pyrekordboxs_xml_with_rekordboxs_format(tmp_path):
    db = _fake_db(tmp_path)
    playlist = SimpleNamespace(ID="571261727", SmartList="<pyrekordbox's version/>")
    db.create_smart_playlist.return_value = playlist
    with patch.object(writer, "ensure_rekordbox_closed"):
        playlist_id = writer.create_tag_smart_playlist(
            "P", [TagCondition("2938978579", "zz test", True)], MATCH_ALL, db=db,
        )
    assert playlist_id == "571261727"
    assert playlist.SmartList == REKORDBOX_SAMPLE
    db.commit.assert_called_once()


# --- CLI ---------------------------------------------------------------------------


@pytest.fixture
def cli_env(monkeypatch):
    from djcues import db as db_module

    tag_rows = [
        SimpleNamespace(ID="100", Name="Energy 5", ParentID="1", Attribute=0),
        SimpleNamespace(ID="200", Name="Vocal Intro", ParentID="2", Attribute=0),
    ]
    db = SimpleNamespace(
        get_my_tag=lambda: tag_rows,
        get_content=lambda: [SimpleNamespace(ID="1", Title="Peak Track"), SimpleNamespace(ID="2", Title="Vocal Track")],
    )
    state = SimpleNamespace(created=[], tag_rows=tag_rows)
    monkeypatch.setattr(tags_cli, "get_db", lambda: db)
    monkeypatch.setattr(db_module, "read_tag_links", lambda ids=None, db=None: [
        TagLink("a", "1", "100", "Energy 5", "1"),
        TagLink("b", "2", "100", "Energy 5", "1"), TagLink("c", "2", "200", "Vocal Intro", "2"),
    ])
    monkeypatch.setattr(writer, "create_tag_smart_playlist",
                        lambda name, conds, match, folder_id=None, db=None: state.created.append((name, conds, match)) or "999")
    return state


def _run(*args):
    return CliRunner().invoke(cli, ["tags", "smartlist", *args])


def test_cli_needs_a_condition(cli_env):
    result = _run("P")
    assert result.exit_code == 1 and "at least one --with or --without" in result.output


def test_cli_rejects_without_with_match_any(cli_env):
    result = _run("P", "--with", "Energy 5", "--without", "Vocal Intro", "--match", "any")
    assert result.exit_code == 1 and "--match all" in result.output


def test_cli_rejects_unknown_tags(cli_env):
    result = _run("P", "--with", "Bogus")
    assert result.exit_code == 1 and "unknown tag" in result.output


def test_cli_explains_when_tags_are_not_in_rekordbox_yet(cli_env):
    cli_env.tag_rows.clear()
    result = _run("P", "--with", "Energy 5")
    assert result.exit_code == 1 and "run `djcues tags apply`" in result.output


def test_cli_dry_run_previews_matches_and_writes_nothing(cli_env):
    result = _run("P", "--with", "Energy 5", "--without", "Vocal Intro", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "has 'Energy 5' AND not 'Vocal Intro'" in result.output
    assert "Matches 1 track(s)" in result.output and "Peak Track" in result.output
    assert 'Id="<new playlist id>"' in result.output and 'Operator="9"' in result.output
    assert cli_env.created == []


def test_cli_creates_the_playlist(cli_env):
    result = _run("Peak", "--with", "Energy 5", "--yes")
    assert result.exit_code == 0, result.output
    assert "Created Intelligent Playlist 'Peak' (ID 999)" in result.output
    name, conds, match = cli_env.created[0]
    assert name == "Peak" and match == MATCH_ALL
    assert [(c.tag_id, c.present) for c in conds] == [("100", True)]


def test_cli_declined_creates_nothing(cli_env):
    result = CliRunner().invoke(cli, ["tags", "smartlist", "Peak", "--with", "Energy 5"], input="n\n")
    assert "Aborted" in result.output and cli_env.created == []


# --- integration: a COPY of the real library -------------------------------------------


@requires_rekordbox
@requires_rekordbox_closed
def test_creates_a_working_intelligent_playlist_in_a_library_copy(tmp_path):
    from pyrekordbox.db6 import Rekordbox6Database

    from djcues.db import get_db, read_tag_links

    live_dir = get_db().db_directory
    for name in ("master.db", "masterPlaylists6.xml"):
        if (live_dir / name).exists():
            shutil.copy2(live_dir / name, tmp_path / name)
    db = Rekordbox6Database(path=tmp_path / "master.db")
    try:
        # Make the tags and links this test needs, so it doesn't depend on
        # whatever the live library happens to have tagged.
        contents = [str(c.ID) for c in db.get_content()][:3]
        writer.apply_tag_changes([
            {"content_id": contents[0], "add": ["Energy 5"]},
            {"content_id": contents[1], "add": ["Energy 5", "Vocal Intro"]},
            {"content_id": contents[2], "add": ["Vocal Intro"]},
        ], db=db)
        ids = {t.Name: str(t.ID) for t in db.get_my_tag() if t.Attribute == 0}
        conds = [TagCondition(ids["Energy 5"], "Energy 5", True),
                 TagCondition(ids["Vocal Intro"], "Vocal Intro", False)]
        expected = matching_tracks(read_tag_links(db=db), conds, MATCH_ALL, [str(c.ID) for c in db.get_content()])
        n_root = db.get_playlist(ParentID="root").count()

        playlist_id = writer.create_tag_smart_playlist("djcues test", conds, MATCH_ALL, db=db)
        db.close()

        db = Rekordbox6Database(path=tmp_path / "master.db")
        playlist = db.get_playlist(ID=playlist_id)
        assert playlist.Attribute == 4 and playlist.ParentID == "root"
        assert playlist.Seq == n_root + 1  # appended; nothing else renumbered
        assert playlist.SmartList == build_smartlist_xml(playlist_id, conds, MATCH_ALL)
        # pyrekordbox evaluates the stored conditions independently of
        # djcues's preview -- the two must agree. (The copy may already
        # carry the live library's own tags, so other tracks can match too.)
        selected = sorted(str(c.ID) for c in db.get_playlist_contents(playlist))
        assert selected == sorted(expected)
        assert contents[0] in selected      # Energy 5, no Vocal Intro
        assert contents[1] not in selected  # Energy 5 but has Vocal Intro
        assert contents[2] not in selected  # no Energy 5
        if (tmp_path / "masterPlaylists6.xml").exists():
            xml_text = (tmp_path / "masterPlaylists6.xml").read_text(encoding="utf-8")
            assert f'Id="{int(playlist_id):X}"' in xml_text
    finally:
        if db.session is not None:
            db.close()
