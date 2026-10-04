"""Tests for djcues.tag_apply (deciding tag changes without ever
overwriting the user's edits) and writer's My Tag functions.

Everything except the last class uses fakes. TestTagWriteIntegration
writes to a *copy* of the real Rekordbox database in tmp_path -- never
the live one."""

from __future__ import annotations

import re
import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from djcues import tag_apply, writer
from djcues.models import TagLink
from djcues.tag_apply import (
    apply_plans,
    current_tags_by_category,
    decide_category,
    desired_from_session_track,
    plan_session,
    writer_changes,
)
from djcues.tag_store import TagStore
from tests.conftest import requires_rekordbox, requires_rekordbox_closed


def _link(content_id, tag, column, link_id=None):
    return TagLink(link_id=link_id or f"L-{content_id}-{tag}", content_id=str(content_id),
                   tag_id=f"T-{tag}", tag_name=tag, column_id=column)


# --- decide_category: the whole safety model --------------------------------


def test_unchanged_when_rekordbox_already_matches():
    plan = decide_category("Energy", {"Energy 3"}, {"Energy 3"}, None)
    assert plan.action == "unchanged"


def test_first_write_into_an_empty_category():
    plan = decide_category("Mix In", {"Long Intro", "Vocal Intro"}, set(), None)
    assert plan.action == "set"
    assert plan.add == {"Long Intro", "Vocal Intro"} and plan.remove == frozenset()


def test_djcues_may_update_what_it_wrote_itself():
    plan = decide_category("Energy", {"Energy 4"}, {"Energy 3"}, {"Energy 3"})
    assert plan.action == "set"
    assert plan.add == {"Energy 4"} and plan.remove == {"Energy 3"}


def test_djcues_may_clear_its_own_tag_when_no_longer_proposed():
    plan = decide_category("Vocals", set(), {"Vocal-Led"}, {"Vocal-Led"})
    assert plan.action == "set" and plan.remove == {"Vocal-Led"}


def test_a_tag_the_user_changed_is_left_alone():
    plan = decide_category("Energy", {"Energy 3"}, {"Energy 5"}, {"Energy 3"})
    assert plan.action == "user_edited"
    assert plan.add == frozenset() and plan.remove == frozenset()


def test_a_tag_the_user_removed_is_never_re_added():
    plan = decide_category("Mix Out", {"Vocal Outro"}, set(), {"Vocal Outro"})
    assert plan.action == "user_edited"


def test_tags_the_user_set_before_djcues_ever_wrote_are_left_alone():
    plan = decide_category("Energy", {"Energy 2"}, {"Energy 4"}, None)
    assert plan.action == "user_edited"


def test_force_overwrites_a_user_edit_and_flags_it():
    plan = decide_category("Energy", {"Energy 3"}, {"Energy 5"}, {"Energy 3"}, force=True)
    assert plan.action == "set" and plan.forced
    assert plan.add == {"Energy 3"} and plan.remove == {"Energy 5"}


def test_force_is_not_flagged_when_nothing_was_overridden():
    plan = decide_category("Energy", {"Energy 4"}, {"Energy 3"}, {"Energy 3"}, force=True)
    assert plan.action == "set" and not plan.forced


# --- what counts as djcues's own tags -----------------------------------------


def test_current_tags_only_counts_catalog_tags_in_their_own_column():
    links = [
        _link(1, "Energy 3", "1"),
        _link(1, "Long Intro", "2"),
        _link(1, "Vocal Outro", "2"),
        _link(1, "Vocal", "2"),          # Rekordbox default tag: not djcues's
        _link(1, "my own tag", "4"),     # user's custom tag: not djcues's
        _link(1, "Energy 4", "3"),       # catalog name, wrong column: not djcues's
    ]
    current = current_tags_by_category(links)
    assert current == {"1": {"Energy": {"Energy 3"}, "Mix In": {"Long Intro"}, "Mix Out": {"Vocal Outro"}}}


# --- reading a session ------------------------------------------------------------


def _session_track(tags=None, skipped=None, status="pending"):
    return {"title": "T", "status": status, "tags": tags or {}, "skipped": skipped or {}}


def test_desired_covers_evaluated_categories_even_when_empty():
    track = _session_track(tags={"Energy": [{"name": "Energy 2", "status": "pending"}]})
    desired = desired_from_session_track(track, ["Energy", "Vocals"])
    assert desired == {"Energy": {"Energy 2"}, "Vocals": set()}  # Vocals evaluated, no tag -> clear ours


def test_categories_skipped_for_missing_data_are_left_alone():
    track = _session_track(skipped={"Vocals": "no_vocal_data"})
    assert "Vocals" not in desired_from_session_track(track, ["Energy", "Vocals"])


def test_reviewer_skipped_tags_are_dropped():
    track = _session_track(tags={"Mix In": [
        {"name": "Long Intro", "status": "pending"}, {"name": "Vocal Intro", "status": "skipped"},
    ]})
    assert desired_from_session_track(track, ["Mix In"])["Mix In"] == {"Long Intro"}


def test_non_catalog_names_in_a_session_are_ignored():
    track = _session_track(tags={"Energy": [{"name": "Energy 99", "status": "pending"}]})
    assert desired_from_session_track(track, ["Energy"])["Energy"] == set()


# --- plan_session / writer_changes ----------------------------------------------


def _session(tracks, categories=None):
    return {"kind": "tags", "tracks": tracks, "categories": categories or ["Energy", "Mix In"]}


def test_plan_session_skips_whole_skipped_tracks(tmp_path):
    session = _session({
        "1": _session_track(tags={"Energy": [{"name": "Energy 1", "status": "pending"}]}),
        "2": _session_track(tags={"Energy": [{"name": "Energy 2", "status": "pending"}]}, status="skipped"),
    })
    with TagStore(tmp_path / "t.db") as store:
        plans = plan_session(session, [], store)
    assert [p.track_id for p in plans] == ["1"]


def test_plan_session_uses_the_ledger(tmp_path):
    session = _session({"1": _session_track(tags={"Energy": [{"name": "Energy 4", "status": "pending"}]})})
    links = [_link(1, "Energy 3", "1")]
    with TagStore(tmp_path / "t.db") as store:
        assert plan_session(session, links, store)[0].categories[0].action == "user_edited"
        store.record_written("1", "Energy", {"Energy 3"})
        assert plan_session(session, links, store)[0].categories[0].action == "set"


def test_writer_changes_removes_every_copy_of_a_dropped_tag(tmp_path):
    session = _session({"1": _session_track(tags={"Energy": [{"name": "Energy 4", "status": "pending"}]})})
    links = [_link(1, "Energy 3", "1", "a"), _link(1, "Energy 3", "1", "b")]  # linked twice
    with TagStore(tmp_path / "t.db") as store:
        store.record_written("1", "Energy", {"Energy 3"})
        changes = writer_changes(plan_session(session, links, store))
    assert changes == [{"content_id": "1", "add": ["Energy 4"], "remove_link_ids": ["a", "b"]}]


def test_writer_changes_is_empty_when_nothing_changes(tmp_path):
    session = _session({"1": _session_track(tags={"Energy": [{"name": "Energy 3", "status": "pending"}]})})
    with TagStore(tmp_path / "t.db") as store:
        assert writer_changes(plan_session(session, [_link(1, "Energy 3", "1")], store)) == []


# --- apply_plans: ledger only after a successful write -----------------------------


def test_apply_plans_records_ledger_and_overrides(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(writer, "apply_tag_changes",
                        lambda changes, db=None: calls.append(changes) or {"added": 1, "removed": 0, "backup": "b"})
    session = _session({
        "1": _session_track(tags={"Energy": [{"name": "Energy 2", "status": "pending"}]}),
        "2": _session_track(tags={"Energy": [{"name": "Energy 2", "status": "pending"}]}),
    }, categories=["Energy"])
    links = [_link(2, "Energy 5", "1")]  # track 2: user's own tag, never written by djcues
    with TagStore(tmp_path / "t.db") as store:
        apply_plans(plan_session(session, links, store), store)
        assert store.last_written("1", "Energy") == {"Energy 2"}
        assert store.last_written("2", "Energy") is None  # untouched -> not djcues's
        overrides = store.overrides()
    assert len(calls) == 1 and calls[0][0]["content_id"] == "1"
    assert [(o["track_id"], o["found"]) for o in overrides] == [("2", ["Energy 5"])]


def test_apply_plans_does_not_touch_the_ledger_when_the_write_fails(tmp_path, monkeypatch):
    def boom(changes, db=None):
        raise writer.RekordboxRunningError("Rekordbox is running")

    monkeypatch.setattr(writer, "apply_tag_changes", boom)
    session = _session({"1": _session_track(tags={"Energy": [{"name": "Energy 2", "status": "pending"}]})},
                       categories=["Energy"])
    with TagStore(tmp_path / "t.db") as store:
        with pytest.raises(writer.RekordboxRunningError):
            apply_plans(plan_session(session, [], store), store)
        assert store.last_written("1", "Energy") is None


def test_apply_plans_skips_the_writer_when_nothing_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(writer, "apply_tag_changes", lambda *a, **k: pytest.fail("writer called"))
    session = _session({"1": _session_track(tags={"Energy": [{"name": "Energy 3", "status": "pending"}]})},
                       categories=["Energy"])
    with TagStore(tmp_path / "t.db") as store:
        result = apply_plans(plan_session(session, [_link(1, "Energy 3", "1")], store), store)
    assert result["added"] == 0


# --- writer.apply_tag_changes envelope (MagicMock db, like test_writer.py) ----------


def _fake_db(tmp_path):
    db = MagicMock()
    db.db_directory = tmp_path
    (tmp_path / "master.db").write_bytes(b"x")
    return db


def test_apply_tag_changes_checks_rekordbox_closed_before_anything(tmp_path):
    db = _fake_db(tmp_path)
    with patch.object(writer, "ensure_rekordbox_closed", side_effect=writer.RekordboxRunningError("running")):
        with pytest.raises(writer.RekordboxRunningError):
            writer.apply_tag_changes([{"content_id": "1", "add": ["Energy 1"]}], db=db)
    db.commit.assert_not_called()
    assert not list(tmp_path.glob("master-backup-*"))


def test_apply_tag_changes_validates_tracks_before_backing_up(tmp_path):
    db = _fake_db(tmp_path)
    db.get_content.return_value = None
    with patch.object(writer, "ensure_rekordbox_closed"):
        with pytest.raises(ValueError, match="not found"):
            writer.apply_tag_changes([{"content_id": "1", "add": ["Energy 1"]}], db=db)
    assert not list(tmp_path.glob("master-backup-*"))


def test_apply_tag_changes_rolls_back_when_commit_is_refused(tmp_path):
    db = _fake_db(tmp_path)
    db.commit.side_effect = RuntimeError("Rekordbox is running")
    with patch.object(writer, "ensure_rekordbox_closed"), \
         patch.object(writer, "ensure_tag_columns", return_value={"Energy 1": "100"}):
        with pytest.raises(writer.RekordboxRunningError):
            writer.apply_tag_changes([{"content_id": "1", "add": ["Energy 1"]}], db=db)
    db.rollback.assert_called_once()
    assert list(tmp_path.glob("master-backup-*"))  # backup happened before the attempt


def test_apply_tag_changes_rejects_unknown_tags_and_rolls_back(tmp_path):
    db = _fake_db(tmp_path)
    with patch.object(writer, "ensure_rekordbox_closed"), \
         patch.object(writer, "ensure_tag_columns", return_value={"Energy 1": "100"}):
        with pytest.raises(writer.TagWriteError, match="unknown tag"):
            writer.apply_tag_changes([{"content_id": "1", "add": ["Bogus"]}], db=db)
    db.rollback.assert_called_once()
    db.commit.assert_not_called()


# --- integration: a COPY of the real library ---------------------------------------


_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


@requires_rekordbox
@requires_rekordbox_closed
class TestTagWriteIntegration:
    @pytest.fixture
    def db_copy(self, tmp_path):
        from pyrekordbox.db6 import Rekordbox6Database

        from djcues.db import get_db

        shutil.copy2(get_db().db_directory / "master.db", tmp_path / "master.db")
        db = Rekordbox6Database(path=tmp_path / "master.db")
        yield db
        if db.session is not None:  # a test may already have closed it to re-open fresh
            db.close()

    def test_writes_rows_in_rekordboxs_own_format(self, db_copy, tmp_path):
        from pyrekordbox.db6 import Rekordbox6Database

        from djcues.db import read_tag_links

        content_id = str(next(iter(db_copy.get_content())).ID)
        before = {l.link_id for l in read_tag_links([content_id], db=db_copy)}
        usn_before = db_copy.get_agent_registry(registry_id="localUpdateCount").int_1

        result = writer.apply_tag_changes(
            [{"content_id": content_id, "add": ["Energy 2", "Long Intro"]}], db=db_copy
        )
        assert result["added"] == 2 and result["backup"].exists()
        db_copy.close()

        db = Rekordbox6Database(path=tmp_path / "master.db")
        try:
            columns = {str(t.ID): t.Name for t in db.get_my_tag() if t.Attribute == 1}
            assert columns == {"1": "Energy", "2": "Mix", "3": "Vocals", "4": "Check"}
            new = [l for l in read_tag_links([content_id], db=db) if l.link_id not in before]
            assert sorted(l.tag_name for l in new) == ["Energy 2", "Long Intro"]
            for l in new:
                row = db.get_my_tag_songs(ID=l.link_id)
                row = row.one() if hasattr(row, "one") else row
                assert _UUID.match(row.ID) and _UUID.match(row.UUID) and row.ID != row.UUID
                assert row.TrackNo is None
                assert row.rb_local_usn is not None and row.rb_local_usn > usn_before
            assert db.get_agent_registry(registry_id="localUpdateCount").int_1 > usn_before
        finally:
            db.close()

    def test_ensure_tag_columns_is_idempotent(self, db_copy):
        first = writer.ensure_tag_columns(db_copy)
        db_copy.commit()
        n_tags = len(list(db_copy.get_my_tag()))
        second = writer.ensure_tag_columns(db_copy)
        db_copy.commit()
        assert first == second
        assert len(list(db_copy.get_my_tag())) == n_tags  # nothing duplicated
