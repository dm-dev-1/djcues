"""Applying a tag session to Rekordbox -- deciding, per track and per
category, what to change, while never overwriting the user's own edits.

The decision is a three-way comparison per (track, category):

- D = desired: what this session proposes (minus anything marked skipped),
- C = current: what Rekordbox has right now for that category's tags,
- L = last written: what djcues itself last wrote there (tag_store's
  ledger; None if djcues has never touched it).

Rules (decide_category):
- C == D                    -> "unchanged"
- C == L (L=None counts as {}) -> "set": djcues owns what's there, so it's
                               safe to change C into D
- otherwise                 -> "user_edited": someone changed it in
                               Rekordbox since djcues wrote it (or tagged
                               it themselves before djcues ever did) --
                               leave it alone and log an override.
                               force=True overrides this and sets D anyway.

So a tag the user removed is never silently re-added, and a tag the user
added is never silently removed.

plan_session() is pure. apply_plans() does the I/O: one writer call
(one transaction, backup first), then the ledger is updated only after
that commit succeeded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from djcues.tagging import CATEGORY_NAMES, TAG_CATALOG, category_of, column_for

if TYPE_CHECKING:
    from djcues.models import TagLink
    from djcues.tag_store import TagStore

_TAGS_BY_CATEGORY = {cat.name: set(cat.tags) for cat in TAG_CATALOG}


@dataclass(frozen=True)
class CategoryPlan:
    category: str
    action: str  # "unchanged" | "set" | "user_edited"
    desired: frozenset[str]
    current: frozenset[str]
    written: frozenset[str] | None
    add: frozenset[str] = frozenset()
    remove: frozenset[str] = frozenset()
    # True when force=True overwrote a user edit -- still logged as an
    # override, since it's exactly the same feedback signal.
    forced: bool = False


@dataclass
class TrackPlan:
    track_id: str
    title: str
    categories: list[CategoryPlan] = field(default_factory=list)
    # tag name -> djmdSongMyTag link IDs currently carrying it (a tag can
    # legitimately be linked twice; removing it removes every copy).
    link_ids: dict[str, list[str]] = field(default_factory=dict)

    def changes(self) -> list[CategoryPlan]:
        return [c for c in self.categories if c.action == "set" and (c.add or c.remove)]


def decide_category(
    category: str,
    desired: set[str],
    current: set[str],
    written: set[str] | None,
    *,
    force: bool = False,
) -> CategoryPlan:
    d, c = frozenset(desired), frozenset(current)
    w = frozenset(written) if written is not None else None
    if c == d:
        return CategoryPlan(category, "unchanged", d, c, w)
    baseline = w if w is not None else frozenset()
    if c == baseline:
        return CategoryPlan(category, "set", d, c, w, add=d - c, remove=c - d)
    if force:
        return CategoryPlan(category, "set", d, c, w, add=d - c, remove=c - d, forced=True)
    return CategoryPlan(category, "user_edited", d, c, w)


def current_tags_by_category(links: list["TagLink"]) -> dict[str, dict[str, set[str]]]:
    """{track_id: {category: tag names}} for djcues's own tags only. A
    link counts only if its tag name is a catalog tag AND it sits in that
    tag's own column -- a user tag that happens to share a name but lives
    elsewhere is not djcues's. Every other tag (Rekordbox defaults, the
    user's own) is invisible here, so it can never be changed."""
    out: dict[str, dict[str, set[str]]] = {}
    for link in links:
        category = category_of(link.tag_name)
        if category is None or column_for(category).category_id != link.column_id:
            continue
        out.setdefault(link.content_id, {}).setdefault(category, set()).add(link.tag_name)
    return out


def _link_ids(links: list["TagLink"], track_id: str) -> dict[str, list[str]]:
    ids: dict[str, list[str]] = {}
    for link in links:
        if link.content_id == track_id:
            category = category_of(link.tag_name)
            if category is not None and column_for(category).category_id == link.column_id:
                ids.setdefault(link.tag_name, []).append(link.link_id)
    return ids


def desired_from_session_track(track: dict, evaluated: list[str]) -> dict[str, set[str]]:
    """{category: tag names} this session wants for one track, over the
    categories it actually evaluated (requested, and not skipped for
    missing data -- a skipped category is left alone entirely, not
    cleared). Tags marked "skipped" by the reviewer are dropped."""
    skipped = set(track.get("skipped", {}))
    desired: dict[str, set[str]] = {}
    for category in evaluated:
        if category in skipped:
            continue
        entries = track.get("tags", {}).get(category, [])
        desired[category] = {
            e["name"] for e in entries
            if e.get("status") != "skipped" and e["name"] in _TAGS_BY_CATEGORY[category]
        }
    return desired


def plan_session(
    session: dict, links: list["TagLink"], store: "TagStore", *, force: bool = False,
) -> list[TrackPlan]:
    """Decide every change for a tag session (see module docstring).
    Tracks marked skipped are left out entirely."""
    evaluated = session.get("categories") or list(CATEGORY_NAMES)
    current = current_tags_by_category(links)
    plans: list[TrackPlan] = []
    for track_id, track in session.get("tracks", {}).items():
        if track.get("status") == "skipped":
            continue
        plan = TrackPlan(track_id=str(track_id), title=track.get("title", str(track_id)),
                         link_ids=_link_ids(links, str(track_id)))
        for category, desired in desired_from_session_track(track, evaluated).items():
            plan.categories.append(decide_category(
                category, desired, current.get(str(track_id), {}).get(category, set()),
                store.last_written(str(track_id), category), force=force,
            ))
        plans.append(plan)
    return plans


def writer_changes(plans: list[TrackPlan]) -> list[dict]:
    """The writer.apply_tag_changes() payload for every "set" category."""
    changes = []
    for plan in plans:
        add: list[str] = []
        remove_ids: list[str] = []
        for c in plan.changes():
            add.extend(sorted(c.add))
            for tag in sorted(c.remove):
                remove_ids.extend(plan.link_ids.get(tag, []))
        if add or remove_ids:
            changes.append({"content_id": plan.track_id, "add": add, "remove_link_ids": remove_ids})
    return changes


def apply_plans(plans: list[TrackPlan], store: "TagStore", *, db: Any = None) -> dict:
    """Write every planned change in one transaction, then -- only once
    that commit succeeded -- record what was written in the ledger and
    log every user override. Returns the writer's summary."""
    from djcues.writer import apply_tag_changes

    changes = writer_changes(plans)
    result = apply_tag_changes(changes, db=db) if changes else {"added": 0, "removed": 0, "backup": None}

    for plan in plans:
        for c in plan.categories:
            if c.action == "set":
                store.record_written(plan.track_id, c.category, set(c.desired), commit=False)
            if c.action == "user_edited" or c.forced:
                store.record_override(
                    plan.track_id, c.category, written=set(c.written or ()), found=set(c.current),
                    proposed=set(c.desired), commit=False,
                )
    store.commit()
    return result
