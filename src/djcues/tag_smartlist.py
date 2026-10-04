"""Intelligent Playlists from djcues's tags -- e.g. "Energy 5 + Long
Intro, no Vocal Intro" as a Rekordbox smart playlist that keeps itself
up to date as tags change.

Pure functions only: the XML Rekordbox stores, and which tracks a set of
conditions matches (for a preview). The write itself is
writer.create_tag_smart_playlist().

The XML format is copied from an Intelligent Playlist the user created
in Rekordbox 7.2.9 itself (the plan's Phase 0 diff):

    <NODE Id="571261727" LogicalOperator="1" AutomaticUpdate="0"><CONDITION
    PropertyName="myTag" Operator="8" ValueUnit="" ValueLeft="-1355988717"
    ValueRight=""/></NODE>

Both numbers are the IDs as SIGNED 32-bit integers: the playlist ID
571261727 is below 2^31, so it's stored as-is; the tag ID 2938978579 is
above it, so it wraps to -1355988717. pyrekordbox's own
SmartList.to_xml() instead always subtracts 2^32 from the playlist ID
(its left_bitshift), which would store -3723705569 -- not what Rekordbox
writes -- so djcues builds the XML itself rather than trusting that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from xml.sax.saxutils import quoteattr

if TYPE_CHECKING:
    from djcues.models import TagLink

# Rekordbox's own values (pyrekordbox.db6.smartlist confirms the names).
MATCH_ALL = 1
MATCH_ANY = 2
_CONTAINS = 8
_NOT_CONTAINS = 9


@dataclass(frozen=True)
class TagCondition:
    tag_id: str
    tag_name: str
    present: bool  # True = track has the tag, False = track doesn't


def signed_int32(value: int | str) -> int:
    """An unsigned 32-bit ID as Rekordbox stores it in smart-list XML."""
    v = int(value)
    if not 0 <= v < 2**32:
        raise ValueError(f"{value} is not a 32-bit ID")
    return v - 2**32 if v >= 2**31 else v


def build_smartlist_xml(playlist_id: int | str, conditions: list[TagCondition], match: int) -> str:
    """The SmartList column value for an Intelligent Playlist, in exactly
    the format Rekordbox writes (see module docstring)."""
    if match not in (MATCH_ALL, MATCH_ANY):
        raise ValueError(f"match must be {MATCH_ALL} (all) or {MATCH_ANY} (any)")
    if not conditions:
        raise ValueError("an Intelligent Playlist needs at least one condition")
    parts = [
        f'<NODE Id="{signed_int32(playlist_id)}" LogicalOperator="{match}" AutomaticUpdate="0">'
    ]
    for c in conditions:
        op = _CONTAINS if c.present else _NOT_CONTAINS
        parts.append(
            f'<CONDITION PropertyName="myTag" Operator="{op}" ValueUnit="" '
            f'ValueLeft={quoteattr(str(signed_int32(c.tag_id)))} ValueRight=""/>'
        )
    parts.append("</NODE>")
    return "".join(parts)


def matching_tracks(links: list["TagLink"], conditions: list[TagCondition], match: int,
                    all_track_ids: list[str]) -> list[str]:
    """Which tracks the conditions would select right now -- a preview,
    computed from the real tag links. all_track_ids is the whole
    collection (a "doesn't have tag X" condition can match untagged
    tracks too)."""
    tags_by_track: dict[str, set[str]] = {}
    for link in links:
        tags_by_track.setdefault(link.content_id, set()).add(link.tag_id)

    def hit(track_id: str, c: TagCondition) -> bool:
        return (c.tag_id in tags_by_track.get(track_id, set())) == c.present

    combine = all if match == MATCH_ALL else any
    return [t for t in all_track_ids if combine(hit(t, c) for c in conditions)]
