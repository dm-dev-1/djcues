"""BPM/Key data-quality audit -- cross-checks each track's stored BPM/
Key against a curator's own "<key> - <bpm>" hint left in Rekordbox's
Comment field, when present, and separately flags any track whose Key
tag is missing, non-Camelot, or unreadable (encrypted, streaming-linked
metadata). Read-only -- djcues has no way to write a BPM/Key tag back
to Rekordbox, and this doesn't attempt one.

Pure functions -- no I/O, no ``click.echo`` -- so they're usable both
from the CLI and from tests without a live Rekordbox connection.
Mirrors harmony.py's own shape, and reuses its logic directly rather
than duplicating it: parse_camelot_key() for key validity,
classify_bpm_relation() for tempo comparison (including half/double-
time, so a legitimate double-time relationship between a comment's
tempo and the stored one is never mistaken for an error), and
_has_encrypted_metadata() for streaming-linked tracks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from djcues.harmony import (
    DEFAULT_BPM_TOLERANCE_PCT,
    BpmRelation,
    CamelotKey,
    _has_encrypted_metadata,
    classify_bpm_relation,
    parse_camelot_key,
)

if TYPE_CHECKING:
    from djcues.models import TrackSummary

# Confirmed live against this real library: every real "<key> - <bpm>"
# curator hint uses this exact separator style (space-dash-space), no
# colons/underscores/no-space variants -- a fixed pattern is sufficient,
# no need for flexible parsing. .fullmatch() (not .search()) is
# deliberate: real near-miss comments like "9A - LDS #231 (12.01.21)"
# must be rejected, not partially matched.
COMMENT_HINT_PATTERN = re.compile(r"^(?P<key>[^-]+) - (?P<bpm>\d+(?:\.\d+)?)$")


def parse_comment_hint(comment: str | None) -> tuple[CamelotKey, float] | None:
    """Parse a "<camelot-key> - <bpm>" curator hint out of a Commnt value.

    None if the comment doesn't match this shape at all, or matches but
    the key portion isn't valid Camelot notation -- defers entirely to
    harmony.parse_camelot_key for that judgment, no second key-validation
    regex here.
    """
    if not comment:
        return None
    match = COMMENT_HINT_PATTERN.fullmatch(comment.strip())
    if not match:
        return None
    key = parse_camelot_key(match.group("key"))
    if key is None:
        return None
    return key, float(match.group("bpm"))


@dataclass(frozen=True)
class CommentHintFinding:
    track: "TrackSummary"
    comment: str
    comment_key: CamelotKey
    comment_bpm: float
    actual_key: str | None  # track.key, verbatim raw string -- not re-parsed
    actual_bpm: float
    bpm_mismatch: bool  # == (bpm_relation is None)
    key_mismatch: bool  # == (parse_camelot_key(track.key) != comment_key)
    bpm_relation: BpmRelation | None


def find_comment_hint_mismatches(
    tracks: list["TrackSummary"],
    *,
    bpm_tolerance_pct: float = DEFAULT_BPM_TOLERANCE_PCT,
    allow_half_double: bool = True,
) -> tuple[list[CommentHintFinding], int]:
    """Returns (findings, comment_hint_count).

    findings holds only tracks where the comment DISAGREES (bpm and/or
    key) with the actual tag; comment_hint_count is every track with
    any parseable hint at all, agreeing or not, for a report's "X of Y
    tracks had a comment hint" line.

    Skips encrypted-metadata tracks (harmony._has_encrypted_metadata) --
    Title/Artist can be ciphertext for streaming-linked tracks even when
    Commnt/BPM/Key are real, so a finding here would otherwise show a
    ciphertext title. That track is still caught separately by
    find_unusable_keys(), nothing is silently lost.
    """
    findings: list[CommentHintFinding] = []
    comment_hint_count = 0

    for track in tracks:
        if _has_encrypted_metadata(track):
            continue

        hint = parse_comment_hint(track.comment)
        if hint is None:
            continue
        comment_key, comment_bpm = hint
        comment_hint_count += 1

        actual_camelot = parse_camelot_key(track.key)
        key_mismatch = actual_camelot != comment_key

        bpm_relation = classify_bpm_relation(
            comment_bpm, track.bpm, tolerance_pct=bpm_tolerance_pct, allow_half_double=allow_half_double
        )
        bpm_mismatch = bpm_relation is None

        if bpm_mismatch or key_mismatch:
            findings.append(CommentHintFinding(
                track=track, comment=track.comment, comment_key=comment_key, comment_bpm=comment_bpm,
                actual_key=track.key, actual_bpm=track.bpm,
                bpm_mismatch=bpm_mismatch, key_mismatch=key_mismatch, bpm_relation=bpm_relation,
            ))

    return findings, comment_hint_count


@dataclass(frozen=True)
class UnusableKeyTrack:
    """A track whose Key tag can't be used at all -- missing, non-Camelot,
    or unreadable (encrypted). Same three reasons and precedence as
    harmony.suggest_compatible_tracks's own .excluded list."""

    track: "TrackSummary"
    reason: str  # "no_key" | "non_camelot_key" | "encrypted_metadata"


def find_unusable_keys(tracks: list["TrackSummary"]) -> list[UnusableKeyTrack]:
    """Independent second pass over the same tracks as
    find_comment_hint_mismatches() -- a track can legitimately appear in
    both (e.g. a comment says "2A" but the actual Key tag is empty
    entirely): both are true, both are useful, answering different
    questions ("does the tag match the curator's own note" vs. "is the
    tag itself even present/valid")."""
    unusable: list[UnusableKeyTrack] = []
    for track in tracks:
        if _has_encrypted_metadata(track):
            unusable.append(UnusableKeyTrack(track=track, reason="encrypted_metadata"))
            continue
        if parse_camelot_key(track.key) is None:
            reason = "no_key" if not track.key else "non_camelot_key"
            unusable.append(UnusableKeyTrack(track=track, reason=reason))
    return unusable


@dataclass(frozen=True)
class AuditResult:
    scanned: int
    comment_hint_count: int
    findings: list[CommentHintFinding] = field(default_factory=list)
    unusable_keys: list[UnusableKeyTrack] = field(default_factory=list)


def audit_tracks(
    tracks: list["TrackSummary"],
    *,
    bpm_tolerance_pct: float = DEFAULT_BPM_TOLERANCE_PCT,
    allow_half_double: bool = True,
) -> AuditResult:
    findings, comment_hint_count = find_comment_hint_mismatches(
        tracks, bpm_tolerance_pct=bpm_tolerance_pct, allow_half_double=allow_half_double
    )
    return AuditResult(
        scanned=len(tracks),
        comment_hint_count=comment_hint_count,
        findings=findings,
        unusable_keys=find_unusable_keys(tracks),
    )
