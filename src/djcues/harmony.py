"""Harmonic mixing suggestions -- Camelot Wheel key compatibility plus
BPM closeness (including half/double-time relationships).

Pure functions -- no I/O, no ``click.echo`` -- so they're usable both
from the CLI and from tests without a live Rekordbox connection.
Mirrors metrics.py's own shape for the same reason.

Camelot Wheel rules verified directly against Mixed In Key's own pages
(the system's creator), not a synthesized web summary -- which
initially and incorrectly suggested diagonal moves (e.g. 8A->9B) are
standard-compatible. The real, standard rule set is exactly three
moves: same key; adjacent number, same letter (+-1, "energy boost/
drop"); same number, different letter ("relative major/minor").
Diagonal moves are deliberately NOT treated as compatible here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from djcues.models import TrackSummary

CAMELOT_PATTERN = re.compile(r"^(1[0-2]|[1-9])(A|B)$")


@dataclass(frozen=True)
class CamelotKey:
    """A parsed Camelot Wheel position -- number 1-12, letter A (minor) or B (major)."""

    number: int
    letter: str

    def __str__(self) -> str:
        return f"{self.number}{self.letter}"


def parse_camelot_key(raw: str | None) -> CamelotKey | None:
    """Parse a raw DjmdKey.ScaleName value into a CamelotKey, or None if
    it's missing or not standard Camelot notation (confirmed live: this
    real library has non-Camelot values like "E"/"Dm" sitting in the
    same lookup table alongside the 24 standard positions)."""
    if not raw:
        return None
    match = CAMELOT_PATTERN.match(raw.strip())
    if not match:
        return None
    return CamelotKey(number=int(match.group(1)), letter=match.group(2))


def camelot_relationship(a: CamelotKey, b: CamelotKey) -> str | None:
    """"same" | "energy_boost" | "energy_drop" | "relative" | None.

    Wraps around the 12-position wheel (12 <-> 1). Diagonal moves
    (different number AND different letter) are None -- not part of
    the standard rule set, confirmed against Mixed In Key's own docs.
    """
    if a == b:
        return "same"
    if a.number == b.number and a.letter != b.letter:
        return "relative"
    if a.letter == b.letter:
        if b.number == a.number % 12 + 1:
            return "energy_boost"
        if a.number == b.number % 12 + 1:
            return "energy_drop"
    return None


KEY_RELATION_LABELS: dict[str, str] = {
    "same": "Same key",
    "energy_boost": "Energy boost (+1)",
    "energy_drop": "Energy drop (-1)",
    "relative": "Relative major/minor",
}
_KEY_RELATION_RANK: dict[str, int] = {"same": 0, "energy_boost": 1, "energy_drop": 1, "relative": 2}

# Rekordbox/Pioneer CDJ's own default pitch fader range -- a real,
# non-arbitrary basis for how close two tracks' tempos need to be to
# mix cleanly with a small pitch nudge, since djcues is Rekordbox-specific.
DEFAULT_BPM_TOLERANCE_PCT = 6.0


@dataclass(frozen=True)
class BpmRelation:
    kind: str  # "same_tempo" | "half_time" | "double_time"
    target_bpm: float  # reference_bpm, reference_bpm/2, or reference_bpm*2
    pitch_shift_pct: float  # >= 0


BPM_RELATION_LABELS: dict[str, str] = {
    "same_tempo": "Same tempo",
    "half_time": "Half-time",
    "double_time": "Double-time",
}


def classify_bpm_relation(
    reference_bpm: float,
    candidate_bpm: float,
    tolerance_pct: float = DEFAULT_BPM_TOLERANCE_PCT,
    allow_half_double: bool = True,
) -> BpmRelation | None:
    """Closest tempo relationship between candidate_bpm and
    reference_bpm within tolerance_pct, or None if nothing matches.

    Checks up to three reference-derived targets (reference_bpm, and --
    when allow_half_double -- reference_bpm/2, reference_bpm*2), each
    using the SAME tolerance window: what makes half/double-time mixing
    work is the ratio being almost exactly 2:1, correctable with the
    same small pitch-fader nudge as a same-tempo match, not a
    separately-loose tolerance. Same ratio-based-detection shape as
    beat_verify.py's _detect_octave_error(), applied to two tracks'
    BPMs instead of a detected beat grid.
    """
    if reference_bpm <= 0 or candidate_bpm <= 0:
        return None

    targets: list[tuple[str, float]] = [("same_tempo", reference_bpm)]
    if allow_half_double:
        targets.append(("half_time", reference_bpm / 2))
        targets.append(("double_time", reference_bpm * 2))

    best: BpmRelation | None = None
    for kind, target_bpm in targets:
        pitch_shift_pct = abs(candidate_bpm - target_bpm) / target_bpm * 100
        if pitch_shift_pct <= tolerance_pct and (best is None or pitch_shift_pct < best.pitch_shift_pct):
            best = BpmRelation(kind=kind, target_bpm=target_bpm, pitch_shift_pct=pitch_shift_pct)
    return best


# Rekordbox encrypts Title/Artist at rest for streaming-linked tracks
# (confirmed live: a real track in this library with FolderPath
# "spotify:track:..." has Title/Artist starting with this exact prefix,
# e.g. "$A7:v1:cXRGX5Nhvaa9rt8efO7O4g==:9qEcCSYCFHtls1UhP1IHWTd312u5L59/
# 27m9UzFbZs4="). These tracks can have real, usable BPM/Key data
# (apparently sourced from the streaming provider's own audio analysis),
# so they aren't caught by the no-key check -- showing raw ciphertext as
# a "track title" in a suggestion would be actively confusing, so they
# get their own, explicit exclusion reason rather than either crashing
# on unreadable text or silently showing it.
_ENCRYPTED_METADATA_PREFIX = "$A7:"


def _has_encrypted_metadata(track: "TrackSummary") -> bool:
    return bool(
        (track.title and track.title.startswith(_ENCRYPTED_METADATA_PREFIX))
        or (track.artist and track.artist.startswith(_ENCRYPTED_METADATA_PREFIX))
    )


@dataclass(frozen=True)
class ExcludedTrack:
    """A candidate that couldn't be evaluated at all -- unusable Key
    data or unreadable (encrypted) metadata, not merely incompatible
    (see suggest_compatible_tracks)."""

    track: "TrackSummary"
    reason: str  # "no_key" | "non_camelot_key" | "encrypted_metadata"


@dataclass(frozen=True)
class Suggestion:
    track: "TrackSummary"
    key_relation: str
    bpm_relation: BpmRelation


@dataclass(frozen=True)
class SuggestionResult:
    reference: "TrackSummary"
    suggestions: list[Suggestion] = field(default_factory=list)
    excluded: list[ExcludedTrack] = field(default_factory=list)


def suggest_compatible_tracks(
    reference: "TrackSummary",
    candidates: list["TrackSummary"],
    *,
    bpm_tolerance_pct: float = DEFAULT_BPM_TOLERANCE_PCT,
    allow_half_double: bool = True,
    limit: int = 10,
) -> SuggestionResult:
    """Rank candidates by harmonic + tempo compatibility with reference.

    Excludes reference's own id from its own results. Raises ValueError
    if reference itself has no parseable Camelot key or no usable BPM --
    there's nothing to suggest compatibility against.

    A candidate with unusable Key data (missing or non-Camelot) is
    recorded in .excluded with a reason, mirroring this project's
    existing convention for un-analyzable tracks (propose/compare
    printing "Skipping ... (no phrase data)") rather than silently
    vanishing it. A candidate with a *parseable but incompatible* key,
    or a BPM outside tolerance, is normal filtering -- not a data-
    quality issue -- so it's just left out, not recorded in .excluded.
    """
    if _has_encrypted_metadata(reference):
        raise ValueError(
            f"track {reference.id} has encrypted, unreadable metadata (likely a streaming-linked "
            "track, e.g. Spotify) -- can't suggest matches for it"
        )
    ref_key = parse_camelot_key(reference.key)
    if ref_key is None:
        raise ValueError(
            f"'{reference.title}' has no usable Camelot key ({reference.key!r}) "
            "-- can't suggest harmonic matches against it"
        )
    if not reference.bpm or reference.bpm <= 0:
        raise ValueError(f"'{reference.title}' has no usable BPM -- can't suggest tempo-compatible matches")

    suggestions: list[Suggestion] = []
    excluded: list[ExcludedTrack] = []

    for candidate in candidates:
        if candidate.id == reference.id:
            continue

        if _has_encrypted_metadata(candidate):
            excluded.append(ExcludedTrack(track=candidate, reason="encrypted_metadata"))
            continue

        cand_key = parse_camelot_key(candidate.key)
        if cand_key is None:
            reason = "no_key" if not candidate.key else "non_camelot_key"
            excluded.append(ExcludedTrack(track=candidate, reason=reason))
            continue

        key_relation = camelot_relationship(ref_key, cand_key)
        if key_relation is None:
            continue

        bpm_relation = classify_bpm_relation(
            reference.bpm, candidate.bpm, tolerance_pct=bpm_tolerance_pct, allow_half_double=allow_half_double
        )
        if bpm_relation is None:
            continue

        suggestions.append(Suggestion(track=candidate, key_relation=key_relation, bpm_relation=bpm_relation))

    suggestions.sort(key=lambda s: (_KEY_RELATION_RANK[s.key_relation], s.bpm_relation.pitch_shift_pct, s.track.title))

    return SuggestionResult(reference=reference, suggestions=suggestions[:limit], excluded=excluded)
