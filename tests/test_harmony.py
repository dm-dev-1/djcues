"""Tests for djcues.harmony -- Camelot Wheel key compatibility and
BPM-closeness (including half/double-time) matching.
"""

from __future__ import annotations

import pytest

from djcues.harmony import (
    BpmRelation,
    CamelotKey,
    camelot_relationship,
    classify_bpm_relation,
    parse_camelot_key,
    suggest_compatible_tracks,
)
from djcues.models import TrackSummary


def _track(id_: str, title: str, artist: str = "Artist", bpm: float = 128.0, key: str | None = "8A") -> TrackSummary:
    return TrackSummary(id=id_, track_no=None, title=title, artist=artist, bpm=bpm, duration_ms=200_000.0, key=key)


# ---------------------------------------------------------------------------
# parse_camelot_key
# ---------------------------------------------------------------------------


def test_parses_valid_camelot_keys():
    assert parse_camelot_key("9A") == CamelotKey(9, "A")
    assert parse_camelot_key("12B") == CamelotKey(12, "B")
    assert parse_camelot_key("1A") == CamelotKey(1, "A")


def test_parse_strips_whitespace():
    assert parse_camelot_key(" 9A ") == CamelotKey(9, "A")


def test_rejects_real_non_camelot_values_confirmed_in_this_library():
    # Confirmed live: this real Rekordbox library has these exact raw
    # values sitting in the same djmdKey lookup table as the 24 standard
    # Camelot positions.
    assert parse_camelot_key("E") is None
    assert parse_camelot_key("Dm") is None


def test_rejects_out_of_range_number():
    assert parse_camelot_key("13A") is None
    assert parse_camelot_key("0A") is None


def test_rejects_invalid_letter():
    assert parse_camelot_key("9C") is None


def test_rejects_missing_key():
    assert parse_camelot_key(None) is None
    assert parse_camelot_key("") is None


def test_str_roundtrips():
    assert str(CamelotKey(9, "A")) == "9A"


# ---------------------------------------------------------------------------
# camelot_relationship -- the correctness guarantee for the whole feature
# ---------------------------------------------------------------------------


def test_camelot_relationship_exhaustive_matrix():
    """Every one of the 24 Camelot positions against every other: exactly
    1 "same" (itself), 2 "energy_*" (+-1, same letter), 1 "relative"
    (same number, other letter), and the remaining 20 are None --
    including every diagonal move, which is not part of the standard
    rule set (verified against Mixed In Key's own docs, not a
    synthesized summary that initially claimed otherwise)."""
    all_keys = [CamelotKey(n, l) for n in range(1, 13) for l in ("A", "B")]
    assert len(all_keys) == 24

    for a in all_keys:
        results = [camelot_relationship(a, b) for b in all_keys]
        counts: dict[str | None, int] = {}
        for r in results:
            counts[r] = counts.get(r, 0) + 1
        assert counts.get("same") == 1, f"{a}: expected exactly 1 same, got {counts}"
        assert counts.get("energy_boost") == 1, f"{a}: expected exactly 1 energy_boost, got {counts}"
        assert counts.get("energy_drop") == 1, f"{a}: expected exactly 1 energy_drop, got {counts}"
        assert counts.get("relative") == 1, f"{a}: expected exactly 1 relative, got {counts}"
        assert counts.get(None) == 20, f"{a}: expected exactly 20 None, got {counts}"


def test_camelot_relationship_wraparound_12_to_1():
    assert camelot_relationship(CamelotKey(12, "A"), CamelotKey(1, "A")) == "energy_boost"
    assert camelot_relationship(CamelotKey(1, "A"), CamelotKey(12, "A")) == "energy_drop"


def test_camelot_relationship_diagonal_move_not_compatible():
    # The concrete case an incorrect web summary initially got wrong --
    # a diagonal move (both number AND letter change) is NOT one of the
    # three standard compatible moves.
    assert camelot_relationship(CamelotKey(8, "A"), CamelotKey(9, "B")) is None
    assert camelot_relationship(CamelotKey(8, "A"), CamelotKey(7, "B")) is None


def test_camelot_relationship_same_key():
    assert camelot_relationship(CamelotKey(8, "A"), CamelotKey(8, "A")) == "same"


def test_camelot_relationship_relative_major_minor():
    assert camelot_relationship(CamelotKey(8, "A"), CamelotKey(8, "B")) == "relative"


# ---------------------------------------------------------------------------
# classify_bpm_relation
# ---------------------------------------------------------------------------


def test_same_tempo_within_tolerance():
    result = classify_bpm_relation(128.0, 130.0)  # 1.56% off, well within default 6%
    assert result is not None
    assert result.kind == "same_tempo"
    assert result.target_bpm == 128.0


def test_same_tempo_outside_tolerance():
    assert classify_bpm_relation(128.0, 145.0, tolerance_pct=6.0) is None


def test_same_tempo_just_inside_vs_just_outside_tolerance():
    # 128 * 1.05 = 134.4 (5% over, inside a 6% tolerance) vs 128 * 1.07 =
    # 136.96 (7% over, outside) -- avoids asserting an exact float
    # boundary (128 * 1.06 doesn't land on precisely 6.0% due to binary
    # float rounding), while still confirming the tolerance is real and
    # not off by an order of magnitude or inverted.
    just_inside = classify_bpm_relation(128.0, 134.4, tolerance_pct=6.0)
    assert just_inside is not None
    assert just_inside.kind == "same_tempo"

    just_outside = classify_bpm_relation(128.0, 136.96, tolerance_pct=6.0)
    assert just_outside is None


def test_half_time_match():
    result = classify_bpm_relation(174.0, 87.0)  # exactly half
    assert result is not None
    assert result.kind == "half_time"
    assert result.target_bpm == 87.0
    assert result.pitch_shift_pct == pytest.approx(0.0)


def test_double_time_match():
    result = classify_bpm_relation(87.0, 174.0)
    assert result is not None
    assert result.kind == "double_time"
    assert result.target_bpm == 174.0


def test_allow_half_double_false_suppresses_both():
    assert classify_bpm_relation(174.0, 87.0, allow_half_double=False) is None
    assert classify_bpm_relation(87.0, 174.0, allow_half_double=False) is None


def test_zero_or_negative_bpm_returns_none():
    assert classify_bpm_relation(0.0, 128.0) is None
    assert classify_bpm_relation(128.0, 0.0) is None
    assert classify_bpm_relation(-5.0, 128.0) is None


def test_closest_target_wins_under_wide_tolerance():
    # With a deliberately wide tolerance, 130 BPM is close to both
    # reference (128, same_tempo) and could theoretically be considered
    # against half/double targets too -- same_tempo should win since
    # it's numerically closest.
    result = classify_bpm_relation(128.0, 130.0, tolerance_pct=50.0)
    assert result is not None
    assert result.kind == "same_tempo"


# ---------------------------------------------------------------------------
# suggest_compatible_tracks
# ---------------------------------------------------------------------------


def test_reference_with_no_key_raises():
    reference = _track("1", "Ref", key=None)
    with pytest.raises(ValueError, match="no usable Camelot key"):
        suggest_compatible_tracks(reference, [])


def test_reference_with_non_camelot_key_raises():
    reference = _track("1", "Ref", key="E")
    with pytest.raises(ValueError, match="no usable Camelot key"):
        suggest_compatible_tracks(reference, [])


def test_reference_with_no_bpm_raises():
    reference = _track("1", "Ref", bpm=0.0, key="8A")
    with pytest.raises(ValueError, match="no usable BPM"):
        suggest_compatible_tracks(reference, [])


def test_empty_candidates_does_not_crash():
    reference = _track("1", "Ref")
    result = suggest_compatible_tracks(reference, [])
    assert result.suggestions == []
    assert result.excluded == []


def test_excludes_reference_own_id_from_results():
    reference = _track("1", "Ref", bpm=128.0, key="8A")
    same_track_again = _track("1", "Ref", bpm=128.0, key="8A")
    result = suggest_compatible_tracks(reference, [same_track_again])
    assert result.suggestions == []


def test_no_key_candidate_is_excluded_with_reason():
    reference = _track("1", "Ref", bpm=128.0, key="8A")
    candidate = _track("2", "No Key Track", bpm=128.0, key=None)
    result = suggest_compatible_tracks(reference, [candidate])
    assert result.suggestions == []
    assert len(result.excluded) == 1
    assert result.excluded[0].reason == "no_key"
    assert result.excluded[0].track.id == "2"


def test_non_camelot_key_candidate_is_excluded_with_reason_using_real_values():
    reference = _track("1", "Ref", bpm=128.0, key="8A")
    candidate = _track("2", "Weird Key Track", bpm=128.0, key="Dm")
    result = suggest_compatible_tracks(reference, [candidate])
    assert len(result.excluded) == 1
    assert result.excluded[0].reason == "non_camelot_key"


def test_incompatible_key_is_filtered_not_excluded():
    # A parseable but harmonically-incompatible key is normal filtering,
    # not a data-quality issue -- must not show up in .excluded.
    reference = _track("1", "Ref", bpm=128.0, key="8A")
    candidate = _track("2", "Diagonal", bpm=128.0, key="9B")  # diagonal from 8A
    result = suggest_compatible_tracks(reference, [candidate])
    assert result.suggestions == []
    assert result.excluded == []


def test_bpm_out_of_tolerance_is_filtered_not_excluded():
    reference = _track("1", "Ref", bpm=128.0, key="8A")
    candidate = _track("2", "Too Fast", bpm=145.0, key="8A")  # same key, bad tempo
    result = suggest_compatible_tracks(reference, [candidate])
    assert result.suggestions == []
    assert result.excluded == []


def test_sorts_by_key_rank_then_bpm_closeness_then_title():
    reference = _track("1", "Ref", bpm=128.0, key="8A")
    relative = _track("2", "Relative Match", bpm=128.0, key="8B")  # rank 2
    boost = _track("3", "Boost Match", bpm=128.0, key="9A")  # rank 1
    same_far = _track("4", "Same Key Far BPM", bpm=131.0, key="8A")  # rank 0, further BPM
    same_close = _track("5", "Same Key Close BPM", bpm=128.5, key="8A")  # rank 0, closer BPM

    result = suggest_compatible_tracks(reference, [relative, boost, same_far, same_close])

    assert [s.track.id for s in result.suggestions] == ["5", "4", "3", "2"]


def test_limit_caps_suggestions_but_not_excluded_count():
    reference = _track("1", "Ref", bpm=128.0, key="8A")
    matches = [_track(str(i), f"Match {i}", bpm=128.0, key="8A") for i in range(2, 7)]  # 5 matches
    no_key = [_track(str(i), f"NoKey {i}", bpm=128.0, key=None) for i in range(7, 10)]  # 3 excluded

    result = suggest_compatible_tracks(reference, matches + no_key, limit=2)

    assert len(result.suggestions) == 2
    assert len(result.excluded) == 3  # not capped by limit


def test_half_time_match_surfaced_by_default():
    reference = _track("1", "Ref", bpm=174.0, key="8A")
    candidate = _track("2", "Half Time", bpm=87.0, key="8A")
    result = suggest_compatible_tracks(reference, [candidate])
    assert len(result.suggestions) == 1
    assert result.suggestions[0].bpm_relation.kind == "half_time"


def test_allow_half_double_false_excludes_half_time_match():
    reference = _track("1", "Ref", bpm=174.0, key="8A")
    candidate = _track("2", "Half Time", bpm=87.0, key="8A")
    result = suggest_compatible_tracks(reference, [candidate], allow_half_double=False)
    assert result.suggestions == []


# ---------------------------------------------------------------------------
# Encrypted-metadata handling (Rekordbox streaming-linked tracks, e.g.
# Spotify -- confirmed live: Title/Artist can be encrypted ciphertext
# starting with "$A7:" while BPM/Key are still real, usable values, so
# these aren't caught by the no-key/non-Camelot checks and need their
# own explicit handling to avoid ever showing raw ciphertext as a
# "track title" in a suggestion.
# ---------------------------------------------------------------------------

_ENCRYPTED_TITLE = "$A7:v1:cXRGX5Nhvaa9rt8efO7O4g==:9qEcCSYCFHtls1UhP1IHWTd312u5L59/27m9UzFbZs4="


def test_candidate_with_encrypted_title_is_excluded_with_reason():
    reference = _track("1", "Ref", bpm=128.0, key="8A")
    candidate = _track("2", _ENCRYPTED_TITLE, bpm=128.0, key="8A")  # real key/BPM, garbled title
    result = suggest_compatible_tracks(reference, [candidate])
    assert result.suggestions == []
    assert len(result.excluded) == 1
    assert result.excluded[0].reason == "encrypted_metadata"


def test_candidate_with_encrypted_artist_is_excluded_with_reason():
    reference = _track("1", "Ref", bpm=128.0, key="8A")
    candidate = _track("2", "Real Title", artist=_ENCRYPTED_TITLE, bpm=128.0, key="8A")
    result = suggest_compatible_tracks(reference, [candidate])
    assert len(result.excluded) == 1
    assert result.excluded[0].reason == "encrypted_metadata"


def test_reference_with_encrypted_title_raises_without_embedding_ciphertext():
    reference = _track("1", _ENCRYPTED_TITLE, bpm=128.0, key="8A")
    with pytest.raises(ValueError) as exc_info:
        suggest_compatible_tracks(reference, [])
    # The whole point: never echo the ciphertext back in a message.
    assert _ENCRYPTED_TITLE not in str(exc_info.value)
    assert "encrypted" in str(exc_info.value)
