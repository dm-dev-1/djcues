"""Tests for djcues.audit -- BPM/Key comment-hint cross-checking and
unusable-key detection.
"""

from __future__ import annotations

from djcues.audit import (
    audit_tracks,
    find_comment_hint_mismatches,
    find_unusable_keys,
    parse_comment_hint,
)
from djcues.harmony import CamelotKey
from djcues.models import TrackSummary
from tests.conftest import requires_rekordbox


def _track(
    id_: str, title: str = "Track", artist: str = "Artist",
    bpm: float = 128.0, key: str | None = "8A", comment: str | None = None,
) -> TrackSummary:
    return TrackSummary(
        id=id_, track_no=None, title=title, artist=artist, bpm=bpm,
        duration_ms=200_000.0, key=key, comment=comment,
    )


# ---------------------------------------------------------------------------
# parse_comment_hint
# ---------------------------------------------------------------------------


def test_parses_confirmed_real_formats():
    assert parse_comment_hint("2A - 118") == (CamelotKey(2, "A"), 118.0)
    assert parse_comment_hint("10B - 91") == (CamelotKey(10, "B"), 91.0)


def test_strips_whitespace():
    assert parse_comment_hint("  2A - 118  ") == (CamelotKey(2, "A"), 118.0)


def test_rejects_real_near_miss_comments_confirmed_in_this_library():
    # Confirmed live: real comments in this library that look superficially
    # similar but are not actual hints.
    assert parse_comment_hint("9A - LDS #231 (12.01.21)") is None
    assert parse_comment_hint("8A - Energy 7") is None


def test_rejects_non_camelot_key_prefix():
    # Matches the regex shape but "E" isn't valid Camelot notation --
    # deferred entirely to parse_camelot_key's own judgment.
    assert parse_comment_hint("E - 118") is None


def test_rejects_missing_or_empty_comment():
    assert parse_comment_hint(None) is None
    assert parse_comment_hint("") is None


def test_rejects_wrong_separator_spacing():
    # Confirmed live: every real match in this library uses exactly
    # " - " (space-dash-space) -- these variants are correctly rejected.
    assert parse_comment_hint("2A -118") is None
    assert parse_comment_hint("2A- 118") is None
    assert parse_comment_hint("2A-118") is None


def test_parses_decimal_bpm():
    assert parse_comment_hint("2A - 118.5") == (CamelotKey(2, "A"), 118.5)


def test_zero_bpm_parses_syntactically():
    # Syntactically valid -- classify_bpm_relation's own non-positive
    # guard is what would later treat this as a mismatch, not this function.
    assert parse_comment_hint("2A - 0") == (CamelotKey(2, "A"), 0.0)


# ---------------------------------------------------------------------------
# find_comment_hint_mismatches -- the two real, live-confirmed cases from
# this library are the actual correctness guarantee here.
# ---------------------------------------------------------------------------


def test_dreamin_case_flags_bpm_mismatch():
    # Real track: stored BPM 146.92, comment "2A - 118", key matches.
    # 146.92 isn't within tolerance of 118, 59, or 236 -- a genuine error.
    track = _track("1", title="Dreamin' Original Mix", bpm=146.92, key="2A", comment="2A - 118")
    findings, hint_count = find_comment_hint_mismatches([track])
    assert hint_count == 1
    assert len(findings) == 1
    assert findings[0].bpm_mismatch is True
    assert findings[0].key_mismatch is False


def test_time_to_pretend_case_not_flagged():
    # Real track: stored BPM 180.0, comment "10B - 91", key matches.
    # 180 is a clean double-time of 91 (1.1% off) -- not a real error.
    track = _track("1", title="Time To Pretend (High Contrast remix)", bpm=180.0, key="10B", comment="10B - 91")
    findings, hint_count = find_comment_hint_mismatches([track])
    assert hint_count == 1
    assert findings == []


def test_agreeing_comment_not_flagged():
    track = _track("1", bpm=128.0, key="8A", comment="8A - 128")
    findings, hint_count = find_comment_hint_mismatches([track])
    assert hint_count == 1
    assert findings == []


def test_key_mismatch_only_still_flagged():
    track = _track("1", bpm=128.0, key="9A", comment="8A - 128")  # BPM agrees, key doesn't
    findings, _hint_count = find_comment_hint_mismatches([track])
    assert len(findings) == 1
    assert findings[0].bpm_mismatch is False
    assert findings[0].key_mismatch is True


def test_no_hint_track_skipped_entirely():
    track = _track("1", bpm=128.0, key="8A", comment=None)
    findings, hint_count = find_comment_hint_mismatches([track])
    assert hint_count == 0
    assert findings == []


def test_near_miss_comment_not_counted_as_a_hint():
    track = _track("1", bpm=128.0, key="8A", comment="9A - LDS #231 (12.01.21)")
    findings, hint_count = find_comment_hint_mismatches([track])
    assert hint_count == 0
    assert findings == []


def test_comment_hint_count_counts_agreeing_and_disagreeing_not_no_hint():
    agreeing = _track("1", bpm=128.0, key="8A", comment="8A - 128")
    disagreeing = _track("2", bpm=146.92, key="2A", comment="2A - 118")
    no_hint = _track("3", bpm=128.0, key="8A", comment=None)
    _findings, hint_count = find_comment_hint_mismatches([agreeing, disagreeing, no_hint])
    assert hint_count == 2


def test_tolerance_pct_changes_outcome():
    # 128 vs comment 120 is ~6.67% off -- flagged at the default 6%
    # tolerance, not flagged with a wider one.
    track = _track("1", bpm=128.0, key="8A", comment="8A - 120")
    findings_default, _ = find_comment_hint_mismatches([track])
    assert len(findings_default) == 1

    findings_wide, _ = find_comment_hint_mismatches([track], bpm_tolerance_pct=10.0)
    assert findings_wide == []


def test_allow_half_double_false_flags_the_time_to_pretend_case():
    # Same real case as above, but with half/double-time matching turned
    # off -- now it SHOULD flag, since only the same-tempo check applies.
    track = _track("1", bpm=180.0, key="10B", comment="10B - 91")
    findings, _ = find_comment_hint_mismatches([track], allow_half_double=False)
    assert len(findings) == 1
    assert findings[0].bpm_mismatch is True


def test_encrypted_metadata_track_with_real_looking_hint_is_skipped_entirely():
    # A streaming-linked track can have a real, disagreeing comment hint
    # but a ciphertext title -- must never appear in findings (it's
    # caught separately by find_unusable_keys instead).
    encrypted_title = "$A7:v1:cXRGX5Nhvaa9rt8efO7O4g==:9qEcCSYCFHtls1UhP1IHWTd312u5L59/27m9UzFbZs4="
    track = _track("1", title=encrypted_title, bpm=146.92, key="2A", comment="2A - 118")
    findings, hint_count = find_comment_hint_mismatches([track])
    assert findings == []
    assert hint_count == 0


# ---------------------------------------------------------------------------
# find_unusable_keys
# ---------------------------------------------------------------------------


def test_no_key_reason():
    track = _track("1", key=None)
    result = find_unusable_keys([track])
    assert len(result) == 1
    assert result[0].reason == "no_key"


def test_non_camelot_key_reason_using_real_value():
    track = _track("1", key="Dm")
    result = find_unusable_keys([track])
    assert len(result) == 1
    assert result[0].reason == "non_camelot_key"


def test_encrypted_metadata_reason():
    encrypted_title = "$A7:v1:cXRGX5Nhvaa9rt8efO7O4g==:9qEcCSYCFHtls1UhP1IHWTd312u5L59/27m9UzFbZs4="
    track = _track("1", title=encrypted_title, key="8A")
    result = find_unusable_keys([track])
    assert len(result) == 1
    assert result[0].reason == "encrypted_metadata"


def test_valid_key_track_produces_no_entry():
    track = _track("1", key="8A")
    assert find_unusable_keys([track]) == []


def test_track_with_bad_key_and_disagreeing_hint_appears_in_both_results():
    # Confirms the two passes are genuinely independent -- both true,
    # both useful, answering different questions.
    track = _track("1", bpm=146.92, key=None, comment="2A - 118")

    findings, _hint_count = find_comment_hint_mismatches([track])
    unusable = find_unusable_keys([track])

    assert len(findings) == 1
    assert findings[0].key_mismatch is True  # None != CamelotKey(2, "A")
    assert len(unusable) == 1
    assert unusable[0].reason == "no_key"


# ---------------------------------------------------------------------------
# audit_tracks
# ---------------------------------------------------------------------------


def test_scanned_equals_input_length():
    tracks = [_track("1"), _track("2"), _track("3")]
    result = audit_tracks(tracks)
    assert result.scanned == 3


def test_combined_categories_integration():
    dreamin = _track("1", title="Dreamin'", bpm=146.92, key="2A", comment="2A - 118")
    agreeing = _track("2", bpm=128.0, key="8A", comment="8A - 128")
    no_key = _track("3", key=None)
    non_camelot = _track("4", key="Dm")
    no_comment = _track("5", key="8A", comment=None)

    result = audit_tracks([dreamin, agreeing, no_key, non_camelot, no_comment])

    assert result.scanned == 5
    assert result.comment_hint_count == 2  # dreamin + agreeing
    assert len(result.findings) == 1
    assert result.findings[0].track.id == "1"
    assert {u.track.id for u in result.unusable_keys} == {"3", "4"}


# ---------------------------------------------------------------------------
# Real-library tests. Structural properties only, never hardcoded counts --
# this project's own lesson, learned twice this session already from a
# stale Tech House track-count assertion breaking after a real playlist
# move. The Time To Pretend check is a real, unguarded regression test
# (fails loudly if that double-time case is ever misclassified again).
# ---------------------------------------------------------------------------


@requires_rekordbox
def test_audit_real_library_structural_properties():
    from djcues.db import list_all_tracks

    tracks = list_all_tracks()
    result = audit_tracks(tracks)

    assert result.scanned == len(tracks) > 0
    assert 0 <= result.comment_hint_count <= result.scanned
    assert len(result.findings) <= result.comment_hint_count
    assert all(f.bpm_mismatch or f.key_mismatch for f in result.findings)
    assert all(u.reason in ("no_key", "non_camelot_key", "encrypted_metadata") for u in result.unusable_keys)


@requires_rekordbox
def test_audit_real_library_does_not_flag_known_time_to_pretend_double_time():
    from djcues.db import list_all_tracks

    result = audit_tracks(list_all_tracks())
    assert not any("Time To Pretend" in f.track.title for f in result.findings)
