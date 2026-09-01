from djcues.metrics import PadStats, compare_cues, merge_pad_stats, overall_stats
from djcues.models import CuePoint


def _cue(kind: int, position_ms: float) -> CuePoint:
    return CuePoint(
        kind=kind, position_ms=position_ms, loop_end_ms=None,
        color_table_index=0, color=0, comment="",
    )


def test_match_within_tolerance():
    existing = [_cue(1, 1000.0)]
    proposed = [_cue(1, 1500.0)]  # 500ms off, within default 1000ms tolerance
    stats = compare_cues(existing, proposed)
    assert stats["A"].matches == 1
    assert stats["A"].misses == 0
    assert stats["A"].false_positives == 0


def test_match_exactly_at_tolerance_boundary():
    existing = [_cue(1, 1000.0)]
    proposed = [_cue(1, 2000.0)]  # exactly 1000ms -- inclusive boundary
    stats = compare_cues(existing, proposed, tolerance_ms=1000.0)
    assert stats["A"].matches == 1
    assert stats["A"].misses == 0


def test_miss_just_outside_tolerance_boundary():
    existing = [_cue(1, 1000.0)]
    proposed = [_cue(1, 2001.0)]  # 1001ms -- just outside
    stats = compare_cues(existing, proposed, tolerance_ms=1000.0)
    assert stats["A"].matches == 0
    assert stats["A"].misses == 1


def test_existing_only_is_a_miss():
    existing = [_cue(1, 1000.0)]
    proposed = []
    stats = compare_cues(existing, proposed)
    assert stats["A"].matches == 0
    assert stats["A"].misses == 1
    assert stats["A"].false_positives == 0


def test_proposed_only_is_a_false_positive():
    """The literal regression test for the old bug: a proposed cue with no
    ground-truth counterpart used to be invisible to the score entirely."""
    existing = []
    proposed = [_cue(1, 1000.0)]
    stats = compare_cues(existing, proposed)
    assert stats["A"].matches == 0
    assert stats["A"].misses == 0
    assert stats["A"].false_positives == 1


def test_empty_both_sides_no_stats_and_no_divide_by_zero():
    stats = compare_cues([], [])
    assert stats == {}
    empty = PadStats()
    assert empty.precision == 0.0
    assert empty.recall == 0.0
    assert empty.f1 == 0.0


def test_memory_cues_kind_zero_excluded():
    """Memory cues (kind == 0) shouldn't be compared -- only hot cues (kind > 0)."""
    existing = [_cue(0, 1000.0)]
    proposed = [_cue(0, 1000.0)]
    stats = compare_cues(existing, proposed)
    assert stats == {}


def test_precision_recall_f1():
    s = PadStats(matches=3, misses=1, false_positives=1)
    assert s.precision == 3 / 4
    assert s.recall == 3 / 4
    assert s.f1 == 3 / 4


def test_merge_pad_stats_sums_across_tracks():
    track1 = {"A": PadStats(matches=1, misses=0, false_positives=0)}
    track2 = {"A": PadStats(matches=0, misses=1, false_positives=0),
              "D": PadStats(matches=1, misses=0, false_positives=1)}
    merged = merge_pad_stats([track1, track2])
    assert merged["A"].matches == 1
    assert merged["A"].misses == 1
    assert merged["D"].matches == 1
    assert merged["D"].false_positives == 1


def test_overall_stats_flattens_all_pads():
    pad_stats = {
        "A": PadStats(matches=2, misses=1, false_positives=0),
        "D": PadStats(matches=1, misses=0, false_positives=1),
    }
    total = overall_stats(pad_stats)
    assert total.matches == 3
    assert total.misses == 1
    assert total.false_positives == 1
