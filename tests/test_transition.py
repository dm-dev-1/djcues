"""Tests for djcues.transition -- transition point suggestions."""

from __future__ import annotations

from djcues.models import BeatGrid, Phrase, Track
from djcues.strategy import DEFAULT_MIN_VOCAL_REGION_MS
from djcues.transition import suggest_transitions
from tests.conftest import requires_rekordbox

_FRAME_MS = 1024 / 22050 * 1000


def _phrase(label: str, position_ms: float, duration_ms: float) -> Phrase:
    return Phrase(beat_start=1, beat_end=2, kind=1, label=label, position_ms=position_ms, duration_ms=duration_ms)


def _vocal_track(total_ms: float, windows: tuple[tuple[float, float], ...] = ()) -> list[int]:
    """An all-zero per-frame vocal array for `total_ms`, with confidence
    4 inside each (start_ms, end_ms) window -- each window is assumed
    long enough to register as its own sustained region at the default
    floor unless a test says otherwise."""
    n_frames = int(total_ms / _FRAME_MS) + 1
    vt = [0] * n_frames
    for start_ms, end_ms in windows:
        i0 = int(start_ms / _FRAME_MS)
        i1 = int(end_ms / _FRAME_MS)
        for i in range(i0, min(i1, n_frames)):
            vt[i] = 4
    return vt


def _track(
    id_: int, title: str, *, phrases: list[Phrase], vocal_track: list[int] | None,
    duration_ms: float = 200_000.0, first_beat_ms: float = 0.0,
) -> Track:
    return Track(
        id=id_, title=title, artist="Artist", bpm=128.0, duration_ms=duration_ms,
        analysis_path="", cues=[], phrases=phrases, beat_grid=BeatGrid(first_beat_ms=first_beat_ms, bpm=128.0),
        vocal_track=vocal_track,
    )


# Standard 200s structure: Intro 0-20s (head zone), Chorus 20-150s, Outro
# 150-200s (tail zone). Reused by every scorable-track test below -- same
# fixture shape as test_clash.py, for cross-feature consistency.
_STANDARD_PHRASES = [
    _phrase("Intro", 0.0, 20_000.0),
    _phrase("Chorus", 20_000.0, 130_000.0),
    _phrase("Outro", 150_000.0, 50_000.0),
]


def _scorable_track(
    id_: int, title: str, *, vocal_windows: tuple[tuple[float, float], ...] = (),
    first_beat_ms: float = 0.0, no_vocal_track: bool = False,
) -> Track:
    return _track(
        id_, title, phrases=list(_STANDARD_PHRASES),
        vocal_track=None if no_vocal_track else _vocal_track(200_000.0, vocal_windows),
        first_beat_ms=first_beat_ms,
    )


# ---------------------------------------------------------------------------
# Core anchor/overlap math
# ---------------------------------------------------------------------------


def test_mix_out_is_outro_phrase_start():
    a = _scorable_track(1, "A")
    b = _scorable_track(2, "B")
    s = suggest_transitions([a, b]).suggestions[0]
    assert s.mix_out_ms == 150_000.0


def test_mix_in_is_first_beat_not_literal_zero():
    # Centerpiece regression test: B's real cue-in point is beat 1, not
    # the raw head-zone start (0.0) -- see transition.py's own docstring
    # for why this differs from clash.head_zone()'s convention.
    a = _scorable_track(1, "A")
    b = _scorable_track(2, "B", first_beat_ms=500.0)
    s = suggest_transitions([a, b]).suggestions[0]
    assert s.mix_in_ms == 500.0


def test_mix_in_is_zero_when_first_beat_is_zero():
    a = _scorable_track(1, "A")
    b = _scorable_track(2, "B", first_beat_ms=0.0)
    s = suggest_transitions([a, b]).suggestions[0]
    assert s.mix_in_ms == 0.0


def test_a_tail_and_b_head_available_exact_values():
    a = _scorable_track(1, "A")
    b = _scorable_track(2, "B", first_beat_ms=500.0)
    s = suggest_transitions([a, b]).suggestions[0]
    assert s.a_tail_available_ms == 50_000.0  # 200_000 - 150_000
    assert s.b_head_available_ms == 19_500.0  # 20_000 - 500


def test_overlap_limited_by_b_head_when_shorter():
    a = _scorable_track(1, "A")  # a_tail_available = 50_000
    b = _scorable_track(2, "B")  # b_head_available = 20_000
    s = suggest_transitions([a, b]).suggestions[0]
    assert s.overlap_ms == 20_000.0
    assert s.limiting_side == "b_head"


def test_overlap_limited_by_a_tail_when_shorter():
    short_tail_phrases = [
        _phrase("Intro", 0.0, 20_000.0),
        _phrase("Chorus", 20_000.0, 175_000.0),
        _phrase("Outro", 195_000.0, 5_000.0),
    ]
    a = _track(1, "A", phrases=short_tail_phrases, vocal_track=_vocal_track(200_000.0))  # a_tail_available = 5_000
    b = _scorable_track(2, "B")  # b_head_available = 20_000
    s = suggest_transitions([a, b]).suggestions[0]
    assert s.overlap_ms == 5_000.0
    assert s.limiting_side == "a_tail"


def test_overlap_clamped_to_zero_when_first_beat_after_intro_end():
    a = _scorable_track(1, "A")
    b = _scorable_track(2, "B", first_beat_ms=25_000.0)  # after B's own Intro (ends 20_000) already ends
    result = suggest_transitions([a, b])
    s = result.suggestions[0]
    assert s.b_head_available_ms == 0.0  # clamped, not negative
    assert s.overlap_ms == 0.0
    assert len(result.suggestions) == 1  # still emitted, not dropped


def test_positions_are_sequential():
    tracks = [_scorable_track(i, str(i)) for i in range(1, 4)]
    result = suggest_transitions(tracks)
    assert [(s.position_a, s.position_b) for s in result.suggestions] == [(1, 2), (2, 3)]
    assert result.suggestions[0].track_a.id == 1 and result.suggestions[0].track_b.id == 2


# ---------------------------------------------------------------------------
# Vocal-region cross-check
# ---------------------------------------------------------------------------


def test_vocal_regions_filtered_to_overlap_window_not_full_zone():
    # A's tail zone is 150_000-200_000, but overlap_ms is only 20_000 (B's
    # head is the limiter), so the real overlap window is 150_000-170_000.
    # A vocal region inside the raw tail zone but outside that narrower
    # window must NOT appear.
    a = _scorable_track(1, "A", vocal_windows=((180_000.0, 185_000.0),))  # in tail zone, outside overlap window
    b = _scorable_track(2, "B")
    s = suggest_transitions([a, b]).suggestions[0]
    assert s.overlap_ms == 20_000.0
    assert s.vocal_regions_a == []


def test_vocal_regions_populated_when_present_in_window():
    a = _scorable_track(1, "A", vocal_windows=((155_000.0, 160_000.0),))  # inside the 150_000-170_000 overlap window
    b = _scorable_track(2, "B", vocal_windows=((2_000.0, 7_000.0),))  # inside the 0-20_000 overlap window
    s = suggest_transitions([a, b]).suggestions[0]
    assert len(s.vocal_regions_a) == 1
    assert len(s.vocal_regions_b) == 1


def test_vocal_regions_on_b_side_filtered_to_overlap_window_not_full_zone():
    # Mirror of the A-side test, using the a_tail-limited scenario so B's
    # overlap window (0-5_000) is narrower than its raw head zone (0-20_000).
    short_tail_phrases = [
        _phrase("Intro", 0.0, 20_000.0),
        _phrase("Chorus", 20_000.0, 175_000.0),
        _phrase("Outro", 195_000.0, 5_000.0),
    ]
    a = _track(1, "A", phrases=short_tail_phrases, vocal_track=_vocal_track(200_000.0))
    b = _scorable_track(2, "B", vocal_windows=((10_000.0, 15_000.0),))  # in head zone, outside overlap window
    s = suggest_transitions([a, b]).suggestions[0]
    assert s.overlap_ms == 5_000.0
    assert s.vocal_regions_b == []


def test_vocal_regions_none_when_vocal_track_missing():
    a = _scorable_track(1, "A", no_vocal_track=True)
    b = _scorable_track(2, "B", no_vocal_track=True)
    s = suggest_transitions([a, b]).suggestions[0]
    assert s.vocal_regions_a is None
    assert s.vocal_regions_b is None


def test_vocal_regions_empty_list_when_present_but_no_overlap():
    a = _scorable_track(1, "A")  # has vocal_track, but no regions anywhere
    b = _scorable_track(2, "B")
    s = suggest_transitions([a, b]).suggestions[0]
    assert s.vocal_regions_a == []
    assert s.vocal_regions_b == []


def test_missing_vocal_track_does_not_block_the_base_suggestion():
    # Key divergence from clash.py: no vocal_track data is NOT a gate here.
    a = _scorable_track(1, "A", no_vocal_track=True)
    b = _scorable_track(2, "B", no_vocal_track=True)
    result = suggest_transitions([a, b])
    assert len(result.suggestions) == 1
    assert result.unscorable == []


# ---------------------------------------------------------------------------
# Unscorable tracks
# ---------------------------------------------------------------------------


def test_no_phrase_data_is_unscorable():
    track = _track(1, "No Phrases", phrases=[], vocal_track=_vocal_track(200_000.0))
    result = suggest_transitions([track, _scorable_track(2, "B")])
    assert [u.reason for u in result.unscorable] == ["no_phrase_data"]
    assert result.suggestions == []


def test_missing_intro_phrase_is_unscorable():
    phrases = [p for p in _STANDARD_PHRASES if p.label != "Intro"]
    track = _track(1, "No Intro", phrases=phrases, vocal_track=_vocal_track(200_000.0))
    result = suggest_transitions([track, _scorable_track(2, "B")])
    assert [u.reason for u in result.unscorable] == ["no_intro_phrase"]


def test_missing_outro_phrase_is_unscorable():
    phrases = [p for p in _STANDARD_PHRASES if p.label != "Outro"]
    track = _track(1, "No Outro", phrases=phrases, vocal_track=_vocal_track(200_000.0))
    result = suggest_transitions([track, _scorable_track(2, "B")])
    assert [u.reason for u in result.unscorable] == ["no_outro_phrase"]


def test_unscorable_middle_track_excludes_both_pairs_touching_it():
    a = _scorable_track(1, "A")
    b = _track(2, "B", phrases=[], vocal_track=_vocal_track(200_000.0))  # unscorable
    c = _scorable_track(3, "C")
    result = suggest_transitions([a, b, c])
    assert result.suggestions == []
    assert len(result.unscorable) == 1
    assert result.scanned_pairs == 2


# ---------------------------------------------------------------------------
# scanned_pairs / invariants / min_vocal_region_ms
# ---------------------------------------------------------------------------


def test_scanned_pairs_zero_tracks():
    assert suggest_transitions([]).scanned_pairs == 0


def test_scanned_pairs_one_track():
    assert suggest_transitions([_scorable_track(1, "A")]).scanned_pairs == 0


def test_scanned_pairs_n_tracks():
    tracks = [_scorable_track(i, str(i)) for i in range(1, 6)]
    assert suggest_transitions(tracks).scanned_pairs == 4


def test_every_scorable_pair_produces_exactly_one_suggestion():
    # 4 tracks, index 1 (0-based) unscorable -> pairs (0,1) and (1,2) are
    # excluded, only pair (2,3) remains scorable.
    a = _scorable_track(1, "A")
    b = _track(2, "B", phrases=[], vocal_track=_vocal_track(200_000.0))
    c = _scorable_track(3, "C")
    d = _scorable_track(4, "D")
    result = suggest_transitions([a, b, c, d])
    assert result.scanned_pairs == 3
    assert len(result.suggestions) == 1
    assert (result.suggestions[0].position_a, result.suggestions[0].position_b) == (3, 4)


def test_min_vocal_region_ms_passthrough():
    # A 500ms burst -- well under the default 2000ms floor, so no region
    # by default, but one once the threshold is lowered below 500ms.
    a = _scorable_track(1, "A", vocal_windows=((155_000.0, 155_500.0),))
    b = _scorable_track(2, "B")
    assert suggest_transitions([a, b]).suggestions[0].vocal_regions_a == []
    result = suggest_transitions([a, b], min_vocal_region_ms=400.0)
    assert len(result.suggestions[0].vocal_regions_a) == 1


def test_default_min_vocal_region_ms_matches_strategy():
    assert DEFAULT_MIN_VOCAL_REGION_MS == 2000.0


# ---------------------------------------------------------------------------
# Real-library test. Structural properties only, never hardcoded counts --
# mirrors test_clash.py's own established convention. An anchored regression
# test against a specific known-adjacent real pair (like clash's Ladbroke
# Grove -> I Go To Work) is deliberately deferred to live verification during
# rollout -- see the plan's own note on this.
# ---------------------------------------------------------------------------


def _load_ordered(playlist_id) -> list[Track]:
    """Mirrors cli.py's/server.py's own reordering step exactly: the
    cheap, TrackNo-sorted list_playlist_tracks() gives the real playlist
    order; load_playlist_tracks() alone does not (see clash.py's/
    transition.py's own module docstrings)."""
    from djcues.db import list_playlist_tracks, load_playlist_tracks

    summaries = list_playlist_tracks(playlist_id)
    loaded = load_playlist_tracks(playlist_id)
    by_id = {str(t.id): t for t in loaded}
    return [by_id[str(s.id)] for s in summaries if str(s.id) in by_id]


@requires_rekordbox
def test_transition_real_library_structural_properties():
    from djcues.db import find_playlist

    playlist = find_playlist("Tech House")
    assert playlist is not None
    tracks = _load_ordered(playlist.ID)
    assert len(tracks) > 0

    result = suggest_transitions(tracks)

    assert result.scanned_pairs == len(tracks) - 1
    assert len(result.suggestions) + sum(
        1 for i in range(len(tracks) - 1)
        if any(u.track.id in (tracks[i].id, tracks[i + 1].id) for u in result.unscorable)
    ) == result.scanned_pairs
    for s in result.suggestions:
        assert s.position_b == s.position_a + 1
        assert s.overlap_ms >= 0.0
        assert 0.0 <= s.mix_out_ms <= s.track_a.duration_ms
        assert 0.0 <= s.mix_in_ms <= s.track_b.duration_ms
    assert all(u.reason in ("no_phrase_data", "no_intro_phrase", "no_outro_phrase") for u in result.unscorable)
