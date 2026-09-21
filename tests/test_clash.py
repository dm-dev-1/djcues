"""Tests for djcues.clash -- vocal-clash detection."""

from __future__ import annotations

from djcues.clash import find_vocal_clashes
from djcues.models import BeatGrid, Phrase, Track
from djcues.strategy import DEFAULT_MIN_VOCAL_REGION_MS
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
    id_: int, title: str, *, phrases: list[Phrase], vocal_track: list[int] | None, duration_ms: float = 200_000.0,
) -> Track:
    return Track(
        id=id_, title=title, artist="Artist", bpm=128.0, duration_ms=duration_ms,
        analysis_path="", cues=[], phrases=phrases, beat_grid=BeatGrid(first_beat_ms=0.0, bpm=128.0),
        vocal_track=vocal_track,
    )


# Standard 200s structure: Intro 0-20s (head zone), Chorus 20-150s, Outro
# 150-200s (tail zone). Reused by every scorable-track test below.
_STANDARD_PHRASES = [
    _phrase("Intro", 0.0, 20_000.0),
    _phrase("Chorus", 20_000.0, 130_000.0),
    _phrase("Outro", 150_000.0, 50_000.0),
]


def _scorable_track(id_: int, title: str, *, vocal_windows: tuple[tuple[float, float], ...] = ()) -> Track:
    return _track(
        id_, title, phrases=list(_STANDARD_PHRASES), vocal_track=_vocal_track(200_000.0, vocal_windows),
    )


# ---------------------------------------------------------------------------
# find_vocal_clashes -- core pairwise logic
# ---------------------------------------------------------------------------


def test_clash_flagged_when_both_zones_have_overlapping_vocals():
    a = _scorable_track(1, "A", vocal_windows=((160_000.0, 165_000.0),))  # inside A's tail zone
    b = _scorable_track(2, "B", vocal_windows=((2_000.0, 7_000.0),))  # inside B's head zone
    result = find_vocal_clashes([a, b])
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f.track_a.id == 1 and f.track_b.id == 2
    assert f.position_a == 1 and f.position_b == 2
    assert len(f.regions_a) == 1 and len(f.regions_b) == 1


def test_not_flagged_when_only_a_has_tail_vocals():
    a = _scorable_track(1, "A", vocal_windows=((160_000.0, 165_000.0),))
    b = _scorable_track(2, "B")  # clean
    result = find_vocal_clashes([a, b])
    assert result.findings == []


def test_not_flagged_when_only_b_has_head_vocals():
    a = _scorable_track(1, "A")  # clean
    b = _scorable_track(2, "B", vocal_windows=((2_000.0, 7_000.0),))
    result = find_vocal_clashes([a, b])
    assert result.findings == []


def test_not_flagged_when_vocal_region_exists_outside_the_relevant_zone():
    # A has vocals in the middle of the track (well outside its 150-200s
    # tail zone); B has vocals in its head zone. A's own tail is clean,
    # so this must not be flagged even though B's head isn't.
    a = _scorable_track(1, "A", vocal_windows=((80_000.0, 85_000.0),))
    b = _scorable_track(2, "B", vocal_windows=((2_000.0, 7_000.0),))
    result = find_vocal_clashes([a, b])
    assert result.findings == []


def test_only_literally_adjacent_pairs_are_compared():
    # A (tail vocals) and C (head vocals) would "clash" if adjacent, but
    # B sits between them and is clean on both sides -- A-C is never a
    # checked pair, so no finding should appear.
    a = _scorable_track(1, "A", vocal_windows=((160_000.0, 165_000.0),))
    b = _scorable_track(2, "B")
    c = _scorable_track(3, "C", vocal_windows=((2_000.0, 7_000.0),))
    result = find_vocal_clashes([a, b, c])
    assert result.findings == []
    assert result.scanned_pairs == 2


def test_multiple_independent_findings_with_correct_positions():
    a = _scorable_track(1, "A", vocal_windows=((160_000.0, 165_000.0),))
    b = _scorable_track(2, "B", vocal_windows=((2_000.0, 7_000.0),))
    c = _scorable_track(3, "C")  # clean pair in the middle
    d = _scorable_track(4, "D", vocal_windows=((160_000.0, 165_000.0),))
    e = _scorable_track(5, "E", vocal_windows=((2_000.0, 7_000.0),))
    result = find_vocal_clashes([a, b, c, d, e])
    assert len(result.findings) == 2
    assert (result.findings[0].position_a, result.findings[0].position_b) == (1, 2)
    assert (result.findings[1].position_a, result.findings[1].position_b) == (4, 5)


# ---------------------------------------------------------------------------
# Unscorable tracks
# ---------------------------------------------------------------------------


def test_no_phrase_data_is_unscorable():
    track = _track(1, "No Phrases", phrases=[], vocal_track=_vocal_track(200_000.0))
    result = find_vocal_clashes([track, _scorable_track(2, "B")])
    assert [u.reason for u in result.unscorable] == ["no_phrase_data"]


def test_no_vocal_data_is_unscorable():
    track = _track(1, "No Vocal", phrases=list(_STANDARD_PHRASES), vocal_track=None)
    result = find_vocal_clashes([track, _scorable_track(2, "B")])
    assert [u.reason for u in result.unscorable] == ["no_vocal_data"]


def test_missing_intro_phrase_is_unscorable():
    phrases = [p for p in _STANDARD_PHRASES if p.label != "Intro"]
    track = _track(1, "No Intro", phrases=phrases, vocal_track=_vocal_track(200_000.0))
    result = find_vocal_clashes([track, _scorable_track(2, "B")])
    assert [u.reason for u in result.unscorable] == ["no_intro_phrase"]


def test_missing_outro_phrase_is_unscorable():
    phrases = [p for p in _STANDARD_PHRASES if p.label != "Outro"]
    track = _track(1, "No Outro", phrases=phrases, vocal_track=_vocal_track(200_000.0))
    result = find_vocal_clashes([track, _scorable_track(2, "B")])
    assert [u.reason for u in result.unscorable] == ["no_outro_phrase"]


def test_unscorable_middle_track_excludes_both_pairs_touching_it():
    a = _scorable_track(1, "A", vocal_windows=((160_000.0, 165_000.0),))
    b = _track(2, "B", phrases=[], vocal_track=_vocal_track(200_000.0))  # unscorable
    c = _scorable_track(3, "C", vocal_windows=((2_000.0, 7_000.0),))
    result = find_vocal_clashes([a, b, c])
    assert result.findings == []
    assert len(result.unscorable) == 1
    assert result.scanned_pairs == 2


# ---------------------------------------------------------------------------
# scanned_pairs / min_vocal_region_ms
# ---------------------------------------------------------------------------


def test_scanned_pairs_zero_tracks():
    assert find_vocal_clashes([]).scanned_pairs == 0


def test_scanned_pairs_one_track():
    assert find_vocal_clashes([_scorable_track(1, "A")]).scanned_pairs == 0


def test_scanned_pairs_n_tracks():
    tracks = [_scorable_track(i, str(i)) for i in range(1, 6)]
    assert find_vocal_clashes(tracks).scanned_pairs == 4


def test_min_vocal_region_ms_passthrough():
    # A 500ms burst -- well under the default 2000ms floor, so no finding
    # by default, but a finding once the threshold is lowered below 500ms.
    a = _scorable_track(1, "A", vocal_windows=((160_000.0, 160_500.0),))
    b = _scorable_track(2, "B", vocal_windows=((2_000.0, 2_500.0),))
    assert find_vocal_clashes([a, b]).findings == []
    result = find_vocal_clashes([a, b], min_vocal_region_ms=400.0)
    assert len(result.findings) == 1


def test_default_min_vocal_region_ms_matches_strategy():
    assert DEFAULT_MIN_VOCAL_REGION_MS == 2000.0


# ---------------------------------------------------------------------------
# Real-library tests. Structural properties only, never hardcoded counts --
# this project's own established lesson from a stale Tech House track-count
# assertion (learned repeatedly this session). The two named-pair regression
# tests are the real correctness guarantee, grounded in real, already-
# verified data rather than synthetic assumptions -- mirrors audit.py's own
# Dreamin'/Time-To-Pretend precedent exactly.
# ---------------------------------------------------------------------------


def _load_ordered(playlist_id) -> list[Track]:
    """Mirrors cli.py's/server.py's own reordering step exactly: the
    cheap, TrackNo-sorted list_playlist_tracks() gives the real playlist
    order; load_playlist_tracks() alone does not (see clash.py's own
    module docstring)."""
    from djcues.db import list_playlist_tracks, load_playlist_tracks

    summaries = list_playlist_tracks(playlist_id)
    loaded = load_playlist_tracks(playlist_id)
    by_id = {str(t.id): t for t in loaded}
    return [by_id[str(s.id)] for s in summaries if str(s.id) in by_id]


@requires_rekordbox
def test_clash_real_library_structural_properties():
    from djcues.db import find_playlist

    playlist = find_playlist("Nu Disco - Disco House")
    assert playlist is not None
    tracks = _load_ordered(playlist.ID)
    assert len(tracks) > 0

    result = find_vocal_clashes(tracks)

    assert result.scanned_pairs == len(tracks) - 1
    for f in result.findings:
        assert f.position_b == f.position_a + 1
        assert f.regions_a and f.regions_b
    assert all(
        u.reason in ("no_phrase_data", "no_vocal_data", "no_intro_phrase", "no_outro_phrase")
        for u in result.unscorable
    )


@requires_rekordbox
def test_clash_real_library_flags_known_pairs():
    # Real, CORRECTLY-ordered pairs in this library (verified live against
    # list_playlist_tracks()'s real TrackNo order, not load_playlist_
    # tracks()'s unordered rows -- an earlier verification pass during
    # planning found different-looking "real" pairs by trusting
    # load_playlist_tracks()'s row order directly, which turned out to be
    # exactly the ordering bug this feature's whole design works around;
    # re-verified after the fix and these two are genuinely adjacent).
    from djcues.db import find_playlist

    playlist = find_playlist("Tech House")
    tracks = _load_ordered(playlist.ID)
    result = find_vocal_clashes(tracks)

    found_pairs = {(f.track_a.title, f.track_b.title) for f in result.findings}
    assert any(
        "Ladbroke Grove" in a and "I Go To Work" in b for a, b in found_pairs
    ), found_pairs
    assert any(
        "Supersonic" in a and "Cha Cha Slide" in b for a, b in found_pairs
    ), found_pairs
