"""Tests for djcues.tagging -- the pure smart-tagging rules."""

from __future__ import annotations

import json

import pytest

from djcues.harmony import _ENCRYPTED_METADATA_PREFIX
from djcues.models import BeatGrid, Phrase, RawBeatGridEntry, Track, TrackSummary, WaveformPoint
from djcues.tagging import (
    CATEGORY_NAMES,
    DEFAULT_THRESHOLDS,
    MIN_CALIBRATION_TRACKS,
    TAG_CATALOG,
    AnalysisFeatures,
    TagThresholds,
    category_of,
    compute_energy_cutpoints,
    derive_tags,
    energy_band,
    extract_features,
    is_calibration_eligible,
    key_status,
)

_FRAME_MS = 1024 / 22050 * 1000
_BAR_MS = 1875.0  # one bar at 128 BPM


def _features(**overrides) -> AnalysisFeatures:
    """A fully-populated, boring track: nothing here should trigger any
    tag except what a test overrides."""
    base = dict(
        track_id="1", duration_ms=240_000.0, bpm=128.0, mean_energy=0.5, peak_energy=0.9,
        intro_bars=8.0, outro_bars=8.0, vocal_pct=20.0, vocal_intro=[], vocal_outro=[],
        tempo_varies=False, grid_consistent=True,
    )
    base.update(overrides)
    return AnalysisFeatures(**base)


def _tags_of(result) -> set[str]:
    return {d.tag for d in result.decisions}


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------


def test_catalog_shape_matches_the_plan():
    assert CATEGORY_NAMES == ("Energy", "Mix In", "Mix Out", "Vocals", "Check")
    all_tags = [t for c in TAG_CATALOG for t in c.tags]
    assert len(all_tags) == 16
    assert len(set(all_tags)) == 16  # no name used twice across categories


def test_category_of():
    assert category_of("Energy 3") == "Energy"
    assert category_of("Vocal Outro") == "Mix Out"
    assert category_of("Instrumental") == "Vocals"
    assert category_of("No Key") == "Check"
    assert category_of("nonsense") is None


def test_only_energy_and_vocals_are_single_select():
    assert {c.name for c in TAG_CATALOG if c.select == "single"} == {"Energy", "Vocals"}


# ---------------------------------------------------------------------------
# energy cutpoints / bands
# ---------------------------------------------------------------------------


def test_cutpoints_split_into_five_equal_count_bands():
    values = [i / 100 for i in range(1, 101)]
    cutpoints = compute_energy_cutpoints(values)
    assert len(cutpoints) == 4
    assert cutpoints == sorted(cutpoints)
    counts = [0] * 5
    for v in values:
        counts[energy_band(v, cutpoints) - 1] += 1
    assert max(counts) - min(counts) <= 2  # ~20 each


def test_cutpoints_are_order_independent():
    values = [0.9, 0.1, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 1.0]
    assert compute_energy_cutpoints(values) == compute_energy_cutpoints(sorted(values))


def test_cutpoints_refuse_too_few_tracks():
    with pytest.raises(ValueError, match="at least"):
        compute_energy_cutpoints([0.1, 0.2, 0.3, 0.4][: MIN_CALIBRATION_TRACKS - 1])


def test_cutpoints_with_all_tied_values_do_not_crash():
    cutpoints = compute_energy_cutpoints([0.5] * 20)
    assert cutpoints == [0.5, 0.5, 0.5, 0.5]
    # everything lands in the top band -- degenerate but well-defined
    assert energy_band(0.5, cutpoints) == 5


def test_energy_band_edges():
    cutpoints = [0.2, 0.4, 0.6, 0.8]
    assert energy_band(0.0, cutpoints) == 1
    assert energy_band(0.19, cutpoints) == 1
    assert energy_band(0.2, cutpoints) == 2  # exactly on a cutpoint -> higher band
    assert energy_band(0.79, cutpoints) == 4
    assert energy_band(0.8, cutpoints) == 5
    assert energy_band(1.0, cutpoints) == 5


# ---------------------------------------------------------------------------
# derive_tags: Energy
# ---------------------------------------------------------------------------

_CUTS = [0.2, 0.4, 0.6, 0.8]


def test_energy_tag_and_evidence():
    result = derive_tags(_features(mean_energy=0.65), cutpoints=_CUTS)
    energy = [d for d in result.decisions if d.category == "Energy"]
    assert [d.tag for d in energy] == ["Energy 4"]
    assert "0.65" in energy[0].evidence and "band 4" in energy[0].evidence


def test_energy_skipped_without_calibration():
    result = derive_tags(_features(), cutpoints=None)
    assert [(s.category, s.reason) for s in result.skipped] == [("Energy", "no_calibration")]
    assert not result.tags_in("Energy")


def test_energy_skipped_without_energy_data():
    result = derive_tags(_features(mean_energy=None, peak_energy=None), cutpoints=_CUTS)
    assert ("Energy", "no_energy_data") in [(s.category, s.reason) for s in result.skipped]


@pytest.mark.parametrize("overrides", [{"duration_ms": 30_000.0}, {"bpm": 0.0}])
def test_loops_and_samples_get_no_energy_tag(overrides):
    result = derive_tags(_features(**overrides), cutpoints=_CUTS)
    assert ("Energy", "sample_or_loop") in [(s.category, s.reason) for s in result.skipped]
    assert not result.tags_in("Energy")


def test_calibration_eligibility_boundary():
    assert is_calibration_eligible(_features(duration_ms=60_000.0))
    assert not is_calibration_eligible(_features(duration_ms=59_999.0))
    assert not is_calibration_eligible(_features(mean_energy=None))


# ---------------------------------------------------------------------------
# derive_tags: Mix In / Mix Out
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bars,expected", [
    (16.0, {"Long Intro"}), (24.0, {"Long Intro"}), (15.0, set()), (8.0, set()),
    (3.0, set()), (2.0, {"Short Intro"}), (0.0, {"Short Intro"}),
])
def test_intro_length_boundaries(bars, expected):
    result = derive_tags(_features(intro_bars=bars), cutpoints=_CUTS)
    assert set(result.tags_in("Mix In")) == expected


@pytest.mark.parametrize("bars,expected", [
    (16.0, {"Long Outro"}), (15.0, set()), (5.0, set()), (4.0, {"Short Outro"}), (2.0, {"Short Outro"}),
])
def test_outro_length_boundaries(bars, expected):
    result = derive_tags(_features(outro_bars=bars), cutpoints=_CUTS)
    assert set(result.tags_in("Mix Out")) == expected


def test_vocal_intro_and_outro_from_overlapping_regions():
    result = derive_tags(
        _features(vocal_intro=[(5000.0, 9000.0)], vocal_outro=[(170_000.0, 174_000.0)]), cutpoints=_CUTS,
    )
    assert "Vocal Intro" in result.tags_in("Mix In")
    assert "Vocal Outro" in result.tags_in("Mix Out")
    intro_evidence = next(d.evidence for d in result.decisions if d.tag == "Vocal Intro")
    assert "0:05-0:09" in intro_evidence


def test_no_vocal_data_means_no_vocal_tag_but_no_skip_either():
    # Length tags can still be evaluated without vocal data, so Mix In/Out
    # aren't skipped -- only the Vocal tag itself is unavailable.
    result = derive_tags(_features(intro_bars=16.0, vocal_intro=None, vocal_outro=None), cutpoints=_CUTS)
    assert result.tags_in("Mix In") == ["Long Intro"]
    assert not any(s.category in ("Mix In", "Mix Out") for s in result.skipped)


def test_mix_categories_skipped_without_phrases():
    result = derive_tags(_features(intro_bars=None, outro_bars=None), cutpoints=_CUTS)
    reasons = {(s.category, s.reason) for s in result.skipped}
    assert ("Mix In", "no_intro_phrase") in reasons
    assert ("Mix Out", "no_outro_phrase") in reasons


# ---------------------------------------------------------------------------
# derive_tags: Vocals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pct,expected", [
    (0.0, ["Instrumental"]), (2.9, ["Instrumental"]), (3.0, []), (20.0, []),
    (44.9, []), (45.0, ["Vocal-Led"]), (90.0, ["Vocal-Led"]),
])
def test_vocal_coverage_boundaries(pct, expected):
    result = derive_tags(_features(vocal_pct=pct), cutpoints=_CUTS)
    assert result.tags_in("Vocals") == expected


def test_vocals_skipped_without_vocal_data():
    result = derive_tags(_features(vocal_pct=None), cutpoints=_CUTS)
    assert ("Vocals", "no_vocal_data") in [(s.category, s.reason) for s in result.skipped]


# ---------------------------------------------------------------------------
# derive_tags: Check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reason,fragment", [
    ("no_key", "no key tag"),
    ("non_camelot_key", "'Dm'"),
])
def test_no_key_tag_and_evidence(reason, fragment):
    result = derive_tags(_features(), key_reason=reason, key_raw="Dm", cutpoints=_CUTS)
    check = [d for d in result.decisions if d.category == "Check"]
    assert [d.tag for d in check] == ["No Key"]
    assert fragment in check[0].evidence


def test_grid_tags():
    result = derive_tags(_features(tempo_varies=True, grid_consistent=False), cutpoints=_CUTS)
    assert set(result.tags_in("Check")) == {"Tempo Varies", "Grid Suspect"}


def test_unknown_grid_state_adds_no_check_tags():
    result = derive_tags(_features(tempo_varies=None, grid_consistent=None), cutpoints=_CUTS)
    assert result.tags_in("Check") == []


def test_a_healthy_track_gets_only_its_energy_tag():
    result = derive_tags(_features(), cutpoints=_CUTS)
    assert _tags_of(result) == {"Energy 3"}
    assert result.skipped == []


# ---------------------------------------------------------------------------
# derive_tags: category filtering and independence
# ---------------------------------------------------------------------------


def test_categories_filter_limits_evaluation_and_skips():
    result = derive_tags(
        _features(intro_bars=None, vocal_pct=None), cutpoints=None, categories=["Vocals"],
    )
    # Energy/Mix In weren't requested, so their missing data is not reported.
    assert [(s.category, s.reason) for s in result.skipped] == [("Vocals", "no_vocal_data")]


def test_every_decision_is_a_catalog_tag_in_its_own_category():
    result = derive_tags(
        _features(
            mean_energy=0.9, intro_bars=16.0, outro_bars=2.0, vocal_pct=60.0,
            vocal_intro=[(0.0, 4000.0)], vocal_outro=[(1.0, 2.0)],
            tempo_varies=True, grid_consistent=False,
        ),
        key_reason="no_key", cutpoints=_CUTS,
    )
    assert result.decisions
    for d in result.decisions:
        assert category_of(d.tag) == d.category


def test_single_select_categories_never_emit_two_tags():
    for pct in (0.0, 10.0, 50.0, 99.0):
        result = derive_tags(_features(vocal_pct=pct), cutpoints=_CUTS)
        assert len(result.tags_in("Vocals")) <= 1
        assert len(result.tags_in("Energy")) <= 1


def test_custom_thresholds_are_respected():
    strict = TagThresholds(long_intro_bars=8.0)
    assert derive_tags(_features(intro_bars=8.0), cutpoints=_CUTS, thresholds=strict).tags_in("Mix In") == ["Long Intro"]
    assert derive_tags(_features(intro_bars=8.0), cutpoints=_CUTS, thresholds=DEFAULT_THRESHOLDS).tags_in("Mix In") == []


# ---------------------------------------------------------------------------
# key_status (reuses audit's own logic)
# ---------------------------------------------------------------------------


def _summary(key, title="T", artist="A") -> TrackSummary:
    return TrackSummary(id="1", track_no=None, title=title, artist=artist, bpm=128.0, duration_ms=1.0, key=key)


def test_key_status_usable_key():
    assert key_status(_summary("8A")) is None


def test_key_status_reasons():
    assert key_status(_summary(None)) == "no_key"
    assert key_status(_summary("")) == "no_key"
    assert key_status(_summary("Dm")) == "non_camelot_key"


def test_encrypted_title_with_a_real_key_is_not_no_key():
    # Regression: audit's precedence reports any encrypted-title track as
    # "encrypted_metadata" before checking its key. On the real library 290
    # of 460 such tracks have a valid Camelot key -- tagging them "No Key"
    # would have been wrong. The key decides, not the title.
    encrypted = _ENCRYPTED_METADATA_PREFIX + "xyz"
    assert key_status(_summary("8A", title=encrypted)) is None
    assert key_status(_summary(None, title=encrypted)) == "no_key"


# ---------------------------------------------------------------------------
# extract_features (synthetic Tracks)
# ---------------------------------------------------------------------------


def _phrase(label: str, bars: int, position_ms: float, beat_start: int = 1) -> Phrase:
    return Phrase(
        beat_start=beat_start, beat_end=beat_start + 4 * bars, kind=1, label=label,
        position_ms=position_ms, duration_ms=bars * _BAR_MS,
    )


def _vocal_array(total_ms: float, windows=()) -> list[int]:
    n = int(total_ms / _FRAME_MS) + 1
    vt = [0] * n
    for start_ms, end_ms in windows:
        for i in range(int(start_ms / _FRAME_MS), min(int(end_ms / _FRAME_MS), n)):
            vt[i] = 4
    return vt


def _track(*, phrases=None, vocal_windows=None, waveform_height=0.6, with_waveform=True) -> Track:
    # Intro 8 bars (0-15s), Chorus 80 bars (15-165s), Outro 8 bars (165-180s)
    if phrases is None:
        phrases = [
            _phrase("Intro", 8, 0.0, beat_start=1),
            _phrase("Chorus", 80, 8 * _BAR_MS, beat_start=33),
            _phrase("Outro", 8, 88 * _BAR_MS, beat_start=353),
        ]
    duration = 96 * _BAR_MS  # 180s
    return Track(
        id=7, title="T", artist="A", bpm=128.0, duration_ms=duration, analysis_path="", cues=[],
        phrases=phrases, beat_grid=BeatGrid(first_beat_ms=0.0, bpm=128.0),
        waveform=[WaveformPoint(height=waveform_height, red=4, green=4, blue=4) for _ in range(50)]
        if with_waveform else None,
        vocal_track=None if vocal_windows is None else _vocal_array(duration, vocal_windows),
    )


def test_extract_intro_outro_bars_from_phrase_beat_length():
    f = extract_features(_track())
    assert f.track_id == "7"
    assert f.intro_bars == 8.0
    assert f.outro_bars == 8.0


def test_extract_energy():
    f = extract_features(_track(waveform_height=0.6))
    assert f.mean_energy == pytest.approx(0.6)
    assert f.peak_energy == pytest.approx(0.6)


def test_extract_without_waveform_has_no_energy():
    f = extract_features(_track(with_waveform=False))
    assert f.mean_energy is None and f.peak_energy is None


def test_extract_vocal_regions_overlapping_intro_and_outro():
    f = extract_features(_track(vocal_windows=[(5000.0, 9000.0), (170_000.0, 174_000.0)]))
    assert f.vocal_pct == pytest.approx(100 * 8000 / 180_000, abs=0.5)
    assert f.vocal_intro is not None and len(f.vocal_intro) == 1
    assert f.vocal_outro is not None and len(f.vocal_outro) == 1


def test_extract_vocals_in_the_middle_only_leave_intro_and_outro_clear():
    f = extract_features(_track(vocal_windows=[(60_000.0, 90_000.0)]))
    assert f.vocal_intro == [] and f.vocal_outro == []  # checked, genuinely clear
    assert f.vocal_pct > 10


def test_extract_without_vocal_data_is_none_not_empty():
    f = extract_features(_track(vocal_windows=None))
    assert f.vocal_pct is None and f.vocal_intro is None and f.vocal_outro is None


def test_extract_missing_intro_phrase():
    phrases = [_phrase("Chorus", 80, 0.0), _phrase("Outro", 8, 80 * _BAR_MS, beat_start=321)]
    f = extract_features(_track(phrases=phrases, vocal_windows=[(1000.0, 5000.0)]))
    assert f.intro_bars is None
    assert f.vocal_intro is None  # can't check an intro that doesn't exist
    assert f.outro_bars == 8.0


def _grid(n=40, bpm=128.0, drift_ms_per_beat=0.0) -> list[RawBeatGridEntry]:
    gap = 60_000.0 / bpm
    return [RawBeatGridEntry(beat_in_bar=(i % 4) + 1, bpm=bpm, time_ms=i * (gap + drift_ms_per_beat)) for i in range(n)]


def test_extract_grid_health_consistent():
    f = extract_features(_track(), _grid())
    assert f.grid_consistent is True and f.tempo_varies is False


def test_extract_grid_health_inconsistent():
    f = extract_features(_track(), _grid(drift_ms_per_beat=10.0))
    assert f.grid_consistent is False


def test_extract_without_grid_leaves_health_unknown():
    f = extract_features(_track(), None)
    assert f.grid_consistent is None and f.tempo_varies is None


# ---------------------------------------------------------------------------
# AnalysisFeatures serialization (it's what the cache stores)
# ---------------------------------------------------------------------------


def test_features_round_trip_through_json():
    f = _features(vocal_intro=[(1.0, 2.0)], vocal_outro=[], tempo_varies=None, mean_energy=None)
    restored = AnalysisFeatures.from_dict(json.loads(json.dumps(f.to_dict())))
    assert restored == f
    assert restored.vocal_intro == [(1.0, 2.0)]  # tuples restored, not lists
    assert restored.vocal_outro == []
