"""Tests for djcues.viz -- the HTML timeline visualizer. Every function
here is pure (no I/O), so these run for real against synthetic Track/
CueProposal fixtures rather than mocking anything, same approach as
test_review.py.
"""

from __future__ import annotations

import pytest

from djcues.constants import CUE_SYSTEM_BY_PAD
from djcues.models import BeatGrid, CueProposal, CuePoint, Phrase, Track, WaveformPoint
from djcues.strategy import CueStrategy
from djcues.viz import (
    CUE_COLORS,
    PHRASE_COLORS,
    _format_time,
    _render_confidence_bars,
    _render_cue_markers,
    _render_phrase_bar,
    _render_timeline_section,
    _render_track_body,
    _render_waveform,
    render_playlist,
    render_timeline,
)

_FRAME_MS = 1024 / 22050 * 1000  # matches _render_waveform's own constant


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def beat_grid() -> BeatGrid:
    return BeatGrid(first_beat_ms=77.0, bpm=128.0)


@pytest.fixture
def phrases(beat_grid: BeatGrid) -> list[Phrase]:
    raw = [(1, 1, "Intro"), (33, 2, "Up"), (145, 5, "Chorus"), (401, 6, "Outro")]
    end_beat = 433
    result = []
    for i, (beat, kind, label) in enumerate(raw):
        next_beat = raw[i + 1][0] if i + 1 < len(raw) else end_beat
        pos = beat_grid.beat_to_ms(beat)
        end_pos = beat_grid.beat_to_ms(next_beat)
        result.append(Phrase(
            beat_start=beat, beat_end=next_beat, kind=kind, label=label,
            position_ms=pos, duration_ms=end_pos - pos,
        ))
    return result


@pytest.fixture
def track(beat_grid: BeatGrid, phrases: list[Phrase]) -> Track:
    return Track(
        id=1, title="Test Track", artist="Test Artist", bpm=128.0,
        duration_ms=240_000.0, analysis_path="", cues=[], phrases=phrases,
        beat_grid=beat_grid,
    )


@pytest.fixture
def proposal(track: Track) -> CueProposal:
    return CueStrategy().propose(track)


@pytest.fixture
def track_with_existing_cues(beat_grid: BeatGrid, phrases: list[Phrase]) -> Track:
    existing = [
        CuePoint(kind=1, position_ms=77.0, loop_end_ms=None, color_table_index=18, color=-1, comment="First Beat"),
        CuePoint(kind=2, position_ms=1000.0, loop_end_ms=8500.0, color_table_index=18, color=-1, comment="Loop In"),
        CuePoint(kind=0, position_ms=500.0, loop_end_ms=None, color_table_index=4, color=-1, comment="First Beat"),
    ]
    return Track(
        id=2, title="Existing Cues Track", artist="Test Artist", bpm=128.0,
        duration_ms=240_000.0, analysis_path="", cues=existing, phrases=phrases,
        beat_grid=beat_grid,
    )


# ---------------------------------------------------------------------------
# _format_time
# ---------------------------------------------------------------------------


def test_format_time_under_a_minute():
    assert _format_time(5_500) == "0:05.5"


def test_format_time_over_a_minute():
    assert _format_time(90_250) == "1:30.2"


# ---------------------------------------------------------------------------
# _render_phrase_bar
# ---------------------------------------------------------------------------


class TestRenderPhraseBar:
    def test_position_and_width_percentages(self):
        phrases = [Phrase(beat_start=1, beat_end=2, kind=1, label="Intro", position_ms=100.0, duration_ms=400.0)]
        result = _render_phrase_bar(phrases, total_ms=1000.0)
        assert "left:10.0000%" in result
        assert "width:40.0000%" in result

    def test_known_label_uses_its_color(self):
        phrases = [Phrase(beat_start=1, beat_end=2, kind=5, label="Chorus", position_ms=0.0, duration_ms=100.0)]
        result = _render_phrase_bar(phrases, total_ms=100.0)
        assert PHRASE_COLORS["Chorus"] in result

    def test_unknown_label_falls_back_to_unknown_color(self):
        phrases = [Phrase(beat_start=1, beat_end=2, kind=99, label="Weird", position_ms=0.0, duration_ms=100.0)]
        result = _render_phrase_bar(phrases, total_ms=100.0)
        assert PHRASE_COLORS["Unknown"] in result

    def test_label_is_html_escaped(self):
        phrases = [Phrase(beat_start=1, beat_end=2, kind=1, label="<b>Intro</b>", position_ms=0.0, duration_ms=100.0)]
        result = _render_phrase_bar(phrases, total_ms=100.0)
        assert "<b>Intro</b>" not in result
        assert "&lt;b&gt;Intro&lt;/b&gt;" in result


# ---------------------------------------------------------------------------
# _render_cue_markers
# ---------------------------------------------------------------------------


class TestRenderCueMarkers:
    def _hot(self, kind, position_ms, comment="C", loop_end_ms=None):
        return CuePoint(kind=kind, position_ms=position_ms, loop_end_ms=loop_end_ms, color_table_index=None, color=-1, comment=comment)

    def test_hot_cue_marker_position_color_and_title(self):
        cue = self._hot(kind=1, position_ms=500.0, comment="First Beat")  # pad A
        result = _render_cue_markers([cue], [], total_ms=1000.0)
        assert "left:50.0000%" in result
        assert CUE_COLORS["A"] in result
        assert 'title="First Beat @ 0:00.5"' in result
        assert ">A<" in result

    def test_hot_cue_comment_is_html_escaped_in_title(self):
        cue = self._hot(kind=1, position_ms=0.0, comment="<script>")
        result = _render_cue_markers([cue], [], total_ms=1000.0)
        assert "<script> @" not in result
        assert "&lt;script&gt; @" in result

    def test_unknown_kind_renders_question_mark_pad(self):
        cue = self._hot(kind=99, position_ms=0.0)
        result = _render_cue_markers([cue], [], total_ms=1000.0)
        assert ">?<" in result

    def test_loop_range_rendered_when_is_loop(self):
        cue = self._hot(kind=2, position_ms=1000.0, loop_end_ms=1500.0)  # pad B, is_loop
        result = _render_cue_markers([cue], [], total_ms=10_000.0)
        assert "hot-loop" in result
        assert "width:5.0000%" in result  # (1500-1000)/10000 * 100

    def test_no_loop_range_when_not_a_loop(self):
        cue = self._hot(kind=1, position_ms=1000.0, loop_end_ms=None)
        result = _render_cue_markers([cue], [], total_ms=10_000.0)
        assert "hot-loop" not in result

    def test_css_class_prefix_applied(self):
        cue = self._hot(kind=1, position_ms=0.0)
        result = _render_cue_markers([cue], [], total_ms=1000.0, css_class_prefix="existing")
        assert 'hot-cue-marker existing"' in result

    def test_memory_cue_numbered_by_sorted_position_not_input_order(self):
        early = CuePoint(kind=0, position_ms=100.0, loop_end_ms=None, color_table_index=None, color=-1, comment="Second In List")
        late = CuePoint(kind=0, position_ms=900.0, loop_end_ms=None, color_table_index=None, color=-1, comment="First In List")
        # Passed in reverse-of-position order -- output should still number by position.
        result = _render_cue_markers([], [late, early], total_ms=1000.0)
        early_idx = result.index("Second In List")
        late_idx = result.index("First In List")
        assert early_idx < late_idx  # earlier position rendered first
        assert ">1<" in result and ">2<" in result

    def test_memory_cue_color_resolved_from_matching_slot(self):
        slot = CUE_SYSTEM_BY_PAD["D"]
        cue = CuePoint(kind=0, position_ms=0.0, loop_end_ms=None, color_table_index=None, color=-1, comment=slot.memory_cue_label)
        result = _render_cue_markers([], [cue], total_ms=1000.0)
        assert CUE_COLORS["D"] in result

    def test_memory_cue_unmatched_comment_uses_default_color(self):
        cue = CuePoint(kind=0, position_ms=0.0, loop_end_ms=None, color_table_index=None, color=-1, comment="Nothing matches this")
        result = _render_cue_markers([], [cue], total_ms=1000.0)
        assert "#aaa" in result

    def test_overlapping_hot_cues_get_nudged_labels(self):
        # Two hot cues at (nearly) the same position -- the second's
        # label should be pushed right by NUDGE_PX so they don't overlap.
        cue_a = self._hot(kind=1, position_ms=500.0)
        cue_b = self._hot(kind=2, position_ms=500.5)  # within OVERLAP_THRESHOLD_PCT
        result = _render_cue_markers([cue_a, cue_b], [], total_ms=1000.0)
        assert "left:4px" in result
        assert "left:18px" in result  # 4 + 1*14


# ---------------------------------------------------------------------------
# _render_confidence_bars
# ---------------------------------------------------------------------------


class TestRenderConfidenceBars:
    def test_renders_all_8_pads_regardless_of_input(self):
        result = _render_confidence_bars({"A": 1.0})
        for pad in "ABCDEFGH":
            assert f"[{pad}]" in result

    def test_missing_pad_defaults_to_zero_percent(self):
        result = _render_confidence_bars({})
        assert "width:0%" in result

    def test_percentage_computed_correctly(self):
        result = _render_confidence_bars({"D": 0.85})
        assert "width:85%" in result
        assert "85%</span>" in result


# ---------------------------------------------------------------------------
# _render_waveform
# ---------------------------------------------------------------------------


class TestRenderWaveform:
    def test_empty_waveform_returns_empty_string(self):
        assert _render_waveform(None) == ""
        assert _render_waveform([]) == ""

    def test_renders_one_rect_per_point_with_correct_viewbox(self):
        wf = [WaveformPoint(height=0.5, red=4, green=4, blue=4) for _ in range(10)]
        result = _render_waveform(wf)
        assert 'viewBox="0 0 10 36"' in result
        assert result.count("<rect") == 10

    def test_bar_height_from_point_height_and_min_clamp(self):
        wf = [WaveformPoint(height=0.0, red=0, green=0, blue=0)]  # would be 0, clamped to 1
        result = _render_waveform(wf)
        assert 'height="1"' in result

    def test_bar_color_from_rgb_hex(self):
        pt = WaveformPoint(height=1.0, red=7, green=0, blue=0)
        result = _render_waveform([pt])
        assert pt.rgb_hex in result

    def test_no_vocal_rects_without_vocal_track(self):
        wf = [WaveformPoint(height=0.5, red=4, green=4, blue=4)]
        result = _render_waveform(wf, vocal_track=None, total_ms=1000.0)
        assert "#ff9800" not in result

    def test_no_vocal_rects_when_total_ms_zero(self):
        wf = [WaveformPoint(height=0.5, red=4, green=4, blue=4)]
        result = _render_waveform(wf, vocal_track=[1, 2, 3], total_ms=0)
        assert "#ff9800" not in result

    def test_vocal_run_renders_one_rect_spanning_it(self):
        wf = [WaveformPoint(height=0.5, red=4, green=4, blue=4) for _ in range(5)]
        vocal_track = [0, 2, 3, 0, 0]  # one contiguous "on" run: frames 1-2
        total_ms = 1000.0
        result = _render_waveform(wf, vocal_track=vocal_track, total_ms=total_ms)
        assert result.count("#ff9800") == 1
        expected_x1 = 1 * _FRAME_MS / total_ms * len(wf)
        assert f'x="{expected_x1:.1f}"' in result

    def test_two_separate_vocal_runs_render_two_rects(self):
        wf = [WaveformPoint(height=0.5, red=4, green=4, blue=4) for _ in range(6)]
        vocal_track = [1, 0, 0, 2, 0, 0]  # two isolated "on" frames
        result = _render_waveform(wf, vocal_track=vocal_track, total_ms=1000.0)
        assert result.count("#ff9800") == 2


# ---------------------------------------------------------------------------
# _render_timeline_section
# ---------------------------------------------------------------------------


class TestRenderTimelineSection:
    def test_title_is_escaped(self, phrases):
        result = _render_timeline_section("<b>Cues</b>", [], [], phrases, total_ms=1000.0)
        assert "<h3>&lt;b&gt;Cues&lt;/b&gt;</h3>" in result

    def test_composes_phrase_bar_and_markers(self, phrases):
        hot = [CuePoint(kind=1, position_ms=0.0, loop_end_ms=None, color_table_index=None, color=-1, comment="First Beat")]
        result = _render_timeline_section("Proposed Cues", hot, [], phrases, total_ms=phrases[-1].position_ms + phrases[-1].duration_ms)
        assert "timeline-section" in result
        assert "phrase-segment" in result
        assert "hot-cue-marker" in result

    def test_timeline_container_wrapped_in_scroll_wrapper(self, phrases):
        # Readability pass: .timeline-container's real width now grows
        # for zoom (JS-driven), so it needs a scrollable ancestor rather
        # than clipping/overflowing the page -- confirm the structural
        # nesting is actually there, not just present anywhere in the
        # string.
        result = _render_timeline_section(
            "Proposed Cues", [], [], phrases,
            total_ms=phrases[-1].position_ms + phrases[-1].duration_ms,
        )
        wrapper_idx = result.index('class="timeline-scroll-wrapper"')
        container_idx = result.index('class="timeline-container"')
        assert wrapper_idx < container_idx
        assert result.count("timeline-scroll-wrapper") == 1


# ---------------------------------------------------------------------------
# _render_track_body
# ---------------------------------------------------------------------------


class TestRenderTrackBody:
    def test_header_fields(self, track: Track, proposal: CueProposal):
        result = _render_track_body(track, proposal)
        assert "<h1>Test Track</h1>" in result
        assert "<h2>Test Artist</h2>" in result
        assert "BPM: 128.0" in result
        assert f"Phrases: {len(track.phrases)}" in result

    def test_title_and_artist_are_html_escaped(self, beat_grid: BeatGrid, phrases: list[Phrase]):
        t = Track(
            id=9, title="<b>Evil</b>", artist="A & B", bpm=128.0, duration_ms=1000.0,
            analysis_path="", cues=[], phrases=phrases, beat_grid=beat_grid,
        )
        result = _render_track_body(t, CueStrategy().propose(t))
        assert "<b>Evil</b>" not in result
        assert "&lt;b&gt;Evil&lt;/b&gt;" in result
        assert "A &amp; B" in result

    def test_phrase_legend_deduplicates_preserving_first_seen_order(self, track: Track, proposal: CueProposal):
        result = _render_track_body(track, proposal)
        # "Up" and "Chorus" and "Intro" and "Outro" each appear exactly once
        # in the legend even though real playlists repeat labels often.
        assert result.count('legend-item') == len({p.label for p in track.phrases})

    def test_default_compare_false_shows_only_proposed_section(self, track: Track, proposal: CueProposal):
        result = _render_track_body(track, proposal)
        assert "Proposed Cues" in result
        assert "Existing Cues" not in result

    def test_compare_true_but_no_existing_cues_still_shows_only_proposed(self, track: Track, proposal: CueProposal):
        result = _render_track_body(track, proposal, compare=True)
        assert "Proposed Cues" in result
        assert "Existing Cues" not in result

    def test_compare_true_with_existing_cues_shows_both_sections(self, track_with_existing_cues: Track):
        proposal = CueStrategy().propose(track_with_existing_cues)
        result = _render_track_body(track_with_existing_cues, proposal, compare=True)
        assert "Existing Cues" in result
        assert "Proposed Cues" in result

    def test_existing_hot_and_memory_cues_split_by_kind(self, track_with_existing_cues: Track):
        proposal = CueStrategy().propose(track_with_existing_cues)
        result = _render_track_body(track_with_existing_cues, proposal, compare=True)
        # 2 existing hot cues (kind>0) + 1 existing memory cue (kind==0) from the fixture
        assert 'hot-cue-marker existing' in result
        assert 'mem-cue-marker existing' in result

    def test_confidence_section_always_present(self, track: Track, proposal: CueProposal):
        result = _render_track_body(track, proposal)
        assert "confidence-section" in result
        assert "Confidence" in result

    def test_notes_section_present_when_notes_exist(self, track: Track, proposal: CueProposal):
        assert proposal.notes  # sanity: the real heuristic always leaves notes
        result = _render_track_body(track, proposal)
        assert "notes-section" in result
        assert "Placement Notes" in result

    def test_notes_section_absent_when_no_notes(self, track: Track, proposal: CueProposal):
        empty_notes_proposal = CueProposal(
            track=proposal.track, hot_cues=proposal.hot_cues,
            memory_cues=proposal.memory_cues, confidence=proposal.confidence, notes=[],
        )
        result = _render_track_body(track, empty_notes_proposal)
        assert "notes-section" not in result

    def test_notes_are_html_escaped(self, track: Track, proposal: CueProposal):
        unsafe_notes_proposal = CueProposal(
            track=proposal.track, hot_cues=proposal.hot_cues,
            memory_cues=proposal.memory_cues, confidence=proposal.confidence,
            notes=["<img src=x>"],
        )
        result = _render_track_body(track, unsafe_notes_proposal)
        assert "<img src=x>" not in result
        assert "&lt;img src=x&gt;" in result


# ---------------------------------------------------------------------------
# render_timeline (single-track full page)
# ---------------------------------------------------------------------------


class TestRenderTimeline:
    def test_full_page_structure(self, track: Track, proposal: CueProposal):
        result = render_timeline(track, proposal)
        assert result.startswith("<!DOCTYPE html>")
        assert result.endswith("</html>")
        assert "<title>djcues &mdash; Test Track</title>" in result
        assert "<h1>Test Track</h1>" in result

    def test_title_is_html_escaped(self, beat_grid: BeatGrid, phrases: list[Phrase]):
        t = Track(
            id=9, title="<b>X</b>", artist="A", bpm=128.0, duration_ms=1000.0,
            analysis_path="", cues=[], phrases=phrases, beat_grid=beat_grid,
        )
        result = render_timeline(t, CueStrategy().propose(t))
        assert "<title>djcues &mdash; &lt;b&gt;X&lt;/b&gt;</title>" in result

    def test_compare_flag_passed_through(self, track_with_existing_cues: Track):
        proposal = CueStrategy().propose(track_with_existing_cues)
        result = render_timeline(track_with_existing_cues, proposal, compare=True)
        assert "Existing Cues" in result


# ---------------------------------------------------------------------------
# render_playlist (multi-track full page)
# ---------------------------------------------------------------------------


class TestRenderPlaylist:
    def test_full_page_structure_and_count(self, track: Track, proposal: CueProposal):
        result = render_playlist("My Playlist", [(track, proposal)])
        assert result.startswith("<!DOCTYPE html>")
        assert "<title>djcues &mdash; My Playlist (1 tracks)</title>" in result
        assert "1 tracks" in result

    def test_playlist_name_is_html_escaped(self, track: Track, proposal: CueProposal):
        result = render_playlist("<script>x</script>", [(track, proposal)])
        assert "<script>x</script>" not in result
        assert "&lt;script&gt;x&lt;/script&gt;" in result

    def test_one_track_card_per_pair(self, track: Track, proposal: CueProposal, track_with_existing_cues: Track):
        proposal2 = CueStrategy().propose(track_with_existing_cues)
        result = render_playlist("P", [(track, proposal), (track_with_existing_cues, proposal2)])
        assert result.count('<div class="track-card">') == 2
        assert "Test Track" in result
        assert "Existing Cues Track" in result

    def test_compare_flag_passed_through_to_every_track(self, track_with_existing_cues: Track):
        proposal = CueStrategy().propose(track_with_existing_cues)
        result = render_playlist("P", [(track_with_existing_cues, proposal)], compare=True)
        assert "Existing Cues" in result

    def test_empty_playlist_renders_valid_shell(self):
        result = render_playlist("Empty", [])
        assert "0 tracks" in result
        assert result.count('<div class="track-card">') == 0
