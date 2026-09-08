"""Tests for djcues.review -- session-dict building and the interactive
review HTML page. Both create_session and render_review_html are pure
functions (no I/O), so these run for real against synthetic Track/
CueProposal fixtures rather than mocking anything.
"""

from __future__ import annotations

import json

import pytest

from djcues.constants import KIND_TO_PAD
from djcues.models import BeatGrid, CueProposal, CuePoint, Phrase, Track, WaveformPoint
from djcues.review import create_session, render_review_html, _render_review_js
from djcues.strategy import CueStrategy


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
def track_with_existing_cues(beat_grid: BeatGrid, phrases: list[Phrase]) -> Track:
    existing = [
        CuePoint(kind=1, position_ms=77.0, loop_end_ms=None, color_table_index=18, color=-1, comment="First Beat"),
        CuePoint(kind=0, position_ms=500.0, loop_end_ms=None, color_table_index=4, color=-1, comment="Memory"),
    ]
    return Track(
        id=2, title="Existing Cues Track", artist="Test Artist", bpm=128.0,
        duration_ms=240_000.0, analysis_path="", cues=existing, phrases=phrases,
        beat_grid=beat_grid,
    )


@pytest.fixture
def proposal(track: Track) -> CueProposal:
    return CueStrategy().propose(track)


@pytest.fixture
def track_with_vocal_and_waveform(beat_grid: BeatGrid, phrases: list[Phrase]) -> Track:
    return Track(
        id=3, title="Rich Track", artist="Test Artist", bpm=128.0,
        duration_ms=240_000.0, analysis_path="", cues=[], phrases=phrases,
        beat_grid=beat_grid,
        waveform=[WaveformPoint(height=0.5, red=4, green=4, blue=4)],
        vocal_track=[0, 1, 2, 3],
    )


# ---------------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------------


class TestCreateSession:
    def test_top_level_fields(self, track: Track, proposal: CueProposal):
        session = create_session("My Playlist", 42, [(track, proposal)])
        assert session["playlist"] == "My Playlist"
        assert session["playlist_id"] == 42
        assert session["settings"] == {"memory_offset_bars": 16, "loop_length_bars": 4}
        # "created" should be a real, parseable ISO timestamp
        from datetime import datetime
        datetime.fromisoformat(session["created"])

    def test_custom_offset_and_loop_bars_recorded(self, track: Track, proposal: CueProposal):
        session = create_session(
            "P", 1, [(track, proposal)], memory_offset_bars=8, loop_length_bars=2
        )
        assert session["settings"] == {"memory_offset_bars": 8, "loop_length_bars": 2}

    def test_track_keyed_by_id_with_expected_fields(self, track: Track, proposal: CueProposal):
        session = create_session("P", 1, [(track, proposal)])
        entry = session["tracks"]["1"]
        assert entry["title"] == "Test Track"
        assert entry["artist"] == "Test Artist"
        assert entry["bpm"] == 128.0
        assert entry["duration_ms"] == 240_000.0
        assert entry["phrase_count"] == len(track.phrases)
        assert entry["first_beat_ms"] == track.beat_grid.first_beat_ms
        assert entry["status"] == "pending"

    def test_has_vocal_and_waveform_data_false_when_absent(self, track: Track, proposal: CueProposal):
        session = create_session("P", 1, [(track, proposal)])
        entry = session["tracks"]["1"]
        assert entry["has_vocal_data"] is False
        assert entry["has_waveform_data"] is False

    def test_has_vocal_and_waveform_data_true_when_present(
        self, track_with_vocal_and_waveform: Track
    ):
        proposal = CueStrategy().propose(track_with_vocal_and_waveform)
        session = create_session("P", 1, [(track_with_vocal_and_waveform, proposal)])
        entry = session["tracks"][str(track_with_vocal_and_waveform.id)]
        assert entry["has_vocal_data"] is True
        assert entry["has_waveform_data"] is True

    def test_has_existing_cues_false_for_fresh_track(self, track: Track, proposal: CueProposal):
        session = create_session("P", 1, [(track, proposal)])
        assert session["tracks"]["1"]["has_existing_cues"] is False

    def test_has_existing_cues_true_when_hot_cue_present(self, track_with_existing_cues: Track):
        proposal = CueStrategy().propose(track_with_existing_cues)
        session = create_session("P", 1, [(track_with_existing_cues, proposal)])
        entry = session["tracks"][str(track_with_existing_cues.id)]
        assert entry["has_existing_cues"] is True

    def test_memory_only_cues_dont_count_as_existing(self, beat_grid: BeatGrid, phrases: list[Phrase]):
        memory_only_track = Track(
            id=5, title="Memory Only", artist="A", bpm=128.0, duration_ms=1000.0,
            analysis_path="", cues=[
                CuePoint(kind=0, position_ms=1.0, loop_end_ms=None, color_table_index=None, color=-1, comment="m"),
            ],
            phrases=phrases, beat_grid=beat_grid,
        )
        proposal = CueStrategy().propose(memory_only_track)
        session = create_session("P", 1, [(memory_only_track, proposal)])
        assert session["tracks"]["5"]["has_existing_cues"] is False

    def test_cues_dict_keyed_by_pad_with_position_and_confidence(
        self, track: Track, proposal: CueProposal
    ):
        session = create_session("P", 1, [(track, proposal)])
        cues = session["tracks"]["1"]["cues"]
        assert set(cues.keys()) == {KIND_TO_PAD[hc.kind] for hc in proposal.hot_cues}
        for hc in proposal.hot_cues:
            pad = KIND_TO_PAD[hc.kind]
            entry = cues[pad]
            assert entry["position_ms"] == hc.position_ms
            assert entry["loop_end_ms"] == hc.loop_end_ms
            assert entry["status"] == "pending"
            assert entry["confidence"] == proposal.confidence.get(pad, 0.0)

    def test_hot_cue_with_unknown_kind_is_skipped(self, track: Track, proposal: CueProposal):
        bogus = CuePoint(kind=99, position_ms=1234.0, loop_end_ms=None, color_table_index=None, color=-1, comment="bogus")
        proposal_with_bogus = CueProposal(
            track=proposal.track,
            hot_cues=[*proposal.hot_cues, bogus],
            memory_cues=proposal.memory_cues,
            confidence=proposal.confidence,
            notes=proposal.notes,
        )
        session = create_session("P", 1, [(track, proposal_with_bogus)])
        cues = session["tracks"]["1"]["cues"]
        assert len(cues) == len(proposal.hot_cues)  # bogus kind=99 excluded, nothing else changed

    def test_memory_cues_keyed_by_1_indexed_position(self, track: Track, proposal: CueProposal):
        session = create_session("P", 1, [(track, proposal)])
        mem = session["tracks"]["1"]["memory_cues"]
        assert set(mem.keys()) == {str(i + 1) for i in range(len(proposal.memory_cues))}
        for i, mc in enumerate(proposal.memory_cues):
            entry = mem[str(i + 1)]
            assert entry["position_ms"] == mc.position_ms
            assert entry["loop_end_ms"] == mc.loop_end_ms
            assert entry["status"] == "pending"

    def test_multiple_tracks_all_present(self, track: Track, proposal: CueProposal, track_with_existing_cues: Track):
        proposal2 = CueStrategy().propose(track_with_existing_cues)
        session = create_session("P", 1, [(track, proposal), (track_with_existing_cues, proposal2)])
        assert set(session["tracks"].keys()) == {"1", "2"}

    def test_result_is_json_serializable(self, track: Track, proposal: CueProposal):
        session = create_session("P", 1, [(track, proposal)])
        # create_session's own docstring promises this -- verify it for real
        # rather than trusting the promise.
        round_tripped = json.loads(json.dumps(session))
        assert round_tripped["playlist"] == "P"


# ---------------------------------------------------------------------------
# render_review_html
# ---------------------------------------------------------------------------


class TestRenderReviewHtml:
    def test_basic_structure(self, track: Track, proposal: CueProposal):
        result = render_review_html("My Playlist", [(track, proposal)], "sess.json", "http://127.0.0.1:8080")
        assert result.startswith("<!DOCTYPE html>")
        assert "My Playlist" in result
        assert "1 tracks" in result
        assert "sess.json" in result
        assert "uv run djcues apply sess.json" in result

    def test_playlist_name_is_html_escaped(self, track: Track, proposal: CueProposal):
        result = render_review_html(
            "<script>alert(1)</script>", [(track, proposal)], "sess.json", "http://127.0.0.1:8080"
        )
        assert "<script>alert(1)</script>" not in result
        assert "&lt;script&gt;" in result

    def test_session_path_is_html_escaped(self, track: Track, proposal: CueProposal):
        result = render_review_html(
            "P", [(track, proposal)], '"><img src=x>', "http://127.0.0.1:8080"
        )
        assert "<img src=x>" not in result
        assert "&lt;img" in result

    def test_server_url_embedded_in_js(self, track: Track, proposal: CueProposal):
        result = render_review_html("P", [(track, proposal)], "sess.json", "http://127.0.0.1:54321")
        assert "const SERVER = 'http://127.0.0.1:54321';" in result

    def test_track_card_data_attributes(self, track: Track, proposal: CueProposal):
        result = render_review_html("P", [(track, proposal)], "sess.json", "http://127.0.0.1:8080")
        assert f'data-track-id="{track.id}"' in result
        assert f'data-bpm="{track.bpm}"' in result
        assert f'data-first-beat-ms="{track.beat_grid.first_beat_ms}"' in result
        assert f'data-duration-ms="{track.duration_ms}"' in result

    def test_data_cues_attribute_has_correct_positions(self, track: Track, proposal: CueProposal):
        import html as html_mod

        result = render_review_html("P", [(track, proposal)], "sess.json", "http://127.0.0.1:8080")
        start = result.index("data-cues='") + len("data-cues='")
        end = result.index("'", start)
        cues_json = html_mod.unescape(result[start:end])
        cues_data = json.loads(cues_json)
        for hc in proposal.hot_cues:
            pad = KIND_TO_PAD[hc.kind]
            assert cues_data[pad]["position_ms"] == hc.position_ms

    def test_no_overwrite_badge_for_fresh_track(self, track: Track, proposal: CueProposal):
        result = render_review_html("P", [(track, proposal)], "sess.json", "http://127.0.0.1:8080")
        assert "OVERWRITE" not in result

    def test_overwrite_badge_for_track_with_existing_cues(self, track_with_existing_cues: Track):
        proposal = CueStrategy().propose(track_with_existing_cues)
        result = render_review_html(
            "P", [(track_with_existing_cues, proposal)], "sess.json", "http://127.0.0.1:8080"
        )
        assert "OVERWRITE" in result

    def test_accept_skip_buttons_reference_track_id(self, track: Track, proposal: CueProposal):
        result = render_review_html("P", [(track, proposal)], "sess.json", "http://127.0.0.1:8080")
        assert f"acceptTrack('{track.id}')" in result
        assert f"skipTrack('{track.id}')" in result

    def test_multiple_tracks_render_multiple_cards(
        self, track: Track, proposal: CueProposal, track_with_existing_cues: Track
    ):
        proposal2 = CueStrategy().propose(track_with_existing_cues)
        result = render_review_html(
            "P", [(track, proposal), (track_with_existing_cues, proposal2)], "sess.json", "http://127.0.0.1:8080"
        )
        assert "2 tracks" in result
        assert result.count('class="track-card review-card"') == 2


# ---------------------------------------------------------------------------
# _render_review_js (URL escaping)
# ---------------------------------------------------------------------------


class TestRenderReviewJs:
    def test_plain_url_embedded_verbatim(self):
        js = _render_review_js("http://127.0.0.1:8080")
        assert "const SERVER = 'http://127.0.0.1:8080';" in js

    def test_single_quote_in_url_is_escaped(self):
        js = _render_review_js("http://127.0.0.1:8080/it's-a-path")
        assert "const SERVER = 'http://127.0.0.1:8080/it\\'s-a-path';" in js
        # And the string literal must actually stay closed correctly --
        # an unescaped quote here would break out of the JS string.
        assert js.count("const SERVER = '") == 1

    def test_backslash_in_url_is_escaped(self):
        js = _render_review_js("http://127.0.0.1:8080\\evil")
        assert "http://127.0.0.1:8080\\\\evil" in js
