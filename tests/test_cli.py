"""Tests for djcues.cli -- the Click command surface (propose, compare,
viz, review, apply, beatgrid, auth, history).

The DB layer (find_playlist/load_playlist_tracks) is always mocked --
these tests never touch a real Rekordbox database. Where a command's
own logic (heuristic cue placement, HTML rendering, session-dict
building) is pure and cheap, it's exercised for real against synthetic
Track fixtures rather than mocked, for stronger coverage; only real I/O
(the local HTTP servers, the browser launch, the blocking wait loops)
is mocked.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from djcues.cli import cli
from djcues.models import BeatGrid, CueProposal, Phrase, Track
from djcues.strategy import CueStrategy


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


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
def track_no_phrases(beat_grid: BeatGrid) -> Track:
    return Track(
        id=2, title="No Phrases Track", artist="Test Artist", bpm=128.0,
        duration_ms=240_000.0, analysis_path="", cues=[], phrases=[],
        beat_grid=beat_grid,
    )


@pytest.fixture
def proposal(track: Track) -> CueProposal:
    return CueStrategy().propose(track)


@pytest.fixture(autouse=True)
def no_analysis_cache():
    """Every existing test in this file predates the analysis cache and
    isn't testing it -- without this, a CliRunner invocation of propose/
    compare/review/beatgrid would hit the REAL ~/.djcues/analysis_cache.db
    (by design, analysis_cache has no test-injectable db_path threaded
    through cli.py -- it's meant to be one real, single, user-global
    cache). Neutralizing it here matches test_writer.py's own precedent:
    get_cached/store_result are mocked at the source, the same way that
    file mocks djcues.history.log_session_corrections rather than letting
    apply_session touch real state. Tests that actually exercise caching
    behavior (TestAnalysisCacheIntegration) override this locally with
    their own patch()."""
    with patch("djcues.analysis_cache.get_cached", return_value=None), \
         patch("djcues.analysis_cache.store_result"):
        yield


def _mock_playlist(playlist_id: int = 1, name: str = "Test Playlist") -> MagicMock:
    pl = MagicMock()
    pl.ID = playlist_id
    pl.Name = name
    return pl


# ---------------------------------------------------------------------------
# Helper function unit tests (no CliRunner needed)
# ---------------------------------------------------------------------------


def test_format_time_under_a_minute():
    from djcues.cli import _format_time
    assert _format_time(5_500) == "0:05.5"


def test_format_time_over_a_minute():
    from djcues.cli import _format_time
    assert _format_time(90_250) == "1:30.2"


def test_resolve_agentic_provider_exits_without_key():
    from djcues.cli import _resolve_agentic_provider
    with patch("djcues.auth.load_config", return_value={}), \
         patch("djcues.auth.resolve_api_key", return_value=(None, None)):
        with pytest.raises(SystemExit) as exc_info:
            _resolve_agentic_provider(None, None)
        assert exc_info.value.code == 1


def test_resolve_agentic_provider_returns_resolved_triple():
    from djcues.cli import _resolve_agentic_provider
    with patch("djcues.auth.load_config", return_value={"provider": "anthropic", "model": "saved-model"}), \
         patch("djcues.auth.resolve_api_key", return_value=("sk-real-key", "keyring")):
        provider_name, api_key, model = _resolve_agentic_provider(None, None)
        assert provider_name == "anthropic"
        assert api_key == "sk-real-key"
        assert model == "saved-model"


def test_resolve_agentic_provider_explicit_args_override_config():
    from djcues.cli import _resolve_agentic_provider
    with patch("djcues.auth.load_config", return_value={"provider": "anthropic", "model": "saved-model"}), \
         patch("djcues.auth.resolve_api_key", return_value=("sk-real-key", "env")):
        provider_name, api_key, model = _resolve_agentic_provider("gemini", "explicit-model")
        assert provider_name == "gemini"
        assert model == "explicit-model"


# ---------------------------------------------------------------------------
# propose
# ---------------------------------------------------------------------------


class TestPropose:
    def test_playlist_not_found(self, runner: CliRunner):
        with patch("djcues.cli.find_playlist", return_value=None):
            result = runner.invoke(cli, ["propose", "Nope", "--all"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_empty_playlist(self, runner: CliRunner):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[]):
            result = runner.invoke(cli, ["propose", "Empty", "--all"])
        assert result.exit_code == 1
        assert "no tracks found" in result.output

    def test_no_track_name_and_no_all_flag(self, runner: CliRunner, track: Track):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]):
            result = runner.invoke(cli, ["propose", "Test Playlist"])
        assert result.exit_code == 1
        assert "provide a track name or use --all" in result.output

    def test_track_name_not_matched(self, runner: CliRunner, track: Track):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]):
            result = runner.invoke(cli, ["propose", "Test Playlist", "Nonexistent Song"])
        assert result.exit_code == 1
        assert "no track matching" in result.output

    def test_estimate_only_without_agentic_errors(self, runner: CliRunner, track: Track):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]):
            result = runner.invoke(cli, ["propose", "Test Playlist", "--all", "--estimate-only"])
        assert result.exit_code == 1
        assert "--estimate-only only applies with --agentic" in result.output

    def test_deep_without_refine_drops_errors(self, runner: CliRunner, track: Track):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]):
            result = runner.invoke(cli, ["propose", "Test Playlist", "--all", "--deep"])
        assert result.exit_code == 1
        assert "--deep only applies with --refine-drops" in result.output

    def test_heuristic_happy_path_prints_proposal(self, runner: CliRunner, track: Track):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]):
            result = runner.invoke(cli, ["propose", "Test Playlist", "Test Track"])
        assert result.exit_code == 0
        assert "Test Track" in result.output
        assert "Test Artist" in result.output
        # This fixture's 4 phrases don't give the heuristic enough structure
        # to place all 8 slots (no Down/Bridge for E, no later Chorus for F)
        # -- 6 is the real, correct count for this input, not a stand-in for 8.
        assert "Hot Cues (6)" in result.output
        assert "Memory Cues (6)" in result.output

    def test_all_flag_processes_every_track(self, runner: CliRunner, track: Track, phrases):
        second = Track(
            id=3, title="Second Track", artist="Test Artist", bpm=128.0,
            duration_ms=240_000.0, analysis_path="", cues=[], phrases=phrases,
            beat_grid=track.beat_grid,
        )
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track, second]):
            result = runner.invoke(cli, ["propose", "Test Playlist", "--all"])
        assert result.exit_code == 0
        assert "Test Track" in result.output
        assert "Second Track" in result.output

    def test_all_flag_skips_tracks_with_no_phrase_data_instead_of_crashing(
        self, runner: CliRunner, track: Track, track_no_phrases: Track
    ):
        # Regression test: a track with no phrase data also typically has
        # bpm=0 in practice (Rekordbox never finished analyzing it), and
        # CueStrategy.propose() divides by bpm -- this used to crash
        # --all outright instead of skipping the one bad track, found via
        # a real ZeroDivisionError against a real Rekordbox library.
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track, track_no_phrases]):
            result = runner.invoke(cli, ["propose", "Test Playlist", "--all"])
        assert result.exit_code == 0
        assert "Test Track" in result.output
        assert "Skipping No Phrases Track (no phrase data)" in result.output

    def test_refine_drops_composes_and_prints_summary(self, runner: CliRunner, track: Track):
        from djcues.models import DropRefinement

        fake_refinement = DropRefinement(
            pad="D", outcome="refined", original_ms=1000.0, refined_ms=1200.0,
            offset_ms=200.0, strength=2.0, source="full_mix", note="test",
        )

        def fake_enhance(proposal, t, offset, loop_bars, deep=False):
            return proposal, [fake_refinement]

        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.drop_enhance.enhance_proposal_drops", side_effect=fake_enhance):
            result = runner.invoke(cli, ["propose", "Test Playlist", "Test Track", "--refine-drops"])
        assert result.exit_code == 0
        assert "1 cue(s) checked against real audio" in result.output
        assert "1 refined" in result.output
        assert "1.0s -> 1.2s" in result.output

    def test_refine_drops_without_librosa_errors(self, runner: CliRunner, track: Track):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch.dict("sys.modules", {"librosa": None}):
            result = runner.invoke(cli, ["propose", "Test Playlist", "Test Track", "--refine-drops"])
        assert result.exit_code == 1
        assert "djcues[audio]" in result.output

    def test_agentic_estimate_only_never_calls_proposer(self, runner: CliRunner, track: Track):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.auth.load_config", return_value={"provider": "anthropic", "model": "m"}), \
             patch("djcues.auth.resolve_api_key", return_value=("sk-key", "keyring")), \
             patch("djcues.providers.get_provider", return_value=MagicMock()), \
             patch("djcues.agentic.estimate_track_cost", return_value=(100, 50, 0.001)), \
             patch("djcues.agentic.propose_with_telemetry") as mock_propose:
            result = runner.invoke(
                cli, ["propose", "Test Playlist", "--all", "--agentic", "--estimate-only"]
            )
        assert result.exit_code == 0
        assert "Estimated cost" in result.output
        mock_propose.assert_not_called()

    def test_agentic_happy_path_prints_cost_summary(self, runner: CliRunner, track: Track):
        from djcues.agentic import AgenticTelemetry

        telemetry = AgenticTelemetry(calls_made=4, input_tokens=500, output_tokens=200)

        def fake_propose_with_telemetry(t, provider, api_key, model, **kwargs):
            return CueStrategy().propose(t), telemetry

        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.auth.load_config", return_value={"provider": "anthropic", "model": "claude-x"}), \
             patch("djcues.auth.resolve_api_key", return_value=("sk-key", "keyring")), \
             patch("djcues.providers.get_provider", return_value=MagicMock()), \
             patch("djcues.providers.estimate_cost", return_value=0.0042), \
             patch("djcues.agentic.propose_with_telemetry", side_effect=fake_propose_with_telemetry):
            result = runner.invoke(cli, ["propose", "Test Playlist", "Test Track", "--agentic"])
        assert result.exit_code == 0
        assert "Actual cost: $0.0042" in result.output
        assert "4 calls" in result.output

    def test_agentic_provider_error_aborts_cleanly(self, runner: CliRunner, track: Track):
        def raising_propose(t, provider, api_key, model, **kwargs):
            raise RuntimeError("401 unauthorized")

        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.auth.load_config", return_value={"provider": "anthropic", "model": "claude-x"}), \
             patch("djcues.auth.resolve_api_key", return_value=("sk-bad-key", "keyring")), \
             patch("djcues.providers.get_provider", return_value=MagicMock()), \
             patch("djcues.agentic.propose_with_telemetry", side_effect=raising_propose):
            result = runner.invoke(cli, ["propose", "Test Playlist", "Test Track", "--agentic"])
        assert result.exit_code == 1
        assert "aborted" in result.output
        assert "djcues auth status" in result.output


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


class TestCompare:
    def test_playlist_not_found(self, runner: CliRunner):
        with patch("djcues.cli.find_playlist", return_value=None):
            result = runner.invoke(cli, ["compare", "Nope", "--all"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_no_track_name_and_no_all_flag(self, runner: CliRunner, track: Track):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]):
            result = runner.invoke(cli, ["compare", "Test Playlist"])
        assert result.exit_code == 1
        assert "provide a track name or use --all" in result.output

    def test_single_track_prints_comparison_table(self, runner: CliRunner, track: Track):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]):
            result = runner.invoke(cli, ["compare", "Test Playlist", "Test Track"])
        assert result.exit_code == 0
        assert "Test Track" in result.output
        assert "Existing" in result.output and "Proposed" in result.output

    def test_all_flag_prints_overall_stats(self, runner: CliRunner, track: Track):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]):
            result = runner.invoke(cli, ["compare", "Test Playlist", "--all"])
        assert result.exit_code == 0
        assert "Overall" in result.output

    def test_all_flag_skips_tracks_with_no_phrase_data_instead_of_crashing(
        self, runner: CliRunner, track: Track, track_no_phrases: Track
    ):
        # Same real-library regression as propose's version of this test.
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track, track_no_phrases]):
            result = runner.invoke(cli, ["compare", "Test Playlist", "--all"])
        assert result.exit_code == 0
        assert "Skipping No Phrases Track (no phrase data)" in result.output
        assert "Overall" in result.output


# ---------------------------------------------------------------------------
# viz
# ---------------------------------------------------------------------------


class TestViz:
    def test_playlist_not_found(self, runner: CliRunner):
        with patch("djcues.cli.find_playlist", return_value=None):
            result = runner.invoke(cli, ["viz", "Nope", "--all"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_track_with_no_phrases_errors(self, runner: CliRunner, track_no_phrases: Track):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track_no_phrases]), \
             patch("webbrowser.open"):
            result = runner.invoke(cli, ["viz", "Test Playlist", "No Phrases Track"])
        assert result.exit_code == 1
        assert "no phrase data" in result.output

    def test_single_track_writes_real_html(self, runner: CliRunner, track: Track, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("webbrowser.open") as mock_open:
            result = runner.invoke(cli, ["viz", "Test Playlist", "Test Track"])
        assert result.exit_code == 0
        mock_open.assert_called_once()

        written = list(tmp_path.glob("*.html"))
        assert len(written) == 1
        html_content = written[0].read_text(encoding="utf-8")
        assert "Test Track" in html_content
        assert "<!DOCTYPE html>" in html_content

    def test_all_flag_skips_tracks_without_phrases(
        self, runner: CliRunner, track: Track, track_no_phrases: Track, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track, track_no_phrases]), \
             patch("webbrowser.open"):
            result = runner.invoke(cli, ["viz", "Test Playlist", "--all"])
        assert result.exit_code == 0
        assert "Rendering 1 tracks" in result.output
        assert "Skipping No Phrases Track" in result.output


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------


class TestReview:
    def test_playlist_not_found(self, runner: CliRunner):
        with patch("djcues.cli.find_playlist", return_value=None):
            result = runner.invoke(cli, ["review", "Nope", "--all"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_empty_playlist(self, runner: CliRunner):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[]):
            result = runner.invoke(cli, ["review", "Test Playlist", "--all"])
        assert result.exit_code == 1
        assert "No tracks found" in result.output

    def test_happy_path_starts_server_and_writes_session(
        self, runner: CliRunner, track: Track, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        fake_server = MagicMock()
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.server.start_server", return_value=(fake_server, 54123)) as mock_start, \
             patch("webbrowser.open") as mock_open, \
             patch("time.sleep", side_effect=KeyboardInterrupt):
            result = runner.invoke(cli, ["review", "Test Playlist", "--all"])

        assert result.exit_code == 0
        assert "Server stopped." in result.output
        mock_start.assert_called_once()
        mock_open.assert_called_once_with("http://127.0.0.1:54123")

        sessions = list(tmp_path.glob("*-session.json"))
        htmls = list(tmp_path.glob("*-review.html"))
        assert len(sessions) == 1
        assert len(htmls) == 1
        assert "Test Track" in htmls[0].read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


class TestApply:
    def test_missing_session_file_errors(self, runner: CliRunner):
        result = runner.invoke(cli, ["apply", "does-not-exist.json"])
        assert result.exit_code != 0

    def test_delegates_to_apply_session(self, runner: CliRunner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "session.json").write_text("{}")
        with patch("djcues.writer.apply_session", return_value={}) as mock_apply:
            result = runner.invoke(cli, ["apply", "session.json", "--dry-run"])
        assert result.exit_code == 0
        mock_apply.assert_called_once()
        _, kwargs = mock_apply.call_args
        assert kwargs["dry_run"] is True
        assert kwargs["force"] is False


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------


class TestHistory:
    def test_no_history_yet(self, runner: CliRunner, tmp_path):
        db_path = tmp_path / "history.db"
        with patch("djcues.history.summary", return_value=[]), \
             patch("djcues.history.default_db_path", return_value=db_path):
            result = runner.invoke(cli, ["history"])
        assert result.exit_code == 0
        assert "No correction history yet" in result.output

    def test_prints_per_pad_stats(self, runner: CliRunner, tmp_path):
        db_path = tmp_path / "history.db"
        rows = [
            {"pad": "D", "total": 10, "corrected": 3, "first_seen": "2026-01-01", "last_seen": "2026-02-01"},
        ]
        with patch("djcues.history.summary", return_value=rows), \
             patch("djcues.history.default_db_path", return_value=db_path):
            result = runner.invoke(cli, ["history"])
        assert result.exit_code == 0
        assert "D" in result.output
        assert "10" in result.output and "3" in result.output


# ---------------------------------------------------------------------------
# beatgrid
# ---------------------------------------------------------------------------


class TestBeatgrid:
    def test_playlist_not_found(self, runner: CliRunner):
        with patch("djcues.cli.find_playlist", return_value=None):
            result = runner.invoke(cli, ["beatgrid", "Nope", "--all"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_ok_track_prints_self_consistent(self, runner: CliRunner, track: Track):
        from djcues.models import BeatGridReport, SelfConsistencyResult

        report = BeatGridReport(
            track_id=track.id, title=track.title,
            self_consistency=SelfConsistencyResult(
                is_consistent=True, tempo_varies=False,
                max_pairwise_gap_error_ms=0.0, cumulative_drift_at_end_ms=0.0,
                entry_count=10, notes=[],
            ),
            audio=None, status="ok",
        )
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.db.get_db", return_value=MagicMock()), \
             patch("djcues.db.extract_raw_beat_grid", return_value=[]), \
             patch("djcues.beat_verify.verify_beat_grid", return_value=report):
            result = runner.invoke(cli, ["beatgrid", "Test Playlist", "Test Track"])
        assert result.exit_code == 0
        assert "self-consistent" in result.output

    def test_octave_error_shows_explanation(self, runner: CliRunner, track: Track):
        from djcues.models import AudioBeatVerification, BeatGridReport, SelfConsistencyResult

        report = BeatGridReport(
            track_id=track.id, title=track.title,
            self_consistency=SelfConsistencyResult(
                is_consistent=False, tempo_varies=False,
                max_pairwise_gap_error_ms=500.0, cumulative_drift_at_end_ms=1000.0,
                entry_count=10, notes=["flagged"],
            ),
            audio=AudioBeatVerification(
                matched_beats=100, mean_abs_drift_ms=200.0, max_abs_drift_ms=300.0,
                pct_within_tolerance=10.0, tracker_name="beat_this",
                verdict="octave_error", octave_error="double",
            ),
            status="flagged",
        )
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.db.get_db", return_value=MagicMock()), \
             patch("djcues.db.extract_raw_beat_grid", return_value=[]), \
             patch("djcues.beat_verify.verify_beat_grid", return_value=report):
            result = runner.invoke(cli, ["beatgrid", "Test Playlist", "Test Track"])
        assert result.exit_code == 0
        assert "POSSIBLE OCTAVE ERROR" in result.output
        assert "phantom beats" in result.output

    def test_all_flag_prints_aggregate_summary(self, runner: CliRunner, track: Track, phrases):
        from djcues.models import BeatGridReport, SelfConsistencyResult

        second = Track(
            id=4, title="Second", artist="A", bpm=128.0, duration_ms=1000.0,
            analysis_path="", cues=[], phrases=phrases, beat_grid=track.beat_grid,
        )
        report = BeatGridReport(
            track_id=1, title="x",
            self_consistency=SelfConsistencyResult(
                is_consistent=True, tempo_varies=False,
                max_pairwise_gap_error_ms=0.0, cumulative_drift_at_end_ms=0.0,
                entry_count=1, notes=[],
            ),
            audio=None, status="ok",
        )
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track, second]), \
             patch("djcues.db.get_db", return_value=MagicMock()), \
             patch("djcues.db.extract_raw_beat_grid", return_value=[]), \
             patch("djcues.beat_verify.verify_beat_grid", return_value=report):
            result = runner.invoke(cli, ["beatgrid", "Test Playlist", "--all"])
        assert result.exit_code == 0
        assert "2 tracks checked, 2 OK" in result.output


# ---------------------------------------------------------------------------
# analysis cache (propose/compare/review/beatgrid wiring, cache group)
# ---------------------------------------------------------------------------


class TestAnalysisCacheIntegration:
    """propose/compare/review/beatgrid's own analysis_cache wiring.
    djcues.analysis_cache.get_cached/store_result are mocked at the
    source in each test here (matching test_writer.py's precedent for
    cross-module persistence calls -- that file mocks
    djcues.history.log_session_corrections rather than re-testing it),
    overriding the file's own no_analysis_cache autouse fixture locally
    wherever a specific hit/miss/failure scenario needs asserting on.
    """

    def test_propose_cache_miss_calls_proposer_and_stores_result(
        self, runner: CliRunner, track: Track
    ):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.analysis_cache.get_cached", return_value=None) as mock_get, \
             patch("djcues.analysis_cache.store_result") as mock_store:
            result = runner.invoke(cli, ["propose", "Test Playlist", "Test Track"])

        assert result.exit_code == 0
        mock_get.assert_called_once()
        mock_store.assert_called_once()
        _, kwargs = mock_store.call_args
        assert kwargs["track_title"] == "Test Track"
        assert "Using cached analysis" not in result.output

    def test_propose_cache_hit_skips_proposer_and_prints_indicator(
        self, runner: CliRunner, track: Track
    ):
        from djcues.analysis_cache import CachedResult

        cached = CachedResult(
            result={"positions": {"A": 77.0}, "confidence": {"A": 1.0}, "notes": ["cached"]},
            cost_usd=None, compute_seconds=190.4, source="cli",
            created_at="2026-09-09T12:00:00", updated_at="2026-09-09T12:00:00",
        )
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.analysis_cache.get_cached", return_value=cached), \
             patch("djcues.analysis_cache.store_result") as mock_store, \
             patch("djcues.strategy.CueStrategy.propose") as mock_propose:
            result = runner.invoke(cli, ["propose", "Test Playlist", "Test Track"])

        assert result.exit_code == 0
        mock_propose.assert_not_called()
        mock_store.assert_not_called()
        assert "Using cached analysis for Test Track (analyzed 2026-09-09T12:00:00)" in result.output
        assert "Cache: 1/1 track(s) reused" in result.output

    def test_propose_no_cache_flag_skips_read_but_still_stores(
        self, runner: CliRunner, track: Track
    ):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.analysis_cache.get_cached") as mock_get, \
             patch("djcues.analysis_cache.store_result") as mock_store:
            result = runner.invoke(cli, ["propose", "Test Playlist", "Test Track", "--no-cache"])

        assert result.exit_code == 0
        mock_get.assert_not_called()
        mock_store.assert_called_once()

    def test_compare_all_and_single_track_both_use_cache(
        self, runner: CliRunner, track: Track
    ):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.analysis_cache.get_cached", return_value=None) as mock_get, \
             patch("djcues.analysis_cache.store_result") as mock_store:
            result_all = runner.invoke(cli, ["compare", "Test Playlist", "--all"])
        assert result_all.exit_code == 0
        assert mock_get.call_count == 1
        assert mock_store.call_count == 1

        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.analysis_cache.get_cached", return_value=None) as mock_get2, \
             patch("djcues.analysis_cache.store_result") as mock_store2:
            result_single = runner.invoke(cli, ["compare", "Test Playlist", "Test Track"])
        assert result_single.exit_code == 0
        assert mock_get2.call_count == 1
        assert mock_store2.call_count == 1

    def test_review_session_uses_cached_proposal_transparently(
        self, runner: CliRunner, track: Track, tmp_path, monkeypatch
    ):
        from djcues.analysis_cache import CachedResult

        monkeypatch.chdir(tmp_path)
        cached = CachedResult(
            result={"positions": {"A": 77.0}, "confidence": {"A": 1.0}, "notes": []},
            cost_usd=None, compute_seconds=1.0, source="cli",
            created_at="2026-09-09T12:00:00", updated_at="2026-09-09T12:00:00",
        )
        fake_server = MagicMock()
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.analysis_cache.get_cached", return_value=cached), \
             patch("djcues.analysis_cache.store_result") as mock_store, \
             patch("djcues.strategy.CueStrategy.propose") as mock_propose, \
             patch("djcues.server.start_server", return_value=(fake_server, 54123)), \
             patch("webbrowser.open"), \
             patch("time.sleep", side_effect=KeyboardInterrupt):
            result = runner.invoke(cli, ["review", "Test Playlist", "--all"])

        assert result.exit_code == 0
        mock_propose.assert_not_called()
        mock_store.assert_not_called()
        assert "Using cached analysis" in result.output

    def test_beatgrid_second_invocation_pattern_uses_cache(
        self, runner: CliRunner, track: Track
    ):
        from djcues.analysis_cache import CachedResult

        cached = CachedResult(
            result={
                "status": "ok",
                "self_consistency": {
                    "is_consistent": True, "tempo_varies": False,
                    "max_pairwise_gap_error_ms": 0.0, "cumulative_drift_at_end_ms": 0.0,
                    "entry_count": 10, "notes": [],
                },
                "audio": None,
            },
            cost_usd=None, compute_seconds=0.1, source="cli",
            created_at="2026-09-09T12:00:00", updated_at="2026-09-09T12:00:00",
        )
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.db.get_db", return_value=MagicMock()), \
             patch("djcues.db.extract_raw_beat_grid", return_value=[]), \
             patch("djcues.analysis_cache.get_cached", return_value=cached), \
             patch("djcues.analysis_cache.store_result") as mock_store, \
             patch("djcues.beat_verify.verify_beat_grid") as mock_verify:
            result = runner.invoke(cli, ["beatgrid", "Test Playlist", "Test Track"])

        assert result.exit_code == 0
        mock_verify.assert_not_called()
        mock_store.assert_not_called()
        assert "Using cached beat-grid check" in result.output
        assert "self-consistent" in result.output

    def test_cache_io_failure_degrades_to_recompute_not_crash(
        self, runner: CliRunner, track: Track
    ):
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.analysis_cache.get_cached", side_effect=RuntimeError("disk full")), \
             patch("djcues.analysis_cache.store_result", side_effect=RuntimeError("disk full")):
            result = runner.invoke(cli, ["propose", "Test Playlist", "Test Track"])

        assert result.exit_code == 0
        assert "Warning: analysis cache unavailable" in result.output
        # Real proposal output is still printed despite the cache being down.
        assert "Test Track" in result.output

    def test_beatgrid_cache_io_failure_degrades_to_recompute_not_crash(
        self, runner: CliRunner, track: Track
    ):
        """beatgrid wires the cache inline (no _apply_cache composition,
        unlike propose/compare/review -- see cli.py) so its own
        get_cached/store_result failure-degradation branches need their
        own, separate coverage from test_cache_io_failure_degrades_to_recompute_not_crash."""
        from djcues.models import BeatGridReport, SelfConsistencyResult

        report = BeatGridReport(
            track_id=track.id, title=track.title,
            self_consistency=SelfConsistencyResult(
                is_consistent=True, tempo_varies=False,
                max_pairwise_gap_error_ms=0.0, cumulative_drift_at_end_ms=0.0,
                entry_count=10, notes=[],
            ),
            audio=None, status="ok",
        )
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.db.get_db", return_value=MagicMock()), \
             patch("djcues.db.extract_raw_beat_grid", return_value=[]), \
             patch("djcues.beat_verify.verify_beat_grid", return_value=report), \
             patch("djcues.analysis_cache.get_cached", side_effect=RuntimeError("disk full")), \
             patch("djcues.analysis_cache.store_result", side_effect=RuntimeError("disk full")):
            result = runner.invoke(cli, ["beatgrid", "Test Playlist", "Test Track"])

        assert result.exit_code == 0
        assert "Warning: analysis cache unavailable" in result.output
        assert "self-consistent" in result.output

    def test_beatgrid_store_failure_alone_still_warns(self, runner: CliRunner, track: Track):
        """Same as above but with a clean cache read (a real miss, not a
        failure) followed by a store failure in isolation -- the
        previous test's simultaneous read+write failure only ever
        exercises the *first* warn-once branch (the read one), since
        cache_warned is already True by the time the write is attempted;
        this is what actually reaches the write branch's own
        `if not cache_warned` check with it still False."""
        from djcues.models import BeatGridReport, SelfConsistencyResult

        report = BeatGridReport(
            track_id=track.id, title=track.title,
            self_consistency=SelfConsistencyResult(
                is_consistent=True, tempo_varies=False,
                max_pairwise_gap_error_ms=0.0, cumulative_drift_at_end_ms=0.0,
                entry_count=10, notes=[],
            ),
            audio=None, status="ok",
        )
        with patch("djcues.cli.find_playlist", return_value=_mock_playlist()), \
             patch("djcues.cli.load_playlist_tracks", return_value=[track]), \
             patch("djcues.db.get_db", return_value=MagicMock()), \
             patch("djcues.db.extract_raw_beat_grid", return_value=[]), \
             patch("djcues.beat_verify.verify_beat_grid", return_value=report), \
             patch("djcues.analysis_cache.get_cached", return_value=None), \
             patch("djcues.analysis_cache.store_result", side_effect=RuntimeError("disk full")):
            result = runner.invoke(cli, ["beatgrid", "Test Playlist", "Test Track"])

        assert result.exit_code == 0
        assert "Warning: analysis cache unavailable" in result.output
        assert "self-consistent" in result.output

    def test_get_proposer_returns_resolved_provider_heuristic_is_none(self):
        from djcues.cli import _get_proposer

        proposer, telemetry_list, resolved_model, resolved_provider = _get_proposer(
            agentic=False, provider_name=None, model=None, offset=16, loop_bars=4, skip_critic=False,
        )
        assert telemetry_list is None
        assert resolved_model is None
        assert resolved_provider is None

    def test_get_proposer_agentic_returns_real_resolved_provider(self):
        from djcues.cli import _get_proposer

        with patch("djcues.auth.load_config", return_value={"provider": "gemini", "model": "gemini-x"}), \
             patch("djcues.auth.resolve_api_key", return_value=("sk-key", "keyring")), \
             patch("djcues.providers.get_provider", return_value=MagicMock()):
            proposer, telemetry_list, resolved_model, resolved_provider = _get_proposer(
                agentic=True, provider_name=None, model=None, offset=16, loop_bars=4, skip_critic=False,
            )
        assert telemetry_list == []
        assert resolved_model == "gemini-x"
        assert resolved_provider == "gemini"


class TestCacheCommand:
    def test_status_empty_cache(self, runner: CliRunner, tmp_path):
        db_path = tmp_path / "analysis_cache.db"
        with patch("djcues.analysis_cache.summary", return_value=[]), \
             patch("djcues.analysis_cache.default_db_path", return_value=db_path):
            result = runner.invoke(cli, ["cache", "status"])
        assert result.exit_code == 0
        assert "No cached analysis yet" in result.output

    def test_status_prints_rows(self, runner: CliRunner, tmp_path):
        db_path = tmp_path / "analysis_cache.db"
        rows = [
            {"analysis_kind": "cue_proposal", "engine": "heuristic", "total": 5,
             "total_cost_usd": 0.0, "first_seen": "2026-01-01", "last_seen": "2026-01-02"},
        ]
        with patch("djcues.analysis_cache.summary", return_value=rows), \
             patch("djcues.analysis_cache.default_db_path", return_value=db_path):
            result = runner.invoke(cli, ["cache", "status"])
        assert result.exit_code == 0
        assert "cue_proposal" in result.output
        assert "heuristic" in result.output

    def test_clear_already_empty(self, runner: CliRunner):
        with patch("djcues.analysis_cache.summary", return_value=[]):
            result = runner.invoke(cli, ["cache", "clear", "--force"])
        assert result.exit_code == 0
        assert "already empty" in result.output

    def test_clear_with_force_skips_confirmation(self, runner: CliRunner):
        rows = [{"analysis_kind": "cue_proposal", "engine": "heuristic", "total": 3,
                 "total_cost_usd": 0.0, "first_seen": "x", "last_seen": "y"}]
        with patch("djcues.analysis_cache.summary", return_value=rows), \
             patch("djcues.analysis_cache.clear_all", return_value=3) as mock_clear:
            result = runner.invoke(cli, ["cache", "clear", "--force"])
        assert result.exit_code == 0
        mock_clear.assert_called_once()
        assert "Deleted 3 cached result(s)" in result.output

    def test_clear_without_force_prompts_and_aborts_on_no(self, runner: CliRunner):
        rows = [{"analysis_kind": "cue_proposal", "engine": "heuristic", "total": 3,
                 "total_cost_usd": 0.0, "first_seen": "x", "last_seen": "y"}]
        with patch("djcues.analysis_cache.summary", return_value=rows), \
             patch("djcues.analysis_cache.clear_all") as mock_clear:
            result = runner.invoke(cli, ["cache", "clear"], input="n\n")
        assert result.exit_code == 0
        assert "Aborted" in result.output
        mock_clear.assert_not_called()


# ---------------------------------------------------------------------------
# auth set / status / clear / models / web
# ---------------------------------------------------------------------------


class TestAuthSet:
    def test_happy_path_saves_config_and_key(self, runner: CliRunner):
        from djcues.providers import ModelInfo

        fake_provider = MagicMock()
        fake_provider.list_models.return_value = [
            ModelInfo(id="claude-cheap", display_name="Cheap", provider="anthropic"),
            ModelInfo(id="claude-good", display_name="Good", provider="anthropic", context_window=200_000),
        ]
        saved_config = {}
        with patch("djcues.providers.get_provider", return_value=fake_provider), \
             patch("djcues.providers.DEFAULT_MODEL", {"anthropic": "claude-cheap"}), \
             patch("djcues.providers.RECOMMENDED_FOR_ACCURACY", {"anthropic": "claude-good"}), \
             patch("djcues.auth.set_api_key") as mock_set_key, \
             patch("djcues.auth.load_config", return_value={}), \
             patch("djcues.auth.save_config", side_effect=saved_config.update):
            result = runner.invoke(
                cli, ["auth", "set", "--provider", "anthropic"],
                input="sk-test-key\n1\n",
            )
        assert result.exit_code == 0, result.output
        mock_set_key.assert_called_once_with("anthropic", "sk-test-key")
        assert saved_config["provider"] == "anthropic"
        # The accuracy-recommended model sorts first in the picker (see
        # models_sorted's key in auth_set) -- choice "1" is claude-good,
        # not claude-cheap, despite claude-cheap being listed first above.
        assert saved_config["model"] == "claude-good"
        assert "Saved" in result.output

    def test_no_models_returned_errors(self, runner: CliRunner):
        fake_provider = MagicMock()
        fake_provider.list_models.return_value = []
        with patch("djcues.providers.get_provider", return_value=fake_provider):
            result = runner.invoke(
                cli, ["auth", "set", "--provider", "anthropic"], input="sk-test-key\n"
            )
        assert result.exit_code == 1
        assert "no models returned" in result.output

    def test_list_models_failure_errors(self, runner: CliRunner):
        fake_provider = MagicMock()
        fake_provider.list_models.side_effect = RuntimeError("invalid key")
        with patch("djcues.providers.get_provider", return_value=fake_provider):
            result = runner.invoke(
                cli, ["auth", "set", "--provider", "anthropic"], input="sk-bad-key\n"
            )
        assert result.exit_code == 1
        assert "could not validate key" in result.output


class TestAuthStatus:
    def test_not_configured(self, runner: CliRunner):
        with patch("djcues.auth.load_config", return_value={}):
            result = runner.invoke(cli, ["auth", "status"])
        assert result.exit_code == 0
        assert "No agentic provider configured" in result.output

    def test_configured_with_key(self, runner: CliRunner):
        with patch("djcues.auth.load_config", return_value={"provider": "anthropic", "model": "claude-x"}), \
             patch("djcues.auth.resolve_api_key", return_value=("sk-key", "keyring")):
            result = runner.invoke(cli, ["auth", "status"])
        assert result.exit_code == 0
        assert "anthropic" in result.output
        assert "configured (source: keyring)" in result.output

    def test_configured_without_key(self, runner: CliRunner):
        with patch("djcues.auth.load_config", return_value={"provider": "anthropic", "model": "claude-x"}), \
             patch("djcues.auth.resolve_api_key", return_value=(None, None)):
            result = runner.invoke(cli, ["auth", "status"])
        assert result.exit_code == 0
        assert "not found" in result.output


class TestAuthClear:
    def test_confirmed_removes_key(self, runner: CliRunner):
        with patch("djcues.auth.clear_api_key") as mock_clear:
            result = runner.invoke(
                cli, ["auth", "clear", "--provider", "anthropic"], input="y\n"
            )
        assert result.exit_code == 0
        mock_clear.assert_called_once_with("anthropic")
        assert "Removed" in result.output

    def test_declined_does_not_remove(self, runner: CliRunner):
        with patch("djcues.auth.clear_api_key") as mock_clear:
            result = runner.invoke(
                cli, ["auth", "clear", "--provider", "anthropic"], input="n\n"
            )
        assert result.exit_code == 0
        mock_clear.assert_not_called()


class TestAuthModels:
    def test_no_provider_configured_or_specified_errors(self, runner: CliRunner):
        with patch("djcues.auth.load_config", return_value={}):
            result = runner.invoke(cli, ["auth", "models"])
        assert result.exit_code == 1
        assert "no provider configured" in result.output

    def test_lists_models_for_configured_provider(self, runner: CliRunner):
        from djcues.providers import ModelInfo

        fake_provider = MagicMock()
        fake_provider.list_models.return_value = [
            ModelInfo(id="claude-x", display_name="Claude X", provider="anthropic", context_window=200_000),
        ]
        with patch("djcues.auth.load_config", return_value={"provider": "anthropic"}), \
             patch("djcues.auth.resolve_api_key", return_value=("sk-key", "keyring")), \
             patch("djcues.providers.get_provider", return_value=fake_provider):
            result = runner.invoke(cli, ["auth", "models"])
        assert result.exit_code == 0
        assert "claude-x" in result.output
        assert "Claude X" in result.output


class TestAuthWeb:
    def _fake_server(self, **overrides):
        base = dict(_setup_complete=False, _shutdown_flag=False, _timed_out=False)
        base.update(overrides)
        return types.SimpleNamespace(**base)

    def test_happy_path_prints_saved(self, runner: CliRunner):
        fake_server = self._fake_server(_setup_complete=True)
        with patch("djcues.auth_web.render_auth_setup_html", return_value="<html></html>"), \
             patch("djcues.server.start_auth_server", return_value=(fake_server, 5555)), \
             patch("webbrowser.open") as mock_open, \
             patch("djcues.auth.load_config", return_value={"provider": "anthropic", "model": "claude-x"}):
            result = runner.invoke(cli, ["auth", "web"])
        assert result.exit_code == 0
        mock_open.assert_called_once_with("http://127.0.0.1:5555")
        assert "Saved. Provider: anthropic, model: claude-x." in result.output

    def test_keyboard_interrupt_cancels(self, runner: CliRunner):
        fake_server = self._fake_server()
        with patch("djcues.auth_web.render_auth_setup_html", return_value="<html></html>"), \
             patch("djcues.server.start_auth_server", return_value=(fake_server, 5555)), \
             patch("webbrowser.open"), \
             patch("time.sleep", side_effect=KeyboardInterrupt):
            result = runner.invoke(cli, ["auth", "web"])
        assert result.exit_code == 0
        assert "Cancelled" in result.output
        assert fake_server._shutdown_flag is True

    def test_timeout_errors(self, runner: CliRunner):
        fake_server = self._fake_server(_timed_out=True)
        with patch("djcues.auth_web.render_auth_setup_html", return_value="<html></html>"), \
             patch("djcues.server.start_auth_server", return_value=(fake_server, 5555)), \
             patch("webbrowser.open"):
            result = runner.invoke(cli, ["auth", "web"])
        assert result.exit_code == 1
        assert "Timed out" in result.output
