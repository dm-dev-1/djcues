"""Tests for djcues.tags_cli -- the `djcues tags ...` command group.

Rekordbox is always faked (same approach as test_cli.py/test_tag_analysis.py):
the DB lookups and the three ANLZ-touching db functions are patched, and
the local tag store is redirected into tmp_path so nothing ever touches
~/.djcues or a real database.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from djcues import tag_analysis, tag_store, tags_cli
from djcues.cli import cli
from djcues.models import BeatGrid, Phrase, TagLink, Track, WaveformPoint
from djcues.tag_store import TagStore


def _content(cid, title="Song", key="8A"):
    return SimpleNamespace(
        ID=str(cid), Title=title, Artist=SimpleNamespace(Name="Artist"), Key=SimpleNamespace(ScaleName=key),
        BPM=12800, Length=200, Commnt=None,
    )


def _track(cid, *, energy=0.6, intro_bars=8, duration_ms=200_000.0) -> Track:
    phrases = [
        Phrase(beat_start=1, beat_end=1 + 4 * intro_bars, kind=1, label="Intro", position_ms=0.0, duration_ms=10_000.0),
        Phrase(beat_start=100, beat_end=132, kind=1, label="Outro", position_ms=150_000.0, duration_ms=15_000.0),
    ]
    return Track(
        id=str(cid), title="Song", artist="Artist", bpm=128.0, duration_ms=duration_ms, analysis_path="", cues=[],
        phrases=phrases, beat_grid=BeatGrid(first_beat_ms=0.0, bpm=128.0),
        waveform=[WaveformPoint(height=energy, red=1, green=1, blue=1) for _ in range(20)],
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A fake library of 10 tracks (energies 0.1..1.0) in one playlist,
    with an isolated tag store. Returns a namespace for tweaking."""
    monkeypatch.setattr(tag_store, "default_db_path", lambda: tmp_path / "tags.db")

    state = SimpleNamespace(
        contents=[_content(i, title=f"Song {i}") for i in range(1, 11)],
        energies={str(i): i / 10 for i in range(1, 11)},
        intro_bars={},  # per-id override
        no_analysis=set(),  # ids that load with no waveform/phrases (streaming-linked etc.)
        load_calls=[],
    )

    def load_track(content, db=None):
        state.load_calls.append(content.ID)
        if content.ID in state.no_analysis:
            return Track(
                id=content.ID, title="Song", artist="Artist", bpm=128.0, duration_ms=200_000.0,
                analysis_path="", cues=[], phrases=[], beat_grid=BeatGrid(first_beat_ms=0.0, bpm=128.0),
            )
        return _track(
            content.ID, energy=state.energies[content.ID], intro_bars=state.intro_bars.get(content.ID, 8),
        )

    db = SimpleNamespace(get_content=lambda: state.contents)
    monkeypatch.setattr(tags_cli, "get_db", lambda: db)
    monkeypatch.setattr(tags_cli, "find_playlist", lambda name, _db=None: SimpleNamespace(ID="p1") if name == "Mix" else None)
    monkeypatch.setattr(tag_analysis, "playlist_contents", lambda _db, pid: list(state.contents))
    monkeypatch.setattr(tag_analysis.db_module, "fingerprint_anlz", lambda c, db=None: f"fp-{c.ID}")
    monkeypatch.setattr(tag_analysis.db_module, "load_track", load_track)
    monkeypatch.setattr(tag_analysis.db_module, "extract_raw_beat_grid", lambda c, db=None: None)
    state.tmp_path = tmp_path
    return state


@pytest.fixture
def runner():
    return CliRunner()


def _calibrate(runner):
    result = runner.invoke(cli, ["tags", "calibrate"])
    assert result.exit_code == 0, result.output
    return result


# --- rules ---------------------------------------------------------------


def test_rules_lists_vocabulary_and_reports_uncalibrated(env, runner):
    result = runner.invoke(cli, ["tags", "rules"])
    assert result.exit_code == 0
    for tag in ("Energy 1", "Long Intro", "Vocal Outro", "Instrumental", "Grid Suspect"):
        assert tag in result.output
    assert "Not calibrated yet" in result.output


def test_rules_shows_cutpoints_once_calibrated(env, runner):
    _calibrate(runner)
    result = runner.invoke(cli, ["tags", "rules"])
    assert "Not calibrated" not in result.output
    assert "Energy 1: min -" in result.output and "Energy 5:" in result.output


# --- calibrate -----------------------------------------------------------


def test_calibrate_reports_bands_and_stores_cutpoints(env, runner):
    result = _calibrate(runner)
    assert "10 tracks" in result.output
    assert "Tracks per band:" in result.output
    with TagStore() as store:
        assert len(store.latest_calibration().cutpoints) == 4


def test_second_calibrate_is_served_from_cache(env, runner):
    _calibrate(runner)
    env.load_calls.clear()
    result = _calibrate(runner)
    assert env.load_calls == []
    assert "10 from cache" in result.output


def test_calibrate_summarises_tracks_with_no_analysis(env, runner):
    env.no_analysis = {"1", "2"}
    result = _calibrate(runner)
    assert "2 track(s) have no usable waveform/phrase analysis" in result.output
    with TagStore() as store:
        assert store.latest_calibration().n_tracks == 8  # the other 8 fed the quintiles


def test_quiet_db_warnings_silences_then_restores_the_logger():
    import logging

    logger = logging.getLogger("djcues.db")
    before = logger.level
    with tags_cli._quiet_db_warnings():
        assert logger.level == logging.ERROR
    assert logger.level == before


def test_calibrate_fails_cleanly_on_a_tiny_library(env, runner):
    env.contents = env.contents[:2]
    result = runner.invoke(cli, ["tags", "calibrate"])
    assert result.exit_code == 1
    assert "at least" in result.output


# --- propose -------------------------------------------------------------


def test_propose_needs_a_track_or_all(env, runner):
    result = runner.invoke(cli, ["tags", "propose", "Mix"])
    assert result.exit_code == 1
    assert "provide a track name or use --all" in result.output


def test_propose_rejects_unknown_category(env, runner):
    result = runner.invoke(cli, ["tags", "propose", "Mix", "--all", "--categories", "Nonsense"])
    assert result.exit_code == 1
    assert "unknown category" in result.output


def test_propose_unknown_playlist(env, runner):
    result = runner.invoke(cli, ["tags", "propose", "Nope", "--all"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_propose_all_without_calibration_warns_and_skips_energy(env, runner):
    result = runner.invoke(cli, ["tags", "propose", "Mix", "--all"])
    assert result.exit_code == 0, result.output
    assert "isn't calibrated" in result.output
    assert "Energy 1" not in result.output


def test_propose_all_after_calibration_tags_energy_bands(env, runner):
    _calibrate(runner)
    result = runner.invoke(cli, ["tags", "propose", "Mix", "--all"])
    assert result.exit_code == 0, result.output
    # lowest- and highest-energy tracks land in the end bands
    assert "Energy 1" in result.output and "Energy 5" in result.output
    assert "10 track(s)" in result.output


def test_propose_single_track_shows_evidence(env, runner):
    _calibrate(runner)
    result = runner.invoke(cli, ["tags", "propose", "Mix", "Song 10"])
    assert result.exit_code == 0, result.output
    assert "[Energy] Energy 5" in result.output
    assert "mean energy 1.00" in result.output


def test_propose_ambiguous_track_name_lists_matches(env, runner):
    result = runner.invoke(cli, ["tags", "propose", "Mix", "Song 1"])  # matches "Song 1" and "Song 10"
    assert result.exit_code == 1
    assert "matches 2 tracks" in result.output
    assert "Song 10" in result.output


def test_propose_no_matching_track(env, runner):
    result = runner.invoke(cli, ["tags", "propose", "Mix", "zzz"])
    assert result.exit_code == 1
    assert "no track matching" in result.output


def test_propose_categories_filter(env, runner):
    env.intro_bars["3"] = 16  # set before calibrating -- features are cached after that
    _calibrate(runner)
    result = runner.invoke(cli, ["tags", "propose", "Mix", "--all", "--categories", "Mix In", "--evidence"])
    assert result.exit_code == 0, result.output
    assert "Long Intro" in result.output
    assert "[Energy]" not in result.output


def test_propose_out_writes_a_pending_session_and_nothing_else(env, runner):
    _calibrate(runner)
    out = env.tmp_path / "session.json"
    result = runner.invoke(cli, ["tags", "propose", "Mix", "--all", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "nothing has been written to Rekordbox" in result.output
    session = json.loads(out.read_text(encoding="utf-8"))
    assert session["kind"] == "tags" and session["playlist"] == "Mix"
    assert len(session["tracks"]) == 10
    one = session["tracks"]["10"]
    assert one["status"] == "pending"
    assert one["tags"]["Energy"][0] == {
        "name": "Energy 5", "evidence": one["tags"]["Energy"][0]["evidence"], "status": "pending",
    }


def test_propose_no_cache_reparses(env, runner):
    _calibrate(runner)
    env.load_calls.clear()
    runner.invoke(cli, ["tags", "propose", "Mix", "--all"])
    assert env.load_calls == []
    runner.invoke(cli, ["tags", "propose", "Mix", "--all", "--no-cache"])
    assert len(env.load_calls) == 10


# --- stats ---------------------------------------------------------------


def test_stats_without_any_cache_explains_what_to_do(env, runner):
    result = runner.invoke(cli, ["tags", "stats"])
    assert result.exit_code == 1
    assert "tags calibrate" in result.output


def test_stats_reports_prevalence_and_flags_outliers(env, runner):
    _calibrate(runner)
    result = runner.invoke(cli, ["tags", "stats"])
    assert result.exit_code == 0, result.output
    assert "10 tracks analysed" in result.output
    lines = result.output.splitlines()
    # Every fake track has an 8-bar intro, so no Long Intro anywhere:
    # 0 of 10 evaluated is outside the healthy 5-35% range -> flagged.
    long_intro = next(line for line in lines if "Long Intro" in line)
    assert "0.0%" in long_intro and "outside 5-35%" in long_intro
    # Energy bands (equal by construction) and Check warnings (rare by
    # nature) are never range-checked.
    assert "outside" not in next(line for line in lines if "Energy 3" in line)
    assert "outside" not in next(line for line in lines if "Tempo Varies" in line)
    # The fake tracks carry no vocal data: Vocals is evaluated on 0 tracks,
    # reported as such rather than as "0% Instrumental".
    assert "Vocals  (evaluated on 0 track(s))" in result.output
    assert "outside" not in next(line for line in lines if "Instrumental" in line)


def test_stats_never_parses_analysis(env, runner):
    _calibrate(runner)
    env.load_calls.clear()
    runner.invoke(cli, ["tags", "stats"])
    assert env.load_calls == []


def test_stats_counts_uncached_tracks_without_parsing_them(env, runner):
    _calibrate(runner)
    env.contents.append(_content(99, title="New"))
    env.energies["99"] = 0.5
    env.load_calls.clear()
    result = runner.invoke(cli, ["tags", "stats"])
    assert "1 track(s) have no cached analysis" in result.output
    assert env.load_calls == []


# --- sample --------------------------------------------------------------


def test_sample_rejects_unknown_tag(env, runner):
    result = runner.invoke(cli, ["tags", "sample", "Bogus"])
    assert result.exit_code == 1
    assert "unknown tag" in result.output


def test_sample_spreads_examples_across_the_tags_range(env, runner):
    _calibrate(runner)
    # Quintiles of 0.1..1.0 put 0.1 and 0.2 in Energy 1; sampling 2 should
    # show both ends rather than two near-identical tracks.
    result = runner.invoke(cli, ["tags", "sample", "Energy 1", "-n", "2"])
    assert result.exit_code == 0, result.output
    assert "Song 1 " in result.output + " " and "Song 2" in result.output


def test_sample_with_no_matches(env, runner):
    _calibrate(runner)
    result = runner.invoke(cli, ["tags", "sample", "Vocal-Led"])
    assert result.exit_code == 0
    assert "No tracks" in result.output


def test_spread_indices():
    assert tags_cli._spread_indices(2, 5) == [0, 1]
    assert tags_cli._spread_indices(10, 1) == [5]
    assert tags_cli._spread_indices(10, 3) == [0, 4, 9] or tags_cli._spread_indices(10, 3) == [0, 5, 9]
    assert tags_cli._spread_indices(10, 2) == [0, 9]


# --- apply / status ----------------------------------------------------------


def _write_session(env, runner, *extra):
    out = env.tmp_path / "session.json"
    _calibrate(runner)
    result = runner.invoke(cli, ["tags", "propose", "Mix", "--all", "--out", str(out), *extra])
    assert result.exit_code == 0, result.output
    return out


@pytest.fixture
def fake_rekordbox_tags(monkeypatch):
    """Current links come from a list; writes are captured, not performed."""
    from djcues import db as db_module
    from djcues import writer

    state = SimpleNamespace(links=[], writes=[], column_plan=[])
    monkeypatch.setattr(db_module, "read_tag_links", lambda ids=None, db=None: list(state.links))
    monkeypatch.setattr(writer, "plan_tag_columns", lambda db: list(state.column_plan))
    monkeypatch.setattr(
        writer, "apply_tag_changes",
        lambda changes, db=None: state.writes.append(changes)
        or {"added": sum(len(c["add"]) for c in changes), "removed": 0, "backup": "backup.db"},
    )
    return state


def test_apply_rejects_a_non_tag_session(env, runner):
    bad = env.tmp_path / "cues.json"
    bad.write_text(json.dumps({"tracks": {}}), encoding="utf-8")
    result = runner.invoke(cli, ["tags", "apply", str(bad)])
    assert result.exit_code == 1 and "not a tag session" in result.output


def test_apply_dry_run_shows_the_plan_and_writes_nothing(env, runner, fake_rekordbox_tags):
    session = _write_session(env, runner)
    result = runner.invoke(cli, ["tags", "apply", str(session), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "+Energy 5" in result.output
    assert "Dry run" in result.output
    assert fake_rekordbox_tags.writes == []
    with TagStore() as store:
        assert store.ledger_entries() == []


def test_apply_writes_and_records_the_ledger(env, runner, fake_rekordbox_tags):
    session = _write_session(env, runner)
    result = runner.invoke(cli, ["tags", "apply", str(session), "--yes"])
    assert result.exit_code == 0, result.output
    assert "Done: 10 tag(s) added" in result.output  # one Energy band per track
    assert len(fake_rekordbox_tags.writes) == 1
    with TagStore() as store:
        assert store.last_written("10", "Energy") == {"Energy 5"}


def test_apply_declined_writes_nothing(env, runner, fake_rekordbox_tags):
    session = _write_session(env, runner)
    result = runner.invoke(cli, ["tags", "apply", str(session)], input="n\n")
    assert "Aborted" in result.output
    assert fake_rekordbox_tags.writes == []


def test_apply_leaves_a_user_edit_alone_and_says_so(env, runner, fake_rekordbox_tags):
    session = _write_session(env, runner)
    # Before djcues ever wrote anything, the user tagged track 10 themselves.
    fake_rekordbox_tags.links = [TagLink("L1", "10", "T", "Energy 2", "1")]
    result = runner.invoke(cli, ["tags", "apply", str(session), "--yes"])
    assert "left alone (edited in Rekordbox)" in result.output
    assert all(c["content_id"] != "10" for c in fake_rekordbox_tags.writes[0])


def test_apply_reports_a_running_rekordbox_cleanly(env, runner, monkeypatch, fake_rekordbox_tags):
    from djcues import writer

    def running(changes, db=None):
        raise writer.RekordboxRunningError("Rekordbox is running. Close it first.")

    monkeypatch.setattr(writer, "apply_tag_changes", running)
    session = _write_session(env, runner)
    result = runner.invoke(cli, ["tags", "apply", str(session), "--yes"])
    assert result.exit_code == 1 and "Rekordbox is running" in result.output


def test_status_before_anything_was_written(env, runner):
    result = runner.invoke(cli, ["tags", "status"])
    assert "hasn't written any tags yet" in result.output


def test_status_after_apply_and_a_user_edit(env, runner, fake_rekordbox_tags):
    session = _write_session(env, runner)
    runner.invoke(cli, ["tags", "apply", str(session), "--yes"])
    # Rekordbox now has what djcues wrote, except the user changed track 10.
    with TagStore() as store:
        written = store.ledger_entries()
    fake_rekordbox_tags.links = [
        TagLink(f"L{t}", t, "T", next(iter(tags)), "1") for t, c, tags in written if c == "Energy" and tags
    ]
    fake_rekordbox_tags.links = [l for l in fake_rekordbox_tags.links if l.content_id != "10"] + [
        TagLink("Lx", "10", "T", "Energy 1", "1")
    ]
    result = runner.invoke(cli, ["tags", "status"])
    assert result.exit_code == 0, result.output
    assert "djcues has tagged 10 track(s)" in result.output
    assert "track 10 [Energy]: djcues wrote ['Energy 5'], now ['Energy 1']" in result.output


def test_apply_shows_the_column_setup_before_the_first_write(env, runner, fake_rekordbox_tags):
    session = _write_session(env, runner)
    fake_rekordbox_tags.column_plan = [{
        "column": SimpleNamespace(Name="Genre"), "catalog": [], "rename_to": "Energy",
        "remove": [SimpleNamespace(Name="Acid House"), SimpleNamespace(Name="Techno")],
        "create": ["Energy 1", "Energy 2"], "kept": [],
    }]
    result = runner.invoke(cli, ["tags", "apply", str(session), "--dry-run"])
    assert "My Tag columns will be set up first:" in result.output
    assert "rename 'Genre' -> 'Energy'; add 2 tag(s); remove unused Rekordbox defaults: Acid House, Techno" in result.output


def test_apply_says_nothing_about_columns_once_they_are_set_up(env, runner, fake_rekordbox_tags):
    session = _write_session(env, runner)
    result = runner.invoke(cli, ["tags", "apply", str(session), "--dry-run"])
    assert "columns will be set up" not in result.output
