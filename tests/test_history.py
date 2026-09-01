import sqlite3

from djcues.history import log_session_corrections, summary


def _session_with_track(track_id: int, status: str, cues: dict) -> dict:
    return {
        "tracks": {
            str(track_id): {
                "title": "Test Track",
                "artist": "Test Artist",
                "bpm": 128.0,
                "duration_ms": 200000.0,
                "phrase_count": 10,
                "has_vocal_data": True,
                "has_waveform_data": True,
                "status": status,
                "has_existing_cues": False,
                "cues": cues,
                "memory_cues": {},
            }
        }
    }


def test_fully_accepted_track_logs_one_row_per_pad(tmp_path):
    db_path = tmp_path / "history.db"
    cues = {
        pad: {"position_ms": 1000.0 * i, "loop_end_ms": None,
              "status": "accepted", "confidence": 0.85}
        for i, pad in enumerate("ABCDEFGH")
    }
    session = _session_with_track(101, "accepted", cues)
    written = log_session_corrections(session, "session1.json", db_path=db_path)
    assert written == 8

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT pad, final_status, proposed_position_ms, final_position_ms FROM corrections"
    ).fetchall()
    conn.close()
    assert len(rows) == 8
    for _pad, status, proposed, final in rows:
        assert status == "accepted"
        assert proposed == final  # nothing was adjusted


def test_adjusted_cue_captures_original_ms_as_proposed(tmp_path):
    db_path = tmp_path / "history.db"
    cues = {
        "D": {"position_ms": 5000.0, "original_ms": 4000.0, "loop_end_ms": None,
              "status": "adjusted", "confidence": 0.85},
    }
    session = _session_with_track(102, "adjusted", cues)
    log_session_corrections(session, "session2.json", db_path=db_path)

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT proposed_position_ms, final_position_ms, final_status "
        "FROM corrections WHERE pad='D'"
    ).fetchone()
    conn.close()
    assert row == (4000.0, 5000.0, "adjusted")


def test_per_cue_skip_logged_with_null_final_position(tmp_path):
    db_path = tmp_path / "history.db"
    cues = {
        "F": {"position_ms": 3000.0, "loop_end_ms": None,
              "status": "skipped", "confidence": 0.5},
    }
    # A single skipped cue flips the track to "adjusted", not "skipped".
    session = _session_with_track(103, "adjusted", cues)
    log_session_corrections(session, "session3.json", db_path=db_path)

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT final_position_ms, final_status FROM corrections WHERE pad='F'"
    ).fetchone()
    conn.close()
    assert row == (None, "skipped")


def test_whole_track_skip_logged_for_every_cue(tmp_path):
    db_path = tmp_path / "history.db"
    # server.py cascades "skipped" status to every cue when a whole track is skipped.
    cues = {
        pad: {"position_ms": 1000.0 * i, "loop_end_ms": None,
              "status": "skipped", "confidence": 0.5}
        for i, pad in enumerate("ABCDEFGH")
    }
    session = _session_with_track(104, "skipped", cues)
    written = log_session_corrections(session, "session4.json", db_path=db_path)
    assert written == 8

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT final_status, final_position_ms FROM corrections").fetchall()
    conn.close()
    assert all(status == "skipped" and pos is None for status, pos in rows)


def test_pending_cue_in_adjusted_track_logged_as_implicit_accept(tmp_path):
    """A track flips to "adjusted" the moment any single cue is nudged, but
    its other cues stay cue-level "pending" -- they still ship to Rekordbox
    (writer.py's build_cue_rows only excludes status == "skipped"), so they
    must be logged as accepted, not silently dropped."""
    db_path = tmp_path / "history.db"
    cues = {
        "D": {"position_ms": 5000.0, "original_ms": 4000.0, "loop_end_ms": None,
              "status": "adjusted", "confidence": 0.85},
        "A": {"position_ms": 100.0, "loop_end_ms": None,
              "status": "pending", "confidence": 1.0},
    }
    session = _session_with_track(105, "adjusted", cues)
    log_session_corrections(session, "session5.json", db_path=db_path)

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT final_status, final_position_ms FROM corrections WHERE pad='A'"
    ).fetchone()
    conn.close()
    assert row == ("accepted", 100.0)


def test_pending_track_not_logged(tmp_path):
    """A track never touched at all (still track-level "pending") was never
    applied -- it shouldn't produce any correction rows."""
    db_path = tmp_path / "history.db"
    cues = {"A": {"position_ms": 100.0, "loop_end_ms": None,
                  "status": "pending", "confidence": 1.0}}
    session = _session_with_track(106, "pending", cues)
    written = log_session_corrections(session, "session6.json", db_path=db_path)
    assert written == 0


def test_repeat_apply_same_session_upserts_not_duplicates(tmp_path):
    db_path = tmp_path / "history.db"
    cues = {"A": {"position_ms": 100.0, "loop_end_ms": None,
                  "status": "accepted", "confidence": 1.0}}
    session = _session_with_track(107, "accepted", cues)
    log_session_corrections(session, "session7.json", db_path=db_path)
    log_session_corrections(session, "session7.json", db_path=db_path)

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
    conn.close()
    assert count == 1


def test_summary_groups_by_pad(tmp_path):
    db_path = tmp_path / "history.db"
    cues = {
        "A": {"position_ms": 100.0, "loop_end_ms": None,
              "status": "accepted", "confidence": 1.0},
        "D": {"position_ms": 5000.0, "original_ms": 4000.0, "loop_end_ms": None,
              "status": "adjusted", "confidence": 0.85},
    }
    session = _session_with_track(108, "adjusted", cues)
    log_session_corrections(session, "session8.json", db_path=db_path)

    stats = summary(db_path=db_path)
    by_pad = {row["pad"]: row for row in stats}
    assert by_pad["A"]["total"] == 1
    assert by_pad["A"]["corrected"] == 0
    assert by_pad["D"]["total"] == 1
    assert by_pad["D"]["corrected"] == 1


def test_summary_empty_db_returns_empty_list(tmp_path):
    db_path = tmp_path / "does_not_exist.db"
    assert summary(db_path=db_path) == []
