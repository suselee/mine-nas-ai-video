from datetime import datetime, timedelta, timezone
from pathlib import Path

from nas_video_summarizer.database import Database


def _insert_segment(db: Database, *, stream_role: str, path: str, started_at: str) -> None:
    ended = (datetime.fromisoformat(started_at) + timedelta(seconds=120)).isoformat(timespec="seconds")
    db.upsert_segment(
        camera_name="test-camera",
        stream_role=stream_role,
        path=Path(path),
        started_at=started_at,
        ended_at=ended,
        duration_seconds=120,
        size_bytes=1024,
    )


def test_get_pending_segments_filters_by_role(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    db.migrate()

    ready_before = "2026-07-08T12:00:00+00:00"

    _insert_segment(db, stream_role="low", path="/buffer/low/seg1.mp4", started_at="2026-07-08T10:00:00+00:00")
    _insert_segment(db, stream_role="high", path="/buffer/high/seg1.mp4", started_at="2026-07-08T10:00:00+00:00")
    _insert_segment(db, stream_role="high", path="/buffer/high/seg2.mp4", started_at="2026-07-08T10:02:00+00:00")

    low_only = db.get_pending_segments(stream_role="low", ready_before=ready_before, max_attempts=3, limit=10)
    high_only = db.get_pending_segments(stream_role="high", ready_before=ready_before, max_attempts=3, limit=10)

    assert len(low_only) == 1
    assert low_only[0]["stream_role"] == "low"
    assert len(high_only) == 2
    assert all(row["stream_role"] == "high" for row in high_only)

    assert db.count_pending_segments(stream_role="low") == 1
    assert db.count_pending_segments(stream_role="high") == 2


def test_get_pending_low_segments_still_works(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    db.migrate()
    ready_before = "2026-07-08T12:00:00+00:00"
    _insert_segment(
        db,
        stream_role="low",
        path="/buffer/low/seg1.mp4",
        started_at="2026-07-08T10:00:00+00:00",
    )

    rows = db.get_pending_low_segments(
        ready_before=ready_before, max_attempts=3, limit=10
    )

    assert len(rows) == 1
    assert rows[0]["stream_role"] == "low"


def test_batch_upsert_inserts_and_updates_segments(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    db.migrate()
    segment = {
        "camera_name": "test-camera",
        "stream_role": "low",
        "path": Path("/buffer/low/seg1.mp4"),
        "started_at": "2026-07-08T10:00:00+00:00",
        "ended_at": "2026-07-08T10:02:00+00:00",
        "duration_seconds": 120,
        "size_bytes": 1024,
    }

    db.upsert_segments([segment])
    db.upsert_segments([{**segment, "size_bytes": 2048}])

    latest = db.latest_segment("low")
    assert latest is not None
    assert latest["size_bytes"] == 2048


def test_connections_configure_busy_timeout(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    db.migrate()

    with db.connect() as conn:
        busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]

    assert busy_timeout == 30_000
    assert journal_mode == "wal"


def test_recent_segments_returns_newest_first(tmp_path):
    db = Database(tmp_path / "test.sqlite3")
    db.migrate()
    _insert_segment(
        db,
        stream_role="low",
        path="/buffer/low/older.mp4",
        started_at="2026-07-08T10:00:00+00:00",
    )
    _insert_segment(
        db,
        stream_role="low",
        path="/buffer/low/newer.mp4",
        started_at="2026-07-08T10:02:00+00:00",
    )

    rows = db.recent_segments("low", limit=1)

    assert len(rows) == 1
    assert rows[0]["path"] == "/buffer/low/newer.mp4"


def test_migrate_adds_detector_columns_to_existing_moments_table(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE moments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_name TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0,
                source_low_segment_id INTEGER,
                source_started_at TEXT NOT NULL,
                source_ended_at TEXT NOT NULL,
                clip_path TEXT NOT NULL UNIQUE,
                metadata_path TEXT NOT NULL,
                favorited INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
    database = Database(path)
    database.migrate()

    with database.connect() as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(moments)")}

    assert {"analysis_backend", "category", "selection_score", "clip_started_at"} <= columns


def test_migrate_converts_legacy_utc_created_at_to_local_iso(tmp_path):
    import sqlite3

    path = tmp_path / "legacy-time.sqlite3"
    database = Database(path)
    database.migrate()
    with database.connect() as conn:
        conn.execute(
            "INSERT INTO events(event_type, message, created_at) VALUES (?, ?, ?)",
            ("legacy", "old", "2026-07-17 01:02:03"),
        )

    database.migrate()

    event = database.recent_events(limit=1)[0]
    expected = datetime(2026, 7, 17, 1, 2, 3, tzinfo=timezone.utc).astimezone()
    assert event["created_at"] == expected.isoformat(timespec="seconds")


def test_new_database_timestamps_are_local_iso(tmp_path):
    database = Database(tmp_path / "local-time.sqlite3")
    database.migrate()
    database.add_event("test", "message")
    _insert_segment(
        database,
        stream_role="low",
        path="/buffer/low/local-time.mp4",
        started_at="2026-07-17T10:00:00+08:00",
    )

    event_time = datetime.fromisoformat(database.recent_events(limit=1)[0]["created_at"])
    segment_time = datetime.fromisoformat(database.latest_segment("low")["created_at"])
    assert event_time.tzinfo is not None
    assert segment_time.tzinfo is not None


def test_detector_events_merge_board_and_yolo_and_deduplicate(tmp_path):
    database = Database(tmp_path / "comparison.sqlite3")
    database.migrate()

    board_case, inserted = database.record_detector_event(
        event_key="board:1",
        source="rv1106_face",
        camera_name="home-camera",
        started_at="2026-07-19T10:00:10+08:00",
        ended_at="2026-07-19T10:00:11+08:00",
        confidence=0.72,
        payload={"seq": 1},
        merge_gap_seconds=15,
    )
    yolo_case, yolo_inserted = database.record_detector_event(
        event_key="yolo:1",
        source="nas_yolo11n",
        camera_name="home-camera",
        started_at="2026-07-19T10:00:20+08:00",
        ended_at="2026-07-19T10:00:25+08:00",
        confidence=0.61,
        payload={"segment_id": 1},
        merge_gap_seconds=15,
    )
    duplicate_case, duplicate_inserted = database.record_detector_event(
        event_key="board:1",
        source="rv1106_face",
        camera_name="home-camera",
        started_at="2026-07-19T10:00:10+08:00",
        ended_at="2026-07-19T10:00:11+08:00",
        confidence=0.72,
        payload={"seq": 1},
        merge_gap_seconds=15,
    )

    assert inserted is True
    assert yolo_inserted is True
    assert duplicate_inserted is False
    assert board_case["id"] == yolo_case["id"] == duplicate_case["id"]
    stored = database.get_comparison_case(int(board_case["id"]))
    assert stored["match_status"] == "both"
    assert stored["board_score"] == 0.72
    assert stored["yolo_score"] == 0.61
    assert stored["board_payload"] == {"seq": 1}


def test_rv1106_session_updates_merge_beyond_time_gap(tmp_path):
    database = Database(tmp_path / "fusion-session.sqlite3")
    database.migrate()

    start, _ = database.record_detector_event(
        event_key="edge:s1:start",
        source="rv1106_edge",
        camera_name="home-camera",
        started_at="2026-07-19T10:00:00+08:00",
        ended_at="2026-07-19T10:00:01+08:00",
        confidence=0.55,
        payload={
            "session_id": "s1",
            "event": "start",
            "identity": "probable",
            "best_ts": 100.0,
        },
        merge_gap_seconds=15,
    )
    update, _ = database.record_detector_event(
        event_key="edge:s1:update",
        source="rv1106_edge",
        camera_name="home-camera",
        started_at="2026-07-19T10:00:00+08:00",
        ended_at="2026-07-19T10:02:00+08:00",
        confidence=0.72,
        payload={
            "session_id": "s1",
            "event": "update",
            "identity": "confirmed",
            "best_ts": 120.0,
        },
        merge_gap_seconds=15,
    )
    end, _ = database.record_detector_event(
        event_key="edge:s1:end",
        source="rv1106_edge",
        camera_name="home-camera",
        started_at="2026-07-19T10:00:00+08:00",
        ended_at="2026-07-19T10:03:00+08:00",
        confidence=0.72,
        payload={
            "session_id": "s1",
            "event": "end",
            "identity": "confirmed",
            "best_ts": 120.0,
        },
        merge_gap_seconds=15,
    )

    assert start["id"] == update["id"] == end["id"]
    stored = database.get_comparison_case(int(start["id"]))
    assert stored["board_session_id"] == "s1"
    assert stored["board_event_state"] == "end"
    assert stored["board_identity"] == "confirmed"
    assert stored["board_score"] == 0.72
    assert stored["board_best_ts"] == 120.0


def test_comparison_review_metrics(tmp_path):
    database = Database(tmp_path / "metrics.sqlite3")
    database.migrate()
    case, _ = database.record_detector_event(
        event_key="board:positive",
        source="rv1106_face",
        camera_name="home-camera",
        started_at="2026-07-19T10:00:00+08:00",
        ended_at="2026-07-19T10:00:01+08:00",
        confidence=0.8,
        payload={},
        merge_gap_seconds=1,
    )
    assert database.set_comparison_review(int(case["id"]), "present") is True

    metrics = database.comparison_metrics()

    assert metrics["board"]["precision"] == 1.0
    assert metrics["board"]["relative_union_recall"] == 1.0
    assert metrics["yolo"]["relative_union_recall"] == 0.0


def test_control_case_is_listed_separately(tmp_path):
    database = Database(tmp_path / "controls.sqlite3")
    database.migrate()
    segment_id = database.upsert_segment(
        camera_name="home-camera",
        stream_role="low",
        path=tmp_path / "low.mp4",
        started_at="2026-07-18T10:00:00+08:00",
        ended_at="2026-07-18T10:02:00+08:00",
        duration_seconds=120,
        size_bytes=100,
    )
    case_id = database.create_control_case(
        segment_id=segment_id,
        camera_name="home-camera",
        started_at="2026-07-18T10:00:30+08:00",
        ended_at="2026-07-18T10:00:50+08:00",
    )

    case = database.get_comparison_case(case_id)
    metrics = database.comparison_metrics()

    assert case["control_sample"] is True
    assert case["match_status"] == "control"
    assert metrics["cases"] == 0
    assert metrics["controls"]["total"] == 1
