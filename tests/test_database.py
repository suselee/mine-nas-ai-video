from datetime import datetime, timedelta
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
