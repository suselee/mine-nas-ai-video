from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def local_now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def utc_now_iso() -> str:
    """Backward-compatible name; timestamps are local ISO values."""
    return local_now_iso()


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


class Database:
    _BUSY_TIMEOUT_MS = 30_000

    def __init__(self, path: Path):
        self.path = path

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path,
            timeout=self._BUSY_TIMEOUT_MS / 1000,
        )
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={self._BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def migrate(self) -> None:
        with self.connect() as conn:
            # WAL is persistent for the database file. Setting it once during
            # startup avoids lock-taking journal mode checks on every connection.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_name TEXT NOT NULL,
                    stream_role TEXT NOT NULL CHECK (stream_role IN ('low', 'high')),
                    path TEXT NOT NULL UNIQUE,
                    started_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    processed_at TEXT,
                    analysis_attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    deleted_at TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_segments_role_time
                ON segments(stream_role, started_at, ended_at);

                CREATE INDEX IF NOT EXISTS idx_segments_pending
                ON segments(stream_role, processed_at, analysis_attempts, ended_at);

                CREATE TABLE IF NOT EXISTS moments (
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
                    analysis_backend TEXT NOT NULL DEFAULT 'vlm',
                    category TEXT NOT NULL DEFAULT 'semantic',
                    selection_score REAL NOT NULL DEFAULT 0,
                    clip_started_at TEXT,
                    clip_ended_at TEXT,
                    favorited INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(source_low_segment_id) REFERENCES segments(id)
                );

                CREATE INDEX IF NOT EXISTS idx_moments_created_at
                ON moments(created_at DESC);

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(moments)").fetchall()
            }
            additions = (
                ("analysis_backend", "TEXT NOT NULL DEFAULT 'vlm'"),
                ("category", "TEXT NOT NULL DEFAULT 'semantic'"),
                ("selection_score", "REAL NOT NULL DEFAULT 0"),
                ("clip_started_at", "TEXT"),
                ("clip_ended_at", "TEXT"),
            )
            for name, declaration in additions:
                if name not in columns:
                    conn.execute(f"ALTER TABLE moments ADD COLUMN {name} {declaration}")
            conn.execute(
                "UPDATE moments SET selection_score=confidence "
                "WHERE selection_score=0 AND confidence!=0"
            )
            conn.execute(
                "UPDATE moments SET clip_started_at=source_started_at "
                "WHERE clip_started_at IS NULL"
            )
            conn.execute(
                "UPDATE moments SET clip_ended_at=source_ended_at "
                "WHERE clip_ended_at IS NULL"
            )
            self._migrate_created_at_to_local(conn)

    @staticmethod
    def _migrate_created_at_to_local(conn: sqlite3.Connection) -> None:
        """Convert legacy SQLite CURRENT_TIMESTAMP values from UTC to local ISO."""
        for table in ("segments", "moments", "events"):
            rows = conn.execute(
                f"SELECT id, created_at FROM {table} WHERE created_at NOT LIKE '%T%'"
            ).fetchall()
            updates: list[tuple[str, int]] = []
            for row in rows:
                value = str(row["created_at"] or "").strip()
                if not value:
                    continue
                try:
                    parsed = datetime.fromisoformat(value)
                except ValueError:
                    continue
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                updates.append(
                    (parsed.astimezone().isoformat(timespec="seconds"), int(row["id"]))
                )
            if updates:
                conn.executemany(
                    f"UPDATE {table} SET created_at=? WHERE id=?", updates
                )

    def add_event(self, event_type: str, message: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO events(event_type, message, created_at) VALUES (?, ?, ?)",
                (event_type, message[:2000], local_now_iso()),
            )

    def upsert_segment(
        self,
        *,
        camera_name: str,
        stream_role: str,
        path: Path,
        started_at: str,
        ended_at: str,
        duration_seconds: float,
        size_bytes: int,
    ) -> int:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO segments(
                    camera_name, stream_role, path, started_at, ended_at,
                    duration_seconds, size_bytes, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    ended_at=excluded.ended_at,
                    duration_seconds=excluded.duration_seconds,
                    size_bytes=excluded.size_bytes
                """,
                (
                    camera_name,
                    stream_role,
                    str(path),
                    started_at,
                    ended_at,
                    duration_seconds,
                    size_bytes,
                    local_now_iso(),
                ),
            )
            row = conn.execute("SELECT id FROM segments WHERE path = ?", (str(path),)).fetchone()
            return int(row["id"])

    def upsert_segments(self, segments: list[dict[str, Any]]) -> None:
        """Insert/update one scanner pass in a single short transaction."""
        if not segments:
            return
        values = [
            (
                segment["camera_name"],
                segment["stream_role"],
                str(segment["path"]),
                segment["started_at"],
                segment["ended_at"],
                segment["duration_seconds"],
                segment["size_bytes"],
                local_now_iso(),
            )
            for segment in segments
        ]
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO segments(
                    camera_name, stream_role, path, started_at, ended_at,
                    duration_seconds, size_bytes, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    ended_at=excluded.ended_at,
                    duration_seconds=excluded.duration_seconds,
                    size_bytes=excluded.size_bytes
                WHERE segments.ended_at != excluded.ended_at
                   OR segments.duration_seconds != excluded.duration_seconds
                   OR segments.size_bytes != excluded.size_bytes
                """,
                values,
            )

    def get_pending_segments(
        self,
        *,
        stream_role: str,
        ready_before: str,
        max_attempts: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM segments
                WHERE stream_role = ?
                  AND deleted_at IS NULL
                  AND processed_at IS NULL
                  AND analysis_attempts < ?
                  AND ended_at <= ?
                ORDER BY started_at ASC
                LIMIT ?
                """,
                (stream_role, max_attempts, ready_before, limit),
            ).fetchall()
            return [row_to_dict(row) for row in rows]

    def get_pending_low_segments(
        self,
        *,
        ready_before: str,
        max_attempts: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Backward-compatible wrapper for callers that only analyze low stream."""
        return self.get_pending_segments(
            stream_role="low",
            ready_before=ready_before,
            max_attempts=max_attempts,
            limit=limit,
        )

    def mark_segment_processed(self, segment_id: int) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE segments
                SET processed_at = ?, last_error = NULL
                WHERE id = ?
                """,
                (utc_now_iso(), segment_id),
            )

    def record_analysis_error(self, segment_id: int, error: str, *, final: bool) -> None:
        processed_at = utc_now_iso() if final else None
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE segments
                SET analysis_attempts = analysis_attempts + 1,
                    last_error = ?,
                    processed_at = COALESCE(?, processed_at)
                WHERE id = ?
                """,
                (error[:2000], processed_at, segment_id),
            )

    def find_segments_between(
        self,
        *,
        stream_role: str,
        started_before: str,
        ended_after: str,
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM segments
                WHERE stream_role = ?
                  AND deleted_at IS NULL
                  AND started_at < ?
                  AND ended_at > ?
                ORDER BY started_at ASC
                """,
                (stream_role, started_before, ended_after),
            ).fetchall()
            return [row_to_dict(row) for row in rows]

    def create_moment(
        self,
        *,
        camera_name: str,
        title: str,
        summary: str,
        tags: list[str],
        confidence: float,
        source_low_segment_id: int | None,
        source_started_at: str,
        source_ended_at: str,
        clip_path: Path,
        metadata_path: Path,
        analysis_backend: str = "vlm",
        category: str = "semantic",
        selection_score: float | None = None,
        clip_started_at: str | None = None,
        clip_ended_at: str | None = None,
    ) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO moments(
                    camera_name, title, summary, tags_json, confidence,
                    source_low_segment_id, source_started_at, source_ended_at,
                    clip_path, metadata_path, analysis_backend, category,
                    selection_score, clip_started_at, clip_ended_at, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    camera_name,
                    title,
                    summary,
                    json.dumps(tags, ensure_ascii=True),
                    confidence,
                    source_low_segment_id,
                    source_started_at,
                    source_ended_at,
                    str(clip_path),
                    str(metadata_path),
                    analysis_backend,
                    category,
                    confidence if selection_score is None else selection_score,
                    clip_started_at or source_started_at,
                    clip_ended_at or source_ended_at,
                    local_now_iso(),
                ),
            )
            return int(cursor.lastrowid)

    def count_moments_on_day(self, day: str) -> int:
        """Number of moments whose source started on ``day`` (``YYYY-MM-DD``)."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM moments WHERE clip_started_at LIKE ?",
                (f"{day}%",),
            ).fetchone()
            return int(row[0]) if row else 0

    def min_confidence_on_day(self, day: str) -> float:
        """Return the weakest confidence, or 1.0 when the day is empty."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT MIN(selection_score) FROM moments WHERE clip_started_at LIKE ?",
                (f"{day}%",),
            ).fetchone()
            return float(row[0]) if row and row[0] is not None else 1.0

    def weakest_moment_on_day(self, day: str) -> dict[str, Any] | None:
        """The lowest-confidence moment starting on ``day`` (for daily-cap eviction)."""
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT id, title, confidence, selection_score, category,
                       clip_path, metadata_path
                FROM moments
                WHERE clip_started_at LIKE ?
                ORDER BY selection_score ASC
                LIMIT 1
                """,
                (f"{day}%",),
            ).fetchone()
            if not row:
                return None
            return row_to_dict(row)

    def count_moments_between(self, start_iso: str, end_iso: str) -> int:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM moments
                WHERE clip_started_at >= ? AND clip_started_at < ?
                """,
                (start_iso, end_iso),
            ).fetchone()
            return int(row[0]) if row else 0

    def moments_between(self, start_iso: str, end_iso: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, title, confidence, selection_score, category,
                       clip_path, metadata_path
                FROM moments
                WHERE clip_started_at >= ? AND clip_started_at < ?
                ORDER BY selection_score ASC
                """,
                (start_iso, end_iso),
            ).fetchall()
        return [row_to_dict(row) for row in rows]

    def weakest_moment_between(
        self, start_iso: str, end_iso: str
    ) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT id, title, confidence, selection_score, category,
                       clip_path, metadata_path
                FROM moments
                WHERE clip_started_at >= ? AND clip_started_at < ?
                ORDER BY selection_score ASC
                LIMIT 1
                """,
                (start_iso, end_iso),
            ).fetchone()
            if not row:
                return None
            return row_to_dict(row)

    def moments_on_day(self, day: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM moments
                WHERE clip_started_at LIKE ?
                ORDER BY clip_started_at ASC, id ASC
                """,
                (f"{day}%",),
            ).fetchall()
        return [row_to_dict(row) for row in rows]

    def nearest_moment_before(self, clip_started_at: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM moments
                WHERE clip_started_at <= ?
                ORDER BY clip_started_at DESC
                LIMIT 1
                """,
                (clip_started_at,),
            ).fetchone()
        return row_to_dict(row) if row else None

    def delete_moment_by_clip(self, clip_path: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "DELETE FROM moments WHERE clip_path = ?", (str(clip_path),)
            )

    def list_moments(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM moments
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        moments = [row_to_dict(row) for row in rows]
        for moment in moments:
            moment["tags"] = json.loads(moment.pop("tags_json"))
            moment["favorited"] = bool(moment["favorited"])
        return moments

    def get_moment(self, moment_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM moments WHERE id = ?", (moment_id,)).fetchone()
        if row is None:
            return None
        moment = row_to_dict(row)
        moment["tags"] = json.loads(moment.pop("tags_json"))
        moment["favorited"] = bool(moment["favorited"])
        return moment

    def set_favorite(self, moment_id: int, favorited: bool) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE moments SET favorited = ? WHERE id = ?",
                (1 if favorited else 0, moment_id),
            )

    def delete_moment_record(self, moment_id: int) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM moments WHERE id = ?", (moment_id,))

    def count_pending_segments(self, *, stream_role: str = "low") -> int:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM segments
                WHERE stream_role = ?
                  AND deleted_at IS NULL
                  AND processed_at IS NULL
                """,
                (stream_role,),
            ).fetchone()
            return int(row["count"])

    def segment_status_between(
        self, *, stream_role: str, start_iso: str, end_iso: str
    ) -> tuple[int, int, int]:
        """Return pending, final-error, and total counts for a time window."""
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    SUM(processed_at IS NULL) AS pending,
                    SUM(processed_at IS NOT NULL AND last_error IS NOT NULL) AS errors,
                    COUNT(*) AS total
                FROM segments
                WHERE stream_role = ? AND started_at >= ? AND started_at < ?
                """,
                (stream_role, start_iso, end_iso),
            ).fetchone()
        return (
            int(row["pending"] or 0),
            int(row["errors"] or 0),
            int(row["total"] or 0),
        )

    def latest_segment(self, stream_role: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM segments
                WHERE stream_role = ? AND deleted_at IS NULL
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (stream_role,),
            ).fetchone()
        return row_to_dict(row) if row else None

    def recent_segments(
        self, stream_role: str, *, limit: int = 10
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM segments
                WHERE stream_role = ? AND deleted_at IS NULL
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (stream_role, max(1, limit)),
            ).fetchall()
            return [row_to_dict(row) for row in rows]

    def expired_segments(self, cutoff_iso: str, *, limit: int = 500) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM segments
                WHERE deleted_at IS NULL
                  AND ended_at < ?
                ORDER BY ended_at ASC
                LIMIT ?
                """,
                (cutoff_iso, limit),
            ).fetchall()
            return [row_to_dict(row) for row in rows]

    def mark_segment_deleted(self, segment_id: int) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE segments SET deleted_at = ? WHERE id = ?",
                (utc_now_iso(), segment_id),
            )

    def recent_events(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM events
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [row_to_dict(row) for row in rows]
