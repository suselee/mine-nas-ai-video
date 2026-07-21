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

                CREATE TABLE IF NOT EXISTS comparison_cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_name TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL,
                    board_score REAL,
                    yolo_score REAL,
                    board_payload_json TEXT,
                    yolo_payload_json TEXT,
                    board_session_id TEXT,
                    board_event_state TEXT,
                    board_identity TEXT,
                    board_best_ts REAL,
                    board_last_event_at TEXT,
                    match_status TEXT NOT NULL DEFAULT 'pending',
                    moment_id INTEGER,
                    save_status TEXT,
                    source_low_segment_id INTEGER,
                    control_sample INTEGER NOT NULL DEFAULT 0,
                    control_clip_path TEXT,
                    review_label TEXT NOT NULL DEFAULT 'unreviewed',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(moment_id) REFERENCES moments(id) ON DELETE SET NULL,
                    FOREIGN KEY(source_low_segment_id) REFERENCES segments(id)
                );

                CREATE INDEX IF NOT EXISTS idx_comparison_cases_time
                ON comparison_cases(camera_name, started_at, ended_at);

                CREATE TABLE IF NOT EXISTS detector_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_key TEXT NOT NULL UNIQUE,
                    source TEXT NOT NULL,
                    camera_name TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL,
                    comparison_case_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(comparison_case_id) REFERENCES comparison_cases(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_detector_events_case
                ON detector_events(comparison_case_id, source);
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
            comparison_columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(comparison_cases)").fetchall()
            }
            comparison_additions = (
                ("board_session_id", "TEXT"),
                ("board_event_state", "TEXT"),
                ("board_identity", "TEXT"),
                ("board_best_ts", "REAL"),
                ("board_last_event_at", "TEXT"),
                ("save_status", "TEXT"),
            )
            for name, declaration in comparison_additions:
                if name not in comparison_columns:
                    conn.execute(
                        f"ALTER TABLE comparison_cases ADD COLUMN {name} {declaration}"
                    )
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
            conn.execute(
                "UPDATE comparison_cases SET save_status='saved' "
                "WHERE moment_id IS NOT NULL AND save_status IS NULL"
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

    @staticmethod
    def _comparison_status(board_score: float | None, yolo_score: float | None) -> str:
        if board_score is not None and yolo_score is not None:
            return "both"
        if board_score is not None:
            return "board_only"
        if yolo_score is not None:
            return "yolo_only"
        return "pending"

    def record_detector_event(
        self,
        *,
        event_key: str,
        source: str,
        camera_name: str,
        started_at: str,
        ended_at: str,
        confidence: float,
        payload: dict[str, Any],
        merge_gap_seconds: float,
    ) -> tuple[dict[str, Any], bool]:
        """Persist one detector hit and merge it into a nearby comparison case."""
        if source not in {"rv1106_face", "rv1106_edge", "nas_yolo11n"}:
            raise ValueError(f"unsupported detector event source: {source}")
        is_board = source in {"rv1106_face", "rv1106_edge"}
        session_id = str(payload.get("session_id") or "").strip() if is_board else ""
        event_state = str(payload.get("event") or "hit").strip().lower() if is_board else None
        identity = str(payload.get("identity") or "confirmed").strip().lower() if is_board else None
        try:
            best_ts = float(payload.get("best_ts")) if is_board and payload.get("best_ts") is not None else None
        except (TypeError, ValueError):
            best_ts = None
        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(ended_at)
        gap = max(0.0, merge_gap_seconds)
        search_start = (start.timestamp() - gap)
        search_end = (end.timestamp() + gap)
        payload_text = json.dumps(payload, ensure_ascii=False)
        now = local_now_iso()
        with self.connect() as conn:
            existing_event = conn.execute(
                "SELECT comparison_case_id FROM detector_events WHERE event_key=?",
                (event_key,),
            ).fetchone()
            if existing_event:
                row = conn.execute(
                    "SELECT * FROM comparison_cases WHERE id=?",
                    (int(existing_event["comparison_case_id"]),),
                ).fetchone()
                return row_to_dict(row), False

            rows = conn.execute(
                """
                SELECT * FROM comparison_cases
                WHERE camera_name=? AND control_sample=0
                ORDER BY updated_at DESC
                """,
                (camera_name,),
            ).fetchall()
            case_row = None
            if is_board and session_id:
                case_row = next(
                    (row for row in rows if str(row["board_session_id"] or "") == session_id),
                    None,
                )
            for row in rows:
                if case_row is not None:
                    break
                row_start = datetime.fromisoformat(str(row["started_at"])).timestamp()
                row_end = datetime.fromisoformat(str(row["ended_at"])).timestamp()
                if row_end >= search_start and row_start <= search_end:
                    case_row = row
                    break

            if case_row is None:
                board_score = confidence if is_board else None
                yolo_score = confidence if source == "nas_yolo11n" else None
                cursor = conn.execute(
                    """
                    INSERT INTO comparison_cases(
                        camera_name, started_at, ended_at, board_score, yolo_score,
                        board_payload_json, yolo_payload_json, match_status,
                        board_session_id, board_event_state, board_identity,
                        board_best_ts, board_last_event_at,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        camera_name,
                        started_at,
                        ended_at,
                        board_score,
                        yolo_score,
                        payload_text if is_board else None,
                        payload_text if source == "nas_yolo11n" else None,
                        self._comparison_status(board_score, yolo_score),
                        session_id or None,
                        event_state if is_board else None,
                        identity if is_board else None,
                        best_ts,
                        ended_at if is_board else None,
                        now,
                        now,
                    ),
                )
                case_id = int(cursor.lastrowid)
            else:
                case_id = int(case_row["id"])
                merged_start = min(start, datetime.fromisoformat(str(case_row["started_at"])))
                merged_end = max(end, datetime.fromisoformat(str(case_row["ended_at"])))
                board_score = (
                    max(float(case_row["board_score"] or 0), confidence)
                    if is_board
                    else case_row["board_score"]
                )
                yolo_score = (
                    max(float(case_row["yolo_score"] or 0), confidence)
                    if source == "nas_yolo11n"
                    else case_row["yolo_score"]
                )
                conn.execute(
                    """
                    UPDATE comparison_cases
                    SET started_at=?, ended_at=?, board_score=?, yolo_score=?,
                        board_payload_json=COALESCE(?, board_payload_json),
                        yolo_payload_json=COALESCE(?, yolo_payload_json),
                        board_session_id=COALESCE(?, board_session_id),
                        board_event_state=COALESCE(?, board_event_state),
                        board_identity=CASE
                            WHEN ?='confirmed' THEN 'confirmed'
                            ELSE COALESCE(board_identity, ?)
                        END,
                        board_best_ts=COALESCE(?, board_best_ts),
                        board_last_event_at=COALESCE(?, board_last_event_at),
                        match_status=?, updated_at=?
                    WHERE id=?
                    """,
                    (
                        merged_start.isoformat(timespec="seconds"),
                        merged_end.isoformat(timespec="seconds"),
                        board_score,
                        yolo_score,
                        payload_text if is_board else None,
                        payload_text if source == "nas_yolo11n" else None,
                        session_id or None if is_board else None,
                        event_state if is_board else None,
                        identity if is_board else None,
                        identity if is_board else None,
                        best_ts,
                        ended_at if is_board else None,
                        self._comparison_status(board_score, yolo_score),
                        now,
                        case_id,
                    ),
                )

            conn.execute(
                """
                INSERT INTO detector_events(
                    event_key, source, camera_name, started_at, ended_at,
                    confidence, payload_json, comparison_case_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_key,
                    source,
                    camera_name,
                    started_at,
                    ended_at,
                    confidence,
                    payload_text,
                    case_id,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM comparison_cases WHERE id=?", (case_id,)
            ).fetchone()
            return row_to_dict(row), True

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

    def find_segment_at(
        self, *, stream_role: str, timestamp: str
    ) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM segments
                WHERE stream_role=? AND deleted_at IS NULL
                  AND started_at <= ? AND ended_at > ?
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (stream_role, timestamp, timestamp),
            ).fetchone()
        return row_to_dict(row) if row else None

    def get_comparison_case(self, case_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM comparison_cases WHERE id=?", (case_id,)
            ).fetchone()
        return self._decode_comparison_case(row_to_dict(row)) if row else None

    def pending_board_cases(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM comparison_cases
                WHERE control_sample=0 AND board_score IS NOT NULL
                  AND moment_id IS NULL AND save_status IS NULL
                ORDER BY started_at ASC
                LIMIT ?
                """,
                (max(1, limit),),
            ).fetchall()
        return [self._decode_comparison_case(row_to_dict(row)) for row in rows]

    def count_pending_board_cases(self) -> int:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) FROM comparison_cases
                WHERE control_sample=0 AND board_score IS NOT NULL
                  AND moment_id IS NULL AND save_status IS NULL
                """
            ).fetchone()
        return int(row[0]) if row else 0

    def attach_comparison_moment(
        self, case_id: int, moment_id: int, source_low_segment_id: int
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE comparison_cases
                SET moment_id=?, source_low_segment_id=?, save_status='saved', updated_at=?
                WHERE id=?
                """,
                (moment_id, source_low_segment_id, local_now_iso(), case_id),
            )

    def mark_comparison_case_skipped(
        self, case_id: int, source_low_segment_id: int, reason: str
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE comparison_cases
                SET source_low_segment_id=?, save_status=?, updated_at=?
                WHERE id=?
                """,
                (source_low_segment_id, reason[:100], local_now_iso(), case_id),
            )

    def list_comparison_cases(
        self,
        *,
        limit: int = 200,
        since_iso: str | None = None,
        match_status: str | None = None,
        review_label: str | None = None,
        clip_state: str | None = None,
        order: str = "newest",
    ) -> list[dict[str, Any]]:
        if match_status not in {None, "board_only", "yolo_only", "both", "control"}:
            raise ValueError("invalid comparison match_status")
        if review_label not in {
            None,
            "unreviewed",
            "present",
            "false_positive",
            "uncertain",
        }:
            raise ValueError("invalid comparison review_label")
        if clip_state not in {None, "ready", "pending", "skipped"}:
            raise ValueError("invalid comparison clip_state")
        if order not in {"newest", "random"}:
            raise ValueError("invalid comparison order")

        conditions: list[str] = []
        values: list[Any] = []
        if since_iso:
            conditions.append("started_at >= ?")
            values.append(since_iso)
        if match_status:
            conditions.append("match_status = ?")
            values.append(match_status)
        if review_label:
            conditions.append("review_label = ?")
            values.append(review_label)
        if clip_state == "ready":
            conditions.append("(moment_id IS NOT NULL OR control_clip_path IS NOT NULL)")
        elif clip_state == "skipped":
            conditions.append(
                "moment_id IS NULL AND control_clip_path IS NULL AND save_status IS NOT NULL"
            )
        elif clip_state == "pending":
            conditions.append(
                "moment_id IS NULL AND control_clip_path IS NULL AND save_status IS NULL"
            )

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        ordering = "RANDOM()" if order == "random" else "started_at DESC, id DESC"
        values.append(max(1, limit))
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM comparison_cases
                {where}
                ORDER BY {ordering}
                LIMIT ?
                """,
                values,
            ).fetchall()
        return [self._decode_comparison_case(row_to_dict(row)) for row in rows]

    @staticmethod
    def _decode_comparison_case(case: dict[str, Any]) -> dict[str, Any]:
        for key in ("board_payload_json", "yolo_payload_json"):
            raw = case.pop(key, None)
            try:
                case[key.removesuffix("_json")] = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                case[key.removesuffix("_json")] = None
        case["control_sample"] = bool(case.get("control_sample"))
        if case.get("moment_id") or case.get("control_clip_path"):
            case["clip_state"] = "ready"
        elif case.get("save_status"):
            case["clip_state"] = "skipped"
        else:
            case["clip_state"] = "pending"
        return case

    def set_comparison_review(self, case_id: int, label: str) -> bool:
        if label not in {"present", "false_positive", "uncertain", "unreviewed"}:
            raise ValueError("invalid comparison review label")
        with self.connect() as conn:
            cursor = conn.execute(
                "UPDATE comparison_cases SET review_label=?, updated_at=? WHERE id=?",
                (label, local_now_iso(), case_id),
            )
            return cursor.rowcount > 0

    def comparison_metrics(self, *, since_iso: str | None = None) -> dict[str, Any]:
        cases = self.list_comparison_cases(limit=10000, since_iso=since_iso)
        detections = [case for case in cases if not case["control_sample"]]
        controls = [case for case in cases if case["control_sample"]]
        reviewed_positive = [case for case in detections if case["review_label"] == "present"]

        def source_metrics(score_key: str) -> dict[str, Any]:
            source_cases = [case for case in detections if case.get(score_key) is not None]
            reviewed = [
                case
                for case in source_cases
                if case["review_label"] in {"present", "false_positive"}
            ]
            true_hits = sum(case["review_label"] == "present" for case in reviewed)
            relative_hits = sum(case.get(score_key) is not None for case in reviewed_positive)
            return {
                "hits": len(source_cases),
                "reviewed": len(reviewed),
                "confirmed": true_hits,
                "precision": round(true_hits / len(reviewed), 4) if reviewed else None,
                "relative_union_recall": (
                    round(relative_hits / len(reviewed_positive), 4)
                    if reviewed_positive
                    else None
                ),
            }

        reviewed_controls = [
            case
            for case in controls
            if case["review_label"] in {"present", "false_positive"}
        ]
        missed_controls = sum(case["review_label"] == "present" for case in reviewed_controls)
        status_counts = {
            status: sum(case["match_status"] == status for case in detections)
            for status in ("board_only", "yolo_only", "both")
        }
        identity_counts = {
            identity: sum(case.get("board_identity") == identity for case in detections)
            for identity in ("confirmed", "probable")
        }
        return {
            "cases": len(detections),
            "reviewed_cases": sum(
                case["review_label"] in {"present", "false_positive"}
                for case in detections
            ),
            "status_counts": status_counts,
            "identity_counts": identity_counts,
            "board": source_metrics("board_score"),
            "yolo": source_metrics("yolo_score"),
            "controls": {
                "total": len(controls),
                "reviewed": len(reviewed_controls),
                "daughter_present": missed_controls,
                "common_miss_rate": (
                    round(missed_controls / len(reviewed_controls), 4)
                    if reviewed_controls
                    else None
                ),
            },
        }

    def control_candidates_between(
        self, *, start_iso: str, end_iso: str, limit: int
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT s.* FROM segments s
                WHERE s.stream_role='low' AND s.deleted_at IS NULL
                  AND s.started_at >= ? AND s.started_at < ?
                  AND s.processed_at IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM comparison_cases c
                      WHERE c.control_sample=0
                        AND c.started_at < s.ended_at AND c.ended_at > s.started_at
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM comparison_cases c
                      WHERE c.control_sample=1 AND c.source_low_segment_id=s.id
                  )
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (start_iso, end_iso, max(0, limit)),
            ).fetchall()
        return [row_to_dict(row) for row in rows]

    def create_control_case(
        self,
        *,
        segment_id: int,
        camera_name: str,
        started_at: str,
        ended_at: str,
    ) -> int:
        now = local_now_iso()
        with self.connect() as conn:
            existing = conn.execute(
                """
                SELECT id FROM comparison_cases
                WHERE control_sample=1 AND source_low_segment_id=?
                """,
                (segment_id,),
            ).fetchone()
            if existing:
                return int(existing["id"])
            cursor = conn.execute(
                """
                INSERT INTO comparison_cases(
                    camera_name, started_at, ended_at, match_status,
                    source_low_segment_id, control_sample, created_at, updated_at
                ) VALUES (?, ?, ?, 'control', ?, 1, ?, ?)
                """,
                (camera_name, started_at, ended_at, segment_id, now, now),
            )
            return int(cursor.lastrowid)

    def count_control_cases_between(self, start_iso: str, end_iso: str) -> int:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) FROM comparison_cases
                WHERE control_sample=1 AND started_at >= ? AND started_at < ?
                """,
                (start_iso, end_iso),
            ).fetchone()
        return int(row[0]) if row else 0

    def delete_comparison_case(self, case_id: int) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM comparison_cases WHERE id=?", (case_id,))

    def set_control_clip_path(self, case_id: int, path: Path) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE comparison_cases SET control_clip_path=?, updated_at=? WHERE id=?",
                (str(path), local_now_iso(), case_id),
            )

    def expired_control_cases(self, cutoff_iso: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM comparison_cases
                WHERE control_sample=1 AND ended_at < ?
                ORDER BY ended_at ASC
                """,
                (cutoff_iso,),
            ).fetchall()
        return [self._decode_comparison_case(row_to_dict(row)) for row in rows]

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
            row = conn.execute(
                "SELECT id FROM moments WHERE clip_path=?", (str(clip_path),)
            ).fetchone()
            if row:
                conn.execute(
                    "DELETE FROM comparison_cases WHERE moment_id=?",
                    (int(row["id"]),),
                )
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
            conn.execute(
                "DELETE FROM comparison_cases WHERE moment_id=?", (moment_id,)
            )
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
