from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any

from .config import Settings
from .database import Database, utc_now_iso
from .ffmpeg_tools import (
    PersonFilterDecision,
    SampledFrame,
    _extract_frame,
    build_contact_sheet_from_frames,
    build_recorder_command,
    extract_clip,
    ffmpeg_available,
    ffprobe_available,
    filter_frames_by_person_detection,
    filter_out_blank_frames,
    parse_segment_filename,
    sample_frames_with_offsets,
    segment_time_window,
)
from .llm import AnalysisResult, DaughterVerification, LlamaAnalyzer


class PersonFilterSkip(Exception):
    """Raised when person filter detects no person in any sampled frame."""


def _is_timeout_error(exc: Exception) -> bool:
    return "timed out" in str(exc).lower() or isinstance(exc, asyncio.TimeoutError)

def _now() -> datetime:
    return datetime.now().astimezone()


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except asyncio.TimeoutError:
        if process.returncode is None:
            process.kill()
        await process.wait()


def _in_time_window(now: datetime, start: str, end: str) -> bool:
    """Check whether `now` falls inside the [start, end) time window.

    start/end are "HH:MM" strings. If either is empty, the window is
    disabled and the function returns True (always active). Handles
    windows that cross midnight (e.g. 21:15 -> 06:00).
    """
    if not start or not end:
        return True

    def _to_minutes(value: str) -> int:
        parts = value.split(":")
        return int(parts[0]) * 60 + int(parts[1])

    now_min = now.hour * 60 + now.minute
    start_min = _to_minutes(start)
    end_min = _to_minutes(end)

    if start_min <= end_min:
        return start_min <= now_min < end_min
    # Crosses midnight: e.g. 21:15 -> 06:00.
    return now_min >= start_min or now_min < end_min


def _in_analysis_window(now: datetime, start: str, end: str) -> bool:
    return _in_time_window(now, start, end)


def _in_record_window(now: datetime, start: str, end: str) -> bool:
    return _in_time_window(now, start, end)


def _moment_period(
    started_at: datetime, boundaries_value: str
) -> tuple[str, datetime, datetime] | None:
    parts = [part.strip() for part in boundaries_value.split(",")]
    if len(parts) != 4:
        return None
    try:
        minutes = []
        for part in parts:
            hour_text, minute_text = part.split(":", 1)
            hour, minute = int(hour_text), int(minute_text)
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                return None
            minutes.append(hour * 60 + minute)
    except (TypeError, ValueError):
        return None
    if minutes != sorted(set(minutes)):
        return None

    current = started_at.hour * 60 + started_at.minute
    labels = ("morning", "afternoon", "evening")
    for index, label in enumerate(labels):
        if minutes[index] <= current < minutes[index + 1]:
            start = started_at.replace(
                hour=minutes[index] // 60,
                minute=minutes[index] % 60,
                second=0,
                microsecond=0,
            )
            end = started_at.replace(
                hour=minutes[index + 1] // 60,
                minute=minutes[index + 1] % 60,
                second=0,
                microsecond=0,
            )
            return label, start, end
    return None


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _stream_alignment_snapshot(
    low_segments: list[dict[str, Any]],
    high_segments: list[dict[str, Any]],
    *,
    tolerance_seconds: float,
    required_samples: int,
    segment_seconds: int,
) -> dict[str, Any]:
    required = max(1, required_samples)
    result: dict[str, Any] = {
        "status": "insufficient",
        "offset_seconds": None,
        "tolerance_seconds": tolerance_seconds,
        "paired_segments": 0,
        "required_samples": required,
        "updated_at": utc_now_iso(),
    }
    if low_segments:
        result["latest_low"] = low_segments[0]["path"]
        result["latest_low_started_at"] = low_segments[0]["started_at"]
    if high_segments:
        result["latest_high"] = high_segments[0]["path"]
        result["latest_high_started_at"] = high_segments[0]["started_at"]
    if not low_segments or not high_segments:
        return result

    available_high = list(high_segments)
    offsets: list[float] = []
    max_pair_distance = max(float(segment_seconds) / 2, tolerance_seconds * 4, 5.0)
    for low in low_segments:
        if not available_high or len(offsets) >= required:
            break
        low_started = _parse_iso(low["started_at"])
        nearest = min(
            available_high,
            key=lambda row: abs(
                (_parse_iso(row["started_at"]) - low_started).total_seconds()
            ),
        )
        offset = (_parse_iso(nearest["started_at"]) - low_started).total_seconds()
        if abs(offset) <= max_pair_distance:
            offsets.append(offset)
            available_high.remove(nearest)

    result["paired_segments"] = len(offsets)
    if offsets:
        result["offset_seconds"] = round(float(median(offsets)), 3)
        result["max_abs_offset_seconds"] = round(max(abs(v) for v in offsets), 3)
    if len(offsets) < required:
        return result
    result["status"] = (
        "stable"
        if abs(float(result["offset_seconds"])) <= tolerance_seconds
        else "drifted"
    )
    return result


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug[:64] or "family-moment"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 1000):
        candidate = path.with_name(f"{stem}-{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not allocate unique path for {path}")


def _append_daily_summary(
    *,
    day_dir: Path,
    clip_path: Path,
    result: AnalysisResult,
    clip_start: datetime,
    clip_end: datetime,
) -> Path:
    summary_path = day_dir / "summary.md"
    if not summary_path.exists():
        summary_path.write_text(
            f"# Family Moments - {clip_start.strftime('%Y-%m-%d')}\n\n",
            encoding="utf-8",
        )

    tags = ", ".join(result.tags) if result.tags else "untagged"
    entry = (
        f"## {clip_start.strftime('%H:%M:%S')} - {result.title}\n\n"
        f"- Clip: [{clip_path.name}]({clip_path.name})\n"
        f"- Time: {clip_start.isoformat(timespec='seconds')} to {clip_end.isoformat(timespec='seconds')}\n"
        f"- Confidence: {result.confidence:.2f}\n"
        f"- Tags: {tags}\n\n"
        f"{result.summary}\n\n"
    )
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write(entry)
    return summary_path


@dataclass(frozen=True)
class CapDecision:
    """Outcome of a per-day or per-period moment-cap check.

    ``outcome`` is one of:
      * ``"ok"``      — under the cap; save normally, no eviction.
      * ``"evict"``   — at the cap but the new clip is stronger; ``weakest``
                        should be evicted before saving.
      * ``"blocked"`` — at the cap and the new clip is not stronger; skip it.
    """

    outcome: str
    scope: str
    weakest: dict[str, Any] | None = None


@dataclass(frozen=True)
class MomentSkip:
    """A decision to not save the current candidate, with the event to log."""

    event_type: str
    message: str


@dataclass(frozen=True)
class CapPlan:
    skip: MomentSkip | None = None
    evictions: tuple[tuple[dict[str, Any], str, str], ...] = ()


class Supervisor:
    def __init__(self, settings: Settings, database: Database):
        self.settings = settings
        self.database = database
        self.stop_event = asyncio.Event()
        self.tasks: list[asyncio.Task[None]] = []
        self._last_moment_saved_at: datetime | None = None
        self._llama_timeout_count = 0
        self._llama_circuit_open_until: datetime | None = None
        self.state: dict[str, Any] = {
            "started_at": utc_now_iso(),
            "recorders": {
                "low": {"status": "not-started"},
                "high": {"status": "not-started"},
            },
            "scanner": {"status": "not-started"},
            "prefilter": {"status": "not-started"},
            "stream_alignment": {"status": "unknown"},
            "analyzer": {"status": "not-started"},
            "cleanup": {"status": "not-started"},
        }

    async def start(self) -> None:
        if not self.settings.workers_enabled:
            self.state["workers"] = "disabled"
            return

        self.tasks.append(asyncio.create_task(self._scan_loop(), name="segment-scanner"))
        self.tasks.append(asyncio.create_task(self._cleanup_loop(), name="buffer-cleanup"))
        self.tasks.append(asyncio.create_task(self._analyzer_loop(), name="segment-analyzer"))

        low_rtsp_url = self.settings.rtsp_low_url_for_ffmpeg
        high_rtsp_url = self.settings.rtsp_high_url_for_ffmpeg

        if low_rtsp_url:
            self.tasks.append(
                asyncio.create_task(
                    self._recorder_loop("low", low_rtsp_url),
                    name="recorder-low",
                )
            )
        else:
            self.state["recorders"]["low"] = {"status": "disabled", "reason": "RTSP_LOW_URL is empty"}

        if high_rtsp_url:
            self.tasks.append(
                asyncio.create_task(
                    self._recorder_loop("high", high_rtsp_url),
                    name="recorder-high",
                )
            )
        else:
            self.state["recorders"]["high"] = {"status": "disabled", "reason": "RTSP_HIGH_URL is empty"}

    async def stop(self) -> None:
        self.stop_event.set()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    def snapshot(self) -> dict[str, Any]:
        return self.state

    def _safe_add_event(self, event_type: str, message: str) -> None:
        try:
            self.database.add_event(event_type, message)
        except Exception:
            # Worker error reporting must never terminate the worker itself.
            pass

    async def _recorder_loop(self, role: str, rtsp_url: str) -> None:
        while not self.stop_event.is_set():
            if not ffmpeg_available(self.settings):
                self.state["recorders"][role] = {
                    "status": "error",
                    "message": f"{self.settings.ffmpeg_bin} not found on PATH",
                    "updated_at": utc_now_iso(),
                }
                await asyncio.sleep(30)
                continue
            if not _in_record_window(
                _now(),
                self.settings.record_window_start,
                self.settings.record_window_end,
            ):
                self.state["recorders"][role] = {
                    "status": "waiting_for_record_window",
                    "window_start": self.settings.record_window_start,
                    "window_end": self.settings.record_window_end,
                    "updated_at": utc_now_iso(),
                }
                await asyncio.sleep(60)
                continue

            command = build_recorder_command(self.settings, role, rtsp_url)
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            self.state["recorders"][role] = {
                "status": "running",
                "pid": process.pid,
                "updated_at": utc_now_iso(),
            }
            try:
                window_closed = False
                while process.returncode is None and not self.stop_event.is_set():
                    await asyncio.sleep(1)
                    if not _in_record_window(
                        _now(),
                        self.settings.record_window_start,
                        self.settings.record_window_end,
                    ):
                        window_closed = True
                        await _stop_process(process)
                        break
                if self.stop_event.is_set() and process.returncode is None:
                    await _stop_process(process)
                    return
                if window_closed:
                    self.state["recorders"][role] = {
                        "status": "waiting_for_record_window",
                        "window_start": self.settings.record_window_start,
                        "window_end": self.settings.record_window_end,
                        "updated_at": utc_now_iso(),
                    }
                    self.database.add_event(
                        "recorder-window",
                        f"{role} recorder stopped at window end",
                    )
                    continue
                return_code = await process.wait()
                self.state["recorders"][role] = {
                    "status": "restarting",
                    "return_code": return_code,
                    "updated_at": utc_now_iso(),
                }
                self.database.add_event("recorder", f"{role} recorder exited with code {return_code}")
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                await _stop_process(process)
                raise

    async def _scan_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                count = await asyncio.to_thread(self._scan_once)
                self.state["scanner"] = {
                    "status": "ok",
                    "last_scan_at": utc_now_iso(),
                    "segments_seen": count,
                }
                self._update_stream_alignment()
            except Exception as exc:
                self.state["scanner"] = {
                    "status": "error",
                    "message": str(exc),
                    "updated_at": utc_now_iso(),
                }
                self._safe_add_event("scanner-error", str(exc))
            await asyncio.sleep(10)

    def _update_stream_alignment(self) -> None:
        sample_count = max(1, self.settings.stream_alignment_sample_count)
        fetch_limit = max(sample_count * 2, sample_count)
        alignment = _stream_alignment_snapshot(
            self.database.recent_segments("low", limit=fetch_limit),
            self.database.recent_segments("high", limit=fetch_limit),
            tolerance_seconds=self.settings.stream_alignment_tolerance_seconds,
            required_samples=sample_count,
            segment_seconds=self.settings.segment_seconds,
        )
        previous = self.state.get("stream_alignment", {}).get("status")
        self.state["stream_alignment"] = alignment
        status = alignment["status"]
        if status != previous and status in {"stable", "drifted", "insufficient"}:
            self._safe_add_event(
                f"stream-alignment-{status}",
                json.dumps(alignment, ensure_ascii=False),
            )

    def _scan_once(self) -> int:
        now = _now()
        segments: list[dict[str, Any]] = []
        for directory, role in (
            (self.settings.low_buffer_dir, "low"),
            (self.settings.high_buffer_dir, "high"),
        ):
            directory.mkdir(parents=True, exist_ok=True)
            for path in sorted(directory.glob("*.mp4")):
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue
                file_age = now.timestamp() - stat.st_mtime
                if file_age < self.settings.segment_stable_seconds:
                    continue

                parsed = parse_segment_filename(path)
                if parsed is None:
                    started_at = datetime.fromtimestamp(stat.st_mtime).astimezone()
                    camera_name = self.settings.camera_name
                else:
                    camera_name, parsed_role, started_at = parsed
                    if parsed_role != role:
                        continue

                started, ended = segment_time_window(started_at, self.settings.segment_seconds)
                segments.append(
                    {
                        "camera_name": camera_name,
                        "stream_role": role,
                        "path": path,
                        "started_at": started.isoformat(timespec="seconds"),
                        "ended_at": ended.isoformat(timespec="seconds"),
                        "duration_seconds": self.settings.segment_seconds,
                        "size_bytes": stat.st_size,
                    }
                )
        self.database.upsert_segments(segments)
        return len(segments)

    def _moment_cooldown_active(self) -> bool:
        if self.settings.moment_cooldown_seconds <= 0:
            return False
        if self._last_moment_saved_at is None:
            return False
        return (_now() - self._last_moment_saved_at).total_seconds() < self.settings.moment_cooldown_seconds

    def _display_started_at(self, segment: dict[str, Any]) -> datetime | None:
        value = str(segment.get("started_at", ""))
        if not value:
            return None
        return _parse_iso(value) + timedelta(
            seconds=self.settings.camera_time_offset_seconds
        )

    def _daily_cap_decision(
        self, segment: dict[str, Any], result: AnalysisResult
    ) -> CapDecision:
        """Decide the daily keep-best-N cap for this candidate (see CapDecision)."""
        cap = self.settings.max_moments_per_day
        if cap <= 0:
            return CapDecision("ok", "daily")
        started_at = self._display_started_at(segment)
        if started_at is None:
            return CapDecision("ok", "daily")
        day = started_at.strftime("%Y-%m-%d")
        if self.database.count_moments_on_day(day) < cap:
            return CapDecision("ok", "daily")
        weakest = self.database.weakest_moment_on_day(day)
        if weakest is None:
            return CapDecision("ok", "daily")
        if result.confidence > float(weakest["confidence"]):
            return CapDecision("evict", "daily", weakest)
        return CapDecision("blocked", "daily")

    def _period_cap_decision(
        self, segment: dict[str, Any], result: AnalysisResult
    ) -> CapDecision:
        """Decide the per-period keep-best-N cap for this candidate.

        ``scope`` carries the period label ("morning"/"afternoon"/"evening"),
        or "" when the period cap does not apply (disabled, no timestamp, or
        the timestamp falls outside every configured period).
        """
        cap = self.settings.max_moments_per_period
        if cap <= 0:
            return CapDecision("ok", "")
        started_at = self._display_started_at(segment)
        if started_at is None:
            return CapDecision("ok", "")
        period = _moment_period(started_at, self.settings.moment_period_boundaries)
        if period is None:
            return CapDecision("ok", "")
        label, start, end = period
        start_iso = start.isoformat(timespec="seconds")
        end_iso = end.isoformat(timespec="seconds")
        if self.database.count_moments_between(start_iso, end_iso) < cap:
            return CapDecision("ok", label)
        weakest = self.database.weakest_moment_between(start_iso, end_iso)
        if weakest is None:
            return CapDecision("ok", label)
        if result.confidence > float(weakest["confidence"]):
            return CapDecision("evict", label, weakest)
        return CapDecision("blocked", label)

    def _evict_moment(
        self,
        moment: dict[str, Any],
        *,
        event_type: str = "daily-cap-evict",
        scope: str = "daily",
    ) -> None:
        for key in ("clip_path", "metadata_path"):
            path = Path(str(moment[key]))
            path.unlink(missing_ok=True)
        self.database.delete_moment_by_clip(str(moment["clip_path"]))
        self.database.add_event(
            event_type,
            f"replaced weakest {scope} moment '{moment['title']}' "
            f"(confidence {float(moment['confidence']):.2f})",
        )

    def _apply_moment_caps(
        self, segment: dict[str, Any], result: AnalysisResult
    ) -> CapPlan:
        """Apply cooldown + period + daily caps for a keep-worthy candidate.

        Return a plan without deleting anything. Evictions are applied only
        after the new clip has passed final verification and been registered.
        This prevents a false positive or ffmpeg failure from deleting an old
        good moment.
        """
        if self._moment_cooldown_active():
            return CapPlan(
                skip=MomentSkip(
                    "moment-cooldown", f"skipped '{result.title}' due to cooldown"
                )
            )

        evictions: list[tuple[dict[str, Any], str, str]] = []
        period = self._period_cap_decision(segment, result)
        if period.outcome == "blocked":
            return CapPlan(
                skip=MomentSkip(
                    "period-cap",
                    f"skipped '{result.title}' ({period.scope} limit "
                    f"{self.settings.max_moments_per_period} reached, "
                    f"confidence {result.confidence:.2f} not above weakest)",
                )
            )
        if period.outcome == "evict":
            assert period.weakest is not None
            evictions.append((period.weakest, "period-cap-evict", period.scope))

        display_started = self._display_started_at(segment)
        day = display_started.strftime("%Y-%m-%d") if display_started else ""
        daily_count = self.database.count_moments_on_day(day) if day else 0
        # A period eviction belongs to this same source day, so account for it
        # before applying the daily cap without mutating the database yet.
        daily_count -= len(evictions)
        daily = self._daily_cap_decision_after_count(
            segment, result, daily_count
        )
        if daily.outcome == "blocked":
            return CapPlan(
                skip=MomentSkip(
                    "daily-cap",
                    f"skipped '{result.title}' (daily limit "
                    f"{self.settings.max_moments_per_day} reached, "
                    f"confidence {result.confidence:.2f} not above weakest)",
                )
            )
        if daily.outcome == "evict":
            assert daily.weakest is not None
            if not any(item[0]["id"] == daily.weakest["id"] for item in evictions):
                evictions.append((daily.weakest, "daily-cap-evict", "daily"))
        return CapPlan(evictions=tuple(evictions))

    def _daily_cap_decision_after_count(
        self,
        segment: dict[str, Any],
        result: AnalysisResult,
        count: int,
    ) -> CapDecision:
        cap = self.settings.max_moments_per_day
        if cap <= 0 or count < cap:
            return CapDecision("ok", "daily")
        started_at = self._display_started_at(segment)
        if started_at is None:
            return CapDecision("ok", "daily")
        day = started_at.strftime("%Y-%m-%d")
        weakest = self.database.weakest_moment_on_day(day)
        if weakest is None:
            return CapDecision("ok", "daily")
        if result.confidence > float(weakest["confidence"]):
            return CapDecision("evict", "daily", weakest)
        return CapDecision("blocked", "daily")

    async def _analyzer_loop(self) -> None:
        analyzer = LlamaAnalyzer(self.settings)
        while not self.stop_event.is_set():
            if not self.settings.analysis_enabled:
                self.state["analyzer"] = {"status": "disabled", "reason": "ANALYSIS_ENABLED=false"}
                await asyncio.sleep(self.settings.analysis_interval_seconds)
                continue
            if not _in_analysis_window(
                _now(),
                self.settings.analysis_window_start,
                self.settings.analysis_window_end,
            ):
                self.state["analyzer"] = {
                    "status": "waiting_for_window",
                    "window_start": self.settings.analysis_window_start,
                    "window_end": self.settings.analysis_window_end,
                    "updated_at": utc_now_iso(),
                }
                await asyncio.sleep(60)
                continue
            if not ffmpeg_available(self.settings):
                self.state["analyzer"] = {
                    "status": "waiting",
                    "message": f"{self.settings.ffmpeg_bin} not found on PATH",
                    "updated_at": utc_now_iso(),
                }
                await asyncio.sleep(self.settings.analysis_interval_seconds)
                continue
            if (
                self._llama_circuit_open_until is not None
                and _now() < self._llama_circuit_open_until
            ):
                self.state["analyzer"] = {
                    "status": "circuit-open",
                    "resume_at": self._llama_circuit_open_until.isoformat(timespec="seconds"),
                    "consecutive_timeouts": self._llama_timeout_count,
                    "updated_at": utc_now_iso(),
                }
                await asyncio.sleep(min(30, self.settings.analysis_interval_seconds))
                continue

            ready_before = (_now() - timedelta(seconds=self.settings.analysis_delay_seconds)).isoformat(
                timespec="seconds"
            )
            segments = self.database.get_pending_segments(
                stream_role=self.settings.analysis_stream_role,
                ready_before=ready_before,
                max_attempts=self.settings.analysis_max_attempts,
                limit=1,
            )
            if not segments:
                self.state["analyzer"] = {
                    "status": "idle",
                    "updated_at": utc_now_iso(),
                }
                await asyncio.sleep(self.settings.analysis_interval_seconds)
                continue

            segment = segments[0]
            self.state["analyzer"] = {
                "status": "analyzing",
                "segment": segment["path"],
                "updated_at": utc_now_iso(),
            }
            try:
                result = await self._analyze_segment(analyzer, segment)
                self._llama_timeout_count = 0
                self._llama_circuit_open_until = None
                should_save = result.should_save(self.settings.moment_keep_threshold)
                if result.keep_consistency_repaired(
                    self.settings.moment_keep_threshold
                ):
                    self.database.add_event(
                        "keep-consistency-repair",
                        "corrected keep=false with local child evidence "
                        f"({result.local_child_score:.2f}): {result.title}",
                    )
                if should_save:
                    cap_plan = self._apply_moment_caps(segment, result)
                    if cap_plan.skip is not None:
                        self.database.add_event(
                            cap_plan.skip.event_type, cap_plan.skip.message
                        )
                    else:
                        moment_id = await self._save_moment(
                            segment, result, analyzer
                        )
                        if moment_id >= 0:
                            for moment, event_type, scope in cap_plan.evictions:
                                self._evict_moment(
                                    moment, event_type=event_type, scope=scope
                                )
                            self.database.add_event(
                                "moment",
                                f"saved moment {moment_id}: {result.title}",
                            )
                if not should_save:
                    self.database.add_event(
                        "analysis-skip",
                        json.dumps(
                            {
                                "keep": result.keep,
                                "confidence": result.confidence,
                                "title": result.title,
                                "tags": result.tags,
                                "start_offset": result.start_offset_seconds,
                                "end_offset": result.end_offset_seconds,
                                "raw_text": result.raw_text[:1000],
                            },
                            ensure_ascii=False,
                        ),
                    )
                self.database.mark_segment_processed(int(segment["id"]))
                self.state["analyzer"] = {
                    "status": "ok",
                    "last_segment": segment["path"],
                    "last_keep": should_save,
                    "updated_at": utc_now_iso(),
                }
            except PersonFilterSkip:
                self.state["analyzer"] = {
                    "status": "ok",
                    "last_segment": segment["path"],
                    "last_keep": False,
                    "updated_at": utc_now_iso(),
                }
                if self.settings.analysis_cooldown_seconds > 0:
                    await asyncio.sleep(self.settings.analysis_cooldown_seconds)
                continue
            except Exception as exc:
                if _is_timeout_error(exc):
                    self._llama_timeout_count += 1
                    if (
                        self.settings.llama_circuit_breaker_failures > 0
                        and self._llama_timeout_count
                        >= self.settings.llama_circuit_breaker_failures
                    ):
                        self._llama_circuit_open_until = _now() + timedelta(
                            seconds=self.settings.llama_circuit_breaker_seconds
                        )
                        self.database.add_event(
                            "llama-circuit-open",
                            f"paused analysis for {self.settings.llama_circuit_breaker_seconds}s "
                            f"after {self._llama_timeout_count} consecutive timeouts",
                        )
                else:
                    self._llama_timeout_count = 0
                attempt = int(segment["analysis_attempts"]) + 1
                final = attempt >= self.settings.analysis_max_attempts
                try:
                    self.database.record_analysis_error(
                        int(segment["id"]), str(exc), final=final
                    )
                except Exception:
                    pass
                self.state["analyzer"] = {
                    "status": "error",
                    "message": str(exc),
                    "final": final,
                    "updated_at": utc_now_iso(),
                }
                self._safe_add_event("analysis-error", str(exc))
                await asyncio.sleep(self.settings.analysis_interval_seconds)
            finally:
                if self.settings.analysis_cooldown_seconds > 0:
                    await asyncio.sleep(self.settings.analysis_cooldown_seconds)

    async def _analyze_segment(self, analyzer: LlamaAnalyzer, segment: dict[str, Any]) -> AnalysisResult:
        with tempfile.TemporaryDirectory(prefix="nas-video-frames-") as temp_dir:
            prefilter_started = time.monotonic()
            duration_seconds = int(segment["duration_seconds"])
            sample_count = (
                self.settings.person_filter_sample_count
                if self.settings.person_filter_enabled
                else self.settings.sample_frame_count
            )
            sampled_frames = await sample_frames_with_offsets(
                self.settings,
                Path(segment["path"]),
                Path(temp_dir) / "frames",
                duration_seconds=duration_seconds,
                sample_count=sample_count,
            )
            extracted_frame_count = len(sampled_frames)
            sampled_frames = await filter_out_blank_frames(
                self.settings, sampled_frames
            )
            if not sampled_frames:
                self._record_prefilter(
                    prefilter_started, "blank", extracted_frame_count, 0
                )
                self.database.add_event(
                    "blank-frame-skip",
                    "all sampled frames are near-black (camera masked/off?)",
                )
                self.database.mark_segment_processed(int(segment["id"]))
                raise PersonFilterSkip()

            decision = await filter_frames_by_person_detection(
                self.settings,
                sampled_frames,
            )
            sampled_frames = decision.frames
            if not sampled_frames:
                outcome = decision.skip_reason or "no-person"
                self._record_prefilter(
                    prefilter_started, outcome, extracted_frame_count, 0
                )
                if decision.skip_reason == "adult-only":
                    self.database.add_event(
                        "adult-only-filter-skip",
                        "all visible people were confidently classified as adults",
                    )
                else:
                    self.database.add_event(
                        "person-filter-skip",
                        "no person detected in any frame",
                    )
                self.database.mark_segment_processed(int(segment["id"]))
                raise PersonFilterSkip()

            if self.settings.analysis_image_mode == "frames":
                image_paths = [frame.path for frame in sampled_frames]
                frame_offsets_seconds = [frame.offset_seconds for frame in sampled_frames]
            else:
                sheet = await build_contact_sheet_from_frames(
                    self.settings,
                    sampled_frames,
                    Path(temp_dir),
                )
                image_paths = [sheet.path]
                frame_offsets_seconds = sheet.frame_offsets_seconds
            self._record_prefilter(
                prefilter_started,
                "ready",
                extracted_frame_count,
                len(sampled_frames),
            )
            try:
                result = await analyzer.analyze(
                    video_path=Path(segment["path"]),
                    image_paths=image_paths,
                    duration_seconds=duration_seconds,
                    frame_offsets_seconds=frame_offsets_seconds,
                )
            except Exception as exc:
                if (
                    not _is_timeout_error(exc)
                    or not self.settings.llama_timeout_fallback
                    or len(sampled_frames) <= 1
                ):
                    raise
                fallback_dir = Path(temp_dir) / "fallback"
                fallback_dir.mkdir(parents=True, exist_ok=True)
                fallback_sheet = await build_contact_sheet_from_frames(
                    self.settings,
                    sampled_frames,
                    fallback_dir,
                )
                self.database.add_event(
                    "analysis-timeout-fallback",
                    f"retrying {len(sampled_frames)} frames as one contact sheet",
                )
                result = await analyzer.analyze(
                    video_path=Path(segment["path"]),
                    image_paths=[fallback_sheet.path],
                    duration_seconds=duration_seconds,
                    frame_offsets_seconds=fallback_sheet.frame_offsets_seconds,
                    image_mode="contact_sheet",
                )
            return replace(
                result,
                local_child_confirmed=decision.child_confirmed,
                local_child_score=decision.max_child_score,
            )

    def _record_prefilter(
        self,
        started_at: float,
        outcome: str,
        input_frame_count: int,
        output_frame_count: int,
    ) -> None:
        self.state["prefilter"] = {
            "status": outcome,
            "seconds": round(time.monotonic() - started_at, 3),
            "input_frames": input_frame_count,
            "output_frames": output_frame_count,
            "updated_at": utc_now_iso(),
        }

    async def _save_moment(
        self,
        segment: dict[str, Any],
        result: AnalysisResult,
        analyzer: LlamaAnalyzer,
    ) -> int:
        source_started = _parse_iso(segment["started_at"])
        source_ended = _parse_iso(segment["ended_at"])
        display_offset = timedelta(seconds=self.settings.camera_time_offset_seconds)
        display_source_started = source_started + display_offset
        display_source_ended = source_ended + display_offset

        wanted_start = source_started + timedelta(
            seconds=result.start_offset_seconds - self.settings.context_before_seconds
        )
        wanted_end = source_started + timedelta(
            seconds=result.end_offset_seconds + self.settings.context_after_seconds
        )
        max_end = wanted_start + timedelta(seconds=self.settings.max_moment_seconds)
        wanted_end = min(wanted_end, max_end)

        if self.settings.analysis_stream_role == "high":
            # Analyzing the high stream directly: the segment itself is
            # the 4K source - no need to cross-reference another stream.
            source_paths = [Path(segment["path"])]
            first_started = source_started
            last_ended = source_ended
        else:
            high_segments = self.database.find_segments_between(
                stream_role="high",
                started_before=wanted_end.isoformat(timespec="seconds"),
                ended_after=wanted_start.isoformat(timespec="seconds"),
            )
            source_rows = sorted(
                (row for row in high_segments if Path(row["path"]).exists()),
                key=lambda row: _parse_iso(row["started_at"]),
            )
            if source_rows:
                source_paths = [Path(row["path"]) for row in source_rows]
                first_started = _parse_iso(source_rows[0]["started_at"])
                last_ended = _parse_iso(source_rows[-1]["ended_at"])
            else:
                source_paths = [Path(segment["path"])]
                first_started = source_started
                last_ended = source_ended

        clip_start = max(wanted_start, first_started)
        clip_end = min(wanted_end, last_ended)
        if clip_end <= clip_start:
            clip_start = first_started
            clip_end = min(last_ended, first_started + timedelta(seconds=self.settings.segment_seconds))

        start_offset_seconds = (clip_start - first_started).total_seconds()
        duration_seconds = max(1.0, (clip_end - clip_start).total_seconds())
        if duration_seconds > self.settings.max_moment_seconds:
            clip_end = clip_start + timedelta(seconds=self.settings.max_moment_seconds)
            duration_seconds = float(self.settings.max_moment_seconds)

        display_clip_start = clip_start + display_offset
        display_clip_end = clip_end + display_offset
        day_dir = self.settings.output_dir / display_clip_start.strftime("%Y-%m-%d")
        title_slug = _slugify(result.title)
        clip_path = _unique_path(day_dir / f"{display_clip_start.strftime('%H%M%S')}_{title_slug}.mp4")
        metadata_path = clip_path.with_suffix(".json")

        # Confirm the candidate on the analysis stream before doing expensive 4K
        # extraction. Three higher-resolution frames refine the positive decision.
        verified = await self._verify_candidate(
            analyzer,
            Path(segment["path"]),
            result.start_offset_seconds,
            result.end_offset_seconds,
        )
        if not verified:
            self.database.add_event(
                "moment-verify-failed",
                f"rejected false positive before clip extraction: {result.title}",
            )
            return -1

        day_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".nas-video-staged-", dir=day_dir
        ) as staging_dir:
            staged_clip_path = Path(staging_dir) / clip_path.name
            await extract_clip(
                self.settings,
                source_paths,
                staged_clip_path,
                start_offset_seconds=start_offset_seconds,
                duration_seconds=duration_seconds,
            )
            verified_final = await self._verify_saved_clip(
                analyzer, staged_clip_path, duration_seconds
            )
            if not verified_final:
                self.database.add_event(
                    "moment-verify-failed",
                    f"rejected final source clip before publishing: {result.title}",
                )
                return -1
            clip_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_clip_path, clip_path)

        metadata = {
            "camera_name": segment["camera_name"],
            "title": result.title,
            "summary": result.summary,
            "tags": result.tags,
            "confidence": result.confidence,
            "keep_consistency_repaired": result.keep_consistency_repaired(
                self.settings.moment_keep_threshold
            ),
            "local_child_confirmed": result.local_child_confirmed,
            "local_child_score": result.local_child_score,
            "source_low_segment": segment["path"],
            "source_started_at": display_source_started.isoformat(timespec="seconds"),
            "source_ended_at": display_source_ended.isoformat(timespec="seconds"),
            "wanted_start": (wanted_start + display_offset).isoformat(timespec="seconds"),
            "wanted_end": (wanted_end + display_offset).isoformat(timespec="seconds"),
            "clip_start": display_clip_start.isoformat(timespec="seconds"),
            "clip_end": display_clip_end.isoformat(timespec="seconds"),
            "clip_duration_seconds": duration_seconds,
            "source_paths": [str(path) for path in source_paths],
            "model_raw": result.raw,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")
        daily_summary_path = _append_daily_summary(
            day_dir=day_dir,
            clip_path=clip_path,
            result=result,
            clip_start=display_clip_start,
            clip_end=display_clip_end,
        )
        metadata["daily_summary_path"] = str(daily_summary_path)
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")

        self._last_moment_saved_at = _now()

        return self.database.create_moment(
            camera_name=segment["camera_name"],
            title=result.title,
            summary=result.summary,
            tags=result.tags,
            confidence=result.confidence,
            source_low_segment_id=int(segment["id"]),
            source_started_at=display_source_started.isoformat(timespec="seconds"),
            source_ended_at=display_source_ended.isoformat(timespec="seconds"),
            clip_path=clip_path,
            metadata_path=metadata_path,
        )

    async def _extract_verification_frames(
        self,
        settings: Settings,
        video_path: Path,
        offsets: list[float],
        temp_dir: str,
        *,
        prefix: str,
    ) -> list[SampledFrame]:
        """Extract one frame per offset; return only the frames that were written."""
        frames: list[SampledFrame] = []
        for index, offset in enumerate(offsets, start=1):
            frame_path = Path(temp_dir) / f"{prefix}-{index}.jpg"
            await _extract_frame(settings, video_path, frame_path, offset)
            if frame_path.exists():
                frames.append(SampledFrame(path=frame_path, offset_seconds=offset))
        return frames

    @staticmethod
    def _dual_evidence_keep(
        local_decision: PersonFilterDecision,
        verification: DaughterVerification,
    ) -> bool:
        """Keep an otherwise-unverified clip when local child evidence is strong
        and the VLM is only weakly negative."""
        return local_decision.child_confirmed and verification.confidence < 0.75

    def _log_verification_outcome(
        self,
        kept: bool,
        local_decision: PersonFilterDecision,
        verification: DaughterVerification,
    ) -> None:
        """Record the shared dual-evidence keep / verify-detail reject event."""
        if kept:
            self.database.add_event(
                "verify-local-child-keep",
                json.dumps(
                    {
                        "child_score": local_decision.max_child_score,
                        "confidence": verification.confidence,
                        "description": verification.description,
                    },
                    ensure_ascii=False,
                ),
            )
        else:
            self.database.add_event(
                "verify-detail",
                json.dumps(
                    {
                        "confidence": verification.confidence,
                        "description": verification.description,
                        "raw_text": verification.raw_text[:1000],
                    },
                    ensure_ascii=False,
                ),
            )

    async def _verify_saved_clip(
        self,
        analyzer: LlamaAnalyzer,
        clip_path: Path,
        duration_seconds: float,
    ) -> bool:
        """Verify the saved clip with frames from its beginning, middle, and end."""
        try:
            with tempfile.TemporaryDirectory(prefix="nas-video-verify-") as temp_dir:
                last_offset = max(0.0, duration_seconds - 0.5)
                offsets = sorted(
                    {
                        min(last_offset, max(0.0, duration_seconds * fraction))
                        for fraction in (0.2, 0.5, 0.8)
                    }
                )
                frames = await self._extract_verification_frames(
                    self.settings, clip_path, offsets, temp_dir, prefix="verify"
                )
                if not frames:
                    return False
                local_decision = await filter_frames_by_person_detection(
                    self.settings, frames
                )
                if not local_decision.frames:
                    self.database.add_event(
                        "verify-local-reject",
                        local_decision.skip_reason or "no local child evidence",
                    )
                    return False
                verification = await analyzer.verify_daughter_visible(
                    [frame.path for frame in frames]
                )
                verified = verification.visible and (
                    not verification.repaired or local_decision.child_confirmed
                )
                if verification.repaired and local_decision.child_confirmed:
                    self.database.add_event(
                        "verify-consistency-repair",
                        "corrected has_daughter=false with local child evidence",
                    )
                if not verified and self._dual_evidence_keep(local_decision, verification):
                    self._log_verification_outcome(True, local_decision, verification)
                    return True
                if not verified:
                    self._log_verification_outcome(False, local_decision, verification)
                return verified
        except Exception as exc:
            self.database.add_event("verify-error", str(exc))
            if _is_timeout_error(exc):
                raise
            # Final-source verification is fail-closed: an unreadable or
            # malformed clip must never become a published moment.
            return False

    async def _verify_candidate(
        self,
        analyzer: LlamaAnalyzer,
        video_path: Path,
        start_offset_seconds: float,
        end_offset_seconds: float,
    ) -> bool:
        center = (start_offset_seconds + end_offset_seconds) / 2
        offsets = sorted(
            {
                max(0.0, start_offset_seconds),
                max(0.0, center),
                max(0.0, end_offset_seconds),
            }
        )
        verification_settings = replace(
            self.settings,
            analysis_frame_width=self.settings.verification_frame_width,
        )
        try:
            with tempfile.TemporaryDirectory(prefix="nas-video-candidate-") as temp_dir:
                frames = await self._extract_verification_frames(
                    verification_settings, video_path, offsets, temp_dir, prefix="candidate"
                )
                if not frames:
                    return False
                local_decision = await filter_frames_by_person_detection(
                    self.settings, frames
                )
                if not local_decision.frames:
                    self.database.add_event(
                        "verify-local-reject",
                        local_decision.skip_reason or "no local child evidence",
                    )
                    return False
                verification = await analyzer.verify_daughter_visible(
                    [frame.path for frame in frames]
                )
                if verification.visible:
                    return True
                if self._dual_evidence_keep(local_decision, verification):
                    self._log_verification_outcome(True, local_decision, verification)
                    return True
                self._log_verification_outcome(False, local_decision, verification)
                return False
        except Exception as exc:
            self.database.add_event("verify-error", str(exc))
            if _is_timeout_error(exc):
                raise
            return False

    async def _cleanup_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                cutoff = (_now() - timedelta(hours=self.settings.retention_hours)).isoformat(
                    timespec="seconds"
                )
                deleted = await asyncio.to_thread(self._cleanup_once, cutoff)
                self.state["cleanup"] = {
                    "status": "ok",
                    "deleted": deleted,
                    "updated_at": utc_now_iso(),
                }
            except Exception as exc:
                self.state["cleanup"] = {
                    "status": "error",
                    "message": str(exc),
                    "updated_at": utc_now_iso(),
                }
                self._safe_add_event("cleanup-error", str(exc))
            await asyncio.sleep(300)

    def _cleanup_once(self, cutoff_iso: str) -> int:
        deleted = 0
        for segment in self.database.expired_segments(cutoff_iso):
            path = Path(segment["path"])
            try:
                path.unlink(missing_ok=True)
            finally:
                self.database.mark_segment_deleted(int(segment["id"]))
                deleted += 1
        return deleted


def health_snapshot(settings: Settings, database: Database, supervisor: Supervisor | None) -> dict[str, Any]:
    disk = shutil.disk_usage(settings.data_dir)
    return {
        "configured": {
            "low_rtsp": bool(settings.rtsp_low_url),
            "high_rtsp": bool(settings.rtsp_high_url),
            "rtsp_credentials": bool(settings.rtsp_username or settings.rtsp_password),
            "camera_time_offset_seconds": settings.camera_time_offset_seconds,
            "segment_at_clocktime": settings.segment_at_clocktime,
            "stream_alignment_tolerance_seconds": settings.stream_alignment_tolerance_seconds,
            "stream_alignment_sample_count": settings.stream_alignment_sample_count,
            "llama_base_url": settings.llama_base_url,
            "model": settings.llama_model,
            "analysis_image_mode": settings.analysis_image_mode,
            "analysis_frame_width": settings.analysis_frame_width,
            "sample_frame_count": settings.sample_frame_count,
            "verification_frame_width": settings.verification_frame_width,
            "llama_timeout_fallback": settings.llama_timeout_fallback,
            "llama_circuit_breaker_failures": settings.llama_circuit_breaker_failures,
            "llama_circuit_breaker_seconds": settings.llama_circuit_breaker_seconds,
            "max_moments_per_day": settings.max_moments_per_day,
            "max_moments_per_period": settings.max_moments_per_period,
            "moment_period_boundaries": settings.moment_period_boundaries,
            "person_filter_face_threshold": settings.person_filter_face_threshold,
            "person_filter_adult_threshold": settings.person_filter_adult_threshold,
            "person_filter_child_threshold": settings.person_filter_child_threshold,
            "analysis_stream_role": settings.analysis_stream_role,
            "analysis_window_start": settings.analysis_window_start,
            "analysis_window_end": settings.analysis_window_end,
            "output_dir": str(settings.output_dir),
            "buffer_dir": str(settings.buffer_dir),
        },
        "tools": {
            "ffmpeg": ffmpeg_available(settings),
            "ffprobe": ffprobe_available(settings),
        },
        "storage": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "segments": {
            "pending_low": database.count_pending_segments(stream_role="low"),
            "pending_high": database.count_pending_segments(stream_role="high"),
            "latest_low": database.latest_segment("low"),
            "latest_high": database.latest_segment("high"),
        },
        "stream_alignment": (
            supervisor.snapshot().get("stream_alignment", {"status": "unknown"})
            if supervisor
            else {"status": "not-started"}
        ),
        "workers": supervisor.snapshot() if supervisor else {"status": "not-started"},
        "events": database.recent_events(limit=10),
    }
