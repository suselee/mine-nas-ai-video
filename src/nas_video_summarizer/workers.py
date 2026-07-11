from __future__ import annotations

import asyncio
import json
import re
import shutil
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import Settings
from .database import Database, utc_now_iso
from .ffmpeg_tools import (
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
from .llm import AnalysisResult, LlamaAnalyzer


class PersonFilterSkip(Exception):
    """Raised when person filter detects no person in any sampled frame."""

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


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


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


class Supervisor:
    def __init__(self, settings: Settings, database: Database):
        self.settings = settings
        self.database = database
        self.stop_event = asyncio.Event()
        self.tasks: list[asyncio.Task[None]] = []
        self._last_moment_saved_at: datetime | None = None
        self.state: dict[str, Any] = {
            "started_at": utc_now_iso(),
            "recorders": {
                "low": {"status": "not-started"},
                "high": {"status": "not-started"},
            },
            "scanner": {"status": "not-started"},
            "prefilter": {"status": "not-started"},
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
                while process.returncode is None and not self.stop_event.is_set():
                    await asyncio.sleep(1)
                if self.stop_event.is_set() and process.returncode is None:
                    await _stop_process(process)
                    return
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
            except Exception as exc:
                self.state["scanner"] = {
                    "status": "error",
                    "message": str(exc),
                    "updated_at": utc_now_iso(),
                }
                self.database.add_event("scanner-error", str(exc))
            await asyncio.sleep(10)

    def _scan_once(self) -> int:
        now = _now()
        seen = 0
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
                self.database.upsert_segment(
                    camera_name=camera_name,
                    stream_role=role,
                    path=path,
                    started_at=started.isoformat(timespec="seconds"),
                    ended_at=ended.isoformat(timespec="seconds"),
                    duration_seconds=self.settings.segment_seconds,
                    size_bytes=stat.st_size,
                )
                seen += 1
        return seen

    def _moment_cooldown_active(self) -> bool:
        if self.settings.moment_cooldown_seconds <= 0:
            return False
        if self._last_moment_saved_at is None:
            return False
        return (_now() - self._last_moment_saved_at).total_seconds() < self.settings.moment_cooldown_seconds

    def _daily_cap_eviction_target(self, segment: dict[str, Any], result: AnalysisResult):
        """Return the weakest moment to evict for the daily cap, or a decision.

        Returns:
          * ``None``      — under the cap, save normally (no eviction).
          * a dict       — at the cap and this new clip is better; evict this
                           weakest moment, then save the new one.
          * ``False``    — at the cap and the new clip is not better; skip it.
        """
        cap = self.settings.max_moments_per_day
        if cap <= 0:
            return None
        day = str(segment.get("started_at", ""))[:10]
        if not day:
            return None
        if self.database.count_moments_on_day(day) < cap:
            return None
        weakest = self.database.weakest_moment_on_day(day)
        if weakest is None:
            return None
        if result.confidence > float(weakest["confidence"]):
            return weakest
        return False

    def _evict_moment(self, moment: dict[str, Any]) -> None:
        for key in ("clip_path", "metadata_path"):
            path = Path(str(moment[key]))
            path.unlink(missing_ok=True)
        self.database.delete_moment_by_clip(str(moment["clip_path"]))
        self.database.add_event(
            "daily-cap-evict",
            f"replaced weakest '{moment['title']}' (confidence {float(moment['confidence']):.2f})",
        )

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
                if result.should_save(self.settings.moment_keep_threshold):
                    if self._moment_cooldown_active():
                        self.database.add_event(
                            "moment-cooldown",
                            f"skipped '{result.title}' due to cooldown",
                        )
                    else:
                        evict = self._daily_cap_eviction_target(segment, result)
                        if evict is False:
                            self.database.add_event(
                                "daily-cap",
                                f"skipped '{result.title}' (daily limit {self.settings.max_moments_per_day} reached, "
                                f"confidence {result.confidence:.2f} not above weakest)",
                            )
                        else:
                            if evict is not None:
                                self._evict_moment(evict)
                            moment_id = await self._save_moment(segment, result, analyzer)
                            if moment_id >= 0:
                                self.database.add_event("moment", f"saved moment {moment_id}: {result.title}")
                if not result.should_save(self.settings.moment_keep_threshold):
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
                    "last_keep": result.should_save(self.settings.moment_keep_threshold),
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
                attempt = int(segment["analysis_attempts"]) + 1
                final = attempt >= self.settings.analysis_max_attempts
                self.database.record_analysis_error(int(segment["id"]), str(exc), final=final)
                self.state["analyzer"] = {
                    "status": "error",
                    "message": str(exc),
                    "final": final,
                    "updated_at": utc_now_iso(),
                }
                self.database.add_event("analysis-error", str(exc))
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
            return await analyzer.analyze(
                video_path=Path(segment["path"]),
                image_paths=image_paths,
                duration_seconds=duration_seconds,
                frame_offsets_seconds=frame_offsets_seconds,
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
            source_rows = [row for row in high_segments if Path(row["path"]).exists()]
            if source_rows:

                def _overlap_seconds(row: dict[str, Any]) -> float:
                    row_start = _parse_iso(row["started_at"])
                    row_end = _parse_iso(row["ended_at"])
                    overlap_start = max(row_start, wanted_start)
                    overlap_end = min(row_end, wanted_end)
                    return max(0.0, (overlap_end - overlap_start).total_seconds())

                best_row = max(source_rows, key=_overlap_seconds)
                source_paths = [Path(best_row["path"])]
                first_started = _parse_iso(best_row["started_at"])
                last_ended = _parse_iso(best_row["ended_at"])
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

        day_dir = self.settings.output_dir / source_started.strftime("%Y-%m-%d")
        title_slug = _slugify(result.title)
        clip_path = _unique_path(day_dir / f"{source_started.strftime('%H%M%S')}_{title_slug}.mp4")
        metadata_path = clip_path.with_suffix(".json")

        await extract_clip(
            self.settings,
            source_paths,
            clip_path,
            start_offset_seconds=start_offset_seconds,
            duration_seconds=duration_seconds,
        )

        # Post-save verification: confirm the clip actually contains the daughter.
        # This catches LLM hallucinations where keep=true but the clip is empty.
        verified = await self._verify_saved_clip(analyzer, clip_path, duration_seconds)
        if not verified:
            clip_path.unlink(missing_ok=True)
            self.database.add_event(
                "moment-verify-failed",
                f"deleted false positive clip: {result.title}",
            )
            return -1

        metadata = {
            "camera_name": segment["camera_name"],
            "title": result.title,
            "summary": result.summary,
            "tags": result.tags,
            "confidence": result.confidence,
            "source_low_segment": segment["path"],
            "source_started_at": source_started.isoformat(timespec="seconds"),
            "source_ended_at": source_ended.isoformat(timespec="seconds"),
            "wanted_start": wanted_start.isoformat(timespec="seconds"),
            "wanted_end": wanted_end.isoformat(timespec="seconds"),
            "clip_start": clip_start.isoformat(timespec="seconds"),
            "clip_end": clip_end.isoformat(timespec="seconds"),
            "clip_duration_seconds": duration_seconds,
            "source_paths": [str(path) for path in source_paths],
            "model_raw": result.raw,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")
        daily_summary_path = _append_daily_summary(
            day_dir=day_dir,
            clip_path=clip_path,
            result=result,
            clip_start=clip_start,
            clip_end=clip_end,
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
            source_started_at=source_started.isoformat(timespec="seconds"),
            source_ended_at=source_ended.isoformat(timespec="seconds"),
            clip_path=clip_path,
            metadata_path=metadata_path,
        )

    async def _verify_saved_clip(
        self,
        analyzer: LlamaAnalyzer,
        clip_path: Path,
        duration_seconds: float,
    ) -> bool:
        """Extract a frame from the saved clip and verify it contains the daughter."""
        try:
            with tempfile.TemporaryDirectory(prefix="nas-video-verify-") as temp_dir:
                frame_path = Path(temp_dir) / "verify.jpg"
                middle_offset = max(0.0, duration_seconds / 2 - 1.0)
                await _extract_frame(self.settings, clip_path, frame_path, middle_offset)
                if not frame_path.exists():
                    return False
                return await analyzer.verify_daughter_visible(frame_path)
        except Exception as exc:
            self.database.add_event("verify-error", str(exc))
            # If verification itself fails, be conservative and keep the clip.
            return True

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
                self.database.add_event("cleanup-error", str(exc))
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
            "llama_base_url": settings.llama_base_url,
            "model": settings.llama_model,
            "analysis_image_mode": settings.analysis_image_mode,
            "analysis_frame_width": settings.analysis_frame_width,
            "sample_frame_count": settings.sample_frame_count,
            "person_filter_face_threshold": settings.person_filter_face_threshold,
            "person_filter_adult_threshold": settings.person_filter_adult_threshold,
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
        "workers": supervisor.snapshot() if supervisor else {"status": "not-started"},
        "events": database.recent_events(limit=10),
    }
