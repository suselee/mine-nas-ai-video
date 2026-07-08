from __future__ import annotations

import asyncio
import math
import re
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import Settings


SEGMENT_RE = re.compile(r"^(?P<camera>.+)_(?P<role>low|high)_(?P<stamp>\d{8}T\d{6})\.mp4$")


def _hwaccel_args(settings: Settings) -> list[str]:
    """Return ffmpeg input-level hwaccel flags, or empty list if disabled.

    With FFMPEG_HWACCEL=vaapi the flags -hwaccel vaapi are prepended
    before -i so the HEVC decode runs on the Intel GPU. Frames are
    copied back to system memory for software filters (scale, select,
    showinfo), which keeps all existing filters working without needing
    VAAPI-specific filter chains. Set FFMPEG_HWACCEL to empty (default)
    for pure software decoding.
    """
    accel = settings.ffmpeg_hwaccel
    if not accel or accel == "none" or accel == "auto":
        if accel == "auto":
            return ["-hwaccel", "auto"]
        return []
    return ["-hwaccel", accel]


def ffmpeg_available(settings: Settings) -> bool:
    return shutil.which(settings.ffmpeg_bin) is not None


def ffprobe_available(settings: Settings) -> bool:
    return shutil.which(settings.ffprobe_bin) is not None


def parse_segment_filename(path: Path) -> tuple[str, str, datetime] | None:
    match = SEGMENT_RE.match(path.name)
    if not match:
        return None
    started_at = datetime.strptime(match.group("stamp"), "%Y%m%dT%H%M%S")
    started_at = started_at.astimezone()
    return match.group("camera"), match.group("role"), started_at


def segment_output_pattern(settings: Settings, role: str) -> Path:
    directory = settings.low_buffer_dir if role == "low" else settings.high_buffer_dir
    return directory / f"{settings.camera_name}_{role}_%Y%m%dT%H%M%S.mp4"


def build_recorder_command(settings: Settings, role: str, rtsp_url: str) -> list[str]:
    command = [
        settings.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "warning",
    ]
    if settings.rtsp_transport:
        command.extend(["-rtsp_transport", settings.rtsp_transport])
    command.extend(
        [
            "-i",
            rtsp_url,
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c",
            "copy",
            "-f",
            "segment",
            "-segment_time",
            str(settings.segment_seconds),
            "-reset_timestamps",
            "1",
            "-strftime",
            "1",
            str(segment_output_pattern(settings, role)),
        ]
    )
    return command


def sample_offsets(duration_seconds: int, frame_count: int, minimum_spacing_seconds: int) -> list[float]:
    if duration_seconds <= 0 or frame_count <= 0:
        return []

    spacing = max(minimum_spacing_seconds, 1)
    max_by_spacing = max(1, math.floor(duration_seconds / spacing) + 1)
    count = min(frame_count, max_by_spacing)
    step = duration_seconds / (count + 1)
    last_offset = max(duration_seconds - 1, 0)
    return [min(last_offset, max(0.0, step * (index + 1))) for index in range(count)]


_SHOWINFO_PTS_RE = re.compile(r"pts_time:\s*([0-9.]+)")


async def _detect_motion_timestamps(
    settings: Settings,
    video_path: Path,
    threshold: float,
) -> list[float]:
    """Run ffmpeg with scene detection to find timestamps of frames with motion.

    Uses scale=160:-2,fps=1 before select to minimize CPU on 4K HEVC sources:
    the decode still happens at full resolution, but the scene-detection filter
    chain operates on tiny 160px frames at 1fps instead of full-res 30fps,
    which dramatically cuts filter overhead. Scene detection at 160px/1fps is
    sufficient for detecting a person walking in a room. No JPEG is written.
    """
    command = [
        settings.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "info",
        *_hwaccel_args(settings),
        "-i",
        str(video_path),
        "-an",
        "-vf",
        f"scale=160:-2,fps=1,select='gt(scene\\,{threshold})',showinfo",
        "-f",
        "null",
        "-",
    ]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    # showinfo writes to stderr; non-zero return code is tolerated because
    # some segments may be partially written - we just return whatever
    # timestamps we parsed and let the caller fall back if empty.
    text = stderr.decode("utf-8", errors="replace")
    timestamps: list[float] = []
    for line in text.splitlines():
        if "showinfo" not in line:
            continue
        match = _SHOWINFO_PTS_RE.search(line)
        if match:
            timestamps.append(float(match.group(1)))
    timestamps.sort()
    return timestamps


def _bucket_index(offset: float, bucket_seconds: float) -> int:
    return int(offset // bucket_seconds)


def _motion_aware_offsets(
    duration_seconds: int,
    motion_timestamps: list[float],
    frame_count: int,
    bucket_count: int = 12,
) -> list[float]:
    """Choose frame offsets that cover motion and keep static baselines.

    Treats each motion timestamp as a candidate frame (so a bucket with
    several motion peaks can yield more than one frame), and fills the
    remaining budget with static baseline frames spread across buckets
    that had no motion - so a child sitting still drawing is still
    captured, but an empty static room gets fewer frames and the LLM
    can more reliably judge keep=false. With no motion at all this
    degrades to roughly even sampling across the segment.
    """
    if duration_seconds <= 0 or frame_count <= 0:
        return []

    bucket_count = max(1, min(bucket_count, duration_seconds))
    bucket_seconds = duration_seconds / bucket_count

    # Motion candidates: unique timestamps clamped into range.
    motion_candidates: list[float] = []
    seen = set()
    for ts in motion_timestamps:
        clamped = max(0.0, min(ts, duration_seconds - 1))
        key = round(clamped, 3)
        if key not in seen:
            seen.add(key)
            motion_candidates.append(clamped)
    motion_candidates.sort()

    # Static candidates: midpoint of each bucket that has no motion.
    motion_buckets = {_bucket_index(ts, bucket_seconds) for ts in motion_candidates}
    static_candidates: list[float] = []
    for i in range(bucket_count):
        if i not in motion_buckets:
            static_candidates.append(min((i + 0.5) * bucket_seconds, duration_seconds - 1))

    if len(motion_candidates) >= frame_count:
        # Plenty of motion: subsample it evenly across the segment so we
        # don't cluster all frames in one burst of motion.
        chosen_motion = _even_subsample(motion_candidates, frame_count)
        offsets = chosen_motion
    else:
        # Take all motion candidates, then fill remaining slots with
        # static baselines spread across the segment.
        remaining = frame_count - len(motion_candidates)
        chosen_static = _even_subsample(static_candidates, remaining)
        offsets = sorted(motion_candidates + chosen_static)

    # Dedupe + top up if rounding shrank the list below frame_count.
    offsets = sorted(set(offsets))
    if len(offsets) < frame_count:
        even = sample_offsets(duration_seconds, frame_count, 1)
        for o in even:
            if o not in offsets:
                offsets.append(o)
                if len(offsets) >= frame_count:
                    break
        offsets = sorted(offsets)
    return offsets[:frame_count]


def _even_subsample(candidates: list[float], count: int) -> list[float]:
    """Pick `count` items from `candidates` spread evenly by index."""
    if count <= 0 or not candidates:
        return []
    if count >= len(candidates):
        return list(candidates)
    step = len(candidates) / count
    return [candidates[int(step * i)] for i in range(count)]


def contact_sheet_layout(frame_count: int, preferred_columns: int) -> tuple[int, int]:
    if frame_count <= 0:
        return 0, 0
    columns = max(1, min(preferred_columns, frame_count))
    rows = math.ceil(frame_count / columns)
    return columns, rows


def _xstack_layout(frame_count: int, columns: int, padding: int) -> str:
    def axis_expr(unit: str, index: int) -> str:
        if index == 0:
            return "0"
        parts = [f"{unit}0" for _ in range(index)]
        if padding:
            parts.append(str(index * padding))
        return "+".join(parts)

    positions: list[str] = []
    for index in range(frame_count):
        row, column = divmod(index, columns)
        x = axis_expr("w", column)
        y = axis_expr("h", row)
        positions.append(f"{x}_{y}")
    return "|".join(positions)


async def _run_command(command: list[str], error_prefix: str) -> None:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if process.returncode != 0:
        message = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{error_prefix}: {message}")


async def _extract_frame(settings: Settings, video_path: Path, output_path: Path, offset_seconds: float) -> None:
    command = [
        settings.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *_hwaccel_args(settings),
        "-ss",
        f"{offset_seconds:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        f"scale={settings.analysis_frame_width}:-2",
        "-q:v",
        "4",
        str(output_path),
    ]
    await _run_command(command, "ffmpeg frame extraction failed")


async def sample_frames(
    settings: Settings,
    video_path: Path,
    output_dir: Path,
    *,
    duration_seconds: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if settings.sample_mode == "motion_aware":
        try:
            motion_ts = await _detect_motion_timestamps(
                settings,
                video_path,
                settings.motion_threshold,
            )
            offsets = _motion_aware_offsets(
                duration_seconds,
                motion_ts,
                settings.sample_frame_count,
            )
        except Exception:
            # Corrupt segment or ffmpeg hiccup: fall back to even sampling
            # so the segment is still analyzed rather than skipped.
            offsets = sample_offsets(
                duration_seconds,
                settings.sample_frame_count,
                settings.sample_every_seconds,
            )
    else:
        offsets = sample_offsets(
            duration_seconds,
            settings.sample_frame_count,
            settings.sample_every_seconds,
        )
    frame_paths: list[Path] = []
    for index, offset in enumerate(offsets, start=1):
        frame_path = output_dir / f"frame_{index:03d}.jpg"
        await _extract_frame(settings, video_path, frame_path, offset)
        if frame_path.exists():
            frame_paths.append(frame_path)
    return frame_paths


async def build_contact_sheet(
    settings: Settings,
    video_path: Path,
    output_dir: Path,
    *,
    duration_seconds: int,
) -> Path:
    frames_dir = output_dir / "frames"
    frame_paths = await sample_frames(
        settings,
        video_path,
        frames_dir,
        duration_seconds=duration_seconds,
    )
    if not frame_paths:
        raise RuntimeError("no frames available for contact sheet")

    sheet_path = output_dir / "contact_sheet.jpg"
    if len(frame_paths) == 1:
        await asyncio.to_thread(shutil.copy2, frame_paths[0], sheet_path)
        return sheet_path

    columns, _ = contact_sheet_layout(len(frame_paths), settings.contact_sheet_columns)
    padding = max(settings.contact_sheet_padding, 0)
    command = [
        settings.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    for frame_path in frame_paths:
        command.extend(["-i", str(frame_path)])
    command.extend(
        [
            "-filter_complex",
            (
                f"xstack=inputs={len(frame_paths)}:"
                f"layout={_xstack_layout(len(frame_paths), columns, padding)}:"
                "fill=black"
            ),
            "-frames:v",
            "1",
            "-q:v",
            "4",
            str(sheet_path),
        ]
    )
    await _run_command(command, "ffmpeg contact sheet failed")
    return sheet_path


def _concat_escape(path: Path) -> str:
    return str(path).replace("'", "'\\''")


async def concat_segments(settings: Settings, segment_paths: list[Path], output_path: Path) -> None:
    if not segment_paths:
        raise ValueError("no segment paths provided")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(segment_paths) == 1:
        await asyncio.to_thread(shutil.copy2, segment_paths[0], output_path)
        return

    concat_list = output_path.with_suffix(".concat.txt")
    concat_list.write_text(
        "".join(f"file '{_concat_escape(path)}'\n" for path in segment_paths),
        encoding="utf-8",
    )
    try:
        command = [
            settings.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(output_path),
        ]
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg concat failed: {message}")
    finally:
        concat_list.unlink(missing_ok=True)


async def extract_clip(
    settings: Settings,
    segment_paths: list[Path],
    output_path: Path,
    *,
    start_offset_seconds: float,
    duration_seconds: float,
) -> None:
    if not segment_paths:
        raise ValueError("no segment paths provided")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="nas-video-clip-") as temp_dir:
        if len(segment_paths) == 1:
            input_path = segment_paths[0]
        else:
            input_path = Path(temp_dir) / "combined.mp4"
            await concat_segments(settings, segment_paths, input_path)

        # Saved clips are served to browsers via /api/moments/{id}/video.
        # Fast path (default): -c copy preserves the RTSP codec untouched,
        # which is fast and lossless but may emit HEVC, which Chrome/
        # Firefox/Edge cannot decode - browsers would play audio only.
        # To make clips browser-friendly, set CLIP_VIDEO_CODEC=libx264
        # (and CLIP_AUDIO_CODEC=aac) to re-encode at extract time.
        reencoding_video = settings.clip_video_codec != "copy"
        reencoding_audio = settings.clip_audio_codec != "copy"

        command = [
            settings.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
        ]
        # For -c copy use input seeking (fast, keyframe-aligned). When
        # re-encoding, use output seeking (decode from 0, trim with -ss
        # after -i) so the encoder starts on a clean keyframe and the
        # output has correct timestamps for player compatibility.
        # hwaccel only helps when decoding (re-encode path); -c copy
        # never decodes so it's omitted there.
        if not reencoding_video and not reencoding_audio:
            command += ["-ss", f"{max(start_offset_seconds, 0):.3f}"]
            command += ["-i", str(input_path), "-t", f"{duration_seconds:.3f}"]
        else:
            command += _hwaccel_args(settings)
            command += ["-i", str(input_path)]
            command += [
                "-ss",
                f"{max(start_offset_seconds, 0):.3f}",
                "-t",
                f"{duration_seconds:.3f}",
            ]

        if settings.clip_video_codec == "copy":
            command += ["-c:v", "copy"]
        else:
            command += [
                "-c:v",
                settings.clip_video_codec,
                "-preset",
                settings.clip_video_preset,
                "-crf",
                str(settings.clip_video_crf),
                "-pix_fmt",
                "yuv420p",
            ]
        if settings.clip_audio_codec == "copy":
            command += ["-c:a", "copy"]
        else:
            command += ["-c:a", settings.clip_audio_codec, "-b:a", "192k"]
        command += ["-avoid_negative_ts", "make_zero"]
        # +faststart moves the moov atom to the front so byte-range
        # seeking works for browser playback. Skip it for -c copy: it
        # requires a second pass that would fragment HEVC streams, and
        # browsers won't play HEVC anyway.
        if reencoding_video or reencoding_audio:
            command += ["-movflags", "+faststart"]
        command += [str(output_path)]
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg clip extraction failed: {message}")


def segment_time_window(started_at: datetime, duration_seconds: int) -> tuple[datetime, datetime]:
    return started_at, started_at + timedelta(seconds=duration_seconds)
