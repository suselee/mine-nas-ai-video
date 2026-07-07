from __future__ import annotations

import asyncio
import math
import re
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from .config import Settings


SEGMENT_RE = re.compile(r"^(?P<camera>.+)_(?P<role>low|high)_(?P<stamp>\d{8}T\d{6})\.mp4$")


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

        command = [
            settings.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{max(start_offset_seconds, 0):.3f}",
            "-i",
            str(input_path),
            "-t",
            f"{duration_seconds:.3f}",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
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
            raise RuntimeError(f"ffmpeg clip extraction failed: {message}")


def segment_time_window(started_at: datetime, duration_seconds: int) -> tuple[datetime, datetime]:
    return started_at, started_at + timedelta(seconds=duration_seconds)
