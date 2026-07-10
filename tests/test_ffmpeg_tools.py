from pathlib import Path

import asyncio

from nas_video_summarizer.ffmpeg_tools import (
    SampledFrame,
    _hwaccel_args,
    _motion_aware_offsets,
    contact_sheet_layout,
    filter_out_blank_frames,
    sample_offsets,
)
from nas_video_summarizer.config import Settings, load_settings


def test_sample_offsets_are_evenly_distributed():
    offsets = sample_offsets(duration_seconds=120, frame_count=4, minimum_spacing_seconds=30)

    assert offsets == [24.0, 48.0, 72.0, 96.0]


def test_sample_offsets_respect_minimum_spacing_limit():
    offsets = sample_offsets(duration_seconds=60, frame_count=8, minimum_spacing_seconds=30)

    assert offsets == [15.0, 30.0, 45.0]


def test_contact_sheet_layout_caps_columns_to_frame_count():
    assert contact_sheet_layout(frame_count=4, preferred_columns=2) == (2, 2)
    assert contact_sheet_layout(frame_count=1, preferred_columns=2) == (1, 1)


def test_motion_aware_offsets_keeps_static_baseline():
    # 120s segment, motion only in seconds 30-60, 6 frames.
    # At least one sampled offset must land in a static region (0-25 or
    # 65-120) so a child sitting still outside the motion window is still
    # captured instead of being dropped.
    motion_ts = [32.0, 35.0, 40.0, 50.0, 55.0]
    offsets = _motion_aware_offsets(120, motion_ts, 6, bucket_count=12)

    assert len(offsets) == 6
    assert any(o < 25 or o > 65 for o in offsets), offsets


def test_motion_aware_offsets_prefers_high_motion_buckets():
    # Motion concentrated in buckets 3 (30-40s) and 4 (40-50s).
    # Every motion timestamp should become a candidate frame, so the
    # 30-50s region gets at least 3 frames (one per timestamp), while
    # pure even sampling over 120s with 6 frames would put only ~1
    # frame in that region.
    motion_ts = [35.0, 40.0, 45.0]
    offsets = _motion_aware_offsets(120, motion_ts, 6, bucket_count=12)

    assert any(30 <= o < 40 for o in offsets), offsets
    assert any(40 <= o < 50 for o in offsets), offsets
    motion_region_count = sum(1 for o in offsets if 30 <= o < 50)
    assert motion_region_count >= len(motion_ts), offsets


def test_motion_aware_offsets_all_static_falls_back_to_even():
    # No motion timestamps at all: still produce frame_count offsets
    # spread across the segment (every bucket gets 1 frame).
    offsets = _motion_aware_offsets(120, [], 6, bucket_count=12)

    assert len(offsets) == 6
    assert offsets[0] > 0
    assert offsets[-1] < 120


def _settings_with_hwaccel(hwaccel: str) -> Settings:
    import os
    os.environ["FFMPEG_HWACCEL"] = hwaccel
    return load_settings("/nonexistent.env")


def test_hwaccel_args_empty_when_disabled():
    settings = _settings_with_hwaccel("")
    assert _hwaccel_args(settings) == []


def test_hwaccel_args_vaapi():
    import os
    os.environ["FFMPEG_HWACCEL"] = "vaapi"
    settings = load_settings("/nonexistent.env")
    assert _hwaccel_args(settings) == ["-hwaccel", "vaapi"]


class _FakeProc:
    def __init__(self, stderr):
        self._stderr = stderr.encode() if isinstance(stderr, str) else stderr

    async def communicate(self):
        return (b"", self._stderr)


def _fake_spawn(monkeypatch, luma):
    async def _spawn(*args, **kwargs):
        return _FakeProc(f"lavfi.signalstats.1.luma_average={luma}")
    monkeypatch.setattr("asyncio.create_subprocess_exec", _spawn)


def test_is_blank_frame_black(monkeypatch):
    from nas_video_summarizer import ffmpeg_tools as ft

    _fake_spawn(monkeypatch, 2.34)
    settings = load_settings("/nonexistent.env")
    assert asyncio.run(ft._is_blank_frame(settings, Path("/nope.jpg"))) is True


def test_is_blank_frame_normal(monkeypatch):
    from nas_video_summarizer import ffmpeg_tools as ft

    _fake_spawn(monkeypatch, 120.5)
    settings = load_settings("/nonexistent.env")
    assert asyncio.run(ft._is_blank_frame(settings, Path("/nope.jpg"))) is False


def test_filter_out_blank_frames_drops_all_blank(monkeypatch):
    from nas_video_summarizer import ffmpeg_tools as ft

    _fake_spawn(monkeypatch, 1.0)
    settings = load_settings("/nonexistent.env")
    frames = [SampledFrame(path=Path(f"/f{i}.jpg"), offset_seconds=i) for i in range(3)]
    out = asyncio.run(filter_out_blank_frames(settings, frames))
    assert out == []


def test_filter_out_blank_frames_keeps_mixed(monkeypatch):
    from nas_video_summarizer import ffmpeg_tools as ft

    _fake_spawn(monkeypatch, 100.0)
    settings = load_settings("/nonexistent.env")
    frames = [SampledFrame(path=Path("/f.jpg"), offset_seconds=0)]
    out = asyncio.run(filter_out_blank_frames(settings, frames))
    assert len(out) == 1

