import asyncio
from dataclasses import replace
from datetime import datetime, timedelta

from nas_video_summarizer.config import load_settings
from nas_video_summarizer.database import Database
from nas_video_summarizer.ffmpeg_tools import SampledFrame
from nas_video_summarizer.llm import AnalysisResult
from nas_video_summarizer.workers import Supervisor, _append_daily_summary, _in_analysis_window


def test_append_daily_summary(tmp_path):
    start = datetime.fromisoformat("2026-07-06T10:00:00+08:00")
    end = start + timedelta(seconds=90)
    clip_path = tmp_path / "100000_painting.mp4"
    result = AnalysisResult(
        keep=True,
        title="Painting together",
        summary="A quiet family moment at the table.",
        tags=["art", "family"],
        confidence=0.82,
        start_offset_seconds=0,
        end_offset_seconds=60,
        raw={},
    )

    summary_path = _append_daily_summary(
        day_dir=tmp_path,
        clip_path=clip_path,
        result=result,
        clip_start=start,
        clip_end=end,
    )

    text = summary_path.read_text(encoding="utf-8")
    assert "# Family Moments - 2026-07-06" in text
    assert "## 10:00:00 - Painting together" in text
    assert "[100000_painting.mp4](100000_painting.mp4)" in text
    assert "art, family" in text


def _dt(hour: int, minute: int) -> datetime:
    return datetime(2026, 7, 8, hour, minute, tzinfo=datetime.now().astimezone().tzinfo)


def test_in_analysis_window_empty_is_always_true():
    assert _in_analysis_window(_dt(3, 0), "", "") is True
    assert _in_analysis_window(_dt(15, 0), "", "") is True
    assert _in_analysis_window(_dt(23, 59), "", "") is True


def test_in_analysis_window_crosses_midnight():
    # Window: 21:15 -> 06:00 (next day)
    assert _in_analysis_window(_dt(21, 15), "21:15", "06:00") is True
    assert _in_analysis_window(_dt(23, 30), "21:15", "06:00") is True
    assert _in_analysis_window(_dt(3, 0), "21:15", "06:00") is True
    assert _in_analysis_window(_dt(5, 59), "21:15", "06:00") is True
    # Just before window opens
    assert _in_analysis_window(_dt(21, 14), "21:15", "06:00") is False
    # At window close
    assert _in_analysis_window(_dt(6, 0), "21:15", "06:00") is False
    # Middle of day
    assert _in_analysis_window(_dt(12, 0), "21:15", "06:00") is False


def test_in_analysis_window_same_day():
    # Window: 09:00 -> 17:00
    assert _in_analysis_window(_dt(9, 0), "09:00", "17:00") is True
    assert _in_analysis_window(_dt(12, 30), "09:00", "17:00") is True
    assert _in_analysis_window(_dt(16, 59), "09:00", "17:00") is True
    assert _in_analysis_window(_dt(8, 59), "09:00", "17:00") is False
    assert _in_analysis_window(_dt(17, 0), "09:00", "17:00") is False
    assert _in_analysis_window(_dt(22, 0), "09:00", "17:00") is False


def test_analyze_segment_passes_actual_frame_offsets(tmp_path, monkeypatch):
    settings = replace(load_settings("/nonexistent.env"), analysis_image_mode="frames")
    database = Database(tmp_path / "test.sqlite3")
    database.migrate()
    supervisor = Supervisor(settings, database)
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"jpeg")
    captured: dict[str, list[float]] = {}

    async def fake_sample_frames_with_offsets(settings, video_path, output_dir, *, duration_seconds):
        return [SampledFrame(path=frame_path, offset_seconds=73.5)]

    class FakeAnalyzer:
        async def analyze(self, **kwargs):
            captured["offsets"] = kwargs["frame_offsets_seconds"]
            return AnalysisResult(
                keep=False,
                title="quiet",
                summary="none",
                tags=[],
                confidence=0.1,
                start_offset_seconds=0,
                end_offset_seconds=1,
                raw={},
            )

    monkeypatch.setattr(
        "nas_video_summarizer.workers.sample_frames_with_offsets",
        fake_sample_frames_with_offsets,
    )

    asyncio.run(
        supervisor._analyze_segment(
            FakeAnalyzer(),
            {
                "path": str(tmp_path / "segment.mp4"),
                "duration_seconds": 120,
            },
        )
    )

    assert captured["offsets"] == [73.5]
