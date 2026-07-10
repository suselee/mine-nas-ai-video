import asyncio
from dataclasses import replace
from datetime import datetime, timedelta

from nas_video_summarizer.config import load_settings
from nas_video_summarizer.database import Database
from nas_video_summarizer.ffmpeg_tools import SampledFrame
from nas_video_summarizer.llm import AnalysisResult
from nas_video_summarizer.workers import (
    Supervisor,
    _append_daily_summary,
    _in_analysis_window,
    _in_time_window,
    _in_record_window,
)


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


def test_time_window_helpers_and_record_alias():
    # _in_time_window is the generic backend for both analysis and record.
    assert _in_time_window(_dt(3, 0), "21:15", "06:00") is True
    assert _in_time_window(_dt(12, 0), "21:15", "06:00") is False
    assert _in_time_window(_dt(10, 0), "", "") is True
    # record window same-day (07:00 .. 21:00, end exclusive)
    assert _in_record_window(_dt(20, 0), "07:00", "21:00") is True
    assert _in_record_window(_dt(21, 30), "07:00", "21:00") is False
    assert _in_record_window(_dt(7, 0), "07:00", "21:00") is True
    assert _in_record_window(_dt(10, 0), "", "") is True


class _FakeCapDB:
    def __init__(self, count, weakest):
        self._count = count
        self._weakest = weakest
        self.evicted = None

    def count_moments_on_day(self, day):
        return self._count

    def min_confidence_on_day(self, day):
        return float(self._weakest["confidence"]) if self._weakest else 1.0

    def weakest_moment_on_day(self, day):
        return self._weakest

    def delete_moment_by_clip(self, clip):
        self.evicted = clip

    def add_event(self, *args, **kwargs):
        pass


def _supervisor_with(db, max_per_day=20):
    settings = replace(
        load_settings("/nonexistent.env"),
        max_moments_per_day=max_per_day,
        moment_keep_threshold=0.5,
    )
    return Supervisor(settings, db)


def _segment(day="2026-07-10T10:00:00+08:00"):
    return {"started_at": day}


def _result(confidence):
    return AnalysisResult(
        keep=True,
        title="x",
        summary="",
        tags=[],
        confidence=confidence,
        start_offset_seconds=0,
        end_offset_seconds=1,
        raw={},
    )


def test_daily_cap_under_limit_saves():
    db = _FakeCapDB(
        count=5,
        weakest={"confidence": 0.3, "clip_path": "/x", "metadata_path": "/y", "title": "t", "id": 1},
    )
    sup = _supervisor_with(db, max_per_day=20)
    assert sup._daily_cap_eviction_target(_segment(), _result(0.6)) is None


def test_daily_cap_at_limit_better_evicts():
    db = _FakeCapDB(
        count=20,
        weakest={"confidence": 0.3, "clip_path": "/x.mp4", "metadata_path": "/x.json", "title": "weak", "id": 1},
    )
    sup = _supervisor_with(db, max_per_day=20)
    target = sup._daily_cap_eviction_target(_segment(), _result(0.8))
    assert isinstance(target, dict) and target["title"] == "weak"


def test_daily_cap_at_limit_worse_skips():
    db = _FakeCapDB(
        count=20,
        weakest={"confidence": 0.7, "clip_path": "/x.mp4", "metadata_path": "/x.json", "title": "strong", "id": 1},
    )
    sup = _supervisor_with(db, max_per_day=20)
    assert sup._daily_cap_eviction_target(_segment(), _result(0.5)) is False


def test_evict_moment_removes_files(tmp_path):
    clip = tmp_path / "c.mp4"
    meta = tmp_path / "c.json"
    clip.write_bytes(b"x")
    meta.write_bytes(b"y")
    db = _FakeCapDB(0, None)
    sup = _supervisor_with(db)
    sup._evict_moment(
        {"clip_path": str(clip), "metadata_path": str(meta), "title": "t", "confidence": 0.3}
    )
    assert not clip.exists() and not meta.exists()


def test_analyze_segment_passes_actual_frame_offsets(tmp_path, monkeypatch):
    settings = replace(load_settings("/nonexistent.env"), analysis_image_mode="frames")
    database = Database(tmp_path / "test.sqlite3")
    database.migrate()
    supervisor = Supervisor(settings, database)
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"jpeg")
    captured: dict[str, list[float]] = {}

    async def fake_sample_frames_with_offsets(settings, video_path, output_dir, *, duration_seconds, sample_count=None):
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
