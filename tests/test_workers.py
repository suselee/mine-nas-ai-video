from datetime import datetime, timedelta

from nas_video_summarizer.llm import AnalysisResult
from nas_video_summarizer.workers import _append_daily_summary, _in_analysis_window


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

