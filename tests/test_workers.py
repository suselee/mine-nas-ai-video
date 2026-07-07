from datetime import datetime, timedelta

from nas_video_summarizer.llm import AnalysisResult
from nas_video_summarizer.workers import _append_daily_summary


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

