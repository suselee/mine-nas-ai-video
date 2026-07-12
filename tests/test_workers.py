import asyncio
from dataclasses import replace
from datetime import datetime, timedelta

from nas_video_summarizer.config import load_settings
from nas_video_summarizer.database import Database
from nas_video_summarizer.ffmpeg_tools import (
    ContactSheet,
    PersonFilterDecision,
    SampledFrame,
)
from nas_video_summarizer.llm import AnalysisResult, DaughterVerification
from nas_video_summarizer.workers import (
    Supervisor,
    _append_daily_summary,
    _in_analysis_window,
    _in_time_window,
    _in_record_window,
    _moment_period,
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


def test_running_recorder_stops_when_record_window_closes(tmp_path, monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"),
        record_window_start="07:00",
        record_window_end="21:00",
    )
    database = Database(tmp_path / "test.sqlite3")
    database.migrate()
    supervisor = Supervisor(settings, database)

    class FakeProcess:
        pid = 123
        returncode = None

        async def wait(self):
            return self.returncode

    process = FakeProcess()
    times = iter((_dt(20, 59), _dt(21, 0)))

    async def fake_spawn(*args, **kwargs):
        return process

    async def fake_stop(proc):
        proc.returncode = 0
        supervisor.stop_event.set()

    async def fake_sleep(seconds):
        return None

    monkeypatch.setattr("nas_video_summarizer.workers.ffmpeg_available", lambda settings: True)
    monkeypatch.setattr(
        "nas_video_summarizer.workers.build_recorder_command",
        lambda settings, role, url: ["ffmpeg"],
    )
    monkeypatch.setattr("nas_video_summarizer.workers._now", lambda: next(times))
    monkeypatch.setattr("nas_video_summarizer.workers._stop_process", fake_stop)
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_spawn)
    monkeypatch.setattr("asyncio.sleep", fake_sleep)

    asyncio.run(supervisor._recorder_loop("low", "rtsp://camera/low"))

    assert supervisor.state["recorders"]["low"]["status"] == "waiting_for_record_window"
    assert database.recent_events(limit=1)[0]["event_type"] == "recorder-window"


class _FakeCapDB:
    def __init__(self, count, weakest, period_count=None):
        self._count = count
        self._period_count = count if period_count is None else period_count
        self._weakest = weakest
        self.evicted = None

    def count_moments_on_day(self, day):
        return self._count

    def min_confidence_on_day(self, day):
        return float(self._weakest["confidence"]) if self._weakest else 1.0

    def weakest_moment_on_day(self, day):
        return self._weakest

    def count_moments_between(self, start_iso, end_iso):
        return self._period_count

    def weakest_moment_between(self, start_iso, end_iso):
        return self._weakest

    def delete_moment_by_clip(self, clip):
        self.evicted = clip

    def add_event(self, *args, **kwargs):
        pass


def _supervisor_with(db, max_per_day=20, max_per_period=0):
    settings = replace(
        load_settings("/nonexistent.env"),
        max_moments_per_day=max_per_day,
        max_moments_per_period=max_per_period,
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


def test_moment_period_boundaries():
    boundaries = "07:00,12:00,17:00,21:00"

    assert _moment_period(_dt(11, 59), boundaries)[0] == "morning"
    assert _moment_period(_dt(12, 0), boundaries)[0] == "afternoon"
    assert _moment_period(_dt(16, 59), boundaries)[0] == "afternoon"
    assert _moment_period(_dt(17, 0), boundaries)[0] == "evening"
    assert _moment_period(_dt(21, 0), boundaries) is None
    assert _moment_period(_dt(10, 0), "invalid") is None


def test_period_cap_under_limit_saves():
    db = _FakeCapDB(count=20, period_count=7, weakest=None)
    sup = _supervisor_with(db, max_per_day=24, max_per_period=8)

    assert sup._period_cap_eviction_target(
        {"started_at": "2026-07-10T10:00:00+08:00"}, _result(0.8)
    ) == ("morning", None)


def test_period_cap_replaces_weaker_moment_in_same_period():
    weakest = {
        "confidence": 0.7,
        "clip_path": "/x.mp4",
        "metadata_path": "/x.json",
        "title": "weak",
        "id": 1,
    }
    db = _FakeCapDB(count=20, period_count=8, weakest=weakest)
    sup = _supervisor_with(db, max_per_day=24, max_per_period=8)

    label, target = sup._period_cap_eviction_target(
        {"started_at": "2026-07-10T18:00:00+08:00"}, _result(0.85)
    )

    assert label == "evening"
    assert target == weakest


def test_period_cap_skips_equal_or_weaker_moment():
    weakest = {
        "confidence": 0.85,
        "clip_path": "/x.mp4",
        "metadata_path": "/x.json",
        "title": "strong",
        "id": 1,
    }
    db = _FakeCapDB(count=20, period_count=8, weakest=weakest)
    sup = _supervisor_with(db, max_per_day=24, max_per_period=8)

    assert sup._period_cap_eviction_target(
        {"started_at": "2026-07-10T13:00:00+08:00"}, _result(0.85)
    ) == ("afternoon", False)


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


def test_contact_sheet_uses_prefiltered_frames(tmp_path, monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"),
        analysis_image_mode="contact_sheet",
        person_filter_enabled=True,
    )
    database = Database(tmp_path / "test.sqlite3")
    database.migrate()
    supervisor = Supervisor(settings, database)
    frames = []
    for index, offset in enumerate((20.0, 80.0)):
        path = tmp_path / f"frame-{index}.jpg"
        path.write_bytes(b"jpeg")
        frames.append(SampledFrame(path=path, offset_seconds=offset))
    captured = {}

    async def fake_sample(*args, **kwargs):
        return frames

    async def fake_blank(settings, sampled_frames):
        return sampled_frames

    async def fake_filter(settings, sampled_frames):
        return PersonFilterDecision([sampled_frames[1]])

    async def fake_sheet(settings, sampled_frames, output_dir):
        captured["sheet_frames"] = sampled_frames
        path = output_dir / "contact_sheet.jpg"
        path.write_bytes(b"sheet")
        return ContactSheet(path=path, frame_offsets_seconds=[80.0])

    class FakeAnalyzer:
        async def analyze(self, **kwargs):
            captured["image_paths"] = kwargs["image_paths"]
            captured["offsets"] = kwargs["frame_offsets_seconds"]
            return _result(0.1)

    monkeypatch.setattr(
        "nas_video_summarizer.workers.sample_frames_with_offsets", fake_sample
    )
    monkeypatch.setattr(
        "nas_video_summarizer.workers.filter_out_blank_frames", fake_blank
    )
    monkeypatch.setattr(
        "nas_video_summarizer.workers.filter_frames_by_person_detection",
        fake_filter,
    )
    monkeypatch.setattr(
        "nas_video_summarizer.workers.build_contact_sheet_from_frames",
        fake_sheet,
    )

    asyncio.run(
        supervisor._analyze_segment(
            FakeAnalyzer(),
            {"path": str(tmp_path / "segment.mp4"), "duration_seconds": 120},
        )
    )

    assert captured["sheet_frames"] == [frames[1]]
    assert captured["offsets"] == [80.0]
    assert captured["image_paths"][0].name == "contact_sheet.jpg"
    assert supervisor.state["prefilter"]["status"] == "ready"
    assert supervisor.state["prefilter"]["input_frames"] == 2
    assert supervisor.state["prefilter"]["output_frames"] == 1


def test_adult_only_filter_records_distinct_event(tmp_path, monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"),
        analysis_image_mode="frames",
        person_filter_enabled=True,
    )
    database = Database(tmp_path / "test.sqlite3")
    database.migrate()
    supervisor = Supervisor(settings, database)
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"jpeg")

    async def fake_sample(*args, **kwargs):
        return [SampledFrame(path=frame_path, offset_seconds=30.0)]

    async def fake_blank(settings, sampled_frames):
        return sampled_frames

    async def fake_filter(settings, sampled_frames):
        return PersonFilterDecision([], "adult-only")

    monkeypatch.setattr(
        "nas_video_summarizer.workers.sample_frames_with_offsets", fake_sample
    )
    monkeypatch.setattr(
        "nas_video_summarizer.workers.filter_out_blank_frames", fake_blank
    )
    monkeypatch.setattr(
        "nas_video_summarizer.workers.filter_frames_by_person_detection",
        fake_filter,
    )

    try:
        asyncio.run(
            supervisor._analyze_segment(
                object(),
                {
                    "id": 1,
                    "path": str(tmp_path / "segment.mp4"),
                    "duration_seconds": 120,
                },
            )
        )
    except Exception as exc:
        assert exc.__class__.__name__ == "PersonFilterSkip"
    else:
        raise AssertionError("adult-only segment should skip analysis")

    events = database.recent_events(limit=1)
    assert events[0]["event_type"] == "adult-only-filter-skip"
    assert supervisor.state["prefilter"]["status"] == "adult-only"


def test_saved_clip_verification_uses_three_frames(tmp_path, monkeypatch):
    settings = load_settings("/nonexistent.env")
    database = Database(tmp_path / "test.sqlite3")
    database.migrate()
    supervisor = Supervisor(settings, database)
    offsets = []

    async def fake_extract(settings, clip_path, frame_path, offset):
        offsets.append(offset)
        frame_path.write_bytes(b"jpeg")

    class FakeAnalyzer:
        async def verify_daughter_visible(self, frame_paths):
            assert len(frame_paths) == 3
            return DaughterVerification(
                visible=True,
                confidence=0.9,
                description="A young girl is visible.",
                repaired=False,
                raw_text="{}",
            )

    monkeypatch.setattr("nas_video_summarizer.workers._extract_frame", fake_extract)

    verified = asyncio.run(
        supervisor._verify_saved_clip(
            FakeAnalyzer(), tmp_path / "clip.mp4", duration_seconds=20
        )
    )

    assert verified is True
    assert offsets == [4.0, 10.0, 16.0]


def test_low_confidence_negative_without_local_child_rejects_clip(
    tmp_path, monkeypatch
):
    settings = load_settings("/nonexistent.env")
    database = Database(tmp_path / "test.sqlite3")
    database.migrate()
    supervisor = Supervisor(settings, database)

    async def fake_extract(settings, clip_path, frame_path, offset):
        frame_path.write_bytes(b"jpeg")

    class FakeAnalyzer:
        async def verify_daughter_visible(self, frame_paths):
            return DaughterVerification(
                visible=False,
                confidence=0.1,
                description="A blurry person, possibly interacting with a child.",
                repaired=False,
                raw_text="{}",
            )

    monkeypatch.setattr("nas_video_summarizer.workers._extract_frame", fake_extract)

    verified = asyncio.run(
        supervisor._verify_saved_clip(
            FakeAnalyzer(), tmp_path / "clip.mp4", duration_seconds=20
        )
    )

    assert verified is False
    assert database.recent_events(limit=1)[0]["event_type"] == "verify-detail"


def test_local_child_evidence_can_keep_uncertain_verification(tmp_path, monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"), person_filter_enabled=True
    )
    database = Database(tmp_path / "test.sqlite3")
    database.migrate()
    supervisor = Supervisor(settings, database)

    async def fake_extract(settings, clip_path, frame_path, offset):
        frame_path.write_bytes(b"jpeg")

    async def fake_local_filter(settings, frames):
        return PersonFilterDecision(
            frames,
            max_child_score=0.82,
            child_confirmed=True,
        )

    class FakeAnalyzer:
        async def verify_daughter_visible(self, frame_paths):
            return DaughterVerification(
                visible=False,
                confidence=0.1,
                description="A blurry person, possibly interacting with a child.",
                repaired=False,
                raw_text="{}",
            )

    monkeypatch.setattr("nas_video_summarizer.workers._extract_frame", fake_extract)
    monkeypatch.setattr(
        "nas_video_summarizer.workers.filter_frames_by_person_detection",
        fake_local_filter,
    )

    verified = asyncio.run(
        supervisor._verify_saved_clip(
            FakeAnalyzer(), tmp_path / "clip.mp4", duration_seconds=20
        )
    )

    assert verified is True
    assert database.recent_events(limit=1)[0]["event_type"] == "verify-local-child-keep"


def test_candidate_verification_uses_high_resolution_frames(tmp_path, monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"), verification_frame_width=512
    )
    database = Database(tmp_path / "test.sqlite3")
    database.migrate()
    supervisor = Supervisor(settings, database)
    captured = []

    async def fake_extract(frame_settings, video_path, frame_path, offset):
        captured.append((frame_settings.analysis_frame_width, offset))
        frame_path.write_bytes(b"jpeg")

    class FakeAnalyzer:
        async def verify_daughter_visible(self, frame_paths):
            return DaughterVerification(True, 0.9, "daughter visible", False, "{}")

    monkeypatch.setattr("nas_video_summarizer.workers._extract_frame", fake_extract)

    verified = asyncio.run(
        supervisor._verify_candidate(
            FakeAnalyzer(), tmp_path / "segment.mp4", 20, 40
        )
    )

    assert verified is True
    assert captured == [(512, 20), (512, 30.0), (512, 40)]
