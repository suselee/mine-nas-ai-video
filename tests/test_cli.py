from dataclasses import replace

from nas_video_summarizer.cli import _person_filter_check, _redact_url
from nas_video_summarizer.config import load_settings


def test_redact_url_hides_rtsp_credentials():
    value = _redact_url("rtsp://alice:secret@example.local:554/live/low?profile=1")

    assert value == "rtsp://<credentials>@example.local:554/live/low?profile=1"
    assert "alice" not in value
    assert "secret" not in value


def test_redact_url_keeps_url_without_credentials():
    value = _redact_url("http://127.0.0.1:8080/v1")

    assert value == "http://127.0.0.1:8080/v1"


def test_person_filter_check_is_optional_when_disabled():
    check = _person_filter_check(load_settings("/nonexistent.env"))

    assert check.ok is True
    assert check.required is False
    assert check.detail == "disabled"


def test_person_filter_check_explains_freebsd_venv_fix(monkeypatch):
    settings = replace(
        load_settings("/nonexistent.env"), person_filter_enabled=True
    )

    def fail_prepare(self):
        raise ImportError("No module named 'cv2'")

    monkeypatch.setattr(
        "nas_video_summarizer.cli.PersonFilter.prepare", fail_prepare
    )

    check = _person_filter_check(settings)

    assert check.ok is False
    assert check.required is True
    assert "--system-site-packages" in check.detail
