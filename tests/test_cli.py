from nas_video_summarizer.cli import _redact_url


def test_redact_url_hides_rtsp_credentials():
    value = _redact_url("rtsp://alice:secret@example.local:554/live/low?profile=1")

    assert value == "rtsp://<credentials>@example.local:554/live/low?profile=1"
    assert "alice" not in value
    assert "secret" not in value


def test_redact_url_keeps_url_without_credentials():
    value = _redact_url("http://127.0.0.1:8080/v1")

    assert value == "http://127.0.0.1:8080/v1"

