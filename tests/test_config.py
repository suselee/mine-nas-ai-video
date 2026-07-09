from nas_video_summarizer.config import load_env_file, load_settings, with_rtsp_credentials


def test_load_settings_from_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "APP_PORT=9001",
                "CAMERA_NAME=test-camera",
                "RTSP_USERNAME=alice",
                "RTSP_PASSWORD=s3cret!",
                "RTSP_LOW_URL=rtsp://example/low",
                "RTSP_HIGH_URL=rtsp://example/high",
                "NEXTCLOUD_OUTPUT_DIR=./out",
                "ANALYSIS_IMAGE_MODE=contact_sheet",
                "ANALYSIS_FRAME_WIDTH=320",
                "CONTACT_SHEET_COLUMNS=2",
                "MAX_MOMENT_SECONDS=180",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(env_file)

    assert settings.app_port == 9001
    assert settings.camera_name == "test-camera"
    assert settings.rtsp_username == "alice"
    assert settings.rtsp_password == "s3cret!"
    assert settings.rtsp_low_url == "rtsp://example/low"
    assert settings.rtsp_high_url == "rtsp://example/high"
    assert settings.rtsp_low_url_for_ffmpeg == "rtsp://alice:s3cret%21@example/low"
    assert settings.output_dir.name == "out"
    assert settings.analysis_image_mode == "contact_sheet"
    assert settings.analysis_frame_width == 320
    assert settings.contact_sheet_columns == 2
    assert settings.max_moment_seconds == 180
    assert settings.analysis_stream_role == "low"
    assert settings.analysis_window_start == ""
    assert settings.analysis_window_end == ""
    assert settings.ffmpeg_hwaccel == ""
    assert settings.analysis_cooldown_seconds == 5


def test_with_rtsp_credentials_does_not_override_embedded_credentials():
    url = with_rtsp_credentials(
        "rtsp://camera-user:camera-password@example/stream",
        "other-user",
        "other-password",
    )

    assert url == "rtsp://camera-user:camera-password@example/stream"


def test_with_rtsp_credentials_handles_blank_credentials():
    url = with_rtsp_credentials("rtsp://example/stream", "", "")

    assert url == "rtsp://example/stream"


def test_load_env_file_handles_quoted_values(tmp_path, monkeypatch):
    monkeypatch.delenv("RTSP_PASSWORD", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text('RTSP_PASSWORD="abc#123&xyz"\n', encoding="utf-8")

    load_env_file(env_file)

    import os

    assert os.environ["RTSP_PASSWORD"] == "abc#123&xyz"
