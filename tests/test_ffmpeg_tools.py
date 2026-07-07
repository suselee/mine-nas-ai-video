from nas_video_summarizer.ffmpeg_tools import contact_sheet_layout, sample_offsets


def test_sample_offsets_are_evenly_distributed():
    offsets = sample_offsets(duration_seconds=120, frame_count=4, minimum_spacing_seconds=30)

    assert offsets == [24.0, 48.0, 72.0, 96.0]


def test_sample_offsets_respect_minimum_spacing_limit():
    offsets = sample_offsets(duration_seconds=60, frame_count=8, minimum_spacing_seconds=30)

    assert offsets == [15.0, 30.0, 45.0]


def test_contact_sheet_layout_caps_columns_to_frame_count():
    assert contact_sheet_layout(frame_count=4, preferred_columns=2) == (2, 2)
    assert contact_sheet_layout(frame_count=1, preferred_columns=2) == (1, 1)

