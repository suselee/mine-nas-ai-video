from nas_video_summarizer.llm import AnalysisResult, _extract_json


def _result(keep: bool, confidence: float) -> AnalysisResult:
    return AnalysisResult(
        keep=keep,
        title="t",
        summary="s",
        tags=[],
        confidence=confidence,
        start_offset_seconds=0,
        end_offset_seconds=1,
        raw={},
    )


def test_extract_json_from_plain_json():
    data = _extract_json('{"keep": true, "confidence": 0.9}')

    assert data["keep"] is True
    assert data["confidence"] == 0.9


def test_extract_json_from_wrapped_text():
    data = _extract_json('Here is the result:\n{"keep": false, "tags": ["quiet"]}\nDone.')

    assert data["keep"] is False
    assert data["tags"] == ["quiet"]


def test_should_save_requires_both_keep_and_confidence():
    # A weak "keep=true" guess must NOT save when confidence is low -
    # this is the regression guard for the static/empty-room problem:
    # 2B models on low-res contact sheets often emit keep=true for
    # rooms that contain toys but no visible child.
    assert not _result(keep=True, confidence=0.3).should_save(0.55)
    assert _result(keep=True, confidence=0.6).should_save(0.55)
    # Even very high confidence does not override an explicit keep=false.
    assert not _result(keep=False, confidence=0.99).should_save(0.55)

