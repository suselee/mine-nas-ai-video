from nas_video_summarizer.llm import _extract_json


def test_extract_json_from_plain_json():
    data = _extract_json('{"keep": true, "confidence": 0.9}')

    assert data["keep"] is True
    assert data["confidence"] == 0.9


def test_extract_json_from_wrapped_text():
    data = _extract_json('Here is the result:\n{"keep": false, "tags": ["quiet"]}\nDone.')

    assert data["keep"] is False
    assert data["tags"] == ["quiet"]

