from io import BytesIO

from nas_video_summarizer import person_filter as pf


def test_custom_model_dir_is_used(tmp_path, monkeypatch):
    model = tmp_path / "yolov11n.onnx"
    monkeypatch.setattr(pf, "_download", lambda url, dest: dest.write_bytes(b"model"))

    assert pf._ensure_models("yolov11n", model_dir=tmp_path) == model
    assert model.read_bytes() == b"model"


def test_download_replaces_partial_file_atomically(tmp_path, monkeypatch):
    destination = tmp_path / "model.onnx"
    partial = tmp_path / "model.onnx.part"
    partial.write_bytes(b"incomplete")

    monkeypatch.setattr(pf.request, "urlopen", lambda url: BytesIO(b"complete"))

    pf._download("https://example.invalid/model.onnx", destination)

    assert destination.read_bytes() == b"complete"
    assert not partial.exists()
