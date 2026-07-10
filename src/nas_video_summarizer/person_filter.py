from __future__ import annotations

import base64
from pathlib import Path
from urllib import request


_MODEL_DIR = Path(__file__).resolve().parent / "_person_filter_models"
_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/"
    "master/deploy.prototxt"
)
_CAFFEMODEL_URL = (
    "https://github.com/chuanqi305/MobileNet-SSD/raw/"
    "master/mobilenet_iter_73000.caffemodel"
)
_PERSON_CLASS_ID = 15


def _ensure_models() -> tuple[Path, Path]:
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    prototxt = _MODEL_DIR / "deploy.prototxt"
    caffemodel = _MODEL_DIR / "mobilenet_iter_73000.caffemodel"

    if not prototxt.exists():
        _download(_PROTOTXT_URL, prototxt)
    if not caffemodel.exists():
        _download(_CAFFEMODEL_URL, caffemodel)

    return prototxt, caffemodel


def _download(url: str, dest: Path) -> None:
    print(f"Downloading {dest.name} ...")
    with request.urlopen(url) as resp:
        dest.write_bytes(resp.read())


class PersonFilter:
    """Person detection using OpenCV DNN + MobileNet-SSD.

    Single dependency (opencv-python-headless), pre-built ARM64 wheel,
    no compilation needed.  ~400ms/frame on ARM Cortex-A53.
    """

    def __init__(self, threshold: float = 0.2):
        self._threshold = threshold
        self._net = None
        self._cv2 = None
        self._np = None

    def _init_net(self):
        if self._net is not None:
            return
        import cv2
        import numpy as np

        self._cv2 = cv2
        self._np = np
        prototxt, caffemodel = _ensure_models()
        self._net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))

    def detect(self, image_b64: str) -> dict[str, float]:
        self._init_net()

        img_bytes = base64.b64decode(image_b64)
        arr = self._np.frombuffer(img_bytes, dtype=self._np.uint8)
        img = self._cv2.imdecode(arr, self._cv2.IMREAD_COLOR)

        blob = self._cv2.dnn.blobFromImage(
            img, 0.007843, (300, 300), 127.5, swapRB=False
        )
        self._net.setInput(blob)
        detections = self._net.forward()

        person_score = 0.0
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self._threshold:
                continue
            class_id = int(detections[0, 0, i, 1])
            if class_id == _PERSON_CLASS_ID:
                person_score = max(person_score, confidence)

        return {
            "person_score": person_score,
            "face_score": 0.0,
            "face_bbox_area": 0.0,
        }
