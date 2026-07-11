from __future__ import annotations

import base64
from pathlib import Path
from urllib import request


_MODEL_DIR = Path(__file__).resolve().parent / "_person_filter_models"

# YOLO-family models. Two ONNX export layouts are supported and detected
# automatically by the decoder:
#   * raw  (one-to-many head):  output (1, nc+4, N)  -> needs NMS
#   * e2e  (one-to-one head):   output (1, 300, 6)   -> already decoded, no NMS
# v8/v11/v12 use the raw layout; YOLO26 uses e2e by default (or raw with
# end2end=False). Weights are downloaded on first use. A `PERSON_FILTER_MODEL_URL`
# env override may point at any compatible ONNX export.
_YOLO_NETS = {
    "yolov11n": {
        "filename": "yolov11n.onnx",
        "url": (
            "https://huggingface.co/unity/inference-engine-yolo/"
            "resolve/main/models/yolo11n.onnx"
        ),
        "input_size": 640,
    },
    "yolov8n": {
        "filename": "yolov8n.onnx",
        "url": (
            "https://github.com/ultralytics/assets/releases/download/"
            "v8.4.0/yolov8n.onnx"
        ),
        "input_size": 640,
    },
    "yolov26n": {
        "filename": "yolov26n.onnx",
        "url": (
            "https://huggingface.co/zwh20081/yolo26-onnx/"
            "resolve/main/yolo26n.onnx"
        ),
        "input_size": 640,
    },
}

# MobileNet-SSD (OpenCV Caffe sample, VOC 20 classes, person = 15)
_MOBILENET_PROTOTXT_URL = (
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/"
    "master/deploy.prototxt"
)
_MOBILENET_CAFFEMODEL_URL = (
    "https://github.com/chuanqi305/MobileNet-SSD/raw/"
    "master/mobilenet_iter_73000.caffemodel"
)
_MOBILENET_PERSON_CLASS_ID = 15

_YOLO_PERSON_CLASS_ID = 0
_YOLO_INPUT_SIZE = 640
_DEFAULT_MODEL = "yolov11n"


def _ensure_models(
    model_key: str, model_url: str = "", model_dir: Path = _MODEL_DIR
) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)

    if model_key == "mobilenet_ssd":
        prototxt = model_dir / "deploy.prototxt"
        caffemodel = model_dir / "mobilenet_iter_73000.caffemodel"
        if not prototxt.exists():
            _download(_MOBILENET_PROTOTXT_URL, prototxt)
        if not caffemodel.exists():
            _download(_MOBILENET_CAFFEMODEL_URL, caffemodel)
        return caffemodel

    spec = _YOLO_NETS[model_key]
    onnx = model_dir / spec["filename"]
    if not onnx.exists():
        _download(model_url or spec["url"], onnx)
    return onnx


def _download(url: str, dest: Path) -> None:
    print(f"Downloading {dest.name} ...")
    temporary = dest.with_suffix(f"{dest.suffix}.part")
    try:
        with request.urlopen(url) as resp, temporary.open("wb") as output:
            while chunk := resp.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(dest)
    finally:
        temporary.unlink(missing_ok=True)


class PersonFilter:
    """Local person detection, runs inline on the NAS (no HTTP server).

    Both backends run via OpenCV DNN, so the **only** runtime dependency is
    opencv — no onnxruntime / torch needed.

    Backends (``PERSON_FILTER_BACKEND``):
      * ``yolov11n``    (default) — YOLOv11n ONNX, fastest + most accurate on
                       CPU. Output format is compatible with v8/v12 too.
      * ``yolov8n``     — YOLOv8n ONNX, slightly older but well proven.
      * ``mobilenet_ssd`` — the older OpenCV Caffe sample model, lower
                       accuracy / more false positives.

    Returns a dict with ``person_score`` (0..1 max person confidence seen),
    plus ``face_score`` / ``face_bbox_area`` placeholders kept for interface
    stability (the YOLO models have no dedicated face class).
    """

    def __init__(
        self,
        threshold: float = 0.2,
        backend: str = _DEFAULT_MODEL,
        model_url: str = "",
        model_dir: Path | None = None,
    ):
        if backend not in _YOLO_NETS and backend != "mobilenet_ssd":
            backend = _DEFAULT_MODEL
        self._backend = backend
        self._threshold = threshold
        self._model_url = model_url
        self._model_dir = model_dir or _MODEL_DIR
        self._is_yolo = backend != "mobilenet_ssd"
        self._input_size = _YOLO_NETS.get(backend, {}).get("input_size", _YOLO_INPUT_SIZE)
        self._net = None
        self._cv2 = None
        self._np = None
        self._model_path = None

    def prepare(self) -> Path:
        """Download and load the configured model, returning its local path."""
        self._init_net()
        assert self._model_path is not None
        return self._model_path

    def _init_net(self):
        if self._net is not None:
            return
        import cv2
        import numpy as np

        self._cv2 = cv2
        self._np = np

        model_path = _ensure_models(
            self._backend, self._model_url, self._model_dir
        )
        self._model_path = model_path
        if self._is_yolo:
            self._net = cv2.dnn.readNetFromONNX(str(model_path))
        else:
            prototxt = self._model_dir / "deploy.prototxt"
            self._net = cv2.dnn.readNetFromCaffe(str(prototxt), str(model_path))
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self, image_b64: str) -> dict[str, float]:
        self._init_net()

        img_bytes = base64.b64decode(image_b64)
        arr = self._np.frombuffer(img_bytes, dtype=self._np.uint8)
        img = self._cv2.imdecode(arr, self._cv2.IMREAD_COLOR)
        if img is None:
            return {
                "person_score": 0.0,
                "face_score": 0.0,
                "face_bbox_area": 0.0,
            }

        if self._is_yolo:
            return self._detect_yolo(img)
        return self._detect_mobilenet(img)

    def _detect_mobilenet(self, img) -> dict[str, float]:
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
            if class_id == _MOBILENET_PERSON_CLASS_ID:
                person_score = max(person_score, confidence)

        return {
            "person_score": person_score,
            "face_score": 0.0,
            "face_bbox_area": 0.0,
        }

    def _detect_yolo(self, img) -> dict[str, float]:
        size = self._input_size
        blob = self._cv2.dnn.blobFromImage(
            img,
            1 / 255.0,
            (size, size),
            swapRB=True,
            crop=False,
        )
        self._net.setInput(blob)
        outputs = self._net.forward()
        preds = outputs[0]

        # e2e (one-to-one head) export: (N, 300, 6) -> [x1,y1,x2,y2,score,cls]
        if preds.shape[-1] == 6:
            return self._decode_yolo_e2e(preds, size)
        # raw (one-to-many head) export: (1, nc+4, N) -> transpose to (N, nc+4)
        return self._decode_yolo_raw(preds.T, size)

    def _decode_yolo_raw(self, predictions, size: int) -> dict[str, float]:
        boxes = []
        scores = []
        for pred in predictions:
            class_scores = pred[4:]
            class_id = int(self._np.argmax(class_scores))
            confidence = float(class_scores[class_id])
            if class_id != _YOLO_PERSON_CLASS_ID or confidence < self._threshold:
                continue
            x, y, bw, bh = pred[0:4]
            # bbox is center_x, center_y, width, height in input-pixel space
            x1 = x - bw / 2
            y1 = y - bh / 2
            boxes.append([float(x1), float(y1), float(bw), float(bh)])
            scores.append(confidence)

        person_score = 0.0
        max_area = 0.0
        if boxes:
            keep = self._nms(boxes, scores, iou_threshold=0.45)
            for i in keep:
                person_score = max(person_score, scores[i])
                area = boxes[i][2] * boxes[i][3]
                max_area = max(max_area, area)

        return self._pack(size, person_score, max_area)

    def _decode_yolo_e2e(self, preds, size: int) -> dict[str, float]:
        # Already-decoded detections: rows of [x1, y1, x2, y2, score, cls].
        person_score = 0.0
        max_area = 0.0
        for row in preds:
            score = float(row[4])
            class_id = int(row[5])
            if class_id != _YOLO_PERSON_CLASS_ID or score < self._threshold:
                continue
            x1, y1, x2, y2 = (float(v) for v in row[0:4])
            w = x2 - x1
            h = y2 - y1
            person_score = max(person_score, score)
            max_area = max(max_area, w * h)

        return self._pack(size, person_score, max_area)

    @staticmethod
    def _pack(size: int, person_score: float, max_area: float) -> dict[str, float]:
        # Normalize bbox area by input resolution for a rough 0..1 occupant
        # signal. e2e coordinates are in input-pixel space when max > 1.
        input_area = size * size
        if input_area <= 0:
            norm_area = 0.0
        elif max_area > 1.0:
            norm_area = max_area / input_area
        else:
            norm_area = max_area
        return {
            "person_score": person_score,
            "face_score": 0.0,
            "face_bbox_area": min(1.0, norm_area),
        }

    def _nms(self, boxes, scores, iou_threshold: float) -> list[int]:
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        keep: list[int] = []
        used = [False] * len(boxes)
        for i in order:
            if used[i]:
                continue
            keep.append(i)
            for j in order:
                if j == i or used[j]:
                    continue
                if self._iou(boxes[i], boxes[j]) > iou_threshold:
                    used[j] = True
        return keep

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        if union <= 0:
            return 0.0
        return inter / union
