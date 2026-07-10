from __future__ import annotations

import base64
from typing import Any


class PersonFilter:
    """MediaPipe Object Detection + Face Detection for person-presence check.

    Uses EfficientDet-Lite0 for person detection and BlazeFace-Short for
    face detection. Both run on CPU ~300-400ms/frame combined on ARM A53.
    """

    def __init__(self, object_threshold: float = 0.2, face_threshold: float = 0.3):
        import cv2
        import mediapipe as mp
        import numpy as np

        self._cv2 = cv2
        self._np = np

        self._object_detector = mp.solutions.object_detection.ObjectDetection(
            model_selection=0,
            min_detection_confidence=object_threshold,
        )
        self._face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=face_threshold,
        )

    def detect(self, image_b64: str) -> dict[str, float]:
        img_bytes = base64.b64decode(image_b64)
        arr = self._np.frombuffer(img_bytes, dtype=self._np.uint8)
        img = self._cv2.imdecode(arr, self._cv2.IMREAD_COLOR)
        rgb = self._cv2.cvtColor(img, self._cv2.COLOR_BGR2RGB)

        obj_result = self._object_detector.process(rgb)
        person_score = 0.0
        if obj_result.detections:
            for det in obj_result.detections:
                if det.label_id[0] == 0:
                    person_score = max(person_score, det.score[0])

        face_result = self._face_detector.process(rgb)
        face_score = 0.0
        face_bbox_area = 0.0
        if face_result.detections:
            for det in face_result.detections:
                face_score = max(face_score, det.score[0])
                bbox = det.location_data.relative_bounding_box
                face_bbox_area = max(face_bbox_area, bbox.width * bbox.height)

        return {
            "person_score": float(person_score),
            "face_score": float(face_score),
            "face_bbox_area": float(face_bbox_area),
        }
