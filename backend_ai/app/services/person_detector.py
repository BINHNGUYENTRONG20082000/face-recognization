"""
person_detector.py - Detect người trong frame bằng YOLO26n
Output: list[{bbox, score}]
"""

import logging
import numpy as np
from ultralytics import YOLO

from app.config import YOLO_MODEL_PATH, PERSON_CONF_THRESHOLD
from app.core.exceptions import ModelNotLoadedError

logger = logging.getLogger(__name__)

PERSON_CLASS_ID = 0   # COCO class 0 = person


class PersonDetector:
    def __init__(self):
        try:
            self._model = YOLO(YOLO_MODEL_PATH)
            logger.info(f"[PersonDetector] Đã load model: {YOLO_MODEL_PATH}")
        except Exception as e:
            raise ModelNotLoadedError(f"Không load được YOLO model: {e}") from e

    # ──────────────────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Trả về list[{"bbox": [x1,y1,x2,y2], "score": float}]
        """
        results = self._model.predict(
            frame,
            conf=PERSON_CONF_THRESHOLD,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )
        persons = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])
                persons.append({"bbox": [x1, y1, x2, y2], "score": score})
        return persons
