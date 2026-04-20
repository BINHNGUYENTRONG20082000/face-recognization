"""
person_detector.py - Detect người trong frame bằng YOLO26n
Output: list[{bbox, score}]
"""

import logging
import sys
from pathlib import Path

# Thêm backend_ai vào sys.path để có thể import app khi chạy trực tiếp
if __name__ == "__main__":
    backend_ai_dir = Path(__file__).resolve().parent.parent.parent
    if str(backend_ai_dir) not in sys.path:
        sys.path.insert(0, str(backend_ai_dir))

import numpy as np
from ultralytics import YOLO

from app.config import PERSON_MODEL_PATH, PERSON_CONF_THRESHOLD
from app.core.exceptions import ModelNotLoadedError

logger = logging.getLogger(__name__)

PERSON_CLASS_ID = 0   # COCO class 0 = person


class PersonDetector:
    def __init__(self):
        try:
            self._model = YOLO(PERSON_MODEL_PATH)
            logger.info(f"[PersonDetector] Đã load model: {PERSON_MODEL_PATH}")
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
def _run_demo() -> None:
    import cv2
    import os
    person_detector = PersonDetector()
    image_path = r"E:\face recognition\data test\photo_2026-04-09_11-09-00.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh demo: {image_path}")

    persons = person_detector.detect(image)
    preview = image.copy()

    for person in persons:
        x1, y1, x2, y2 = [int(value) for value in person["bbox"]]
        score = person["score"]
        label = f"person {score:.2f}"
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(preview, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Hiển thị ảnh kết quả hoặc lưu lại
    cv2.imshow("Person Detection Demo", preview)
    cv2.waitKey(0)
    # output_path = OUTPUT_DIR / "person_detection_demo.jpg"
    # cv2.imwrite(str(output_path), preview)   

if __name__ == "__main__":
    _run_demo()