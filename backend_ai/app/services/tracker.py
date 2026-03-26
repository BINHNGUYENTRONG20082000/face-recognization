"""
tracker.py - Tracking person theo frame bằng Ultralytics ByteTrack / BoT-SORT
Output: list[{track_id, bbox, score, age}]
"""

import logging
import numpy as np
from ultralytics import YOLO

from app.config import YOLO_MODEL_PATH, TRACKER_TYPE
from app.core.exceptions import ModelNotLoadedError

logger = logging.getLogger(__name__)


class PersonTracker:
    """
    Wrap Ultralytics tracker để nhận detection từ PersonDetector
    và trả về track_id ổn định theo thời gian.
    """

    def __init__(self):
        try:
            self._model = YOLO(YOLO_MODEL_PATH)
            self._tracker_cfg = f"{TRACKER_TYPE}.yaml"
            logger.info(f"[Tracker] Sử dụng tracker: {TRACKER_TYPE}")
        except Exception as e:
            raise ModelNotLoadedError(f"Không khởi tạo được tracker: {e}") from e

    # ──────────────────────────────────────────────────────────────────────────
    def update(self, persons: list[dict], frame: np.ndarray) -> list[dict]:
        """
        Chạy tracking trực tiếp trên frame (YOLO track mode).
        Trả về list[{"track_id": int, "bbox": [x1,y1,x2,y2], "score": float}]
        """
        results = self._model.track(
            frame,
            persist=True,
            tracker=self._tracker_cfg,
            classes=[0],
            verbose=False,
        )
        tracks = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                if box.id is None:
                    continue
                track_id = int(box.id[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])
                tracks.append({
                    "track_id": track_id,
                    "bbox":     [x1, y1, x2, y2],
                    "score":    score,
                })
        return tracks
