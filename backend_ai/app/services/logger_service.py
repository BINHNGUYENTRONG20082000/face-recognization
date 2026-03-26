"""
logger_service.py - Ghi kết quả ra JSONL và CSV
"""

import csv
import json
import logging
import os
from datetime import datetime

from app.config import LOGS_DIR

logger = logging.getLogger(__name__)


class LoggerService:
    def __init__(self, camera_id: str):
        os.makedirs(LOGS_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._jsonl_path = os.path.join(LOGS_DIR, f"{camera_id}_{ts}.jsonl")
        self._csv_path   = os.path.join(LOGS_DIR, f"{camera_id}_{ts}.csv")

        self._jsonl_f = open(self._jsonl_path, "w", encoding="utf-8")
        self._csv_f   = open(self._csv_path, "w", newline="", encoding="utf-8")

        self._csv_writer = csv.DictWriter(
            self._csv_f,
            fieldnames=[
                "frame_id", "timestamp", "camera_id",
                "track_id", "name", "score", "status", "source",
                "person_bbox", "face_bbox",
            ],
        )
        self._csv_writer.writeheader()
        logger.info(f"[Logger] {self._jsonl_path} | {self._csv_path}")

    # ──────────────────────────────────────────────────────────────────────────
    def write(self, frame_result: dict):
        # JSONL
        self._jsonl_f.write(json.dumps(frame_result, ensure_ascii=False) + "\n")

        # CSV — mỗi detection một dòng
        for det in frame_result.get("detections", []):
            self._csv_writer.writerow({
                "frame_id":   frame_result["frame_id"],
                "timestamp":  frame_result["timestamp"],
                "camera_id":  frame_result["camera_id"],
                "track_id":   det.get("track_id"),
                "name":       det.get("name"),
                "score":      det.get("score"),
                "status":     det.get("status"),
                "source":     det.get("source"),
                "person_bbox": det.get("person_bbox"),
                "face_bbox":  det.get("face_bbox"),
            })

    # ──────────────────────────────────────────────────────────────────────────
    def close(self):
        self._jsonl_f.close()
        self._csv_f.close()
        logger.info("[Logger] Đã đóng file log.")
