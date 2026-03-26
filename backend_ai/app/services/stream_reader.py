"""
stream_reader.py - Đọc luồng video từ RTSP / webcam / file
Output: frame, frame_id, timestamp, camera_id
"""

import cv2
import logging
from app.core.exceptions import StreamOpenError

logger = logging.getLogger(__name__)


class StreamReader:
    """
    Hỗ trợ:
    - RTSP:   "rtsp://user:pass@ip:port/stream"
    - Webcam: "0" hoặc số nguyên
    - File:   đường dẫn .mp4 / .avi
    """

    def __init__(self, camera_id: str, source: str):
        self.camera_id = camera_id
        self.source    = int(source) if source.isdigit() else source
        self._cap: cv2.VideoCapture | None = None

    # ──────────────────────────────────────────────────────────────────────────
    def _open(self):
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise StreamOpenError(
                f"[StreamReader] Không mở được nguồn: {self.source}"
            )
        logger.info(f"[StreamReader] Đã kết nối camera {self.camera_id} → {self.source}")

    # ──────────────────────────────────────────────────────────────────────────
    def read(self):
        """Generator trả về từng frame (numpy array BGR)."""
        self._open()
        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    logger.warning("[StreamReader] Hết frame hoặc mất kết nối.")
                    break
                yield frame
        finally:
            self._cap.release()
            logger.info(f"[StreamReader] Đã đóng camera {self.camera_id}")

    # ──────────────────────────────────────────────────────────────────────────
    def get_fps(self) -> float:
        if self._cap and self._cap.isOpened():
            return self._cap.get(cv2.CAP_PROP_FPS)
        return 25.0
