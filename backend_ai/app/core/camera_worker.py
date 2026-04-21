"""
camera_worker.py - Per-camera RTSP reader thread
Chạy trong thread riêng, đọc stream và push frame vào batch queue
"""

import cv2
import time
import threading
import logging
import numpy as np
from typing import Dict, Optional, Callable

logger = logging.getLogger(__name__)


class CameraWorker:
    """
    Thread-based RTSP stream reader:
    - Liên tục đọc frame từ RTSP
    - Push frame vào BatchQueue
    - Auto-reconnect trên error
    - Backpressure: drop old frame nếu không kịp xử lý
    """

    def __init__(
        self,
        camera_id: int,
        rtsp_url: str,
        batch_queue,
        target_input_fps: float = 0.0,
        read_timeout_s: float = 5.0,
        max_reconnect_attempts: int = 3,
        reconnect_delay_s: float = 2.0,
    ):
        """
        Args:
            camera_id: int (0, 1, 2, ...)
            rtsp_url: "rtsp://..."
            batch_queue: BatchQueue instance
            target_input_fps: giới hạn FPS đẩy vào AI (0 = không giới hạn)
            read_timeout_s: timeout cho mỗi frame read
            max_reconnect_attempts: max retry trước khi fail
            reconnect_delay_s: wait time giữa reconnect attempts
        """
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.batch_queue = batch_queue
        self.target_input_fps = target_input_fps
        self._min_emit_interval = (1.0 / target_input_fps) if target_input_fps and target_input_fps > 0 else 0.0
        self._last_emit_ts = 0.0

        self.read_timeout = read_timeout_s
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay_s

        self._cap = None
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        
        # Keep track of latest frame for rendering
        self._latest_frame = None
        self._frame_lock = threading.Lock()

        self.frame_count = 0
        self.drop_count = 0
        self.error_count = 0
        self._frame_id = 0

    # ──────────────────────────────────────────────────────────────────────────
    def _connect(self) -> bool:
        """Kết nối RTSP stream"""
        for attempt in range(self.max_reconnect_attempts):
            try:
                self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Single frame buffer
                self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self.read_timeout * 1000))

                # Test read
                ret, frame = self._cap.read()
                if ret:
                    logger.info(f"[Cam {self.camera_id}] Kết nối RTSP thành công: {self.rtsp_url}")
                    return True
                else:
                    logger.warning(f"[Cam {self.camera_id}] Read test failed, attempt {attempt + 1}/{self.max_reconnect_attempts}")
                    self._cap.release()
                    self._cap = None

            except Exception as e:
                logger.error(f"[Cam {self.camera_id}] Connect error: {e}, attempt {attempt + 1}/{self.max_reconnect_attempts}")

            if attempt < self.max_reconnect_attempts - 1:
                time.sleep(self.reconnect_delay)

        logger.error(f"[Cam {self.camera_id}] Không thể kết nối sau {self.max_reconnect_attempts} lần")
        return False

    # ──────────────────────────────────────────────────────────────────────────
    def get_latest_frame(self):
        """Get latest frame read from RTSP"""
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    # ──────────────────────────────────────────────────────────────────────────
    def _worker_loop(self):
        """Main loop: liên tục đọc frame và push vào batch_queue"""
        reconnect_attempts = 0

        while not self._stop_event.is_set():
            # Connect if not connected
            if self._cap is None:
                if self._connect():
                    reconnect_attempts = 0
                else:
                    reconnect_attempts += 1
                    time.sleep(self.reconnect_delay)
                    continue

            # Read frame
            try:
                ret, frame = self._cap.read()

                if not ret or frame is None:
                    logger.warning(f"[Cam {self.camera_id}] Read failed, reconnecting...")
                    self._cap.release()
                    self._cap = None
                    self.error_count += 1
                    continue

                # Keep latest frame for rendering.
                with self._frame_lock:
                    self._latest_frame = frame

                now = time.time()
                if self._min_emit_interval > 0 and (now - self._last_emit_ts) < self._min_emit_interval:
                    self.drop_count += 1
                    continue

                frame_data = {
                    "camera_id": self.camera_id,
                    "frame": frame,
                    "timestamp": now,
                    "frame_id": self._frame_id,
                }
                self.batch_queue.put(frame_data)
                self._last_emit_ts = now
                self.frame_count += 1
                self._frame_id += 1

            except Exception as e:
                logger.error(f"[Cam {self.camera_id}] Read exception: {e}")
                self.error_count += 1
                if self._cap:
                    self._cap.release()
                    self._cap = None
                time.sleep(0.1)

    # ──────────────────────────────────────────────────────────────────────────
    def start(self):
        """Bắt đầu reader thread"""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=False)
            self._worker_thread.start()
            logger.info(f"[Cam {self.camera_id}] Worker thread started")

    # ──────────────────────────────────────────────────────────────────────────
    def stop(self):
        """Dừng reader thread và cleanup"""
        self._stop_event.set()

        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

        if self._cap:
            self._cap.release()
            self._cap = None

        logger.info(
            f"[Cam {self.camera_id}] Stopped - "
            f"frames={self.frame_count}, drops={self.drop_count}, errors={self.error_count}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    def is_alive(self) -> bool:
        """Check if worker thread chạy"""
        return self._worker_thread and self._worker_thread.is_alive()

    def get_stats(self) -> Dict:
        """Lấy thống kê"""
        return {
            "camera_id": self.camera_id,
            "frames_read": self.frame_count,
            "frames_dropped": self.drop_count,
            "errors": self.error_count,
            "is_connected": self._cap is not None,
        }
