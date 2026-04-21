"""
batch_queue.py - Unified batch queue cho N cameras
"""

import time
import threading
import logging
from collections import deque
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class BatchQueue:
    """
    Thread-safe batch queue: collect frames từ N cameras,
    trả về batch khi đủ batch_size hoặc timeout
    """

    def __init__(self, batch_size: int = 8, timeout_ms: float = 20):
        """
        Args:
            batch_size: số frames trong 1 batch
            timeout_ms: max wait time trước khi return partial batch (ms)
        """
        self.batch_size = batch_size
        self.timeout = timeout_ms / 1000.0  # convert to seconds

        # Latest-only semantics per camera: keep only newest frame for each cam.
        self._latest_by_camera: Dict[int, Dict] = {}
        self._ready_camera_ids = deque()
        self._ready_camera_set = set()

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        
        self._stop_event = threading.Event()

    # ──────────────────────────────────────────────────────────────────────────
    def put(self, frame_data: Dict):
        """
        Thêm frame vào queue: {camera_id, frame, timestamp, ...}
        Non-blocking, keep queue near real-time.
        Chỉ giữ frame mới nhất cho mỗi camera để tránh tích lũy độ trễ.
        """
        camera_id = int(frame_data.get("camera_id", -1))
        if camera_id < 0:
            return

        with self._cv:
            self._latest_by_camera[camera_id] = frame_data
            if camera_id not in self._ready_camera_set:
                self._ready_camera_ids.append(camera_id)
                self._ready_camera_set.add(camera_id)
            self._cv.notify()

    # ──────────────────────────────────────────────────────────────────────────
    def get_batch(self, timeout_override: Optional[float] = None) -> Optional[List]:
        """
        Collect frames vào batch.
        Return when đủ batch_size hoặc timeout.
        """
        timeout = timeout_override or self.timeout
        batch = []
        deadline = time.time() + timeout

        with self._cv:
            while not self._ready_camera_ids:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._cv.wait(timeout=remaining)

            while len(batch) < self.batch_size:
                remaining = deadline - time.time()
                if remaining <= 0 or not self._ready_camera_ids:
                    break

                camera_id = self._ready_camera_ids.popleft()
                self._ready_camera_set.discard(camera_id)
                frame_data = self._latest_by_camera.pop(camera_id, None)
                if frame_data is not None:
                    batch.append(frame_data)

        return batch if batch else None

    # ──────────────────────────────────────────────────────────────────────────
    def qsize(self) -> int:
        """Current queue size"""
        with self._lock:
            return len(self._latest_by_camera)

    def clear(self):
        """Clear all pending frames"""
        with self._cv:
            self._latest_by_camera.clear()
            self._ready_camera_ids.clear()
            self._ready_camera_set.clear()

    def stop(self):
        """Stop batch queue"""
        self._stop_event.set()
        self.clear()
