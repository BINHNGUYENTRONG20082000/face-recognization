"""
camera_state_manager.py - Per-camera state management
Lấy results từ SharedAIService, maintain tracking state, generate overlay
"""

import cv2
import time
import threading
import logging
import numpy as np
from typing import Dict, Optional
from queue import Queue, Empty
import unicodedata

logger = logging.getLogger(__name__)


def _strip_vietnamese_accents(text: str) -> str:
    """Remove Vietnamese diacritics"""
    if not text:
        return text
    nfkd_form = unicodedata.normalize("NFD", text)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


class CameraStateManager:
    """
    Per-camera state management:
    - Subscribe to result queue từ SharedAIService
    - Maintain face cache per track
    - Generate overlay frame
    - Log statistics
    """

    def __init__(self, camera_id: int, result_queue: Queue):
        """
        Args:
            camera_id: int (0, 1, 2, ...)
            result_queue: Queue từ SharedAIService
        """
        self.camera_id = camera_id
        self.result_queue = result_queue

        # State
        self._current_frame = None
        self._current_overlay = None
        self._current_detections = {}
        self._current_faces = {}
        self._current_frame_id = -1
        self._analysis_delay_ms = 0.0
        self._state_lock = threading.Lock()

        # Face cache
        self._face_cache: Dict[int, Dict] = {}  # {track_id: {face, embedding, timestamp}}
        self._cache_expiry_frames = 300  # ~10s at 30fps

        # Worker thread
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        # Stats
        self.frame_count = 0
        self.detection_count = 0
        self.fps = 0
        self._last_fps_update = time.time()
        self._frames_since_fps_update = 0

    # ──────────────────────────────────────────────────────────────────────────
    def _worker_loop(self):
        """Poll result queue and update state"""
        logger.info(f"[StateManager {self.camera_id}] Worker started")

        while not self._stop_event.is_set():
            try:
                result = self.result_queue.get(timeout=0.1)

                with self._state_lock:
                    analyzed_frame = result.get("frame")
                    if analyzed_frame is not None:
                        self._current_frame = analyzed_frame.copy()

                    new_frame_id = int(result.get("frame_id", -1))
                    if new_frame_id >= 0:
                        self._current_frame_id = new_frame_id

                    # Measure analysis/display delay from frame capture to result publish.
                    input_ts = float(result.get("timestamp", time.time()))
                    self._analysis_delay_ms = max(0.0, (time.time() - input_ts) * 1000.0)

                    # Update detections
                    self._current_detections = {t["track_id"]: t for t in result["tracks"]}

                    # Update faces
                    faces_by_track = {}
                    for face in result["faces"]:
                        track_id = face["track_id"]
                        faces_by_track[track_id] = face

                        # Update cache
                        self._face_cache[track_id] = {
                            "face": face["face"],
                            "name": face["voted_name"],
                            "timestamp": result["timestamp"],
                        }

                    self._current_faces = faces_by_track

                    self.frame_count += 1
                    self.detection_count = len(self._current_detections)

                    # Update FPS
                    self._frames_since_fps_update += 1
                    now = time.time()
                    elapsed = now - self._last_fps_update
                    if elapsed >= 1.0:
                        self.fps = self._frames_since_fps_update / elapsed
                        self._frames_since_fps_update = 0
                        self._last_fps_update = now

            except Empty:
                continue
            except Exception as e:
                logger.error(f"[StateManager {self.camera_id}] Error: {e}")

        logger.info(f"[StateManager {self.camera_id}] Worker stopped")

    # ──────────────────────────────────────────────────────────────────────────
    def start(self):
        """Bắt đầu state manager"""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=False)
            self._worker_thread.start()
            logger.info(f"[StateManager {self.camera_id}] Started")

    # ──────────────────────────────────────────────────────────────────────────
    def stop(self):
        """Dừng state manager"""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
        logger.info(f"[StateManager {self.camera_id}] Stopped - {self.frame_count} frames processed")

    # ──────────────────────────────────────────────────────────────────────────
    def set_current_frame(self, frame: np.ndarray):
        """Set fallback frame (dùng khi stream_frame chưa có)"""
        with self._state_lock:
            self._current_frame = frame.copy() if frame is not None else None

    # ──────────────────────────────────────────────────────────────────────────
    def get_overlay_frame(self, stream_frame: Optional[np.ndarray] = None, render: bool = True) -> Optional[np.ndarray]:
        """Lấy frame với overlay detections + names.
        
        Args:
            stream_frame: Frame mới nhất từ live stream (ưu tiên dùng để hiển thị mượt).
                          Nếu None, fallback về frame đã phân tích trước đó.
        """
        with self._state_lock:
            base_frame = stream_frame if stream_frame is not None else self._current_frame

            if base_frame is None:
                # Return placeholder if no frame yet
                placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Waiting for frames...", (150, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                return placeholder

            frame = base_frame.copy()

            # Draw person detections + identity labels
            for track_id, track in self._current_detections.items():
                x1, y1, x2, y2 = [int(v) for v in track["bbox"]]
                score = track["score"]

                # Get name from cache
                name = "Unknown"
                if track_id in self._current_faces:
                    name = self._current_faces[track_id].get("voted_name", "Unknown")

                # Strip Vietnamese accents
                name_clean = _strip_vietnamese_accents(name)

                # Draw person box (green)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label
                label = f"ID:{track_id} {name_clean} ({score:.2f})"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

                # Draw face box if available (orange)
                if track_id in self._current_faces:
                    face_info = self._current_faces[track_id]
                    face_bbox = face_info.get("face_bbox")
                    if face_bbox and len(face_bbox) == 4:
                        fx1, fy1, fx2, fy2 = [int(v) for v in face_bbox]
                        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 165, 255), 2)

                        match_score = float(face_info.get("match_score", 0.0))
                        face_status = face_info.get("status", "unknown")
                        face_label = f"face:{face_status} {match_score:.2f}"
                        cv2.putText(
                            frame,
                            face_label,
                            (fx1, max(15, fy1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 165, 255),
                            1,
                        )

            # Draw FPS and timing stats
            stats_text = (
                f"FPS: {self.fps:.1f} | Det: {self.detection_count} | "
                f"Fid: {self._current_frame_id} | Delay: {self._analysis_delay_ms:.0f}ms"
            )
            cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            return frame

    # ──────────────────────────────────────────────────────────────────────────
    def is_alive(self) -> bool:
        """Check if worker running"""
        return self._worker_thread and self._worker_thread.is_alive()

    # ──────────────────────────────────────────────────────────────────────────
    def get_stats(self) -> Dict:
        """Lấy stats"""
        with self._state_lock:
            return {
                "camera_id": self.camera_id,
                "frames_processed": self.frame_count,
                "current_detections": self.detection_count,
                "fps": self.fps,
                "cached_faces": len(self._face_cache),
                "frame_id": self._current_frame_id,
                "analysis_delay_ms": self._analysis_delay_ms,
            }
