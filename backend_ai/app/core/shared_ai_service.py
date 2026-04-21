"""
shared_ai_service.py - Shared GPU inference service cho N cameras
Chạy 1 instance duy nhất, xử lý batch frames từ tất cả cameras
"""

import cv2
import time
import threading
import logging
import numpy as np
from typing import Dict, List, Optional
from queue import Queue, Empty
from threading import Lock

from app.services.tracker import PersonTracker
from app.services.face_recognizer import FaceRecognizer
from app.services.track_memory import TrackMemory
from app.config import RECOGNITION_REFRESH_FRAMES, UNKNOWN_RETRY_FRAMES, FACE_REFRESH_FRAMES

logger = logging.getLogger(__name__)


class SharedAIService:
    """
    Centralized GPU inference service:
    - Load models 1 lần
    - Process batches từ unified BatchQueue
    - Return results vào per-camera result queues
    - Manage per-camera state (tracking)
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Load models once"""
        if hasattr(self, "_initialized") and self._initialized:
            return

        logger.info("[SharedAIService] Initializing...")

        # Load shared models
        self.face_recognizer = FaceRecognizer()
        logger.info("[SharedAIService] Model started: FaceRecognizer (InsightFace detection + recognition)")

        # Per-camera tracker instances.
        # NOTE: Ultralytics track(persist=True) keeps internal tracker state,
        # so each camera must use its own tracker instance to avoid state mixing.
        self._cam_trackers: Dict[int, PersonTracker] = {}
        self._cam_trackers_lock = Lock()

        # Per-camera tracking state
        self._cam_memories: Dict[int, TrackMemory] = {}  # {camera_id: TrackMemory}
        self._cam_memories_lock = Lock()
        self._cam_frame_ids: Dict[int, int] = {}  # {camera_id: frame_id}
        self._cam_face_caches: Dict[int, Dict[int, Dict]] = {}  # {camera_id: {track_id: cache}}

        # Result queues per camera
        self._result_queues: Dict[int, Queue] = {}  # {camera_id: Queue}
        self._result_queues_lock = Lock()

        # Worker thread
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._batch_queue = None  # Will be set by caller

        # Stats
        self._frame_count = 0
        self._batch_count = 0
        self._process_times = []

        self._initialized = True
        logger.info("[SharedAIService] Ready (models loaded)")

    # ──────────────────────────────────────────────────────────────────────────
    def get_model_info(self) -> Dict:
        """Return model startup info for monitoring/logging."""
        return {
            "models": [
                {
                    "name": "PersonTracker",
                    "backend": "Ultralytics YOLO",
                    "task": "person detection + tracking",
                    "started": len(self._cam_trackers) > 0,
                },
                {
                    "name": "FaceRecognizer",
                    "backend": "InsightFace",
                    "task": "face detection + embedding + identification",
                    "started": self.face_recognizer is not None,
                },
            ],
            "singleton_initialized": bool(getattr(self, "_initialized", False)),
        }

    # ──────────────────────────────────────────────────────────────────────────
    def set_batch_queue(self, batch_queue):
        """Set the unified batch queue"""
        self._batch_queue = batch_queue

    # ──────────────────────────────────────────────────────────────────────────
    def _get_or_create_memory(self, camera_id: int) -> TrackMemory:
        """Get TrackMemory for camera, create if not exists"""
        with self._cam_memories_lock:
            if camera_id not in self._cam_memories:
                self._cam_memories[camera_id] = TrackMemory()
                self._cam_frame_ids[camera_id] = 0
            return self._cam_memories[camera_id]

    # ──────────────────────────────────────────────────────────────────────────
    def _get_or_create_face_cache(self, camera_id: int) -> Dict[int, Dict]:
        """Get per-camera face cache, create if not exists."""
        with self._cam_memories_lock:
            if camera_id not in self._cam_face_caches:
                self._cam_face_caches[camera_id] = {}
            return self._cam_face_caches[camera_id]

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _should_recognize(memory: TrackMemory, track_id: int, frame_id: int) -> bool:
        """Throttle recognition frequency per track to reduce latency."""
        state = memory.get(track_id)
        if state is None:
            return True

        elapsed = frame_id - int(state.get("last_recognized_frame", -1))
        status = state.get("status")
        if status == "recognized":
            return elapsed >= RECOGNITION_REFRESH_FRAMES
        return elapsed >= UNKNOWN_RETRY_FRAMES

    # ──────────────────────────────────────────────────────────────────────────
    def _get_or_create_tracker(self, camera_id: int) -> PersonTracker:
        """Get PersonTracker for camera, create if not exists."""
        with self._cam_trackers_lock:
            if camera_id not in self._cam_trackers:
                self._cam_trackers[camera_id] = PersonTracker()
                logger.info(
                    f"[SharedAIService] Model started: PersonTracker cam={camera_id} "
                    f"(YOLO + ByteTrack/BoT-SORT)"
                )
            return self._cam_trackers[camera_id]

    # ──────────────────────────────────────────────────────────────────────────
    def _get_or_create_result_queue(self, camera_id: int) -> Queue:
        """Get result queue for camera, create if not exists"""
        with self._result_queues_lock:
            if camera_id not in self._result_queues:
                # Keep latest-only result queue to avoid stale overlays.
                self._result_queues[camera_id] = Queue(maxsize=1)
            return self._result_queues[camera_id]

    # ──────────────────────────────────────────────────────────────────────────
    def get_result_queue(self, camera_id: int) -> Queue:
        """Lấy result queue cho camera"""
        return self._get_or_create_result_queue(camera_id)

    # ──────────────────────────────────────────────────────────────────────────
    def _process_batch(self, batch: List[Dict]):
        """
        Process batch từ N cameras
        batch = [{camera_id, frame, timestamp}, ...]
        """
        start_time = time.time()

        # Group by camera_id để track state riêng
        batch_by_cam = {}
        for item in batch:
            cam_id = item["camera_id"]
            if cam_id not in batch_by_cam:
                batch_by_cam[cam_id] = []
            batch_by_cam[cam_id].append(item)

        # Process each camera's frames
        for cam_id, cam_frames in batch_by_cam.items():
            tracker = self._get_or_create_tracker(cam_id)
            for frame_item in cam_frames:
                frame = frame_item["frame"]
                ts = frame_item["timestamp"]
                input_frame_id = int(frame_item.get("frame_id", -1))

                try:
                    # 1. Person tracking
                    tracks = tracker.update([], frame)  # PersonTracker does detect+track

                    # 2. Face recognition per track (throttled + cached)
                    results_faces = []
                    memory = self._get_or_create_memory(cam_id)
                    face_cache = self._get_or_create_face_cache(cam_id)
                    frame_id = self._cam_frame_ids.get(cam_id, 0)

                    for track in tracks:
                        track_id = track["track_id"]
                        x1, y1, x2, y2 = track["bbox"]

                        # Crop face ROI
                        try:
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(frame.shape[1], x2)
                            y2 = min(frame.shape[0], y2)

                            # Focus ROI on upper-body/head for better face detection hit rate.
                            h = max(1, y2 - y1)
                            head_y2 = y1 + int(h * 0.6)
                            head_y2 = min(frame.shape[0], max(y1 + 1, head_y2))

                            face_roi = frame[y1:head_y2, x1:x2]
                            if face_roi.size == 0:
                                continue

                            cached = face_cache.get(track_id)
                            should_refresh_face = (
                                cached is None
                                or (frame_id - int(cached.get("last_frame", -1))) >= FACE_REFRESH_FRAMES
                            )

                            if should_refresh_face:
                                faces = self.face_recognizer.detect_and_embed(face_roi)
                                face_cache[track_id] = {
                                    "faces": faces,
                                    "last_frame": frame_id,
                                }
                            else:
                                faces = cached.get("faces", []) if cached else []

                            if not faces:
                                current_state = memory.get(track_id)
                                if current_state is None:
                                    memory.update(track_id, None, 0.0, "no_face", frame_id)
                                continue

                            best_face = faces[0]
                            local_fx1, local_fy1, local_fx2, local_fy2 = [int(v) for v in best_face.bbox]
                            face_bbox_abs = [
                                x1 + local_fx1,
                                y1 + local_fy1,
                                x1 + local_fx2,
                                y1 + local_fy2,
                            ]

                            face_cache[track_id]["face_bbox"] = face_bbox_abs

                            if self._should_recognize(memory, track_id, frame_id):
                                embedding = getattr(best_face, "normed_embedding", None)
                                if embedding is not None:
                                    match = self.face_recognizer.match_embedding(embedding)
                                    name = match.get("name")
                                    status = match.get("status", "unknown")
                                    score = float(match.get("score", 0.0))

                                    if status != "recognized" or not name:
                                        name = "Unknown"

                                    memory.update(track_id, name, score, status, frame_id)

                            state = memory.get(track_id) or {}
                            results_faces.append({
                                "track_id": track_id,
                                "person_bbox": track["bbox"],
                                "face_bbox": face_bbox_abs,
                                "face": best_face,
                                "confidence": track["score"],
                                "predicted_name": state.get("name") or "Unknown",
                                "voted_name": state.get("name") or "Unknown",
                                "status": state.get("status", "unknown"),
                                "match_score": float(state.get("score", 0.0)),
                            })
                        except Exception as e:
                            logger.warning(f"[Cam {cam_id}] Face process error: {e}")
                            continue

                    # 3. Push result to camera's result queue
                    result = {
                        "camera_id": cam_id,
                        "timestamp": ts,
                        "frame_id": input_frame_id,
                        "frame": frame,
                        "tracks": tracks,
                        "faces": results_faces,
                        "result_ts": time.time(),
                    }

                    active_track_ids = {t["track_id"] for t in tracks}
                    memory.cleanup(active_track_ids)
                    stale_track_ids = set(face_cache.keys()) - active_track_ids
                    for stale_track_id in stale_track_ids:
                        face_cache.pop(stale_track_id, None)
                    self._cam_frame_ids[cam_id] = frame_id + 1

                    result_queue = self._get_or_create_result_queue(cam_id)
                    try:
                        # Latest-only semantics: discard stale pending result if any.
                        if result_queue.full():
                            try:
                                result_queue.get_nowait()
                            except Empty:
                                pass
                        result_queue.put(result, block=False)
                    except Exception:
                        pass  # Drop if queue full

                    self._frame_count += 1

                except Exception as e:
                    logger.error(f"[Cam {cam_id}] Process error: {e}")

        # Record timing
        elapsed = time.time() - start_time
        self._process_times.append(elapsed)
        self._batch_count += 1

    # ──────────────────────────────────────────────────────────────────────────
    def _worker_loop(self):
        """Main worker loop: poll batch queue, process"""
        logger.info("[SharedAIService] Worker thread started")

        while not self._stop_event.is_set():
            try:
                # Get batch (blocks with timeout)
                batch = self._batch_queue.get_batch(timeout_override=0.05)

                if batch:
                    self._process_batch(batch)
                else:
                    time.sleep(0.01)

            except Exception as e:
                logger.error(f"[SharedAIService] Worker error: {e}")
                time.sleep(0.1)

        logger.info("[SharedAIService] Worker thread stopped")

    # ──────────────────────────────────────────────────────────────────────────
    def start(self, batch_queue):
        """Bắt đầu service"""
        self.set_batch_queue(batch_queue)

        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=False)
            self._worker_thread.start()
            logger.info("[SharedAIService] Started")

    # ──────────────────────────────────────────────────────────────────────────
    def stop(self):
        """Dừng service"""
        self._stop_event.set()

        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

        logger.info(
            f"[SharedAIService] Stopped - "
            f"batches={self._batch_count}, frames={self._frame_count}, "
            f"avg_time={np.mean(self._process_times):.2f}ms"
        )

    # ──────────────────────────────────────────────────────────────────────────
    def is_alive(self) -> bool:
        """Check if service running"""
        return self._worker_thread and self._worker_thread.is_alive()

    def get_stats(self) -> Dict:
        """Lấy stats"""
        avg_process_time = np.mean(self._process_times) if self._process_times else 0
        avg_fps = (self._frame_count / sum(self._process_times)) if sum(self._process_times) > 0 else 0

        return {
            "batches_processed": self._batch_count,
            "frames_processed": self._frame_count,
            "avg_batch_time_ms": avg_process_time * 1000,
            "throughput_fps": avg_fps,
            "active_cameras": len(self._cam_memories),
            "active_trackers": len(self._cam_trackers),
            "model_count": 2,
        }
