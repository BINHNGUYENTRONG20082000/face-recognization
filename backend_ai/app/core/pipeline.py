"""
pipeline.py - Orchestrator kết nối toàn bộ modules realtime
"""

import time
import logging
from typing import Optional

from app.config import (
    CAMERA_SOURCES,
    RECOGNITION_REFRESH_FRAMES,
    UNKNOWN_RETRY_FRAMES,
)
from app.services.stream_reader import StreamReader
from app.services.person_detector import PersonDetector
from app.services.tracker import PersonTracker
from app.services.face_detector import FaceDetector
from app.services.face_recognizer import FaceRecognizer
from app.services.track_memory import TrackMemory
from app.services.result_builder import ResultBuilder
from app.services.logger_service import LoggerService

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Pipeline realtime:
    Frame → PersonDetect → Track → FaceDetect → Recognize → Memory → Result → Log
    """

    def __init__(self, camera_id: str = "cam_01"):
        self.camera_id   = camera_id
        self.stream      = StreamReader(camera_id, CAMERA_SOURCES[camera_id])
        self.detector    = PersonDetector()
        self.tracker     = PersonTracker()
        self.face_det    = FaceDetector()
        self.recognizer  = FaceRecognizer()
        self.memory      = TrackMemory()
        self.builder     = ResultBuilder()
        self.log_service = LoggerService(camera_id)

    # ──────────────────────────────────────────────────────────────────────────
    def _should_recognize(self, track_id: int, frame_id: int) -> bool:
        """
        Chính sách giảm xử lý lặp:
        - Track mới          → recognize ngay
        - Recognized         → refresh mỗi RECOGNITION_REFRESH_FRAMES
        - Unknown            → retry mỗi UNKNOWN_RETRY_FRAMES
        - Face xấu / skip    → False
        """
        state = self.memory.get(track_id)
        if state is None:
            return True
        if state["status"] == "recognized":
            return (frame_id - state["last_recognized_frame"]) >= RECOGNITION_REFRESH_FRAMES
        if state["status"] == "unknown":
            return (frame_id - state["last_recognized_frame"]) >= UNKNOWN_RETRY_FRAMES
        return False

    # ──────────────────────────────────────────────────────────────────────────
    def process_frame(self, frame, frame_id: int, timestamp: float) -> dict:
        # 1. Detect persons
        persons = self.detector.detect(frame)

        # 2. Update tracker
        tracks = self.tracker.update(persons, frame)

        detections = []
        for track in tracks:
            track_id   = track["track_id"]
            person_box = track["bbox"]

            # 3. Crop ROI
            x1, y1, x2, y2 = map(int, person_box)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            face_box: Optional[list] = None
            name        = None
            score       = None
            status      = "skip"
            source      = "cache"

            cached = self.memory.get(track_id)

            if self._should_recognize(track_id, frame_id):
                # 4. Detect face trong ROI
                faces = self.face_det.detect(roi)
                if faces:
                    best_face = faces[0]   # face có score cao nhất
                    face_box  = best_face["bbox"]

                    # 5. Extract embedding & match
                    result = self.recognizer.recognize(roi, best_face)
                    name   = result["name"]
                    score  = result["score"]
                    status = result["status"]
                    source = "recomputed"

                    # 6. Update memory
                    self.memory.update(track_id, name, score, status, frame_id)
                else:
                    status = "no_face"
                    if cached:
                        name, score, status = cached["name"], cached["score"], cached["status"]
                        source = "cache"
            elif cached:
                name  = cached["name"]
                score = cached["score"]
                status = cached["status"]
                source = "cache"

            detections.append({
                "track_id":   track_id,
                "person_bbox": list(person_box),
                "face_bbox":  face_box,
                "name":       name,
                "score":      round(float(score), 4) if score else None,
                "status":     status,
                "source":     source,
            })

        # 7. Build frame result
        result = self.builder.build(
            frame_id=frame_id,
            timestamp=timestamp,
            camera_id=self.camera_id,
            detections=detections,
        )

        # 8. Log
        self.log_service.write(result)
        return result

    # ──────────────────────────────────────────────────────────────────────────
    def run(self):
        logger.info(f"[Pipeline] Bắt đầu camera {self.camera_id}")
        frame_id = 0
        start_ts = time.time()

        for frame in self.stream.read():
            timestamp = round(time.time() - start_ts, 3)
            result = self.process_frame(frame, frame_id, timestamp)
            logger.debug(f"[Frame {frame_id}] {len(result['detections'])} detections")
            frame_id += 1

        self.log_service.close()
        logger.info("[Pipeline] Kết thúc")
