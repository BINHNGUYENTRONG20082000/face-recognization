"""
pipeline_v2.py - Modern pipeline using SharedAIService (new architecture)
Frame → CameraWorker → BatchQueue → SharedAIService (GPU) → StateManager → Overlay → Display
"""

import cv2
import time
import threading
import logging
from typing import Optional, List
from queue import Queue, Empty

from app.core.batch_queue import BatchQueue
from app.core.camera_worker import CameraWorker
from app.core.shared_ai_service import SharedAIService
from app.core.camera_state_manager import CameraStateManager
from app.services.logger_service import LoggerService
from app.config import CAMERA_SOURCES

logger = logging.getLogger(__name__)


class PipelineV2:
    """
    Modern multi-camera architecture using SharedAIService.
    
    Features:
    - Shared GPU models (no duplication)
    - Batch processing for efficiency
    - Per-camera state isolation
    - Streaming RTSP with auto-reconnect
    """

    def __init__(
        self,
        camera_id: int,
        rtsp_url: Optional[str] = None,
        batch_queue: Optional[BatchQueue] = None,
        shared_ai_service: Optional[SharedAIService] = None,
        input_fps_limit: float = 0.0,
    ):
        """
        Args:
            camera_id: int (0, 1, 2, ...)
            rtsp_url: RTSP URL (if None, use CAMERA_SOURCES[camera_id])
            batch_queue: Shared BatchQueue (if None, create new)
            shared_ai_service: Shared AI service (if None, create new singleton)
            input_fps_limit: giới hạn FPS đưa vào AI trên mỗi camera (0 = không giới hạn)
        """
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url or CAMERA_SOURCES.get(f"cam_{camera_id:02d}", "")
        self.input_fps_limit = input_fps_limit
        
        # Initialize components
        self.batch_queue = batch_queue or BatchQueue(batch_size=8, timeout_ms=20)
        self.shared_ai = shared_ai_service or SharedAIService()
        
        # Per-camera components
        self.camera_worker: Optional[CameraWorker] = None
        self.state_manager: Optional[CameraStateManager] = None
        self.log_service = LoggerService(f"cam_{camera_id:02d}")
        
        # Control
        self._stop_event = threading.Event()
        self._running = False
        
        logger.info(f"[PipelineV2 {camera_id}] Initialized with RTSP: {self.rtsp_url}")

    # ──────────────────────────────────────────────────────────────────────────
    def start(self):
        """Start camera worker and state manager"""
        if self._running:
            logger.warning(f"[PipelineV2 {self.camera_id}] Already running")
            return

        try:
            # Start camera worker (RTSP reader thread)
            self.camera_worker = CameraWorker(
                camera_id=self.camera_id,
                rtsp_url=self.rtsp_url,
                batch_queue=self.batch_queue,
                target_input_fps=self.input_fps_limit,
            )
            self.camera_worker.start()

            # Start shared AI service (only once)
            if not self.shared_ai.is_alive():
                self.shared_ai.start(self.batch_queue)

            # Start state manager (receives results from shared AI)
            result_queue = self.shared_ai.get_result_queue(self.camera_id)
            self.state_manager = CameraStateManager(self.camera_id, result_queue)
            self.state_manager.start()

            self._running = True
            self._stop_event.clear()
            logger.info(f"[PipelineV2 {self.camera_id}] Started successfully")

        except Exception as e:
            logger.error(f"[PipelineV2 {self.camera_id}] Start failed: {e}")
            self.stop()

    # ──────────────────────────────────────────────────────────────────────────
    def stop(self):
        """Stop camera worker and state manager"""
        if not self._running:
            return

        self._stop_event.set()

        try:
            if self.camera_worker:
                self.camera_worker.stop()
            if self.state_manager:
                self.state_manager.stop()
            self.log_service.close()
            self._running = False
            logger.info(f"[PipelineV2 {self.camera_id}] Stopped")
        except Exception as e:
            logger.error(f"[PipelineV2 {self.camera_id}] Stop error: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    def get_overlay_frame(self, use_live_stream: bool = True) -> Optional[cv2.Mat]:
        """Get current frame with overlay detections.
        
        Dùng frame mới nhất từ live stream để đảm bảo video mượt (đủ FPS),
        overlay detection boxes từ kết quả inference gần nhất.
        """
        if not self.state_manager:
            return None

        stream_frame = None
        if use_live_stream and self.camera_worker:
            stream_frame = self.camera_worker.get_latest_frame()

        return self.state_manager.get_overlay_frame(stream_frame=stream_frame)

    # ──────────────────────────────────────────────────────────────────────────
    def get_stats(self) -> dict:
        """Get statistics"""
        stats = {
            "camera_id": self.camera_id,
            "rtsp_url": self.rtsp_url,
            "is_running": self._running,
        }

        if self.camera_worker:
            stats.update({
                "worker": self.camera_worker.get_stats(),
            })

        if self.state_manager:
            stats.update({
                "state_manager": self.state_manager.get_stats(),
            })

        return stats

    # ──────────────────────────────────────────────────────────────────────────
    def is_alive(self) -> bool:
        """Check if pipeline running"""
        return self._running and self.camera_worker and self.camera_worker.is_alive()

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def create_shared_components(batch_size: int = 8) -> tuple:
        """
        Create shared components for N cameras.
        Returns: (batch_queue, shared_ai_service)
        
        Usage:
            batch_q, ai_svc = PipelineV2.create_shared_components()
            pipe1 = PipelineV2(0, batch_queue=batch_q, shared_ai_service=ai_svc)
            pipe2 = PipelineV2(1, batch_queue=batch_q, shared_ai_service=ai_svc)
        """
        batch_queue = BatchQueue(batch_size=batch_size, timeout_ms=20)
        shared_ai = SharedAIService()
        return batch_queue, shared_ai
