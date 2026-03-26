"""
routes.py - FastAPI router: health check, stream control, DB reload
"""

import threading
import logging
from fastapi import APIRouter, HTTPException

from app.api.schemas import StatusOut
from app.core.pipeline import Pipeline
from app.config import CAMERA_SOURCES

router = APIRouter()
logger = logging.getLogger(__name__)

# Registry các pipeline đang chạy
_pipelines: dict[str, Pipeline]     = {}
_threads:   dict[str, threading.Thread] = {}


def _run_pipeline(camera_id: str):
    try:
        _pipelines[camera_id].run()
    except Exception as e:
        logger.error(f"[Route] Pipeline {camera_id} lỗi: {e}", exc_info=True)
    finally:
        _pipelines.pop(camera_id, None)
        _threads.pop(camera_id, None)


# ──────────────────────────────────────────────────────────────────────────────
@router.get("/health", tags=["System"])
def health_check():
    return {"status": "ok"}


@router.get("/cameras", tags=["Camera"])
def list_cameras():
    return {"cameras": list(CAMERA_SOURCES.keys())}


@router.post("/cameras/{camera_id}/start", response_model=StatusOut, tags=["Camera"])
def start_camera(camera_id: str):
    if camera_id not in CAMERA_SOURCES:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy camera: {camera_id}")
    if camera_id in _threads and _threads[camera_id].is_alive():
        raise HTTPException(status_code=409, detail=f"Camera {camera_id} đã đang chạy.")

    pipeline = Pipeline(camera_id=camera_id)
    _pipelines[camera_id] = pipeline
    t = threading.Thread(target=_run_pipeline, args=(camera_id,), daemon=True)
    _threads[camera_id] = t
    t.start()
    return StatusOut(status="started", camera_id=camera_id, message="Pipeline đã khởi động.")


@router.post("/cameras/{camera_id}/stop", response_model=StatusOut, tags=["Camera"])
def stop_camera(camera_id: str):
    if camera_id not in _pipelines:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} không chạy.")
    # Báo stream dừng — reader sẽ tự release khi không còn frame
    _pipelines[camera_id].stream._cap and _pipelines[camera_id].stream._cap.release()
    return StatusOut(status="stopped", camera_id=camera_id, message="Đã gửi lệnh dừng.")


@router.post("/face-db/reload", tags=["FaceDB"])
def reload_face_db():
    """Hot-reload face database khi thêm nhân viên mới."""
    for cam_id, pipeline in _pipelines.items():
        pipeline.recognizer.reload_db()
    return {"status": "reloaded", "cameras": list(_pipelines.keys())}
