"""
routes.py - FastAPI router: health check, stream control, DB reload
"""

import threading
import logging
from fastapi import APIRouter, HTTPException

from app.api.schemas import CameraRuntimeOut, StatusOut
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


def _is_running(camera_id: str) -> bool:
    thread = _threads.get(camera_id)
    return thread is not None and thread.is_alive()


def _camera_status(camera_id: str) -> CameraRuntimeOut:
    pipeline = _pipelines.get(camera_id)
    is_alive = _is_running(camera_id)
    if pipeline is None and not is_alive:
        status = "stopped"
    elif pipeline is not None and pipeline.stop_requested:
        status = "stopping"
    else:
        status = "running"
    return CameraRuntimeOut(camera_id=camera_id, status=status, is_alive=is_alive)


def shutdown_pipelines() -> None:
    for pipeline in list(_pipelines.values()):
        pipeline.stop()


# ──────────────────────────────────────────────────────────────────────────────
@router.get("/health", tags=["System"])
def health_check():
    running = [camera_id for camera_id in CAMERA_SOURCES if _is_running(camera_id)]
    return {"status": "ok", "running_cameras": running, "total_configured": len(CAMERA_SOURCES)}


@router.get("/cameras", tags=["Camera"])
def list_cameras():
    return {"cameras": [_camera_status(camera_id).model_dump() for camera_id in CAMERA_SOURCES]}


@router.get("/cameras/{camera_id}", response_model=CameraRuntimeOut, tags=["Camera"])
def camera_status(camera_id: str):
    if camera_id not in CAMERA_SOURCES:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy camera: {camera_id}")
    return _camera_status(camera_id)


@router.post("/cameras/{camera_id}/start", response_model=StatusOut, tags=["Camera"])
def start_camera(camera_id: str):
    if camera_id not in CAMERA_SOURCES:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy camera: {camera_id}")
    if _is_running(camera_id):
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
    _pipelines[camera_id].stop()
    return StatusOut(status="stopping", camera_id=camera_id, message="Đã gửi lệnh dừng.")


@router.post("/face-db/reload", tags=["FaceDB"])
def reload_face_db():
    """Hot-reload face database khi thêm nhân viên mới."""
    for pipeline in _pipelines.values():
        pipeline.recognizer.reload_db()
    return {"status": "reloaded", "cameras": list(_pipelines.keys())}
