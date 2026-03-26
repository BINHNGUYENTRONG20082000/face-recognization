"""
schemas.py - Pydantic schemas cho FastAPI request/response
"""

from pydantic import BaseModel
from typing import Optional


class DetectionOut(BaseModel):
    track_id:    int
    person_bbox: list[float]
    face_bbox:   Optional[list[float]]
    name:        Optional[str]
    score:       Optional[float]
    status:      str
    source:      str


class FrameResultOut(BaseModel):
    frame_id:       int
    timestamp:      float
    camera_id:      str
    num_detections: int
    detections:     list[DetectionOut]


class StatusOut(BaseModel):
    status:    str
    camera_id: str
    message:   str
