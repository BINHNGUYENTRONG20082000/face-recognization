"""
result_builder.py - Tạo frame result theo schema chuẩn
"""

import time


class ResultBuilder:
    """
    Schema output mỗi frame:
    {
        "frame_id": int,
        "timestamp": float,
        "camera_id": str,
        "num_detections": int,
        "detections": [
            {
                "track_id": int,
                "person_bbox": [x1,y1,x2,y2],
                "face_bbox": [x1,y1,x2,y2] | null,
                "name": str | null,
                "score": float | null,
                "status": "recognized" | "unknown" | "no_face" | "skip",
                "source": "recomputed" | "cache"
            }
        ]
    }
    """

    def build(
        self,
        frame_id: int,
        timestamp: float,
        camera_id: str,
        detections: list[dict],
    ) -> dict:
        return {
            "frame_id":       frame_id,
            "timestamp":      timestamp,
            "camera_id":      camera_id,
            "num_detections": len(detections),
            "detections":     detections,
        }
