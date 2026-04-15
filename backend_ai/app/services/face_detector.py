"""
face_detector.py - Detect face trong person ROI bằng InsightFace
Output: list[{bbox, kps, det_score}]  — sắp xếp theo score giảm dần
"""

import logging
import numpy as np
from insightface.app import FaceAnalysis

from app.config import INSIGHTFACE_CTX_ID, INSIGHTFACE_MODEL_NAME, INSIGHTFACE_DET_SIZE
from app.core.exceptions import ModelNotLoadedError

logger = logging.getLogger(__name__)


class FaceDetector:
    def __init__(self):
        try:
            self._app = FaceAnalysis(
                name=INSIGHTFACE_MODEL_NAME,
                allowed_modules=["detection"],
            )
            self._app.prepare(ctx_id=INSIGHTFACE_CTX_ID, det_size=INSIGHTFACE_DET_SIZE)
            logger.info("[FaceDetector] InsightFace detection đã sẵn sàng.")
        except Exception as e:
            raise ModelNotLoadedError(f"Không load được InsightFace: {e}") from e

    # ──────────────────────────────────────────────────────────────────────────
    def detect(self, roi: np.ndarray) -> list[dict]:
        """
        roi: numpy BGR image (person crop)
        Trả về list[{"bbox": [x1,y1,x2,y2], "kps": ..., "det_score": float}]
        sắp xếp theo score giảm dần.
        """
        faces = self._app.get(roi)
        results = []
        for face in faces:
            results.append({
                "bbox":      face.bbox.tolist(),
                "kps":       face.kps.tolist() if face.kps is not None else None,
                "det_score": float(face.det_score),
            })
        # Sắp xếp face tốt nhất lên đầu
        results.sort(key=lambda x: x["det_score"], reverse=True)
        return results
