"""
face_detector.py - Detect face trong person ROI bằng InsightFace
Output: list[{bbox, kps, det_score}]  — sắp xếp theo score giảm dần
"""

import logging
from pathlib import Path
import sys

import numpy as np
from insightface.app import FaceAnalysis

# Cho phép chạy trực tiếp file này mà vẫn import được package `app`.
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from app.config import INSIGHTFACE_CTX_ID, INSIGHTFACE_MODEL_NAME, INSIGHTFACE_DET_SIZE, OUTPUT_DIR
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

# demo chạy thưc tế với ảnh test, vẽ kết quả lên ảnh và hiển thị hoặc lưu lại
def _run_demo() -> None:
    import cv2
    import os

    image_path = Path("E:/face recognition/data test/photo_2026-04-09_11-09-00.jpg")
    face_detector = FaceDetector()
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh demo: {image_path}")
    results = face_detector.detect(img)

    preview = img.copy()
    for result in results:
        x1, y1, x2, y2 = [int(value) for value in result["bbox"]]
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            preview,
            f"score: {result['det_score']:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if result["kps"]:
            for point in result["kps"]:
                px, py = [int(value) for value in point]
                cv2.circle(preview, (px, py), 2, (0, 0, 255), -1)

    print(results)
    try:
        cv2.imshow("Face Detector Preview", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    except cv2.error:
        pass

    preview_path = Path(OUTPUT_DIR) / "face_detector_preview.jpg"
    cv2.imwrite(str(preview_path), preview)
    print(f"OpenCV GUI không khả dụng. Đã lưu ảnh preview tại: {preview_path}")

    if os.name == "nt":
        os.startfile(str(preview_path))


if __name__ == "__main__":
    _run_demo()