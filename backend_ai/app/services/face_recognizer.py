"""
face_recognizer.py - Extract embedding và so khớp với face database
Output: {name, score, status}
"""

import json
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

from app.config import (
    INSIGHTFACE_MODEL_NAME,
    INSIGHTFACE_CTX_ID,
    INSIGHTFACE_DET_SIZE,
    EMBEDDINGS_PATH,
    NAMES_PATH,
    OUTPUT_DIR,
    RECOGNITION_SIM_THRESHOLD,
    FAISS_INDEX_PATH,
    FAISS_META_DB_PATH,
    FAISS_NPROBE,
    FAISS_USE_GPU,
    FAISS_EF_SEARCH,
)
from app.core.exceptions import ModelNotLoadedError, EmbeddingDBNotFoundError
from app.services.embedding_searcher import create_searcher, BaseEmbeddingSearcher

logger = logging.getLogger(__name__)


def _resize_for_preview(
    image: np.ndarray,
    max_width: int | None,
    max_height: int | None,
) -> np.ndarray:
    if max_width is None and max_height is None:
        return image

    height, width = image.shape[:2]
    width_scale = (max_width / width) if max_width else None
    height_scale = (max_height / height) if max_height else None

    scales = [scale for scale in (width_scale, height_scale) if scale is not None]
    if not scales:
        return image

    scale = min(scales)
    if scale <= 0:
        return image

    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))

    import cv2

    return cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return b @ a


class FaceRecognizer:
    def __init__(self):
        # Load InsightFace (recognition module)
        try:
            self._app = FaceAnalysis(
                name=INSIGHTFACE_MODEL_NAME,
                allowed_modules=["detection", "recognition"],
            )
            self._app.prepare(ctx_id=INSIGHTFACE_CTX_ID, det_size=INSIGHTFACE_DET_SIZE)
            logger.info("[FaceRecognizer] InsightFace recognition đã sẵn sàng.")
        except Exception as e:
            raise ModelNotLoadedError(f"Không load được InsightFace recognizer: {e}") from e

        # Load searcher (FAISS nếu có index, numpy nếu không)
        self._load_db()

    # ──────────────────────────────────────────────────────────────────────────
    def _load_db(self):
        try:
            embeddings = np.load(EMBEDDINGS_PATH)
            with open(NAMES_PATH, "r", encoding="utf-8") as f:
                names: list[str] = json.load(f)
        except FileNotFoundError as e:
            raise EmbeddingDBNotFoundError(f"Không tìm thấy face DB: {e}") from e

        self._searcher: BaseEmbeddingSearcher = create_searcher(
            embeddings=embeddings,
            names=names,
            faiss_index_path=str(FAISS_INDEX_PATH) if FAISS_INDEX_PATH else None,
            faiss_db_path=str(FAISS_META_DB_PATH) if FAISS_META_DB_PATH else None,
            use_gpu=FAISS_USE_GPU,
            nprobe=FAISS_NPROBE,
            ef_search=FAISS_EF_SEARCH,
        )
        logger.info(
            "[FaceRecognizer] Đã load DB: %d vectors (%s).",
            self._searcher.total_vectors(),
            type(self._searcher).__name__,
        )

    # ──────────────────────────────────────────────────────────────────────────
    def reload_db(self):
        """Hot-reload face database (dùng khi thêm nhân viên mới)."""
        self._load_db()

    # ──────────────────────────────────────────────────────────────────────────
    def get_embedding(self, roi: np.ndarray, face_info: dict) -> np.ndarray | None:
        """Lấy embedding từ face crop trong ROI."""
        faces = self._app.get(roi)
        if not faces:
            return None
        # Chọn face có det_score cao nhất
        best = max(faces, key=lambda f: f.det_score)
        return best.normed_embedding   # shape (512,)

    # ──────────────────────────────────────────────────────────────────────────
    def recognize(self, roi: np.ndarray, face_info: dict) -> dict:
        """
        Trả về {"name": str|None, "score": float, "status": "recognized"|"unknown"}
        """
        embedding = self.get_embedding(roi, face_info)
        if embedding is None:
            return {"name": None, "score": 0.0, "status": "unknown"}
        return self._searcher.search_one(embedding, RECOGNITION_SIM_THRESHOLD)

    # ──────────────────────────────────────────────────────────────────────────
    def detect_and_embed(self, frame: np.ndarray) -> list:
        """
        Detect tất cả face trong frame VÀ extract embedding trong 1 GPU pass duy nhất.
        Trả về list InsightFace Face objects (có .bbox, .det_score, .normed_embedding).
        """
        return self._app.get(frame)

    # ──────────────────────────────────────────────────────────────────────────
    def match_embedding(self, normed_emb: np.ndarray) -> dict:
        """So khớp 1 embedding với DB. Trả về {"name", "score", "status"}."""
        return self._searcher.search_one(normed_emb, RECOGNITION_SIM_THRESHOLD)

    # ──────────────────────────────────────────────────────────────────────────
    def batch_match_embeddings(self, normed_embs: np.ndarray) -> list[dict]:
        """
        Batch match N embeddings cùng lúc.
        normed_embs: shape (N, 512)
        Trả về list[{"name", "score", "status"}]
        """
        return self._searcher.search_batch(normed_embs, RECOGNITION_SIM_THRESHOLD)

# ──────────────────────────────────────────────────────────────────────────
def _run_demo(preview_width: int = 1280, preview_height: int = 720) -> None:
    import cv2
    import os
    face_recognizer = FaceRecognizer()
    image_path = r"E:\face recognition\data test\IMG_003-A.JPG"
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh demo: {image_path}")

    faces = face_recognizer.detect_and_embed(image)
    preview = image.copy()

    # Batch match tất cả embeddings cùng lúc
    if faces:
        embeddings_batch = np.array([face.normed_embedding for face in faces])  # shape (N, 512)
        matches = face_recognizer.batch_match_embeddings(embeddings_batch)
        results = []
        
        for face, match in zip(faces, matches):
            x1, y1, x2, y2 = [int(value) for value in face.bbox]
            label = match["name"] if match["status"] == "recognized" else "unknown"
            results.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "name": match["name"],
                    "score": match["score"],
                    "status": match["status"],
                }
            )

            color = (0, 200, 0) if match["status"] == "recognized" else (0, 165, 255)
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                preview,
                f"{label}: {match['score']:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
    else:
        results = []

    preview = _resize_for_preview(preview, preview_width, preview_height)
    preview_path = Path(OUTPUT_DIR) / "face_recognizer_preview.jpg"
    for r in results:
        name = r["name"] if r["status"] == "recognized" else "unknown"
        print(f"{name} (score: {r['score']:.4f})")

    try:
        cv2.namedWindow("Face Recognizer Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Face Recognizer Preview", preview.shape[1], preview.shape[0])
        cv2.imshow("Face Recognizer Preview", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    except cv2.error:
        cv2.imwrite(str(preview_path), preview)
        print(f"OpenCV GUI không khả dụng. Đã lưu ảnh preview tại: {preview_path}")
        if os.name == "nt":
            os.startfile(str(preview_path))


if __name__ == "__main__":
    import os

    _run_demo(
        preview_width=int(os.getenv("FACE_RECOGNIZER_PREVIEW_WIDTH", "1280")),
        preview_height=int(os.getenv("FACE_RECOGNIZER_PREVIEW_HEIGHT", "720")),
    )

