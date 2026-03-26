"""
face_recognizer.py - Extract embedding và so khớp với face database
Output: {name, score, status}
"""

import json
import logging
import numpy as np
import insightface
from insightface.app import FaceAnalysis

from app.config import (
    INSIGHTFACE_MODEL_NAME,
    INSIGHTFACE_DET_SIZE,
    EMBEDDINGS_PATH,
    NAMES_PATH,
    RECOGNITION_SIM_THRESHOLD,
)
from app.core.exceptions import ModelNotLoadedError, EmbeddingDBNotFoundError

logger = logging.getLogger(__name__)


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
            self._app.prepare(ctx_id=0, det_size=INSIGHTFACE_DET_SIZE)
            logger.info("[FaceRecognizer] InsightFace recognition đã sẵn sàng.")
        except Exception as e:
            raise ModelNotLoadedError(f"Không load được InsightFace recognizer: {e}") from e

        # Load face database
        self._load_db()

    # ──────────────────────────────────────────────────────────────────────────
    def _load_db(self):
        try:
            self._embeddings: np.ndarray = np.load(EMBEDDINGS_PATH)
            with open(NAMES_PATH, "r", encoding="utf-8") as f:
                self._names: list[str] = json.load(f)
            logger.info(
                f"[FaceRecognizer] Đã load {len(self._names)} nhân viên từ DB."
            )
        except FileNotFoundError as e:
            raise EmbeddingDBNotFoundError(
                f"Không tìm thấy face DB: {e}"
            ) from e

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

        sims = _cosine_similarity(embedding, self._embeddings)
        best_idx  = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score >= RECOGNITION_SIM_THRESHOLD:
            return {
                "name":   self._names[best_idx],
                "score":  best_score,
                "status": "recognized",
            }
        return {"name": None, "score": best_score, "status": "unknown"}

    # ──────────────────────────────────────────────────────────────────────────
    def detect_and_embed(self, frame: np.ndarray) -> list:
        """
        Detect tất cả face trong frame VÀ extract embedding trong 1 GPU pass duy nhất.
        Trả về list InsightFace Face objects (có .bbox, .det_score, .normed_embedding).
        """
        return self._app.get(frame)

    # ──────────────────────────────────────────────────────────────────────────
    def match_embedding(self, normed_emb: np.ndarray) -> dict:
        """
        So khớp embedding đã extract sẵn với DB — thuần numpy, không dùng GPU.
        Trả về {"name", "score", "status"}.
        """
        sims       = _cosine_similarity(normed_emb, self._embeddings)
        best_idx   = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        if best_score >= RECOGNITION_SIM_THRESHOLD:
            return {"name": self._names[best_idx], "score": best_score, "status": "recognized"}
        return {"name": None, "score": best_score, "status": "unknown"}
