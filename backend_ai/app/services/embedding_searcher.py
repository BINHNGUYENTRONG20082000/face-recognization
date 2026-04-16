"""
embedding_searcher.py - Tìm kiếm embedding trong DB (numpy hoặc FAISS)

Interface chung:
    searcher.search_batch(normed_embs)  →  list[dict{name, score, status}]
    searcher.search_one(normed_emb)     →  dict{name, score, status}

Dùng NumpySearcher khi DB nhỏ (< ~20K).
Dùng FaissSearcher khi DB lớn (hàng triệu embedding).
"""

import logging
import os
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Base interface
# ─────────────────────────────────────────────────────────────────────────────
class BaseEmbeddingSearcher(ABC):
    """Interface chung để FaceRecognizer có thể swap backend mà không cần sửa code."""

    @abstractmethod
    def search_batch(self, normed_embs: np.ndarray, threshold: float) -> list[dict]:
        """
        normed_embs: shape (N, 512), đã L2-normalized
        threshold: ngưỡng cosine similarity tối thiểu
        Trả về list[{"name", "score", "status"}] độ dài N.
        """

    @abstractmethod
    def search_one(self, normed_emb: np.ndarray, threshold: float) -> dict:
        """Tìm kiếm 1 embedding đơn. Trả về {"name", "score", "status"}."""

    @abstractmethod
    def total_vectors(self) -> int:
        """Số embedding hiện có trong DB."""


# ─────────────────────────────────────────────────────────────────────────────
# Numpy backend — phù hợp DB nhỏ, exact search
# ─────────────────────────────────────────────────────────────────────────────
class NumpySearcher(BaseEmbeddingSearcher):
    """
    Exact cosine similarity bằng numpy matrix multiplication.
    DB: embeddings.npy + names.json (cấu trúc hiện tại của project).
    """

    def __init__(self, embeddings: np.ndarray, names: list[str]):
        """
        embeddings: shape (K, 512) float32 đã normalize
        names: list[str] độ dài K
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != 512:
            raise ValueError(f"embeddings phải shape (K, 512), nhận được {embeddings.shape}")
        if len(names) != embeddings.shape[0]:
            raise ValueError("Số lượng names và embeddings không khớp.")

        # Pre-normalize để tránh normalize lại mỗi lần query
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        self._embeddings: np.ndarray = (embeddings / norms).astype(np.float32)
        self._names = names
        logger.info("[NumpySearcher] Loaded %d embeddings.", len(names))

    # ──────────────────────────────────────────────────────────────────────────
    def search_batch(self, normed_embs: np.ndarray, threshold: float) -> list[dict]:
        if len(self._names) == 0:
            return [{"name": None, "score": 0.0, "status": "unknown"}] * len(normed_embs)

        # Normalize query
        q_norms = np.linalg.norm(normed_embs, axis=1, keepdims=True) + 1e-8
        query = (normed_embs / q_norms).astype(np.float32)

        # (N, 512) @ (512, K) = (N, K)
        sims = query @ self._embeddings.T

        best_indices = np.argmax(sims, axis=1)
        best_scores = np.take_along_axis(sims, best_indices[:, None], axis=1).flatten()

        results = []
        for best_idx, best_score in zip(best_indices, best_scores):
            score = float(best_score)
            if score >= threshold:
                results.append({
                    "name": self._names[int(best_idx)],
                    "score": score,
                    "status": "recognized",
                })
            else:
                results.append({"name": None, "score": score, "status": "unknown"})
        return results

    def search_one(self, normed_emb: np.ndarray, threshold: float) -> dict:
        return self.search_batch(normed_emb.reshape(1, -1), threshold)[0]

    def total_vectors(self) -> int:
        return self._embeddings.shape[0]


# ─────────────────────────────────────────────────────────────────────────────
# FAISS backend — phù hợp DB lớn (hàng chục nghìn đến hàng triệu)
# ─────────────────────────────────────────────────────────────────────────────
class FaissSearcher(BaseEmbeddingSearcher):
    """
    ANN search bằng FAISS (IVF, HNSW, Flat).
    Index file + SQLite metadata — cùng format với build_faiss_index.py.
    """

    def __init__(
        self,
        index_path: str,
        db_path: str,
        use_gpu: bool = False,
        nprobe: int | None = None,
        ef_search: int | None = None,
    ):
        import faiss

        if not Path(index_path).exists():
            raise FileNotFoundError(f"Không tìm thấy FAISS index: {index_path}")
        if not Path(db_path).exists():
            raise FileNotFoundError(f"Không tìm thấy metadata DB: {db_path}")

        logger.info("[FaissSearcher] Đang tải index: %s", index_path)
        self._index = faiss.read_index(index_path)

        if use_gpu:
            res = faiss.StandardGpuResources()
            self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
            logger.info("[FaissSearcher] Đã chuyển index lên GPU.")

        # Truy cập underlying index (nếu wrap trong IndexIDMap/IndexIDMap2)
        underlying_idx = getattr(self._index, "index", self._index)

        # Tăng efSearch nếu là HNSW
        if hasattr(underlying_idx, "hnsw"):
            underlying_idx.hnsw.efSearch = ef_search if ef_search is not None else 128
            logger.info("[FaissSearcher] HNSW efSearch=%d", underlying_idx.hnsw.efSearch)

        # Thiết lập nprobe nếu là IVF
        if hasattr(underlying_idx, "nprobe"):
            nlist = int(getattr(underlying_idx, "nlist", 1024))
            auto_nprobe = max(16, min(128, nlist // 16))
            underlying_idx.nprobe = int(nprobe) if nprobe is not None else auto_nprobe
            logger.info(
                "[FaissSearcher] IVF nprobe=%d (nlist=%d)",
                underlying_idx.nprobe,
                nlist,
            )

        logger.info("[FaissSearcher] %d vectors trong index.", self._index.ntotal)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        # Cache metadata trong RAM để tránh truy vấn SQLite mỗi query.
        self._id_to_name: dict[int, str] = {
            int(row["faiss_id"]): str(row["name"])
            for row in self._conn.execute("SELECT faiss_id, name FROM persons")
        }

    # ──────────────────────────────────────────────────────────────────────────
    def search_batch(self, normed_embs: np.ndarray, threshold: float) -> list[dict]:
        if self._index.ntotal == 0:
            return [{"name": None, "score": 0.0, "status": "unknown"}] * len(normed_embs)

        query = normed_embs.astype(np.float32)
        norms = np.linalg.norm(query, axis=1, keepdims=True) + 1e-8
        query /= norms

        # FAISS search trả về shape (N, top_k)
        scores, ids = self._index.search(query, 1)
        scores = scores[:, 0]
        ids = ids[:, 0]

        results = []
        for score, faiss_id in zip(scores, ids):
            score = float(score)
            if faiss_id < 0 or score < threshold:
                results.append({"name": None, "score": score, "status": "unknown"})
                continue

            name = self._id_to_name.get(int(faiss_id))
            if name:
                results.append({"name": name, "score": score, "status": "recognized"})
            else:
                results.append({"name": None, "score": score, "status": "unknown"})

        return results

    def search_one(self, normed_emb: np.ndarray, threshold: float) -> dict:
        return self.search_batch(normed_emb.reshape(1, -1), threshold)[0]

    def total_vectors(self) -> int:
        return self._index.ntotal

    def set_nprobe(self, nprobe: int) -> None:
        """Điều chỉnh nprobe runtime để tune speed vs accuracy."""
        underlying_idx = getattr(self._index, "index", self._index)
        if hasattr(underlying_idx, "nprobe"):
            underlying_idx.nprobe = nprobe

    def stats(self) -> dict:
        total_persons = self._conn.execute(
            "SELECT COUNT(DISTINCT person_id) FROM persons"
        ).fetchone()[0]
        underlying_idx = getattr(self._index, "index", self._index)
        return {
            "total_vectors": self._index.ntotal,
            "total_persons": total_persons,
            "nprobe": int(underlying_idx.nprobe) if hasattr(underlying_idx, "nprobe") else None,
        }

    def close(self) -> None:
        self._conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Factory — tự chọn backend dựa vào config
# ─────────────────────────────────────────────────────────────────────────────
def create_searcher(
    embeddings: np.ndarray | None = None,
    names: list[str] | None = None,
    faiss_index_path: str | None = None,
    faiss_db_path: str | None = None,
    use_gpu: bool = False,
    nprobe: int | None = None,
    ef_search: int | None = None,
) -> BaseEmbeddingSearcher:
    """
    Chọn backend tự động:
    - FaissSearcher (ưu tiên): khi có file FAISS index + metadata DB.
      HNSW: ~99% recall, nhanh trên CPU, phù hợp hàng triệu vectors.
      IVFFlat: cần nprobe đủ cao để đảm bảo recall.
    - NumpySearcher (fallback): exact search khi không có FAISS files.
      Phù hợp DB < ~200K vectors (bị giới hạn bởi RAM và tốc độ CPU).
    - Đặt FORCE_NUMPY=1 để ép dùng NumpySearcher dù có FAISS files.
    """
    force_numpy = os.getenv("FORCE_NUMPY", "0") == "1"

    if (
        not force_numpy
        and faiss_index_path
        and faiss_db_path
        and Path(faiss_index_path).exists()
        and Path(faiss_db_path).exists()
    ):
        logger.info("[create_searcher] Dùng FaissSearcher (CPU HNSW/IVF).")
        return FaissSearcher(
            index_path=faiss_index_path,
            db_path=faiss_db_path,
            use_gpu=use_gpu,
            nprobe=nprobe,
            ef_search=ef_search,
        )

    if embeddings is not None and names is not None:
        logger.info("[create_searcher] Dùng NumpySearcher (exact search, recall 100%%).")
        return NumpySearcher(embeddings=embeddings, names=names)

    raise ValueError(
        "Phải cung cấp faiss_index_path + faiss_db_path cho FaissSearcher, "
        "hoặc embeddings + names cho NumpySearcher."
    )
