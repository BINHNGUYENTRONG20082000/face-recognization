"""
track_memory.py - Lưu trạng thái nhận diện theo person track_id
Voting nhiều frame + cache để giảm compute
"""

import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

_VOTE_WINDOW = 7   # số frame dùng để voting


class TrackMemory:
    """
    State per track_id:
    {
        "name": str | None,
        "score": float,
        "status": "recognized" | "unknown" | "no_face",
        "last_recognized_frame": int,
        "vote_buffer": deque of (name, score),
    }
    """

    def __init__(self):
        self._store: dict[int, dict] = {}
        self._votes: dict[int, deque] = defaultdict(lambda: deque(maxlen=_VOTE_WINDOW))

    # ──────────────────────────────────────────────────────────────────────────
    def get(self, track_id: int) -> dict | None:
        return self._store.get(track_id)

    # ──────────────────────────────────────────────────────────────────────────
    def update(
        self,
        track_id: int,
        name: str | None,
        score: float,
        status: str,
        frame_id: int,
    ):
        """
        Cập nhật state + voting.
        - no_face / bad_face: chỉ cập nhật timer, GIỮ NGUYÊN tên/status đã recognized
          để tránh xóa kết quả tốt khi mặt bị che tạm thời.
        - recognized: voting tên phổ biến nhất trong window.
        - unknown: lưu thẳng.
        """
        prev = self._store.get(track_id)

        # no_face / bad_face → chỉ cập nhật timer, không thay đổi tên
        if status in ("no_face", "bad_face"):
            if prev is not None:
                self._store[track_id] = {
                    **prev,
                    "last_recognized_frame": frame_id,
                }
            else:
                self._store[track_id] = {
                    "name": None, "score": 0.0,
                    "status": status,
                    "last_recognized_frame": frame_id,
                }
            return

        # Voting cho recognized
        if status == "recognized":
            self._votes[track_id].append((name, score))
            name_counts: dict[str, float] = defaultdict(float)
            for n, s in self._votes[track_id]:
                if n:
                    name_counts[n] += s
            if name_counts:
                best_name  = max(name_counts, key=lambda k: name_counts[k])
                best_score = name_counts[best_name] / len(self._votes[track_id])
                name  = best_name
                score = best_score

        self._store[track_id] = {
            "name":                  name,
            "score":                 score,
            "status":                status,
            "last_recognized_frame": frame_id,
        }
        logger.debug(f"[TrackMemory] track {track_id} → {name} ({score:.3f}) {status}")

    # ──────────────────────────────────────────────────────────────────────────
    def remove(self, track_id: int):
        """Xoá track khi mất khỏi scene."""
        self._store.pop(track_id, None)
        self._votes.pop(track_id, None)

    # ──────────────────────────────────────────────────────────────────────────
    def cleanup(self, active_track_ids: set[int]):
        """Xoá các track không còn active để giải phóng bộ nhớ."""
        stale = set(self._store.keys()) - active_track_ids
        for tid in stale:
            self.remove(tid)
