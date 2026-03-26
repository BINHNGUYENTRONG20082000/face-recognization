"""
test_rtsp.py - Test toàn bộ pipeline realtime với RTSP camera
Hiển thị overlay: person bbox, track_id, tên nhân viên, FPS

Cách dùng:
    python scripts/test_rtsp.py --rtsp "rtsp://user:pass@ip:port/stream"
    python scripts/test_rtsp.py --rtsp "rtsp://..." --show-fps --save-log

Phím tắt khi chạy:
    q   - Thoát
    s   - Lưu snapshot frame hiện tại
    r   - Reload face database
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import logging
import json
from pathlib import Path
from datetime import datetime

import threading
import queue
import cv2
import numpy as np

# Thêm root vào sys.path để import app.*
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import (
    YOLO_MODEL_PATH,
    EMBEDDINGS_PATH,
    NAMES_PATH,
    PERSON_CONF_THRESHOLD,
    RECOGNITION_SIM_THRESHOLD,
    RECOGNITION_REFRESH_FRAMES,
    UNKNOWN_RETRY_FRAMES,
    SNAPSHOTS_DIR,
    LOGS_DIR,
    INSIGHTFACE_MODEL_NAME,
    INSIGHTFACE_DET_SIZE,
    TRACKER_TYPE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Màu sắc overlay ──────────────────────────────────────────────────────────
COLOR_RECOGNIZED = (0, 220, 0)      # xanh lá
COLOR_UNKNOWN    = (0, 100, 255)    # cam
COLOR_NO_FACE    = (180, 180, 180)  # xám


# ─── Helper: vẽ text có nền ───────────────────────────────────────────────────
def _put_text_bg(frame, text: str, org: tuple, color: tuple, font_scale: float = 0.55):
    font      = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    cv2.rectangle(frame, (x, y - th - baseline - 2), (x + tw + 2, y + baseline), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + 1, y - 1), font, font_scale, color, thickness, cv2.LINE_AA)


# ─── Hàm vẽ overlay lên frame ─────────────────────────────────────────────────
def draw_frame(frame: np.ndarray, result: dict) -> np.ndarray:
    for det in result.get("detections", []):
        x1, y1, x2, y2 = map(int, det["person_bbox"])
        status  = det.get("status", "unknown")
        name    = det.get("name") or "Unknown"
        score   = det.get("score") or 0.0
        track_id = det.get("track_id", -1)
        source  = det.get("source", "")

        color = (COLOR_RECOGNIZED if status == "recognized"
                 else COLOR_NO_FACE if status == "no_face"
                 else COLOR_UNKNOWN)

        # Person bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Face bbox (tọa độ absolute trong frame)
        if det.get("face_bbox"):
            fx1, fy1, fx2, fy2 = map(int, det["face_bbox"])
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 255, 0), 1)

        # Label
        src_tag = "●" if source == "recomputed" else "○"
        if status == "recognized":
            label = f"#{track_id} {name} ({score:.2f}) {src_tag}"
        elif status == "no_face":
            label = f"#{track_id} [no face]"
        else:
            label = f"#{track_id} Unknown ({score:.2f}) {src_tag}"

        _put_text_bg(frame, label, (x1, y1 - 4), color)

    return frame


# ─── Chính sách giảm xử lý lặp (Mục 7) ─────────────────────────────────────
def _should_recognize(cached: dict | None, frame_id: int) -> bool:
    """
    - Track mới (chưa có cached)  → recognize ngay
    - recognized                  → refresh mỗi RECOGNITION_REFRESH_FRAMES (~10)
    - unknown / no_face / bad_face → retry mỗi UNKNOWN_RETRY_FRAMES (~3)
    """
    if cached is None:
        return True
    elapsed = frame_id - cached["last_recognized_frame"]
    if cached["status"] == "recognized":
        return elapsed >= RECOGNITION_REFRESH_FRAMES
    return elapsed >= UNKNOWN_RETRY_FRAMES


# ─── Helper: tìm face khớp với person bbox ────────────────────────────────────
def _find_face_in_box(faces: list, x1: int, y1: int, x2: int, y2: int):
    """
    Tìm InsightFace Face object có center nằm trong person bbox [x1,y1,x2,y2].
    Trả về face có det_score cao nhất, hoặc None nếu không tìm thấy.
    """
    best = None
    for face in faces:
        fb = face.bbox.astype(int)
        cx = (fb[0] + fb[2]) // 2
        cy = (fb[1] + fb[3]) // 2
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            if best is None or face.det_score > best.det_score:
                best = face
    return best


# ─── Main pipeline realtime ───────────────────────────────────────────────────
def run(rtsp_url: str, show_fps: bool, save_log: bool, save_video: bool = False):
    logger.info(f"Kết nối camera: {rtsp_url}")

    from app.services.tracker          import PersonTracker
    from app.services.face_recognizer  import FaceRecognizer
    from app.services.track_memory     import TrackMemory
    from app.services.result_builder   import ResultBuilder

    logger.info("Load models...")
    # Tracker: YOLO detect + ByteTrack trong 1 pass
    # Recognizer: detect + embed toàn frame trong 1 GPU pass (bỏ FaceDetector riêng)
    tracker    = PersonTracker()
    recognizer = FaceRecognizer()
    memory     = TrackMemory()
    builder    = ResultBuilder()
    logger.info("Models sẵn sàng.")

    # Log file
    log_file = None
    if save_log:
        os.makedirs(LOGS_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(LOGS_DIR, f"rtsp_test_{ts}.jsonl")
        log_file = open(log_path, "w", encoding="utf-8")
        logger.info(f"Lưu log: {log_path}")

    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

    # Mở stream — dùng TCP để tránh packet-loss gây H.264 decode error
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error(f"Không mở được RTSP: {rtsp_url}")
        return

    # ── VideoWriter (ghi video overlay) ────────────────────────────────────
    video_writer = None
    if save_video:
        videos_dir = os.path.join(os.path.dirname(LOGS_DIR), "videos")
        os.makedirs(videos_dir, exist_ok=True)
        ts_str     = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(videos_dir, f"rtsp_test_{ts_str}.mp4")
        src_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        src_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, src_fps, (src_w, src_h))
        logger.info(f"[Video] Ghi ra: {video_path}  ({src_w}x{src_h} @ {src_fps:.1f}fps)")
    # ── Reader thread: liên tục drain buffer camera, giữ frame mới nhất ────────────
    _stop_reader = threading.Event()
    _frame_queue = queue.Queue(maxsize=1)

    def _reader(cap_, q, stop):
        while not stop.is_set():
            ret, frm = cap_.read()
            if not ret:
                q.put(None)  # sentinel: stream ended
                break
            # Giữ frame mới nhất — xóa frame cũ nếu queue đầy
            if q.full():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            q.put(frm)

    _reader_thread = threading.Thread(
        target=_reader, args=(cap, _frame_queue, _stop_reader), daemon=True
    )
    _reader_thread.start()
    # ── State ─────────────────────────────────────────────────────────────────
    frame_id   = 0
    start_ts   = time.time()
    fps_counter = 0
    fps_val    = 0.0
    fps_timer  = time.time()

    logger.info("Bắt đầu stream. Nhấn [q] thoát | [s] snapshot | [r] reload DB")

    while True:
        # Bước 1: Lấy frame mới nhất từ reader thread
        try:
            frame = _frame_queue.get(timeout=5.0)
        except queue.Empty:
            logger.warning("Timeout đọc frame — mất kết nối?")
            break
        if frame is None:
            logger.warning("Mất kết nối hoặc hết stream.")
            break
        timestamp = round(time.time() - start_ts, 3)

        # Bước 2+3: Detect person + Update tracker (1 YOLO pass duy nhất)
        tracks = tracker.update([], frame)

        # Bước 4a: 1 InsightFace GPU pass cho toàn frame (detect + embed tất cả face)
        # Chỉ chạy nếu có ít nhất 1 track cần recognize — tránh waste GPU
        _needs_recog = any(
            _should_recognize(memory.get(t["track_id"]), frame_id)
            for t in tracks
        )
        _all_faces = recognizer.detect_and_embed(frame) if _needs_recog else []

        detections = []
        active_ids = set()

        for track in tracks:
            track_id   = track["track_id"]
            person_box = track["bbox"]
            active_ids.add(track_id)

            x1, y1, x2, y2 = map(int, person_box)
            if x2 <= x1 or y2 <= y1:
                continue

            cached   = memory.get(track_id)
            face_box = None
            name = score = None
            status = "skip"
            source = "cache"

            # Bước 4b: Kiểm tra chính sách (mục 7) — dùng face đã detect sẵn
            if _should_recognize(cached, frame_id):
                face_obj = _find_face_in_box(_all_faces, x1, y1, x2, y2)

                if face_obj is None:
                    # Không detect được mặt → no_face, lấy cache nếu có
                    status = "no_face"
                    memory.update(track_id,
                                  cached["name"] if cached else None,
                                  cached["score"] if cached else 0.0,
                                  "no_face", frame_id)
                    if cached:
                        name, score = cached["name"], cached["score"]
                        source = "cache"
                elif face_obj.det_score < FACE_DET_SCORE:
                    # Face xấu (mờ, góc) → skip, giữ cache, retry sau 3 frame
                    status = "bad_face"
                    memory.update(track_id,
                                  cached["name"] if cached else None,
                                  cached["score"] if cached else 0.0,
                                  "bad_face", frame_id)
                    if cached:
                        name, score = cached["name"], cached["score"]
                        source = "cache"
                else:
                    # Face hợp lệ — embedding đã có, match DB bằng numpy (không GPU)
                    fb = face_obj.bbox.astype(int)
                    face_box = [int(fb[0]), int(fb[1]), int(fb[2]), int(fb[3])]

                    res = recognizer.match_embedding(face_obj.normed_embedding)
                    source = "recomputed"

                    # Bước 5: Cập nhật track_memory (voting) — đọc lại kết quả
                    # đã được voting để hiển thị ổn định, tránh nhảy loạn
                    memory.update(track_id, res["name"], res["score"], res["status"], frame_id)
                    stable = memory.get(track_id)
                    name   = stable["name"]
                    score  = stable["score"]
                    status = stable["status"]
            else:
                # Chưa đến lúc refresh → dùng cache
                if cached:
                    name   = cached["name"]
                    score  = cached["score"]
                    status = cached["status"]
                    source = "cache"

            detections.append({
                "track_id":    track_id,
                "person_bbox": list(person_box),
                "face_bbox":   [round(v, 1) for v in face_box] if face_box else None,
                "name":        name,
                "score":       round(float(score), 4) if score else None,
                "status":      status,
                "source":      source,
            })

        # Dọn track đã rời khỏi scene
        memory.cleanup(active_ids)

        # Bước 6: Build frame result
        result = builder.build(
            frame_id=frame_id,
            timestamp=timestamp,
            camera_id="rtsp_test",
            detections=detections,
        )

        # Bước 7: Render + Log
        if log_file:
            log_file.write(json.dumps(result, ensure_ascii=False) + "\n")

        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_val     = fps_counter / (time.time() - fps_timer)
            fps_counter = 0
            fps_timer   = time.time()

        display = draw_frame(frame.copy(), result)
        if show_fps:
            _put_text_bg(display, f"FPS: {fps_val:.1f}", (10, 30), (0, 255, 255), font_scale=0.7)
        _put_text_bg(
            display,
            f"Frame:{frame_id} | Tracks:{len(tracks)} | Thresh:{RECOGNITION_SIM_THRESHOLD} | FaceDet:{FACE_DET_SCORE}",
            (10, display.shape[0] - 10), (200, 200, 200), font_scale=0.45,
        )
        cv2.imshow("RTSP - Face Recognition", display)

        # Ghi frame có overlay vào video
        if video_writer is not None:
            video_writer.write(display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            snap_path = os.path.join(
                SNAPSHOTS_DIR,
                f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}_f{frame_id}.jpg",
            )
            cv2.imwrite(snap_path, display)
            logger.info(f"[Snapshot] {snap_path}")
        elif key == ord("r"):
            recognizer.reload_db()
            logger.info("[Reload] Face DB đã được reload.")

        frame_id += 1

    _stop_reader.set()
    _reader_thread.join(timeout=3.0)
    cap.release()
    if video_writer is not None:
        video_writer.release()
        logger.info("[Video] Đã lưu video.")
    cv2.destroyAllWindows()
    if log_file:
        log_file.close()
    logger.info(f"Kết thúc. Tổng frame: {frame_id}")


# ─── Cấu hình chạy trực tiếp ─────────────────────────────────────────────────
RTSP_URL       = "rtsp://admin:Aipt2025@@10.0.99.27:554/1"
SHOW_FPS       = True    # Hiện FPS trên màn hình
SAVE_LOG       = True    # Lưu kết quả ra file JSONL trong output/logs/
SAVE_VIDEO     = True    # Ghi video có overlay ra output/videos/
THRESHOLD      = None    # None = dùng config.py (0.45); đặt float để override
FACE_DET_SCORE = 0.5     # Chất lượng face detect tối thiểu — thấp hơn → bỏ qua (face xấu)

# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if THRESHOLD is not None:
        import app.config as cfg
        cfg.RECOGNITION_SIM_THRESHOLD = THRESHOLD
        logger.info(f"Override threshold → {THRESHOLD}")

    run(
        rtsp_url=RTSP_URL,
        show_fps=SHOW_FPS,
        save_log=SAVE_LOG,
        save_video=SAVE_VIDEO,
    )
