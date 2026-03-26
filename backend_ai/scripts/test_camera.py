"""
test_camera.py - Hiển thị đồng thời nhiều camera RTSP trên 1 cửa sổ dạng lưới
Phím tắt:
    q   - Thoát
    +/- - Tăng/giảm kích thước ô
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import time
import threading
import queue
import math
import numpy as np

# ─── Danh sách camera ──────────────────────────────────────────────────────────
RTSP_URLS = [
    "rtsp://admin:Aipt2025@@10.0.99.22:554/1",
    "rtsp://admin:Aipt2025@@10.0.99.28:554/1",
    "rtsp://admin:Aipt2025@@10.0.99.23:554/1",
    "rtsp://admin:Aipt2025@@10.0.99.24:554/1",
    "rtsp://admin:Aipt2025@@10.0.99.32:554/1",
    "rtsp://admin:Aipt2025@@10.0.99.31:554/1",
    "rtsp://admin:Aipt2025@@10.0.99.30:554/1",
    "rtsp://admin:Aipt2025@@10.0.99.25:554/1",
    "rtsp://admin:Aipt2025@@10.0.99.27:554/1",
    "rtsp://admin:Aipt2025@@10.0.99.26:554/1",
]

# ─── Cấu hình hiển thị ────────────────────────────────────────────────────────
CELL_W      = 640   # chiều rộng mỗi ô (pixel)
CELL_H      = 360   # chiều cao mỗi ô (pixel)
GRID_COLS   = 4     # số cột
RECONNECT_S = 5     # giây chờ trước khi reconnect khi mất stream

# Màu cho label
_COLOR_OK   = (0, 220, 0)
_COLOR_ERR  = (0, 0, 200)


# ─── Reader thread cho 1 camera ───────────────────────────────────────────────
def _camera_reader(idx: int, url: str, out_queue: queue.Queue, stop: threading.Event):
    """
    Chạy trong thread riêng. Luôn giữ frame mới nhất trong out_queue (maxsize=1).
    Tự động reconnect khi mất kết nối.
    """
    while not stop.is_set():
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            _put_latest(out_queue, None)          # báo lỗi kết nối
            stop.wait(RECONNECT_S)
            continue

        fps_t  = time.time()
        fps_c  = 0
        fps_v  = 0.0

        while not stop.is_set():
            ret, frame = cap.read()
            if not ret:
                break   # mất kết nối → vòng ngoài reconnect

            # Tính FPS thực
            fps_c += 1
            elapsed = time.time() - fps_t
            if elapsed >= 1.0:
                fps_v = fps_c / elapsed
                fps_c = 0
                fps_t = time.time()

            _put_latest(out_queue, (frame, fps_v))

        cap.release()
        if not stop.is_set():
            _put_latest(out_queue, None)          # báo reconnecting
            stop.wait(RECONNECT_S)


def _put_latest(q: queue.Queue, item):
    """Thay thế item cũ bằng item mới nhất (queue không tích lũy)."""
    if q.full():
        try:
            q.get_nowait()
        except queue.Empty:
            pass
    q.put(item)


# ─── Tạo ô lỗi / offline ──────────────────────────────────────────────────────
def _error_cell(label: str, w: int, h: int) -> np.ndarray:
    cell = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(cell, label, (10, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COLOR_ERR, 1, cv2.LINE_AA)
    return cell


# ─── Main ─────────────────────────────────────────────────────────────────────
def run():
    n_cams = len(RTSP_URLS)
    cols   = GRID_COLS
    rows   = math.ceil(n_cams / cols)

    cell_w, cell_h = CELL_W, CELL_H

    queues = [queue.Queue(maxsize=1) for _ in range(n_cams)]
    stop   = threading.Event()

    # Khởi động reader thread cho từng camera
    threads = []
    for i, url in enumerate(RTSP_URLS):
        t = threading.Thread(
            target=_camera_reader, args=(i, url, queues[i], stop), daemon=True
        )
        t.start()
        threads.append(t)

    print(f"Đang mở {n_cams} camera... nhấn [q] để thoát, [+/-] để đổi kích thước.")

    while True:
        cells = []
        for i in range(n_cams):
            ip_tag = RTSP_URLS[i].split("@")[-1].split(":")[0]   # lấy IP hiển thị
            label  = f"CAM {i+1} | {ip_tag}"

            try:
                item = queues[i].get_nowait()
            except queue.Empty:
                item = None   # thread chưa trả frame (đang kết nối)

            if item is None:
                cell = _error_cell(f"{label}  [connecting...]", cell_w, cell_h)
            else:
                frame, fps_v = item
                # Resize về kích thước ô
                cell = cv2.resize(frame, (cell_w, cell_h))
                # Overlay label + FPS
                cv2.rectangle(cell, (0, 0), (cell_w, 22), (0, 0, 0), -1)
                cv2.putText(cell, f"{label}  FPS:{fps_v:.1f}",
                            (6, 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, _COLOR_OK, 1, cv2.LINE_AA)

            cells.append(cell)

        # Padding nếu số camera không đủ lấp đầy lưới
        while len(cells) < rows * cols:
            cells.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

        # Ghép thành lưới
        grid_rows = []
        for r in range(rows):
            row_cells = cells[r * cols: r * cols + cols]
            grid_rows.append(np.hstack(row_cells))
        grid = np.vstack(grid_rows)

        cv2.imshow("Multi-Camera Viewer", grid)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("+") or key == ord("="):
            cell_w = min(cell_w + 80, 1280)
            cell_h = min(cell_h + 45, 720)
        elif key == ord("-"):
            cell_w = max(cell_w - 80, 160)
            cell_h = max(cell_h - 45, 90)

    stop.set()
    cv2.destroyAllWindows()
    print("Đã thoát.")


if __name__ == "__main__":
    run()
