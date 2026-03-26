"""
config.py - Cấu hình toàn bộ hệ thống backend_ai
"""

import os
from pathlib import Path

# ─── Đường dẫn gốc ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ─── Model ────────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH = str(BASE_DIR / "models" / "yolo26m.pt")

# ─── Face database ────────────────────────────────────────────────────────────
EMBEDDINGS_PATH = str(BASE_DIR / "input" / "face_features" / "embeddings.npy")
NAMES_PATH      = str(BASE_DIR / "input" / "face_features" / "names.json")
EMPLOYEE_IMAGES_DIR = str(BASE_DIR / "input" / "employee_images")

# ─── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR       = str(BASE_DIR / "output")
LOGS_DIR         = str(BASE_DIR / "output" / "logs")
SNAPSHOTS_DIR    = str(BASE_DIR / "output" / "snapshots")

# ─── Camera / Stream ──────────────────────────────────────────────────────────
CAMERA_SOURCES: dict[str, str] = {
    # "cam_01": "rtsp://user:pass@192.168.1.100:554/stream1",
    "cam_01": "0",   # webcam để debug
}

# ─── Detection / Tracking ─────────────────────────────────────────────────────
PERSON_CONF_THRESHOLD = 0.5
TRACKER_TYPE          = "botsort"   # bytetrack | botsort

# ─── Recognition ──────────────────────────────────────────────────────────────
RECOGNITION_SIM_THRESHOLD  = 0.3   # cosine similarity tối thiểu
RECOGNITION_REFRESH_FRAMES = 30     # refresh mỗi N frame với track đã nhận diện
UNKNOWN_RETRY_FRAMES       = 10      # retry nhanh với track unknown

# ─── InsightFace ──────────────────────────────────────────────────────────────
INSIGHTFACE_MODEL_NAME = "buffalo_l"
INSIGHTFACE_DET_SIZE   = (640, 640)

# ─── API ──────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
