"""
config.py - Cấu hình toàn bộ hệ thống backend_ai
"""

import json
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = BASE_DIR.parent.parent
ENV_FILE = BASE_DIR / ".env"


def _load_env_file(env_file: Path) -> None:
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _resolve_path(env_name: str, candidates: list[Path]) -> Path:
    override = os.getenv(env_name)
    if override:
        return Path(override)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _parse_camera_sources() -> dict[str, str]:
    raw_json = os.getenv("CAMERA_SOURCES_JSON")
    if raw_json:
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                return {str(key): str(value) for key, value in parsed.items()}
        except json.JSONDecodeError:
            pass

    env_sources = {
        key.removeprefix("CAMERA_SOURCE_").lower(): value
        for key, value in os.environ.items()
        if key.startswith("CAMERA_SOURCE_") and value
    }
    if env_sources:
        return env_sources

    legacy_sources = {
        key[4:-4].lower(): value
        for key, value in os.environ.items()
        if key.startswith("CAM_") and key.endswith("_URL") and value
    }
    if legacy_sources:
        return legacy_sources

    return {"cam_01": os.getenv("DEFAULT_CAMERA_SOURCE", "0")}


_load_env_file(ENV_FILE)

# ─── Đường dẫn runtime ────────────────────────────────────────────────────────
MODELS_DIR = _resolve_path("MODELS_DIR", [BASE_DIR / "models", WORKSPACE_DIR / "models"])
INPUT_DIR = _resolve_path("INPUT_DIR", [BASE_DIR / "input", WORKSPACE_DIR / "input"])
OUTPUT_DIR = _resolve_path("OUTPUT_DIR", [BASE_DIR / "output", WORKSPACE_DIR / "output"])

FACE_FEATURES_DIR = INPUT_DIR / "face_features"
EMPLOYEE_IMAGES_DIR = str(
    _resolve_path(
        "EMPLOYEE_IMAGES_DIR",
        [INPUT_DIR / "employee_images", WORKSPACE_DIR / "data test" / "Ảnh CBNV"],
    )
)

PERSON_MODEL_PATH = str(
    _resolve_path(
        "YOLO_MODEL_PATH",
        [
            MODELS_DIR / "yolo26m.pt",
            MODELS_DIR / "yolo26n.pt",
            WORKSPACE_DIR / "yolo26n.pt",
        ],
    )
)
YOLO_MODEL_WEAPON_PATH = str(
    _resolve_path(
        "YOLO_MODEL_WEAPON_PATH",
        [
            MODELS_DIR / "weapon_yolo26m.pt",
            MODELS_DIR / "best.pt",
            WORKSPACE_DIR / "best.pt",
        ],
    )
)

EMBEDDINGS_PATH = str(
    _resolve_path(
        "EMBEDDINGS_PATH",
        [FACE_FEATURES_DIR / "embeddings.npy", WORKSPACE_DIR / "input" / "face_features" / "embeddings.npy"],
    )
)
NAMES_PATH = str(
    _resolve_path(
        "NAMES_PATH",
        [FACE_FEATURES_DIR / "names.json", WORKSPACE_DIR / "input" / "face_features" / "names.json"],
    )
)

LOGS_DIR = str(Path(OUTPUT_DIR) / "logs")
SNAPSHOTS_DIR = str(Path(OUTPUT_DIR) / "snapshots")

for directory in (Path(MODELS_DIR), Path(INPUT_DIR), FACE_FEATURES_DIR, Path(OUTPUT_DIR), Path(LOGS_DIR), Path(SNAPSHOTS_DIR)):
    directory.mkdir(parents=True, exist_ok=True)

# ─── Camera / Stream ──────────────────────────────────────────────────────────
CAMERA_SOURCES: dict[str, str] = _parse_camera_sources()

# ─── Detection / Tracking ─────────────────────────────────────────────────────
PERSON_CONF_THRESHOLD = _env_float("PERSON_CONF_THRESHOLD", 0.5)
TRACKER_TYPE = os.getenv("TRACKER_TYPE", "botsort")

# ─── Recognition ──────────────────────────────────────────────────────────────
RECOGNITION_SIM_THRESHOLD = _env_float("RECOGNITION_SIM_THRESHOLD", 0.3)
RECOGNITION_REFRESH_FRAMES = _env_int("RECOGNITION_REFRESH_FRAMES", 30)
UNKNOWN_RETRY_FRAMES = _env_int("UNKNOWN_RETRY_FRAMES", 10)

# ─── InsightFace ──────────────────────────────────────────────────────────────
INSIGHTFACE_MODEL_NAME = os.getenv("INSIGHTFACE_MODEL_NAME", "buffalo_l")
INSIGHTFACE_CTX_ID = _env_int("INSIGHTFACE_CTX_ID", 0)
INSIGHTFACE_DET_SIZE = (
    _env_int("INSIGHTFACE_DET_WIDTH", 640),
    _env_int("INSIGHTFACE_DET_HEIGHT", 640),
)

# ─── FAISS Search ──────────────────────────────────────────────────────────────
_faiss_index = os.getenv("FAISS_INDEX_PATH")
_faiss_db = os.getenv("FAISS_META_DB_PATH")
FAISS_INDEX_PATH: Path | None = (
    Path(_faiss_index) if _faiss_index
    else (FACE_FEATURES_DIR / "face_index.faiss") if (FACE_FEATURES_DIR / "face_index.faiss").exists()
    else None
)
FAISS_META_DB_PATH: Path | None = (
    Path(_faiss_db) if _faiss_db
    else (FACE_FEATURES_DIR / "face_meta.db") if (FACE_FEATURES_DIR / "face_meta.db").exists()
    else None
)
FAISS_NPROBE = _env_int("FAISS_NPROBE", 8)
FAISS_USE_GPU = os.getenv("FAISS_USE_GPU", "0") == "1"
# HNSW: số lượng nodes mở rộng khi search (cao hơn = recall tốt hơn, chậm hơn chút)
FAISS_EF_SEARCH = _env_int("FAISS_EF_SEARCH", 128)

# ─── API ──────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = _env_int("API_PORT", 8000)
