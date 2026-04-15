"""
build_face_embeddings.py - Script offline xây dựng face database
Luồng: ảnh nhân viên → detect + align → embedding → lưu embeddings.npy + names.json
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import logging
import os
import numpy as np
from insightface.app import FaceAnalysis
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Đường dẫn mặc định ───────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
IMAGES_DIR      = str(BASE_DIR / "input" / "employee_images")
EMBEDDINGS_OUT  = str(BASE_DIR / "input" / "face_features" / "embeddings.npy")
NAMES_OUT       = str(BASE_DIR / "input" / "face_features" / "names.json")
MODEL_NAME      = "buffalo_l"
DET_SIZE        = (640, 640)
CTX_ID          = int(os.getenv("INSIGHTFACE_CTX_ID", "0"))


import re
import cv2

# ─── Bảng chuyển tiếng Việt → ASCII ──────────────────────────────────────────
_VI_MAP = str.maketrans(
    'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'
    'ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ'
    'ĐƯỪỨỰỬỮƠỜỚỢỞỠÂÊÔÃẼĨÕŨỸĂÀÁẠẢÈÉẸẺÌÍỊỈÒÓỌỎÙÚỤỦ',
    'aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyyd'
    'AAAAAAAAAAAAAAAAAEEEEEEEEEEEIIIIIOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYD'
    'DUUUUUUOOOOOOAEOAEIOUYAAAAAEEEEIIIIOOOOUUUU'
)


def _remove_accents(s: str) -> str:
    return s.translate(_VI_MAP)


def _extract_name_from_filename(stem: str) -> str:
    """
    Trích tên nhân viên từ tên file.
    Ví dụ: "Nguyen Van A 1"  → "Nguyen Van A"
            "Nguyen Van A"    → "Nguyen Van A"
    Loại bỏ phần số thứ tự ở cuối (` 1`, ` 2`, ...).
    """
    return re.sub(r"\s+\d+$", "", stem).strip()


def _collect_flat(images_path: Path) -> dict[str, list[Path]]:
    """
    Chế độ flat: ảnh để thẳng trong thư mục.
    Tên file: "<Tên nhân viên> <số>.jpg"
    Gom các ảnh có cùng tên nhân viên lại.
    """
    name_to_files: dict[str, list[Path]] = {}
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for img_path in sorted(images_path.iterdir()):
        if img_path.is_file() and img_path.suffix.lower() in img_exts:
            name = _extract_name_from_filename(img_path.stem)
            name_to_files.setdefault(name, []).append(img_path)
    return name_to_files


def _collect_subdir(images_path: Path) -> dict[str, list[Path]]:
    """
    Chế độ thư mục con: mỗi thư mục = một nhân viên.
    input/employee_images/
        Nguyen_Van_A/
            img1.jpg
    """
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    name_to_files: dict[str, list[Path]] = {}
    for person_dir in sorted(images_path.iterdir()):
        if person_dir.is_dir():
            files = [f for f in person_dir.iterdir() if f.suffix.lower() in img_exts]
            if files:
                name_to_files[person_dir.name] = sorted(files)
    return name_to_files


def build(images_dir: str, embeddings_out: str, names_out: str):
    app = FaceAnalysis(name=MODEL_NAME, allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=CTX_ID, det_size=DET_SIZE)

    embeddings: list[np.ndarray] = []
    names:      list[str]        = []
    skipped:    list[str]        = []

    images_path = Path(images_dir)

    # Tự động phát hiện chế độ: subdirectory hay flat
    has_subdirs = any(d.is_dir() for d in images_path.iterdir())
    if has_subdirs:
        logger.info("[Mode] Thư mục con (mỗi thư mục = 1 nhân viên)")
        name_to_files = _collect_subdir(images_path)
    else:
        logger.info("[Mode] Flat (tên file = tên nhân viên + số)")
        name_to_files = _collect_flat(images_path)

    logger.info(f"Tìm thấy {len(name_to_files)} nhân viên trong {images_dir}")

    for name, img_files in name_to_files.items():
        person_embeds: list[np.ndarray] = []

        for img_path in img_files:
            # cv2.imread không hỗ trợ unicode path trên Windows
            # → đọc bytes trước, decode sau
            img_array = np.fromfile(str(img_path), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning(f"  Không đọc được ảnh: {img_path.name}")
                skipped.append(str(img_path))
                continue
            faces = app.get(img)
            if not faces:
                logger.warning(f"  Không detect face: {img_path.name}")
                skipped.append(str(img_path))
                continue
            best = max(faces, key=lambda f: f.det_score)
            person_embeds.append(best.normed_embedding)

        if not person_embeds:
            logger.warning(f"[SKIP] {name}: không có embedding hợp lệ")
            continue

        avg_embed = np.mean(person_embeds, axis=0)
        avg_embed /= np.linalg.norm(avg_embed) + 1e-8
        embeddings.append(avg_embed)
        names.append(_remove_accents(name))   # lưu tên ASCII để hiển thị trên frame
        logger.info(f"  [{name}] {len(person_embeds)} ảnh → 1 embedding trung bình")

    os.makedirs(os.path.dirname(embeddings_out), exist_ok=True)
    np.save(embeddings_out, np.array(embeddings))
    with open(names_out, "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=2)

    logger.info(f"\nXong! {len(names)} nhân viên → {embeddings_out}")
    if skipped:
        logger.info(f"Bỏ qua {len(skipped)} ảnh không detect được face.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build face embeddings database")
    parser.add_argument("--images-dir",     default=IMAGES_DIR)
    parser.add_argument("--embeddings-out", default=EMBEDDINGS_OUT)
    parser.add_argument("--names-out",      default=NAMES_OUT)
    args = parser.parse_args()

    build(args.images_dir, args.embeddings_out, args.names_out)
