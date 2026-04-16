"""weapon_detector.py - Phát hiện vũ khí trong ảnh

Usage:
    detector = WeaponDetector()
    detections = detector.detect(frame, conf_threshold=0.3)
    annotated = detector.annotate(frame, detections)
"""

import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Bootstrap sys.path để chạy file độc lập
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]  # backend_ai/
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from app.config import YOLO_MODEL_WEAPON_PATH, OUTPUT_DIR
from app.core.exceptions import ModelNotLoadedError

logger = logging.getLogger(__name__)


class WeaponDetector:
    """Phát hiện vũ khí (knife, gun, ...) trong ảnh."""

    def __init__(self, conf_threshold: float = 0.3):
        """
        Args:
            conf_threshold: Ngưỡng confidence mặc định (0-1)
        """
        try:
            self._model = YOLO(YOLO_MODEL_WEAPON_PATH)
            self._conf_threshold = conf_threshold
            
            # Lấy class names từ model
            self._class_names = self._model.names  # dict {class_id: class_name}
            
            logger.info(
                "[WeaponDetector] Đã load model: %s",
                Path(YOLO_MODEL_WEAPON_PATH).name,
            )
            logger.info(
                "[WeaponDetector] Classes: %s",
                ", ".join(f"{k}={v}" for k, v in self._class_names.items()),
            )
        except Exception as e:
            raise ModelNotLoadedError(
                f"Không load được model weapon YOLO: {e}"
            ) from e

    @property
    def class_names(self) -> dict[int, str]:
        """Mapping class_id → class_name."""
        return self._class_names

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float | None = None,
    ) -> list[dict]:
        """
        Phát hiện vũ khí trong ảnh.

        Args:
            image: numpy BGR image (frame hoặc person crop)
            conf_threshold: Ngưỡng confidence (dùng self._conf_threshold nếu None)

        Returns:
            list[{
                "bbox": [x1, y1, x2, y2],
                "confidence": float,
                "class_id": int,
                "class_name": str,
            }]
            Sắp xếp theo confidence giảm dần.
        """
        threshold = conf_threshold if conf_threshold is not None else self._conf_threshold

        results = self._model(image, conf=threshold, verbose=False)[0]
        detections = []

        if results.boxes is None or len(results.boxes) == 0:
            return detections

        for *box, conf, cls in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(conf),
                    "class_id": class_id,
                    "class_name": self._class_names.get(class_id, "unknown"),
                }
            )

        # Sắp xếp theo confidence giảm dần
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        return detections

    def annotate(
        self,
        image: np.ndarray,
        detections: list[dict],
        color: tuple[int, int, int] | None = None,  # None = auto color
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Vẽ bounding box + label lên ảnh.

        Args:
            image: BGR image
            detections: Kết quả từ detect()
            color: Màu bbox (BGR). None = tự động (đỏ cho vũ khí, xanh cho objects khác)
            thickness: Độ dày viền

        Returns:
            Ảnh đã vẽ (copy của input)
        """
        annotated = image.copy()
        dangerous_classes = {"knife", "gun", "weapon"}

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            class_name = det["class_name"]

            # Auto color: red for weapons, green for others
            if color is None:
                is_dangerous = class_name.lower() in dangerous_classes
                box_color = (0, 0, 255) if is_dangerous else (0, 255, 0)  # BGR
            else:
                box_color = color

            # Draw bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

            # Draw label background
            label = f"{class_name} {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w, y1),
                box_color,
                -1,
            )

            # Draw text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated

    def filter_dangerous_weapons(
        self, detections: list[dict]
    ) -> list[dict]:
        """
        Lọc chỉ các vũ khí nguy hiểm (Knife, Gun, Weapon).
        
        Args:
            detections: Kết quả từ detect()
            
        Returns:
            List detections chỉ chứa vũ khí nguy hiểm
        """
        dangerous_classes = {"knife", "gun", "weapon"}
        return [
            det for det in detections
            if det["class_name"].lower() in dangerous_classes
        ]

    def get_statistics(self, detections: list[dict]) -> dict[str, int]:
        """
        Thống kê số lượng từng loại object phát hiện được.
        
        Args:
            detections: Kết quả từ detect()
            
        Returns:
            dict {class_name: count}
        """
        stats = {}
        for det in detections:
            class_name = det["class_name"]
            stats[class_name] = stats.get(class_name, 0) + 1
        return stats

    def has_dangerous_weapon(self, detections: list[dict]) -> bool:
        """
        Kiểm tra có vũ khí nguy hiểm không.
        
        Args:
            detections: Kết quả từ detect()
            
        Returns:
            True nếu phát hiện Knife, Gun hoặc Weapon
        """
        return len(self.filter_dangerous_weapons(detections)) > 0

    def _run_demo(self):
        """Demo: Detect weapons trên ảnh test."""
        test_image = r"E:\face recognition\data test\(136) Bodycam Footage Shows Florida Woman Stab Cop Ahead of Shooting - YouTube_frame_002960.jpg"
        frame = cv2.imread(str(test_image))
        if frame is None:
            print(f"[DEMO] Lỗi đọc ảnh: {test_image}")
            return

        print(f"[DEMO] Kích thước ảnh: {frame.shape[:2]}")

        # Detect
        detections = self.detect(frame)
        print(f"[DEMO] Tìm thấy {len(detections)} objects:")
        for i, det in enumerate(detections, 1):
            print(
                f"  {i}. {det['class_name']:20s} conf={det['confidence']:.3f} "
                f"bbox={det['bbox']}"
            )

        # Statistics
        stats = self.get_statistics(detections)
        print(f"\n[DEMO] Thống kê:")
        for class_name, count in sorted(stats.items()):
            print(f"  - {class_name}: {count}")

        # Dangerous weapon check
        dangerous = self.filter_dangerous_weapons(detections)
        if dangerous:
            print(f"\n[DEMO] ⚠️  CẢNH BÁO: Phát hiện {len(dangerous)} vũ khí nguy hiểm!")
            for det in dangerous:
                print(f"  - {det['class_name']} (confidence: {det['confidence']:.3f})")
        else:
            print(f"\n[DEMO] ✓ Không phát hiện vũ khí nguy hiểm.")

        # Annotate
        annotated = self.annotate(frame, detections)

        # Resize để hiển thị (nếu quá lớn)
        h, w = annotated.shape[:2]
        max_dim = 1200
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            annotated = cv2.resize(annotated, (new_w, new_h))
            print(f"[DEMO] Resize preview: {w}x{h} → {new_w}x{new_h}")

        # Hiển thị hoặc lưu file
        try:
            cv2.imshow("Weapon Detection", annotated)
            print("[DEMO] Nhấn phím bất kỳ để đóng cửa sổ...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            # Fallback: lưu file
            output_path = Path(OUTPUT_DIR) / "weapon_detection_demo.jpg"
            cv2.imwrite(str(output_path), annotated)
            print(f"[DEMO] Đã lưu kết quả: {output_path}")
            if os.name == "nt":  # Windows
                os.startfile(output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    detector = WeaponDetector(conf_threshold=0.25)
    detector._run_demo()
