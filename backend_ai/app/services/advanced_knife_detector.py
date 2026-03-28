"""
advanced_knife_detector.py - Phát hiện dao nâng cao với logging, alerting, thống kê
Version nâng cao của KnifeDetector
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import numpy as np
from ultralytics import YOLO

from app.config import LOGS_DIR
from .knife_detector import KnifeDetector

logger = logging.getLogger(__name__)


class AdvancedKnifeDetector(KnifeDetector):
    """
    Detector dao nâng cao với:
    - Alerting & logging
    - Thống kê, metrics
    - Tracking trạng thái
    - Custom model support
    """
    
    def __init__(self, custom_model_path: str = None, enable_logging: bool = True):
        """
        Args:
            custom_model_path: Đường dẫn đến custom knife detection model
            enable_logging: Bật ghi log JSON
        """
        super().__init__()
        
        self.enable_logging = enable_logging
        self.custom_model = None
        self.stats = {
            "total_frames": 0,
            "frames_with_knife": 0,
            "frames_with_person_and_knife": 0,
            "total_knives_detected": 0,
            "total_persons_with_knife": 0,
        }
        self.alerts = []
        
        # Load custom model nếu có
        if custom_model_path:
            try:
                self.custom_model = YOLO(custom_model_path)
                logger.info(f"[AdvancedKnifeDetector] Custom model đã load: {custom_model_path}")
            except Exception as e:
                logger.warning(f"[AdvancedKnifeDetector] Không load custom model: {e}")
        
        # Setup logging file
        if enable_logging:
            self.log_file = Path(LOGS_DIR) / f"knife_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    def detect_with_stats(self, frame: np.ndarray, frame_id: int = None) -> dict:
        """
        Detect dao với thống kê và logging
        
        Args:
            frame: Input frame
            frame_id: ID của frame (tùy chọn)
        
        Returns:
            dict với detection results + stats
        """
        self.stats["total_frames"] += 1
        
        # Detect từ default model
        result = self.detect_persons_and_knives(frame)
        
        # Nếu có custom model, thử detect thêm từ custom model
        if self.custom_model:
            custom_result = self._detect_with_custom_model(frame)
            result = self._merge_detection_results(result, custom_result)
        
        # Update stats
        if len(result["knives"]) > 0:
            self.stats["frames_with_knife"] += 1
            self.stats["total_knives_detected"] += len(result["knives"])
        
        if len(result["persons_with_knife"]) > 0:
            self.stats["frames_with_person_and_knife"] += 1
            self.stats["total_persons_with_knife"] += len(result["persons_with_knife"])
        
        # Create alert nếu phát hiện người cầm dao
        if result["persons_with_knife"]:
            alert = self._create_alert(frame_id, result)
            self.alerts.append(alert)
            
            # Log to file
            if self.enable_logging:
                self._log_alert(alert)
            
            logger.warning(f"[ALERT] Frame {frame_id}: {len(result['persons_with_knife'])} người cầm dao!")
        
        # Thêm stats vào result
        result["stats"] = self.stats.copy()
        result["frame_id"] = frame_id
        result["timestamp"] = datetime.now().isoformat()
        
        return result

    # ──────────────────────────────────────────────────────────────────────────
    def _detect_with_custom_model(self, frame: np.ndarray) -> dict:
        """Detect sử dụng custom knife model"""
        try:
            results = self.custom_model.predict(frame, verbose=False)
            
            knives = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    score = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    knives.append({
                        "bbox": [x1, y1, x2, y2],
                        "score": score,
                        "class": class_id,
                        "class_name": f"knife_custom",
                        "source": "custom_model"
                    })
            
            return {
                "persons": [],
                "knives": knives,
                "persons_with_knife": []
            }
        except Exception as e:
            logger.error(f"[AdvancedKnifeDetector] Custom model detection error: {e}")
            return {"persons": [], "knives": [], "persons_with_knife": []}

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _merge_detection_results(result1: dict, result2: dict) -> dict:
        """Merge kết quả từ 2 models"""
        # Merge knives từ cả 2 models (có thể có duplicate, nhưng thế là OK)
        merged = result1.copy()
        merged["knives"].extend(result2["knives"])
        
        # Re-match person với knife sau khi merge
        if merged["knives"]:
            persons_with_knife = merged.get("persons_with_knife", [])
            # Có thể thêm logic phức tạp hơn ở đây nếu cần
        
        return merged

    # ──────────────────────────────────────────────────────────────────────────
    def _create_alert(self, frame_id: int, result: dict) -> dict:
        """Tạo object alert"""
        return {
            "timestamp": datetime.now().isoformat(),
            "frame_id": frame_id,
            "type": "PERSON_WITH_KNIFE",
            "severity": "HIGH",
            "count_persons_with_knife": len(result["persons_with_knife"]),
            "details": result["persons_with_knife"]
        }

    # ──────────────────────────────────────────────────────────────────────────
    def _log_alert(self, alert: dict):
        """Ghi alert vào log file - JSONL format"""
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(alert, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"[AdvancedKnifeDetector] Error writing log: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    def get_stats(self) -> dict:
        """Lấy thống kê hiện tại"""
        stats = self.stats.copy()
        stats["total_alerts"] = len(self.alerts)
        stats["avg_knife_confidence"] = self._calc_avg_confidence()
        return stats

    # ──────────────────────────────────────────────────────────────────────────
    def _calc_avg_confidence(self) -> float:
        """Tính confidence trung bình"""
        if not self.alerts:
            return 0.0
        
        all_scores = []
        for alert in self.alerts:
            for person_with_knife in alert.get("details", []):
                all_scores.append(person_with_knife.get("knife_score", 0))
        
        return np.mean(all_scores) if all_scores else 0.0

    # ──────────────────────────────────────────────────────────────────────────
    def save_stats_report(self, output_path: str = None) -> str:
        """Lưu report thống kê"""
        if output_path is None:
            output_path = str(Path(LOGS_DIR) / f"knife_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "statistics": self.get_stats(),
            "alerts_count": len(self.alerts),
            "alerts_sample": self.alerts[-10:] if len(self.alerts) > 10 else self.alerts
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[AdvancedKnifeDetector] Stats report đã lưu: {output_path}")
        return output_path

    # ──────────────────────────────────────────────────────────────────────────
    def clear_stats(self):
        """Reset thống kê"""
        self.stats = {
            "total_frames": 0,
            "frames_with_knife": 0,
            "frames_with_person_and_knife": 0,
            "total_knives_detected": 0,
            "total_persons_with_knife": 0,
        }
        self.alerts = []
