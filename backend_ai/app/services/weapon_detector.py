"""
weapon_detector.py - Phát hiện vũ khí (knife/gun) trong frame bằng YOLO.
Output: list[{bbox, score, class, class_name}]
"""

import logging
import numpy as np
from ultralytics import YOLO

from app.config import YOLO_MODEL_PATH, PERSON_CONF_THRESHOLD

logger = logging.getLogger(__name__)

# COCO class IDs
PERSON_CLASS_ID = 0   # COCO class 0 = person
WEAPON_CLASS_IDS = {43, 2, 5, 7}  # 43=knife, 2/5/7=gun placeholders; map lai theo dataset thuc te.


class WeaponDetector:
    """Phát hiện vũ khí (knife/gun) và match vào người."""
    
    def __init__(self):
        try:
            self._model = YOLO(YOLO_MODEL_PATH)
            logger.info(f"[WeaponDetector] Đã load model: {YOLO_MODEL_PATH}")
        except Exception as e:
            raise Exception(f"Không load được YOLO model: {e}") from e

    # ──────────────────────────────────────────────────────────────────────────
    def detect_weapons(self, frame: np.ndarray) -> list[dict]:
        """
        Phát hiện các vật thể vũ khí trong frame
        
        Returns:
            list[{
                "bbox": [x1, y1, x2, y2],
                "score": float,
                "class": int (YOLO class ID),
                "class_name": str
            }]
        """
        results = self._model.predict(
            frame,
            conf=PERSON_CONF_THRESHOLD,
            verbose=False,
        )
        
        weapons = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                if self._is_weapon_like(class_id):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    score = float(box.conf[0])
                    class_name = self._get_class_name(class_id)
                    
                    weapons.append({
                        "bbox": [x1, y1, x2, y2],
                        "score": score,
                        "class": class_id,
                        "class_name": class_name
                    })
        
        return weapons

    # ──────────────────────────────────────────────────────────────────────────
    def detect_persons_and_weapons(self, frame: np.ndarray) -> dict:
        """
        Phát hiện cả người và vũ khí trong frame
        
        Returns:
            {
                "persons": list[{"bbox": [x1,y1,x2,y2], "score": float}],
                "weapons": list[{"bbox": [x1,y1,x2,y2], "score": float, "class_name": str}],
                "persons_with_weapon": list[{"person_bbox": [...], "weapon_bbox": [...]}]
            }
        """
        results = self._model.predict(
            frame,
            conf=PERSON_CONF_THRESHOLD,
            verbose=False,
        )
        
        persons = []
        weapons = []
        
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])
                
                if class_id == PERSON_CLASS_ID:
                    persons.append({
                        "bbox": [x1, y1, x2, y2],
                        "score": score,
                        "class_id": class_id
                    })
                elif self._is_weapon_like(class_id):
                    weapons.append({
                        "bbox": [x1, y1, x2, y2],
                        "score": score,
                        "class": class_id,
                        "class_name": self._get_class_name(class_id)
                    })
        
        persons_with_weapon = self._match_weapon_to_person(persons, weapons)
        
        return {
            "persons": persons,
            "weapons": weapons,
            "persons_with_weapon": persons_with_weapon,
        }

    # ──────────────────────────────────────────────────────────────────────────
    def _match_weapon_to_person(self, persons: list, weapons: list) -> list:
        """
        Khớp vũ khí với người - nếu tâm vũ khí nằm trong bounding box của người.
        """
        matched = []
        
        for weapon in weapons:
            w_x1, w_y1, w_x2, w_y2 = weapon["bbox"]
            weapon_center_x = (w_x1 + w_x2) / 2
            weapon_center_y = (w_y1 + w_y2) / 2
            
            for person in persons:
                p_x1, p_y1, p_x2, p_y2 = person["bbox"]
                
                if p_x1 <= weapon_center_x <= p_x2 and p_y1 <= weapon_center_y <= p_y2:
                    matched.append({
                        "person_bbox": person["bbox"],
                        "person_score": person["score"],
                        "weapon_bbox": weapon["bbox"],
                        "weapon_score": weapon["score"],
                        "weapon_class_name": weapon["class_name"],
                    })
                    break
        
        return matched

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _is_weapon_like(class_id: int) -> bool:
        """
        Kiểm tra xem class ID có liên quan đến vũ khí không
        
        COCO class IDs liên quan đến vũ khí/nguy hiểm.
        """
        return class_id in WEAPON_CLASS_IDS

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _get_class_name(class_id: int) -> str:
        """
        Lấy tên class từ COCO dataset
        """
        coco_classes = {
            0: "person",
            43: "knife",
            2: "gun",
            5: "gun",
            7: "gun",
            39: "bottle",
            41: "cup",
            42: "fork",
            # Thêm các class khác nếu cần
        }
        return coco_classes.get(class_id, f"class_{class_id}")
