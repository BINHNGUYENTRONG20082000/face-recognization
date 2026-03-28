"""
knife_detector.py - Phát hiện người cầm dao/kiếm trong frame bằng YOLO26
Output: list[{bbox, score, is_person_with_knife}]
"""

import logging
import numpy as np
from ultralytics import YOLO

from app.config import YOLO_MODEL_PATH, PERSON_CONF_THRESHOLD

logger = logging.getLogger(__name__)

# COCO class IDs
PERSON_CLASS_ID = 0   # COCO class 0 = person
KNIFE_CLASS_ID = 43   # COCO class 43 = knife (nếu trong dataset)


class KnifeDetector:
    """Phát hiện dao/kiếm và người cần dao"""
    
    def __init__(self):
        try:
            self._model = YOLO(YOLO_MODEL_PATH)
            logger.info(f"[KnifeDetector] Đã load model: {YOLO_MODEL_PATH}")
        except Exception as e:
            raise Exception(f"Không load được YOLO model: {e}") from e

    # ──────────────────────────────────────────────────────────────────────────
    def detect_knives(self, frame: np.ndarray) -> list[dict]:
        """
        Phát hiện các vật thể giống dao trong frame
        
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
        
        knives = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                # Lọc để tìm kiếm - có thể bao gồm các class liên quan
                # class 43 = knife (nếu có), class 41 = cup, class 39 = bottle, etc.
                # Có thể mở rộng thêm các class khác tùy theo nhu cầu
                if class_id == KNIFE_CLASS_ID or self._is_weapon_like(class_id):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    score = float(box.conf[0])
                    class_name = self._get_class_name(class_id)
                    
                    knives.append({
                        "bbox": [x1, y1, x2, y2],
                        "score": score,
                        "class": class_id,
                        "class_name": class_name
                    })
        
        return knives

    # ──────────────────────────────────────────────────────────────────────────
    def detect_persons_and_knives(self, frame: np.ndarray) -> dict:
        """
        Phát hiện cả người và dao trong frame
        
        Returns:
            {
                "persons": list[{"bbox": [x1,y1,x2,y2], "score": float}],
                "knives": list[{"bbox": [x1,y1,x2,y2], "score": float, "class_name": str}],
                "persons_with_knife": list[{"person_bbox": [...], "knife_bbox": [...]}]
            }
        """
        results = self._model.predict(
            frame,
            conf=PERSON_CONF_THRESHOLD,
            verbose=False,
        )
        
        persons = []
        knives = []
        
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
                elif class_id == KNIFE_CLASS_ID or self._is_weapon_like(class_id):
                    knives.append({
                        "bbox": [x1, y1, x2, y2],
                        "score": score,
                        "class": class_id,
                        "class_name": self._get_class_name(class_id)
                    })
        
        # Tìm người cầm dao (dao nằm trong bounding box của người)
        persons_with_knife = self._match_knife_to_person(persons, knives)
        
        return {
            "persons": persons,
            "knives": knives,
            "persons_with_knife": persons_with_knife
        }

    # ──────────────────────────────────────────────────────────────────────────
    def _match_knife_to_person(self, persons: list, knives: list) -> list:
        """
        Khớp dao với người - nếu dao nằm trong bounding box của người
        """
        matched = []
        
        for knife in knives:
            k_x1, k_y1, k_x2, k_y2 = knife["bbox"]
            knife_center_x = (k_x1 + k_x2) / 2
            knife_center_y = (k_y1 + k_y2) / 2
            
            for person in persons:
                p_x1, p_y1, p_x2, p_y2 = person["bbox"]
                
                # Kiểm tra xem tâm của dao có nằm trong bounding box của người không
                if p_x1 <= knife_center_x <= p_x2 and p_y1 <= knife_center_y <= p_y2:
                    matched.append({
                        "person_bbox": person["bbox"],
                        "person_score": person["score"],
                        "knife_bbox": knife["bbox"],
                        "knife_score": knife["score"],
                        "knife_class_name": knife["class_name"]
                    })
                    break
        
        return matched

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _is_weapon_like(class_id: int) -> bool:
        """
        Kiểm tra xem class ID có liên quan đến vũ khí không
        
        COCO class IDs liên quan đến vũ khí/nguy hiểm:
        - 43: knife
        - 44: microwave (false positive)
        - Có thể thêm các class khác tùy tinh chỉnh mô hình
        """
        weapon_classes = [43]  # knife
        return class_id in weapon_classes

    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _get_class_name(class_id: int) -> str:
        """
        Lấy tên class từ COCO dataset
        """
        coco_classes = {
            0: "person",
            43: "knife",
            39: "bottle",
            41: "cup",
            42: "fork",
            # Thêm các class khác nếu cần
        }
        return coco_classes.get(class_id, f"class_{class_id}")
