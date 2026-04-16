"""
test_weapon_detector.py - Test đầy đủ các tính năng WeaponDetector

Chạy từ thư mục backend_ai:
    python -m pytest tests/test_weapon_detector.py -v
    hoặc
    python tests/test_weapon_detector.py
"""

import sys
from pathlib import Path

# Bootstrap sys.path
project_root = Path(__file__).resolve().parent.parent  # backend_ai/
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging
import cv2
import numpy as np
from app.services.weapon_detector import WeaponDetector
from app.config import OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def test_initialization():
    """Test 1: Khởi tạo detector."""
    print("\n" + "="*60)
    print("TEST 1: KHỞI TẠO DETECTOR")
    print("="*60)
    
    detector = WeaponDetector(conf_threshold=0.25)
    print(f"✓ Detector đã khởi tạo thành công")
    print(f"✓ Confidence threshold: {detector._conf_threshold}")
    print(f"✓ Số classes: {len(detector.class_names)}")
    print(f"✓ Class names:")
    for class_id, class_name in detector.class_names.items():
        print(f"    {class_id}: {class_name}")
    
    return detector


def test_detection_on_blank_image(detector: WeaponDetector):
    """Test 2: Detect trên ảnh trống (phải trả về [])."""
    print("\n" + "="*60)
    print("TEST 2: DETECTION TRÊN ẢNH TRỐNG")
    print("="*60)
    
    blank_image = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = detector.detect(blank_image, conf_threshold=0.3)
    
    assert len(detections) == 0, "Ảnh trống phải trả về 0 detections"
    print(f"✓ Ảnh trống → {len(detections)} detections (OK)")


def test_statistics(detector: WeaponDetector):
    """Test 3: Test get_statistics()."""
    print("\n" + "="*60)
    print("TEST 3: THỐNG KÊ")
    print("="*60)
    
    # Mock detections
    mock_detections = [
        {"bbox": [0, 0, 10, 10], "confidence": 0.9, "class_id": 0, "class_name": "Knife"},
        {"bbox": [0, 0, 10, 10], "confidence": 0.8, "class_id": 1, "class_name": "Gun"},
        {"bbox": [0, 0, 10, 10], "confidence": 0.7, "class_id": 0, "class_name": "Knife"},
        {"bbox": [0, 0, 10, 10], "confidence": 0.6, "class_id": 7, "class_name": "Person"},
    ]
    
    stats = detector.get_statistics(mock_detections)
    print(f"Mock detections: {len(mock_detections)}")
    print(f"Statistics:")
    for class_name, count in sorted(stats.items()):
        print(f"  - {class_name}: {count}")
    
    assert stats["Knife"] == 2, "Phải có 2 Knife"
    assert stats["Gun"] == 1, "Phải có 1 Gun"
    assert stats["Person"] == 1, "Phải có 1 Person"
    print(f"✓ Statistics chính xác")


def test_dangerous_weapons_filter(detector: WeaponDetector):
    """Test 4: Test filter_dangerous_weapons()."""
    print("\n" + "="*60)
    print("TEST 4: LỌC VŨ KHÍ NGUY HIỂM")
    print("="*60)
    
    mock_detections = [
        {"bbox": [0, 0, 10, 10], "confidence": 0.9, "class_id": 0, "class_name": "Knife"},
        {"bbox": [0, 0, 10, 10], "confidence": 0.8, "class_id": 1, "class_name": "Gun"},
        {"bbox": [0, 0, 10, 10], "confidence": 0.7, "class_id": 3, "class_name": "Smartphone"},
        {"bbox": [0, 0, 10, 10], "confidence": 0.6, "class_id": 7, "class_name": "Person"},
        {"bbox": [0, 0, 10, 10], "confidence": 0.85, "class_id": 8, "class_name": "Weapon"},
    ]
    
    dangerous = detector.filter_dangerous_weapons(mock_detections)
    print(f"Total detections: {len(mock_detections)}")
    print(f"Dangerous weapons: {len(dangerous)}")
    for det in dangerous:
        print(f"  - {det['class_name']} (conf: {det['confidence']:.2f})")
    
    assert len(dangerous) == 3, "Phải có 3 vũ khí nguy hiểm (Knife, Gun, Weapon)"
    print(f"✓ Filter đúng 3 vũ khí nguy hiểm")


def test_has_dangerous_weapon(detector: WeaponDetector):
    """Test 5: Test has_dangerous_weapon()."""
    print("\n" + "="*60)
    print("TEST 5: KIỂM TRA CÓ VŨ KHÍ NGUY HIỂM")
    print("="*60)
    
    # Case 1: Có vũ khí
    dangerous_detections = [
        {"bbox": [0, 0, 10, 10], "confidence": 0.9, "class_id": 0, "class_name": "Knife"},
        {"bbox": [0, 0, 10, 10], "confidence": 0.7, "class_id": 7, "class_name": "Person"},
    ]
    result1 = detector.has_dangerous_weapon(dangerous_detections)
    print(f"Case 1 (có Knife): {result1}")
    assert result1 is True, "Phải trả về True khi có vũ khí"
    
    # Case 2: Không có vũ khí
    safe_detections = [
        {"bbox": [0, 0, 10, 10], "confidence": 0.8, "class_id": 3, "class_name": "Smartphone"},
        {"bbox": [0, 0, 10, 10], "confidence": 0.7, "class_id": 7, "class_name": "Person"},
    ]
    result2 = detector.has_dangerous_weapon(safe_detections)
    print(f"Case 2 (chỉ có Smartphone + Person): {result2}")
    assert result2 is False, "Phải trả về False khi không có vũ khí"
    
    print(f"✓ has_dangerous_weapon() hoạt động đúng")


def test_annotate(detector: WeaponDetector):
    """Test 6: Test annotate() với color coding."""
    print("\n" + "="*60)
    print("TEST 6: ANNOTATE VỚI COLOR CODING")
    print("="*60)
    
    # Tạo ảnh test đơn giản
    test_image = np.ones((400, 600, 3), dtype=np.uint8) * 255  # white canvas
    
    mock_detections = [
        {"bbox": [50, 50, 150, 150], "confidence": 0.92, "class_id": 0, "class_name": "Knife"},
        {"bbox": [200, 50, 300, 150], "confidence": 0.88, "class_id": 1, "class_name": "Gun"},
        {"bbox": [350, 50, 450, 150], "confidence": 0.75, "class_id": 3, "class_name": "Smartphone"},
        {"bbox": [50, 200, 200, 350], "confidence": 0.82, "class_id": 7, "class_name": "Person"},
    ]
    
    # Auto color (đỏ cho weapon, xanh cho other)
    annotated_auto = detector.annotate(test_image, mock_detections)
    
    # Manual color
    annotated_manual = detector.annotate(test_image, mock_detections, color=(255, 0, 0))
    
    # Lưu file
    output_dir = Path(OUTPUT_DIR)
    output_auto = output_dir / "test_weapon_annotate_auto.jpg"
    output_manual = output_dir / "test_weapon_annotate_manual.jpg"
    
    cv2.imwrite(str(output_auto), annotated_auto)
    cv2.imwrite(str(output_manual), annotated_manual)
    
    print(f"✓ Annotate với auto color: {output_auto}")
    print(f"  - Knife, Gun → màu đỏ (vũ khí)")
    print(f"  - Smartphone, Person → màu xanh (an toàn)")
    print(f"✓ Annotate với manual color: {output_manual}")
    print(f"  - Tất cả → màu manual (255, 0, 0)")


def test_real_image(detector: WeaponDetector):
    """Test 7: Test trên ảnh thật (nếu có)."""
    print("\n" + "="*60)
    print("TEST 7: DETECTION TRÊN ẢNH THẬT")
    print("="*60)
    
    from app.config import WORKSPACE_DIR
    
    # Tìm ảnh test
    test_dirs = [
        WORKSPACE_DIR / "data test",
        WORKSPACE_DIR / "input",
    ]
    
    test_image_path = None
    for test_dir in test_dirs:
        if not test_dir.exists():
            continue
        for ext in [".jpg", ".jpeg", ".png"]:
            candidates = list(test_dir.rglob(f"*{ext}"))[:3]  # Lấy 3 ảnh đầu
            if candidates:
                test_image_path = candidates[0]
                break
        if test_image_path:
            break
    
    if test_image_path:
        print(f"Ảnh test: {test_image_path}")
        image = cv2.imread(str(test_image_path))
        
        if image is not None:
            detections = detector.detect(image, conf_threshold=0.25)
            print(f"Detections: {len(detections)}")
            
            if len(detections) > 0:
                stats = detector.get_statistics(detections)
                print(f"Statistics:")
                for class_name, count in sorted(stats.items()):
                    print(f"  - {class_name}: {count}")
                
                # Kiểm tra vũ khí nguy hiểm
                if detector.has_dangerous_weapon(detections):
                    print(f"⚠️  CẢNH BÁO: Phát hiện vũ khí nguy hiểm!")
                else:
                    print(f"✓ An toàn: Không có vũ khí nguy hiểm")
                
                # Lưu kết quả
                annotated = detector.annotate(image, detections)
                output_path = Path(OUTPUT_DIR) / "test_weapon_real_image.jpg"
                cv2.imwrite(str(output_path), annotated)
                print(f"✓ Đã lưu kết quả: {output_path}")
            else:
                print(f"✓ Không phát hiện object nào (có thể ảnh không chứa weapon)")
    else:
        print(f"⚠️  Không tìm thấy ảnh test, bỏ qua test này")


def main():
    """Chạy tất cả test cases."""
    print("\n" + "#"*60)
    print("#  TEST WEAPON DETECTOR")
    print("#"*60)
    
    try:
        # Test 1: Initialization
        detector = test_initialization()
        
        # Test 2: Blank image
        test_detection_on_blank_image(detector)
        
        # Test 3: Statistics
        test_statistics(detector)
        
        # Test 4: Dangerous weapons filter
        test_dangerous_weapons_filter(detector)
        
        # Test 5: Has dangerous weapon
        test_has_dangerous_weapon(detector)
        
        # Test 6: Annotate
        test_annotate(detector)
        
        # Test 7: Real image
        test_real_image(detector)
        
        print("\n" + "#"*60)
        print("#  ✓ TẤT CẢ TEST CASES ĐỀU PASS")
        print("#"*60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
