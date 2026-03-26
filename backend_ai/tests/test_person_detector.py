"""
test_person_detector.py
"""
import numpy as np
import pytest

from app.services.person_detector import PersonDetector


def test_detect_returns_list():
    detector = PersonDetector()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(dummy_frame)
    assert isinstance(result, list)


def test_detect_structure():
    detector = PersonDetector()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(dummy_frame)
    for item in result:
        assert "bbox" in item
        assert "score" in item
        assert len(item["bbox"]) == 4
