"""
test_result_builder.py
"""
from app.services.result_builder import ResultBuilder


def test_build_schema():
    builder = ResultBuilder()
    result = builder.build(
        frame_id=10,
        timestamp=0.5,
        camera_id="cam_01",
        detections=[
            {
                "track_id": 1,
                "person_bbox": [0, 0, 100, 200],
                "face_bbox": None,
                "name": "Nguyen Van A",
                "score": 0.85,
                "status": "recognized",
                "source": "recomputed",
            }
        ],
    )
    assert result["frame_id"] == 10
    assert result["num_detections"] == 1
    assert result["detections"][0]["name"] == "Nguyen Van A"


def test_empty_detections():
    builder = ResultBuilder()
    result = builder.build(0, 0.0, "cam_01", [])
    assert result["num_detections"] == 0
    assert result["detections"] == []
