# Backend AI — Face Recognition Realtime

Hệ thống nhận diện nhân viên từ camera realtime theo kiến trúc **Person Tracking + Face Recognition**.

## Kiến trúc tổng thể

```
Camera (RTSP/Webcam)
    │
    ▼
StreamReader          ← đọc frame, chuẩn hóa
    │
    ▼
PersonDetector        ← YOLO26n detect người
    │
    ▼
PersonTracker         ← ByteTrack / BoT-SORT → track_id ổn định
    │
    ▼ (mỗi person ROI)
FaceDetector          ← InsightFace detect face trong ROI
    │
    ▼
FaceRecognizer        ← extract embedding → so khớp face DB
    │
    ▼
TrackMemory           ← voting nhiều frame + cache
    │
    ▼
ResultBuilder         ← frame result JSON chuẩn
    │
    ▼
LoggerService         ← JSONL + CSV log
```

## Cấu trúc thư mục

```
backend_ai/
├── app/
│   ├── main.py                  ← FastAPI app entry point
│   ├── config.py                ← Cấu hình toàn hệ thống
│   ├── core/
│   │   ├── pipeline.py          ← Orchestrator pipeline realtime
│   │   └── exceptions.py        ← Custom exceptions
│   ├── services/
│   │   ├── stream_reader.py     ← Đọc RTSP / webcam / file
│   │   ├── person_detector.py   ← YOLO26n person detect
│   │   ├── tracker.py           ← Ultralytics ByteTrack
│   │   ├── face_detector.py     ← InsightFace face detect
│   │   ├── face_recognizer.py   ← InsightFace embedding + match
│   │   ├── track_memory.py      ← Voting + cache theo track_id
│   │   ├── result_builder.py    ← Build JSON frame result
│   │   └── logger_service.py    ← JSONL + CSV output
│   └── api/
│       ├── routes.py            ← FastAPI routes
│       └── schemas.py           ← Pydantic schemas
├── scripts/
│   ├── build_face_embeddings.py ← Offline: build face DB
│   └── test_camera.py           ← Debug camera + detection
├── tests/
│   ├── test_person_detector.py
│   ├── test_result_builder.py
│   └── test_track_memory.py
├── models/
│   └── yolo26n.pt               ← YOLO model weights
├── input/
│   ├── employee_images/         ← Ảnh nhân viên (tổ chức theo tên thư mục)
│   └── face_features/
│       ├── embeddings.npy       ← Face embeddings DB
│       └── names.json           ← Tên nhân viên tương ứng
├── output/
│   ├── logs/                    ← JSONL + CSV kết quả
│   └── snapshots/               ← Ảnh chụp nếu cần
├── requirements.txt
├── .env.example
└── .gitignore
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Bước 1 — Xây dựng Face Database (offline)

Tổ chức ảnh theo cấu trúc:
```
input/employee_images/
    Nguyen_Van_A/
        img1.jpg
        img2.jpg
    Tran_Thi_B/
        img1.jpg
```

Sau đó chạy:
```bash
python scripts/build_face_embeddings.py
```

## Bước 2 — Chạy API server

```bash
python -m app.main
# hoặc
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Bước 3 — Khởi động pipeline

```http
POST /api/v1/cameras/cam_01/start
```

## Debug camera

```bash
python scripts/test_camera.py cam_01
```

## Chạy tests

```bash
pytest tests/
```

## Output schema mỗi frame

```json
{
  "frame_id": 120,
  "timestamp": 4.0,
  "camera_id": "cam_01",
  "num_detections": 1,
  "detections": [
    {
      "track_id": 5,
      "person_bbox": [60, 40, 220, 360],
      "face_bbox": [100, 80, 180, 190],
      "name": "Nguyen Van A",
      "score": 0.82,
      "status": "recognized",
      "source": "recomputed"
    }
  ]
}
```

## Stack công nghệ

| Component        | Thư viện                |
|------------------|-------------------------|
| Person Detection | YOLO26n (Ultralytics)   |
| Person Tracking  | ByteTrack / BoT-SORT    |
| Face Detection   | InsightFace             |
| Face Recognition | InsightFace (buffalo_l) |
| Inference GPU    | onnxruntime-gpu         |
| API              | FastAPI + Uvicorn       |
