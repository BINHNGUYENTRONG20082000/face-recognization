"""
exceptions.py - Custom exceptions cho backend_ai
"""


class ModelNotLoadedError(RuntimeError):
    """Model AI chưa được khởi tạo."""


class EmbeddingDBNotFoundError(FileNotFoundError):
    """Không tìm thấy file embeddings / names."""


class StreamOpenError(IOError):
    """Không mở được camera stream."""


class FaceDetectionError(RuntimeError):
    """Lỗi trong quá trình detect face."""
