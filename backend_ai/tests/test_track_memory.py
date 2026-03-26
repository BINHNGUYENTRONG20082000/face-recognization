"""
test_track_memory.py
"""
from app.services.track_memory import TrackMemory


def test_initial_get_none():
    mem = TrackMemory()
    assert mem.get(99) is None


def test_update_and_get():
    mem = TrackMemory()
    mem.update(1, "Nguyen Van A", 0.9, "recognized", frame_id=10)
    state = mem.get(1)
    assert state is not None
    assert state["status"] == "recognized"
    assert state["last_recognized_frame"] == 10


def test_cleanup():
    mem = TrackMemory()
    mem.update(1, "A", 0.8, "recognized", 0)
    mem.update(2, None, 0.0, "unknown", 0)
    mem.cleanup(active_track_ids={1})
    assert mem.get(2) is None
    assert mem.get(1) is not None
