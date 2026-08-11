from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Iterable

import cv2
import numpy as np

from app.domain.entities import canonical_stream_id


class FrameStore:
    """Bounded source-frame cache used by ROI tasks outside the probe thread."""

    def __init__(self, capture_stream_ids: Iterable[str] = (), max_size: int = 128) -> None:
        self._capture_stream_ids = {canonical_stream_id(value) for value in capture_stream_ids}
        self._max_size = max(int(max_size), 1)
        self._lock = Lock()
        self._order: deque[tuple[str, int]] = deque()
        self._frames: dict[tuple[str, int], np.ndarray] = {}
        self._dropped = 0

    def should_capture(self, stream_id: str) -> bool:
        return canonical_stream_id(stream_id) in self._capture_stream_ids

    def put(self, stream_id: str, frame_id: int, frame: np.ndarray) -> None:
        key = (canonical_stream_id(stream_id), int(frame_id))
        image = np.asarray(frame)
        if image.ndim != 3:
            return
        with self._lock:
            if key in self._frames:
                self._frames[key] = image.copy()
                return
            while len(self._frames) >= self._max_size:
                old_key = self._order.popleft()
                if old_key in self._frames:
                    del self._frames[old_key]
                    self._dropped += 1
            self._frames[key] = image.copy()
            self._order.append(key)

    def get(self, stream_id: str, frame_id: int) -> np.ndarray | None:
        key = (canonical_stream_id(stream_id), int(frame_id))
        with self._lock:
            frame = self._frames.get(key)
            return None if frame is None else frame.copy()

    def get_bgr(self, stream_id: str, frame_id: int) -> np.ndarray | None:
        frame = self.get(stream_id, frame_id)
        if frame is None:
            return None
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        return frame

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"pending_frames": len(self._frames), "dropped": self._dropped}
