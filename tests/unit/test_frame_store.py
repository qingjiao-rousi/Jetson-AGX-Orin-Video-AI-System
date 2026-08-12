from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from app.infrastructure.inference.frame_store import FrameStore


class FrameStoreTests(unittest.TestCase):
    def test_tracks_consumer_hits_misses_and_frame_age(self) -> None:
        store = FrameStore(max_size=2)
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        with patch("app.infrastructure.inference.frame_store.time.monotonic", side_effect=(10.0, 10.2)):
            store.put("stream-0", 1, frame)
            result = store.get("stream-0", 1, consumer="helmet")
        self.assertIsNotNone(result)
        self.assertIsNone(store.get("stream-0", 99, consumer="helmet"))
        stats = store.stats()
        self.assertEqual(stats["pending_bytes"], frame.nbytes)
        self.assertEqual(stats["by_consumer"]["helmet"]["hits"], 1)
        self.assertEqual(stats["by_consumer"]["helmet"]["misses"], 1)
        self.assertEqual(stats["by_consumer"]["helmet"]["frame_age_ms"]["p50"], 200.0)

    def test_reports_evictions_per_stream(self) -> None:
        store = FrameStore(max_size=1)
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        store.put("stream-0", 1, frame)
        store.put("stream-1", 1, frame)
        stats = store.stats()
        self.assertEqual(stats["evicted"], 1)
        self.assertEqual(stats["dropped"], 1)
        self.assertEqual(stats["evicted_by_stream"], {"stream-0": 1})
        self.assertEqual(stats["pending_frames"], 1)

    def test_per_stream_capacity_preserves_other_stream_frames(self) -> None:
        store = FrameStore(max_size=4, max_per_stream=2)
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        store.put("stream-0", 1, frame)
        store.put("stream-0", 2, frame)
        store.put("stream-1", 1, frame)
        store.put("stream-0", 3, frame)
        self.assertIsNone(store.get("stream-0", 1))
        self.assertIsNotNone(store.get("stream-1", 1))
        stats = store.stats()
        self.assertEqual(stats["pending_by_stream"], {"stream-0": 2, "stream-1": 1})
        self.assertEqual(stats["evicted_per_stream"], 1)
        self.assertEqual(stats["evicted_global"], 0)


if __name__ == "__main__":
    unittest.main()
