from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.domain.entities import BoundingBox, Detection, FrameResult
from app.infrastructure.monitoring.runtime_metrics import RuntimeMetricsRecorder


class RuntimeMetricsTests(unittest.TestCase):
    def test_stream_metrics_expose_current_counts_age_and_frame_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = RuntimeMetricsRecorder(Path(tmp) / "metrics.jsonl")
            recorder.start()
            result = FrameResult(
                stream_id="stream-0",
                frame_id=4,
                timestamp=datetime.now(timezone.utc),
                detections=[
                    Detection(
                        class_id=0,
                        class_name="person",
                        confidence=0.9,
                        bbox=BoundingBox(left=0, top=0, width=1, height=1),
                    )
                ],
            )
            recorder.observe(result)
            payload = recorder.snapshot()
            recorder.close()

        stream = payload["streams"]["stream-0"]
        self.assertEqual(stream["detection_count"], 1)
        self.assertEqual(stream["dropped_frames"], 4)
        self.assertEqual(stream["dropped_frame_rate"], 0.8)
        self.assertGreaterEqual(stream["frame_age_ms"], 0)


if __name__ == "__main__":
    unittest.main()
