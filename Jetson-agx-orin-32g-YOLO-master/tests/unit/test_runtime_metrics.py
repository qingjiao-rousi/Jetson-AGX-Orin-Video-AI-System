from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.domain.entities import BoundingBox, Detection, FrameResult
from app.infrastructure.monitoring.runtime_metrics import RuntimeMetricsRecorder


class RuntimeMetricsTests(unittest.TestCase):
    def test_stream_metrics_expose_current_counts_age_and_frame_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "metrics.jsonl"
            recorder = RuntimeMetricsRecorder(metrics_path)
            recorder.set_queue_metrics_provider(
                lambda: {"writer": {"dropped": 2}, "task_buffer": {"dropped": 3}}
            )
            recorder.set_probe_metrics_provider(lambda: {"native_calls": 3})
            recorder.set_control_metrics_provider(lambda: {"fps": {"dropped_frames": 2}})
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

            emitted = json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[-1])

        stream = payload["streams"]["stream-0"]
        self.assertEqual(stream["detection_count"], 1)
        self.assertEqual(stream["dropped_frames"], 4)
        self.assertEqual(stream["dropped_frame_rate"], 0.8)
        self.assertGreaterEqual(stream["frame_age_ms"], 0)
        self.assertEqual(payload["probe"]["native_calls"], 3)
        self.assertEqual(payload["controls"]["fps"]["dropped_frames"], 2)
        self.assertEqual(emitted["queues"]["writer"]["dropped"], 2)
        self.assertEqual(emitted["queues"]["task_buffer"]["dropped"], 3)

    def test_latency_tracks_pre_inference_to_json_write_percentiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = RuntimeMetricsRecorder(Path(tmp) / "metrics.jsonl")
            recorder.start()
            for frame_id in range(3):
                result = FrameResult(
                    stream_id="stream-0",
                    frame_id=frame_id,
                    timestamp=datetime.now(timezone.utc),
                )
                recorder.mark_pipeline_start(result.stream_id, result.frame_id)
                time.sleep(0.001)
                recorder.observe(result)
                time.sleep(0.001)
                recorder.mark_result_written(result)
            payload = recorder.snapshot()
            recorder.close()

        latency = payload["latency"]
        self.assertEqual(latency["definition"], "primary_infer_sink_to_json_write_ms")
        self.assertEqual(latency["pipeline"]["samples"], 3)
        self.assertEqual(latency["json_writer"]["samples"], 3)
        self.assertEqual(latency["end_to_end"]["samples"], 3)
        self.assertIsNotNone(latency["end_to_end"]["p50_ms"])
        self.assertIsNotNone(latency["end_to_end"]["p95_ms"])


if __name__ == "__main__":
    unittest.main()
