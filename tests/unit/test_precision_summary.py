from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "benchmark"))

from summarize_precision_run import summarize


class PrecisionSummaryTests(unittest.TestCase):
    def test_summary_includes_latency_and_bounded_queue_drops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            metric = {
                "elapsed_seconds": 10.0,
                "total_frames": 100,
                "processing_fps": 10.0,
                "process": {"max_rss_kb": 1024},
                "gpu": {"utilization_gpu": 50, "temperature_c": 55, "ram_used_mb": 1000},
                "latency": {
                    "definition": "primary_infer_sink_to_json_write_ms",
                    "pipeline": {"samples": 100, "p50_ms": 10, "p95_ms": 20},
                    "json_writer": {"samples": 100, "p50_ms": 1, "p95_ms": 2},
                    "end_to_end": {"samples": 100, "p50_ms": 11, "p95_ms": 22},
                    "unmatched_results": 0,
                    "unmatched_writes": 0,
                },
                "queues": {
                    "writer": {"dropped": 1, "write_errors": 0},
                    "task_buffer": {"dropped": 2},
                    "frame_store": {"dropped": 3},
                },
                "controls": {
                    "fps": {"dropped_frames": 4, "drop_ratio": 0.04},
                    "backpressure": {"max_pending_ever": 5},
                },
                "streams": {},
            }
            (output_dir / "runtime_metrics.jsonl").write_text(json.dumps(metric) + "\n", encoding="utf-8")
            (output_dir / "results.jsonl").write_text("", encoding="utf-8")
            (output_dir / "events.jsonl").write_text("", encoding="utf-8")

            summary = summarize("int8", output_dir, warmup_samples=0)

        self.assertEqual(summary["latency"]["end_to_end_p95_ms"], 22)
        self.assertEqual(summary["drop_and_queue_stats"]["writer_dropped"], 1)
        self.assertEqual(summary["drop_and_queue_stats"]["task_buffer_dropped"], 2)
        self.assertEqual(summary["drop_and_queue_stats"]["frame_store_dropped"], 3)
        self.assertEqual(summary["drop_and_queue_stats"]["fps_controller_dropped"], 4)


if __name__ == "__main__":
    unittest.main()
