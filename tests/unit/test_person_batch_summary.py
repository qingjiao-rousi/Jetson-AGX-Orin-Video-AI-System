from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "legacy" / "person_analytics"))

from summarize_person_batch import summarize_batch


class PersonBatchSummaryTests(unittest.TestCase):
    def test_summarize_batch_combines_success_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            batch_dir = Path(tmp)
            ok_dir = batch_dir / "001_a"
            failed_dir = batch_dir / "002_b"
            ok_dir.mkdir()
            failed_dir.mkdir()

            (ok_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "input_video": "/videos/a.mp4",
                        "status": "ok",
                        "exit_code": 0,
                        "started_at": "2026-07-07T10:00:00-04:00",
                        "finished_at": "2026-07-07T10:00:10-04:00",
                        "error": "",
                        "log_path": str(ok_dir / "run.log"),
                        "batch_jobs": 8,
                    }
                ),
                encoding="utf-8",
            )
            (ok_dir / "run.log").write_text("done\n", encoding="utf-8")
            (ok_dir / "analytics_summary.json").write_text(
                json.dumps(
                    {
                        "global": {"total_unique_persons": 3},
                        "lines": [
                            {"line_id": "gate-a", "line_crossing_in": 2, "line_crossing_out": 1},
                            {"line_id": "gate-b", "line_crossing_in": 4, "line_crossing_out": 0},
                        ],
                        "rois": [
                            {"roi_id": "full", "unique_persons_in_roi": 3},
                            {"roi_id": "left", "unique_persons_in_roi": 1},
                        ],
                        "timeline": {
                            "streams": {
                                "stream-0": {
                                    "frame_count": 100,
                                    "is_frame_continuous": True,
                                    "estimated_fps": 50.0,
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            (failed_dir / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "input_video": "/videos/b.mp4",
                        "status": "failed",
                        "exit_code": 1,
                        "started_at": "2026-07-07T10:01:00-04:00",
                        "finished_at": "2026-07-07T10:01:01-04:00",
                        "error": "probe failed",
                    }
                ),
                encoding="utf-8",
            )

            summary = summarize_batch(batch_dir)

        self.assertEqual(summary["video_count"], 2)
        self.assertEqual(summary["processed_count"], 1)
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(summary["batch_jobs"], 8)
        self.assertEqual(summary["total_duration_seconds"], 61.0)
        self.assertEqual(summary["total_frame_count"], 100)
        self.assertAlmostEqual(summary["processing_fps"], 1.639)
        self.assertEqual(summary["total_unique_persons_sum"], 3)
        self.assertEqual(summary["line_crossing_in_sum"], 6)
        self.assertEqual(summary["line_crossing_out_sum"], 1)
        self.assertEqual(summary["videos"][0]["roi_unique_persons"], {"full": 3, "left": 1})
        self.assertEqual(summary["videos"][0]["streams"]["stream-0"]["frame_count"], 100)
        self.assertEqual(summary["videos"][0]["total_frame_count"], 100)
        self.assertEqual(summary["videos"][0]["processing_fps"], 10.0)
        self.assertEqual(summary["videos"][0]["duration_seconds"], 10.0)
        self.assertIn("log_path", summary["videos"][0])
        self.assertIn("output_summary", summary["videos"][0]["file_sizes"])
        self.assertEqual(summary["videos"][1]["error"], "probe failed")


if __name__ == "__main__":
    unittest.main()
