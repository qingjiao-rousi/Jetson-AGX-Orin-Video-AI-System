from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from check_person_batch_outputs import check_batch


class PersonBatchQualityTests(unittest.TestCase):
    def test_check_batch_marks_passed_review_and_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            passed_dir = root / "001_passed"
            review_dir = root / "002_review"
            failed_dir = root / "003_failed"
            for directory in (passed_dir, review_dir, failed_dir):
                directory.mkdir()

            for directory in (passed_dir, review_dir):
                for name in (
                    "person_analytics.mp4",
                    "person_analytics_overlay.mp4",
                    "results.jsonl",
                    "analytics_summary.json",
                ):
                    (directory / name).write_text("x", encoding="utf-8")

            summary_path = root / "batch_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "videos": [
                            {
                                "input_video": "/videos/passed.mp4",
                                "status": "ok",
                                "total_unique_persons": 2,
                                "line_crossing_in": 1,
                                "line_crossing_out": 0,
                                "output_dir": str(passed_dir),
                                "output_video": str(passed_dir / "person_analytics.mp4"),
                                "output_overlay_video": str(passed_dir / "person_analytics_overlay.mp4"),
                                "output_jsonl": str(passed_dir / "results.jsonl"),
                                "output_summary": str(passed_dir / "analytics_summary.json"),
                                "streams": {
                                    "stream-0": {
                                        "frame_count": 100,
                                        "is_frame_continuous": True,
                                        "estimated_fps": 50.0,
                                    }
                                },
                            },
                            {
                                "input_video": "/videos/review.mp4",
                                "status": "ok",
                                "total_unique_persons": 0,
                                "output_dir": str(review_dir),
                                "output_video": str(review_dir / "person_analytics.mp4"),
                                "output_overlay_video": str(review_dir / "person_analytics_overlay.mp4"),
                                "output_jsonl": str(review_dir / "results.jsonl"),
                                "output_summary": str(review_dir / "analytics_summary.json"),
                                "streams": {
                                    "stream-0": {
                                        "frame_count": 50,
                                        "is_frame_continuous": False,
                                        "estimated_fps": 50.0,
                                    }
                                },
                            },
                            {
                                "input_video": "/videos/failed.mp4",
                                "status": "failed",
                                "error": "pipeline failed",
                                "output_dir": str(failed_dir),
                                "output_video": str(failed_dir / "missing.mp4"),
                                "output_overlay_video": str(failed_dir / "missing_overlay.mp4"),
                                "output_jsonl": str(failed_dir / "missing.jsonl"),
                                "output_summary": str(failed_dir / "missing_summary.json"),
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            quality = check_batch(summary_path)

        self.assertEqual(quality["video_count"], 3)
        self.assertEqual(quality["passed_count"], 1)
        self.assertEqual(quality["review_count"], 1)
        self.assertEqual(quality["failed_count"], 1)
        self.assertEqual(quality["videos"][0]["quality_status"], "passed")
        self.assertEqual(quality["videos"][1]["quality_status"], "review")
        self.assertIn("zero unique persons", quality["videos"][1]["reviews"])
        self.assertEqual(quality["videos"][2]["quality_status"], "failed")
        self.assertIn("pipeline failed", quality["videos"][2]["failures"])


if __name__ == "__main__":
    unittest.main()
