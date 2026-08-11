from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "legacy" / "person_analytics"))

from export_person_batch_report import export_report


class PersonBatchReportTests(unittest.TestCase):
    def test_export_report_writes_csv_and_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path = root / "batch_summary.json"
            csv_path = root / "batch_summary.csv"
            html_path = root / "batch_report.html"
            summary_path.write_text(
                json.dumps(
                    {
                        "batch_dir": str(root),
                        "video_count": 2,
                        "processed_count": 1,
                        "failed_count": 1,
                        "total_unique_persons_sum": 3,
                        "line_crossing_in_sum": 2,
                        "line_crossing_out_sum": 1,
                        "videos": [
                            {
                                "input_video": "/videos/a.mp4",
                                "status": "ok",
                                "total_unique_persons": 3,
                                "roi_unique_persons": {"full": 3},
                                "line_crossing_in": 2,
                                "line_crossing_out": 1,
                                "streams": {
                                    "stream-0": {
                                        "frame_count": 100,
                                        "estimated_fps": 49.98,
                                        "is_frame_continuous": True,
                                    }
                                },
                                "output_video": str(root / "001_a/person_analytics.mp4"),
                                "output_overlay_video": str(root / "001_a/person_analytics_overlay.mp4"),
                                "output_jsonl": str(root / "001_a/results.jsonl"),
                                "output_summary": str(root / "001_a/analytics_summary.json"),
                            },
                            {
                                "input_video": "/videos/b.mp4",
                                "status": "failed",
                                "exit_code": 1,
                                "error": "probe failed",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            export_report(summary_path, csv_path, html_path)

            with csv_path.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            html = html_path.read_text(encoding="utf-8")

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["input_video"], "/videos/a.mp4")
        self.assertEqual(rows[0]["total_unique_persons"], "3")
        self.assertEqual(rows[0]["roi_unique_persons"], "full=3")
        self.assertEqual(rows[0]["estimated_fps"], "49.98")
        self.assertEqual(rows[1]["status"], "failed")
        self.assertEqual(rows[1]["error"], "probe failed")
        self.assertIn("Person Batch Report", html)
        self.assertIn("a.mp4", html)
        self.assertIn("probe failed", html)
        self.assertIn("001_a/person_analytics_overlay.mp4", html)


if __name__ == "__main__":
    unittest.main()
