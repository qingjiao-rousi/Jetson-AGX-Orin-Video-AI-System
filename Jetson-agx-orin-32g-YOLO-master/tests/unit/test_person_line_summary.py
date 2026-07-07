from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from summarize_person_line import parse_line, summarize_line_crossings


class PersonLineSummaryTests(unittest.TestCase):
    def test_summarize_line_crossings_counts_direction_changes(self) -> None:
        rows = [
            {
                "frame_id": 0,
                "timestamp": "t0",
                "tracks": [
                    {
                        "track_id": 1,
                        "class_id": 0,
                        "confidence": 0.8,
                        "bbox": {"left": 20, "top": 40, "width": 20, "height": 20},
                    },
                    {
                        "track_id": 2,
                        "class_id": 0,
                        "confidence": 0.7,
                        "bbox": {"left": 120, "top": 40, "width": 20, "height": 20},
                    },
                ],
            },
            {
                "frame_id": 1,
                "timestamp": "t1",
                "tracks": [
                    {
                        "track_id": 1,
                        "class_id": 0,
                        "confidence": 0.9,
                        "bbox": {"left": 120, "top": 40, "width": 20, "height": 20},
                    },
                    {
                        "track_id": 2,
                        "class_id": 0,
                        "confidence": 0.6,
                        "bbox": {"left": 20, "top": 40, "width": 20, "height": 20},
                    },
                ],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            summary = summarize_line_crossings(
                path,
                line=parse_line("100,0,100,100"),
                line_id="gate",
                min_side_distance=1.0,
            )

        self.assertEqual(summary["line_id"], "gate")
        self.assertEqual(summary["line_crossing_in"], 1)
        self.assertEqual(summary["line_crossing_out"], 1)
        self.assertEqual(summary["in_track_ids"], [2])
        self.assertEqual(summary["out_track_ids"], [1])
        self.assertEqual(summary["crossing_count"], 2)


if __name__ == "__main__":
    unittest.main()
