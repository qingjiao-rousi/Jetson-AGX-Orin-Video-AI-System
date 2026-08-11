from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from summarize_person_analytics import summarize_analytics


class PersonAnalyticsSummaryTests(unittest.TestCase):
    def test_summarize_analytics_uses_yaml_rois_and_lines(self) -> None:
        rows = [
            {
                "frame_id": 0,
                "timestamp": "t0",
                "detections": [
                    {
                        "class_id": 0,
                        "class_name": "person",
                        "confidence": 0.8,
                        "bbox": {"left": 10, "top": 10, "width": 20, "height": 20},
                    }
                ],
                "tracks": [
                    {
                        "track_id": 1,
                        "class_id": 0,
                        "confidence": 0.8,
                        "bbox": {"left": 10, "top": 10, "width": 20, "height": 20},
                    }
                ],
            },
            {
                "frame_id": 1,
                "timestamp": "t1",
                "detections": [
                    {
                        "class_id": 0,
                        "class_name": "person",
                        "confidence": 0.9,
                        "bbox": {"left": 120, "top": 10, "width": 20, "height": 20},
                    }
                ],
                "tracks": [
                    {
                        "track_id": 1,
                        "class_id": 0,
                        "confidence": 0.9,
                        "bbox": {"left": 120, "top": 10, "width": 20, "height": 20},
                    }
                ],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl = root / "results.jsonl"
            config = root / "analytics.yaml"
            jsonl.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            config.write_text(
                "\n".join(
                    [
                        "min_track_frames: 2",
                        "rois:",
                        "  - id: full",
                        "    rect: [0, 0, 200, 100]",
                        "lines:",
                        "  - id: gate",
                        "    points: [100, 0, 100, 100]",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = summarize_analytics(jsonl, config)

        self.assertEqual(summary["global"]["total_unique_persons"], 1)
        self.assertTrue(summary["timeline"]["streams"]["stream-0"]["is_frame_continuous"])
        self.assertEqual(summary["rois"][0]["unique_persons_in_roi"], 1)
        self.assertEqual(summary["lines"][0]["crossing_count"], 1)


if __name__ == "__main__":
    unittest.main()
