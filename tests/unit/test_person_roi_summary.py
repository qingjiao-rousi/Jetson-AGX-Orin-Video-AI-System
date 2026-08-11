from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "legacy" / "person_analytics"))

from summarize_person_roi import parse_roi, summarize_roi


class PersonRoiSummaryTests(unittest.TestCase):
    def test_summarize_roi_counts_tracks_by_bbox_center(self) -> None:
        rows = [
            {
                "frame_id": 0,
                "timestamp": "t0",
                "tracks": [
                    {
                        "track_id": 1,
                        "class_id": 0,
                        "confidence": 0.8,
                        "bbox": {"left": 10, "top": 10, "width": 20, "height": 20},
                    },
                    {
                        "track_id": 2,
                        "class_id": 0,
                        "confidence": 0.9,
                        "bbox": {"left": 200, "top": 200, "width": 20, "height": 20},
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
                        "confidence": 0.6,
                        "bbox": {"left": 20, "top": 20, "width": 20, "height": 20},
                    }
                ],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            summary = summarize_roi(
                path,
                roi=parse_roi("0,0,100,100"),
                roi_id="entrance",
                min_track_frames=2,
            )

        self.assertEqual(summary["roi_id"], "entrance")
        self.assertEqual(summary["frames_with_roi_person"], 2)
        self.assertEqual(summary["unique_persons_in_roi"], 1)
        self.assertEqual(summary["stable_track_ids"], [1])
        self.assertAlmostEqual(summary["tracks"][0]["average_confidence"], 0.7)
        self.assertEqual(summary["tracks"][0]["frames_in_roi"], 2)


if __name__ == "__main__":
    unittest.main()
