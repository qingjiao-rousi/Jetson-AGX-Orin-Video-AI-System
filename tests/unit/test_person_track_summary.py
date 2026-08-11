from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from summarize_person_tracks import summarize_tracks


class PersonTrackSummaryTests(unittest.TestCase):
    def test_summarize_tracks_counts_stable_unique_persons(self) -> None:
        rows = [
            {
                "frame_id": 0,
                "timestamp": "t0",
                "detections": [
                    {
                        "class_id": 0,
                        "class_name": "person",
                        "confidence": 0.9,
                        "bbox": {"left": 1, "top": 2, "width": 3, "height": 4},
                    }
                ],
                "tracks": [
                    {
                        "track_id": 10,
                        "class_id": 0,
                        "confidence": 0.9,
                        "bbox": {"left": 1, "top": 2, "width": 3, "height": 4},
                    }
                ],
            },
            {
                "frame_id": 1,
                "timestamp": "t1",
                "detections": [],
                "tracks": [
                    {
                        "track_id": 10,
                        "class_id": 0,
                        "confidence": 0.8,
                        "bbox": {"left": 2, "top": 3, "width": 4, "height": 5},
                    },
                    {
                        "track_id": 99,
                        "class_id": 0,
                        "confidence": 0.7,
                        "bbox": {"left": 10, "top": 20, "width": 30, "height": 40},
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

            summary = summarize_tracks(path, min_track_frames=2)

        self.assertEqual(summary["total_frames"], 2)
        self.assertEqual(summary["frames_with_person"], 1)
        self.assertEqual(summary["max_tracks_in_frame"], 2)
        self.assertEqual(summary["total_unique_persons"], 1)
        self.assertEqual(summary["stable_track_ids"], [10])
        self.assertEqual(summary["tracks"][0]["first_frame"], 0)
        self.assertEqual(summary["tracks"][0]["last_frame"], 1)
        self.assertEqual(summary["tracks"][0]["frame_count"], 2)


if __name__ == "__main__":
    unittest.main()
