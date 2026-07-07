from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from summarize_person_timeline import summarize_timeline


class PersonTimelineSummaryTests(unittest.TestCase):
    def test_summarize_timeline_reports_continuity_and_fps(self) -> None:
        rows = [
            {"stream_id": "stream-0", "frame_id": 0, "timestamp": "2026-07-07 00:00:00+00:00"},
            {"stream_id": "stream-0", "frame_id": 1, "timestamp": "2026-07-07 00:00:00.020000+00:00"},
            {"stream_id": "stream-0", "frame_id": 2, "timestamp": "2026-07-07 00:00:00.040000+00:00"},
            {"stream_id": "stream-1", "frame_id": 0, "timestamp": "2026-07-07 00:00:01+00:00"},
            {"stream_id": "stream-1", "frame_id": 2, "timestamp": "2026-07-07 00:00:01.040000+00:00"},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            summary = summarize_timeline(path)

        stream0 = summary["streams"]["stream-0"]
        stream1 = summary["streams"]["stream-1"]
        self.assertEqual(summary["stream_count"], 2)
        self.assertTrue(stream0["is_frame_continuous"])
        self.assertAlmostEqual(stream0["estimated_fps"], 50.0)
        self.assertFalse(stream1["is_frame_continuous"])
        self.assertEqual(stream1["missing_frames"], [1])


if __name__ == "__main__":
    unittest.main()
