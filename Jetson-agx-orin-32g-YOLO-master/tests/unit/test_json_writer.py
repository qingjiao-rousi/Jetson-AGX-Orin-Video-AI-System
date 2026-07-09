from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.domain.entities import BoundingBox, Detection, FrameResult
from app.infrastructure.output.json_writer import JsonWriter


class JsonWriterTests(unittest.TestCase):
    def test_write_serializes_datetimes_as_iso_utc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "results.jsonl"
            writer = JsonWriter(output_path)
            writer.write(
                FrameResult(
                    stream_id="stream-0",
                    frame_id=1,
                    timestamp=datetime(2026, 7, 9, 8, 30, tzinfo=timezone.utc),
                    detections=[
                        Detection(
                            class_id=0,
                            class_name="person",
                            confidence=0.9,
                            bbox=BoundingBox(left=1, top=2, width=3, height=4),
                        )
                    ],
                    extra={"source_timestamp": datetime(2026, 7, 9, 8, 31, tzinfo=timezone.utc)},
                )
            )
            writer.close()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["timestamp"], "2026-07-09T08:30:00+00:00")
        self.assertEqual(payload["extra"]["source_timestamp"], "2026-07-09T08:31:00+00:00")
        self.assertEqual(payload["detections"][0]["bbox"]["width"], 3)


if __name__ == "__main__":
    unittest.main()
