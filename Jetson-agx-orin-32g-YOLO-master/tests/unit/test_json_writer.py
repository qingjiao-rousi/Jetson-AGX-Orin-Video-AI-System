from __future__ import annotations

import json
import sys
import tempfile
import unittest
from threading import Event
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

    def test_writer_drops_oldest_when_queue_is_full(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "results.jsonl"
            writer = JsonWriter(output_path, queue_size=1)
            started = Event()
            release = Event()
            original_write_item = writer._write_item

            def blocked_write_item(result) -> None:
                started.set()
                release.wait(timeout=2)
                original_write_item(result)

            writer._write_item = blocked_write_item
            writer.write(FrameResult("stream-0", 1, datetime.now(timezone.utc)))
            self.assertTrue(started.wait(timeout=2))
            writer.write(FrameResult("stream-0", 2, datetime.now(timezone.utc)))
            writer.write(FrameResult("stream-0", 3, datetime.now(timezone.utc)))
            release.set()
            writer.close()

            stats = writer.stats()
            rows = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(stats["dropped"], 1)
        self.assertEqual(stats["lines_written"], 2)
        self.assertEqual([json.loads(row)["frame_id"] for row in rows], [1, 3])
        self.assertFalse(stats["worker_alive"])

    def test_written_callback_receives_committed_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            written = []
            writer = JsonWriter(Path(tmp) / "results.jsonl", on_written=written.append)
            result = FrameResult("stream-0", 7, datetime.now(timezone.utc))
            writer.write(result)
            writer.close()

        self.assertEqual(written, [result])


if __name__ == "__main__":
    unittest.main()
