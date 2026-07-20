from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest

from app.infrastructure.output.event_writer import EventWriter


@dataclass(frozen=True)
class SampleEvent:
    event_type: str
    track_id: int


class EventWriterTests(unittest.TestCase):
    def test_writes_dataclass_event_as_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            writer = EventWriter(path)
            writer.write(SampleEvent("helmet_violation", 7))
            writer.close()

            payload = json.loads(path.read_text(encoding="utf-8").strip())
        self.assertEqual(payload, {"event_type": "helmet_violation", "track_id": 7})


if __name__ == "__main__":
    unittest.main()
