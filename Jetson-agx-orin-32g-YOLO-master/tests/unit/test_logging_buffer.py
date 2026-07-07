from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.shared.logger import InMemoryLogBuffer


class LoggingBufferTests(unittest.TestCase):
    def test_tail_keeps_latest_records_only(self) -> None:
        buffer = InMemoryLogBuffer(capacity=2)

        for message in ("one", "two", "three"):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname=__file__,
                lineno=1,
                msg=message,
                args=(),
                exc_info=None,
            )
            buffer.append(record)

        tail = buffer.tail(2)

        self.assertEqual(len(tail), 2)
        self.assertEqual(tail[0]["message"], "two")
        self.assertEqual(tail[1]["message"], "three")


if __name__ == "__main__":
    unittest.main()
