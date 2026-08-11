from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.application.debug_service import DebugService
from app.optimization.strategy_advisor import OptimizationAdvisor
from app.shared.logger import InMemoryLogBuffer


class FakeOrchestrator:
    class Settings:
        app_name = "demo-app"

    settings = Settings()

    def status_snapshot(self) -> dict:
        return {
            "app": {"started": True},
            "pipeline": {"running": True},
            "pipeline_status": {"pipeline_state": "PLAYING"},
            "bus": {"watch_attached": True, "last_message_type": "STARTED", "last_warning": None},
            "writer": {"lines_written": 3},
            "monitor": {"status": "started"},
            "controllers": {"fps": {"observations": 3}, "backpressure": {"queue_limit": 32}},
            "last_result": None,
            "is_running": True,
            "source_count": 6,
            "last_error": None,
        }


class DebugServiceTests(unittest.TestCase):
    def test_debug_snapshot_contains_status_logs_and_optimization(self) -> None:
        buffer = InMemoryLogBuffer(capacity=10)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello dashboard",
            args=(),
            exc_info=None,
        )
        buffer.append(record)

        service = DebugService(
            orchestrator=FakeOrchestrator(),
            log_buffer=buffer,
            optimization_advisor=OptimizationAdvisor(),
        )

        snapshot = service.debug_snapshot(limit=5)

        self.assertIn("health", snapshot)
        self.assertIn("status", snapshot)
        self.assertIn("optimization", snapshot)
        self.assertEqual(snapshot["recent_logs"]["items"][0]["message"], "hello dashboard")


if __name__ == "__main__":
    unittest.main()
