from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.application.orchestrator import Orchestrator
from app.domain.entities import FrameResult


@dataclass
class FakeSettings:
    def validate(self) -> None:
        return None


class FakePipelineManager:
    def __init__(self) -> None:
        self._error_cleared = False
        self._started = False
        self._stopped = False
        self._runtime = {"blueprint": object()}
        self._last_error = None

    def clear_error(self) -> None:
        self._error_cleared = True

    def probes(self):
        class FakeProbes:
            def register_frame_result_handler(self, handler) -> None:
                self.handler = handler

        if not hasattr(self, "_probes"):
            self._probes = FakeProbes()
        return self._probes

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._stopped = True

    def set_error(self, message: str) -> None:
        self._last_error = message

    def state(self):
        class S:
            is_running = True
            source_count = 2
            last_error = None

        return S()

    def describe(self) -> dict:
        return {"source_count": 2, "nodes": ("a", "b"), "probes": (("p", "src"),)}

    def bus_state(self) -> dict:
        return {"watch_attached": True, "last_message_type": "STARTED", "last_error": None, "last_warning": None, "running": True}


class FakeWriter:
    def __init__(self) -> None:
        self.closed = False

    def write(self, result) -> None:
        self.last = result

    def close(self) -> None:
        self.closed = True

    def stats(self) -> dict:
        return {"lines_written": 1, "is_closed": self.closed}


class ExplodingWriter(FakeWriter):
    def write(self, result) -> None:
        raise OSError("disk full")


class ExplodingParser:
    def parse(self, result):
        raise AssertionError("FrameResult should not be parsed twice")


class FakeMonitor:
    def __init__(self) -> None:
        self._running = False

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def snapshot(self) -> dict:
        return {"status": "started"}


class FakeController:
    def __init__(self) -> None:
        self._count = 0
        self._consumed = 0

    def observe(self, result) -> None:
        self._count += 1

    def mark_consumed(self) -> None:
        self._consumed += 1

    def stats(self) -> dict:
        return {"observations": self._count, "consumed": self._consumed}


class OrchestratorStateTests(unittest.TestCase):
    def test_status_snapshot_returns_all_subsystem_states(self) -> None:
        orchestrator = Orchestrator(
            settings=FakeSettings(),
            pipeline_manager=FakePipelineManager(),
            meta_parser=ExplodingParser(),
            json_writer=FakeWriter(),
            gpu_monitor=FakeMonitor(),
            fps_controller=FakeController(),
            backpressure_controller=FakeController(),
        )

        snapshot = orchestrator.status_snapshot()

        self.assertIn("pipeline", snapshot)
        self.assertIn("bus", snapshot)
        self.assertIn("writer", snapshot)
        self.assertIn("monitor", snapshot)
        self.assertIn("controllers", snapshot)

    def test_frame_result_is_written_without_second_parse(self) -> None:
        writer = FakeWriter()
        backpressure = FakeController()
        fps = FakeController()
        orchestrator = Orchestrator(
            settings=FakeSettings(),
            pipeline_manager=FakePipelineManager(),
            meta_parser=ExplodingParser(),
            json_writer=writer,
            gpu_monitor=FakeMonitor(),
            fps_controller=fps,
            backpressure_controller=backpressure,
        )
        result = FrameResult(
            stream_id="stream-0",
            frame_id=1,
            timestamp=datetime(2026, 7, 9, tzinfo=timezone.utc),
        )

        orchestrator.on_frame_result(result)

        self.assertIs(writer.last, result)
        self.assertIs(orchestrator._last_result, result)
        self.assertEqual(backpressure.stats()["observations"], 1)
        self.assertEqual(backpressure.stats()["consumed"], 0)
        self.assertEqual(fps.stats()["observations"], 0)

    def test_frame_result_errors_are_captured_without_escaping_probe_callback(self) -> None:
        pipeline_manager = FakePipelineManager()
        orchestrator = Orchestrator(
            settings=FakeSettings(),
            pipeline_manager=pipeline_manager,
            meta_parser=ExplodingParser(),
            json_writer=ExplodingWriter(),
            gpu_monitor=FakeMonitor(),
            fps_controller=FakeController(),
            backpressure_controller=FakeController(),
        )
        result = FrameResult(
            stream_id="stream-0",
            frame_id=1,
            timestamp=datetime(2026, 7, 9, tzinfo=timezone.utc),
        )

        with self.assertLogs(level="ERROR") as logs:
            orchestrator.on_frame_result(result)

        self.assertTrue(orchestrator._stop_event.is_set())
        self.assertIn("frame result handler failed", pipeline_manager._last_error)
        self.assertIn("disk full", pipeline_manager._last_error)
        self.assertTrue(any("frame result handler failed" in entry for entry in logs.output))


if __name__ == "__main__":
    unittest.main()
