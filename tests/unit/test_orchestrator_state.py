from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
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

    def observe(self, result) -> None:
        self._count += 1

    def stats(self) -> dict:
        return {"observations": self._count}


class OrchestratorStateTests(unittest.TestCase):
    def test_status_snapshot_returns_all_subsystem_states(self) -> None:
        orchestrator = Orchestrator(
            settings=FakeSettings(),
            pipeline_manager=FakePipelineManager(),
            meta_parser=type("P", (), {"parse": staticmethod(lambda x: x)})(),
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


if __name__ == "__main__":
    unittest.main()

