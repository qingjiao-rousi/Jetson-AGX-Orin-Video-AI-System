from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.infrastructure.pipeline.manager import PipelineManager


class FakeBus:
    def __init__(self) -> None:
        self.watch_added = False
        self.connected: dict[str, object] = {}

    def add_signal_watch(self) -> None:
        self.watch_added = True

    def connect(self, signal_name: str, callback) -> int:
        self.connected[signal_name] = callback
        return 1


class FakePipeline:
    def __init__(self) -> None:
        self.bus = FakeBus()

    def get_bus(self) -> FakeBus:
        return self.bus


class FakeBuilder:
    def build(self):
        class FakeBlueprint:
            app_name = "demo"
            source_count = 2
            probes = ()
            nodes = ()
            links = ()

        return FakeBlueprint()

    def build_runtime(self):
        return {
            "blueprint": self.build(),
            "pipeline": FakePipeline(),
            "elements": {},
            "gstreamer_available": True,
            "static_links": (),
            "dynamic_links": (),
            "streammux_requests": (),
            "probe_attachments": (),
        }


class FakeBuilderWithFakeOutput(FakeBuilder):
    class Settings:
        class DeepStream:
            output_sink = "fake"

        deepstream = DeepStream()

    settings = Settings()
    fallback_called = False

    def build_runtime_with_fake_output(self):
        self.fallback_called = True
        raise AssertionError("fake output must not fall back to fake output")

    def has_output_fallback_active(self) -> bool:
        return False


class PipelineManagerBusTests(unittest.TestCase):
    def test_fake_output_does_not_attempt_redundant_output_fallback(self) -> None:
        builder = FakeBuilderWithFakeOutput()
        manager = PipelineManager(builder)

        self.assertFalse(manager._try_rebuild_with_output_fallback(RuntimeError("device unavailable")))
        self.assertFalse(builder.fallback_called)

    def test_start_registers_bus_watch_when_runtime_pipeline_exists(self) -> None:
        manager = PipelineManager(FakeBuilder())

        manager.start()

        runtime = manager.runtime()
        self.assertIsNotNone(runtime)
        bus = runtime["pipeline"].get_bus()
        self.assertTrue(bus.watch_added)
        self.assertIn("message", bus.connected)

    def test_error_message_updates_error_and_stops_running(self) -> None:
        manager = PipelineManager(FakeBuilder())
        manager.start()

        class FakeMessage:
            type = "ERROR"

            def parse_error(self):
                return RuntimeError("boom"), "debug-info"

        manager._on_bus_message(None, FakeMessage())

        self.assertFalse(manager.running)
        self.assertEqual(manager.state().last_error, "boom")

    def test_eos_message_stops_running_without_error(self) -> None:
        manager = PipelineManager(FakeBuilder())
        manager.start()

        class FakeMessage:
            type = "EOS"

        manager._on_bus_message(None, FakeMessage())

        self.assertFalse(manager.running)
        self.assertIsNone(manager.state().last_error)

    def test_warning_message_keeps_running_and_records_warning(self) -> None:
        manager = PipelineManager(FakeBuilder())
        manager.start()

        class FakeMessage:
            type = "WARNING"

            def parse_warning(self):
                return RuntimeError("careful"), "warn-debug"

        manager._on_bus_message(None, FakeMessage())

        self.assertTrue(manager.running)
        self.assertEqual(manager.bus_state()["last_warning"], "careful")

    def test_bus_state_exposes_clear_runtime_status(self) -> None:
        manager = PipelineManager(FakeBuilder())
        manager.start()

        state = manager.bus_state()

        self.assertTrue(state["watch_attached"])
        self.assertEqual(state["last_message_type"], "STARTED")
        self.assertIsNone(state["last_error"])

    def test_state_summary_reports_playing_when_running(self) -> None:
        manager = PipelineManager(FakeBuilder())
        manager.start()

        state = manager.pipeline_status()

        self.assertEqual(state["pipeline_state"], "PLAYING")
        self.assertTrue(state["running"])

    def test_state_summary_reports_null_when_stopped(self) -> None:
        manager = PipelineManager(FakeBuilder())
        manager.start()
        manager.stop()

        state = manager.pipeline_status()

        self.assertEqual(state["pipeline_state"], "NULL")
        self.assertFalse(state["running"])

    def test_stop_joins_existing_bus_thread(self) -> None:
        manager = PipelineManager(FakeBuilder())

        class FakeThread:
            def __init__(self) -> None:
                self.joined_with = None

            def is_alive(self) -> bool:
                return True

            def join(self, timeout=None) -> None:
                self.joined_with = timeout

        bus_thread = FakeThread()
        manager._bus_thread = bus_thread

        manager.stop()

        self.assertEqual(bus_thread.joined_with, 1.0)
        self.assertIsNone(manager._bus_thread)

    def test_state_changed_message_updates_pipeline_state(self) -> None:
        manager = PipelineManager(FakeBuilder())
        manager.start()

        class FakeMessage:
            type = "STATE_CHANGED"

            def parse_state_changed(self):
                return "NULL", "PLAYING", "VOID_PENDING"

        manager._on_bus_message(None, FakeMessage())

        self.assertEqual(manager.bus_state()["last_message_type"], "STATE_CHANGED")
        self.assertEqual(manager.pipeline_status()["pipeline_state"], "PLAYING")


if __name__ == "__main__":
    unittest.main()
