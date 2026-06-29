from __future__ import annotations

from app.domain.entities import PipelineState
from app.infrastructure.pipeline.builder import PipelineBlueprint
from app.infrastructure.pipeline.probes import ProbeRegistry


class PipelineManager:
    def __init__(self, builder, probes: ProbeRegistry | None = None, meta_parser=None) -> None:
        self._builder = builder
        self._probes = probes or ProbeRegistry()
        self._meta_parser = meta_parser
        self._pipeline: PipelineBlueprint | None = None
        self._runtime: dict | None = None
        self._running = False
        self._last_error: str | None = None
        self._last_warning: str | None = None
        self._last_message_type: str | None = None
        self._bus_watch_attached = False

    def start(self) -> None:
        if self._running:
            return
        self._runtime = self._builder.build_runtime()
        self._runtime["probe_registry"] = self._probes
        self._runtime["meta_parser"] = self._meta_parser
        self._pipeline = self._runtime["blueprint"]
        self._attach_bus_watch()
        self._running = True
        self._last_error = None
        self._last_warning = None
        self._last_message_type = "STARTED"

    def stop(self) -> None:
        self._running = False
        self._pipeline = None
        self._runtime = None
        self._bus_watch_attached = False

    def restart(self) -> None:
        self.stop()
        self.start()

    def state(self) -> PipelineState:
        source_count = 0
        if isinstance(self._pipeline, PipelineBlueprint):
            source_count = self._pipeline.source_count
        return PipelineState(
            is_running=self._running,
            source_count=source_count,
            last_error=self._last_error,
        )

    def set_error(self, message: str) -> None:
        self._last_error = message
        self._running = False

    def clear_error(self) -> None:
        self._last_error = None

    def pipeline(self) -> PipelineBlueprint | None:
        return self._pipeline

    def runtime(self) -> dict | None:
        return self._runtime

    def probes(self) -> ProbeRegistry:
        return self._probes

    def probe_specs(self) -> tuple[tuple[str, str], ...]:
        if isinstance(self._pipeline, PipelineBlueprint):
            return self._pipeline.probes
        return ()

    def describe(self) -> dict:
        if not isinstance(self._pipeline, PipelineBlueprint):
            return {
                "running": self._running,
                "source_count": 0,
                "app_name": None,
                "nodes": (),
                "links": (),
                "probes": (),
            }
        return {
            "running": self._running,
            "source_count": self._pipeline.source_count,
            "app_name": self._pipeline.app_name,
            "nodes": tuple(node.name for node in self._pipeline.nodes),
            "node_specs": tuple(
                {
                    "name": node.name,
                    "element": node.element,
                    "properties": node.properties,
                }
                for node in self._pipeline.nodes
            ),
            "links": self._pipeline.links,
            "probes": self._pipeline.probes,
        }

    @property
    def running(self) -> bool:
        return self._running

    def bus_state(self) -> dict:
        return {
            "watch_attached": self._bus_watch_attached,
            "last_message_type": self._last_message_type,
            "last_error": self._last_error,
            "last_warning": self._last_warning,
            "running": self._running,
        }

    def pipeline_status(self) -> dict:
        return {
            "pipeline_state": self._pipeline_state(),
            "running": self._running,
            "has_runtime": self._runtime is not None,
            "watch_attached": self._bus_watch_attached,
            "last_message_type": self._last_message_type,
            "last_error": self._last_error,
            "last_warning": self._last_warning,
        }

    def _attach_bus_watch(self) -> None:
        if not self._runtime:
            return
        pipeline = self._runtime.get("pipeline")
        if pipeline is None or not hasattr(pipeline, "get_bus"):
            return
        bus = pipeline.get_bus()
        if bus is None:
            return
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)
        self._bus_watch_attached = True

    def _on_bus_message(self, bus, message) -> None:
        _ = bus
        message_type = getattr(message, "type", None)
        self._last_message_type = str(message_type) if message_type is not None else None

        if message_type == "ERROR":
            error, _debug = message.parse_error()
            self._last_error = str(error)
            self._running = False
            return

        if message_type == "EOS":
            self._running = False
            return

        if message_type == "WARNING":
            warning, _debug = message.parse_warning()
            self._last_warning = str(warning)
            return

        if message_type == "STATE_CHANGED":
            old_state, new_state, pending_state = message.parse_state_changed()
            _ = old_state
            _ = pending_state
            self._running = str(new_state) == "PLAYING"
            return

    def _pipeline_state(self) -> str:
        if self._last_error is not None:
            return "ERROR"
        if self._running:
            return "PLAYING"
        return "NULL"
