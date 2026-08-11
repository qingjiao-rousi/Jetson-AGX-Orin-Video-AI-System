from __future__ import annotations

import logging
from threading import Event, Thread, current_thread

from app.domain.entities import PipelineState
from app.infrastructure.pipeline.builder import PipelineBlueprint
from app.infrastructure.pipeline.probes import ProbeRegistry


class PipelineManager:
    def __init__(self, builder, probes: ProbeRegistry | None = None, meta_parser=None, frame_store=None) -> None:
        self._builder = builder
        self._probes = probes or ProbeRegistry()
        self._meta_parser = meta_parser
        self._frame_store = frame_store
        self._pipeline: PipelineBlueprint | None = None
        self._runtime: dict | None = None
        self._running = False
        self._last_error: str | None = None
        self._last_warning: str | None = None
        self._last_message_type: str | None = None
        self._bus_watch_attached = False
        self._stop_event = Event()
        self._bus_thread: Thread | None = None
        self._frame_gate = None
        self._runtime_metrics = None

    def set_frame_gate(self, gate) -> None:
        """Set a pre-inference buffer gate, typically the FPS controller."""
        self._frame_gate = gate

    def set_runtime_metrics(self, runtime_metrics) -> None:
        self._runtime_metrics = runtime_metrics

    def register_plate_annotation(self, stream_id: str, track_id: int, event: dict) -> None:
        if hasattr(self._builder, "register_plate_annotation"):
            self._builder.register_plate_annotation(stream_id, track_id, event)

    def start(self) -> None:
        if self._running:
            return
        self._runtime = self._builder.build_runtime()
        self._runtime["probe_registry"] = self._probes
        self._runtime["meta_parser"] = self._meta_parser
        self._runtime["frame_store"] = self._frame_store
        self._runtime["frame_gate"] = self._frame_gate
        self._runtime["runtime_metrics"] = self._runtime_metrics
        self._register_probe_points()
        self._pipeline = self._runtime["blueprint"]
        self._stop_event.clear()
        self._attach_bus_watch()
        try:
            self._set_pipeline_state_playing()
        except RuntimeError as exc:
            if not self._try_rebuild_with_output_fallback(exc):
                raise
        self._start_bus_polling()
        self._running = True
        self._last_error = None
        self._last_warning = None
        self._last_message_type = "STARTED"

    def stop(self) -> None:
        self._stop_event.set()
        bus_thread = self._bus_thread
        if self._runtime is not None:
            pipeline = self._runtime.get("pipeline")
            gst = self._runtime.get("gst")
            if pipeline is not None and gst is not None and hasattr(pipeline, "set_state"):
                self._finalize_file_output(pipeline, gst)
                pipeline.set_state(gst.State.NULL)
        if bus_thread is not None and bus_thread is not current_thread() and bus_thread.is_alive():
            bus_thread.join(timeout=1.0)
        self._running = False
        self._pipeline = None
        self._runtime = None
        self._bus_watch_attached = False
        self._bus_thread = None

    def _finalize_file_output(self, pipeline, gst) -> None:
        if not self._runtime:
            return
        blueprint = self._runtime.get("blueprint")
        output_policy = getattr(blueprint, "output_policy", {}) if blueprint is not None else {}
        if not output_policy.get("enable_file_sink"):
            return
        if not hasattr(pipeline, "send_event") or not hasattr(gst, "Event"):
            return
        bus = pipeline.get_bus() if hasattr(pipeline, "get_bus") else None
        try:
            pipeline.send_event(gst.Event.new_eos())
            if bus is not None and hasattr(bus, "timed_pop_filtered"):
                message_types = gst.MessageType.ERROR | gst.MessageType.EOS
                bus.timed_pop_filtered(3_000_000_000, message_types)
        except Exception as exc:
            logging.warning("failed to finalize file output with EOS: %s", exc)

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
            "sources": tuple(
                {
                    "name": source.name,
                    "kind": source.kind,
                    "scene": getattr(source, "scene", "normal"),
                    "priority": getattr(source, "priority", "medium"),
                    "zones": tuple(getattr(source, "zones", ())),
                    "capabilities": tuple(getattr(source, "capabilities", ())),
                }
                for source in self._pipeline.sources
            ),
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

    def _start_bus_polling(self) -> None:
        if not self._runtime:
            return
        pipeline = self._runtime.get("pipeline")
        gst = self._runtime.get("gst")
        if pipeline is None or gst is None or not hasattr(pipeline, "get_bus"):
            return
        bus = pipeline.get_bus()
        if bus is None or not hasattr(bus, "timed_pop_filtered"):
            return
        message_types = gst.MessageType.ERROR | gst.MessageType.EOS | gst.MessageType.WARNING
        self._bus_thread = Thread(
            target=self._poll_bus,
            args=(bus, message_types),
            name="gstreamer-bus-poller",
            daemon=True,
        )
        self._bus_thread.start()

    def _poll_bus(self, bus, message_types) -> None:
        while not self._stop_event.is_set():
            message = bus.timed_pop_filtered(200_000_000, message_types)
            if message is None:
                continue
            self._on_bus_message(bus, message)
            if not self._running or self._last_error is not None:
                self._stop_event.set()
                return

    def _set_pipeline_state_playing(self) -> None:
        if not self._runtime:
            return
        pipeline = self._runtime.get("pipeline")
        gst = self._runtime.get("gst")
        if pipeline is None or gst is None or not hasattr(pipeline, "set_state"):
            return
        result = pipeline.set_state(gst.State.PLAYING)
        if hasattr(gst, "StateChangeReturn") and result == gst.StateChangeReturn.FAILURE:
            self._last_error = "failed to set GStreamer pipeline to PLAYING"
            self._running = False
            raise RuntimeError(self._last_error)

    def _try_rebuild_with_output_fallback(self, exc: RuntimeError) -> bool:
        if not hasattr(self._builder, "build_runtime_with_fake_output"):
            return False
        settings = getattr(self._builder, "settings", None)
        deepstream = getattr(settings, "deepstream", None)
        if getattr(deepstream, "output_sink", None) == "fake":
            return False
        if hasattr(self._builder, "has_output_fallback_active") and self._builder.has_output_fallback_active():
            return False
        logging_message = f"pipeline PLAYING failed; trying fake output fallback: {exc}"
        logging.warning(logging_message)
        old_runtime = self._runtime or {}
        old_pipeline = old_runtime.get("pipeline")
        gst = old_runtime.get("gst")
        if old_pipeline is not None and gst is not None and hasattr(old_pipeline, "set_state"):
            old_pipeline.set_state(gst.State.NULL)

        self._runtime = self._builder.build_runtime_with_fake_output()
        self._runtime["probe_registry"] = self._probes
        self._runtime["meta_parser"] = self._meta_parser
        self._runtime["frame_store"] = self._frame_store
        self._runtime["frame_gate"] = self._frame_gate
        self._register_probe_points()
        self._pipeline = self._runtime["blueprint"]
        self._bus_watch_attached = False
        self._attach_bus_watch()
        self._set_pipeline_state_playing()
        self._last_warning = logging_message
        self._last_error = None
        return True

    def _register_probe_points(self) -> tuple[dict, ...]:
        """Register probes once, after their registry/parser dependencies exist."""
        if self._runtime is None:
            return ()
        if self._runtime.get("probe_points_registered", False):
            return self._runtime.get("probe_attachments", ())
        attachments: tuple[dict, ...] = ()
        if hasattr(self._builder, "attach_probe_points"):
            attachments = self._builder.attach_probe_points(self._runtime)
        self._runtime["probe_attachments"] = attachments
        self._runtime["probe_points_registered"] = True
        return attachments

    def _on_bus_message(self, bus, message) -> None:
        _ = bus
        message_type = getattr(message, "type", None)
        self._last_message_type = str(message_type) if message_type is not None else None
        message_name = self._message_type_name(message_type)

        if message_name == "ERROR":
            error, _debug = message.parse_error()
            if self._is_live_source_recoverable_error(str(error)):
                self._last_warning = f"recoverable live source error: {error}"
                return
            self._last_error = str(error)
            self._running = False
            return

        if message_name == "EOS":
            if self._is_live_source_runtime():
                self._last_warning = "live source EOS received; keeping pipeline active for reconnect"
                return
            self._running = False
            return

        if message_name == "WARNING":
            warning, _debug = message.parse_warning()
            self._last_warning = str(warning)
            return

        if message_name == "STATE_CHANGED":
            old_state, new_state, pending_state = message.parse_state_changed()
            _ = old_state
            _ = pending_state
            self._running = str(new_state) == "PLAYING"
            return

    def _message_type_name(self, message_type) -> str:
        if message_type is None:
            return ""
        name = getattr(message_type, "value_nick", None) or getattr(message_type, "name", None)
        if name:
            return str(name).replace("-", "_").upper()
        text = str(message_type).upper()
        for candidate in ("ERROR", "EOS", "WARNING", "STATE_CHANGED"):
            if candidate in text:
                return candidate
        return text

    def _pipeline_state(self) -> str:
        if self._last_error is not None:
            return "ERROR"
        if self._running:
            return "PLAYING"
        return "NULL"

    def _is_live_source_runtime(self) -> bool:
        runtime = self._runtime or {}
        flags = runtime.get("runtime_flags", {})
        return bool(flags.get("live_source"))

    def _is_live_source_recoverable_error(self, error: str) -> bool:
        if not self._is_live_source_runtime():
            return False
        recoverable_markers = (
            "Could not read from resource",
            "Could not connect",
            "Connection refused",
            "Connection timed out",
            "The server closed the connection",
            "Internal data stream error",
            "streaming stopped",
            "No route to host",
            "Network is unreachable",
        )
        lowered = error.lower()
        return any(marker.lower() in lowered for marker in recoverable_markers)
