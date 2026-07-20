from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging
from threading import Lock
import time

from app.domain.entities import canonical_stream_id
from app.infrastructure.pipeline.source_factory import SourceBranchSpec, SourceFactory, SourceSpec
from app.infrastructure.pipeline.cpp_probe import CppProbeHandler

UNTRACKED_OBJECT_ID = 0xFFFFFFFFFFFFFFFF
OSD_CONFIDENCE_UPDATE_THRESHOLD = 0.05


@dataclass(frozen=True)
class PipelineNodeSpec:
    name: str
    element: str
    stage: str
    properties: dict[str, Any]
    flags: dict[str, Any]


@dataclass(frozen=True)
class PipelineBlueprint:
    app_name: str
    source_count: int
    sources: tuple[SourceSpec, ...]
    nodes: tuple[PipelineNodeSpec, ...]
    links: tuple[tuple[str, str], ...]
    probes: tuple[tuple[str, str], ...]
    timestamp_policy: dict[str, Any]
    output_policy: dict[str, Any]
    status: str = "configured"


class GStreamerRuntimeFactory:
    def __init__(self) -> None:
        self._gst = None
        self._available = False
        self._pyds = None
        self._load()
        self._load_pyds()

    @property
    def available(self) -> bool:
        return self._available

    def create_elements(self, blueprint: PipelineBlueprint) -> dict[str, Any]:
        if not self._available or self._gst is None:
            return {}

        elements: dict[str, Any] = {}
        for node in blueprint.nodes:
            if node.element == "gst_pipeline":
                continue
            element = self._gst.ElementFactory.make(node.element, node.name)
            if element is None:
                raise RuntimeError(f"failed to create GStreamer element `{node.element}` as `{node.name}`")
            for key, value in node.properties.items():
                if node.element == "capsfilter" and key == "caps" and isinstance(value, str):
                    value = self._gst.Caps.from_string(value)
                self._set_property_if_available(element, key, value)
            elements[node.name] = element
        return elements

    def create_pipeline(self, name: str) -> Any | None:
        if not self._available or self._gst is None:
            return None
        return self._gst.Pipeline.new(name)

    def buffer_probe_type(self) -> Any | None:
        if not self._available or self._gst is None:
            return None
        return self._gst.PadProbeType.BUFFER

    @property
    def gst(self) -> Any | None:
        return self._gst

    @property
    def pyds(self) -> Any | None:
        return self._pyds

    def _load(self) -> None:
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst

            Gst.init(None)
            self._gst = Gst
            self._available = True
        except Exception as exc:
            logging.warning("GStreamer runtime unavailable: %s", exc)
            self._gst = None
            self._available = False

    def _load_pyds(self) -> None:
        try:
            import pyds  # type: ignore

            self._pyds = pyds
        except Exception as exc:
            logging.warning("DeepStream pyds bindings unavailable: %s", exc)
            self._pyds = None

    def _set_property_if_available(self, element: Any, key: str, value: Any) -> None:
        if hasattr(element, "find_property") and element.find_property(key) is None:
            logging.warning(
                "GStreamer element `%s` does not support property `%s`; skipping it",
                getattr(element, "name", element),
                key,
            )
            return
        element.set_property(key, value)


class PipelineBuilder:
    def __init__(self, settings) -> None:
        self.settings = settings
        self._runtime_factory = GStreamerRuntimeFactory()
        self._probe_warning_counts: dict[str, int] = {}
        self._forced_output_sink: str | None = None
        self._stream_track_id_maps: dict[int, dict[int, int]] = {}
        self._stream_next_track_ids: dict[int, int] = {}
        self._osd_last_label_state: dict[tuple[int, int], tuple[str, float, int]] = {}
        self._plate_annotations: dict[tuple[int, int], dict[str, Any]] = {}
        self._plate_annotations_lock = Lock()
        self._probe_metrics_lock = Lock()
        self._probe_metrics_state: dict[str, int] = {
            "probe_batches": 0,
            "native_calls": 0,
            "native_total_ns": 0,
            "python_calls": 0,
            "python_total_ns": 0,
            "python_fallback_calls": 0,
            "osd_frames": 0,
            "osd_objects": 0,
            "osd_updates": 0,
        }
        self._cpp_probe = CppProbeHandler(
            getattr(self.settings.deepstream, "probe_handler_path", None)
        )

    def register_plate_annotation(self, stream_id: str, track_id: int, event: dict[str, Any]) -> None:
        try:
            source_id = int(str(stream_id).rsplit("-", 1)[-1])
        except (TypeError, ValueError):
            return
        with self._plate_annotations_lock:
            self._plate_annotations[(source_id, int(track_id))] = dict(event)

    def _plate_annotation(self, source_id: int, local_track_id: int) -> dict[str, Any] | None:
        with self._plate_annotations_lock:
            return self._plate_annotations.get((source_id, local_track_id))

    def build(self) -> PipelineBlueprint:
        self.settings.validate()

        source_factory = SourceFactory(getattr(self.settings, "sources", ()))
        sources = tuple(source_factory.list_sources())
        branches = source_factory.build_branches()
        source_count = source_factory.count_enabled() or self.settings.source_count
        self._validate_source_count(source_count)
        self._validate_sources(sources)
        self._validate_deepstream_paths()
        self._validate_output_policy()

        nodes = self._build_nodes(branches)
        links = self._build_links(branches)
        probes = self._build_probes()
        timestamp_policy = self._build_timestamp_policy()
        output_policy = self._build_output_policy()

        return PipelineBlueprint(
            app_name=getattr(self.settings, "app_name", "deepstream-multistream"),
            source_count=source_count,
            sources=sources,
            nodes=nodes,
            links=links,
            probes=probes,
            timestamp_policy=timestamp_policy,
            output_policy=output_policy,
        )

    def build_runtime(self) -> dict[str, Any]:
        blueprint = self.build()
        runtime = {
            "blueprint": blueprint,
            "gstreamer_available": self._runtime_factory.available,
            "elements": {},
            "static_links": blueprint.links,
            "dynamic_links": self._build_dynamic_links(blueprint),
            "streammux_requests": (),
            "probe_attachments": (),
            "assembly_steps": (
                "create_pipeline",
                "add_elements_to_pipeline",
                "link_static_elements",
                "prepare_dynamic_pad_handlers",
                "prepare_streammux_requests",
                "register_probe_points",
            ),
            "probe_points": blueprint.probes,
            "runtime_flags": self._build_runtime_flags(blueprint),
            "gst": self._runtime_factory.gst,
        }
        if not self._runtime_factory.available:
            runtime["dynamic_links"] = self.prepare_dynamic_pad_handlers(runtime)
            runtime["streammux_requests"] = self.prepare_streammux_requests(runtime)
            logging.warning("GStreamer runtime is not available in the current environment")
            return runtime

        try:
            return self._assemble_runtime(runtime, blueprint)
        except RuntimeError as exc:
            if not self._should_fallback_to_fake_output(exc):
                raise
            logging.warning("output hardware path failed, falling back to fakesink: %s", exc)
            self._forced_output_sink = "fake"
            blueprint = self.build()
            runtime = {
                **runtime,
                "blueprint": blueprint,
                "static_links": blueprint.links,
                "dynamic_links": self._build_dynamic_links(blueprint),
                "runtime_flags": self._build_runtime_flags(blueprint),
                "probe_points": blueprint.probes,
            }
            return self._assemble_runtime(runtime, blueprint)

    def build_runtime_with_fake_output(self) -> dict[str, Any]:
        if not bool(getattr(self.settings.deepstream, "enable_hardware_fallback", True)):
            raise RuntimeError("hardware fallback is disabled")
        if getattr(self.settings.deepstream, "output_sink", "rtmp") == "fake":
            raise RuntimeError("output sink is already fake")
        self._forced_output_sink = "fake"
        return self.build_runtime()

    def has_output_fallback_active(self) -> bool:
        return self._forced_output_sink == "fake"

    def _assemble_runtime(self, runtime: dict[str, Any], blueprint: PipelineBlueprint) -> dict[str, Any]:
        runtime["pipeline"] = self._runtime_factory.create_pipeline(blueprint.app_name)
        runtime["elements"] = self._runtime_factory.create_elements(blueprint)
        self.add_elements_to_pipeline(runtime)
        runtime["linked_pairs"] = self.link_static_elements(runtime)
        runtime["dynamic_links"] = self.prepare_dynamic_pad_handlers(runtime)
        runtime["streammux_requests"] = self.prepare_streammux_requests(runtime)
        return runtime

    def _should_fallback_to_fake_output(self, exc: RuntimeError) -> bool:
        if not (
            bool(getattr(self.settings.deepstream, "enable_hardware_fallback", True))
            and self._forced_output_sink is None
            and getattr(self.settings.deepstream, "output_sink", "rtmp") != "fake"
        ):
            return False
        text = str(exc)
        output_markers = (
            "post-osd-convert",
            "encoder-caps",
            "encoder",
            "nvv4l2h264enc",
            "h264-parser",
            "output-mux",
            "qtmux",
            "flvmux",
            "filesink",
            "rtmpsink",
            "sink",
        )
        return any(marker in text for marker in output_markers)

    def _build_nodes(self, branches: tuple[SourceBranchSpec, ...]) -> tuple[PipelineNodeSpec, ...]:
        ds = self.settings.deepstream
        tracker_enabled = bool(getattr(ds, "enable_tracker", True))
        osd_enabled = bool(getattr(ds, "enable_osd", True))
        output_sink = self._effective_output_sink()
        use_fake_sink = output_sink == "fake"
        use_file_sink = output_sink == "file"
        use_rtsp_sink = output_sink == "rtsp"
        tiler_enabled = bool(getattr(ds, "enable_tiler", False)) and len(branches) > 1
        live_source = any(branch.source.is_rtsp for branch in branches)
        nodes: list[PipelineNodeSpec] = [
            PipelineNodeSpec(
                "pipeline",
                "gst_pipeline",
                "root",
                {},
                {"required": True, "live": True, "supports_probe": False, "hardware_accelerated": False},
            ),
            PipelineNodeSpec(
                "streammux",
                "nvstreammux",
                "mux",
                {
                    "batch-size": ds.batch_size,
                    "batched-push-timeout": ds.batched_push_timeout_us,
                    "width": ds.inference_width,
                    "height": ds.inference_height,
                    "live-source": live_source,
                    "attach-sys-ts": True,
                    "enable-padding": True,
                },
                {"required": True, "live": True, "supports_probe": False, "hardware_accelerated": True},
            ),
            PipelineNodeSpec(
                "primary-infer",
                "nvinfer",
                "infer",
                {
                    "config-file-path": str(ds.infer_config_path),
                    "model-engine-file": str(ds.model_engine_path),
                    "batch-size": ds.batch_size,
                },
                {"required": True, "live": True, "supports_probe": True, "hardware_accelerated": True},
            ),
            PipelineNodeSpec(
                "post-osd-queue",
                "queue",
                "output",
                self._queue_properties(max_buffers=8),
                {"required": True, "live": True, "supports_probe": False, "hardware_accelerated": False},
            ),
            PipelineNodeSpec(
                "sink",
                self._build_sink_element_name(output_sink),
                "output",
                self._build_sink_properties(output_sink),
                {"required": True, "live": True, "supports_probe": False, "hardware_accelerated": False},
            ),
        ]

        if osd_enabled:
            insert_at = 4 if tracker_enabled else 3
            nodes[insert_at:insert_at] = [
                PipelineNodeSpec(
                    "pre-osd-convert",
                    "nvvideoconvert",
                    "overlay",
                    {},
                    {"required": True, "live": True, "supports_probe": False, "hardware_accelerated": True},
                ),
                PipelineNodeSpec(
                    "pre-osd-caps",
                    "capsfilter",
                    "overlay",
                    {"caps": "video/x-raw(memory:NVMM),format=RGBA"},
                    {"required": True, "live": True, "supports_probe": False, "hardware_accelerated": False},
                ),
                PipelineNodeSpec(
                    "osd",
                    "nvdsosd",
                    "overlay",
                    {},
                    {"required": True, "live": True, "supports_probe": True, "hardware_accelerated": True},
                ),
            ]

        if tiler_enabled:
            insert_at = 4 if tracker_enabled else 3
            nodes.insert(
                insert_at,
                PipelineNodeSpec(
                    "tiler",
                    "nvmultistreamtiler",
                    "compose",
                    {
                        "rows": ds.tiler_rows,
                        "columns": ds.tiler_columns,
                        "width": ds.tiler_width,
                        "height": ds.tiler_height,
                    },
                    {"required": True, "live": True, "supports_probe": False, "hardware_accelerated": True},
                ),
            )
            nodes.extend(
                [
                    PipelineNodeSpec(
                        "pre-tiler-convert",
                        "nvvideoconvert",
                        "compose",
                        {},
                        {"required": True, "live": True, "supports_probe": False, "hardware_accelerated": True},
                    ),
                    PipelineNodeSpec(
                        "pre-tiler-caps",
                        "capsfilter",
                        "compose",
                        {"caps": "video/x-raw(memory:NVMM),format=RGBA"},
                        {"required": True, "live": True, "supports_probe": False, "hardware_accelerated": False},
                    ),
                ]
            )

        if not use_fake_sink:
            nodes[5:5] = [
                PipelineNodeSpec(
                    "post-osd-convert",
                    "nvvideoconvert",
                    "encode",
                    {},
                    {"required": True, "live": True, "supports_probe": False, "hardware_accelerated": True},
                ),
                PipelineNodeSpec(
                    "encoder-caps",
                    "capsfilter",
                    "encode",
                    {"caps": self._build_encoder_caps(ds, tiler_enabled=tiler_enabled)},
                    {"required": True, "live": True, "supports_probe": False, "hardware_accelerated": False},
                ),
                PipelineNodeSpec(
                    "encoder",
                    "nvv4l2h264enc",
                    "encode",
                    {
                        "bitrate": int(getattr(ds, "encoder_bitrate", 4000000)),
                        "insert-sps-pps": True,
                    },
                    {"required": True, "live": True, "supports_probe": False, "hardware_accelerated": True},
                ),
                PipelineNodeSpec(
                    "h264-parser",
                    "h264parse",
                    "encode",
                    {},
                    {"required": True, "live": True, "supports_probe": False, "hardware_accelerated": False},
                ),
                *self._build_output_mux_nodes(use_file_sink=use_file_sink, use_rtsp_sink=use_rtsp_sink),
            ]

        if tracker_enabled:
            nodes.insert(
                3,
                PipelineNodeSpec(
                    "tracker",
                    "nvtracker",
                    "track",
                    self._build_tracker_properties(),
                    {"required": True, "live": True, "supports_probe": True, "hardware_accelerated": True},
                ),
            )

        for branch in branches:
            nodes.extend(
                PipelineNodeSpec(
                    name=node.name,
                    element=node.element,
                    stage=node.stage,
                    properties=self._merge_branch_properties(node),
                    flags=node.flags,
                )
                for node in branch.nodes
            )

        return tuple(nodes)

    def _build_links(self, branches: tuple[SourceBranchSpec, ...]) -> tuple[tuple[str, str], ...]:
        links: list[tuple[str, str]] = []
        use_rtsp_sink = self._effective_output_sink() == "rtsp"
        for branch in branches:
            links.extend(branch.links)
            links.append((branch.mux_input, "streammux"))

        links.extend(
            [
                ("streammux", "primary-infer"),
            ]
        )
        current = "primary-infer"
        if self.settings.deepstream.enable_tracker:
            links.append((current, "tracker"))
            current = "tracker"
        if bool(getattr(self.settings.deepstream, "enable_tiler", False)) and len(branches) > 1:
            links.extend(
                [
                    (current, "pre-tiler-convert"),
                    ("pre-tiler-convert", "pre-tiler-caps"),
                    ("pre-tiler-caps", "tiler"),
                ]
            )
            current = "tiler"
        if getattr(self.settings.deepstream, "enable_osd", True):
            links.extend(
                [
                    (current, "pre-osd-convert"),
                    ("pre-osd-convert", "pre-osd-caps"),
                    ("pre-osd-caps", "osd"),
                ]
            )
            current = "osd"
        links.append((current, "post-osd-queue"))
        if self._effective_output_sink() == "fake":
            links.append(("post-osd-queue", "sink"))
        else:
            links.extend(
                [
                    ("post-osd-queue", "post-osd-convert"),
                    ("post-osd-convert", "encoder-caps"),
                    ("encoder-caps", "encoder"),
                    ("encoder", "h264-parser"),
                    *([("h264-parser", "rtsp-pay"), ("rtsp-pay", "sink")] if use_rtsp_sink else [("h264-parser", "output-mux"), ("output-mux", "sink")]),
                ]
            )
        return tuple(links)

    def _build_probes(self) -> tuple[tuple[str, str], ...]:
        probes: list[tuple[str, str]] = []
        if bool(getattr(self.settings.optimization, "enable_fps_control", True)):
            probes.append(("primary-infer", "sink"))
        if (
            getattr(self.settings.deepstream, "enable_osd", True)
            and bool(getattr(self.settings.deepstream, "enable_tiler", False))
            and self.settings.effective_source_count() > 1
        ):
            # Keep metadata before tiler and capture RGBA frames immediately
            # before tiler. Tiler output no longer owns per-source metadata.
            probes.append(("tracker", "src") if self.settings.deepstream.enable_tracker else ("primary-infer", "src"))
            probes.append(("pre-tiler-caps", "sink"))
            return tuple(probes)
        if getattr(self.settings.deepstream, "enable_osd", True):
            probes.append(("osd", "sink"))
        else:
            probes.append(("primary-infer", "src"))
        return tuple(probes)

    def _build_timestamp_policy(self) -> dict[str, Any]:
        live_source = any(source.kind == "rtsp" for source in getattr(self.settings, "sources", ()))
        return {
            "timestamp_mode": "system_ntp",
            "sync_enabled": True,
            "live_source": live_source,
            "attach_sys_ts_as_ntp": True,
            "result_fields": ("stream_id", "frame_id", "timestamp"),
        }

    def _build_output_policy(self) -> dict[str, Any]:
        output_sink = self._effective_output_sink()
        return {
            "enable_osd": getattr(self.settings.deepstream, "enable_osd", True),
            "enable_encoder": output_sink != "fake",
            "enable_rtmp_sink": output_sink == "rtmp",
            "enable_rtsp_sink": output_sink == "rtsp",
            "enable_file_sink": output_sink == "file",
            "enable_tiler": bool(getattr(self.settings.deepstream, "enable_tiler", False)),
            "fallback_output_sink": self._forced_output_sink,
            "enable_hardware_fallback": bool(getattr(self.settings.deepstream, "enable_hardware_fallback", True)),
            "enable_last_frame_keepalive": bool(getattr(self.settings.deepstream, "enable_last_frame_keepalive", True)),
            "last_frame_keepalive_timeout_ms": int(
                getattr(self.settings.deepstream, "last_frame_keepalive_timeout_ms", 1000)
            ),
            "enable_json_output": self.settings.output.enable_jsonl,
            "enable_mqtt_output": self.settings.output.enable_mqtt,
            "enable_kafka_output": self.settings.output.enable_kafka,
        }

    def _build_sink_element_name(self, output_sink: str) -> str:
        if output_sink == "fake":
            return "fakesink"
        if output_sink == "file":
            return "filesink"
        if output_sink == "rtsp":
            return "rtspclientsink"
        return "rtmpsink"

    def _build_sink_properties(self, output_sink: str) -> dict[str, Any]:
        if output_sink == "fake":
            return {"sync": False}
        if output_sink == "file":
            path = self.settings.deepstream.output_video_path
            path.parent.mkdir(parents=True, exist_ok=True)
            return {"location": str(path), "sync": False}
        return {"location": self.settings.deepstream.output_url}

    def _build_output_mux_nodes(
        self,
        *,
        use_file_sink: bool,
        use_rtsp_sink: bool,
    ) -> tuple[PipelineNodeSpec, ...]:
        if use_rtsp_sink:
            return (
                PipelineNodeSpec(
                    "rtsp-pay",
                    "rtph264pay",
                    "encode",
                    {"pt": 96, "config-interval": 1},
                    {"required": True, "live": True, "supports_probe": False, "hardware_accelerated": False},
                ),
            )
        return (
            PipelineNodeSpec(
                "output-mux",
                "qtmux" if use_file_sink else "flvmux",
                "output",
                {"streamable": 1} if not use_file_sink else {},
                {"required": True, "live": True, "supports_probe": False, "hardware_accelerated": False},
            ),
        )

    def _build_encoder_caps(self, ds: object, *, tiler_enabled: bool) -> str:
        width = int(getattr(ds, "tiler_width", 0) if tiler_enabled else getattr(ds, "inference_width", 0))
        height = int(getattr(ds, "tiler_height", 0) if tiler_enabled else getattr(ds, "inference_height", 0))
        if width > 0 and height > 0:
            return f"video/x-raw(memory:NVMM),format=NV12,width={width},height={height}"
        return "video/x-raw(memory:NVMM),format=NV12"

    def _build_tracker_properties(self) -> dict[str, Any]:
        tracker_path = self.settings.deepstream.tracker_config_path
        defaults: dict[str, Any] = {
            "tracker-width": 640,
            "tracker-height": 640,
            "gpu-id": 0,
            "ll-lib-file": "/opt/nvidia/deepstream/deepstream-7.1/lib/libnvds_nvmultiobjecttracker.so",
            "ll-config-file": "/opt/nvidia/deepstream/deepstream-7.1/samples/configs/deepstream-app/config_tracker_IOU.yml",
        }
        if not tracker_path.exists():
            return defaults

        try:
            import yaml

            with tracker_path.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except Exception:
            logging.warning("failed to load tracker config `%s`; using defaults", tracker_path)
            return defaults

        return {
            **defaults,
            **{key: value for key, value in raw.items() if value is not None},
        }

    def _validate_source_count(self, enabled_count: int) -> None:
        expected = getattr(self.settings, "source_count", 0)
        if enabled_count != expected:
            raise ValueError(
                f"enabled source count ({enabled_count}) does not match configured source_count ({expected})"
            )

    def _validate_sources(self, sources: tuple[SourceSpec, ...]) -> None:
        if not sources:
            raise ValueError("at least one enabled source is required")

        for source in sources:
            if not source.uri:
                raise ValueError(f"source `{source.name}` has an empty uri")
            if source.is_rtsp and not source.uri.startswith("rtsp://"):
                raise ValueError(f"rtsp source `{source.name}` must start with rtsp://")
            if source.is_file and not (source.uri.startswith("file://") or Path(source.uri).exists()):
                raise ValueError(
                    f"file source `{source.name}` must be a valid file path or file:// uri"
                )

    def _validate_deepstream_paths(self) -> None:
        ds = self.settings.deepstream
        required_paths = {
            "model_engine_path": ds.model_engine_path,
            "custom_lib_path": ds.custom_lib_path,
            "tracker_config_path": ds.tracker_config_path,
            "infer_config_path": ds.infer_config_path,
            "streammux_config_path": ds.streammux_config_path,
        }
        for label, path in required_paths.items():
            if not str(path).strip():
                raise ValueError(f"{label} must not be empty")

    def _validate_output_policy(self) -> None:
        output = self.settings.output
        if output.enable_mqtt and not output.mqtt_topic:
            raise ValueError("mqtt_topic must be set when MQTT output is enabled")
        if not output.enable_jsonl and not output.enable_mqtt and not output.enable_kafka:
            raise ValueError("at least one structured output must be enabled")

    def _merge_branch_properties(self, node) -> dict[str, Any]:
        properties = dict(node.properties)
        if node.name.startswith("pre-mux-queue"):
            properties.update(self._queue_properties(max_buffers=self.settings.optimization.max_queue_size))
        return properties

    def _effective_output_sink(self) -> str:
        return self._forced_output_sink or getattr(self.settings.deepstream, "output_sink", "rtmp")

    def _queue_properties(self, *, max_buffers: int) -> dict[str, Any]:
        properties: dict[str, Any] = {
            "max-size-buffers": max_buffers,
            "max-size-time": 0,
            "max-size-bytes": 0,
        }
        if bool(getattr(self.settings.optimization, "enable_drop_old_frames", True)):
            properties["leaky"] = 2
        return properties

    def _build_dynamic_links(self, blueprint: PipelineBlueprint) -> tuple[dict[str, Any], ...]:
        dynamic_links: list[dict[str, Any]] = []
        for node in blueprint.nodes:
            if node.flags.get("dynamic_pad"):
                dynamic_links.append(
                    {
                        "source": node.name,
                        "element": node.element,
                        "stage": node.stage,
                        "pad_type": "src",
                        "mode": "dynamic-pad",
                    }
                )
        return tuple(dynamic_links)

    def _build_runtime_flags(self, blueprint: PipelineBlueprint) -> dict[str, Any]:
        return {
            "source_count": blueprint.source_count,
            "live_source": blueprint.timestamp_policy.get("live_source", True),
            "sync_enabled": blueprint.timestamp_policy.get("sync_enabled", True),
            "enable_rtmp_sink": blueprint.output_policy.get("enable_rtmp_sink", True),
            "enable_tiler": blueprint.output_policy.get("enable_tiler", False),
            "output_sink": self._effective_output_sink(),
            "requested_output_sink": getattr(self.settings.deepstream, "output_sink", "rtmp"),
            "fallback_output_sink": self._forced_output_sink,
            "enable_json_output": blueprint.output_policy.get("enable_json_output", True),
        }

    def add_elements_to_pipeline(self, runtime: dict[str, Any]) -> None:
        pipeline = runtime.get("pipeline")
        elements = runtime.get("elements", {})
        if pipeline is None or not elements:
            return
        for element in elements.values():
            added = pipeline.add(element)
            if added is False:
                raise RuntimeError(f"failed to add element `{getattr(element, 'name', element)}` to pipeline")

    def link_static_elements(self, runtime: dict[str, Any]) -> tuple[tuple[str, str], ...]:
        elements = runtime.get("elements", {})
        static_links: tuple[tuple[str, str], ...] = runtime.get("static_links", ())
        linked_pairs: list[tuple[str, str]] = []
        if not elements:
            return static_links

        dynamic_sources = {plan["source"] for plan in runtime.get("dynamic_links", ())}
        for source_name, target_name in static_links:
            if source_name in dynamic_sources or target_name == "streammux":
                continue
            source_element = elements.get(source_name)
            target_element = elements.get(target_name)
            if source_element is None or target_element is None:
                raise RuntimeError(f"missing element for static link: {source_name} -> {target_name}")
            linked = source_element.link(target_element)
            if linked is False:
                raise RuntimeError(f"failed to link elements: {source_name} -> {target_name}")
            linked_pairs.append((source_name, target_name))
        return tuple(linked_pairs)

    def prepare_dynamic_pad_handlers(self, runtime: dict[str, Any]) -> tuple[dict[str, Any], ...]:
        blueprint: PipelineBlueprint = runtime["blueprint"]
        dynamic_links = self._build_dynamic_links(blueprint)
        elements = runtime.get("elements", {})

        for plan in dynamic_links:
            element = elements.get(plan["source"])
            if element is None:
                continue
            target_name = self._resolve_dynamic_target(plan["source"], blueprint)
            target = elements.get(target_name)
            if target is None:
                raise RuntimeError(f"missing dynamic link target `{target_name}` for `{plan['source']}`")
            if hasattr(element, "connect"):
                element.connect("pad-added", self._on_dynamic_pad_added, {"target": target})

        runtime["dynamic_links"] = dynamic_links
        return dynamic_links

    def prepare_streammux_requests(self, runtime: dict[str, Any]) -> tuple[dict[str, Any], ...]:
        blueprint: PipelineBlueprint = runtime["blueprint"]
        requests: list[dict[str, Any]] = []
        elements = runtime.get("elements", {})
        streammux = elements.get("streammux")
        for index in range(blueprint.source_count):
            queue_name = f"pre-mux-queue-{index + 1}"
            request = {
                "source": queue_name,
                "target": "streammux",
                "request_pad": f"sink_{index}",
                "src_pad": "src",
            }
            requests.append(request)
            if streammux is not None:
                if not hasattr(streammux, "get_request_pad"):
                    raise RuntimeError("streammux element does not support request pads")
                request_pad = streammux.get_request_pad(request["request_pad"])
                if request_pad is None:
                    raise RuntimeError(f"failed to request pad `{request['request_pad']}` from streammux")
                queue_element = elements.get(queue_name)
                if queue_element is None:
                    raise RuntimeError(f"missing source queue element `{queue_name}` for streammux link")
                src_pad = queue_element.get_static_pad("src")
                if src_pad is None:
                    raise RuntimeError(f"missing src pad on `{queue_name}`")
                link_result = src_pad.link(request_pad)
                if link_result not in (None, 0):
                    raise RuntimeError(f"failed to link `{queue_name}` to streammux pad `{request['request_pad']}`")
        runtime["streammux_requests"] = tuple(requests)
        return tuple(requests)

    def attach_probe_points(self, runtime: dict[str, Any]) -> tuple[dict[str, Any], ...]:
        blueprint: PipelineBlueprint = runtime["blueprint"]
        attachments: list[dict[str, Any]] = []
        elements = runtime.get("elements", {})
        probe_type = self._runtime_factory.buffer_probe_type()
        for element_name, pad_name in blueprint.probes:
            attachments.append(
                {
                    "element": element_name,
                    "pad": pad_name,
                    "mode": "buffer",
                }
            )
            element = elements.get(element_name)
            if element is None:
                continue
            pad = element.get_static_pad(pad_name)
            if pad is None:
                raise RuntimeError(f"failed to get pad `{pad_name}` from `{element_name}`")
            if hasattr(pad, "add_probe"):
                user_data = {
                    "probe_registry": runtime.get("probe_registry"),
                    "meta_parser": runtime.get("meta_parser"),
                    "cpp_probe": self._cpp_probe,
                    "mode": (
                        "pre_infer_gate" if (element_name, pad_name) == ("primary-infer", "sink")
                        else "frame_only" if element_name == "pre-tiler-caps"
                        else "result"
                    ),
                    "frame_gate": runtime.get("frame_gate"),
                    "frame_store": runtime.get("frame_store"),
                }
                try:
                    pad.add_probe(
                        probe_type if probe_type is not None else 0,
                        self._on_probe_buffer,
                        user_data,
                    )
                except TypeError:
                    pad.add_probe(
                        probe_type if probe_type is not None else 0,
                        lambda probe_pad, probe_info: self._on_probe_buffer(probe_pad, probe_info, user_data),
                    )
        runtime["probe_attachments"] = tuple(attachments)
        return tuple(attachments)

    def _resolve_dynamic_target(self, source_name: str, blueprint: PipelineBlueprint) -> str:
        for left, right in blueprint.links:
            if left == source_name:
                return right
        raise RuntimeError(f"no dynamic target found for `{source_name}`")

    def _on_dynamic_pad_added(self, src, pad, user_data) -> None:
        _ = src
        target = user_data["target"]
        sink_pad = target.get_static_pad("sink")
        if sink_pad is None:
            raise RuntimeError(f"target `{getattr(target, 'name', target)}` has no sink pad")
        if hasattr(sink_pad, "is_linked") and sink_pad.is_linked():
            return
        media_type = self._dynamic_pad_media_type(pad)
        if not self._is_video_dynamic_pad(media_type):
            logging.debug("ignoring non-video dynamic pad: %s", media_type or "unknown")
            return
        result = pad.link(sink_pad)
        if result not in (None, 0):
            if str(result).endswith("WAS_LINKED"):
                return
            raise RuntimeError(
                f"failed to dynamically link pad to `{getattr(target, 'name', target)}`"
            )
        logging.info("dynamic pad linked: %s -> %s (%s)", getattr(src, "name", src), getattr(target, "name", target), media_type)

    def _dynamic_pad_media_type(self, pad) -> str:
        caps = pad.get_current_caps() if hasattr(pad, "get_current_caps") else None
        if (caps is None or caps.get_size() == 0) and hasattr(pad, "query_caps"):
            caps = pad.query_caps(None)
        if caps is None or caps.get_size() == 0:
            return ""
        structure = caps.get_structure(0)
        return structure.get_name() if structure is not None else ""

    def _is_video_dynamic_pad(self, media_type: str) -> bool:
        return media_type.startswith("video/") or media_type == "application/x-rtp"

    def _on_probe_buffer(self, pad, info, user_data=None) -> int:
        _ = pad
        user_data = user_data or {}
        if user_data.get("mode") == "pre_infer_gate":
            gate = user_data.get("frame_gate")
            dropped = gate() if gate is not None else False
            gst = self._runtime_factory.gst
            if gst is not None and hasattr(gst, "PadProbeReturn"):
                return gst.PadProbeReturn.DROP if dropped else gst.PadProbeReturn.OK
            return 0 if dropped else 1
        if user_data.get("mode") == "frame_only":
            self._capture_probe_frames(info, user_data.get("frame_store"))
            gst = self._runtime_factory.gst
            if gst is not None and hasattr(gst, "PadProbeReturn"):
                return gst.PadProbeReturn.OK
            return 1
        payload = self._extract_probe_payload(info)
        registry = None if user_data is None else user_data.get("probe_registry")
        parser = None if user_data is None else user_data.get("meta_parser")
        if registry is not None and parser is not None:
            registry.emit_probe_payload(payload, parser)
        gst = self._runtime_factory.gst
        if gst is not None and hasattr(gst, "PadProbeReturn"):
            return gst.PadProbeReturn.OK
        return 1

    def _capture_probe_frames(self, info: object, frame_store: object | None) -> None:
        """Copy only configured task streams out of the DeepStream surface."""
        if frame_store is None or not hasattr(frame_store, "should_capture"):
            return
        buffer = self._extract_probe_buffer(info)
        pyds = self._runtime_factory.pyds
        if buffer is None or pyds is None or not hasattr(pyds, "get_nvds_buf_surface"):
            return
        batch_meta = self._extract_nvds_batch_meta(info)
        if batch_meta is None:
            return
        try:
            import numpy as np

            frame_list = getattr(batch_meta, "frame_meta_list", None)
            for frame_meta in self._iterate_glist(frame_list, "NvDsFrameMeta"):
                source_id = self._safe_int(
                    self._safe_get(frame_meta, "source_id", self._safe_get(frame_meta, "pad_index", 0)),
                    0,
                )
                stream_id = canonical_stream_id(source_id)
                if not frame_store.should_capture(stream_id):
                    continue
                batch_id = self._safe_int(self._safe_get(frame_meta, "batch_id", -1), -1)
                frame_id = self._safe_int(self._safe_get(frame_meta, "frame_num", 0), 0)
                if batch_id < 0:
                    continue
                surface = pyds.get_nvds_buf_surface(hash(buffer), batch_id)
                frame_store.put(stream_id, frame_id, np.array(surface, copy=True))
        except Exception as exc:
            self._log_probe_warning(
                "frame_store",
                "failed to capture ROI frame from DeepStream surface: %s",
                exc,
            )

    def _extract_probe_payload(self, info: object) -> object:
        self._record_probe_metric("probe_batches")
        batch_meta = self._extract_nvds_batch_meta(info)
        if batch_meta is not None:
            if bool(getattr(self.settings.deepstream, "enable_osd", True)):
                osd_stats = self._apply_osd_track_labels(batch_meta)
                self._add_probe_metrics(**osd_stats)
            if self._cpp_probe.available:
                try:
                    buffer = self._extract_probe_buffer(info)
                    if buffer is not None:
                        started_ns = time.perf_counter_ns()
                        payload = self._cpp_probe.parse_buffer(buffer)
                        self._normalize_native_track_ids(payload)
                        self._add_probe_metrics(
                            native_calls=1,
                            native_total_ns=time.perf_counter_ns() - started_ns,
                        )
                        return payload
                except Exception as exc:
                    self._record_probe_metric("python_fallback_calls")
                    self._log_probe_warning(
                        "cpp_probe",
                        "native C++ probe parser failed; falling back to Python traversal: %s",
                        exc,
                    )
            started_ns = time.perf_counter_ns()
            payload = self._batch_meta_to_payload(batch_meta)
            self._add_probe_metrics(
                python_calls=1,
                python_total_ns=time.perf_counter_ns() - started_ns,
            )
            return payload
        if isinstance(info, dict):
            return info
        if hasattr(info, "payload"):
            return getattr(info, "payload")
        return {"raw_probe_info": info}

    def _extract_probe_buffer(self, info: object) -> object | None:
        if info is None:
            return None
        if hasattr(info, "get_buffer"):
            return info.get_buffer()
        if hasattr(info, "buffer"):
            return getattr(info, "buffer")
        if isinstance(info, dict):
            return info.get("buffer")
        return None

    def _normalize_native_track_ids(self, payload: object) -> None:
        """Apply the same local/global track contract as Python metadata parsing."""
        if not isinstance(payload, dict):
            return
        frame_list = payload.get("frame_meta_list", [])
        for frame in frame_list if isinstance(frame_list, list) else ():
            if not isinstance(frame, dict):
                continue
            source_id = frame.get("source_id", frame.get("stream_id", 0))
            source_index = self._safe_int(source_id, 0)
            objects = frame.get("obj_meta_list", [])
            tracks: list[dict[str, Any]] = []
            for obj in objects if isinstance(objects, list) else ():
                if not isinstance(obj, dict):
                    continue
                global_track_id = obj.get("object_id", obj.get("global_track_id", UNTRACKED_OBJECT_ID))
                local_track_id = self._local_track_id(source_index, global_track_id)
                obj["object_id"] = global_track_id
                obj["global_track_id"] = global_track_id
                obj["track_id"] = local_track_id
                if self._is_valid_track_id(global_track_id):
                    tracks.append(
                        {
                            "track_id": local_track_id,
                            "global_track_id": global_track_id,
                            "object_id": global_track_id,
                            "class_id": obj.get("class_id", 0),
                            "class_name": obj.get("class_name", obj.get("obj_label", "unknown")),
                            "obj_label": obj.get("obj_label", obj.get("class_name", "unknown")),
                            "confidence": obj.get("confidence", 0.0),
                            "rect_params": obj.get("rect_params", {}),
                        }
                    )
            frame["tracks"] = tracks

    def _extract_nvds_batch_meta(self, info: object) -> object | None:
        if info is None:
            return None

        buffer = self._extract_probe_buffer(info)

        if buffer is None or self._runtime_factory.pyds is None:
            return None

        try:
            batch_meta = self._runtime_factory.pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        except Exception as exc:
            self._log_probe_warning(
                "extract_nvds_batch_meta",
                "failed to extract NvDsBatchMeta from probe buffer: %s",
                exc,
            )
            return None
        return batch_meta

    def _batch_meta_to_payload(self, batch_meta: object) -> dict[str, Any]:
        payload: dict[str, Any] = {"frame_meta_list": []}
        if batch_meta is None:
            return payload

        frame_list = getattr(batch_meta, "frame_meta_list", None)
        for frame_meta in self._iterate_glist(frame_list, "NvDsFrameMeta"):
            frame_payload = self._frame_meta_to_payload(frame_meta)
            payload["frame_meta_list"].append(frame_payload)

        return payload

    def _frame_meta_to_payload(self, frame_meta: object) -> dict[str, Any]:
        source_id = self._safe_get(frame_meta, "source_id", self._safe_get(frame_meta, "pad_index", 0))
        source_index = self._safe_int(source_id, 0)
        ntp_timestamp = self._safe_get(frame_meta, "ntp_timestamp", None)
        buf_pts = self._safe_get(frame_meta, "buf_pts", None)
        frame: dict[str, Any] = {
            "stream_id": canonical_stream_id(source_index),
            "source_id": source_id,
            "frame_id": self._safe_get(frame_meta, "frame_num", 0),
            "frame_num": self._safe_get(frame_meta, "frame_num", 0),
            "timestamp": ntp_timestamp,
            "ntp_timestamp": ntp_timestamp,
            "buf_pts": buf_pts,
            "obj_meta_list": [],
            "tracks": [],
        }

        obj_list = getattr(frame_meta, "obj_meta_list", None)
        for obj_meta in self._iterate_glist(obj_list, "NvDsObjectMeta"):
            object_payload = self._object_meta_to_payload(obj_meta, source_id=source_index)
            frame["obj_meta_list"].append(object_payload)
            if self._is_valid_track_id(object_payload.get("track_id")):
                frame["tracks"].append(
                    {
                        "track_id": object_payload["track_id"],
                        "global_track_id": object_payload["object_id"],
                        "object_id": object_payload["object_id"],
                        "class_id": object_payload["class_id"],
                        "class_name": object_payload["obj_label"],
                        "obj_label": object_payload["obj_label"],
                        "confidence": object_payload["confidence"],
                        "rect_params": object_payload["rect_params"],
                    }
                )
        return frame

    def _object_meta_to_payload(self, obj_meta: object, *, source_id: int) -> dict[str, Any]:
        rect = getattr(obj_meta, "rect_params", None)
        rect_payload = {
            "left": self._safe_get(rect, "left", 0.0),
            "top": self._safe_get(rect, "top", 0.0),
            "width": self._safe_get(rect, "width", 0.0),
            "height": self._safe_get(rect, "height", 0.0),
        }
        object_id = self._safe_get(obj_meta, "object_id", 0)
        local_track_id = self._local_track_id(source_id, object_id)
        return {
            "class_id": self._safe_get(obj_meta, "class_id", 0),
            "obj_label": self._safe_get(obj_meta, "obj_label", "unknown"),
            "confidence": self._safe_get(obj_meta, "confidence", 0.0),
            "track_id": local_track_id,
            "global_track_id": object_id,
            "object_id": object_id,
            "rect_params": rect_payload,
        }

    def _apply_osd_track_labels(self, batch_meta: object) -> dict[str, int]:
        stats = {"osd_frames": 0, "osd_objects": 0, "osd_updates": 0}
        frame_list = getattr(batch_meta, "frame_meta_list", None)
        for frame_meta in self._iterate_glist(frame_list, "NvDsFrameMeta"):
            stats["osd_frames"] += 1
            source_id = self._safe_int(
                self._safe_get(frame_meta, "source_id", self._safe_get(frame_meta, "pad_index", 0)),
                0,
            )
            obj_list = getattr(frame_meta, "obj_meta_list", None)
            active_track_ids: set[int] = set()
            for obj_meta in self._iterate_glist(obj_list, "NvDsObjectMeta"):
                stats["osd_objects"] += 1
                global_track_id = self._safe_get(obj_meta, "object_id", UNTRACKED_OBJECT_ID)
                if not self._is_valid_track_id(global_track_id):
                    continue
                local_track_id = self._local_track_id(source_id, global_track_id)
                active_track_ids.add(self._safe_int(global_track_id, -1))
                label = self._safe_get(obj_meta, "obj_label", "person")
                if not label or label == "unknown":
                    label = "person"
                confidence = float(self._safe_get(obj_meta, "confidence", 0.0))
                state_key = (source_id, self._safe_int(global_track_id, -1))
                annotation = self._plate_annotation(source_id, local_track_id)
                previous_state = self._osd_last_label_state.get(state_key)
                should_update = (
                    previous_state is None
                    or previous_state[0] != label
                    or previous_state[2] != local_track_id
                    or abs(previous_state[1] - confidence) >= OSD_CONFIDENCE_UPDATE_THRESHOLD
                    or annotation is not None
                )
                text_params = getattr(obj_meta, "text_params", None)
                if text_params is not None and should_update:
                    plate_text = "" if annotation is None else f" PLATE:{annotation.get('plate_text', '')}"
                    display_text = f"{label} ID:{local_track_id} {confidence:.2f}{plate_text}"
                    if getattr(text_params, "display_text", None) != display_text:
                        text_params.display_text = display_text
                        stats["osd_updates"] += 1
                if annotation is not None:
                    self._add_plate_osd(frame_meta, annotation)
                self._osd_last_label_state[state_key] = (label, confidence, local_track_id)
            for state_key in tuple(self._osd_last_label_state):
                if state_key[0] == source_id and state_key[1] not in active_track_ids:
                    del self._osd_last_label_state[state_key]
        return stats

    def _add_plate_osd(self, frame_meta: object, annotation: dict[str, Any]) -> None:
        pyds = self._runtime_factory.pyds
        if pyds is None or not hasattr(pyds, "nvds_acquire_display_meta_from_pool"):
            return
        bbox = annotation.get("plate_bbox") or {}
        try:
            display_meta = pyds.nvds_acquire_display_meta_from_pool(frame_meta)
            display_meta.num_rects = 1
            rect = display_meta.rect_params[0]
            rect.left = float(bbox.get("left", 0.0))
            rect.top = float(bbox.get("top", 0.0))
            rect.width = float(bbox.get("width", 0.0))
            rect.height = float(bbox.get("height", 0.0))
            rect.border_width = 3
            rect.has_bg_color = 0
            rect.border_color.set(0.0, 1.0, 0.0, 1.0)
            display_meta.num_labels = 1
            text = display_meta.text_params[0]
            text.display_text = f"{annotation.get('plate_text', '')} {float(annotation.get('confidence', 0.0)):.2f}"
            text.x_offset = int(rect.left)
            text.y_offset = max(int(rect.top) - 24, 0)
            text.font_params.font_name = "Sans"
            text.font_params.font_size = 16
            text.font_params.font_color.set(1.0, 1.0, 0.0, 1.0)
            text.set_bg_clr = 1
            text.text_bg_clr.set(0.0, 0.0, 0.0, 0.7)
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        except Exception as exc:
            self._log_probe_warning("plate_osd", "failed to add plate OSD: %s", exc)

    def _record_probe_metric(self, key: str, value: int = 1) -> None:
        with self._probe_metrics_lock:
            self._probe_metrics_state[key] = self._probe_metrics_state.get(key, 0) + value

    def _add_probe_metrics(self, **values: int) -> None:
        with self._probe_metrics_lock:
            for key, value in values.items():
                self._probe_metrics_state[key] = self._probe_metrics_state.get(key, 0) + int(value)

    def probe_metrics(self) -> dict[str, Any]:
        with self._probe_metrics_lock:
            state = dict(self._probe_metrics_state)
        native_calls = state["native_calls"]
        python_calls = state["python_calls"]
        return {
            "probe_batches": state["probe_batches"],
            "native_calls": native_calls,
            "native_avg_ms": round(state["native_total_ns"] / max(native_calls, 1) / 1_000_000, 3),
            "python_calls": python_calls,
            "python_avg_ms": round(state["python_total_ns"] / max(python_calls, 1) / 1_000_000, 3),
            "python_fallback_calls": state["python_fallback_calls"],
            "native_ratio": round(native_calls / max(state["probe_batches"], 1), 4),
            "osd_frames": state["osd_frames"],
            "osd_objects": state["osd_objects"],
            "osd_updates": state["osd_updates"],
        }

    def _local_track_id(self, source_id: object, global_track_id: object) -> int:
        if not self._is_valid_track_id(global_track_id):
            return self._safe_int(global_track_id, -1)
        source_index = self._safe_int(source_id, 0)
        global_id = self._safe_int(global_track_id, -1)
        stream_map = self._stream_track_id_maps.setdefault(source_index, {})
        existing = stream_map.get(global_id)
        if existing is not None:
            return existing
        next_id = self._stream_next_track_ids.get(source_index, 1)
        stream_map[global_id] = next_id
        self._stream_next_track_ids[source_index] = next_id + 1
        return next_id

    def _is_valid_track_id(self, value: object) -> bool:
        try:
            track_id = int(value)
        except (TypeError, ValueError):
            return False
        return track_id >= 0 and track_id != UNTRACKED_OBJECT_ID

    def _iterate_glist(self, node: object, meta_type: str | None = None):
        current = node
        while current is not None:
            data = getattr(current, "data", None)
            if data is not None:
                yield self._cast_meta(data, meta_type)
            current = getattr(current, "next", None)

    def _cast_meta(self, value: object, meta_type: str | None = None) -> object:
        if self._runtime_factory.pyds is None or value is None or meta_type is None:
            return value
        cls = getattr(self._runtime_factory.pyds, meta_type, None)
        if cls is not None and hasattr(cls, "cast"):
            try:
                return cls.cast(value)
            except Exception as exc:
                self._log_probe_warning(
                    f"cast_meta_{meta_type}",
                    "failed to cast DeepStream %s metadata: %s",
                    meta_type,
                    exc,
                )
                return value
        return value

    def _log_probe_warning(self, key: str, message: str, *args: Any) -> None:
        count = self._probe_warning_counts.get(key, 0) + 1
        self._probe_warning_counts[key] = count
        if count <= 3 or count % 100 == 0:
            logging.warning("%s (count=%s)", message % args, count)

    def _safe_get(self, obj: object, attr: str, default: Any) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)

    def _safe_int(self, value: object, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
