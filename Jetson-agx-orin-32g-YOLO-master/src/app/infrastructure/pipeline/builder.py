from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging

from app.infrastructure.pipeline.source_factory import SourceBranchSpec, SourceFactory, SourceSpec

UNTRACKED_OBJECT_ID = 0xFFFFFFFFFFFFFFFF


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
                "attach_probe_points",
            ),
            "probe_points": blueprint.probes,
            "runtime_flags": self._build_runtime_flags(blueprint),
            "gst": self._runtime_factory.gst,
        }
        if not self._runtime_factory.available:
            runtime["dynamic_links"] = self.prepare_dynamic_pad_handlers(runtime)
            runtime["streammux_requests"] = self.prepare_streammux_requests(runtime)
            runtime["probe_attachments"] = self.attach_probe_points(runtime)
            logging.warning("GStreamer runtime is not available in the current environment")
            return runtime

        runtime["pipeline"] = self._runtime_factory.create_pipeline(blueprint.app_name)
        runtime["elements"] = self._runtime_factory.create_elements(blueprint)
        self.add_elements_to_pipeline(runtime)
        runtime["linked_pairs"] = self.link_static_elements(runtime)
        runtime["dynamic_links"] = self.prepare_dynamic_pad_handlers(runtime)
        runtime["streammux_requests"] = self.prepare_streammux_requests(runtime)
        runtime["probe_attachments"] = self.attach_probe_points(runtime)
        return runtime

    def _build_nodes(self, branches: tuple[SourceBranchSpec, ...]) -> tuple[PipelineNodeSpec, ...]:
        ds = self.settings.deepstream
        tracker_enabled = bool(getattr(ds, "enable_tracker", True))
        osd_enabled = bool(getattr(ds, "enable_osd", True))
        output_sink = getattr(ds, "output_sink", "rtmp")
        use_fake_sink = output_sink == "fake"
        use_file_sink = output_sink == "file"
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
                    "custom-lib-path": str(ds.custom_lib_path),
                },
                {"required": True, "live": True, "supports_probe": True, "hardware_accelerated": True},
            ),
            PipelineNodeSpec(
                "post-osd-queue",
                "queue",
                "output",
                {"max-size-buffers": 8},
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
                    {"caps": "video/x-raw(memory:NVMM),format=NV12"},
                    {"required": True, "live": True, "supports_probe": False, "hardware_accelerated": False},
                ),
                PipelineNodeSpec(
                    "encoder",
                    "nvv4l2h264enc",
                    "encode",
                    {"bitrate": 4000000},
                    {"required": True, "live": True, "supports_probe": False, "hardware_accelerated": True},
                ),
                PipelineNodeSpec(
                    "h264-parser",
                    "h264parse",
                    "encode",
                    {},
                    {"required": True, "live": True, "supports_probe": False, "hardware_accelerated": False},
                ),
                PipelineNodeSpec(
                    "output-mux",
                    "qtmux" if use_file_sink else "flvmux",
                    "output",
                    {"streamable": 1} if not use_file_sink else {},
                    {"required": True, "live": True, "supports_probe": False, "hardware_accelerated": False},
                ),
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
            links.append((current, "tiler"))
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
        if getattr(self.settings.deepstream, "output_sink", "rtmp") == "fake":
            links.append(("post-osd-queue", "sink"))
        else:
            links.extend(
                [
                    ("post-osd-queue", "post-osd-convert"),
                    ("post-osd-convert", "encoder-caps"),
                    ("encoder-caps", "encoder"),
                    ("encoder", "h264-parser"),
                    ("h264-parser", "output-mux"),
                    ("output-mux", "sink"),
                ]
            )
        return tuple(links)

    def _build_probes(self) -> tuple[tuple[str, str], ...]:
        if (
            getattr(self.settings.deepstream, "enable_osd", True)
            and bool(getattr(self.settings.deepstream, "enable_tiler", False))
            and self.settings.effective_source_count() > 1
        ):
            return (("tiler", "sink"),)
        if getattr(self.settings.deepstream, "enable_osd", True):
            probes = [("osd", "sink")]
        else:
            probes = [("primary-infer", "src")]
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
        output_sink = getattr(self.settings.deepstream, "output_sink", "rtmp")
        return {
            "enable_osd": getattr(self.settings.deepstream, "enable_osd", True),
            "enable_encoder": output_sink != "fake",
            "enable_rtmp_sink": output_sink == "rtmp",
            "enable_file_sink": output_sink == "file",
            "enable_tiler": bool(getattr(self.settings.deepstream, "enable_tiler", False)),
            "enable_json_output": self.settings.output.enable_jsonl,
            "enable_mqtt_output": self.settings.output.enable_mqtt,
            "enable_kafka_output": self.settings.output.enable_kafka,
        }

    def _build_sink_element_name(self, output_sink: str) -> str:
        if output_sink == "fake":
            return "fakesink"
        if output_sink == "file":
            return "filesink"
        return "rtmpsink"

    def _build_sink_properties(self, output_sink: str) -> dict[str, Any]:
        if output_sink == "fake":
            return {"sync": False}
        if output_sink == "file":
            path = self.settings.deepstream.output_video_path
            path.parent.mkdir(parents=True, exist_ok=True)
            return {"location": str(path), "sync": False}
        return {"location": "rtmp://127.0.0.1/live/stream"}

    def _build_tracker_properties(self) -> dict[str, Any]:
        tracker_path = self.settings.deepstream.tracker_config_path
        defaults: dict[str, Any] = {
            "tracker-width": 640,
            "tracker-height": 640,
            "gpu-id": 0,
            "ll-lib-file": "/opt/nvidia/deepstream/deepstream-7.1/lib/libnvds_nvmultiobjecttracker.so",
            "ll-config-file": "/opt/nvidia/deepstream/deepstream-7.1/samples/configs/deepstream-app/config_tracker_IOU.yml",
            "enable-batch-process": 1,
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
            properties["max-size-buffers"] = self.settings.optimization.max_queue_size
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
            "output_sink": getattr(self.settings.deepstream, "output_sink", "rtmp"),
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
        if not media_type.startswith("video/"):
            logging.debug("ignoring non-video dynamic pad: %s", media_type or "unknown")
            return
        result = pad.link(sink_pad)
        if result not in (None, 0):
            if str(result).endswith("WAS_LINKED"):
                return
            raise RuntimeError(
                f"failed to dynamically link pad to `{getattr(target, 'name', target)}`"
            )

    def _dynamic_pad_media_type(self, pad) -> str:
        caps = pad.get_current_caps() if hasattr(pad, "get_current_caps") else None
        if (caps is None or caps.get_size() == 0) and hasattr(pad, "query_caps"):
            caps = pad.query_caps(None)
        if caps is None or caps.get_size() == 0:
            return ""
        structure = caps.get_structure(0)
        return structure.get_name() if structure is not None else ""

    def _on_probe_buffer(self, pad, info, user_data=None) -> int:
        _ = pad
        payload = self._extract_probe_payload(info)
        registry = None if user_data is None else user_data.get("probe_registry")
        parser = None if user_data is None else user_data.get("meta_parser")
        if registry is not None and parser is not None:
            registry.emit_probe_payload(payload, parser)
        gst = self._runtime_factory.gst
        if gst is not None and hasattr(gst, "PadProbeReturn"):
            return gst.PadProbeReturn.OK
        return 1

    def _extract_probe_payload(self, info: object) -> object:
        batch_meta = self._extract_nvds_batch_meta(info)
        if batch_meta is not None:
            self._apply_osd_track_labels(batch_meta)
            return self._batch_meta_to_payload(batch_meta)
        if isinstance(info, dict):
            return info
        if hasattr(info, "payload"):
            return getattr(info, "payload")
        return {"raw_probe_info": info}

    def _extract_nvds_batch_meta(self, info: object) -> object | None:
        if info is None:
            return None

        buffer = None
        if hasattr(info, "get_buffer"):
            buffer = info.get_buffer()
        elif hasattr(info, "buffer"):
            buffer = getattr(info, "buffer")
        elif isinstance(info, dict):
            buffer = info.get("buffer")

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
        ntp_timestamp = self._safe_get(frame_meta, "ntp_timestamp", None)
        buf_pts = self._safe_get(frame_meta, "buf_pts", None)
        frame: dict[str, Any] = {
            "stream_id": source_id,
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
            object_payload = self._object_meta_to_payload(obj_meta)
            frame["obj_meta_list"].append(object_payload)
            if self._is_valid_track_id(object_payload.get("object_id")):
                frame["tracks"].append(
                    {
                        "track_id": object_payload["object_id"],
                        "class_id": object_payload["class_id"],
                        "confidence": object_payload["confidence"],
                        "rect_params": object_payload["rect_params"],
                    }
                )
        return frame

    def _object_meta_to_payload(self, obj_meta: object) -> dict[str, Any]:
        rect = getattr(obj_meta, "rect_params", None)
        rect_payload = {
            "left": self._safe_get(rect, "left", 0.0),
            "top": self._safe_get(rect, "top", 0.0),
            "width": self._safe_get(rect, "width", 0.0),
            "height": self._safe_get(rect, "height", 0.0),
        }
        return {
            "class_id": self._safe_get(obj_meta, "class_id", 0),
            "obj_label": self._safe_get(obj_meta, "obj_label", "unknown"),
            "confidence": self._safe_get(obj_meta, "confidence", 0.0),
            "object_id": self._safe_get(obj_meta, "object_id", 0),
            "rect_params": rect_payload,
        }

    def _apply_osd_track_labels(self, batch_meta: object) -> None:
        frame_list = getattr(batch_meta, "frame_meta_list", None)
        for frame_meta in self._iterate_glist(frame_list, "NvDsFrameMeta"):
            obj_list = getattr(frame_meta, "obj_meta_list", None)
            for obj_meta in self._iterate_glist(obj_list, "NvDsObjectMeta"):
                track_id = self._safe_get(obj_meta, "object_id", UNTRACKED_OBJECT_ID)
                if not self._is_valid_track_id(track_id):
                    continue
                label = self._safe_get(obj_meta, "obj_label", "person")
                if not label or label == "unknown":
                    label = "person"
                confidence = float(self._safe_get(obj_meta, "confidence", 0.0))
                text_params = getattr(obj_meta, "text_params", None)
                if text_params is not None:
                    text_params.display_text = f"{label} ID:{int(track_id)} {confidence:.2f}"

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
