from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class WebSettings:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8080
    batch_dir: Path = Path("outputs/batch")
    multifile_dir: Path = Path("outputs/multifile_inproc")
    rtsp_dir: Path = Path("outputs/rtsp_inproc")
    enable_status_api: bool = True
    enable_debug_api: bool = True
    enable_logs_api: bool = True
    refresh_interval_ms: int = 1000
    log_buffer_size: int = 200


@dataclass(frozen=True)
class LoggingSettings:
    level: str = "INFO"
    file_path: Path = Path("outputs/logs/app.log")
    console: bool = True


@dataclass(frozen=True)
class OutputSettings:
    jsonl_path: Path = Path("outputs/results.jsonl")
    events_jsonl_path: Path = Path("outputs/events.jsonl")
    metrics_jsonl_path: Path | None = None
    enable_jsonl: bool = True
    enable_mqtt: bool = False
    enable_kafka: bool = False
    mqtt_host: str = "127.0.0.1"
    mqtt_port: int = 1883
    mqtt_topic: str = "deepstream/results"


@dataclass(frozen=True)
class OptimizationSettings:
    max_queue_size: int = 32
    fps_min: float = 5.0
    fps_max: float = 30.0
    enable_fps_control: bool = True
    enable_backpressure: bool = True
    enable_drop_old_frames: bool = True
    stale_after_seconds: float = 5.0


@dataclass(frozen=True)
class SceneSettings:
    """Static description of a camera business scene.

    Capabilities and model-task routing are intentionally added in a later
    step.  This object only establishes the stable scene vocabulary used by
    camera profiles.
    """

    name: str
    description: str = ""
    enabled: bool = True


@dataclass(frozen=True)
class ModelSettings:
    """A deployable model artifact; it does not define when it runs."""

    name: str
    engine_path: Path
    labels_path: Path | None = None
    config_path: Path | None = None
    backend: str = "tensorrt"
    input_width: int = 640
    input_height: int = 640
    confidence_threshold: float = 0.25
    nms_iou_threshold: float = 0.45
    enabled: bool = True


@dataclass(frozen=True)
class ModelTaskSettings:
    """A schedulable inference task referenced by camera capabilities."""

    name: str
    model: str
    trigger_classes: tuple[str, ...] = ()
    interval: int = 1
    min_track_frames: int = 1
    cache_frames: int = 0
    frame_trigger: bool = False
    micro_batch_size: int = 1
    micro_batch_wait_ms: int = 0
    enabled: bool = True


@dataclass(frozen=True)
class CapabilitySettings:
    """Business capability mapped to one or more model tasks."""

    name: str
    tasks: tuple[str, ...] = ()
    enabled: bool = True


@dataclass(frozen=True)
class SourceSettings:
    name: str
    uri: str
    kind: Literal["rtsp", "file"] = "rtsp"
    enabled: bool = True
    scene: str = "normal"
    priority: Literal["low", "medium", "high"] = "medium"
    zones: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeepStreamSettings:
    batch_size: int = 6
    batched_push_timeout_us: int = 40000
    inference_width: int = 640
    inference_height: int = 640
    # nvinfer interval: 0 = infer every frame, 1 = infer every 2nd frame.
    infer_interval: int = 1
    enable_tracker: bool = True
    enable_osd: bool = True
    enable_tiler: bool = False
    tiler_rows: int = 2
    tiler_columns: int = 4
    tiler_width: int = 1280
    tiler_height: int = 720
    output_sink: Literal["rtmp", "rtsp", "fake", "file"] = "rtmp"
    output_url: str = "rtmp://127.0.0.1/live/stream"
    output_video_path: Path = Path("outputs/person_detect.mp4")
    model_engine_path: Path = Path("models/yolov8s.engine")
    custom_lib_path: Path = Path("custom_libs/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so")
    tracker_config_path: Path = Path("configs/deepstream/tracker_iou.yml")
    probe_handler_path: Path = Path("build/probe_handler/libprobe_handler.so")
    infer_config_path: Path = Path("configs/deepstream/infer_primary_yolo.txt")
    streammux_config_path: Path = Path("configs/deepstream/streammux.yaml")
    enable_hardware_fallback: bool = True
    enable_last_frame_keepalive: bool = True
    last_frame_keepalive_timeout_ms: int = 1000
    encoder_bitrate: int = 4000000


@dataclass(frozen=True)
class AppSettings:
    app_name: str = "deepstream-multistream"
    source_count: int = 6
    sources: tuple[SourceSettings, ...] = ()
    scenes: tuple[SceneSettings, ...] = (
        SceneSettings(name="normal", description="基础检测场景"),
    )
    models: tuple[ModelSettings, ...] = ()
    model_tasks: tuple[ModelTaskSettings, ...] = ()
    capabilities: tuple[CapabilitySettings, ...] = ()
    analytics: dict[str, object] = field(default_factory=dict)
    enable_web: bool = False
    web: WebSettings = WebSettings()
    logging: LoggingSettings = LoggingSettings()
    output: OutputSettings = OutputSettings()
    optimization: OptimizationSettings = OptimizationSettings()
    deepstream: DeepStreamSettings = DeepStreamSettings()

    def enabled_sources(self) -> tuple[SourceSettings, ...]:
        return tuple(source for source in self.sources if source.enabled)

    def effective_source_count(self) -> int:
        enabled = self.enabled_sources()
        return len(enabled) if enabled else self.source_count

    def validate(self) -> None:
        if self.source_count <= 0:
            raise ValueError("source_count must be greater than zero")
        if self.deepstream.batch_size <= 0:
            raise ValueError("deepstream.batch_size must be greater than zero")
        if self.deepstream.inference_width <= 0 or self.deepstream.inference_height <= 0:
            raise ValueError("deepstream inference size must be greater than zero")
        if self.deepstream.infer_interval < 0:
            raise ValueError("deepstream.infer_interval must be zero or greater")
        if self.deepstream.output_sink not in {"fake", "file", "rtmp", "rtsp"}:
            raise ValueError(f"unsupported output sink: {self.deepstream.output_sink}")
        if self.deepstream.output_sink in {"rtmp", "rtsp"} and not self.deepstream.output_url:
            raise ValueError("deepstream.output_url is required for stream output")
        if self.deepstream.enable_tiler:
            if self.deepstream.tiler_rows <= 0 or self.deepstream.tiler_columns <= 0:
                raise ValueError("deepstream tiler rows/columns must be greater than zero")
            if self.deepstream.tiler_width <= 0 or self.deepstream.tiler_height <= 0:
                raise ValueError("deepstream tiler size must be greater than zero")
        if self.web.port <= 0:
            raise ValueError("web.port must be greater than zero")
        if self.web.refresh_interval_ms <= 0:
            raise ValueError("web.refresh_interval_ms must be greater than zero")
        if self.web.log_buffer_size <= 0:
            raise ValueError("web.log_buffer_size must be greater than zero")
        if self.output.enable_mqtt and not self.output.mqtt_host:
            raise ValueError("mqtt_host must be set when MQTT output is enabled")
        scene_names = {scene.name for scene in self.scenes if scene.enabled}
        if len(scene_names) != len(tuple(scene.name for scene in self.scenes if scene.enabled)):
            raise ValueError("scene names must be unique")
        if "normal" not in scene_names:
            raise ValueError("a `normal` scene must be configured")
        source_names = [source.name for source in self.sources]
        if len(source_names) != len(set(source_names)):
            raise ValueError("source names must be unique")
        for source in self.sources:
            if source.scene not in scene_names:
                raise ValueError(
                    f"source `{source.name}` references unknown scene `{source.scene}`"
                )
            if source.priority not in {"low", "medium", "high"}:
                raise ValueError(f"unsupported priority for source `{source.name}`: {source.priority}")
            if any(not zone.strip() for zone in source.zones):
                raise ValueError(f"source `{source.name}` contains an empty zone name")
        model_names = {model.name for model in self.models if model.enabled}
        if len(model_names) != len(tuple(model.name for model in self.models if model.enabled)):
            raise ValueError("model names must be unique")
        task_names = {task.name for task in self.model_tasks if task.enabled}
        if len(task_names) != len(tuple(task.name for task in self.model_tasks if task.enabled)):
            raise ValueError("model task names must be unique")
        capability_names = {capability.name for capability in self.capabilities if capability.enabled}
        if len(capability_names) != len(
            tuple(capability.name for capability in self.capabilities if capability.enabled)
        ):
            raise ValueError("capability names must be unique")
        for task in self.model_tasks:
            if task.enabled and task.model not in model_names:
                raise ValueError(f"model task `{task.name}` references unknown model `{task.model}`")
            if task.interval < 0:
                raise ValueError(f"model task `{task.name}` interval must be zero or greater")
            if task.min_track_frames <= 0:
                raise ValueError(f"model task `{task.name}` min_track_frames must be greater than zero")
            if task.cache_frames < 0:
                raise ValueError(f"model task `{task.name}` cache_frames must be zero or greater")
            if task.micro_batch_size <= 0:
                raise ValueError(f"model task `{task.name}` micro_batch_size must be greater than zero")
            if task.micro_batch_wait_ms < 0:
                raise ValueError(f"model task `{task.name}` micro_batch_wait_ms must not be negative")
        for model in self.models:
            if model.input_width <= 0 or model.input_height <= 0:
                raise ValueError(f"model `{model.name}` input size must be greater than zero")
            if not 0.0 <= model.confidence_threshold <= 1.0:
                raise ValueError(f"model `{model.name}` confidence_threshold must be between 0 and 1")
            if not 0.0 <= model.nms_iou_threshold <= 1.0:
                raise ValueError(f"model `{model.name}` nms_iou_threshold must be between 0 and 1")
        for capability in self.capabilities:
            if not capability.enabled:
                continue
            unknown_tasks = set(capability.tasks) - task_names
            if unknown_tasks:
                raise ValueError(
                    f"capability `{capability.name}` references unknown tasks: "
                    f"{sorted(unknown_tasks)}"
                )
        for source in self.sources:
            unknown_capabilities = set(source.capabilities) - capability_names
            if unknown_capabilities:
                raise ValueError(
                    f"source `{source.name}` references unknown capabilities: "
                    f"{sorted(unknown_capabilities)}"
                )
