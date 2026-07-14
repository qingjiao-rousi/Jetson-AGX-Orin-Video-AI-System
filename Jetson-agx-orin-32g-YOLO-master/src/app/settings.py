from __future__ import annotations

from dataclasses import dataclass
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
class SourceSettings:
    name: str
    uri: str
    kind: Literal["rtsp", "file"] = "rtsp"
    enabled: bool = True


@dataclass(frozen=True)
class DeepStreamSettings:
    batch_size: int = 6
    batched_push_timeout_us: int = 40000
    inference_width: int = 640
    inference_height: int = 640
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
