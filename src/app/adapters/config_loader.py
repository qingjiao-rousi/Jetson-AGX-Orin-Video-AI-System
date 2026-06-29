from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    yaml = None

from app.settings import AppSettings


def load_settings(config_path: Path) -> AppSettings:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to load configuration files. Install with `pip install pyyaml`."
        )

    if not config_path.exists():
        return AppSettings()

    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    app_cfg = raw.get("app", {})
    logging_cfg = raw.get("logging", {})
    output_cfg = raw.get("output", {})
    optimization_cfg = raw.get("optimization", {})
    deepstream_cfg = raw.get("deepstream", {})
    web_cfg = raw.get("web", {})
    sources_cfg = raw.get("sources", [])

    from app.settings import (
        AppSettings,
        DeepStreamSettings,
        LoggingSettings,
        OptimizationSettings,
        OutputSettings,
        SourceSettings,
        WebSettings,
    )

    sources = tuple(
        SourceSettings(
            name=item["name"],
            uri=item["uri"],
            kind=item.get("kind", "rtsp"),
            enabled=bool(item.get("enabled", True)),
        )
        for item in sources_cfg
    )

    return AppSettings(
        app_name=app_cfg.get("app_name", "deepstream-multistream"),
        source_count=int(app_cfg.get("source_count", len(sources) or 6)),
        sources=sources,
        enable_web=bool(app_cfg.get("enable_web", False)),
        web=WebSettings(
            enabled=bool(web_cfg.get("enabled", app_cfg.get("enable_web", False))),
            host=web_cfg.get("host", "127.0.0.1"),
            port=int(web_cfg.get("port", 8080)),
            enable_status_api=bool(web_cfg.get("enable_status_api", True)),
            enable_debug_api=bool(web_cfg.get("enable_debug_api", True)),
            enable_logs_api=bool(web_cfg.get("enable_logs_api", True)),
            refresh_interval_ms=int(web_cfg.get("refresh_interval_ms", 1000)),
            log_buffer_size=int(web_cfg.get("log_buffer_size", 200)),
        ),
        logging=LoggingSettings(
            level=logging_cfg.get("level", "INFO"),
            file_path=Path(logging_cfg.get("file_path", "outputs/logs/app.log")),
            console=bool(logging_cfg.get("console", True)),
        ),
        output=OutputSettings(
            jsonl_path=Path(output_cfg.get("jsonl_path", "outputs/results.jsonl")),
            enable_jsonl=bool(output_cfg.get("enable_jsonl", True)),
            enable_mqtt=bool(output_cfg.get("enable_mqtt", False)),
            enable_kafka=bool(output_cfg.get("enable_kafka", False)),
            mqtt_host=output_cfg.get("mqtt_host", "127.0.0.1"),
            mqtt_port=int(output_cfg.get("mqtt_port", 1883)),
            mqtt_topic=output_cfg.get("mqtt_topic", "deepstream/results"),
        ),
        optimization=OptimizationSettings(
            max_queue_size=int(optimization_cfg.get("max_queue_size", 32)),
            fps_min=float(optimization_cfg.get("fps_min", 5.0)),
            fps_max=float(optimization_cfg.get("fps_max", 30.0)),
            enable_fps_control=bool(optimization_cfg.get("enable_fps_control", True)),
            enable_backpressure=bool(optimization_cfg.get("enable_backpressure", True)),
        ),
        deepstream=DeepStreamSettings(
            batch_size=int(deepstream_cfg.get("batch_size", 6)),
            batched_push_timeout_us=int(deepstream_cfg.get("batched_push_timeout_us", 40000)),
            inference_width=int(deepstream_cfg.get("inference_width", 640)),
            inference_height=int(deepstream_cfg.get("inference_height", 640)),
            model_engine_path=Path(deepstream_cfg.get("model_engine_path", "models/yolov8n.engine")),
            custom_lib_path=Path(
                deepstream_cfg.get(
                    "custom_lib_path",
                    "custom_libs/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so",
                )
            ),
            tracker_config_path=Path(
                deepstream_cfg.get("tracker_config_path", "configs/deepstream/tracker_iou.yml")
            ),
            infer_config_path=Path(
                deepstream_cfg.get("infer_config_path", "configs/deepstream/infer_primary_yolo.txt")
            ),
            streammux_config_path=Path(
                deepstream_cfg.get("streammux_config_path", "configs/deepstream/streammux.yaml")
            ),
        ),
    )
