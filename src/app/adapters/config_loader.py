from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    yaml = None

from app.settings import (
    AppSettings,
    CapabilitySettings,
    DeepStreamSettings,
    LoggingSettings,
    ModelSettings,
    ModelTaskSettings,
    OptimizationSettings,
    OutputSettings,
    SceneSettings,
    SourceSettings,
    WebSettings,
)


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
    scenes_cfg = raw.get("scenes", {})
    models_cfg = raw.get("models", {})
    model_tasks_cfg = raw.get("model_tasks", {})
    capabilities_cfg = raw.get("capabilities", {})
    analytics_cfg = raw.get("analytics", {})

    scenes = _parse_scenes(scenes_cfg)
    models = _parse_models(models_cfg)
    model_tasks = _parse_model_tasks(model_tasks_cfg)
    capabilities = _parse_capabilities(capabilities_cfg)

    sources = tuple(
        SourceSettings(
            name=item["name"],
            uri=item["uri"],
            kind=item.get("kind", "rtsp"),
            enabled=bool(item.get("enabled", True)),
            scene=str(item.get("scene", "normal")),
            priority=str(item.get("priority", "medium")),
            zones=tuple(str(zone) for zone in item.get("zones", ())),
            capabilities=tuple(str(value) for value in item.get("capabilities", ())),
        )
        for item in sources_cfg
    )

    return AppSettings(
        app_name=app_cfg.get("app_name", "deepstream-multistream"),
        source_count=int(app_cfg.get("source_count", len(sources) or 6)),
        sources=sources,
        scenes=scenes,
        models=models,
        model_tasks=model_tasks,
        capabilities=capabilities,
        analytics=analytics_cfg if isinstance(analytics_cfg, dict) else {},
        enable_web=bool(app_cfg.get("enable_web", False)),
        web=WebSettings(
            enabled=bool(web_cfg.get("enabled", app_cfg.get("enable_web", False))),
            host=web_cfg.get("host", "127.0.0.1"),
            port=int(web_cfg.get("port", 8080)),
            batch_dir=Path(web_cfg.get("batch_dir", "outputs/batch")),
            multifile_dir=Path(web_cfg.get("multifile_dir", "outputs/multifile_inproc")),
            rtsp_dir=Path(web_cfg.get("rtsp_dir", "outputs/rtsp_inproc")),
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
            events_jsonl_path=Path(output_cfg.get("events_jsonl_path", "outputs/events.jsonl")),
            metrics_jsonl_path=Path(output_cfg["metrics_jsonl_path"])
            if output_cfg.get("metrics_jsonl_path")
            else None,
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
            enable_drop_old_frames=bool(optimization_cfg.get("enable_drop_old_frames", True)),
            stale_after_seconds=float(optimization_cfg.get("stale_after_seconds", 5.0)),
        ),
        deepstream=DeepStreamSettings(
            batch_size=int(deepstream_cfg.get("batch_size", 6)),
            batched_push_timeout_us=int(deepstream_cfg.get("batched_push_timeout_us", 40000)),
            inference_width=int(deepstream_cfg.get("inference_width", 640)),
            inference_height=int(deepstream_cfg.get("inference_height", 640)),
            infer_interval=int(deepstream_cfg.get("infer_interval", 1)),
            enable_tracker=bool(deepstream_cfg.get("enable_tracker", True)),
            enable_osd=bool(deepstream_cfg.get("enable_osd", True)),
            enable_tiler=bool(deepstream_cfg.get("enable_tiler", False)),
            tiler_rows=int(deepstream_cfg.get("tiler_rows", 2)),
            tiler_columns=int(deepstream_cfg.get("tiler_columns", 4)),
            tiler_width=int(deepstream_cfg.get("tiler_width", 1280)),
            tiler_height=int(deepstream_cfg.get("tiler_height", 720)),
            output_sink=deepstream_cfg.get("output_sink", "rtmp"),
            output_url=deepstream_cfg.get(
                "output_url",
                "rtmp://127.0.0.1/live/stream",
            ),
            output_video_path=Path(deepstream_cfg.get("output_video_path", "outputs/person_detect.mp4")),
            model_engine_path=Path(deepstream_cfg.get("model_engine_path", "models/yolov8s.engine")),
            custom_lib_path=Path(
                deepstream_cfg.get(
                    "custom_lib_path",
                    "custom_libs/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so",
                )
            ),
            tracker_config_path=Path(
                deepstream_cfg.get("tracker_config_path", "configs/deepstream/tracker_iou.yml")
            ),
            probe_handler_path=Path(
                deepstream_cfg.get("probe_handler_path", "build/probe_handler/libprobe_handler.so")
            ),
            infer_config_path=Path(
                deepstream_cfg.get("infer_config_path", "configs/deepstream/infer_primary_yolo.txt")
            ),
            streammux_config_path=Path(
                deepstream_cfg.get("streammux_config_path", "configs/deepstream/streammux.yaml")
            ),
            enable_hardware_fallback=bool(deepstream_cfg.get("enable_hardware_fallback", True)),
            enable_last_frame_keepalive=bool(deepstream_cfg.get("enable_last_frame_keepalive", True)),
            last_frame_keepalive_timeout_ms=int(deepstream_cfg.get("last_frame_keepalive_timeout_ms", 1000)),
            encoder_bitrate=int(deepstream_cfg.get("encoder_bitrate", 4000000)),
        ),
    )


def _parse_scenes(raw: Any) -> tuple[SceneSettings, ...]:
    """Parse scene definitions while keeping old configs backwards compatible."""
    if not raw:
        return (SceneSettings(name="normal", description="基础检测场景"),)

    parsed: list[SceneSettings] = []
    if isinstance(raw, dict):
        items = ((name, value) for name, value in raw.items())
    elif isinstance(raw, list):
        items = ((item.get("name"), item) for item in raw if isinstance(item, dict))
    else:
        raise ValueError("scenes must be a mapping or a list")

    for name, value in items:
        if not name or not str(name).strip():
            raise ValueError("scene name must not be empty")
        options = value if isinstance(value, dict) else {}
        parsed.append(
            SceneSettings(
                name=str(name),
                description=str(options.get("description", "")),
                enabled=bool(options.get("enabled", True)),
            )
        )

    if not any(scene.name == "normal" and scene.enabled for scene in parsed):
        parsed.append(SceneSettings(name="normal", description="基础检测场景"))
    return tuple(parsed)


def _parse_models(raw: Any) -> tuple[ModelSettings, ...]:
    if not raw:
        return ()
    items = raw.items() if isinstance(raw, dict) else (
        (item.get("name"), item) for item in raw if isinstance(item, dict)
    )
    parsed: list[ModelSettings] = []
    for name, value in items:
        if not name:
            raise ValueError("model name must not be empty")
        options = value if isinstance(value, dict) else {}
        engine = options.get("engine", options.get("path"))
        if not engine:
            raise ValueError(f"model `{name}` requires `engine` or `path`")
        parsed.append(
            ModelSettings(
                name=str(name),
                engine_path=Path(str(engine)),
                labels_path=Path(str(options["labels"])) if options.get("labels") else None,
                config_path=Path(str(options["config"])) if options.get("config") else None,
                backend=str(options.get("backend", "tensorrt")),
                input_width=int(options.get("input_width", 640)),
                input_height=int(options.get("input_height", 640)),
                confidence_threshold=float(options.get("confidence_threshold", 0.25)),
                nms_iou_threshold=float(options.get("nms_iou_threshold", 0.45)),
                enabled=bool(options.get("enabled", True)),
            )
        )
    return tuple(parsed)


def _parse_model_tasks(raw: Any) -> tuple[ModelTaskSettings, ...]:
    if not raw:
        return ()
    items = raw.items() if isinstance(raw, dict) else (
        (item.get("name"), item) for item in raw if isinstance(item, dict)
    )
    parsed: list[ModelTaskSettings] = []
    for name, value in items:
        if not name:
            raise ValueError("model task name must not be empty")
        options = value if isinstance(value, dict) else {}
        trigger_classes = options.get("trigger_classes", options.get("trigger_class", ()))
        if isinstance(trigger_classes, str):
            trigger_classes = (trigger_classes,)
        parsed.append(
            ModelTaskSettings(
                name=str(name),
                model=str(options.get("model", "")),
                trigger_classes=tuple(str(item) for item in trigger_classes),
                interval=int(options.get("interval", 1)),
                min_track_frames=int(options.get("min_track_frames", 1)),
                cache_frames=int(options.get("cache_frames", 0)),
                frame_trigger=bool(options.get("frame_trigger", False)),
                enabled=bool(options.get("enabled", True)),
            )
        )
    return tuple(parsed)


def _parse_capabilities(raw: Any) -> tuple[CapabilitySettings, ...]:
    if not raw:
        return ()
    items = raw.items() if isinstance(raw, dict) else (
        (item.get("name"), item) for item in raw if isinstance(item, dict)
    )
    parsed: list[CapabilitySettings] = []
    for name, value in items:
        if not name:
            raise ValueError("capability name must not be empty")
        options = value if isinstance(value, dict) else {}
        tasks = options.get("tasks", ())
        if isinstance(tasks, str):
            tasks = (tasks,)
        parsed.append(
            CapabilitySettings(
                name=str(name),
                tasks=tuple(str(task) for task in tasks),
                enabled=bool(options.get("enabled", True)),
            )
        )
    return tuple(parsed)
