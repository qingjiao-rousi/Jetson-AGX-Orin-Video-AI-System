from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.adapters.config_loader import load_settings
from app.application.debug_service import DebugService
from app.application.orchestrator import Orchestrator
from app.application.routing_policy import RoutingPolicy, TaskRequestBuffer
from app.application.helmet_service import HelmetTaskWorker
from app.application.plate_service import PlateTaskWorker
from app.application.scene_analytics import SceneAnalytics
from app.application.pose_service import PoseTaskWorker
from app.application.fire_smoke_service import FireSmokeTaskWorker
from app.infrastructure.inference.meta_parser import MetaParser
from app.infrastructure.inference.frame_store import FrameStore
from app.infrastructure.monitoring.gpu_monitor import GpuMonitor
from app.infrastructure.monitoring.runtime_metrics import RuntimeMetricsRecorder
from app.infrastructure.output.json_writer import JsonWriter
from app.infrastructure.output.event_writer import EventWriter
from app.infrastructure.pipeline.builder import PipelineBuilder
from app.infrastructure.pipeline.manager import PipelineManager
from app.infrastructure.web.dashboard import DashboardServer
from app.optimization.backpressure_controller import BackpressureController
from app.optimization.fps_controller import FpsController
from app.optimization.strategy_advisor import OptimizationAdvisor
from app.shared.logger import setup_logging


@dataclass(frozen=True)
class Application:
    orchestrator: Orchestrator
    debug_service: DebugService
    dashboard_server: DashboardServer | None


def create_application(config_path: Path, settings=None) -> Application:
    if settings is None:
        settings = load_settings(config_path)
    log_buffer = setup_logging(settings.logging, buffer_size=settings.web.log_buffer_size)

    pipeline_builder = PipelineBuilder(settings)
    meta_parser = MetaParser()
    capture_ids: list[str] = []
    enabled_index = 0
    for source in settings.sources:
        if not source.enabled:
            continue
        if source.capabilities:
            capture_ids.append(f"stream-{enabled_index}")
        enabled_index += 1
    capture_stream_ids = tuple(capture_ids)
    frame_store = FrameStore(
        capture_stream_ids=capture_stream_ids,
        max_size=max(settings.optimization.max_queue_size * 4, 32),
    )
    pipeline_manager = PipelineManager(
        pipeline_builder,
        meta_parser=meta_parser,
        frame_store=frame_store,
    )
    backpressure_controller = BackpressureController(settings.optimization)
    json_writer = JsonWriter(
        settings.output.jsonl_path,
        queue_size=settings.optimization.max_queue_size,
        drop_oldest=settings.optimization.enable_drop_old_frames,
        on_error=pipeline_manager.set_error,
        on_written=lambda result: _mark_result_written(
            backpressure_controller, runtime_metrics, result
        ),
    )
    event_writer = EventWriter(settings.output.events_jsonl_path)
    gpu_monitor = GpuMonitor()
    runtime_metrics = RuntimeMetricsRecorder(
        settings.output.metrics_jsonl_path,
        stale_after_seconds=settings.optimization.stale_after_seconds,
        enable_last_frame_keepalive=settings.deepstream.enable_last_frame_keepalive,
        keepalive_timeout_ms=settings.deepstream.last_frame_keepalive_timeout_ms,
    )
    runtime_metrics.set_probe_metrics_provider(pipeline_builder.probe_metrics)
    pipeline_manager.set_runtime_metrics(runtime_metrics)
    fps_controller = FpsController(settings.optimization)
    runtime_metrics.set_control_metrics_provider(
        lambda: {
            "fps": fps_controller.stats(),
            "backpressure": backpressure_controller.stats(),
        }
    )

    # 互联：FPS 控制器读取 GPU 监控和背压状态
    fps_controller.bind_gpu_monitor(gpu_monitor)
    fps_controller.bind_backpressure(backpressure_controller)

    routing_policy = RoutingPolicy(settings)
    task_buffer = TaskRequestBuffer(settings.optimization.max_queue_size)
    runtime_metrics.set_queue_metrics_provider(
        lambda: {
            "writer": json_writer.stats(),
            "task_buffer": task_buffer.stats(),
            "frame_store": frame_store.stats(),
        }
    )
    helmet_model = next(
        (model for model in settings.models if model.name == "helmet" and model.enabled),
        None,
    )
    helmet_task = next(
        (task for task in settings.model_tasks if task.name == "helmet" and task.enabled),
        None,
    )
    helmet_worker = HelmetTaskWorker(task_buffer, frame_store, helmet_model, helmet_task)
    plate_models = {
        model.name: model
        for model in settings.models
        if model.name in {"plate_detector", "plate_ocr"} and model.enabled
    }
    plate_worker = PlateTaskWorker(task_buffer, frame_store, plate_models)
    pose_model = next(
        (model for model in settings.models if model.name == "pose" and model.enabled),
        None,
    )
    pose_worker = PoseTaskWorker(task_buffer, frame_store, pose_model)
    fire_smoke_model = next(
        (model for model in settings.models if model.name == "fire_smoke" and model.enabled),
        None,
    )
    fire_smoke_worker = FireSmokeTaskWorker(task_buffer, frame_store, fire_smoke_model)
    runtime_metrics.set_queue_metrics_provider(
        lambda: {
            "writer": json_writer.stats(),
            "task_buffer": task_buffer.stats(),
            "frame_store": frame_store.stats(),
            "workers": {"helmet": helmet_worker.stats()},
        }
    )

    orchestrator = Orchestrator(
        settings=settings,
        pipeline_manager=pipeline_manager,
        meta_parser=meta_parser,
        json_writer=json_writer,
        gpu_monitor=gpu_monitor,
        runtime_metrics=runtime_metrics,
        fps_controller=fps_controller,
        backpressure_controller=backpressure_controller,
        routing_policy=routing_policy,
        task_buffer=task_buffer,
        helmet_worker=helmet_worker,
        plate_worker=plate_worker,
        event_writer=event_writer,
        scene_analytics=SceneAnalytics(getattr(settings, "analytics", {})),
        pose_worker=pose_worker,
        fire_smoke_worker=fire_smoke_worker,
    )
    helmet_worker.set_event_handler(orchestrator.on_helmet_event)
    plate_worker.set_event_handler(orchestrator.on_vehicle_event)
    pose_worker.set_event_handler(orchestrator.on_pose_event)
    fire_smoke_worker.set_event_handler(orchestrator.on_fire_smoke_event)
    debug_service = DebugService(
        orchestrator=orchestrator,
        log_buffer=log_buffer,
        optimization_advisor=OptimizationAdvisor(),
    )
    dashboard_server = DashboardServer(debug_service, settings.web) if settings.web.enabled else None
    return Application(
        orchestrator=orchestrator,
        debug_service=debug_service,
        dashboard_server=dashboard_server,
    )


def _mark_result_written(backpressure_controller, runtime_metrics, result) -> None:
    backpressure_controller.mark_consumed()
    if hasattr(runtime_metrics, "mark_result_written"):
        runtime_metrics.mark_result_written(result)
