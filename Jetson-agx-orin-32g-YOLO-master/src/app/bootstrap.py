from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.adapters.config_loader import load_settings
from app.application.debug_service import DebugService
from app.application.orchestrator import Orchestrator
from app.infrastructure.inference.meta_parser import MetaParser
from app.infrastructure.monitoring.gpu_monitor import GpuMonitor
from app.infrastructure.monitoring.runtime_metrics import RuntimeMetricsRecorder
from app.infrastructure.output.json_writer import JsonWriter
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
    pipeline_manager = PipelineManager(pipeline_builder, meta_parser=meta_parser)
    json_writer = JsonWriter(
        settings.output.jsonl_path,
        queue_size=settings.optimization.max_queue_size,
        drop_oldest=settings.optimization.enable_drop_old_frames,
        on_error=pipeline_manager.set_error,
    )
    gpu_monitor = GpuMonitor()
    runtime_metrics = RuntimeMetricsRecorder(
        settings.output.metrics_jsonl_path,
        stale_after_seconds=settings.optimization.stale_after_seconds,
        enable_last_frame_keepalive=settings.deepstream.enable_last_frame_keepalive,
        keepalive_timeout_ms=settings.deepstream.last_frame_keepalive_timeout_ms,
    )
    fps_controller = FpsController(settings.optimization)
    backpressure_controller = BackpressureController(settings.optimization)

    # 互联：FPS 控制器读取 GPU 监控和背压状态
    fps_controller.bind_gpu_monitor(gpu_monitor)
    fps_controller.bind_backpressure(backpressure_controller)

    orchestrator = Orchestrator(
        settings=settings,
        pipeline_manager=pipeline_manager,
        meta_parser=meta_parser,
        json_writer=json_writer,
        gpu_monitor=gpu_monitor,
        runtime_metrics=runtime_metrics,
        fps_controller=fps_controller,
        backpressure_controller=backpressure_controller,
    )
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
