from __future__ import annotations

"""应用 composition root：在这里连接组件，但不承载具体业务决策。"""

from dataclasses import dataclass
from pathlib import Path

# 配置适配器：调用方未注入 settings 时，在 composition root 创建本次运行的配置快照。
from app.adapters.config_loader import load_settings
# 应用层编排器负责生命周期；DebugService 只整理只读状态给 Web/UI。
from app.application.debug_service import DebugService
from app.application.orchestrator import Orchestrator
# 主检测结果先被路由为任务请求，再经有界队列交给慢速专用模型 worker。
from app.application.routing_policy import RoutingPolicy, TaskRequestBuffer
# 各 worker 不属于 DeepStream 主图，只消费 FrameStore 中与请求帧号匹配的 CPU 帧副本。
from app.application.helmet_service import HelmetTaskWorker
from app.application.plate_service import PlateTaskWorker
# 轻量场景分析只使用 tracker 结果，例如区域、越线和人车关系，不调用额外模型。
from app.application.scene_analytics import SceneAnalytics
from app.application.pose_service import PoseTaskWorker
from app.application.fire_smoke_service import FireSmokeTaskWorker
# 将松散 metadata 归一化为领域对象；FrameStore 隔离 probe 与异步 ROI worker。
from app.infrastructure.inference.meta_parser import MetaParser
from app.infrastructure.inference.frame_store import FrameStore
from app.infrastructure.monitoring.gpu_monitor import GpuMonitor
from app.infrastructure.monitoring.runtime_metrics import RuntimeMetricsRecorder
from app.infrastructure.output.json_writer import JsonWriter
from app.infrastructure.output.event_writer import EventWriter
# Builder 构造 DeepStream 图；Manager 管理运行态、bus 和 probe 回调。
from app.infrastructure.pipeline.builder import PipelineBuilder
from app.infrastructure.pipeline.manager import PipelineManager
from app.infrastructure.web.dashboard import DashboardServer
# 背压给 FPS gate 提供反馈，Advisor 只产出人工可读建议。
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
    """构建一套彼此隔离的运行期组件，并返回由入口负责启停的应用容器。"""
    if settings is None:
        # 单测或其他入口可注入已构造的 settings，避免再次读取 YAML。
        settings = load_settings(config_path)
    # 日志最先初始化，之后的 engine、路径或配置错误才能同时进入文件和调试缓冲。
    log_buffer = setup_logging(settings.logging, buffer_size=settings.web.log_buffer_size)

    # 主 DeepStream 链路只负责解码、合批、主检测、跟踪与 metadata/probe 采集。
    pipeline_builder = PipelineBuilder(settings)
    meta_parser = MetaParser()
    capture_ids: list[str] = []
    enabled_index = 0
    # Builder 使用“启用 source 的顺序”分配 stream-N；这里必须遵守同一编号规则。
    for source in settings.sources:
        if not source.enabled:
            continue
        if source.capabilities:
            capture_ids.append(f"stream-{enabled_index}")
        enabled_index += 1
    capture_stream_ids = tuple(capture_ids)
    # 只有绑定专用能力的流需要保存原始帧，避免基础检测流无意义地占用 CPU 内存。
    # 默认共享容量随全局任务队列放大；容量实验可用 YAML 显式覆盖。
    frame_store = FrameStore(
        capture_stream_ids=capture_stream_ids,
        max_size=(
            settings.optimization.frame_store_max_size
            or max(settings.optimization.max_queue_size * 4, 32)
        ),
        max_per_stream=settings.optimization.frame_store_per_stream_capacity,
    )
    # Manager 是运行时门面，将 parser/FrameStore 注入真正的 GStreamer probe 回调。
    pipeline_manager = PipelineManager(
        pipeline_builder,
        meta_parser=meta_parser,
        frame_store=frame_store,
    )
    # 输出写入完成才算结果真正被消费，可作为主链路背压控制的闭环信号。
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
    # 事件频率低于逐帧结果，使用独立 writer 避免与主结果队列互相影响。
    event_writer = EventWriter(settings.output.events_jsonl_path)
    gpu_monitor = GpuMonitor()
    # 指标采集与业务输出解耦，后续所有 provider 在采样时按需读取当前状态。
    runtime_metrics = RuntimeMetricsRecorder(
        settings.output.metrics_jsonl_path,
        stale_after_seconds=settings.optimization.stale_after_seconds,
        enable_last_frame_keepalive=settings.deepstream.enable_last_frame_keepalive,
        keepalive_timeout_ms=settings.deepstream.last_frame_keepalive_timeout_ms,
    )
    # Recorder 拉取组件快照而不控制组件，避免观测代码反向改变推理逻辑。
    runtime_metrics.set_probe_metrics_provider(pipeline_builder.probe_metrics)
    pipeline_manager.set_runtime_metrics(runtime_metrics)
    fps_controller = FpsController(settings.optimization)
    runtime_metrics.set_control_metrics_provider(
        lambda: {
            "fps": fps_controller.stats(),
            "backpressure": backpressure_controller.stats(),
        }
    )

    # FPS 控制器根据硬件监控与输出背压调整采样节奏，不直接参与推理。
    fps_controller.bind_gpu_monitor(gpu_monitor)
    fps_controller.bind_backpressure(backpressure_controller)

    # 路由决定哪个 ROI/帧需要专用任务；TaskRequestBuffer 为每个任务维护独立队列。
    routing_policy = RoutingPolicy(settings)
    task_buffer = TaskRequestBuffer(
        settings.optimization.max_queue_size,
        task_settings=settings.model_tasks,
    )
    # 先注册基础队列指标；worker 创建完成后会替换为包含 worker 端到端状态的 provider。
    runtime_metrics.set_queue_metrics_provider(
        lambda: {
            "writer": json_writer.stats(),
            "task_buffer": task_buffer.stats(),
            "frame_store": frame_store.stats(),
        }
    )
    # 专用模型在各自 worker 中独立消费，避免一个慢任务阻塞另一个任务的排队与陈旧丢弃策略。
    # 专用模型允许缺省；worker 延迟初始化，并由自身决定空配置时的行为。
    helmet_model = next(
        (model for model in settings.models if model.name == "helmet" and model.enabled),
        None,
    )
    helmet_task = next(
        (task for task in settings.model_tasks if task.name == "helmet" and task.enabled),
        None,
    )
    helmet_worker = HelmetTaskWorker(task_buffer, frame_store, helmet_model, helmet_task)
    # 车牌任务由 detector 与 OCR 组成，故按名字传入两套模型而非单一 engine。
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
    # 用最终 provider 覆盖基础 provider，将每个 worker 的处理/缺帧/时延一起纳入快照。
    runtime_metrics.set_queue_metrics_provider(
        lambda: {
            "writer": json_writer.stats(),
            "task_buffer": task_buffer.stats(),
            "frame_store": frame_store.stats(),
            "workers": {
                "helmet": helmet_worker.stats(),
                "pose": pose_worker.stats(),
                "fire_smoke": fire_smoke_worker.stats(),
                "plate_detector": plate_worker.stats(),
            },
        }
    )

    # Orchestrator 是唯一的生命周期协调者：主 pipeline 回调、路由、异步任务和输出在此汇合。
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
    # Worker 只产生领域事件；事件写出和主结果关联由 Orchestrator 统一处理。
    helmet_worker.set_event_handler(orchestrator.on_helmet_event)
    plate_worker.set_event_handler(orchestrator.on_vehicle_event)
    pose_worker.set_event_handler(orchestrator.on_pose_event)
    fire_smoke_worker.set_event_handler(orchestrator.on_fire_smoke_event)
    # DebugService 只组合状态和建议，不拥有、不启动核心推理资源。
    debug_service = DebugService(
        orchestrator=orchestrator,
        log_buffer=log_buffer,
        optimization_advisor=OptimizationAdvisor(),
    )
    # Web 是可选观察面，关闭它不能影响主 pipeline 与 benchmark 的可运行性。
    dashboard_server = DashboardServer(debug_service, settings.web) if settings.web.enabled else None
    return Application(
        orchestrator=orchestrator,
        debug_service=debug_service,
        dashboard_server=dashboard_server,
    )


def _mark_result_written(backpressure_controller, runtime_metrics, result) -> None:
    """将异步 JSONL 落盘回传给背压和端到端时延指标。"""
    backpressure_controller.mark_consumed()
    if hasattr(runtime_metrics, "mark_result_written"):
        runtime_metrics.mark_result_written(result)
