from __future__ import annotations

"""GStreamer runtime 的生命周期与 bus 消息管理。"""

import logging
from threading import Event, Thread, current_thread

# PipelineState 是暴露给编排器/UI 的最小运行态，不泄漏 Gst 对象。
from app.domain.entities import PipelineState
# Blueprint 用于描述拓扑；Registry 把 probe 解析结果交回应用层。
from app.infrastructure.pipeline.builder import PipelineBlueprint
from app.infrastructure.pipeline.probes import ProbeRegistry


class PipelineManager:
    """将声明式 ``PipelineBlueprint`` 变为运行时 pipeline，并处理其状态机。

    Builder 负责组装元素和 probe，Manager 负责 PLAYING/NULL 转换、EOS、错误和
    可选输出回退。bus 轮询使用独立守护线程，避免依赖应用层是否运行 GLib 主循环。
    """
    def __init__(self, builder, probes: ProbeRegistry | None = None, meta_parser=None, frame_store=None) -> None:
        # builder 创建 GStreamer 图；parser/FrameStore 是 probe 回调所需的应用侧依赖。
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
        """设置主推理前的 buffer gate，通常由 FPS 控制器按背压丢弃帧。"""
        self._frame_gate = gate

    def set_runtime_metrics(self, runtime_metrics) -> None:
        """注入 probe 时延标记器；Manager 不依赖其具体实现，便于关闭 metrics 的运行。"""
        self._runtime_metrics = runtime_metrics

    def register_plate_annotation(self, stream_id: str, track_id: int, event: dict) -> None:
        """转发异步 OCR 结果给 Builder 的 OSD 状态表，下一帧同轨迹可显示车牌。"""
        if hasattr(self._builder, "register_plate_annotation"):
            self._builder.register_plate_annotation(stream_id, track_id, event)

    def start(self) -> None:
        """装配运行时对象、注册 probe 后切到 PLAYING；编码路径失败时可回退 fake sink。"""
        if self._running:
            return
        # build_runtime 只负责 GStreamer 对象图；这些应用侧依赖必须在 probe 注册前注入。
        self._runtime = self._builder.build_runtime()
        # runtime 字典是 Builder 与 Manager 的窄接口：GStreamer 对象归 Builder，
        # 回调所需的 Python 服务由 Manager 在真正挂 probe 前注入。
        self._runtime["probe_registry"] = self._probes
        self._runtime["meta_parser"] = self._meta_parser
        self._runtime["frame_store"] = self._frame_store
        self._runtime["frame_gate"] = self._frame_gate
        self._runtime["runtime_metrics"] = self._runtime_metrics
        # 必须先挂 probe，再 PLAYING；否则首个 batch 可能没有 metadata/时延标记。
        self._register_probe_points()
        self._pipeline = self._runtime["blueprint"]
        self._stop_event.clear()
        # signal watch 兼容 GLib；poller 保证纯 CLI 模式也能收到 EOS/ERROR。
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
        """请求 bus 线程退出，并在 file sink 下先发 EOS，确保 MP4 容器尾部被写完整。"""
        self._stop_event.set()
        bus_thread = self._bus_thread
        if self._runtime is not None:
            # pipeline 先进入 NULL，随后才 join bus 线程，避免线程继续消费旧 bus 消息。
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
        """文件输出不能直接 NULL：先等待 EOS 或 ERROR，给 qtmux 写入索引的机会。"""
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
        """完全重建运行时 pipeline；用于明确的恢复策略，而非每个 bus warning。"""
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
        """导出可 JSON 化的蓝图摘要，供 dashboard 和调试而非实际 runtime 控制。"""
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
        """注册 signal watch 以兼容 GLib 环境；实际可靠处理仍由轮询线程完成。"""
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
        """启动有限超时的 bus 轮询，EOS/ERROR 能在无 GLib MainLoop 的 CLI 运行中被处理。"""
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
        """在独立线程消费关键 bus 消息；非实时文件 EOS 会自然结束应用主循环。"""
        while not self._stop_event.is_set():
            message = bus.timed_pop_filtered(200_000_000, message_types)
            if message is None:
                continue
            self._on_bus_message(bus, message)
            if not self._running or self._last_error is not None:
                self._stop_event.set()
                return

    def _set_pipeline_state_playing(self) -> None:
        """执行 GStreamer 状态切换，并把同步失败转换为可诊断异常。"""
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
        """仅在输出硬件路径失败时重建为 fakesink，保留解码、推理与指标链路用于诊断。"""
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
        """在 registry/parser/FrameStore 注入后注册一次 probe，避免重复挂载回调。"""
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
        """将 GStreamer bus 消息归一化为运行状态。

        RTSP 网络类错误和 EOS 只标记 warning，交给 source 的重连能力继续恢复；
        文件输入的 EOS 则表示全部输入已完成，应结束运行。
        """
        _ = bus
        message_type = getattr(message, "type", None)
        self._last_message_type = str(message_type) if message_type is not None else None
        message_name = self._message_type_name(message_type)

        if message_name == "ERROR":
            # RTSP 断流类错误可恢复；文件解码/模型等其它错误应让主循环退出。
            error, _debug = message.parse_error()
            if self._is_live_source_recoverable_error(str(error)):
                self._last_warning = f"recoverable live source error: {error}"
                return
            self._last_error = str(error)
            self._running = False
            return

        if message_name == "EOS":
            # 本地 MP4 的 EOS 是正常结束；live source 的 EOS 则等待 source 重连。
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
