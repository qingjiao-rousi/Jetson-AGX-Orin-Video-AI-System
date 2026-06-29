from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.infrastructure.web.dashboard import DashboardServer
from app.optimization.strategy_advisor import OptimizationAdvisor
from app.shared.logger import InMemoryLogBuffer


@dataclass
class DemoOrchestrator:
    class Settings:
        app_name = "deepstream-multistream-demo"

    settings = Settings()

    def status_snapshot(self) -> dict:
        return {
            "app": {
                "started": True,
                "stop_requested": False,
                "snapshot_at": "2026-06-24T12:00:00+08:00",
            },
            "pipeline": {
                "running": True,
                "source_count": 4,
                "app_name": "deepstream-multistream-demo",
                "nodes": ("source", "decode", "streammux", "infer", "tracker", "osd", "encoder", "sink"),
                "links": (
                    ("source", "decode"),
                    ("decode", "streammux"),
                    ("streammux", "infer"),
                    ("infer", "tracker"),
                    ("tracker", "osd"),
                    ("osd", "encoder"),
                    ("encoder", "sink"),
                ),
                "probes": (("tracker", "src"),),
            },
            "pipeline_status": {
                "pipeline_state": "PLAYING",
                "running": True,
                "has_runtime": True,
                "watch_attached": True,
                "last_message_type": "STATE_CHANGED",
                "last_error": None,
                "last_warning": "演示告警：模拟 bus warning",
            },
            "bus": {
                "watch_attached": True,
                "last_message_type": "WARNING",
                "last_error": None,
                "last_warning": "演示告警：模拟 bus warning",
                "running": True,
            },
            "writer": {
                "path": "outputs/results.jsonl",
                "lines_written": 842,
                "is_closed": False,
            },
            "monitor": {
                "status": "started",
                "running": True,
                "utilization_gpu": 68,
                "utilization_memory": 41,
                "temperature_c": 62,
            },
            "controllers": {
                "fps": {
                    "enabled": True,
                    "observations": 842,
                    "fps_min": 5.0,
                    "fps_max": 30.0,
                    "last_stream_id": "stream-1",
                    "last_frame_id": 3820,
                    "last_result_type": "FrameResult",
                    "last_fps": 24.8,
                },
                "backpressure": {
                    "enabled": True,
                    "observations": 842,
                    "queue_limit": 32,
                    "last_stream_id": "stream-1",
                    "last_frame_id": 3820,
                    "last_result_type": "FrameResult",
                },
            },
            "last_result": {
                "stream_id": "stream-1",
                "frame_id": 3820,
                "detection_count": 11,
                "track_count": 2,
                "timestamp": "2026-06-24T12:00:00+08:00",
            },
            "streams": (
                {
                    "name": "stream-1",
                    "title": "Entrance Channel",
                    "source": "rtsp://192.168.1.101/live/main",
                    "kind": "rtsp",
                    "source_type": "network",
                    "state": "RUNNING",
                    "inference_state": "active",
                    "playback": "rtmp://demo/live/stream-1",
                    "fps": 25.1,
                    "latency_ms": 46,
                    "detections": 4,
                    "dropped_frames": 0,
                    "last_warning": None,
                    "last_error": None,
                    "last_message_type": "STATE_CHANGED",
                    "note": "OSD result stream",
                },
                {
                    "name": "stream-2",
                    "title": "Assembly Line A",
                    "source": "rtsp://192.168.1.102/live/main",
                    "kind": "rtsp",
                    "source_type": "network",
                    "state": "RUNNING",
                    "inference_state": "active",
                    "playback": "rtmp://demo/live/stream-2",
                    "fps": 24.7,
                    "latency_ms": 53,
                    "detections": 3,
                    "dropped_frames": 1,
                    "last_warning": None,
                    "last_error": None,
                    "last_message_type": "STATE_CHANGED",
                    "note": "OSD result stream",
                },
                {
                    "name": "stream-3",
                    "title": "Warehouse Route",
                    "source": "/data/videos/warehouse-demo.mp4",
                    "kind": "file",
                    "source_type": "local",
                    "state": "DEGRADED",
                    "inference_state": "active",
                    "playback": "rtmp://demo/live/stream-3",
                    "fps": 18.9,
                    "latency_ms": 88,
                    "detections": 2,
                    "dropped_frames": 7,
                    "last_warning": "帧到达偏慢",
                    "last_error": None,
                    "last_message_type": "WARNING",
                    "note": "存在延迟告警",
                },
                {
                    "name": "stream-4",
                    "title": "Validation Sample",
                    "source": "/data/videos/sample-4.mp4",
                    "kind": "file",
                    "source_type": "local",
                    "state": "RUNNING",
                    "inference_state": "active",
                    "playback": "rtmp://demo/live/stream-4",
                    "fps": 25.5,
                    "latency_ms": 39,
                    "detections": 2,
                    "dropped_frames": 0,
                    "last_warning": None,
                    "last_error": None,
                    "last_message_type": "STATE_CHANGED",
                    "note": "local validation stream",
                },
            ),
            "is_running": True,
            "source_count": 4,
            "last_error": None,
        }


class DemoDebugService:
    def __init__(self) -> None:
        self._orchestrator = DemoOrchestrator()
        self._log_buffer = InMemoryLogBuffer(capacity=20)
        self._advisor = OptimizationAdvisor()
        for msg in (
            "演示服务已启动",
            "已加载 4 路演示视频任务",
            "stream-3 上报帧到达偏慢",
            "状态数据刷新正常",
        ):
            record = logging.LogRecord(
                name="demo",
                level=logging.INFO,
                pathname=__file__,
                lineno=1,
                msg=msg,
                args=(),
                exc_info=None,
            )
            self._log_buffer.append(record)

    def health_snapshot(self) -> dict:
        status = self._orchestrator.status_snapshot()
        return {
            "app_name": self._orchestrator.settings.app_name,
            "healthy": True,
            "is_running": status["is_running"],
            "pipeline_state": status["pipeline_status"]["pipeline_state"],
            "generated_at": "2026-06-24T12:00:00+08:00",
        }

    def status_snapshot(self) -> dict:
        status = self._orchestrator.status_snapshot()
        return {**status, "logs": self._log_buffer.stats()}

    def logs_snapshot(self, limit: int = 100) -> dict:
        return {
            "limit": limit,
            "items": self._log_buffer.tail(limit),
        }

    def debug_snapshot(self, limit: int = 100) -> dict:
        status = self.status_snapshot()
        return {
            "generated_at": "2026-06-24T12:00:00+08:00",
            "health": self.health_snapshot(),
            "status": status,
            "optimization": self._advisor.recommend(status),
            "recent_logs": self.logs_snapshot(limit),
        }


@dataclass
class DemoWebSettings:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8080
    enable_status_api: bool = True
    enable_debug_api: bool = True
    enable_logs_api: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview the dashboard UI with demo data.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    debug_service = DemoDebugService()
    settings = DemoWebSettings(host=args.host, port=args.port)
    server = DashboardServer(debug_service, settings)
    server.start()
    print(f"Dashboard preview running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
