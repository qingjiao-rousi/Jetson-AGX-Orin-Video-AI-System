from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
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
    def __init__(self, *, rtsp_dir: Path | None = None) -> None:
        self._orchestrator = DemoOrchestrator()
        self._log_buffer = InMemoryLogBuffer(capacity=20)
        self._advisor = OptimizationAdvisor()
        self._rtsp_dir = rtsp_dir
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
        rtsp_status = self._rtsp_status_snapshot()
        if rtsp_status:
            status = rtsp_status
        return {
            "app_name": self._orchestrator.settings.app_name,
            "healthy": not bool(status.get("last_error")),
            "is_running": status["is_running"],
            "pipeline_state": status["pipeline_status"]["pipeline_state"],
            "generated_at": status.get("app", {}).get("snapshot_at", "2026-06-24T12:00:00+08:00"),
        }

    def status_snapshot(self) -> dict:
        status = self._rtsp_status_snapshot() or self._orchestrator.status_snapshot()
        return {**status, "logs": self._log_buffer.stats()}

    def logs_snapshot(self, limit: int = 100) -> dict:
        return {
            "limit": limit,
            "items": self._log_buffer.tail(limit),
        }

    def debug_snapshot(self, limit: int = 100) -> dict:
        status = self.status_snapshot()
        return {
            "generated_at": status.get("app", {}).get("snapshot_at", "2026-06-24T12:00:00+08:00"),
            "health": self.health_snapshot(),
            "status": status,
            "optimization": self._advisor.recommend(status),
            "recent_logs": self.logs_snapshot(limit),
        }

    def _rtsp_status_snapshot(self) -> dict | None:
        if self._rtsp_dir is None:
            return None
        summary_path = self._rtsp_dir / "rtsp_summary.json"
        if not summary_path.is_file():
            return None
        summary = _read_json(summary_path)
        quality = _read_json(self._rtsp_dir / "rtsp_quality.json")
        metrics = _read_jsonl_last(self._rtsp_dir / "runtime_metrics.jsonl")
        source_status = _read_json(self._rtsp_dir / "source_status.json")
        metadata = _read_json(self._rtsp_dir / "run_metadata.json")
        preview_video_url = "/rtsp-files/rtsp_preview.mp4" if (self._rtsp_dir / "rtsp_preview.mp4").is_file() else ""
        individual_playback = _individual_playback(self._rtsp_dir)
        streams = _rtsp_streams(
            summary,
            quality,
            metrics,
            source_status,
            rtsp_dir=self._rtsp_dir,
            preview_video_url=preview_video_url,
            individual_playback=individual_playback,
        )
        quality_status = quality.get("quality_status") or "unknown"
        pipeline_state = "FAILED" if quality_status == "failed" else "READY"
        last_error = "; ".join((quality.get("failures") or [])[:3])
        return {
            "app": {
                "started": True,
                "stop_requested": False,
                "snapshot_at": metrics.get("timestamp") or summary.get("finished_at") or "2026-06-24T12:00:00+08:00",
            },
            "pipeline": {
                "running": quality_status != "failed",
                "source_count": summary.get("expected_stream_count", len(streams)),
                "app_name": "rtsp-production-acceptance",
                "nodes": ("rtsp-source", "depay", "decode", "streammux", "infer", "tracker", "osd", "sink"),
                "links": (),
                "probes": (("tracker", "src"),),
            },
            "pipeline_status": {
                "pipeline_state": pipeline_state,
                "running": quality_status != "failed",
                "has_runtime": True,
                "watch_attached": True,
                "last_message_type": "RTSP_ACCEPTANCE",
                "last_error": last_error or None,
                "last_warning": "; ".join((quality.get("reviews") or [])[:3]) or None,
            },
            "bus": {
                "watch_attached": True,
                "last_message_type": "RTSP_ACCEPTANCE",
                "last_error": last_error or None,
                "last_warning": "; ".join((quality.get("reviews") or [])[:3]) or None,
                "running": quality_status != "failed",
            },
            "writer": {
                "path": str(self._rtsp_dir / "results.jsonl"),
                "lines_written": summary.get("total_lines", 0),
                "is_closed": True,
            },
            "monitor": metrics.get("gpu") or summary.get("gpu") or {},
            "runtime_metrics": metrics,
            "controllers": {
                "fps": {
                    "enabled": False,
                    "observations": metrics.get("total_frames", summary.get("total_frame_count", 0)),
                    "last_fps": metrics.get("processing_fps") or summary.get("processing_fps"),
                },
                "backpressure": {
                    "enabled": True,
                    "queue_limit": 32,
                },
            },
            "streams": streams,
            "preview_video": preview_video_url,
            "last_result": None,
            "is_running": quality_status != "failed",
            "source_count": summary.get("expected_stream_count", len(streams)),
            "last_error": last_error or None,
            "rtsp_summary": summary,
            "rtsp_quality": quality,
            "source_status": source_status,
            "run_metadata": metadata,
        }


@dataclass
class DemoWebSettings:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8080
    batch_dir: Path = Path("outputs/batch")
    multifile_dir: Path = Path("outputs/multifile_inproc")
    rtsp_dir: Path = Path("outputs/rtsp_inproc")
    enable_status_api: bool = True
    enable_debug_api: bool = True
    enable_logs_api: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview the dashboard UI with demo data.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--batch-dir", type=Path, default=Path("outputs/batch"))
    parser.add_argument("--multifile-dir", type=Path, default=Path("outputs/multifile_inproc"))
    parser.add_argument("--rtsp-dir", type=Path, default=Path("outputs/rtsp_inproc"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    debug_service = DemoDebugService(rtsp_dir=args.rtsp_dir)
    settings = DemoWebSettings(
        host=args.host,
        port=args.port,
        batch_dir=args.batch_dir,
        multifile_dir=args.multifile_dir,
        rtsp_dir=args.rtsp_dir,
    )
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


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_jsonl_last(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return {}
    return {}


def _rtsp_streams(
    summary: dict,
    quality: dict,
    metrics: dict,
    source_status: dict,
    *,
    rtsp_dir: Path,
    preview_video_url: str = "",
    individual_playback: dict[str, str] | None = None,
) -> list[dict]:
    expected = int(summary.get("expected_stream_count") or source_status.get("stream_count") or 0)
    summary_streams = summary.get("streams") or {}
    metric_streams = metrics.get("streams") or {}
    source_streams = {
        f"stream-{int(item.get('index', 0)) - 1}": item
        for item in source_status.get("streams", [])
        if isinstance(item, dict) and item.get("index")
    }
    quality_by_id = {
        item.get("stream_id"): item
        for item in quality.get("streams", [])
        if isinstance(item, dict) and item.get("stream_id")
    }
    stream_ids = sorted(
        set(summary_streams) | set(metric_streams) | set(source_streams) | {f"stream-{index}" for index in range(expected)}
    )
    streams = []
    for stream_id in stream_ids:
        stream = summary_streams.get(stream_id, {})
        metric = metric_streams.get(stream_id, {})
        source = source_streams.get(stream_id, {})
        stream_quality = quality_by_id.get(stream_id, {})
        quality_status = stream_quality.get("quality_status") or quality.get("quality_status") or "unknown"
        source_uri = source.get("uri") or _rtsp_uri_from_summary(summary, stream_id)
        state = (
            "ERROR"
            if quality_status == "failed" or source.get("status") == "stopped"
            else "DEGRADED"
            if quality_status == "review" or metric.get("status") == "stale"
            else "RUNNING"
        )
        frame_count = int(metric.get("frame_count", stream.get("frame_count", 0)) or 0)
        last_frame_id = metric.get("last_frame_id", stream.get("last_frame", -1))
        try:
            estimated_dropped_frames = max(int(last_frame_id) + 1 - frame_count, 0)
        except (TypeError, ValueError):
            estimated_dropped_frames = 0
        frame_age_ms = metric.get("frame_age_ms")
        if frame_age_ms is None:
            age_seconds = metric.get("last_seen_age_seconds")
            frame_age_ms = round(float(age_seconds) * 1000, 1) if age_seconds is not None else "-"
        streams.append(
            {
                "name": stream_id,
                "title": source.get("stream_id") or stream_id,
                "source": source_uri,
                "kind": "rtsp",
                "source_type": "mediamtx-sim",
                "state": state,
                "inference_state": quality_status,
                "playback": (individual_playback or {}).get(stream_id, ""),
                "preview_playback": (individual_playback or {}).get(stream_id, "") or preview_video_url,
                "fps": metric.get("estimated_processing_fps") or stream.get("estimated_fps") or 0,
                "latency_ms": frame_age_ms,
                "detections": metric.get("detection_count", metric.get("detections", 0)),
                "dropped_frames": metric.get("dropped_frames", estimated_dropped_frames),
                "dropped_frame_rate": metric.get(
                    "dropped_frame_rate",
                    round(estimated_dropped_frames / max(int(last_frame_id) + 1, 1), 4)
                    if last_frame_id is not None
                    else 0,
                ),
                "last_warning": "; ".join(stream_quality.get("reviews", [])) or None,
                "last_error": "; ".join(stream_quality.get("failures", [])) or None,
                "last_message_type": "RTSP_ACCEPTANCE",
                "note": f"source={source.get('status', 'unknown')} / frames={stream.get('frame_count', metric.get('frame_count', 0))}",
            }
        )
    return streams


def _individual_playback(rtsp_dir: Path) -> dict[str, str]:
    index_path = rtsp_dir / "individual" / "individual_outputs.json"
    if not index_path.is_file():
        return {}
    try:
        payload = _read_json(index_path)
    except (OSError, json.JSONDecodeError):
        return {}
    outputs = payload.get("outputs", []) if isinstance(payload, dict) else []
    playback: dict[str, str] = {}
    for item in outputs:
        if not isinstance(item, dict) or not item.get("video_exists"):
            continue
        stream_id = str(item.get("stream_id", ""))
        try:
            index = int(stream_id.rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            continue
        video = Path(str(item.get("video", "")))
        try:
            relative = video.resolve().relative_to(rtsp_dir.resolve())
        except ValueError:
            continue
        playback[f"stream-{index - 1}"] = "/rtsp-files/" + relative.as_posix()
    return playback


def _rtsp_uri_from_summary(summary: dict, stream_id: str) -> str:
    base = str(summary.get("rtsp_base") or "")
    try:
        index = int(stream_id.split("-")[-1]) + 1
    except ValueError:
        return base
    return f"{base}{index}" if base else ""




if __name__ == "__main__":
    raise SystemExit(main())
