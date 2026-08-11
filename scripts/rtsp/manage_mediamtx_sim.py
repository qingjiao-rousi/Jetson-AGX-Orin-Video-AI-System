#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_RUNTIME_DIR = Path(".runtime/mediamtx_sim")


@dataclass
class StreamState:
    stream_id: str
    index: int
    input_video: str
    uri: str
    status: str = "starting"
    pid: int | None = None
    started_at: str | None = None
    last_update_at: str | None = None
    last_error: str = ""
    restart_count: int = 0
    consecutive_failures: int = 0
    log_path: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage a local MediaMTX + FFmpeg RTSP camera simulator.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Start MediaMTX and keep FFmpeg publishers alive in foreground.")
    _add_common_start_args(run_parser)

    start_parser = subparsers.add_parser("start", help="Start the simulator in background.")
    _add_common_start_args(start_parser)

    stop_parser = subparsers.add_parser("stop", help="Stop MediaMTX and all FFmpeg publishers.")
    stop_parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)

    status_parser = subparsers.add_parser("status", help="Print simulator status JSON.")
    status_parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)

    return parser.parse_args()


def _add_common_start_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="Input MP4 file or directory.")
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--host", default="127.0.0.1", help="Host printed in RTSP URLs.")
    parser.add_argument("--rtsp-port", type=int, default=8554)
    parser.add_argument("--rtmp-port", type=int, default=1935)
    parser.add_argument("--webrtc-port", type=int, default=8889)
    parser.add_argument("--hls-port", type=int, default=8888)
    parser.add_argument("--glob", default="*.mp4")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--mount-prefix", default="stream")
    parser.add_argument("--restart-delay", type=float, default=2.0)
    parser.add_argument("--max-consecutive-failures", type=int, default=5)
    parser.add_argument("--ffmpeg-loglevel", default="warning")
    parser.add_argument("--write-queue-size", type=int, default=8192)
    parser.add_argument("--rtsp-pkt-size", type=int, default=1200)
    parser.add_argument("--transcode-fps", type=float, default=15.0)
    parser.add_argument("--transcode-bitrate", default="2500k")
    parser.add_argument("--transcode-gop", type=int, default=30)
    parser.add_argument("--no-realtime", action="store_true", help="Disable ffmpeg -re.")
    parser.add_argument("--transcode", action="store_true", help="Transcode to H.264 instead of -c:v copy.")
    parser.add_argument("--enable-rtmp", action="store_true", help="Enable MediaMTX RTMP input for output validation.")
    parser.add_argument("--force", action="store_true", help="Stop previous simulator before starting.")


def main() -> int:
    args = parse_args()
    if args.command == "run":
        return run_foreground(args)
    if args.command == "start":
        return start_background(args)
    if args.command == "stop":
        stop_runtime(args.runtime_dir)
        return 0
    if args.command == "status":
        print_status(args.runtime_dir)
        return 0
    raise AssertionError(args.command)


def start_background(args: argparse.Namespace) -> int:
    runtime_dir = args.runtime_dir
    if args.force:
        stop_runtime(runtime_dir)
    elif _pid_alive(_read_pid(runtime_dir / "manager.pid")):
        raise SystemExit(f"simulator is already running: {runtime_dir / 'manager.pid'}")

    runtime_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(Path(__file__).resolve()), "run", str(args.input)]
    for option in (
        "runtime_dir",
        "host",
        "rtsp_port",
        "rtmp_port",
        "webrtc_port",
        "hls_port",
        "glob",
        "limit",
        "mount_prefix",
        "restart_delay",
        "max_consecutive_failures",
        "ffmpeg_loglevel",
        "write_queue_size",
        "rtsp_pkt_size",
        "transcode_fps",
        "transcode_bitrate",
        "transcode_gop",
    ):
        value = getattr(args, option)
        flag = "--" + option.replace("_", "-")
        cmd.extend([flag, str(value)])
    if args.no_realtime:
        cmd.append("--no-realtime")
    if args.transcode:
        cmd.append("--transcode")
    if args.enable_rtmp:
        cmd.append("--enable-rtmp")

    log_path = runtime_dir / "manager.log"
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=Path(__file__).resolve().parents[2],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (runtime_dir / "manager.pid").write_text(f"{process.pid}\n", encoding="utf-8")
    print(f"MediaMTX simulator manager started: pid={process.pid}")
    print(f"Runtime dir: {runtime_dir}")
    print(f"Status: {runtime_dir / 'source_status.json'}")
    print(f"Log: {log_path}")
    return 0


def run_foreground(args: argparse.Namespace) -> int:
    _check_dependencies()
    if args.limit <= 0:
        raise SystemExit("--limit must be greater than zero")
    runtime_dir = args.runtime_dir
    if args.force:
        stop_runtime(runtime_dir)
    elif _pid_alive(_read_pid(runtime_dir / "manager.pid")) and not _is_current_manager(runtime_dir):
        raise SystemExit(f"simulator is already running: {runtime_dir / 'manager.pid'}")

    videos = _collect_videos(args.input, args.glob, args.limit)
    if not videos:
        raise SystemExit(f"no input videos found: {args.input}")
    for video in videos:
        if not video.is_file() or video.stat().st_size <= 0:
            raise SystemExit(f"input video is missing or empty: {video}")

    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "manager.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    _write_config(args, videos)

    mediamtx = _start_mediamtx(args)
    states = _initial_states(args, videos)
    publishers: dict[str, subprocess.Popen] = {}
    stopped = False

    def _request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    try:
        time.sleep(1.5)
        if mediamtx.poll() is not None:
            raise RuntimeError(f"MediaMTX exited early with code {mediamtx.returncode}")

        for state in states:
            publishers[state.stream_id] = _start_ffmpeg(args, state)
            _mark_online(state, publishers[state.stream_id].pid)
        _write_status(args, states, mediamtx_pid=mediamtx.pid)
        _print_ready(args, states)

        while not stopped:
            if mediamtx.poll() is not None:
                for state in states:
                    state.status = "failed"
                    state.last_error = f"MediaMTX exited with code {mediamtx.returncode}"
                    state.last_update_at = _now()
                _write_status(args, states, mediamtx_pid=mediamtx.pid)
                return int(mediamtx.returncode or 1)

            for state in states:
                process = publishers.get(state.stream_id)
                if process is None:
                    continue
                exit_code = process.poll()
                if exit_code is None:
                    state.status = "online"
                    state.last_update_at = _now()
                    continue
                state.status = "reconnecting"
                state.pid = None
                state.last_error = f"ffmpeg exited with code {exit_code}"
                state.restart_count += 1
                state.consecutive_failures += 1
                state.last_update_at = _now()
                _write_status(args, states, mediamtx_pid=mediamtx.pid)
                if state.consecutive_failures > args.max_consecutive_failures:
                    state.status = "failed"
                    state.last_error = (
                        f"ffmpeg exceeded max consecutive failures: {args.max_consecutive_failures}"
                    )
                    state.last_update_at = _now()
                    continue
                time.sleep(args.restart_delay)
                publishers[state.stream_id] = _start_ffmpeg(args, state)
                _mark_online(state, publishers[state.stream_id].pid)
            _write_status(args, states, mediamtx_pid=mediamtx.pid)
            time.sleep(1.0)
    finally:
        for process in publishers.values():
            _terminate_process(process)
        _terminate_process(mediamtx)
        for state in states:
            if state.status != "failed":
                state.status = "stopped"
            state.pid = None
            state.last_update_at = _now()
        _write_status(args, states, mediamtx_pid=None)
    return 0


def stop_runtime(runtime_dir: Path) -> None:
    status_path = runtime_dir / "source_status.json"
    status = _read_json(status_path)
    manager_pid = _read_pid(runtime_dir / "manager.pid")
    if _pid_alive(manager_pid):
        try:
            os.kill(manager_pid, signal.SIGTERM)
        except OSError:
            pass
        _wait_for_pid_exit(manager_pid, timeout=5.0)

    for stream in status.get("streams", []):
        pid = stream.get("pid")
        if isinstance(pid, int) and _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
    mediamtx_pid = status.get("mediamtx", {}).get("pid")
    if isinstance(mediamtx_pid, int) and _pid_alive(mediamtx_pid):
        try:
            os.kill(mediamtx_pid, signal.SIGTERM)
        except OSError:
            pass

    for pid_file in runtime_dir.glob("*.pid"):
        try:
            pid_file.unlink()
        except OSError:
            pass
    print(f"Stopped MediaMTX simulator: {runtime_dir}")


def print_status(runtime_dir: Path) -> None:
    status_path = runtime_dir / "source_status.json"
    if not status_path.is_file():
        print(json.dumps({"status": "missing", "source_status": str(status_path)}, indent=2))
        return
    print(status_path.read_text(encoding="utf-8"), end="")


def _check_dependencies() -> None:
    missing = [name for name in ("mediamtx", "ffmpeg") if shutil.which(name) is None]
    if missing:
        raise SystemExit(
            "Missing dependencies: "
            + ", ".join(missing)
            + "\nInstall MediaMTX linux_arm64 from https://github.com/bluenviron/mediamtx/releases "
            + "and FFmpeg with `sudo apt install -y ffmpeg`."
        )


def _collect_videos(path: Path, glob_pattern: str, limit: int) -> list[Path]:
    if path.is_file():
        return [path.resolve()]
    if path.is_dir():
        return [item.resolve() for item in sorted(path.glob(glob_pattern))[:limit] if item.is_file()]
    return []


def _write_config(args: argparse.Namespace, videos: list[Path]) -> None:
    config_path = args.runtime_dir / "mediamtx.yml"
    lines = [
        "logLevel: info",
        "logDestinations: [stdout]",
        "rtsp: true",
        "rtspTransports: [tcp]",
        f"rtspAddress: :{args.rtsp_port}",
        f"writeQueueSize: {args.write_queue_size}",
        f"rtmp: {'true' if getattr(args, 'enable_rtmp', False) else 'false'}",
        f"rtmpAddress: :{args.rtmp_port}",
        "hls: false",
        "webrtc: false",
        "srt: false",
        "moq: false",
        "playback: false",
        "api: false",
        "metrics: false",
        "pprof: false",
        "paths:",
    ]
    for index, _video in enumerate(videos, start=1):
        mount = f"{args.mount_prefix}{index}"
        lines.extend(
            [
                f"  {mount}:",
                "    source: publisher",
                "    sourceOnDemand: no",
            ]
        )
    if getattr(args, "enable_rtmp", False):
        lines.extend(
            [
                "  all_others:",
                "    source: publisher",
                "    sourceOnDemand: no",
            ]
        )
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _start_mediamtx(args: argparse.Namespace) -> subprocess.Popen:
    log_path = args.runtime_dir / "mediamtx.log"
    with log_path.open("ab") as log_file:
        process = subprocess.Popen(
            ["mediamtx", str(args.runtime_dir / "mediamtx.yml")],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (args.runtime_dir / "mediamtx.pid").write_text(f"{process.pid}\n", encoding="utf-8")
    return process


def _initial_states(args: argparse.Namespace, videos: list[Path]) -> list[StreamState]:
    states = []
    for index, video in enumerate(videos, start=1):
        stream_id = f"{args.mount_prefix}{index}"
        uri = f"rtsp://{args.host}:{args.rtsp_port}/{stream_id}"
        states.append(
            StreamState(
                stream_id=stream_id,
                index=index,
                input_video=str(video),
                uri=uri,
                log_path=str(args.runtime_dir / f"ffmpeg_{stream_id}.log"),
                started_at=_now(),
                last_update_at=_now(),
            )
        )
    return states


def _start_ffmpeg(args: argparse.Namespace, state: StreamState) -> subprocess.Popen:
    log_path = Path(state.log_path)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", args.ffmpeg_loglevel]
    if not args.no_realtime:
        cmd.append("-re")
    cmd.extend(["-stream_loop", "-1", "-i", state.input_video, "-an"])
    if args.transcode:
        cmd.extend(
            [
                "-vf",
                f"fps={args.transcode_fps}",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-tune",
                "zerolatency",
                "-b:v",
                args.transcode_bitrate,
                "-maxrate",
                args.transcode_bitrate,
                "-bufsize",
                args.transcode_bitrate,
                "-g",
                str(args.transcode_gop),
                "-keyint_min",
                str(args.transcode_gop),
                "-x264-params",
                "repeat-headers=1:scenecut=0",
            ]
        )
    else:
        cmd.extend(["-c:v", "copy"])
    cmd.extend(["-f", "rtsp", "-rtsp_transport", "tcp", "-pkt_size", str(args.rtsp_pkt_size), state.uri])
    with log_path.open("ab") as log_file:
        return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)


def _mark_online(state: StreamState, pid: int | None) -> None:
    state.status = "online"
    state.pid = pid
    state.last_error = ""
    state.consecutive_failures = 0
    state.last_update_at = _now()


def _write_status(args: argparse.Namespace, states: list[StreamState], mediamtx_pid: int | None) -> None:
    payload: dict[str, Any] = {
        "status": _overall_status(states),
        "updated_at": _now(),
        "runtime_dir": str(args.runtime_dir),
        "rtsp_base": f"rtsp://{args.host}:{args.rtsp_port}/{args.mount_prefix}",
        "stream_count": len(states),
        "online_count": len([state for state in states if state.status == "online"]),
        "mediamtx": {
            "pid": mediamtx_pid,
            "rtsp_port": args.rtsp_port,
            "rtmp_port": args.rtmp_port,
            "webrtc_url": f"http://{args.host}:{args.webrtc_port}",
            "hls_url": f"http://{args.host}:{args.hls_port}",
            "log_path": str(args.runtime_dir / "mediamtx.log"),
            "write_queue_size": args.write_queue_size,
        },
        "ffmpeg": {
            "transcode": bool(args.transcode),
            "rtsp_pkt_size": args.rtsp_pkt_size,
            "transcode_fps": args.transcode_fps,
            "transcode_bitrate": args.transcode_bitrate,
            "transcode_gop": args.transcode_gop,
            "realtime": not args.no_realtime,
        },
        "streams": [asdict(state) for state in states],
    }
    (args.runtime_dir / "source_status.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _overall_status(states: list[StreamState]) -> str:
    if not states:
        return "empty"
    if all(state.status == "online" for state in states):
        return "online"
    if any(state.status == "failed" for state in states):
        return "failed"
    if any(state.status == "reconnecting" for state in states):
        return "reconnecting"
    if all(state.status == "stopped" for state in states):
        return "stopped"
    return "starting"


def _print_ready(args: argparse.Namespace, states: list[StreamState]) -> None:
    print("MediaMTX camera simulator ready:")
    for state in states:
        print(f"  {state.uri}  <-  {state.input_video}")
    print("")
    print(f"WebRTC preview: http://{args.host}:{args.webrtc_port}")
    print(f"Status: {args.runtime_dir / 'source_status.json'}")
    print("")
    print("DeepStream command:")
    print(f"  RTSP_BASE=rtsp://127.0.0.1:{args.rtsp_port}/{args.mount_prefix} \\")
    print(f"  SOURCE_COUNT={len(states)} \\")
    print("  OUTPUT_SINK=file \\")
    print("  scripts/rtsp/run_rtsp_inproc.sh outputs/rtsp_inproc")


def _terminate_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _wait_for_pid_exit(pid: int | None, timeout: float) -> None:
    if pid is None:
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.1)


def _is_current_manager(runtime_dir: Path) -> bool:
    return _read_pid(runtime_dir / "manager.pid") == os.getpid()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
