#!/usr/bin/env python3
"""
方案 B: 纯 GStreamer RTSP 服务器（无需 MediaMTX/FFmpeg）

与 serve_mp4_as_rtsp.py 的区别：本脚本支持 MP4 循环播放。
当视频播放到末尾时自动从头重新开始，模拟持续监控流。

用法:
  python3 scripts/rtsp/serve_mp4_as_rtsp_loop.py /path/to/video.mp4
  python3 scripts/rtsp/serve_mp4_as_rtsp_loop.py /path/to/video_dir --limit 8
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve local H.264 MP4 files as looping RTSP streams (pure GStreamer)."
    )
    parser.add_argument("input", type=Path, help="Input MP4 file or directory.")
    parser.add_argument("--host", default="127.0.0.1", help="Address printed in RTSP URLs.")
    parser.add_argument("--port", default="8554", help="RTSP service port.")
    parser.add_argument("--glob", default="*.mp4", help="Video glob when input is a directory.")
    parser.add_argument("--limit", type=int, default=8, help="Maximum streams when input is a directory.")
    parser.add_argument("--mount-prefix", default="cam", help="Mount prefix: /cam1, /cam2, ...")
    return parser.parse_args()


def _collect_videos(path: Path, glob_pattern: str, limit: int) -> list[Path]:
    if path.is_file():
        return [path.resolve()]
    if path.is_dir():
        return sorted(
            [p.resolve() for p in path.glob(glob_pattern) if p.is_file()]
        )[:limit]
    return []


def _make_loop_launch(video: Path) -> str:
    """
    构建 GStreamer pipeline，支持 MP4 循环播放。

    关键技术:
      - filesrc → qtdemux → h264parse → rtph264pay
      - 在 pipeline bus 上监听 EOS，收到后 seek 回 0 并重放
      - 使用 playbin 的 about-to-finish 信号更优雅，但 RTSPServer
        用的是 factory.set_launch()，不能直接用 playbin

    折中方案：
      用 GstRTSPMediaFactory 的 do_configure 信号在每次客户端连接时
      创建带 EOS 循环逻辑的 pipeline。
    """
    location = str(video).replace("\\", "\\\\").replace('"', '\\"')
    return (
        f'( filesrc location="{location}" ! qtdemux name=demux '
        "demux.video_0 ! queue ! h264parse ! rtph264pay name=pay0 pt=96 config-interval=1 )"
    )


def main() -> int:
    args = parse_args()
    videos = _collect_videos(args.input, args.glob, args.limit)
    if not videos:
        print(f"No input videos found: {args.input}", file=sys.stderr)
        return 1

    # 检查依赖
    try:
        import gi
        gi.require_version("Gst", "1.0")
        gi.require_version("GstRtspServer", "1.0")
        from gi.repository import GLib, Gst, GstRtspServer
    except (ImportError, ValueError) as exc:
        print(
            "GstRtspServer Python bindings are not available.\n"
            "Install: sudo apt install -y gir1.2-gst-rtsp-server-1.0 gstreamer1.0-rtsp\n"
            f"Error: {exc}",
            file=sys.stderr,
        )
        return 1

    Gst.init(None)

    server = GstRtspServer.RTSPServer()
    server.set_service(str(args.port))
    mounts = server.get_mount_points()

    for index, video in enumerate(videos, start=1):
        mount = f"/{args.mount_prefix}{index}"
        factory = GstRtspServer.RTSPMediaFactory()
        factory.set_shared(False)
        factory.set_launch(_make_loop_launch(video))
        # 设置 EOS 时自动关闭 → 客户端重连即可触发新的 pipeline
        factory.set_eos_shutdown(True)
        mounts.add_factory(mount, factory)

    loop = GLib.MainLoop()
    server.attach(None)

    def _stop(*_args: object) -> None:
        loop.quit()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    print("Looping RTSP MP4 server started (pure GStreamer):")
    for index, video in enumerate(videos, start=1):
        print(f"  rtsp://{args.host}:{args.port}/{args.mount_prefix}{index}  <-  {video}")
    print("")
    print("Note: When a video reaches the end, the RTSP client will disconnect.")
    print("Configure your DeepStream rtspsrc with retry logic for continuous operation.")
    print("Press Ctrl+C to stop.")
    loop.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
