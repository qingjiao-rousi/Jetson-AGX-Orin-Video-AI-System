#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve local H.264 MP4 files as local RTSP streams.")
    parser.add_argument("input", type=Path, help="Input MP4 file or directory.")
    parser.add_argument("--host", default="127.0.0.1", help="Address printed in RTSP URLs.")
    parser.add_argument("--port", default="8554", help="RTSP service port.")
    parser.add_argument("--glob", default="*.mp4", help="Video glob when input is a directory.")
    parser.add_argument("--limit", type=int, default=8, help="Maximum streams when input is a directory.")
    parser.add_argument("--mount-prefix", default="stream", help="Mount prefix, for example stream -> /stream1.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = _collect_videos(args.input, args.glob, args.limit)
    if not videos:
        raise SystemExit(f"no input videos found: {args.input}")

    try:
        import gi

        gi.require_version("Gst", "1.0")
        gi.require_version("GstRtspServer", "1.0")
        from gi.repository import GLib, Gst, GstRtspServer
    except (ImportError, ValueError) as exc:
        raise SystemExit(
            "GstRtspServer Python bindings are not available.\n"
            "Install the RTSP server bindings first, for example:\n"
            "  sudo apt install -y gir1.2-gst-rtsp-server-1.0 gstreamer1.0-rtsp\n"
            f"Original error: {exc}"
        ) from exc

    Gst.init(None)
    server = GstRtspServer.RTSPServer()
    server.set_service(str(args.port))
    mounts = server.get_mount_points()

    for index, video in enumerate(videos, start=1):
        mount = f"/{args.mount_prefix}{index}"
        factory = GstRtspServer.RTSPMediaFactory()
        factory.set_shared(False)
        factory.set_launch(_rtsp_launch(video))
        mounts.add_factory(mount, factory)

    loop = GLib.MainLoop()
    server.attach(None)

    def stop(*_args: object) -> None:
        loop.quit()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    print("Local RTSP MP4 simulator started:")
    for index, video in enumerate(videos, start=1):
        print(f"  rtsp://{args.host}:{args.port}/{args.mount_prefix}{index}  <-  {video}")
    print("")
    print("Press Ctrl+C to stop.")
    loop.run()
    return 0


def _collect_videos(path: Path, glob_pattern: str, limit: int) -> list[Path]:
    if path.is_file():
        return [path.resolve()]
    if path.is_dir():
        return [item.resolve() for item in sorted(path.glob(glob_pattern))[:limit] if item.is_file()]
    return []


def _rtsp_launch(video: Path) -> str:
    location = str(video).replace("\\", "\\\\").replace('"', '\\"')
    return (
        f'( filesrc location="{location}" ! qtdemux name=demux '
        "demux.video_0 ! queue ! h264parse ! rtph264pay name=pay0 pt=96 config-interval=1 )"
    )


if __name__ == "__main__":
    raise SystemExit(main())
