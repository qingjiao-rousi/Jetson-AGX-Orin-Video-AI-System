from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "rtsp"))

import manage_mediamtx_sim as sim


class MediaMtxSimulatorTests(unittest.TestCase):
    def test_collect_videos_sorts_and_limits_mp4_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b.mp4").write_bytes(b"b")
            (root / "a.mp4").write_bytes(b"a")
            (root / "note.txt").write_text("ignore", encoding="utf-8")

            videos = sim._collect_videos(root, "*.mp4", 1)

        self.assertEqual([video.name for video in videos], ["a.mp4"])

    def test_write_config_creates_publisher_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime_dir = Path(tmp)
            args = _args(runtime_dir=runtime_dir, mount_prefix="cam", rtsp_port=9554)
            videos = [runtime_dir / "1.mp4", runtime_dir / "2.mp4"]

            sim._write_config(args, videos)

            text = (runtime_dir / "mediamtx.yml").read_text(encoding="utf-8")

        self.assertIn("rtspAddress: :9554", text)
        self.assertIn("  cam1:", text)
        self.assertIn("  cam2:", text)
        self.assertIn("source: publisher", text)

    def test_write_status_contains_rtsp_urls_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime_dir = Path(tmp)
            args = _args(runtime_dir=runtime_dir, mount_prefix="stream")
            states = [
                sim.StreamState(
                    stream_id="stream1",
                    index=1,
                    input_video="/videos/1.mp4",
                    uri="rtsp://127.0.0.1:8554/stream1",
                    status="online",
                ),
                sim.StreamState(
                    stream_id="stream2",
                    index=2,
                    input_video="/videos/2.mp4",
                    uri="rtsp://127.0.0.1:8554/stream2",
                    status="reconnecting",
                ),
            ]

            sim._write_status(args, states, mediamtx_pid=123)
            payload = json.loads((runtime_dir / "source_status.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["status"], "reconnecting")
        self.assertEqual(payload["stream_count"], 2)
        self.assertEqual(payload["online_count"], 1)
        self.assertEqual(payload["mediamtx"]["pid"], 123)
        self.assertEqual(payload["streams"][0]["uri"], "rtsp://127.0.0.1:8554/stream1")

    def test_start_ffmpeg_uses_realtime_loop_copy_and_no_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime_dir = Path(tmp)
            args = _args(runtime_dir=runtime_dir, transcode=False, no_realtime=False)
            state = sim.StreamState(
                stream_id="stream1",
                index=1,
                input_video="/videos/1.mp4",
                uri="rtsp://127.0.0.1:8554/stream1",
                log_path=str(runtime_dir / "ffmpeg_stream1.log"),
            )

            with patch("subprocess.Popen") as popen:
                sim._start_ffmpeg(args, state)
                cmd = popen.call_args.args[0]

        self.assertIn("-re", cmd)
        self.assertIn("-stream_loop", cmd)
        self.assertIn("-1", cmd)
        self.assertIn("-an", cmd)
        self.assertIn("-c:v", cmd)
        self.assertIn("copy", cmd)
        self.assertIn("-pkt_size", cmd)
        self.assertIn("1200", cmd)
        self.assertEqual(cmd[-1], "rtsp://127.0.0.1:8554/stream1")


def _args(**overrides) -> Namespace:
    values = {
        "runtime_dir": Path(".runtime/mediamtx_sim"),
        "host": "127.0.0.1",
        "rtsp_port": 8554,
        "rtmp_port": 1935,
        "webrtc_port": 8889,
        "hls_port": 8888,
        "mount_prefix": "stream",
        "ffmpeg_loglevel": "warning",
        "write_queue_size": 8192,
        "rtsp_pkt_size": 1200,
        "transcode_fps": 15.0,
        "transcode_bitrate": "2500k",
        "transcode_gop": 30,
        "no_realtime": False,
        "transcode": False,
    }
    values.update(overrides)
    return Namespace(**values)


if __name__ == "__main__":
    unittest.main()
