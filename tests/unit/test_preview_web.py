from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "tools"))

from preview_web import _individual_playback, _rtsp_streams


class PreviewWebTests(unittest.TestCase):
    def test_individual_outputs_are_mapped_to_rtsp_stream_playback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rtsp_dir = Path(tmp)
            video = rtsp_dir / "individual" / "stream_01" / "stream_01_osd.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"mp4")
            (rtsp_dir / "individual" / "individual_outputs.json").write_text(
                json.dumps(
                    {
                        "outputs": [
                            {
                                "stream_id": "stream_01",
                                "video": str(video),
                                "video_exists": True,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            playback = _individual_playback(rtsp_dir)
            streams = _rtsp_streams(
                {"expected_stream_count": 1, "streams": {"stream-0": {}}},
                {},
                {},
                {},
                rtsp_dir=rtsp_dir,
                individual_playback=playback,
            )

        expected_url = "/rtsp-files/individual/stream_01/stream_01_osd.mp4"
        self.assertEqual(playback, {"stream-0": expected_url})
        self.assertEqual(streams[0]["playback"], expected_url)
        self.assertEqual(streams[0]["preview_playback"], expected_url)


if __name__ == "__main__":
    unittest.main()
