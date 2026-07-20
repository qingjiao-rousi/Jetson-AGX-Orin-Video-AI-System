from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.infrastructure.pipeline.cpp_probe import CppProbeHandler


class CppProbeAdapterTests(unittest.TestCase):
    def test_native_payload_is_normalized_to_existing_batch_contract(self) -> None:
        payload = CppProbeHandler.to_batch_payload(
            [
                {
                    "stream_id": "stream-2",
                    "frame_id": 17,
                    "ntp_timestamp": 123,
                    "detections": [
                        {
                            "class_id": 0,
                            "class_name": "person",
                            "confidence": 0.9,
                            "track_id": 42,
                            "bbox": {"left": 1, "top": 2, "width": 3, "height": 4},
                        }
                    ],
                }
            ]
        )

        frame = payload["frame_meta_list"][0]
        self.assertEqual(frame["stream_id"], "stream-2")
        self.assertEqual(frame["frame_num"], 17)
        self.assertEqual(frame["obj_meta_list"][0]["object_id"], 42)
        self.assertEqual(frame["obj_meta_list"][0]["rect_params"]["width"], 3)

    def test_invalid_native_payload_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            CppProbeHandler.to_batch_payload({})


if __name__ == "__main__":
    unittest.main()
