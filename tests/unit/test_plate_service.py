from __future__ import annotations

import unittest

import numpy as np

from app.application.plate_service import decode_ocr_output, decode_plate_detector_output


class PlateServiceTests(unittest.TestCase):
    def test_decodes_end2end_plate_box(self) -> None:
        output = np.array([[0, 96, 120, 288, 192, 0, 0.92]], dtype=np.float32)
        detections = decode_plate_detector_output(
            output,
            roi_shape=(384, 384),
            input_size=384,
        )
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].confidence, 0.92, places=2)
        self.assertEqual(detections[0].bbox.width, 192)

    def test_decodes_ocr_slots_and_padding(self) -> None:
        alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
        output = np.zeros((1, 9, len(alphabet)), dtype=np.float32)
        output[0, :, alphabet.index("_")] = 0.6
        output[0, 0, alphabet.index("A")] = 0.9
        output[0, 1, alphabet.index("1")] = 0.8
        output[0, 2, alphabet.index("_")] = 0.95
        text, confidence = decode_ocr_output(output, alphabet=alphabet)
        self.assertEqual(text, "A1")
        self.assertGreater(confidence, 0.6)


if __name__ == "__main__":
    unittest.main()
