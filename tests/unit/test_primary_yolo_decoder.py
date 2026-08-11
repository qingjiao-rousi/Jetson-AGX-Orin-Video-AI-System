from __future__ import annotations

import unittest

import numpy as np

from app.application.primary_yolo_decoder import decode_primary_yolov8_output


class PrimaryYoloDecoderTests(unittest.TestCase):
    def test_decodes_xyxy_score_class_and_undoes_letterbox(self) -> None:
        # A 640x480 source image receives 80 pixels of vertical padding.
        output = np.array([[[100, 100, 300, 400, 0.9, 0], [102, 101, 299, 399, 0.8, 0]]], dtype=np.float32)
        detections = decode_primary_yolov8_output(
            output, image_shape=(480, 640), labels=("person",),
            confidence_threshold=0.25, nms_iou_threshold=0.45,
        )
        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual(detection.class_id, 0)
        self.assertAlmostEqual(detection.bbox.left, 100.0)
        self.assertAlmostEqual(detection.bbox.top, 20.0)
        self.assertAlmostEqual(detection.bbox.width, 200.0)
        self.assertAlmostEqual(detection.bbox.height, 300.0)

    def test_rejects_a_non_primary_output_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "anchors, 6"):
            decode_primary_yolov8_output(
                np.zeros((1, 84, 8400), dtype=np.float32), image_shape=(640, 640), labels=("person",),
                confidence_threshold=0.25, nms_iou_threshold=0.45,
            )

    def test_discards_invalid_class_and_confidence(self) -> None:
        output = np.array([[[0, 0, 10, 10, 1.2, 0], [0, 0, 10, 10, 0.9, 3]]], dtype=np.float32)
        detections = decode_primary_yolov8_output(
            output, image_shape=(640, 640), labels=("person",),
            confidence_threshold=0.25, nms_iou_threshold=0.45,
        )
        self.assertEqual(detections, ())
