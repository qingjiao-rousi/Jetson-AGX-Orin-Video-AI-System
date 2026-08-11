from __future__ import annotations

import unittest

from app.application.detection_alignment import bbox_iou, match_detections
from app.domain.entities import BoundingBox, Detection


def detection(class_id: int, confidence: float, left: float, top: float, width: float, height: float) -> Detection:
    return Detection(class_id, str(class_id), confidence, BoundingBox(left, top, width, height))


class DetectionAlignmentTests(unittest.TestCase):
    def test_matches_same_class_once_at_highest_iou(self) -> None:
        fp16 = (detection(0, 0.90, 0, 0, 10, 10), detection(0, 0.80, 20, 20, 10, 10))
        int8 = (detection(0, 0.85, 1, 1, 10, 10), detection(0, 0.70, 100, 100, 10, 10))
        matches, fp16_only, int8_only = match_detections(fp16, int8, iou_threshold=0.5)
        self.assertEqual(len(matches), 1)
        self.assertEqual((matches[0].reference_index, matches[0].candidate_index), (0, 0))
        self.assertAlmostEqual(matches[0].confidence_delta, -0.05)
        self.assertEqual(fp16_only, (1,))
        self.assertEqual(int8_only, (1,))

    def test_never_matches_different_classes(self) -> None:
        fp16 = (detection(0, 0.9, 0, 0, 10, 10),)
        int8 = (detection(1, 0.9, 0, 0, 10, 10),)
        matches, fp16_only, int8_only = match_detections(fp16, int8, iou_threshold=0.5)
        self.assertEqual(matches, ())
        self.assertEqual(fp16_only, (0,))
        self.assertEqual(int8_only, (0,))

    def test_bbox_iou_handles_no_overlap(self) -> None:
        self.assertEqual(bbox_iou(detection(0, 0.9, 0, 0, 1, 1), detection(0, 0.9, 2, 2, 1, 1)), 0.0)


if __name__ == "__main__":
    unittest.main()
