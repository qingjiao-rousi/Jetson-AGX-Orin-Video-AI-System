from __future__ import annotations

import unittest

import numpy as np

from app.application.helmet_service import (
    HelmetAssociator,
    HelmetDetection,
    HelmetEventTracker,
    bbox_iom,
    crop_person_roi,
    decode_yolov8_output,
)
from app.domain.entities import BoundingBox


class HelmetServiceTests(unittest.TestCase):
    def test_crop_person_roi_and_iom_association(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        roi, rect = crop_person_roi(
            frame,
            BoundingBox(left=100, top=100, width=200, height=300),
            padding_ratio=0,
        )
        self.assertEqual(roi.shape[:2], (300, 200))
        self.assertEqual(rect, (100, 100, 200, 300))

        person = BoundingBox(100, 100, 200, 300)
        hardhat = HelmetDetection(4, "hardhat", 0.9, BoundingBox(140, 120, 40, 30))
        assessment = HelmetAssociator(min_iom=0.2).associate(
            person,
            (hardhat,),
            stream_id="stream-0",
            track_id=3,
            frame_id=10,
        )
        self.assertEqual(assessment.status, "wearing")
        self.assertGreaterEqual(bbox_iom(hardhat.bbox, person), 0.99)

    def test_decode_yolov8_raw_output(self) -> None:
        # 17 PPE classes: 4 box values + 17 class scores.
        output = np.zeros((1, 21, 1), dtype=np.float32)
        output[0, 0:4, 0] = [100, 80, 40, 30]
        output[0, 4 + 4, 0] = 0.91  # hardhat
        detections = decode_yolov8_output(
            output,
            roi_shape=(640, 640),
            labels=("barricade", "dumpster", "excavators", "gloves", "hardhat"),
            confidence_threshold=0.25,
        )
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_name, "hardhat")
        self.assertAlmostEqual(detections[0].bbox.left, 80.0)

    def test_event_requires_consecutive_not_wearing_assessments(self) -> None:
        tracker = HelmetEventTracker(confirm_frames=3, cooldown_frames=30)
        person = BoundingBox(100, 100, 200, 300)
        for frame_id in (1, 2):
            assessment = tracker_assessment(frame_id, "not_wearing")
            self.assertIsNone(tracker.update(assessment, person))
        event = tracker.update(tracker_assessment(3, "not_wearing"), person)
        self.assertIsNotNone(event)
        self.assertEqual(event.event_type, "helmet_violation")
        self.assertIsNone(tracker.update(tracker_assessment(4, "not_wearing"), person))


def tracker_assessment(frame_id: int, status: str):
    from app.application.helmet_service import HelmetAssessment

    return HelmetAssessment("stream-0", 3, frame_id, status, 0.9)


if __name__ == "__main__":
    unittest.main()
