from __future__ import annotations

import unittest

from app.application.coco_calibration import select_calibration_images


class CocoCalibrationTests(unittest.TestCase):
    def test_selection_is_seeded_and_sorted_by_id(self) -> None:
        images = [{"id": index, "file_name": f"{index}.jpg"} for index in range(10)]
        first = select_calibration_images(images, count=4, seed=7)
        second = select_calibration_images(list(reversed(images)), count=4, seed=7)
        self.assertEqual(first, second)
        self.assertEqual([item["id"] for item in first], sorted(item["id"] for item in first))

    def test_rejects_request_larger_than_population(self) -> None:
        with self.assertRaisesRegex(ValueError, "only 1"):
            select_calibration_images([{"id": 1}], count=2, seed=1)


if __name__ == "__main__":
    unittest.main()
