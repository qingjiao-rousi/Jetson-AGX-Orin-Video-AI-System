from __future__ import annotations

import unittest

from app.application.coco_evaluation import coco_box_iou, fixed_threshold_person_metrics, validate_coco_label_mapping


class CocoEvaluationTests(unittest.TestCase):
    def test_maps_contiguous_model_indexes_to_coco_ids(self) -> None:
        mapping = validate_coco_label_mapping(("person", "bicycle"), ({"id": 1, "name": "person"}, {"id": 2, "name": "bicycle"}))
        self.assertEqual(mapping, {0: 1, 1: 2})

    def test_rejects_missing_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "not present"):
            validate_coco_label_mapping(("person", "not-a-coco-class"), ({"id": 1, "name": "person"},))

    def test_fixed_threshold_person_metrics_are_one_to_one(self) -> None:
        annotations = [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 0},
            {"image_id": 1, "category_id": 1, "bbox": [30, 30, 10, 10], "iscrowd": 0},
        ]
        predictions = [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.8},
            {"image_id": 1, "category_id": 1, "bbox": [30, 30, 10, 10], "score": 0.2},
        ]
        result = fixed_threshold_person_metrics(predictions, annotations, person_category_id=1, score_threshold=0.25)
        self.assertEqual((result["true_positive"], result["false_positive"], result["false_negative"]), (1, 1, 1))
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["f1"], 0.5)

    def test_coco_box_iou(self) -> None:
        self.assertAlmostEqual(coco_box_iou([0, 0, 10, 10], [5, 0, 10, 10]), 1 / 3)


if __name__ == "__main__":
    unittest.main()
