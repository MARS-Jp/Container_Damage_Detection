"""Check class threshold alignment and union-area calculations without model inference."""

import unittest

import numpy as np
import torch

from tools.predict_seg_by_class import class_thresholds, filter_class_scores, prediction_metrics
from ultralytics.engine.results import Results


class ClassThresholdTests(unittest.TestCase):
    def test_names_ids_fallback_and_invalid_inputs(self):
        names = {0: "B", 2: "D", 5: "R"}
        self.assertEqual(class_thresholds(["d=0.08", "5=0.20"], names, 0.15), {0: 0.15, 2: 0.08, 5: 0.20})
        for entries in (["unknown=0.1"], ["D=nan"], ["D=-0.1"], ["D=1.1"], ["D=0.1", "2=0.2"], ["0.1"]):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                class_thresholds(entries, names, 0.15)

    def test_filtering_keeps_matching_boxes_and_masks(self):
        names = {0: "B", 2: "D", 5: "R"}
        boxes = torch.tensor([
            [0, 0, 3, 3, 0.09, 2], [1, 1, 4, 4, 0.09, 5],
            [2, 2, 5, 5, 0.30, 5], [3, 3, 6, 6, 0.12, 0],
        ])
        masks = torch.zeros((4, 8, 8))
        for index in range(4):
            masks[index, index:index + 2, index:index + 2] = 1
        result = Results(np.zeros((8, 8, 3), dtype=np.uint8), "image.jpg", names, boxes=boxes, masks=masks)
        filtered = filter_class_scores(result, {0: 0.15, 2: 0.08, 5: 0.20})
        self.assertTrue(torch.equal(filtered.boxes.data, boxes[[0, 2]]))
        self.assertTrue(torch.equal(filtered.masks.data, masks[[0, 2]]))
        self.assertEqual(len(result.boxes), 4)

    def test_union_area_does_not_double_count_overlaps(self):
        masks = np.zeros((3, 4, 4), dtype=bool)
        masks[0, :2, :2] = True
        masks[1, 1:3, 1:3] = True
        masks[2, :2, :2] = True
        metrics = prediction_metrics(masks, np.array([2, 2, 5]), {2: "D", 5: "R"}, (4, 4))
        self.assertEqual(metrics["classes"]["2"]["count"], 2)
        self.assertEqual(metrics["classes"]["2"]["union_pixels"], 7)
        self.assertEqual(metrics["classes"]["2"]["coverage_percent"], 43.75)
        self.assertEqual(metrics["overlap_pixels"]["2:5"], 4)


if __name__ == "__main__":
    unittest.main()
