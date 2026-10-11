"""Numerical checks of pixel metrics, macro averaging, and overlap handling."""

import unittest
import numpy as np

from tools.pixel_metrics import METRIC_COLUMNS, confusion_counts, metric_values, rasterize_boxes, summarize_counts


class PixelMetricsTests(unittest.TestCase):
    def test_all_metrics_from_known_nontrivial_confusion(self):
        counts = {"TP": 2, "FP": 1, "FN": 1, "TN": 4}
        overall, classes = summarize_counts({"D": counts})
        expected = {"Accuracy": .75, "Precision": 2/3, "Recall": 2/3, "Dice": 2/3,
                    "IoU": .5, "mIoU": .5, "Mean Dice": 2/3, "FPR": .2, "FNR": 1/3}
        for name, value in expected.items():
            self.assertAlmostEqual(overall[name], value)
            if name not in {"mIoU", "Mean Dice"}:
                self.assertAlmostEqual(classes["D"][name], value)
        self.assertIsNone(classes["D"]["mIoU"])
        self.assertIsNone(classes["D"]["Mean Dice"])
        self.assertTrue(set(METRIC_COLUMNS).issubset(overall))

    def test_boolean_masks_and_true_negatives_use_all_image_pixels(self):
        truth = np.array([[1,1,1,0], [0,0,0,0]], dtype=bool)
        prediction = np.array([[1,1,0,1], [0,0,0,0]], dtype=bool)
        self.assertEqual(confusion_counts(prediction, truth), {"TP": 2, "FP": 1, "FN": 1, "TN": 4})
        with self.assertRaises(ValueError):
            confusion_counts(prediction.astype(np.uint8), truth)

    def test_empty_class_not_artificially_perfect_or_added_to_macro_mean(self):
        overall, classes = summarize_counts({"D": {"TP": 2, "FP": 1, "FN": 1, "TN": 4},
                                            "R": {"TP": 0, "FP": 0, "FN": 0, "TN": 8}})
        self.assertAlmostEqual(overall["mIoU"], .5)
        self.assertIsNone(classes["R"]["IoU"])
        self.assertIsNone(classes["R"]["Dice"])
        self.assertIsNone(classes["R"]["Precision"])
        self.assertEqual(classes["R"]["Accuracy"], 1)
        self.assertEqual(classes["R"]["FPR"], 0)

    def test_pooled_metrics_differ_from_unweighted_image_average(self):
        first = {"TP": 1, "FP": 0, "FN": 0, "TN": 0}
        second = {"TP": 0, "FP": 9, "FN": 0, "TN": 0}
        pooled = {key: first[key] + second[key] for key in first}
        self.assertEqual(metric_values(pooled)["Accuracy"], .1)
        self.assertEqual(metric_values(pooled)["Precision"], .1)

    def test_box_area_union_and_multilabel_overlap_do_not_count_instances_twice(self):
        entries = [{"class": "D", "box": [0,0,2,2]}, {"class": "D", "box": [1,0,3,2]},
                   {"class": "R", "box": [1,0,2,2]}]
        masks = rasterize_boxes(entries, ["D", "R"], 2, 4)
        self.assertEqual(int(masks["D"].sum()), 6)
        self.assertEqual(int(masks["R"].sum()), 2)
        self.assertEqual(int((masks["D"] & masks["R"]).sum()), 2)


if __name__ == "__main__":
    unittest.main()
