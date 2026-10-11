"""Independent pixel reconstruction and object matching edge cases."""
import unittest
import numpy as np
from tools.audit_inference_metrics import rectangle_union, binary_counts, object_counts, pooled


class AuditInferenceMetricsTests(unittest.TestCase):
    def test_pixel_center_boundaries_and_duplicate_rectangle_union(self):
        result = rectangle_union([[.5, .5, 2.5, 1.5], [.5, .5, 2.5, 1.5]], 2, 4)
        np.testing.assert_array_equal(result, [[True, True, False, False], [False, False, False, False]])

    def test_all_background_baseline_and_false_negative_counts(self):
        truth = np.array([[True, False, False, False]], dtype=bool)
        prediction = np.zeros_like(truth)
        counts = binary_counts(prediction, truth)
        self.assertEqual(counts, {"TP": 0, "FP": 0, "FN": 1, "TN": 3})
        values = pooled({"D": counts})
        self.assertEqual(values["Accuracy"], .75)
        self.assertEqual(values["always_background_Accuracy"], .75)
        self.assertEqual(values["Recall"], 0)
        self.assertIsNone(values["Precision"])

    def test_object_duplicate_cannot_be_second_true_positive(self):
        predictions = [{"class": "D", "box": [0,0,2,2], "score": .8},
                       {"class": "D", "box": [0,0,2,2], "score": .9}]
        truth = [{"class": "D", "box": [0,0,2,2]}]
        result = object_counts(predictions, truth)
        self.assertEqual((result["TP"], result["FP"], result["FN"]), (1,1,0))
        self.assertEqual(result["Precision"], .5)


if __name__ == "__main__":
    unittest.main()
