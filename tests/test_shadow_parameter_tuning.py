import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.predict_shadow import Detection
from tools.diagnose_shadow_inference import in_region
from tools.evaluate_damage_xml import match_boxes
from tools.predict_shadow_recovery import serialize
from tools.predict_tuned_shadow import apply_recipes, validate_preset
from tools.refine_shadow_parameters import fast_match, metadata_selection
from tools.tune_shadow_parameters import consensus_candidates, merge_local_candidates, merge_preserving_baseline, pairwise_iou, save_bank, unflip, validate_config


def detection(view, class_id=2, score=.1):
    mask = np.zeros((10, 12), bool)
    mask[2:8, 2:6] = True
    return Detection(np.array([2, 2, 6, 8, score, class_id]), mask, view)


class ParameterTuningTests(unittest.TestCase):
    gate = {"D": .02, "R": .06, "min_support": 2}

    def test_flip_is_inverted_for_boxes_and_every_mask_pixel(self):
        d = detection("left:flip")
        flipped = unflip(d)
        self.assertEqual(flipped.box[:4].tolist(), [6, 2, 10, 8])
        self.assertTrue(np.array_equal(unflip(flipped).mask, d.mask))
        self.assertTrue(np.array_equal(unflip(flipped).box, d.box))

    def test_duplicates_from_the_same_view_are_one_vote(self):
        accepted, _ = consensus_candidates([detection("left:original"), detection("left:original")], self.gate, .25, 12)
        self.assertEqual(accepted, [])

    def test_independent_views_can_support_the_same_damage(self):
        accepted, support = consensus_candidates([detection("left:original"), detection("left:clahe")], self.gate, .25, 12)
        self.assertEqual(len(accepted), 2)
        self.assertEqual(support, [2, 2])

    def test_other_class_and_below_threshold_cannot_vote(self):
        accepted, _ = consensus_candidates([detection("left:original"), detection("left:clahe", class_id=5),
                                             detection("left:gamma", score=.01)], self.gate, .25, 12)
        self.assertEqual(accepted, [])

    def test_merge_keeps_baseline_and_does_not_suppress_other_class(self):
        base = detection("baseline", score=.15)
        same = detection("left:clahe", score=.8)
        rust = detection("left:gamma", class_id=5)
        result = merge_preserving_baseline([base], [same, rust], .5)
        self.assertIs(result[0], base)
        self.assertEqual(len(result), 2)

    def test_zero_area_iou_is_finite(self):
        self.assertTrue(np.isfinite(pairwise_iou(np.zeros((2, 4)))).all())

    def test_empty_candidate_bank_is_saved_without_invalid_reshape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank"
            save_bank(path, [], {2: "D", 5: "R"}, (10, 12), [])
            with np.load(path / "candidate_masks.npz") as data:
                self.assertEqual(data["shape"].tolist(), [0, 10, 12])

    def test_profiles_cannot_filter_below_the_generated_candidate_floor(self):
        config = json.loads(Path("tools/shadow_parameter_profiles.json").read_text())
        validate_config(config)
        config["gates"][0]["D"] = .001
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_stronger_local_detection_can_replace_weaker_baseline(self):
        old, new = detection("baseline", score=.2), detection("left:original", score=.8)
        regions = {"left": (1, 1, 7, 9)}
        self.assertEqual(merge_local_candidates([old], [new], .5, "highest_confidence", regions), [new])

    def test_exterior_baseline_is_protected_from_replacement(self):
        old, new = detection("baseline", score=.2), detection("left:original", score=.8)
        regions = {"right": (8, 5, 12, 10)}
        self.assertEqual(merge_local_candidates([old], [new], .5, "highest_confidence", regions), [old])

    def test_metadata_screen_matches_actual_mask_selection_for_both_policies(self):
        old = detection("baseline", score=.2)
        outside = detection("baseline", score=.1)
        outside.box[:4] = [8, 5, 12, 9]
        outside.mask[:] = False
        outside.mask[5:9, 8:12] = True
        a, b, weak = (detection("left:original", score=.8), detection("left:clahe", score=.7),
                      detection("left:gamma", score=.4))
        a.box[:4] = b.box[:4] = [3, 2, 7, 8]
        a.mask[:] = b.mask[:] = False
        a.mask[2:8, 3:7] = b.mask[2:8, 3:7] = True
        baseline, candidates = [old, outside], [a, b, weak]
        regions = {"left": (1, 1, 7, 9)}
        protected = [not in_region(d, regions["left"]) for d in baseline]
        config = {"agreement_iou": .25, "min_mask_area": 12}
        for policy in ("preserve", "highest_confidence"):
            for overlap in (.5, .85):
                with self.subTest(policy=policy, overlap=overlap):
                    recipe = {"threshold": .6, "min_support": 2, "merge_iou": overlap, "merge_policy": policy}
                    indices, _ = metadata_selection(serialize(baseline, {2: "D"}), serialize(candidates, {2: "D"}),
                                                    recipe, config, protected)
                    accepted, _ = consensus_candidates(candidates, {"D": .6, "R": 1, "min_support": 2}, .25, 12)
                    actual = merge_local_candidates(baseline, accepted, overlap, policy, regions)
                    expected = baseline + candidates
                    self.assertEqual([id(d) for d in actual], [id(expected[i]) for i in indices])

    def test_fast_matching_agrees_with_global_one_to_one_evaluation(self):
        predictions = serialize([detection("one", score=.8), detection("duplicate", score=.7)], {2: "D"})
        truth = [{"index": 0, "xml_object_id": 1, "class": "D", "box": [2, 2, 6, 8]},
                 {"index": 1, "xml_object_id": 2, "class": "D", "box": [8, 5, 12, 9]}]
        fast = fast_match(predictions, truth, {"left": (0, 0, 8, 10), "right": (8, 0, 12, 10)})
        full = match_boxes(predictions, truth, .5)
        self.assertEqual((fast["tp"], fast["fp"], fast["f1"]),
                         tuple(full["classes"]["D"][key] for key in ("tp", "fp", "f1")))
        self.assertEqual((fast["left_tp"], fast["right_tp"]), (1, 0))
        json.dumps(fast)  # Regression: NumPy integer region counts are not JSON-safe.

    def test_empty_metadata_and_empty_truth_are_supported(self):
        recipe = {"threshold": .6, "min_support": 2, "merge_iou": .85, "merge_policy": "highest_confidence"}
        indices, _ = metadata_selection([], [], recipe, {"min_mask_area": 12, "agreement_iou": .25}, [])
        self.assertEqual(indices, [])
        self.assertEqual(fast_match([{"box": [2, 2, 6, 8], "score": .8}], [], {})["fp"], 1)

    def test_production_selection_does_not_use_annotation_matching_fields(self):
        preset = {"classes": {
            "D": {"bank": "test", "threshold": .6, "min_support": 2, "merge_iou": .85, "merge_policy": "preserve"},
            "R": {"bank": "test", "threshold": .12, "min_support": 1, "merge_iou": .5, "merge_policy": "highest_confidence"}},
            "regions": {"left": (1, 1, 7, 9)}, "agreement_iou": .25, "min_mask_area": 12}
        base = [detection("baseline", class_id=5, score=.15)]
        raw = {"test": [detection("left:original", class_id=5, score=.8)]}
        first, _ = apply_recipes(base, raw, preset)
        for recipe in preset["classes"].values():
            recipe["matched_ids"] = [99999]
            recipe["tp"] = -99999
        second, _ = apply_recipes(base, raw, preset)
        self.assertEqual([id(d) for d in first], [id(d) for d in second])
        self.assertEqual(len(first), 1)
        self.assertIs(first[0], raw["test"][0])

    def test_source_mismatch_is_rejected_before_inference(self):
        preset = {"source_filename": "CBHU0708054-A.JPG", "image_shape": [600, 800]}
        with self.assertRaises(ValueError):
            validate_preset(preset, Path("other.jpg"), (600, 800))
        with self.assertRaises(ValueError):
            validate_preset(preset, Path("CBHU0708054-A.JPG"), (640, 640))


if __name__ == "__main__":
    unittest.main()
