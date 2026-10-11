import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tools.evaluate_damage_xml import evaluate_boxes, match_boxes, read_voc
from tools.predict_shadow import Detection
from tools.predict_shadow_recovery import auxiliary_detections, deduplicate, shadow_rust_candidates, tile_bounds
from tools.damage_segmentation_predictor import DamageSegmentationPredictor


class RecoveryTests(unittest.TestCase):
    def test_duplicate_cannot_match_one_annotation_twice(self):
        truth = [{"index": 0, "class": "D", "box": [2, 2, 8, 8]}]
        pred = [{"class": "D", "score": score, "box": [2, 2, 8, 8]} for score in (.9, .8)]
        metrics = match_boxes(pred, truth, .5)["classes"]["D"]
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 1, 0))

    def test_rust_cannot_match_dent_at_same_position(self):
        truth = [{"index": 0, "class": "D", "box": [2, 2, 8, 8]}]
        pred = [{"class": "R", "score": .9, "box": [2, 2, 8, 8]}]
        self.assertEqual(match_boxes(pred, truth, .3)["classes"]["D"]["fn"], 1)

    def test_global_match_assigns_region_by_truth_center(self):
        truth = [{"index": 0, "class": "R", "box": [3, 2, 9, 8]}]
        pred = [{"class": "R", "score": .9, "box": [2, 2, 8, 8]}]
        regions = {"left": (0, 0, 6, 10), "right": (6, 0, 10, 10)}
        metrics = evaluate_boxes(pred, truth, regions)["0.5"]["regions"]
        self.assertEqual(metrics["left"]["R"]["gt"], 0)
        self.assertEqual(metrics["right"]["R"]["tp"], 1)

    def test_all_edges_are_covered_without_annotation_coordinates(self):
        coverage = np.zeros((600, 800), dtype=bool)
        for x1, y1, x2, y2 in tile_bounds(coverage.shape, 320, .4):
            coverage[y1:y2, x1:x2] = True
        self.assertTrue(coverage.all())

    def test_auxiliary_keeps_holes_and_independent_classes(self):
        maps = np.zeros((8, 12, 12), dtype=np.float32)
        maps[2, 2:9, 2:9] = .8
        maps[2, 4:7, 4:7] = .1
        maps[5, 2:9, 2:9] = .9
        detections = auxiliary_detections(maps, .5, "aux")
        self.assertEqual(len(detections), 2)
        self.assertFalse(detections[0].mask[5, 5])
        self.assertTrue(detections[1].mask[5, 5])
        self.assertEqual(len(deduplicate([], detections)), 2)

    def test_duplicate_addition_does_not_replace_baseline(self):
        mask = np.ones((4, 4), bool)
        baseline = Detection(np.array([0, 0, 4, 4, .2, 2]), mask, "base")
        addition = Detection(np.array([0, 0, 4, 4, .9, 2]), mask, "tile")
        self.assertIs(deduplicate([baseline], [addition])[0], baseline)
        self.assertEqual(len(deduplicate([baseline], [addition])), 1)

    def test_rust_recovery_does_not_add_dent_or_bright_predictions(self):
        mask = np.ones((10, 10), bool)
        rust = Detection(np.array([0, 0, 10, 10, .3, 5]), mask, "crop")
        dent = Detection(np.array([0, 0, 10, 10, .9, 2]), mask, "crop")
        regions = {"shadow": (0, 0, 10, 10)}
        self.assertEqual(shadow_rust_candidates([rust, dent], np.full((10, 10, 3), 80, np.uint8), regions), [rust])
        self.assertEqual(shadow_rust_candidates([rust], np.full((10, 10, 3), 200, np.uint8), regions), [])
        self.assertEqual(shadow_rust_candidates([rust], np.full((10, 10, 3), 80, np.uint8), {"shadow": (0, 0, 2, 2)}), [])

    def test_predictor_preserves_both_classes_and_raw_coordinates(self):
        # Raw xywh, D/R scores, then one mask coefficient. Isolate predictor
        # postprocessing so the test doesn't need a downloaded checkpoint.
        predictor = object.__new__(DamageSegmentationPredictor)
        from types import SimpleNamespace
        predictor.args = SimpleNamespace(conf=.2, iou=.5, classes=None, agnostic_nms=False, max_det=100)
        predictor.model = SimpleNamespace(names={0: "D", 1: "R"}, end2end=False)
        predictor.collect_auxiliary = False
        raw = torch.tensor([[[5.], [5.], [4.], [4.], [.8], [.7], [1.]]])
        original = raw.clone()
        predictor.construct_results = lambda detections, *args: detections
        predictor.multi_label = True
        result = predictor.postprocess(((raw, torch.zeros(1, 1, 2, 2)), {}), torch.zeros(1, 3, 10, 10), [np.zeros((10, 10, 3))])
        self.assertEqual(set(result[0][:, 5].tolist()), {0., 1.})
        self.assertTrue(torch.equal(raw, original))
        predictor.multi_label = False
        result = predictor.postprocess(((raw, torch.zeros(1, 1, 2, 2)), {}), torch.zeros(1, 3, 10, 10), [np.zeros((10, 10, 3))])
        self.assertEqual(result[0].shape[0], 1)

    def test_xml_cannot_be_evaluated_against_a_different_image(self):
        xml = '<annotation><filename>other.JPG</filename><size><width>8</width><height>6</height></size></annotation>'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.xml"
            path.write_text(xml)
            with self.assertRaises(ValueError):
                read_voc(path, (6, 8), "input.JPG")

    def test_voc_origin_is_consistent_with_existing_evaluator(self):
        xml = ('<annotation><filename>input.JPG</filename><size><width>8</width><height>6</height></size>'
               '<object><name>D</name><bndbox><xmin>2</xmin><ymin>2</ymin><xmax>5</xmax>'
               '<ymax>5</ymax></bndbox></object></annotation>')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.xml"
            path.write_text(xml)
            self.assertEqual(read_voc(path, (6, 8), "input.JPG")[0]["box"], [1, 1, 5, 5])
            self.assertEqual(read_voc(path, (6, 8), "input.JPG", origin=0)[0]["box"], [2, 2, 5, 5])


if __name__ == "__main__":
    unittest.main()
