"""Verify crop remapping, regional eligibility, and preservation of existing masks."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from tools.diagnose_shadow_inference import (
    append_candidates,
    evaluate,
    in_region,
    map_detection,
    pixel_bounds,
    process_image,
)
from tools.predict_shadow import Detection


class ShadowDiagnosticTests(unittest.TestCase):
    def inference_args(self, versions=None):
        return SimpleNamespace(
            versions=versions,
            regions=["left", "right"],
            left_roi=(0, 0, 0.4, 1),
            right_roi=(0.6, 0, 1, 1),
            left_context=(0, 0, 0.5, 1),
            right_context=(0.5, 0, 1, 1),
            imgsz=640,
            conf=0.15,
            diagnostic_conf=0.01,
            iou=0.7,
            device="cpu",
            gamma=0.55,
            clahe_clip=3.0,
            shadow_threshold=180,
            merge_iou=0.6,
        )

    def test_crop_mask_and_box_are_translated_without_resizing(self):
        mask = np.zeros((4, 5), dtype=bool)
        mask[1:3, 2:4] = True
        source = Detection(np.array([2, 1, 4, 3, 0.2, 2]), mask, "crop")
        mapped = map_detection(source, (10, 20), (30, 40))
        np.testing.assert_array_equal(mapped.box[:4], [12, 21, 14, 23])
        self.assertEqual(int(mapped.mask.sum()), 4)
        np.testing.assert_array_equal(mapped.mask[20:24, 10:15], mask)
        np.testing.assert_array_equal(source.box[:4], [2, 1, 4, 3])
        with self.assertRaises(ValueError):
            map_detection(source, (39, 29), (30, 40))

    def test_majority_rule_excludes_adjacent_long_rust_masks(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[2:18, 8:12] = True
        detection = Detection(np.array([8, 2, 12, 18, 0.8, 5]), mask, "original")
        self.assertFalse(in_region(detection, (0, 0, 9, 20)))
        self.assertTrue(in_region(detection, (7, 1, 13, 19)))

    def test_baseline_is_preserved_and_different_classes_can_overlap(self):
        mask = np.zeros((12, 12), dtype=bool)
        mask[2:9, 2:9] = True
        original = Detection(np.array([2, 2, 9, 9, 0.2, 2]), mask, "original")
        higher = Detection(np.array([2, 2, 9, 9, 0.8, 2]), mask.copy(), "clahe")
        rust = Detection(np.array([2, 2, 9, 9, 0.4, 5]), mask.copy(), "clahe")
        selected = append_candidates([original], [higher, rust], 0.6)
        self.assertEqual(len(selected), 2)
        self.assertIs(selected[0], original)
        self.assertIs(selected[1], rust)

    def test_invalid_roi_is_rejected(self):
        self.assertEqual(pixel_bounds((0.1, 0.2, 0.8, 0.9), (100, 200)), (20, 20, 160, 90))
        with self.assertRaises(ValueError):
            pixel_bounds((0.8, 0.2, 0.1, 0.9), (100, 200))

    def test_s08_view_selection_predicts_once_without_enhancement(self):
        args = self.inference_args(["S08_diagnostic_only"])
        args.image_output = Path("unused")
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        model = Mock(names={2: "D", 5: "R"})
        model.predict.return_value = [SimpleNamespace(boxes=[])]
        with patch("tools.diagnose_shadow_inference.enhance_shadow") as enhance, patch(
            "tools.diagnose_shadow_inference.extract_detections", return_value=[]
        ), patch("tools.diagnose_shadow_inference.write_image"), redirect_stdout(io.StringIO()):
            views, diagnostics = evaluate(
                model, image, args, {"left": (0, 0, 40, 100)}, {"left": (0, 0, 50, 100)}, {"full_original"}
            )
        model.predict.assert_called_once()
        enhance.assert_not_called()
        self.assertIs(model.predict.call_args.kwargs["source"], image)
        self.assertEqual(model.predict.call_args.kwargs["conf"], 0.01)
        self.assertEqual(list(views), ["full_original"])
        self.assertEqual([d["view"] for d in diagnostics], ["full_original"])

    def test_s08_only_matches_s08_from_full_run_and_keeps_default_nine_views(self):
        def detected(box, score, cid):
            mask = np.zeros((100, 100), dtype=bool)
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = True
            return Detection(np.array([*box, score, cid]), mask, "full_original")

        original = [
            detected([5, 20, 15, 30], 0.2, 2),
            detected([65, 50, 75, 60], 0.02, 2),
            detected([25, 50, 35, 60], 0.015, 5),
            detected([45, 50, 55, 60], 0.02, 5),
            detected([45, 70, 55, 80], 0.7, 5),
        ]

        def extract(result, label, shape):
            return original if label == "full_original" else []

        model = Mock(names={2: "D", 5: "R"})
        model.predict.return_value = [SimpleNamespace(boxes=original)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.jpg"
            cv2.imencode(".jpg", np.zeros((100, 100, 3), np.uint8))[1].tofile(source)
            full_args = self.inference_args()
            single_args = copy(full_args)
            single_args.versions = ["S08_diagnostic_only"]
            with patch("tools.diagnose_shadow_inference.extract_detections", side_effect=extract), redirect_stdout(
                io.StringIO()
            ):
                full, _ = process_image(model, source, root / "full", full_args)
                self.assertEqual(model.predict.call_count, 9)
                model.predict.reset_mock()
                single, _ = process_image(model, source, root / "single", single_args)
                model.predict.assert_called_once()
            self.assertEqual(len(full), 8)
            self.assertEqual(len(single), 1)
            self.assertEqual([p.name for p in (root / "single").glob("S*")], ["S08_diagnostic_only"])
            self.assertEqual(len(list((root / "single" / "inputs").glob("*.png"))), 1)
            for filename in ("predictions.json", "prediction_metrics.json", "input__all.jpg"):
                self.assertEqual(
                    (root / "single/S08_diagnostic_only" / filename).read_bytes(),
                    (root / "full/S08_diagnostic_only" / filename).read_bytes(),
                )
            saved = json.loads((root / "single/S08_diagnostic_only/predictions.json").read_text(encoding="utf-8"))
            self.assertEqual(len(saved["detections"]), 4)
            # Low-score rust outside either inspection ROI remains excluded;
            # high-score baseline rust outside the ROIs remains preserved.
            self.assertEqual(sorted(d["score"] for d in saved["detections"]), [0.015, 0.02, 0.2, 0.7])


if __name__ == "__main__":
    unittest.main()
