"""Offline checks for shadow enhancement, mask fusion and class-separated output.

Run without the repository's model-downloading pytest hooks:
    python -m unittest discover -s tests -p test_predict_shadow.py -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.predict_shadow import (
    Detection,
    collect_images,
    enhance_shadow,
    exclude_background,
    exclusion_masks,
    extract_detections,
    merge_detections,
    parse_args,
    predict_image,
    render,
    write_image,
)


class ArrayTensor:
    """Supply only the tensor interface used to read a prediction."""

    def __init__(self, data):
        self.data = data

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class Boxes:
    def __init__(self, data):
        self.data = ArrayTensor(data)

    def __len__(self):
        return len(self.data.data)


def result_for(detections):
    boxes = np.array([d.box for d in detections], dtype=np.float32).reshape(-1, 6)
    masks = SimpleNamespace(data=ArrayTensor(np.stack([d.mask for d in detections]))) if detections else None
    return SimpleNamespace(boxes=Boxes(boxes), masks=masks)


def detection(mask, view="original", score=0.8, class_id=0):
    return Detection(np.array([0, 0, mask.shape[1], mask.shape[0], score, class_id]), mask, view)


class ShadowInferenceTests(unittest.TestCase):
    def setUp(self):
        self.args = parse_args([])
        self.mask = np.zeros((32, 64), dtype=bool)
        self.mask[4:20, 8:24] = True

    def test_background_exclusion_only_clips_rust_and_keeps_container_pixels(self):
        excluded = np.zeros((32, 64), dtype=bool)
        excluded[:, 48:] = True
        crossing = np.zeros_like(excluded)
        crossing[8:16, 40:56] = True
        rust = detection(crossing, class_id=5)
        dent = detection(crossing.copy(), class_id=2)
        column = detection(excluded.copy(), class_id=5)
        kept = exclude_background([rust, dent, column], {5: excluded})
        self.assertEqual(len(kept), 2)
        np.testing.assert_array_equal(kept[0].mask[:, :48], crossing[:, :48])
        self.assertFalse(kept[0].mask[:, 48:].any())
        self.assertIs(kept[1], dent)
        np.testing.assert_array_equal(rust.mask, crossing)

    def test_exclusion_is_bound_to_full_source_path_not_filename(self):
        source = Path("original/image.jpg").resolve()
        rule = {"source": str(source), "classes": ["R"], "polygon_normalized": [[0.75, 0], [1, 0], [1, 1], [0.75, 1]]}
        masks, matched = exclusion_masks(source, (32, 64), {2: "D", 5: "R"}, [rule])
        self.assertEqual(matched, [rule])
        self.assertEqual(set(masks), {5})
        self.assertFalse(masks[5][:, :47].any())
        self.assertTrue(masks[5][:, 48:].all())
        self.assertEqual(exclusion_masks(Path("another/image.jpg"), (32, 64), {5: "R"}, [rule]), ({}, []))

    def test_enhancement_lifts_dark_pixels_without_moving_or_changing_bright_pixels(self):
        image = np.full((64, 64, 3), 240, dtype=np.uint8)
        image[:, :32] = 40
        saved = image.copy()
        for view in ("gamma", "clahe"):
            with self.subTest(view=view):
                enhanced = enhance_shadow(image, view, self.args)
                self.assertEqual(enhanced.shape, image.shape)
                self.assertEqual(enhanced.dtype, image.dtype)
                self.assertGreater(enhanced[:, :32].mean(), image[:, :32].mean())
                np.testing.assert_array_equal(enhanced[:, 32:], image[:, 32:])
                np.testing.assert_array_equal(image, saved)

    def test_gamma_identity_and_extreme_inputs(self):
        image = np.random.default_rng(12).integers(0, 256, (24, 36, 3), dtype=np.uint8)
        self.args.gamma = 1.0
        np.testing.assert_array_equal(enhance_shadow(image, "gamma", self.args), image)
        for value in (0, 255):
            for view in ("gamma", "clahe"):
                enhanced = enhance_shadow(np.full((8, 8, 3), value, np.uint8), view, self.args)
                self.assertTrue(np.isfinite(enhanced).all())

    def test_duplicate_keeps_higher_score_with_matching_mask_and_provenance(self):
        original = detection(self.mask)
        gamma = detection(self.mask.copy(), "gamma", 0.9)
        merged = merge_detections([original, gamma], 0.6)
        self.assertEqual(len(merged), 1)
        self.assertIs(merged[0], gamma)
        self.assertEqual(merge_detections([], 0.6), [])

    def test_overlapping_classes_and_distinct_masks_are_preserved(self):
        adjacent = np.roll(self.mask, 20, axis=1)
        candidates = [
            detection(self.mask),
            detection(self.mask.copy(), "gamma", class_id=1),
            detection(adjacent, "clahe"),  # Identical boxes, disjoint mask pixels.
        ]
        self.assertEqual(len(merge_detections(candidates, 0.6)), 3)

    def test_predictions_within_one_view_are_not_suppressed_again(self):
        candidates = [detection(self.mask), detection(self.mask.copy())]
        self.assertEqual(len(merge_detections(candidates, 0.6)), 2)

    def test_wrong_mask_geometry_is_rejected(self):
        result = result_for([detection(self.mask)])
        with self.assertRaisesRegex(RuntimeError, "遮罩座標"):
            extract_detections(result, "original", (64, 64))

    def test_render_preserves_mask_holes_and_the_original(self):
        image = np.full((64, 64, 3), 100, np.uint8)
        mask = np.zeros((64, 64), bool)
        mask[4:60, 4:60] = True
        mask[20:44, 20:44] = False
        output = render(image, [detection(mask)], {0: "D"}, self.args)
        np.testing.assert_array_equal(output[28:36, 28:36], image[28:36, 28:36])
        self.assertTrue((image == 100).all())
        self.assertFalse(np.array_equal(output[10, 10], image[10, 10]))

    def test_unicode_io_all_classes_and_empty_predictions(self):
        calls = []
        results = [[], [detection(self.mask, "gamma")], [detection(self.mask, "clahe", 0.9)]]

        def predict(**kwargs):
            calls.append(kwargs)
            return [result_for(results[len(calls) - 1])]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "陰影測試.png"
            image = np.full((32, 64, 3), 60, np.uint8)
            write_image(source, image)
            self.assertEqual(collect_images([root, source]), [source])
            output = root / "result"
            model = SimpleNamespace(names={0: "D", 1: "R"}, predict=predict)
            self.args.save_views = True
            report = predict_image(model, source, output, self.args)
            self.assertEqual(report["view_counts"], {"original": 0, "gamma": 1, "clahe": 1})
            self.assertEqual(report["merged_count"], 1)
            self.assertEqual(report["detections"][0]["view"], "clahe")
            self.assertEqual(len(list(output.glob("*.jpg"))), 4)  # original + all + two classes
            self.assertEqual(len(list((output / "views").glob("*"))), 4)
            saved = json.loads((output / "predictions.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, report)
            for call in calls:
                self.assertTrue(call["retina_masks"])
                self.assertFalse(call["agnostic_nms"])
                self.assertEqual(call["source"].shape, image.shape)
            calls.clear()
            results[:] = [[], [], []]
            report = predict_image(model, source, root / "empty", self.args)
            self.assertEqual(report["merged_count"], 0)
            self.assertTrue((root / "empty" / "陰影測試__all.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
