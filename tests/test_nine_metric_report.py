"""Check all requested columns, missing records, and real-mask evaluation."""

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from tools.pixel_metrics import METRIC_COLUMNS
from tools.report_nine_metrics import evaluate_images, load_mask_archive, main, metric_context


class NineMetricReportTests(unittest.TestCase):
    def fixture(self, root, name, has_annotation):
        results, labels = root / "results/01_model", root / "labels"
        source = root / f"dataset/train/D/{name}.JPG"
        folder = results / "train" / name / "S08_diagnostic_only"
        folder.mkdir(parents=True)
        metrics = {
            "source": str(source),
            "width": 4,
            "height": 2,
            "classes": {"2": {"name": "D", "count": 1, "union_pixels": 4}},
        }
        (folder / "prediction_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (folder / "predictions.json").write_text(
            json.dumps({"detections": [{"class": "D", "score": 0.9, "box_xyxy": [0, 0, 2, 2]}]}), encoding="utf-8"
        )
        if has_annotation:
            label = labels / "train/D" / f"{name}.xml"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text(
                f"<annotation><filename>{name}.JPG</filename><size><width>4</width><height>2</height></size><object><name>D</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>2</xmax><ymax>2</ymax></bndbox></object></annotation>",
                encoding="utf-8",
            )
        return results, labels, folder

    def test_every_source_and_class_row_has_nine_columns_even_without_xml(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, labels, folder = self.fixture(root, "matched", True)
            self.fixture(root, "missing", False)
            before = (folder / "predictions.json").read_bytes()
            with redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "--mode",
                        "boxes",
                        "--results",
                        str(results),
                        "--annotations",
                        str(labels),
                        "--xml-origin",
                        "0",
                        "--output",
                        str(root / "reports"),
                    ]
                )
            self.assertEqual(code, 0)
            output = next((root / "reports").iterdir())
            for group, expected in (("train", 2), ("val", 0), ("train_and_val", 2)):
                for filename in ("image_metrics.csv", "image_class_metrics.csv"):
                    with (output / group / filename).open(encoding="utf-8-sig", newline="") as handle:
                        reader = csv.DictReader(handle)
                        self.assertTrue(set(METRIC_COLUMNS).issubset(reader.fieldnames))
                        rows = list(reader)
                    self.assertEqual(len(rows), expected)
                    if rows:
                        matched = next(row for row in rows if row["image_name"] == "matched.JPG")
                        missing = next(row for row in rows if row["image_name"] == "missing.JPG")
                        self.assertEqual(matched["IoU"], "1.0")
                        self.assertEqual(matched["Accuracy"], "1.0")
                        if filename == "image_class_metrics.csv":
                            self.assertEqual(matched["mIoU"], "NA")
                            self.assertEqual(matched["Mean Dice"], "NA")
                            self.assertIn("Not applicable to a single class", matched["NA_reason"])
                        self.assertTrue(all(missing[field] == "NA" for field in METRIC_COLUMNS))
                        self.assertIn("missing_annotation", missing["pixel_reason"])
                        self.assertIn("missing_annotation", missing["NA_reason"])
            self.assertEqual((folder / "predictions.json").read_bytes(), before)
            self.assertIn("框區域像素評估", (output / "index.html").read_text(encoding="utf-8"))

    def test_background_baseline_exposes_high_accuracy_without_detections(self):
        values = {"TP": 0, "FP": 0, "FN": 1, "TN": 99, "Accuracy": 0.99}
        context = metric_context(values, "image_class", 1, 0)
        self.assertEqual(context["always_background_accuracy"], 0.99)
        self.assertEqual(context["accuracy_gain_over_background"], 0)
        self.assertEqual(context["gt_positive_pixels"], 1)
        self.assertEqual(context["predicted_positive_pixels"], 0)

    def test_true_masks_use_pixels_not_enclosing_boxes_and_apply_score_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth = np.array([[[1, 1, 0, 0], [0, 0, 0, 0]]], dtype=bool)
            prediction = np.array([[[1, 0, 1, 0], [0, 0, 0, 0]]], dtype=bool)
            ground = root / "masks/train/D/sample.npz"
            ground.parent.mkdir(parents=True)
            np.savez_compressed(ground, masks=truth, class_names=np.array(["D"]))
            folder = root / "predictions"
            folder.mkdir()
            np.savez_compressed(
                folder / "prediction_masks.npz",
                packed_masks=np.packbits(prediction, axis=-1),
                width=4,
                height=2,
                class_names=np.array(["D"]),
                scores=np.array([0.8]),
            )
            record = {
                "classes": {"D": {"id": 2}},
                "width": 4,
                "height": 2,
                "source": str(root / "dataset/train/D/sample.JPG"),
                "split": "train",
                "metrics_file": str(folder / "prediction_metrics.json"),
                "annotation": "",
            }
            evaluate_images([record], "masks", [root / "masks"])
            self.assertEqual(record["pixel_status"], "ok")
            self.assertEqual(record["pixel_counts"]["D"], {"TP": 1, "FP": 1, "FN": 1, "TN": 5})
            self.assertAlmostEqual(record["pixel_overall"]["Dice"], 0.5)
            self.assertAlmostEqual(record["pixel_overall"]["IoU"], 1 / 3)
            filtered = load_mask_archive(folder / "prediction_masks.npz", ["D"], 2, 4, 0.9)
            self.assertFalse(filtered["D"].any())

    def test_wrong_mask_dimensions_rejected_and_pickle_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.npz"
            np.savez_compressed(path, masks=np.zeros((1, 2, 2), dtype=bool), class_names=np.array(["D"]))
            with self.assertRaises(ValueError):
                load_mask_archive(path, ["D"], 2, 4)
            np.savez_compressed(path, masks=np.array([{"not": "a mask"}], dtype=object), class_names=np.array(["D"]))
            with self.assertRaises(ValueError):
                load_mask_archive(path, ["D"], 2, 4)


if __name__ == "__main__":
    unittest.main()
