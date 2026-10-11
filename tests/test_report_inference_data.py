"""Check split isolation, matching, pooled metrics, and missing-label handling."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from tools.report_inference_data import AnnotationIndex, average_precision, load_records, main, match_predictions, summarize


class InferenceDataReportTests(unittest.TestCase):
    def result(self, root, split, name, predictions, boxes=None, shape=(100, 100), union_pixels=500):
        results, labels = root / "results/01_model", root / "labels"
        folder = results / split / name / "S08_diagnostic_only"
        folder.mkdir(parents=True)
        source = root / "dataset" / split / "B" / f"{name}.JPG"
        width, height = shape
        metrics = {"source": str(source), "width": width, "height": height,
                   "classes": {"2": {"name": "D", "count": len(predictions), "union_pixels": union_pixels if predictions else 0}}}
        (folder / "prediction_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (folder / "predictions.json").write_text(json.dumps({"detections": predictions}), encoding="utf-8")
        if boxes is not None:
            label = labels / split / "B" / f"{name}.xml"
            label.parent.mkdir(parents=True, exist_ok=True)
            objects = "".join(f"<object><name>D</name><difficult>0</difficult><bndbox><xmin>{box[0]}</xmin><ymin>{box[1]}</ymin><xmax>{box[2]}</xmax><ymax>{box[3]}</ymax></bndbox></object>" for box in boxes)
            label.write_text(f"<annotation><filename>{name}.JPG</filename><size><width>{width}</width><height>{height}</height></size>{objects}</annotation>", encoding="utf-8")
        return results, labels, folder

    def predictions(self, *entries):
        return [{"class": name, "score": score, "box_xyxy": box} for name, score, box in entries]

    def test_one_to_one_matching_uses_class_and_score(self):
        truth = [{"class": "D", "box": [10, 10, 30, 30]}]
        predictions = [{"class": "D", "score": .8, "box": [10, 10, 30, 30]},
                       {"class": "R", "score": .95, "box": [10, 10, 30, 30]},
                       {"class": "D", "score": .9, "box": [10, 10, 30, 30]}]
        self.assertEqual(match_predictions(predictions, truth, .5), [False, False, True])

    def test_ap_handles_perfect_missed_and_no_gt_classes(self):
        self.assertEqual(average_precision([(.9, True)], 1), 1)
        self.assertEqual(average_precision([(.95, False), (.9, True)], 1), .5)
        self.assertEqual(average_precision([], 1), 0)
        self.assertIsNone(average_precision([(.9, False)], 0))

    def test_combined_recomputes_ratios_global_ap_and_pixel_weighting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            box = [10, 10, 30, 30]
            results, labels, _ = self.result(root, "train", "train_image", self.predictions(("D", .9, box), ("D", .8, box)), [box])
            self.result(root, "val", "val_image", self.predictions(("D", .7, box)), [box, [40, 40, 60, 60]], (200, 200), 1000)
            records, issues = load_records(results, AnnotationIndex([labels]), origin=0)
            self.assertFalse(issues)
            train = summarize([record for record in records if record["split"] == "train"], "train")
            val = summarize([record for record in records if record["split"] == "val"], "val")
            combined = summarize(records, "train_and_val")
            summary = combined["summary"]
            self.assertEqual((summary["tp"], summary["fp"], summary["fn"]), (2, 1, 1))
            self.assertAlmostEqual(summary["precision"], 2 / 3)
            self.assertAlmostEqual(summary["recall"], 2 / 3)
            self.assertNotAlmostEqual(summary["precision"], (train["summary"]["precision"] + val["summary"]["precision"]) / 2)
            self.assertNotAlmostEqual(summary["box_map50"], (train["summary"]["box_map50"] + val["summary"]["box_map50"]) / 2)
            self.assertAlmostEqual(combined["classes"][0]["weighted_coverage_percent"], 3)

    def test_missing_label_is_excluded_not_assumed_background_and_empty_xml_is_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root, box = Path(directory), [10, 10, 30, 30]
            results, labels, _ = self.result(root, "train", "missing", self.predictions(("D", .9, box)))
            self.result(root, "train", "background", [], [])
            records, issues = load_records(results, AnnotationIndex([labels]), origin=0)
            report = summarize(records, "train")
            self.assertEqual(report["summary"]["image_count"], 2)
            self.assertEqual(report["summary"]["evaluated_images"], 1)
            self.assertEqual(report["summary"]["excluded_from_evaluation"], 1)
            self.assertEqual(report["classes"][0]["detections"], 1)
            self.assertEqual(report["classes"][0]["fp"], 0)
            self.assertIsNone(report["classes"][0]["box_ap50"])
            self.assertEqual(issues[0]["status"], "missing_annotation")

    def test_annotation_dimensions_and_filename_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results, labels, _ = self.result(root, "val", "wrong_size", [], [], (100, 100))
            label = labels / "val/B/wrong_size.xml"
            label.write_text(label.read_text().replace("<width>100</width>", "<width>101</width>"), encoding="utf-8")
            records, issues = load_records(results, AnnotationIndex([labels]), origin=0)
            self.assertEqual(records[0]["evaluation_status"], "invalid_annotation")
            self.assertIn("dimensions", issues[0]["error"])

    def test_same_filename_uses_split_and_category_then_rejects_ambiguous_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "labels"
            for split, category in (("train", "B"), ("train", "R"), ("val", "B")):
                path = labels / split / category / "same.xml"
                path.parent.mkdir(parents=True)
                path.write_text("<annotation><filename>same.JPG</filename><size><width>100</width><height>100</height></size></annotation>", encoding="utf-8")
            index = AnnotationIndex([labels])
            status, path, _, _ = index.resolve(root / "dataset/train/B/same.JPG", "train", 100, 100, 1)
            self.assertEqual(status, "ok")
            self.assertEqual(path.parent.name, "B")
            self.assertEqual(index.resolve(root / "dataset/train/C/same.JPG", "train", 100, 100, 1)[0], "ambiguous_annotation")
            self.assertEqual(index.resolve(root / "dataset/val/B/same.JPG", "val", 100, 100, 1)[0], "ok")

    def test_command_generates_three_reports_and_preserves_input_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            box = [10, 10, 30, 30]
            results, labels, folder = self.result(root, "train", "sample", self.predictions(("D", .9, box)), [box])
            before = (folder / "prediction_metrics.json").read_bytes()
            output = root / "reports"
            with redirect_stdout(io.StringIO()):
                code = main(["--results", str(results), "--annotations", str(labels), "--xml-origin", "0", "--output", str(output)])
            self.assertEqual(code, 0)
            report_root = next(output.iterdir())
            self.assertTrue((report_root / "index.html").is_file())
            for split in ("train", "val", "train_and_val"):
                self.assertTrue((report_root / split / "report.html").is_file())
                self.assertTrue((report_root / split / "class_summary.csv").is_file())
            with (report_root / "train/summary.csv").open(encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))
                self.assertEqual(row["image_count"], "1")
                self.assertEqual(row["evaluated_images"], "1")
            self.assertEqual((folder / "prediction_metrics.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
