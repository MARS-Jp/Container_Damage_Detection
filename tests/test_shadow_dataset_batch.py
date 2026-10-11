"""Verify dataset discovery, preview, and failure handling without GPU inference."""

from contextlib import redirect_stdout
import csv
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools.diagnose_shadow_inference import infer_dataset_split, main


class ShadowDatasetBatchTests(unittest.TestCase):
    def test_split_recognizes_dataset_directories_without_matching_filename_or_substring(self):
        examples = {
            "dataset/train/B/image.JPG": "train",
            "dataset/val 驗證/X/image.jpg": "val",
            "dataset/VAL驗證/image.jpg": "val",
            "dataset/Validation/images/image.png": "val",
            "dataset/test/image.jpg": "test",
            "dataset/train/val/image.jpg": "val",
            "dataset/valuable/train.jpg": "other",
        }
        for source, expected in examples.items():
            with self.subTest(source=source):
                self.assertEqual(infer_dataset_split(Path(source)), expected)

    def inputs(self, root):
        weight = root / "model.pt"
        weight.touch()
        first, bad, last = root / "a/same.JPG", root / "b/bad.png", root / "b/nested/same.JPG"
        for path in (first, bad, last):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        (root / "b/labels.xml").touch()
        return weight, [first, bad, last]

    def test_directory_preview_recurses_deduplicates_and_does_not_load_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight, sources = self.inputs(root)
            output = root / "output"
            factory, stdout = Mock(), io.StringIO()
            with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=factory)}), redirect_stdout(stdout):
                code = main(["--weights", str(weight), "--source", str(root / "a"), str(root / "b"),
                             str(root / "b/nested"), "--versions", "S08_diagnostic_only", "--dry-run", "--output", str(output)])
            self.assertEqual(code, 0)
            factory.assert_not_called()
            self.assertFalse(output.exists())
            self.assertIn("images: 3", stdout.getvalue())
            self.assertIn("total model calls: 3", stdout.getvalue())
            self.assertNotIn("labels.xml", stdout.getvalue())

    def test_corrupt_image_can_be_recorded_without_stopping_remaining_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight, sources = self.inputs(root)
            output = root / "output"
            factory = Mock(return_value=SimpleNamespace(task="segment"))
            def process(model, source, folder, args):
                if source == sources[1]:
                    raise ValueError("Cannot decode image")
                return [{}], [{"source": str(source), "version": "S08_diagnostic_only", "region": "left",
                               "class": "D", "count": 0, "roi_pixels": 0, "max_conf": 0}]
            with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=factory)}), \
                    patch("tools.diagnose_shadow_inference.process_image", side_effect=process) as processed, \
                    redirect_stdout(io.StringIO()):
                code = main(["--weights", str(weight), "--source", *map(str, sources),
                             "--versions", "S08_diagnostic_only", "--continue-on-error", "--output", str(output)])
            self.assertEqual(code, 1)
            factory.assert_called_once()
            self.assertEqual(processed.call_count, 3)
            run = next(output.iterdir())
            with (run / "image_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([r["status"] for r in rows], ["ok", "failed", "ok"])
            self.assertIn("Cannot decode image", rows[1]["error"])
            with (run / "source_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
                manifest = list(csv.DictReader(handle))
            self.assertEqual(len(manifest), 3)
            self.assertNotEqual(manifest[0]["output_name"], manifest[2]["output_name"])
            with (run / "region_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)

    def test_fail_fast_default_records_failure_before_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight, sources = self.inputs(root)
            output = root / "output"
            factory = Mock(return_value=SimpleNamespace(task="segment"))
            with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=factory)}), \
                    patch("tools.diagnose_shadow_inference.process_image", side_effect=ValueError("Cannot decode image")) as processed, \
                    redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                main(["--weights", str(weight), "--source", *map(str, sources),
                      "--versions", "S08_diagnostic_only", "--output", str(output)])
            self.assertEqual(processed.call_count, 1)
            run = next(output.iterdir())
            with (run / "image_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
                self.assertEqual([row["status"] for row in csv.DictReader(handle)], ["failed"])

    def test_train_and_val_same_names_write_to_separate_folders_and_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight = root / "model.pt"
            weight.touch()
            sources = [root / "dataset/train/B/same.JPG", root / "dataset/val 驗證/X/same.JPG"]
            for source in sources:
                source.parent.mkdir(parents=True)
                source.touch()
            output = root / "output"
            factory = Mock(return_value=SimpleNamespace(task="segment"))

            def process(model, source, folder, args):
                folder.mkdir(parents=True)
                (folder / "result.jpg").write_bytes(b"inference result")
                return [{}], [{"source": str(source), "version": "S08_diagnostic_only", "region": "left",
                               "class": "D", "count": 0, "roi_pixels": 0, "max_conf": 0}]

            with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=factory)}), \
                    patch("tools.diagnose_shadow_inference.process_image", side_effect=process), \
                    redirect_stdout(io.StringIO()):
                code = main(["--weights", str(weight), "--source", str(root / "dataset"),
                             "--versions", "S08_diagnostic_only", "--output", str(output)])
            self.assertEqual(code, 0)
            factory.assert_called_once()
            run = next(output.iterdir())
            self.assertTrue((run / "01_model/train/01_same/result.jpg").is_file())
            self.assertTrue((run / "01_model/val/02_same/result.jpg").is_file())
            with (run / "source_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["dataset_split"] for row in rows], ["train", "val"])
            self.assertEqual([row["output_name"] for row in rows], ["train/01_same", "val/02_same"])
            for filename in ("image_summary.csv", "region_summary.csv"):
                with (run / filename).open(encoding="utf-8-sig", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual([row["dataset_split"] for row in rows], ["train", "val"])
            with (run / "image_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
                self.assertTrue(all(Path(row["output"]).is_dir() for row in csv.DictReader(handle)))


if __name__ == "__main__":
    unittest.main()
