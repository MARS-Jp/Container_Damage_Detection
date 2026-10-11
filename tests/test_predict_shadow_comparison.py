"""Ensure comparisons use per-image identity and exclude failed jobs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from tools.compare_shadow_sweep import read_records
from tools.predict_parameter_sweep import SHADOW_DEFAULTS


class ShadowComparisonTests(unittest.TestCase):
    def test_directory_sources_and_duplicate_filenames_are_not_mixed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs, status = [], []
            for index, state in enumerate(("ok", "ok", "failed")):
                output = root / f"job{index}"
                output.mkdir()
                job = {
                    "group": "shadow", "version": f"v{index}", "weight": str(root / "model.pt"),
                    "source": str(root / "image_directory"), "output": str(output),
                    "imgsz": 640, "conf": 0.15, "iou": 0.7, **SHADOW_DEFAULTS,
                }
                jobs.append(job)
                status.append({"output": str(output), "status": state, "seconds": "1.0"})
                actual_source = root / f"different_source{index}" / "same_name.jpg"
                (output / "same_name__all.jpg").touch()
                (output / "predictions.json").write_text(json.dumps({
                    "source": str(actual_source), "merged_count": 0, "view_counts": {"original": 0},
                    "metrics": {"classes": {}},
                }), encoding="utf-8")
            (root / "parameters.json").write_text(json.dumps(jobs), encoding="utf-8")
            with (root / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("output", "status", "seconds"))
                writer.writeheader()
                writer.writerows(status)
            records = read_records(root)
            self.assertEqual(len(records), 2)
            self.assertNotEqual(records[0]["key"], records[1]["key"])
            self.assertEqual({Path(r["source"]).name for r in records}, {"same_name.jpg"})
            self.assertTrue(all(r["source"] != r["job"]["source"] for r in records))
            self.assertEqual({r["job"]["version"] for r in records}, {"v0", "v1"})


if __name__ == "__main__":
    unittest.main()
