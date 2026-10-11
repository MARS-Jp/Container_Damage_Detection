"""Offline batch-runner checks: python -m unittest discover -s tests -p test_predict_parameter_sweep.py -v."""

from __future__ import annotations

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools import predict_parameter_sweep as sweep


class ParameterSweepTests(unittest.TestCase):
    def test_exclusion_configuration_is_snapshotted_for_the_whole_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight, source = root / "model.pt", root / "image.jpg"
            weight.touch()
            source.touch()
            config = root / "regions.json"
            config.write_text('{"regions": []}', encoding="utf-8")
            snapshots = []

            def run_job(job):
                snapshot = Path(job["exclude_regions"])
                snapshots.append(snapshot)
                self.assertNotEqual(snapshot, config)
                self.assertEqual(json.loads(snapshot.read_text(encoding="utf-8")), {"regions": []})
                self.assertIn(str(snapshot), job["command"])
                config.write_text('{"regions": ["changed while running"]}', encoding="utf-8")
                return 0, 0.1

            with patch.object(sweep, "run_job", side_effect=run_job), redirect_stdout(io.StringIO()):
                code = sweep.main([
                    "--suite", "all", "--versions", "baseline", "--weights", str(weight),
                    "--source", str(source), "--output", str(root / "output"), "--retina-masks",
                    "--exclude-regions", str(config),
                ])
            self.assertEqual(code, 0)
            self.assertEqual(len(snapshots), 3)
            self.assertEqual(len(set(snapshots)), 1)

    def test_three_groups_share_fill_only_rendering_and_isolate_outputs(self):
        args = sweep.parse_args([
            "--suite", "all", "--weights", "m.pt", "--source", "i.jpg", "--retina-masks",
            "--imgsz", "640", "--conf", "0.15", "--iou", "0.7", "--compare",
        ])
        jobs = sweep.build_jobs(args, Path("output"))
        self.assertEqual(len(jobs), 25)
        self.assertEqual(len({job["output"] for job in jobs}), 25)
        self.assertEqual({job["group"] for job in jobs}, {"shadow", "general", "combined"})
        for job in jobs:
            self.assertEqual(Path(job["command"][1]).name, "predict_shadow.py")
            self.assertNotIn("--show-scores", job["command"])
            self.assertEqual(job["alpha"], 0.35)
            if job["group"] == "general" or job["version"] == "baseline":
                self.assertEqual(job["views"], ["original"])
            if job["group"] == "shadow":
                self.assertEqual((job["imgsz"], job["conf"], job["iou"]), (640, 0.15, 0.7))
            command = job["command"]
            self.assertEqual(command[command.index("--output") + 1], job["output"])

    def test_group_selection_and_actual_shadow_arguments(self):
        from tools.predict_shadow import parse_args as parse_shadow_args

        for suite in ("shadow", "general", "combined"):
            with self.subTest(suite=suite):
                args = sweep.parse_args(["--suite", suite, "--weights", "m.pt", "--source", "i.jpg", "--retina-masks"])
                jobs = sweep.build_jobs(args, Path("output"))
                for job in jobs:
                    child_args = parse_shadow_args(job["command"][2:])
                    for key in ("imgsz", "conf", "iou", "views", "gamma", "clahe_clip", "shadow_threshold", "merge_iou"):
                        self.assertEqual(getattr(child_args, key), job[key])

    def test_focused_suite_passes_class_thresholds_and_metrics(self):
        args = sweep.parse_args(["--suite", "focused", "--compare", "--weights", "m.pt", "--source", "i.jpg"])
        jobs = sweep.build_jobs(args, Path("output"))
        self.assertEqual(len(jobs), 10)
        self.assertEqual(len({job["output"] for job in jobs}), 10)
        selected = next(job for job in jobs if job["version"] == "d008_r020")
        self.assertEqual(selected["class_conf"], {"D": 0.08, "R": 0.20})
        self.assertIn("D=0.08", selected["command"])
        self.assertIn("R=0.2", selected["command"])
        self.assertTrue(all("--save-metrics" in job["command"] for job in jobs))

    def test_versions_change_only_one_baseline_parameter(self):
        args = sweep.parse_args([
            "--weights", "model.pt", "--source", "image.jpg", "--imgsz", "640", "--conf", "0.15",
            "--iou", "0.7", "--device", "cpu", "--retina-masks",
        ])
        jobs = sweep.build_jobs(args, Path("output"))
        self.assertEqual(len(jobs), 8)
        self.assertEqual(len({job["output"] for job in jobs}), 8)
        for job in jobs:
            changed = {key for key in sweep.PARAMETER_KEYS if job[key] != jobs[0][key]}
            self.assertEqual(len(changed), 0 if job["version"] == "baseline" else 1)
            command = job["command"]
            for key in ("imgsz", "conf", "iou", "device"):
                self.assertEqual(command[command.index(f"--{key}") + 1], str(job[key]))
            self.assertIn("--hide-labels", command)
            self.assertIn("--retina-masks", command)
            self.assertNotIn("--show-boxes", command)
            self.assertNotIn("--no-fill", command)

    def test_same_model_and_source_names_cannot_overwrite_each_other(self):
        args = sweep.parse_args([
            "--weights", "first/model.pt", "second/model.pt", "--source", "first/image.jpg", "second/image.jpg",
            "--versions", "baseline", "--no-retina-masks",
        ])
        jobs = sweep.build_jobs(args, Path("output"))
        self.assertEqual(len(jobs), 4)
        self.assertEqual(len({job["output"] for job in jobs}), 4)
        self.assertTrue(all("--retina-masks" not in job["command"] for job in jobs))

    def test_dry_run_does_not_create_output_or_launch_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dry_run"
            with patch.object(sweep, "run_job") as run_job, redirect_stdout(io.StringIO()):
                code = sweep.main([
                    "--weights", "missing.pt", "--source", "missing.jpg", "--output", str(output), "--dry-run",
                ])
            self.assertEqual(code, 0)
            self.assertFalse(output.exists())
            run_job.assert_not_called()

    def test_compare_option_runs_comparison_after_successful_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.pt").touch()
            (root / "image.jpg").touch()
            compare = Mock()
            module = SimpleNamespace(create_comparisons=compare)
            with patch.dict(sys.modules, {"tools.compare_parameter_sweep": module}), \
                    patch.object(sweep, "run_job", return_value=(0, 0.1)), redirect_stdout(io.StringIO()):
                code = sweep.main([
                    "--weights", str(root / "model.pt"), "--source", str(root / "image.jpg"),
                    "--output", str(root / "results"), "--versions", "baseline", "--compare",
                ])
            self.assertEqual(code, 0)
            compare.assert_called_once()
            run_dir = compare.call_args.args[0]
            self.assertTrue((run_dir / "summary.csv").is_file())

    def test_execution_logs_errors_continues_and_creates_a_new_run_each_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weight, source, script = root / "model.pt", root / "image.jpg", root / "fake_predict.py"
            weight.touch()
            source.touch()
            script.write_text(
                "import sys\nprint('inference output', flush=True)\n"
                "size = sys.argv[sys.argv.index('--imgsz') + 1]\n"
                "sys.exit(7 if size == '1280' else 0)\n", encoding="utf-8",
            )
            original_build_command = sweep.BASE["build_command"]

            def fake_command(*args):
                command = original_build_command(*args)
                command[1] = str(script)
                return command

            argv = [
                "--weights", str(weight), "--source", str(source), "--output", str(root / "results"),
                "--versions", "imgsz960", "imgsz1280", "baseline", "--imgsz", "640",
            ]
            with patch.dict(sweep.BASE, {"build_command": fake_command}), redirect_stdout(io.StringIO()):
                self.assertEqual(sweep.main(argv), 1)
                self.assertEqual(sweep.main(argv), 1)
            runs = list((root / "results").iterdir())
            self.assertEqual(len(runs), 2)
            for run in runs:
                with (run / "summary.csv").open(encoding="utf-8-sig", newline="") as summary:
                    rows = list(csv.DictReader(summary))
                self.assertEqual([row["status"] for row in rows], ["ok", "failed", "ok"])
                self.assertEqual([row["returncode"] for row in rows], ["0", "7", "0"])
                self.assertEqual(len(json.loads((run / "parameters.json").read_text(encoding="utf-8"))), 3)
                for row in rows:
                    self.assertIn("inference output", Path(row["log"]).read_text(encoding="utf-8"))
                    self.assertTrue((Path(row["output"]) / "parameters.json").is_file())


if __name__ == "__main__":
    unittest.main()
