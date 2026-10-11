"""Compare the baseline, selected parameters, and focused shadow alternatives."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_parameter_sweep import draw_figure, font_manager, plt
from tools.predict_shadow_recovery import REGIONS
from tools.tune_shadow_parameters import performance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-run", type=Path, required=True)
    parser.add_argument("--focused-run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.selected_run / "comparisons"
    output.mkdir(parents=True, exist_ok=True)
    provenance = [
        json.loads((r / "input_provenance.json").read_text(encoding="utf-8"))
        for r in (args.selected_run, args.focused_run)
    ]
    for key in ("model_sha256", "image_sha256", "xml_sha256", "regions", "coordinate_origin"):
        if provenance[0][key] != provenance[1][key]:
            raise ValueError(f"Incompatible comparison inputs: {key}")
    if hashlib.sha256(Path(provenance[0]["source"]).read_bytes()).hexdigest() != provenance[0]["image_sha256"]:
        raise ValueError("Source image changed since inference")
    records, scores = [], []
    for run, folder_name, title in (
        (args.selected_run, "Q01_baseline", "原圖基準"),
        (args.selected_run, "Q04_balanced", "保留版：溫和提亮＋分類合併"),
        (args.focused_run, "Q04_balanced", "強提亮／集中裁切：分類合併"),
        (args.focused_run, "Q05_recall", "強提亮／集中裁切：召回診斷"),
    ):
        folder = run / folder_name
        prediction = json.loads((folder / "predictions.json").read_text(encoding="utf-8"))
        record = {
            "name": title,
            "title": title,
            "source": prediction["source"],
            "image": next(folder.glob("*__all.jpg")),
            "evaluation": json.loads((folder / "evaluation.json").read_text(encoding="utf-8")),
        }
        score = performance(record)
        record["caption"] = (
            f"{title}\nIoU0.50：D {score['D_tp']}/25，R {score['R_tp']}/69；未匹配 {score['D_fp']}/{score['R_fp']}"
        )
        records.append(record)
        scores.append({"version": title, "folder": str(folder.resolve()), **score})
    font = Path("C:/Windows/Fonts/msjh.ttc")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    for filename, crop, title in (
        ("final_all.png", None, ""),
        ("final_left.png", REGIONS["left"], "左側暗部"),
        ("final_right.png", REGIONS["right"], "右下陰影"),
    ):
        draw_figure(records, output / filename, "all", crop, title)
    with (output / "final_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    print(f"Saved three consolidated figures and summary: {output.resolve()}")


if __name__ == "__main__":
    main()
