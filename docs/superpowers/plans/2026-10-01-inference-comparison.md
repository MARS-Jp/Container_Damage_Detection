# Inference Output and Three-group Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 先產生只有分割塗色的圖片，再自動執行陰影、一般、混合三組推論參數比較。

**Architecture:** 保留 predict_shadow.py 的推論與遮罩融合，僅簡化 render。延伸既有 predict_parameter_sweep.py，三組統一呼叫陰影程式，一般組只啟用 original 視圖。獨立的 compare_shadow_sweep.py 讀取已完成工作與 predictions.json，產生同來源、同模型的比較圖。

**Tech Stack:** Python 3、argparse、subprocess、unittest、OpenCV、NumPy、PyTorch、Ultralytics、Pillow、Matplotlib；Windows PowerShell。

**Spec:** docs/superpowers/plans/2026-10-01-inference-comparison-spec.md

## Global Constraints

- 先完成純塗色輸出，實際產生圖片後才執行三組參數比較。
- 純塗色修改維持模型、来源、推論參數、遮罩內容、類別顏色與 alpha=0.35 不變。
- 图片內不顯示信心分數、D/R 文字、偵測框或輪廓線。
- 三组使用 predict-2.py 中已啟用的模型、圖片與一般推論基準。
- 使用現有 Windows/PowerShell、yolov11 Python 環境與本地 GPU，不增加依賴。
- 保留工作區既有修改；本次不建立提交。

---

### Task 1: 先輸出純塗色圖片

**Files:**

- Modify: tools/predict_shadow.py（parse_args、render、使用說明）
- Verify: tests/test_predict_shadow.py（現有遮罩、IO、空預測與融合測試）

**Interfaces:**

- Consumes: `render(image: np.ndarray, detections: list[Detection], names: dict, args: argparse.Namespace) -> np.ndarray`。
- Produces: 同簽名函數；只對 `detection.mask` 為真的像素做原有 alpha 混色。

- [x] **Step 1: 維持推論邏輯，移除 render 的輪廓與文字**

```python
output = image.copy()
for detection in detections:
    color = class_color(int(detection.box[5]))
    mask = detection.mask
    output[mask] = np.rint(output[mask] * (1 - args.alpha) + np.array(color) * args.alpha).astype(np.uint8)
return output
```

移除 `--show-scores` 選項及使用範例；所有 `render` 呼叫繼續使用相同函數。

- [x] **Step 2: 執行現有測試並以原設定推論**

```powershell
& 'C:\Users\USER\anaconda3\envs\yolov11\python.exe' -m unittest discover -s tests -p test_predict_shadow.py -v
& 'C:\Users\USER\anaconda3\envs\yolov11\python.exe' tools/predict_shadow.py
```

預期：8 個測試通過；新增 run 資料夾，原有 JSON 偵測資料仍保留。打開 `*__all.jpg`，確認只有塗色。

### Task 2: 三組參數的批次執行

**Files:**

- Modify: tools/predict_parameter_sweep.py
- Modify: tests/test_predict_parameter_sweep.py
- Create: docs/inference-three-group-comparison-1001.md

**Interfaces:**

- Consumes: `parse_args(argv: list[str] | None) -> argparse.Namespace`、`build_jobs(args: argparse.Namespace, run_dir: Path) -> list[dict]`、`run_job(job: dict) -> tuple[int, float]`。
- Produces: jobs 新增 `group`；保留 `version/weight/source/output/log/command` 與實際參數。`--suite all --compare` 一次跑三組。

- [x] **Step 1: 加入測試，確認 25 個工作與組別隔離**

```python
args = sweep.parse_args(["--suite", "all", "--weights", "m.pt", "--source", "i.jpg"])
jobs = sweep.build_jobs(args, Path("output"))
self.assertEqual(len(jobs), 25)
self.assertEqual(len({job["output"] for job in jobs}), 25)
self.assertEqual({job["group"] for job in jobs}, {"shadow", "general", "combined"})
for job in jobs:
    self.assertIn("predict_shadow.py", job["command"][1])
    self.assertNotIn("--show-scores", job["command"])
    if job["group"] == "general" or job["version"] == "baseline":
        self.assertEqual(job["views"], ["original"])
```

- [x] **Step 2: 定義各組 presets 與批次路徑**

```python
GENERAL_TESTS = {name: {**settings, "views": ["original"]} for name, settings in PARAMETER_TESTS.items()}
COMBINED_TESTS = {
    "baseline": {"views": ["original"]},
    "shadow_default": {},
    "conf010": {"conf": 0.10},
    "iou050": {"iou": 0.50},
    "imgsz960": {"imgsz": 960},
    "imgsz1280": {"imgsz": 1280},
    "size960_conf010_iou050": {"imgsz": 960, "conf": 0.10, "iou": 0.50},
    "size960_gamma045_shadow128": {"imgsz": 960, "gamma": 0.45, "shadow_threshold": 128},
}
```

shadow 共 9 版：baseline、shadow_default、gamma_only、clahe_only、gamma045、gamma085、clahe40、shadow128、merge040。一般組 8 版沿用 PARAMETER_TESTS，混合組 8 版。

三組共用推論預設：Gamma 0.65、CLAHE 2.0、亮度門檻 255、合併 IoU 0.60、max_det 300、alpha 0.35；一般組只使用 original。實際命令明確帶入全部參數與 `--save-views`。驗證 retina masks 必須開啟、imgsz 為 32 的倍數、版本屬於選定組別。

輸出：`runs/segment_shadow/parameter_tests/run_時間/01_shadow|02_general|03_combined/版本/模型/來源/`。維持子程式既有時間與圖片子資料夾，不覆寫歷史輸出。失敗記錄至 summary.csv 後繼續。

- [x] **Step 3: 驗證原批次功能及新增分組**

```powershell
& 'C:\Users\USER\anaconda3\envs\yolov11\python.exe' -m unittest discover -s tests -p test_predict_parameter_sweep.py -v
& 'C:\Users\USER\anaconda3\envs\yolov11\python.exe' tools/predict_parameter_sweep.py --suite all --dry-run
```

預期：原 initial/focused 測試仍通過；dry-run 規劃 25 版且不寫檔、不推論。

### Task 3: 圖片比較與實際執行

**Files:**

- Create: tools/compare_shadow_sweep.py
- Modify: tools/predict_shadow.py（predictions.json 補充原始遮罩聯集統計）
- Modify: tests/test_predict_parameter_sweep.py
- Modify: docs/inference-three-group-comparison-1001.md

**Interfaces:**

- Consumes: run_dir 下的 parameters.json、summary.csv、每張圖片的 predictions.json 與 `*__all.jpg`。
- Produces: `create_comparisons(run_dir: Path) -> list[Path]`；輸出組內全圖/D/R 比較、細部比較、跨組總覽與 comparison_summary.csv。

- [x] **Step 1: 用已成功工作的實際來源建立比較資料**

```python
successful = {row["output"] for row in csv.DictReader(handle) if row["status"] == "ok"}
# 每個 predictions.json 的 source 是展開後的原始圖片路徑，避免資料夾來源混淆。
key = (str(Path(job["weight"]).resolve()).casefold(), str(Path(report["source"]).resolve()).casefold())
```

從二值遮罩計算同類 union_pixels 與 coverage_percent，不由 JPG 顏色估計。沿用 compare_parameter_sweep.draw_figure 的同位置裁切，caption 明確列出組別、版本、imgsz/conf/iou、視圖與增強參數。

- [x] **Step 2: 執行完整批次與檢查輸出**

```powershell
& 'C:\Users\USER\anaconda3\envs\yolov11\python.exe' tools/predict_parameter_sweep.py --suite all --compare
```

預期：25/25 成功；三個組別各自保存結果；比較圖包含原圖與每版純塗色結果。打開代表性比較圖，確認無圖片內文字與框線，且 caption 不遮擋照片。

- [x] **Step 3: 交付可重跑命令、結果位置與驗證紀錄**

使用說明列出三組版本表、實際輸出位置、`--versions` 選版方式與 `--dry-run`。檢查 git diff 僅包含本次修改與保留的既有變更；不建立提交。

## Self-review

- 規格 1–2 對應 Task 1；3–5 對應 Task 2；6–7 對應 Task 3；8 納入 Global Constraints。
- 已明確定義 preset、函數介面、命令、輸出與失敗行為。
- 依使用者「先做輸出，再做參數」的授權，在當前工作階段依序實作，不另開工作任務。

### 追加 Task 4: 排除指定圖片右側橘色柱子的 R 誤檢

**Files:**

- Create: tools/shadow_exclusion_regions.json
- Modify: tools/predict_shadow.py
- Modify: tools/predict_parameter_sweep.py
- Test: tests/test_predict_shadow.py、tests/test_predict_parameter_sweep.py

**Interfaces:**

- `load_exclusion_rules(path: Path) -> list[dict]` 讀取 source/classes/polygon_normalized。
- `exclusion_masks(source: Path, shape: tuple[int, int], names: dict, rules: list[dict]) -> tuple[dict, list]` 只匹配完整圖片路徑。
- `exclude_background(detections: list[Detection], masks: dict[int, np.ndarray]) -> list[Detection]` 在融合前移除指定類別的排除區像素。

- [x] **Step 1: 指定實際圖片與橘色柱子範圍**

正規化多邊形：`[[0.845, 0], [1, 0], [1, 1], [0.880, 1], [0.870, 0.600], [0.858, 0.250]]`，類別 `R`，只匹配目前 CBHU0708054-A.JPG 的完整路徑。

- [x] **Step 2: 實作遮罩裁除與保留排除紀錄**

```python
mask = detection.mask & ~excluded
if mask.any():
    kept.append(Detection(detection.box.copy(), mask, detection.view))
```

無相交像素時保留原物件；排除區外像素、其他類別與其他來源圖片不受影響。predictions.json 記錄 raw_view_counts、view_counts、excluded_regions；提供 `--no-exclusions`。

- [x] **Step 3: 相同參數重新推論並確認修正**

執行 Task 1 的同一個推論命令。結果：總數 65 → 61，右側柱子 4 個誤檢移除；已目視確認修正版。所有批次工作使用 run 內的 exclusion_regions.json 快照。

- [x] **Step 4: 執行新增邊界與設定快照測試**

```powershell
& 'C:\Users\USER\anaconda3\envs\yolov11\python.exe' -m unittest discover -s tests -p 'test_predict_*.py' -v
```

驗證跨界 R 遮罩保留區外像素、D 不變、同名不同目錄圖片不匹配、修改原始設定後整批仍使用同一份快照。

## 實作驗收紀錄

- 純塗色第一次輸出：`runs/segment_shadow/run_20261001_004129_540900/`。
- 柱子排除修正版：`runs/segment_shadow/run_20261001_004913_595036/`，已目視確認。
- 三組完整執行：`runs/segment_shadow/parameter_tests/run_20261001_005102_586762/`。
- 陰影 9、一般 8、混合 8，25/25 成功，工作累計 231.7 秒（包含各次載入與存檔）。
- 17 張比較 PNG 均可完整解碼；組內比較與跨組總覽已目視確認，照片內沒有新增文字或框線。
- 25 份預測報告都套用同一份排除設定；三組 baseline 的預測與 JPG 檔案雜湊完全相同。
- 23 項相關 unittest 全部通過；保留工作區其他既有變更，未建立提交。

## 後續漏檢與 XML 核對紀錄

- [x] 使用者指定左 MSC 凹陷、右下陰影後，新增 `tools/predict_shadow_recall.py`，沿用既有模型與純塗色，執行 33 次全圖／局部推論產生 6 個診斷版本。
- [x] 讀取使用者 XML，確認 800×600、D 25／R 69，共 94 個框，沒有分割輪廓。
- [x] 新增 `tools/evaluate_shadow_annotations.py`，同類別按分數排序一對一匹配，門檻 IoU 0.5，紀錄逐標記漏檢及區域統計。
- [x] 完成 31 版標記比對。MSC D 嚴格框匹配從 2/22 至最高 4/22；右下 D 0/2、R 0/11，未解決右側漏檢且新候選有大量未匹配。
- [x] 保留原推論預設；新版本列為診斷，未改動 XML 或權重，未将框標記轉成塗色。
- [x] 30 項相關測試通過，生成指定標記的裁切比較；詳細結果與重跑命令見 `docs/shadow-recall-annotation-1001.md`。

本輪診斷完成不代表漏檢已修復；剩餘問題為模型對右下陰影與大部分 MSC 凹陷的分類或定位不足，仍需檢查訓練標記的一致性及模型能力。
