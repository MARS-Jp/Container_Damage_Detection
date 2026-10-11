# Shadow Training Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 查清單張漏檢與本機標註、資料切分及 checkpoint 訓練紀錄的關係，產出可追溯核查報告。

**Architecture:** 新增獨立唯讀核查程式，沿用既有 XML 座標與 ROI 定義。解析全部本機分割標註，記錄來源 SHA256，輸出 CSV、JSON、訓練曲線與原圖／舊標註裁切核對頁；不改推論或訓練程式。

**Tech Stack:** Windows PowerShell、既有 yolov11 Python、NumPy、OpenCV、PyYAML、Matplotlib、unittest。

**Spec:** docs/superpowers/plans/2026-10-01-inference-comparison-spec.md（第 10–12 項）

## Global Constraints

- 使用現有 Windows/PowerShell、yolov11 Python 環境與本地 GPU，不增加依賴。
- 不把 XML 方框改成分割真值；所有照片內不畫框、不放文字。
- 不更改原資料、權重與推論預設，不啟動訓練。
- 本機 dataset0608 不能冒稱已確認的 Linux 0912 訓練資料。

## Task 1: 核查工具與數值驗證

**Files:** Create `tools/audit_shadow_training.py`; Test `tests/test_audit_shadow_training.py`.

**Interfaces:** `parse_polygons(text: str, names: dict) -> tuple[list[dict], list[dict]]` 回傳可用多邊形與逐行問題；`compare_annotations(annotation: dict, polygons: list) -> list[dict]` 回傳每個 XML 物件的同類框交集及實際多邊形覆蓋比例。兩者不載入模型，方便測試。

- [x] 驗證混入偵測框、無效座標和退化多邊形不會被當成合法分割標註。

```python
polygons, issues = parse_polygons('2 0.5 0.5 0.2 0.2', {2: 'D'})
assert not polygons
assert issues[0]['issue'] == 'box_row_not_segmentation'
```

- [x] 驗證同類框交集不會被誤當成實際遮罩覆盖，且不同類別不交叉配對。
- [x] 掃描全部圖片／標註配對、class counts、空標記、重複框、內容雜湊及容器命名群組；逐物件比對新 XML 與舊分割。
- [x] 保存 checkpoint 原始 train_args、train_metrics、250 輪 train_results；紀錄已載入模型的架構、類別與追蹤中的程式差異。

```powershell
& 'C:/Users/USER/anaconda3/envs/yolov11/python.exe' -B -m unittest discover -s tests -p 'test_audit_shadow_training.py' -v
& 'C:/Users/USER/anaconda3/envs/yolov11/python.exe' -B tools/audit_shadow_training.py
```

## Task 2: 解讀與交付

**Files:** Create `docs/shadow-training-audit-1001.md`; output `runs/shadow_training_audit/run_*/`.

- [x] 核對來源圖片相同雜湊、新 XML 94 個物件、舊分割 31 個物件；核對報告與 CSV 每筆一致。
- [x] 在審閱頁按 XML ID 列出原图與舊多邊形填色的同一裁切，只作補標參考；沒有新輪廓的物件保持未補標。
- [x] 檢查曲線圖和補標頁，說明 `(B)` 為 box、`(M)` 為 mask，不是損傷類別 B。
- [x] 彙整可確認結論、未確認 Linux 資料來源，以及按順序進行的修正工作；原始標註与權重不寫入。

執行方式沿用本對話已授權的同一工作階段；不建立 Git 提交或影響其他並行變更。

## 完成紀錄

- 核查輸出：`runs/shadow_training_audit/final/run_20261001_200610_416081/`。
- 使用者補充兩份 0911 權重後，另新增 `tools/compare_audited_weights.py`，完成三份權重同條件實測及框 IoU 0.3/0.5 評估。
- 256 張圖片、94 項新標註、188 張核對局部圖已核對；5 項測試通過。
- 確認 25 組完全相同照片、23 對標註遮罩不同，其中 12 對為空白／非空衝突。
- 新舊資料來源的未確認事項與三模型實測結果已寫入 `docs/shadow-training-audit-1001.md`；未修改原始資料、推論入口或權重。
