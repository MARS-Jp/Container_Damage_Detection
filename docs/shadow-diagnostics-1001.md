# 陰影區局部推論檢查

`tools/diagnose_shadow_inference.py` 沿用 `predict-2.py` 的模型、圖片與基準推論參數。
它載入每份權重一次，分別測試原圖、Gamma、CLAHE、左右局部裁切與裁切後的亮度／對比增強，
輸出 8 組對照，並自動建立全圖及左右區域的 all／D／R 比較圖。

```powershell
& "C:\Users\USER\anaconda3\envs\yolov11\python.exe" `
  "C:\Users\USER\Documents\iim_0905\Yolov11_AttnRes\tools\diagnose_shadow_inference.py"
```

結果位於 `tools/runs/segment_by_class/shadow_diagnostics/run_時間/`，每次建立新資料夾。

**預設檢查區與裁切背景範圍針對 CBHU0708054-A 設定。** 換圖片時應以
`--left-roi`、`--right-roi`、`--left-context`、`--right-context` 指定合適範圍。
四個值為相對於原圖寬高的 `(left, top, right, bottom)`，均介於 0 和 1；context 必須包含 roi。
`--regions left` 或 `--regions right` 可只檢查其中一側。

預設 ROI：左 `(0.03, 0.20, 0.54, 0.90)`，右 `(0.65, 0.70, 0.805, 0.89)`。
右側範圍特別位於鋸齒陰影下方，避免將亮部新增候選算成陰影改善。
ROI 統計要求至少 5 個遮罩像素，且至少一半遮罩面積落在指定區內。

局部結果映射回原始座標，保留原有基準預測；新增預測以同類 mask IoU 去重。
遮罩不會被人工連接、膨脹或合併成較大面積。
結果照片保持只塗色；比較圖的說明與數量位於照片外。

0.01 的低分版本是診斷對照。圖上出現更多塗色仍須經真實標註或人工確認，不能直接視為漏檢已修復。

## 只執行 S08

加入 `--versions S08_diagnostic_only`，每個模型／每張圖片只對原圖呼叫模型一次，並只保存 S08 的全圖與各類別結果。不執行 Gamma、CLAHE 或局部裁切推論，也不產生其他版本或版本比較圖。

```powershell
& 'C:\Users\USER\anaconda3\envs\yolov11\python.exe' -B `
  'C:\Users\USER\Documents\iim_0905\Yolov11_AttnRes\tools\diagnose_shadow_inference.py' `
  --weights 'C:\Users\USER\Downloads\last0912.pt' `
  --versions S08_diagnostic_only `
  --imgsz 640 --conf 0.15 --diagnostic-conf 0.01 --iou 0.7 --merge-iou 0.6
```

圖片仍沿用 predict-2 的 DATASET_SOURCES；可用 `--source '圖片完整路徑'` 覆寫。權重在命令內明確指定 last0912.pt，避免受 predict-2 目前模型清單影響。

S08 的篩選保持原樣：全圖基準信心度至少 0.15；只在左右檢查區補入信心度至少 0.01 的 D/R 候選，再以同類 mask IoU > 0.60 去重。顏色、0.35 透明度和純塗色輸出保持原設定。

已用 CBHU0708054-A.JPG 實際核對：只進行一次模型推論，輸出 50 D／142 R 候選；predictions.json、prediction_metrics.json 與原 S08 相同，9 張 JPEG 逐位元組一致。此次驗證結果位於 `runs/segment_shadow_s08/run_20261002_225637_759929/`，另有 verification.json。

不提供 `--versions` 時仍執行原本 9 次推論／8 個版本。也可一次指定多版，例如 `--versions S01_baseline S08_diagnostic_only`，兩版會共用同一次原圖推論。

## 大量 dataset 批次推論

`--source` 可指定圖片檔案、圖片資料夾，或一次指定多個資料夾。資料夾會遞迴搜尋所有子資料夾內的 JPG、JPEG、PNG、BMP、TIF、TIFF、WEBP；副檔名不區分大小寫。重疊的來源資料夾會去除重複圖片，XML/TXT 等標註檔不會作為圖片推論。

現有貨櫃 dataset 掃描結果：train 876 張、val 驗證 174 張，共 1,050 張。選 train 就包含 B、C、D、H、O、R、X 子資料夾；也可只指定 `train/B` 或 `train/R`。

在專案根目錄先預覽數量：

```powershell
$datasetRoot = 'C:\Users\USER\Desktop\碩論\貨櫃-20221012T025654Z-001\貨櫃\Container_damage Dataset'
& 'C:\Users\USER\anaconda3\envs\yolov11\python.exe' -B tools/diagnose_shadow_inference.py `
  --weights 'C:\Users\USER\Downloads\last0912.pt' `
  --source "$datasetRoot\train" "$datasetRoot\val 驗證" `
  --versions S08_diagnostic_only --dry-run
```

`--dry-run` 顯示唯一圖片數量、版本數及預計模型呼叫次數；不載入模型、不建立輸出資料夾。上述命令會列出 1,050 張圖片／1,050 次模型推論。

確認來源後，移除 `--dry-run` 即可實際執行：

```powershell
& 'C:\Users\USER\anaconda3\envs\yolov11\python.exe' -B tools/diagnose_shadow_inference.py `
  --weights 'C:\Users\USER\Downloads\last0912.pt' `
  --source "$datasetRoot\train" "$datasetRoot\val 驗證" `
  --versions S08_diagnostic_only --continue-on-error `
  --imgsz 640 --conf 0.15 --diagnostic-conf 0.01 --iou 0.7 --merge-iou 0.6
```

替換 `--source` 後面的路徑，即可選其他 dataset；只跑 train 時刪除第二個路徑。也可在 predict-2.py 的 DATASET_SOURCES 清單中填入資料夾完整路徑，不提供 `--source` 時會使用該清單。

程式每個權重只載入一次，逐張處理並顯示 `[模型序號/模型數][圖片序號/圖片數]`。每張圖片仍只跑一次 S08，輸出全圖加 8 類別圖，共 9 張純塗色 JPEG。圖片資料夾帶有索引，所以不同來源的同名檔案會分開保存。

輸出依原圖所在的資料集子資料夾自動分開：`train` 存入每個模型結果下的 `train/`，`val`、`val 驗證`、`valid`、`validation` 存入 `val/`，`test` 存入 `test/`；來源路徑沒有可辨識的名稱時存入 `other/`。例如：

```text
run_時間/
└── 01_last0912/
    ├── train/
    │   └── 01_CBHU0708054-A/S08_diagnostic_only/...
    └── val/
        └── 877_原圖名稱/S08_diagnostic_only/...
```

無須新增推論參數，同時指定 train、val 的原命令即可。`--dry-run` 也會顯示各資料集張數及預計輸出位置。圖片檢視器可選擇 `01_last0912/train` 或 `01_last0912/val` 單獨查看。

每次執行使用新的 run 資料夾，並額外保存：

- `source_manifest.csv`：本輪選到的每張原圖、dataset_split（train/val/test/other）及相對於模型資料夾的輸出路徑。
- `image_summary.csv`：dataset_split、逐張成功／失敗狀態及錯誤訊息，處理時持續寫入。
- `region_summary.csv`：dataset_split、各版左右區域 D/R 數量、面積與最高信心分數，逐張寫入，避免累積整個 dataset 的統計。

`--continue-on-error` 會記錄無法處理的圖片並繼續後續圖片；完成後若有失敗項，程式回傳非零狀態。未指定時仍在第一個錯誤停止。中斷後重新執行會建立新 run，目前不會自動接續舊 run。

S08 的低分候選檢查區仍採用 CBHU0708054-A 的固定左右相對座標，不會自動尋找其他圖片的陰影。圖片視角改變時需調整 ROI；批次功能保持原 S08 推論及篩選規則。
