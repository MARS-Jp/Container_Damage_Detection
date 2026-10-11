# 純塗色輸出與三組參數比較

`predict_shadow.py` 的所有結果圖只保留原有 alpha=0.35 分割塗色，不畫信心分數、D/R 標籤、偵測框或輪廓線。
此顯示修改保留原本推論設定。信心分數與類別仍記錄在 JSON，以便檢查；移除圖片文字不代表取消 conf 推論門檻。

## 一次執行三組

在專案根目錄執行：

```powershell
& 'C:\Users\USER\anaconda3\envs\yolov11\python.exe' tools/predict_parameter_sweep.py --suite all --compare
```

程式每次讀取 `tools/predict-2.py` 中已啟用的模型、來源、imgsz/conf/IoU/device/retina_masks。
目前是 last0912.pt、一張 CBHU0708054-A.JPG，基準為 640 / 0.15 / 0.7。
三組一律使用相同的 `predict_shadow.py` 純塗色方法，方便比較；retina_masks 必須開啟。

預設增強設定：original+gamma+clahe、Gamma=0.65、CLAHE=2.0、亮度門檻=255、合併 IoU=0.60。
baseline 一律只啟用 original；一般組所有版本都只啟用 original。

| 組別 | 版數 | 版本與調整 |
|---|---:|---|
| 01_shadow 陰影增強 | 9 | baseline、shadow_default、gamma_only、clahe_only、gamma045、gamma085、clahe40、shadow128、merge040 |
| 02_general 一般推論 | 8 | baseline；conf=0.05/0.10/0.25；IoU=0.30/0.50；imgsz=960/1280 |
| 03_combined 兩者搭配 | 8 | baseline、shadow_default；增強+conf=0.10、IoU=0.50、imgsz=960/1280；960+conf0.10+IoU0.50；960+Gamma0.45+亮度128 |

陰影組固定一般參數；混合組除 baseline 外都啟用三視圖。
Gamma、CLAHE、亮度與合併 IoU 的版本，除列出項目外沿用相同增強預設。
參數表集中在 `tools/predict_parameter_sweep.py` 的 SHADOW_DEFAULTS、SHADOW_TESTS、GENERAL_TESTS、COMBINED_TESTS。
若自行修改 predict-2.py 的基準，個別 preset 可能與基準相同；实际參數以 parameters.json 為準。

只跑一組或少數版本：

```powershell
& 'C:\Users\USER\anaconda3\envs\yolov11\python.exe' tools/predict_parameter_sweep.py --suite shadow --compare
& 'C:\Users\USER\anaconda3\envs\yolov11\python.exe' tools/predict_parameter_sweep.py --suite general --compare
& 'C:\Users\USER\anaconda3\envs\yolov11\python.exe' tools/predict_parameter_sweep.py --suite combined --versions baseline imgsz960 size960_conf010_iou050 --compare
& 'C:\Users\USER\anaconda3\envs\yolov11\python.exe' tools/predict_parameter_sweep.py --suite all --dry-run
```

`--suite all --versions baseline imgsz960` 會在含有該名稱的各組選版。`--dry-run` 只列出指令，不推論、不建立結果資料夾。
原本 initial/focused 的用法仍保留。

## 右側橘色柱子的背景排除

`tools/shadow_exclusion_regions.json` 指定目前 CBHU0708054-A.JPG 右側橘色柱子的多邊形，只排除 R 類別。
程式以完整原圖路徑匹配，不會把其他圖片的右側一律刪除；多邊形採 0..1 正規化座標。
排除在各視圖的遮罩上執行，跨越边界的遮罩只移除排除區內像素，完全位於區內的預測會被移除。
貨櫃本體與 D 類別維持原有處理。這是針對這張圖指定背景範圍，不代表模型已學會区分油漆和鏽蝕。

每次批次會先保存一份 `exclusion_regions.json` 快照，讓三組與所有版本使用相同設定。
每張 `predictions.json` 記錄排除區、排除前各視圖數量 raw_view_counts，以及排除後數量 view_counts。
若要查看模型未排除的預測，可在單張或批次指令加 `--no-exclusions`。
換圖或換位置時，請明確新增該圖片的 source 與多邊形，不會自動沿用本張的柱子位置。

## 結果位置

```text
runs/segment_shadow/parameter_tests/run_時間/
  parameters.json
  summary.csv
  exclusion_regions.json
  01_shadow/版本/模型/來源/
  02_general/版本/模型/來源/
  03_combined/版本/模型/來源/
    parameters.json
    inference.log
    run_時間/模型/圖片/
      *__original.jpg
      *__all.jpg
      *__class_*.jpg
      predictions.json
      views/（增強輸入與各視圖結果）
  comparisons/
    README.md
    comparison_summary.csv
    001_圖片名/
      01_shadow/（全圖、D、R、局部比較）
      02_general/
      03_combined/
      three_groups_overview.png
      three_groups_detail.png
```

每次執行建立新資料夾；單一工作失敗會留下日誌並繼續。比較只納入成功工作的結果。
比較圖片的版本與參數放在照片外；跨組總覽使用固定選版，不自動宣稱最佳版本。
每個模型與實際來源圖片分開比較，即使不同資料夾有同名圖片也不會混在一起。
遮罩覆蓋率使用同類二值遮罩聯集計算，重疊不重複計算。數量或覆蓋率增加不等於準確率提升。

僅重做比較图，不再推論：

```powershell
& 'C:\Users\USER\anaconda3\envs\yolov11\python.exe' tools/compare_shadow_sweep.py --run-dir '結果資料夾完整路徑'
```

## 本次已完成結果

- 柱子修正版：[純塗色圖片](../runs/segment_shadow/run_20261001_004913_595036/01_last0912/000001_CBHU0708054-A/CBHU0708054-A__all.jpg)。
- 批次結果：`runs/segment_shadow/parameter_tests/run_20261001_005102_586762/`，25/25 成功。
- [跨組比較總覽](../runs/segment_shadow/parameter_tests/run_20261001_005102_586762/comparisons/001_CBHU0708054-A/three_groups_overview.png)。
- [完整比較索引與數據表](../runs/segment_shadow/parameter_tests/run_20261001_005102_586762/comparisons/README.md)。
- 已生成 17 張比較圖，23 項相關測試通過。三組 baseline 的圖片完全相同，25 版皆使用同一份排除設定。
