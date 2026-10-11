# 九個指標逐筆輸出

程式：`tools/report_nine_metrics.py`。每筆輸出都有：Accuracy、Precision、Recall、Dice、IoU、mIoU、Mean Dice、FPR、FNR。分為 train、val、train_and_val 三組；每組都保存逐張、逐張逐類、逐類及整體資料。

## 必須指定評估基礎

- `--mode boxes`：XML 方框區域對照預測方框區域的像素評估；頁面與 CSV 明確標示這不是分割精度。只有矩形框資料時，可計算此模式。
- `--mode masks`：真實分割標記對照無損預測遮罩。不能由塗色 JPEG、方框或遮罩像素總數還原真實分割 IoU。

框區域模式：

```powershell
& 'C:\Users\USER\anaconda3\python.exe' -B `
  'C:\Users\USER\Documents\iim_0905\Yolov11_AttnRes\tools\report_nine_metrics.py' --mode boxes --open
```

預設來源與 XML 路徑沿用 `report_inference_data.py`；`--results`、`--annotations` 可覆寫。每次建立新的 `tools/runs/nine_metric_reports/report_時間`，原推論資料保持原樣。

## 真實分割模式

後續執行 `diagnose_shadow_inference.py` 時加上 `--save-masks`，每版會新增 `prediction_masks.npz`，保存原圖尺寸下的每個實例遮罩、類別、分數、來源與版本。原本的推論及篩選規則保持不變。舊結果沒有此檔案，必須重新推論取得原始遮罩；不能從 JPEG 猜測。

```powershell
& 'C:\Users\USER\anaconda3\envs\yolov11\python.exe' -B `
  'C:\Users\USER\Documents\iim_0905\Yolov11_AttnRes\tools\report_nine_metrics.py' `
  --mode masks --results '含 prediction_masks.npz 的推論資料夾' `
  --mask-annotations '真實分割標記資料夾' --open
```

真實標記可使用 YOLO 分割 TXT（類別 ID 與模型相同，不接受 box TXT），或無損 NPZ。NPZ 使用 `masks`（bool、N×H×W）與 `class_names`（N 個類別名稱，例如 D/R）；每張影像的遮罩大小必須與原圖相同。N 可代表實例或類別，多個同類遮罩會取聯集。空白且確實存在的 TXT 表示背景，缺少 TXT 不視為背景。TXT 解析需要 yolov11 環境中的 OpenCV。

## 檔案與逐筆政策

- `image_metrics.csv`：每張圖片一筆，九個指標全部保留。
- `image_class_metrics.csv`：每張圖片×每類別一筆，九個指標全部保留。
- `class_metrics.csv`：同一模型／版本的每類別彙總。
- `summary.csv`：同一模型／版本的整體九個指標。
- `not_evaluated.csv`：無法評估的圖片及原因；這些圖片仍保留在逐張 CSV。
- `report.html`、主目錄 `index.html`：三組頁籤及可搜尋、每頁 50 筆的逐張表格。

CSV 數值為 0–1；網頁呈現百分比。缺少有效標記、缺少原始預測遮罩、分母為零等情況以 NA 表示，並寫入 `NA_reason`，不補造分數。

## 定義

每個前景類別獨立使用一對其餘像素的二元遮罩，可保留不同類別的重疊。TP/FP/FN/TN 均為像素數，不是物件匹配數。Accuracy=(TP+TN)/(TP+FP+FN+TN)，Precision=TP/(TP+FP)，Recall=TP/(TP+FN)，Dice=2TP/(2TP+FP+FN)，IoU=TP/(TP+FP+FN)，FPR=FP/(FP+TN)，FNR=FN/(TP+FN)。

逐張整體與資料集整體的這七項使用所有前景類別的像素計數加總（micro）。mIoU、Mean Dice 使用所屬圖片／資料集的逐類 IoU、Dice 平均（macro），不包含背景類別，預測與標記都不存在的類別不納入平均。**單一類別的 mIoU、Mean Dice 不適用，九欄仍保留，但這兩欄為 NA。** 2026-10-08 核查後不再把所屬圖片／資料集的宏觀平均重複到每個類別。

train_and_val 先加總像素計數再重新計算，不直接平均 train 與 val。Accuracy 的分母包含各類別的背景／其他類別像素，因此可能高於前景 IoU；評估方式明確記錄在設定與每筆 CSV。

## 核查後的解讀與新增欄位

`metric_scope` 與 `macro_scope` 指出分數的範圍；`gt_positive_pixels`、`predicted_positive_pixels`、逐類的 `gt_positive_images`、`predicted_positive_images` 顯示評估支持量。`always_background_accuracy` 是每個類別全部預測為負的基準，`accuracy_gain_over_background` 是模型 Accuracy 超過該基準的差值，單位仍為 0–1。

舊表 train+val 的 97.7762% Accuracy，全部預測背景也可得到 97.0160%，差值僅 0.7602 個百分點，不能解讀為 97.8% 損傷辨識成功。616 張可評估、434 張無有效對應 XML；不能將此結果推廣為全部 1,050 張的分割精度。

框區域 Precision 66.2522% 是矩形聯集的像素比例；同一批有效圖片以同類、IoU≥0.5、一對一物件匹配計算，Precision 為 28.2287%。兩者分母與評估目標不同；矩形像素數值不能替代偵測物件的命中率。

完整核查程式 `tools/audit_inference_metrics.py` 獨立重建矩形像素，核對推論輸入快照、原始 JSON、XML、重複圖片與可取得的標記來源照片；另以 CBHU0708054-A 的原始遮罩對照舊多邊形標記。執行：

```powershell
& 'C:\Users\USER\anaconda3\envs\yolov11\python.exe' -B tools/audit_inference_metrics.py
```

原始報表及推論結果保留。修正後重新執行 `report_nine_metrics.py` 會建立新目錄；舊 PDF／statistics_charts 不會自動更新，不應沿用其逐類重複宏觀分數。
