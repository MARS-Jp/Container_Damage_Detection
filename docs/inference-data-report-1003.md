# 推論完成後的 train、val、綜合數據報表

程式：`tools/report_inference_data.py`。直接讀取推論產生的 `prediction_metrics.json`、`predictions.json`，不載入權重，也不重新推論。

## 執行

```powershell
& 'C:\Users\USER\anaconda3\python.exe' -B `
  'C:\Users\USER\Documents\iim_0905\Yolov11_AttnRes\tools\report_inference_data.py' --open
```

預設讀取 `run_20261002_232129_326448/01_last0912`，標記使用 `C:\Users\USER\Desktop\已標記\labels`。主頁有三個頁籤：只有 train、只有 val、綜合 train + val。每次執行建立新的 `tools/runs/inference_reports/report_時間`。

更換來源或標記資料夾：

```powershell
& 'C:\Users\USER\anaconda3\python.exe' -B `
  'C:\Users\USER\Documents\iim_0905\Yolov11_AttnRes\tools\report_inference_data.py' `
  --results '推論結果模型資料夾完整路徑' `
  --annotations 'VOC XML 標記資料夾完整路徑' `
  --versions S08_diagnostic_only --iou 0.5 --eval-conf 0 --xml-origin 1 --open
```

`--annotations` 可指定多個資料夾或個別 XML。程式優先對應 train/val 加上類別子資料夾的相對路徑；找不到時，只在同一資料集內尋找唯一同名 XML。沒有 train/val 路徑的單一 XML 也可提供；多份候選不自動挑選。尺寸與 XML 宣告的圖片檔名必須相符，difficult 標記目前不參與評估。

## 三組資料夾

每組均有：

- `report.html`、`series_01.png`：數據頁面及可分享的圖表；多模型／版本會有多張圖表。
- `summary.csv`、`summary.json`：图片數、總偵測數、平均偵測數、空預測張數、評估張數及整體框指標。
- `class_summary.csv`：逐類偵測數、有此類別的圖片數、像素覆蓋率，以及 TP/FP/FN、Precision/Recall/F1、框 AP50/AP50–95。
- `images.csv`：逐張來源、各類預測數、對應 XML 與是否納入評估。
- `evaluation_exclusions.csv`：缺少標記、對應不明或資料不符等未納入評估的原因。
- `region_summary.csv`：原推論檔已有的左右區域數據，不把跨區物件加總當作全圖偵測數。

根目錄另有 `index.html` 三頁籤主頁、`data_quality.csv` 全部資料問題，以及 `settings.json`，記錄來源、標記路徑與評估參數。

## 數值定義

推論統計包含此組全部可讀取的推論圖片，無預測圖片也計入分母。各類遮罩覆蓋率使用同類聯集像素總和／圖片像素總和，不直接平均大小不同圖片的覆蓋率，也不把不同類別重疊面積相加當成唯一損傷面積。

評估只使用可對應且驗證通過的 XML／預測檔。缺少標記不視為沒有物件；無物件的有效 XML 才能作為背景圖片評估。頁面明確顯示實際評估張數與涵蓋率，沒有可評估資料時以空值／「—」呈現。

框匹配採同類別、信心分數由高到低、一對一、IoU ≥ `--iou`。Precision = TP/(TP+FP)，Recall = TP/(TP+FN)，F1 = 2TP/(2TP+FP+FN)。整體值採 micro（先加總 TP/FP/FN）；逐類值獨立計算。`--eval-conf 0` 使用全部已保存候選；較高值只影響評估，不改推論統計。

AP 使用 101 個 recall 取樣點的 precision envelope 平均，AP50–95 使用十個 IoU 門檻 0.50、0.55、…、0.95。mAP 平均有 GT 的類別，沒有 GT 的類別 AP 留空。綜合 train + val 使用合併後的原始 TP/FP/FN 與重新排序的預測計算 AP，不直接平均兩組指標。

XML 為矩形框，無法驗證分割邊界；這裡的 AP 是已保存、已篩選預測的框指標，與重新執行模型驗證的分割 mAP 不同。模型與推論版本分開呈現，不將多版結果算成多張資料集圖片。

預設完整標記來自「已標記/labels」，不會自動混入「0928-1」的新單張標記版本。若只提供更新的單張 XML，其他沒有標記的圖片會從評估排除，推論統計仍可正常呈現。
