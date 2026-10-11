# 1001 陰影漏檢：標註核對與 13 版實驗

這次執行了 13 個版本、117 張逐類／全圖結果與 8 張比較圖。右下鏽蝕有局部找回，凹陷尚未穩定改善；原推論入口與權重保留原設定。所有推論照片維持純遮罩塗色，沒有信心度、D/R 文字、偵測框或輪廓線。

## 標註差異

使用者提供的 `C:/Users/USER/Desktop/0928-1/CBHU0708054-A.xml` 有 **69 R、25 D**。本機 `dataset0608/dataset0608/dataset/labels/train/B/CBHU0708054-A.txt` 只有 **19 R、12 D**；`Desktop/已標記/labels/train/B/CBHU0708054-A.xml` 也只有 19 R、12 D。

原推論圖片、dataset0608 的圖片、已標記目錄的圖片 SHA256 完全相同：`67c87d8a468f252e48527bc9d087648c6f18528659fcb56c1c017bd2f8df6d79`。差異來自標註版本，沒有發現三份圖片內容不同。

新 XML 中 **14 D、44 R** 與舊分割多邊形的同類包圍框沒有任何交集。這是標註核對結果，**不能直接斷言 checkpoint 使用的就是本機這份資料**：權重只有 `data.yaml` 名稱，沒有訓練標註 hash。兩套標註還可能使用不同的物件拆分方式；例如長條鏽蝕可以被分成亮部、暗部兩個框。

新 XML 沒有分割輪廓。因此本輪用框比對評估物件定位，沒有把 XML 方框填成損傷遮罩，也沒有將它當成新的分割訓練真值。

## 實作

- `tools/damage_segmentation_predictor.py`：獨立的可選 predictor，允許同一候選保留多類別；保留原始 proto/mask coefficients，也能讀取已訓練的 multi-label 像素分支。未修改 Ultralytics 預設 predictor。
- `tools/predict_shadow_recovery.py`：讀取 predict-2.py 的模型、來源、imgsz/conf/IoU/device，固定重疊網格裁切整張圖片，原圖／Gamma／CLAHE 推論後映回原座標。基準遮罩保留；新增遮罩只去除同類重複，不膨脹、不強制聯集。沿用右側橘色柱子的既有 R 排除設定。
- `tools/evaluate_damage_xml.py`：沿用既有 XML 座標轉換，同類、分數排序、一對一配對，分別記錄框 IoU 0.30 與 0.50；區域由真值框中心分組。
- `label_review.csv` 和 `annotation_audit.json`：逐項標記新 XML 與舊多邊形的交集、基準是否命中、各版命中結果。

輔助分割版為實驗性輸出，直接對已有 auxiliary logits 取 sigmoid、閾值及連通區域，不等同於重新訓練。像素機率與主偵測頭的信心分數分開記錄。這個 checkpoint 的前兩個輔助頭順序是 P3、P2。

## 實測

相同 640 / conf0.15 / IoU0.7，候選檢查門檻 0.03。單張框 IoU 0.50：

| 版本 | 全圖 D 命中/25 | 全圖 R 命中/69 | 未匹配 D/R | 右下 D 命中/2 | 右下 R 命中/9 |
|---|---:|---:|---:|---:|---:|
| T01 基準 | 2 | 8 | 14 / 34 | 0 | 0 |
| T03 基準＋512 重疊裁切 | 2 | 10 | 24 / 52 | 0 | 1 |
| T09 裁切 CLAHE＋P3 像素0.30 | 3 | 13 | 77 / 164 | 0 | 0 |
| T10 裁切 CLAHE＋P2/P3 均值0.50 | 3 | 9 | 49 / 105 | 0 | 0 |
| T13 512 裁切只補暗部 R | 2 | 10 | 14 / 49 | 0 | 1 |

T03/T13 找回新 XML #7（左側 R）和 #30（右下 R），框 IoU 分別約 0.592、0.681。T13 只保留至少半數遮罩落在左右檢查區、遮罩亮度中位數不超過 128 的 R 新增候選；原基準 D 不變。裁切、區域和亮度條件不讀取 XML 座標。

這些版本增加的未匹配預測仍多，不能把新增候選數當成準確率提高，也沒有足夠證據把它們切換成預設版本。部分未匹配項可能是重複、定位偏差或標註拆分方式不同。XML 只有框，無法判定分割邊界是否正確。

## 結果與重跑

結果根目錄：`runs/segment_shadow_recovery/run_20261001_101924_904578/`。

- [全圖比較](../runs/segment_shadow_recovery/run_20261001_101924_904578/comparisons/all_variants.png)
- [右下重點版本](../runs/segment_shadow_recovery/run_20261001_101924_904578/comparisons/selected_right.png)
- [新舊標註與推論局部比較](../runs/segment_shadow_recovery/run_20261001_101924_904578/comparisons/annotation_review.png)
- [逐項補標核對清單](../runs/segment_shadow_recovery/run_20261001_101924_904578/label_review.csv)
- [完整評估表](../runs/segment_shadow_recovery/run_20261001_101924_904578/evaluation_summary.csv)

```powershell
& 'C:\Users\USER\anaconda3\envs\yolov11\python.exe' -B tools/predict_shadow_recovery.py
```

只重新產生評估與比較，不再推論：

```powershell
& 'C:\Users\USER\anaconda3\envs\yolov11\python.exe' -B tools/predict_shadow_recovery.py --rebuild-report 'runs/segment_shadow_recovery/run_20261001_101924_904578'
```

這個工具的左右區域針對 CBHU0708054-A；換圖時需修改 REGIONS，並提供匹配的 XML 與舊分割檔。各版本保存二值 `masks.npz`，評估不從 JPEG 顏色反推遮罩。

## 後續模型修正所需資料

先確認 0912 實際訓練的 data.yaml 與分割標註，再依 `label_review.csv` 補齊陰影損傷輪廓。既有正確多邊形應保留；凹陷與鏽蝕可重疊，但不能把正常浪板或投影標成損傷。只有框的新增項需要補畫多邊形。

本機 data.yaml 的 train、val 都指向 images/train；若它也是正式訓練設定，重訓時應改用分離的驗證圖片，且同一貨櫃的 A/B/C/D 圖片放在同一切分。這張圖已用於選參數，不能再把它的結果當成獨立驗證。

確認並修正標註後，可從 last0912.pt 初始化同架構的新訓練，使用 resume=False、overlap_mask=False；結果需與獨立的陰影樣本比對。本次沒有用單張 XML 方框代替多邊形來訓練模型。
