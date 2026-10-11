# 訓練標註與三份權重核查（2026-10-01）

已完成本機 256 張資料核查及三份權重的同條件推論。三份權重仍未解決右下陰影漏檢；本機資料存在標註版本落差、相同照片標註衝突及訓練／驗證使用同一路徑的問題。尚未取得 Linux 當時實際使用的資料內容，以下區分本機資料事實和 checkpoint 自帶的訓練紀錄。

## 交付檔案

- 完整核查：`runs/shadow_training_audit/final/run_20261001_200610_416081/`。
- [94 個物件的原圖／舊標註核對頁](C:/Users/USER/Documents/iim_0905/Yolov11_AttnRes/runs/shadow_training_audit/final/run_20261001_200610_416081/review.html)。頁面可切換右下陰影、左側 MSC、其他區域。
- [逐項補標核對清單](C:/Users/USER/Documents/iim_0905/Yolov11_AttnRes/runs/shadow_training_audit/final/run_20261001_200610_416081/annotation_review.csv)。座標為零起點 xyxy，XML ID 沿用檔案順序。
- [相同照片的標註衝突清單](C:/Users/USER/Documents/iim_0905/Yolov11_AttnRes/runs/shadow_training_audit/final/run_20261001_200610_416081/duplicate_label_conflicts.csv)。列出兩份圖片的路徑、類別及逐類遮罩相異像素數。
- [三份權重的純塗色比較圖](C:/Users/USER/Documents/iim_0905/Yolov11_AttnRes/tools/runs/shadow_training_audit/model_comparison/run_20261001_200415_640203/comparison/models_comparison.png)。每個模型亦保存獨立資料夾及逐類圖片。
- `audit_summary.json`、`checkpoint_audit.json`、`dataset_audit.json`、`dataset_inventory.csv` 保存依據；`sources/` 是原始 XML、TXT 與 data.yaml 快照，JSON 保存來源 SHA256。`model_reports/` 收集三份權重的訓練紀錄。

## 同一張照片的新舊標註

來源照片和 dataset0608 中的同名照片 SHA256 相同：`67c87d8a468f252e48527bc9d087648c6f18528659fcb56c1c017bd2f8df6d79`，大小 800×600。

| 標註來源                                                                      | 凹陷 D | 鏽蝕 R | 型態   |
| ----------------------------------------------------------------------------- | -----: | -----: | ------ |
| Desktop/0928-1/CBHU0708054-A.xml                                              |     25 |     69 | 方框   |
| Desktop/已標記/labels/train/B/CBHU0708054-A.xml                               |     12 |     19 | 方框   |
| Desktop/碩論/dataset0608/dataset0608/dataset/labels/train/B/CBHU0708054-A.txt |     12 |     19 | 多邊形 |

新 XML 有 14 處 D、44 處 R 與舊同類多邊形的包圍框完全不相交。實際把舊多邊形轉成像素核查，則有 14 處 D、45 處 R 的新 XML 框內沒有任何舊同類遮罩像素。兩個統計用途不同，不能把框交集當作真實輪廓覆蓋。

| 重點區域    | 新標註 | 與舊同類框不相交 | 框內没有舊同類遮罩像素 |
| ----------- | -----: | ---------------: | ---------------------: |
| 左 MSC 凹陷 |     22 |               13 |                     13 |
| 右下凹陷    |      2 |                1 |                      1 |
| 右下鏽蝕    |     11 |                8 |                      8 |

例如右下 D #84、R #30、R #86 缺少對應舊標註；D #85 有部分舊同類遮罩，但範圍與新框不同。這是補標與物件切分的一致性檢查，不是分割準確率。新 XML 沒有分割邊界，所以核對頁只呈現原圖與舊多邊形填色，未生成新訓練標註。

## 全部本機資料

核查根目錄：`C:/Users/USER/Desktop/碩論/dataset0608/dataset0608/dataset`。

- 256 張圖片均在 images/train，images/val 為空；256 份對應 TXT 都存在，沒有孤立 TXT。
- data.yaml 寫的是 `train: images/train`、`val: images/train`。本次以明確的本機 dataset-root 檢查此資料；相對 `path: ./dataset` 在其他工作目錄的解析結果另需確認。
- 69 份 TXT 為空。空標記可能是真實背景，不能全數判錯；但其中 12 份有內容完全相同、另一份非空標記的照片副本。
- 25 組照片 SHA256 完全相同，其中 23 對逐類分割遮罩不同。差異包括類別不同、輪廓不同，以及上述 12 對有標註／空白衝突。
- 非空標記皆通過本工具的分割欄位、有限值、[0,1] 座標、類別、非退化面積與重複同類包圍框檢查。這不等於已完成人工判讀，也未全面檢查自交或視覺漏標。
- 物件數：B 18、C 19、D 326、H 26、O 19、R 623、X 89、U 4。

具體衝突例子：`train/C/CIMG4463(1).JPG` 與 `train/C/CIMG4463.JPG` 圖片完全相同，前者標 O/R，後者 TXT 空白；`train/B/CBHU0708054-B.JPG` 與 `train/B/CIMG6427.JPG` 也完全相同，但 D/R 輪廓不同。資料切分需同時檢查內容雜湊及貨櫃身分，僅靠檔名不能隔離這些副本。

## 權重核查與比較

本輪明確指定以下三份 `.pt`。檢查時 `predict-2.py` 已啟用 `best0929-3.pt`，所以命令使用 `--weights`／`--model` 指定三份歷史權重；本輪未修改這個入口設定。交付的正式主核查為 `final/run_20261001_200610_416081/`，不可用早期暫存目錄名稱推斷模型身分。

三者的類別表皆為 B,C,D,H,O,R,X,U（D=2、R=5）；未發現浮點權重／緩衝張量的 NaN 或 Inf。三者 checkpoint 記錄的 head.py、block.py、資料處理與標準分割 predictor，與本機對應追蹤版本一致。這些結果沒有證明模型精度正確，但排除了這幾項明顯的不一致。

| 權重                  | head                | 訓練紀錄輪數 | overlap_mask | 最末分割 mAP50 | 歷史最高分割 mAP50 |
| --------------------- | ------------------- | -----------: | ------------ | -------------: | -----------------: |
| last0912.pt           | Segment26MultiLabel |          250 | False        |         2.054% |   4.410%（102 輪） |
| last0911-multicasr.pt | Segment26           |          600 | True         |         0.984% |   2.466%（332 輪） |
| last0911-2.pt         | Segment26           |          600 | True         |         1.825% |   4.343%（256 輪） |

上表是 checkpoint 保存的歷史驗證紀錄，並非本次重新測得的獨立驗證成績；三者都記錄相對名稱 `data.yaml`，未附標註內容雜湊，不能證明驗證集相同。紀錄中的 `(B)` 指 boxes、`(M)` 指 masks；不是損傷類別 B。

同條件實測使用原圖、imgsz=640、conf=0.15、NMS IoU=0.7、retina_masks=True、max_det=300。三者沿用同一份橘色柱子 R 排除規則；該规则仍屬手動背景排除。沒有影像增強、裁切追加或模型融合。

新 XML 評估為同類、按信心分數排序的一對一框匹配，IoU≥0.5：

| 權重               | 全圖 D 命中/25 | 全圖 R 命中/69 | 未匹配預測 D/R | 左 MSC D/22 | 右下 D/2 | 右下 R/11 |
| ------------------ | -------------: | -------------: | -------------: | ----------: | -------: | --------: |
| last0912           |              2 |              8 |        14 / 34 |           2 |        0 |         0 |
| last0911-multicasr |              2 |              5 |        11 / 31 |           2 |        0 |         0 |
| last0911-2         |              2 |              6 |        13 / 28 |           2 |        0 |         0 |

將匹配 IoU 放寬為 0.3，右下凹陷仍皆為 0/2；右下 R 只有 last0912 命中 1/11，另兩份為 0/11。這張照片已用於診斷和參數選擇，結果只能用於本圖比較。XML 只有框，不能評估分割邊界的好壞。未匹配預測包括可能的誤檢、重複、定位偏差及物件拆分差異。

## 依據核查結果安排修正

1. 取得 Linux 原始訓練的 data.yaml、labels 與對應圖片清單，確認 checkpoint 的資料來源。使用者補充的兩個路徑為模型權重，已完成比較；訓練資料來源仍未確認。
2. 優先處理 23 對相同照片的標註衝突，逐圖保留經確認的完整輪廓。空白／非空副本不能同時沿用互相矛盾的標註，亦不宜直接把兩份輪廓聯集合併。
3. 依本次 94 項清單補畫實際損傷多邊形，先檢查右下 D #84、R #30/#86 和左 MSC 的 13 個無交集凹陷。確認長條鏽蝕和局部鏽斑的物件拆分規則；新 XML 方框本身不能當成分割遮罩。
4. 在資料完成核對後建立貨櫃與重複影像均分離的 train/val。這張已多次調參的照片只能作診斷樣本，另外準備未參與調整的陰影損傷照片。
5. 使用確認的分割資料重訓／微調，再與目前三份權重比較。單純換這兩份 0911 權重尚無改善證據；本輪未改模型參數或開始重訓。

## 重跑方式與驗證

```powershell
& 'C:/Users/USER/anaconda3/envs/yolov11/python.exe' -B tools/audit_shadow_training.py --model 'C:/Users/USER/Downloads/last0912.pt'

& 'C:/Users/USER/anaconda3/envs/yolov11/python.exe' -B tools/predict_shadow.py --weights 'C:/Users/USER/Downloads/last0912.pt' 'C:/Users/USER/Downloads/last0911-multicasr.pt' 'C:/Users/USER/Downloads/last0911-2.pt' --views original --imgsz 640 --conf 0.15 --iou 0.7 --device 0 --output 'C:/Users/USER/Documents/iim_0905/Yolov11_AttnRes/runs/shadow_training_audit/model_comparison'

& 'C:/Users/USER/anaconda3/envs/yolov11/python.exe' -B tools/compare_audited_weights.py --run 'C:/Users/USER/Documents/iim_0905/Yolov11_AttnRes/tools/runs/shadow_training_audit/model_comparison/run_20261001_200415_640203'

& 'C:/Users/USER/anaconda3/envs/yolov11/python.exe' -B -m unittest discover -s tests -p 'test_audit_shadow_training.py' -v
```

最後一個比較命令重建已完成的三模型結果；要比較新推論，改用新輸出的 run 資料夾。核查工具預設讀取 predict-2.py 的模型和圖片，因此正式比對請如上明確提供 `--model`。

5 個針對核查語義的測試通過，涵蓋不將框當分割、多邊形異常、框交集與像素覆蓋的區別、類別／ROI 對應、完全相同圖片的標註衝突。三份權重推論及 XML 評估皆完成；核對頁有 94 項、188 張原圖／舊標註局部圖，照片內維持無文字、無框、無輪廓線。
