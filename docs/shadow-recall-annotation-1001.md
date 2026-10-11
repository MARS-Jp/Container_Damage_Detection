# MSC 凹陷與右下陰影的漏檢檢查

使用者提供 `C:/Users/USER/Desktop/0928-1/CBHU0708054-A.xml`：800×600，25 個 D 框、69 個 R 框，合計 94 個。
這份 XML 沒有分割輪廓。以下以同類別、依分數排序的一對一框 IoU ≥ 0.5 比對，不能當成分割遮罩準確率。
VOC 的 xmin/ymin 從一開始計數，讀取後各减 1；xmax/ymax 保留。可用 `--coordinate-origin 0` 改成直接使用起點。

## 已執行的檢查

`tools/predict_shadow_recall.py` 延用 predict-2.py 的模型與來源，保留既有純塗色方式與橘色柱子排除設定。
全圖基準為 imgsz=640、conf=0.15、IoU=0.7；局部推論尺寸為 960。
針對兩區各做一個背景裁切與四個重疊裁切，各自測試原圖、Gamma 與 CLAHE。整張圖片共推論 33 次。
各版本保留既有三視圖遮罩，只補入實際模型候選，再以同類遮罩 IoU 去重；不膨脹遮罩，也不把 XML 框轉成填色。

區域座標是正規化 xyxy：左 MSC `(0.03,0.20,0.54,0.90)`，右下 `(0.64,0.58,0.85,0.94)`。
XML 的區域統計按標記框中心分組：左 MSC 有 22 個 D，右下有 2 個 D、11 個 R。
局部推論保留規則則要求候選至少五個遮罩像素、至少半數面積落在該區。
換圖片時需用 `--left-roi`、`--right-roi`、`--left-context`、`--right-context` 指定合適範圍。

| 版本 | 左 MSC D 命中／標記 | 右下 D | 右下 R | 全圖 D 命中／預測 | 全圖 R 命中／預測 |
|---|---:|---:|---:|---:|---:|
| R02 既有三視圖 | 2/22 | 0/2 | 0/11 | 2/18 | 8/47 |
| R03 加局部背景裁切，相同門檻 | 2/22 | 0/2 | 0/11 | 2/29 | 12/70 |
| R04 加重疊裁切，相同門檻 | 2/22 | 0/2 | 0/11 | 2/43 | 13/83 |
| R05 局部 D≥0.08、R≥0.10 | 3/22 | 0/2 | 0/11 | 3/56 | 13/96 |
| R06 局部 D≥0.05、R≥0.08 | 4/22 | 0/2 | 0/11 | 4/73 | 14/102 |

新增候選很多，但真正符合標記的增加有限。R06 的 D 未匹配預測為 69 個、R 為 88 個，可能包含誤檢、重複或定位不夠準。
右下即使將比對 IoU 放寬至 0.3，D 仍是 0/2、R 是 1/11，不能只把問題歸因於 0.5 的匹配門檻。
右下部分預測將較大面積陰影當成凹陷，並沒有逐一定位標記中的損傷。因此新程式是診斷工具，沒有取代原先推論預設。
上述參數與裁切是針對此圖調整的回測結果，不能當成未見影像或獨立驗證集的準確率。

## 輸出

- 六版推論：`runs/segment_shadow/recall_tests/run_20261001_101515_915036/`。
- 前次 25 版與本次 6 版的標記比對：`runs/segment_shadow/annotation_evaluation/run_20261001_101612_958101/`。
- `summary.csv`：全圖與兩區的命中、漏檢、precision/recall。
- `missed_annotations.csv`：未命中的標記 ID、座標、同類最高 IoU 與預測分數。
- 每版 `*__all.jpg`：純塗色完整結果；`*__additions.jpg`：只有本版新增候選的塗色。
- `comparisons/left_D.png`、`right_all.png`：固定區域比較。
- [指定標記的原圖／既有結果／局部診斷比較](../runs/segment_shadow/annotation_evaluation/run_20261001_101612_958101/annotated_targets.png)：從 XML 定位裁切窗口；沒有將 XML 框塗成預測。

## 重跑

```powershell
& 'C:\Users\USER\anaconda3\envs\yolov11\python.exe' tools/predict_shadow_recall.py --annotation 'C:\Users\USER\Desktop\0928-1\CBHU0708054-A.xml'
```

推論後自動輸出框比對。若只評估已有結果，不必重新推論：

```powershell
& 'C:\Users\USER\anaconda3\envs\yolov11\python.exe' tools/evaluate_shadow_annotations.py --annotation 'C:\Users\USER\Desktop\0928-1\CBHU0708054-A.xml' --run-dir '既有結果資料夾'
```

模型權重與原始 XML 均未修改。下一步應檢查這些漏檢標記在訓練資料中的類別、框／遮罩範圍是否一致，再用獨立影像驗證訓練或模型變更。

## 驗證

共 30 項相關 unittest 通過，涵蓋裁切映射、完整覆蓋、保留既有遮罩、柱子排除、一對一且同類別的框匹配及 VOC 座標转换。
6 個新版本與既有 25 版均已完成實際推論／標記比對；本次六版的基準保持既有 D/R 數量與遮罩。
也已讀取權重資訊：task=segment，D 類別 ID=2、R=5、訓練 imgsz=640；類別索引沒有對調。
