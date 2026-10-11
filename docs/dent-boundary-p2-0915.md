# 0915 架構整合版：陰影細節＋損傷邊界＋原生 P2 遮罩

> 2026-09-18 狀態：使用者回報後兩版結果不理想，目前回到 0912 作為使用基準。主版本與 DCNv2 版均保留作實驗紀錄，下列建議僅為當時的實驗安排；目前請使用 [0912 基準版本](rollback-0912.md)。

這版將上一版的亮度／細節改動與本次的架構改動合併，保留舊 YAML 供對照，沒有覆蓋 0911 或 0915 shadow 版。目的為改善陰影、低對比、小型及不規則 D（凹洞）的辨識；程式驗證通過不代表真實資料召回率已改善，仍須重新訓練與驗證。

## 先使用哪個檔案

- 主版本：`ultralytics/cfg/models/11_myself/yolo11-10csar-dr-boundary-p2-0915.yaml`。
- 可選 DCNv2 版本：`ultralytics/cfg/models/11_myself/yolo11-10csar-dr-boundary-p2-dcn-0915.yaml`。
- 上一版對照：`ultralytics/cfg/models/11_myself/yolo11-10csar-dr-shadow-0915.yaml`。

兩個新 YAML 的解析內容只差最後一個 `use_dcn` 布林參數。建議先測主版本，再以相同資料、初始化及訓練預算測 DCNv2，才能分辨額外採樣分支的效果。DCNv2 版不是已確認較好的版本。

## 修改及新增內容

| 位置            | 改動                                        | 用途與限制                                                                             |
| --------------- | ------------------------------------------- | -------------------------------------------------------------------------------------- |
| 第 0 層         | 沿用 `ShadowIIMStem`                        | RGB／IIM 之外保留局部亮度、細／粗對比、有正負號的梯度；不是把暗處直接判成凹洞          |
| 第 1～45 層     | 保持 0915 shadow 版不變                     | 保留 backbone、5 CSAR → 5 FSNetShuffle → 5 CSAR，以及 5 個 MSATMultiLabel 分支         |
| 第 46 層        | 改為 `Segment26MultiLabelBoundary`          | 仍接收原來的五尺度特徵、五尺度類別 logits，以及 backbone P3／P2 細節旁路               |
| P3／P2 細節融合 | 新增 P4 語意引導及邊界預測                  | 利用深層上下文決定淺層形狀細節的使用程度，目的是降低正常浪板／雜紋干擾；效果需實驗驗證 |
| 邊界監督        | 新增 `v8BoundaryMultiLabelSegmentationLoss` | 從每個損傷實例的 mask 產生類別邊界，不需另外標註邊界                                   |
| 遮罩解碼        | 新增 `NativeP2Proto`                        | 將各尺度直接融合到 P2，避免 P2 細節先縮至 P3，再經轉置卷積放大                         |
| DCNv2 可選分支  | 新增 `ModulatedShapeConv`                   | 在 P3／P2 形狀分支加入可學習位移及調制採樣，主版本預設關閉                             |

總 YAML 層數仍是 47。相對實際的 `last0912.pt`／0911 架構，只改第 0、46 層；相對上一版 0915 shadow，只改第 46 層。偵測輸入順序保持 P3／P2／P4／P5／P6，stride 仍為 `[8, 4, 16, 32, 64]`。prototype mask 仍為 stride 4：640×640 輸入對應 160×160，不是改成 stride 2。

### 語意與邊界如何影響預測

P2／P3 的淺層特徵先產生細／粗局部差分，再與 P4 上下文共同預測每一類的損傷邊界。融合採用：

```text
輸出 = 原 neck 特徵 + tanh(可學習 detail_scale) × 語意 gate × 邊界 gate × 形狀細節
```

`detail_scale` 初始為 0.1；邊界 gate 是 `0.5 + 0.5 × max(sigmoid(各類邊界 logits))`，有 0.5 的下限，不做硬式篩選。語意 gate 來自 neck、淺層與 P4 上下文，沒有使用「R 必須存在」的判定。這些融合後特徵同時送往 box、分類、mask coefficient 與 prototype，不只是多畫一張與預測無關的邊界圖。

### 邊界標籤與 loss

1. 在現有 mask 解析度上，對**每個實例**做 3×3 膨脹減侵蝕，產生內外側邊界帶。
2. 再依圖片、類別合併邊界，縮小時採 max-preserving pooling；順序不能相反，否則同類別相鄰實例的交界可能消失。
3. P3／P2 兩個邊界分支分別接受正樣本加權 BCE 與 class-wise Dice 監督，再取平均乘上 `boundary_gain=0.2`。BCE 正樣本權重按每類前景比例計算並限制於 1～20，經總像素權重正規化。

監督適用於資料集**所有類別**，包含 D／R，不把 D=2、R=5 寫死在模組中。D 與 R 重疊時各自保留，同一區域可以同時有兩種邊界；只有 D、沒有 R 的樣本也正常監督。這是損傷標註的邊界，不是用影像中的所有陰影、文字或浪板邊緣當真值。

沿用現有七個 loss 欄位：`sem_loss` 現在合計「原 MSAT multi-label loss + 新邊界 loss」，不新增第八欄。新舊 `sem_loss` 數值不能直接當成同一目標比較。舊 heatmap／seedmap 分支保留，特徵解析度改為 P2，loss 與目標的相容流程不變。

仍然必須設定 `overlap_mask=False`，否則合併實例遮罩會破壞獨立監督，原本的 ValueError 會繼續阻止不相容訓練。這個檢查沒有移除。

### 原生 P2 解碼與權重相容性

舊版把 P2 投影後縮至 P3，與其他尺度融合，再用轉置卷積回到 P2。新版保留 P2 原解析度，將 P3／P4／P5／P6 對齊至 P2，保留原通道投影與細化卷積，將內部 `upsample` 換為 `Identity`，避免輸出意外升至 P1。

相同權重的運算解析度已改變，因此不是與舊模型數值等價的替換。多個卷積會處理較大的特徵圖，參數量小幅變動**不代表**顯存或延遲也只小幅變動；尚未量測 CUDA 峰值顯存與速度。

## 已完成驗證

CPU 環境：PyTorch 2.6.0、torchvision 0.21.0。以下參數量均為 nano、nc=8：

| 版本                  |     參數量 | 相對原 0911 增量 |
| --------------------- | ---------: | ---------------: |
| 原 `last0912.pt` 架構 | 11,927,937 |                — |
| 上一版 0915 shadow    | 11,951,015 |          +23,078 |
| 本次邊界／P2 主版本   | 11,948,985 |          +21,048 |
| 本次邊界／P2＋DCNv2   | 11,973,425 |          +45,488 |

移除舊轉置卷積的參數抵銷了一部分新增分支，因此主版的參數略少於上一版，但不能據此推論更快。

- 實際載入 `last0912.pt`：兩版皆可沿用原 2102 個 state tensors 中的 **2100 個**，並逐項驗證載入值一致。
- 兩個未沿用項目為 `model.46.proto.upsample.weight`、`model.46.proto.upsample.bias`，因為該轉置卷積已移除。
- 主版共有 2191 個 state entries、DCNv2 版 2209 個；新增項目需訓練。也測試了從上一版 0915 shadow 沿用權重，除上述兩項外原有 shapes 均相容。
- 15 項新版測試通過，涵蓋重疊 D／R、同類別相鄰實例、D 單獨出現、空標註、P2 細節保留、DCNv2 offset／mask 梯度、完整 loss/backward、驗證 loss、EMA deepcopy、非正方形推論與 Conv-BN 融合。
- 舊 0915 shadow、IIM、MSATMultiLabel 的 16 項迴歸測試通過，合計 **31 項通過**。
- 實際 checkpoint 初始化後，兩版的低亮度合成輸入預測與 prototype 均為有限值。

重跑測試（需已安裝相容的 torch／torchvision／pytest 與專案相依套件）：

```bash
python -m pytest tests/test_dent_boundary.py tests/test_shadow_dent.py tests/test_msat_multilabel.py tests/test_yolo11_6csar_iim.py --noconftest -o addopts= -q
```

尚未在真實訓練資料上重新訓練或計算 D 的 recall／AP，沒有驗證 CUDA／多 GPU／長訓練穩定性。測試只確認 export 分支的回傳結構，**沒有驗證 ONNX／TensorRT 匯出或部署**；DCNv2 原生運算尤其需另外確認部署支援。

## 訓練主機要同步哪些檔案

只複製 YAML 不夠。請一起同步：

```text
ultralytics/cfg/models/11_myself/yolo11-10csar-dr-boundary-p2-0915.yaml
ultralytics/cfg/models/11_myself/yolo11-10csar-dr-boundary-p2-dcn-0915.yaml
ultralytics/nn/modules/shadow_dent.py             # 上一版的必要依賴
ultralytics/nn/modules/dent_boundary.py           # 本次新增
ultralytics/utils/dent_boundary_loss.py           # 本次新增
ultralytics/nn/modules/__init__.py                # 新 head 匯出
ultralytics/nn/tasks.py                           # YAML 解析與新 criterion 註冊
```

如果遠端同一檔案也有其他修改，請合併本次差異，不要直接覆蓋。`utils/loss.py` 需保留原有的 `v8MultiLabelSegmentationLoss`，本次沒有修改它。

在訓練主機的專案根目錄使用（`data.yaml` 與 checkpoint 請換成實際路徑）：

```bash
yolo segment train \
  model=ultralytics/cfg/models/11_myself/yolo11-10csar-dr-boundary-p2-0915.yaml \
  pretrained=/path/to/last0912.pt \
  data=/path/to/data.yaml \
  epochs=250 batch=8 imgsz=640 \
  optimizer=auto amp=False overlap_mask=False mask_ratio=4 \
  resume=False name=yolo11-dr-boundary-p2-0915
```

DCNv2 實驗只需改模型路徑為 `yolo11-10csar-dr-boundary-p2-dcn-0915.yaml`，並改用不同的輸出名稱，例如 `name=yolo11-dr-boundary-p2-dcn-0915`。需要 torchvision 原生 deform-conv 運算與相符的 PyTorch／CUDA build；沒有支援時會明確報錯，不會默默改回普通卷積。DCNv2 僅加在形狀分支，採 depthwise 權重、一組 18 通道 offset、9 通道 sigmoid modulation，採樣用 FP32 後回到輸入 dtype。

這些訓練參數是沿用既有 checkpoint 的比較起點，不是本次改進重點。先短跑確認主機可用，再使用一致的正式預算；`optimizer=auto` 會隨總步數改變選擇，短跑不是正式訓練的直接對照。P2 計算增加，若顯存不足可先降低 batch，但對照實驗也應記錄此差異。

`last0912.pt` 已剝除 optimizer／續訓狀態，應用於初始化新架構，設定 `resume=False`。不要只寫 `model=last0912.pt`，那會繼續使用舊架構；載入後也必須訓練新增分支。若使用已訓練的 0915 shadow 權重初始化整合版，請將該 warm-start 條件另記，避免與直接從 0912 初始化的模型混為同一比較。

## 效果比較

可同時準備／分開訓練上一版與本版，但每個實驗需用不同 `name`／輸出目錄、不要共同覆寫 checkpoint；同 GPU 並行還需自行確認顯存足夠。建議對照原 0911、0915 shadow、整合主版、整合 DCNv2 版，在相同資料切分、conf／NMS 與訓練條件下比較。

主要看陰影 D、亮部 D、D/R 共存、D 單獨出現的 mask recall／AP，以及正常浪板、純投影的誤報、R 的 precision。邊界分支依賴正確的實例標註，漏標凹洞或過粗多邊形會限制效果；不能靠架構保證找回所有未標出的 D。只有推論疊圖，沒有真值與原始影像，無法量化本次真實改善。

## 研究依據與本次實作界線

- [Gated-SCNN（ICCV 2019）](https://arxiv.org/abs/1907.05740) 提供高層語意引導淺層形狀分支的方向；本次是接在既有 YOLO 特徵上的輕量化自訂設計，並非完整移植。
- [PIDNet（CVPR 2023）](https://arxiv.org/abs/2206.02066) 提供細節、上下文及邊界協作的設計參考，並未直接採用 PIDNet backbone。
- [torchvision DCNv2 文件](https://docs.pytorch.org/vision/0.21/generated/torchvision.ops.deform_conv2d.html) 定義帶 modulation mask 的 deformable convolution；本次可選分支實際呼叫此運算，不是僅以普通卷積命名成 DCN。

上述研究與文件支持設計原理和運算用法，不代表已證實能改善這個貨櫃凹洞資料集。本次也沒有宣稱實作完整 FreqFusion。
