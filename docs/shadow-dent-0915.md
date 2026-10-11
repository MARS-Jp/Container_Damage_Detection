# 0915 陰影／低對比凹洞模型候選版

> 2026-09-18 狀態：使用者回報後兩版結果不理想，目前回到 0912 作為使用基準。本頁與模型保留作實驗紀錄，不再作為目前建議版本；請見 [0912 回退說明](rollback-0912.md)。

新增模型：`ultralytics/cfg/models/11_myself/yolo11-10csar-dr-shadow-0915.yaml`。

這版根據使用者提供的 `last0912.pt` 實際架構修改，目的為改善陰影、低對比與缺少鏽色提示時的 D（凹洞）召回率。已完成程式與權重相容性驗證，尚未重新訓練或證明實際召回率提高。

## 確認過的原模型

- checkpoint 內 YAML：`yolo11-10csar-dr-msat-0911.yaml`，nano 寬深度。
- 類別：`0:B, 1:C, 2:D, 3:H, 4:O, 5:R, 6:X, 7:U`。
- 架構：RGB/IIM → backbone → 5 CSAR → 5 FSNetShuffle → 5 CSAR → 5 MSATMultiLabel → Segment26MultiLabel。
- 保存的參數：`epochs=250, batch=8, imgsz=640, amp=False, overlap_mask=False, mask_ratio=4, optimizer=auto`。
- 所有權重／buffer 均為有限值，未發現 NaN/Inf。
- `epoch=-1` 且沒有 optimizer，屬於已剝除續訓狀態的 checkpoint；用它初始化新模型時須 `resume=False`。
- 保存的全類別 Mask recall 為 `0.03295`、Mask mAP50 為 `0.02054`。這是 checkpoint 中的歷史驗證結果，並非本次重新驗證，也不是 D 單一類別或這張圖片的 recall。

## 架構修改

| 位置       | 修改                                                | 目的                                                                   |
| ---------- | --------------------------------------------------- | ---------------------------------------------------------------------- |
| 第 0 層    | `IIMStem` → `ShadowIIMStem`                         | 保留 RGB/IIM，新增局部亮度、細／粗尺度對比、水平／垂直有正負號梯度分支 |
| 第 46 層   | `Segment26MultiLabel` → `Segment26MultiLabelShadow` | 增加 backbone 第 6 層 P3、第 3 層 P2 的直接輸入，在預測前補回淺層細節  |
| 新分支融合 | 小幅初始化的可學習 residual gain 與空間 gate        | 讓原模型提供主體特徵，由訓練決定新增形狀線索的使用程度                 |

總 YAML 層數仍為 47。第 1～45 層保持原架構；P3/P2/P4/P5/P6 輸出順序不變，prototype mask 仍為影像的 1/4 解析度。原本的多標籤 D/R 共存監督繼續使用，沒有設定「必須有 R 才能預測 D」。

### 為什麼補亮度而不只加注意力

本專案 IIM 的 RGB 對數通道差會抵消共同的亮度變化。對相同灰階的 RGB 輸入，R/G/B 通道差可為零，但局部明暗仍可能提供形狀線索；原 RGB 分支仍保留這些訊號，不能據此斷言 IIM 就是漏檢原因。

新分支顯式保留這類訊號：局部均值採 5×5／15×15，梯度採 signed Sobel；比值使用分母下限與 tanh，統計運算用 FP32。邊界使用複製填補，避免平坦影像四周出現人工邊緣。這些都是可供網路學習的影像特徵，並非直接量測深度，也不能單憑梯度分辨凹洞、正常浪板和投影。

P2/P3 的細節旁路另外提供淺層特徵，以及 3×3／7×7 局部差分；在分割頭內完成融合後，box、分類、mask coefficient 與 prototype 都能取得更新後的特徵。

YOLA 的光照不變特徵原理可參閱 [YOLA／NeurIPS 2024](https://arxiv.org/abs/2410.18398)。新分支是本專案的待驗證設計，並非該論文已驗證過的凹洞偵測方法。

## 權重與測試

- 實際載入 `last0912.pt`：原有 **2102 個 state tensors 全數相容，載入後值一致**。新增的 59 個 state entries 使用初始化值；原分割頭沒有因層索引改動而丟失。
- nc=8、nano：參數由 **11,927,937** 增至 **11,951,015**，新增 **23,078（約 0.19%）**。這個比例只描述參數量，不等同於記憶體或延遲比例。
- CPU／PyTorch 2.6：10 項新測試通過，涵蓋黑色與均勻表面、低亮度形狀特徵、半精度描述子、舊 stem 相容性、細節旁路梯度、完整模型 loss/backward、D/R 重疊 masks、EMA deepcopy 與非正方形輸入。
- 原 IIM／MSATMultiLabel 的 6 項迴歸測試通過。
- 使用實際 checkpoint 初始化新版後，低亮度合成輸入的預測與 prototype 都是有限值。
- 尚未驗證 CUDA AMP、長時間訓練穩定性或真實影像上的效果；沿用本次權重的 `amp=False`。

重跑測試：

```bash
python -m pytest tests/test_shadow_dent.py tests/test_yolo11_6csar_iim.py tests/test_msat_multilabel.py --noconftest -o addopts= -q
```

## 在訓練主機使用

請同步新 YAML、`ultralytics/nn/modules/shadow_dent.py`，以及修改過的 `ultralytics/nn/modules/__init__.py` 和 `ultralytics/nn/tasks.py`。只複製 YAML 不足以解析新模組。

以下範例沿用 checkpoint 的主要訓練設定。將 `last0912.pt` 和 `data.yaml` 換成主機上的實際路徑，其餘資料增強／排程設定也應與原訓練一致，以利比較：

```bash
yolo segment train \
  model=ultralytics/cfg/models/11_myself/yolo11-10csar-dr-shadow-0915.yaml \
  pretrained=/path/to/last0912.pt \
  data=data.yaml \
  epochs=250 batch=8 imgsz=640 \
  optimizer=auto amp=False overlap_mask=False mask_ratio=4 \
  resume=False name=yolo11-dr-shadow-0915
```

先用短程 smoke run 確認新模組可載入且 loss 有限；正式效果比較時使用相同的訓練預算與實際 optimizer。`optimizer=auto` 的選擇會受總步數影響，短跑不能直接視為完整訓練的對照。

請用 `model=新YAML pretrained=last0912.pt`，不要只指定 `model=last0912.pt`；後者仍會建立舊架構。僅載入舊權重而不訓練新增分支，也不能視為完成改進。

## 效果驗證重點

比較相同切分、相同 conf／NMS 下的陰影 D、亮部 D、D 與 R 共存、D 單獨出現四組召回率，同時查看 R 精度與正常浪板／純陰影的誤報。將漏掉的陰影 D 正確標註，並保留正常浪板與沒有凹洞的投影作負樣本；不能把所有暗處當成 D。

本次只有預測疊圖，沒有逐處的凹洞真值，因此無法從這張圖量化漏檢數。若還有低信心候選遭閥值過濾，應另做 conf sweep；調低門檻帶來的召回率改變須與架構改進分開比較。

原有 multi-label 參數也保持不變：`cooccurrence_weight=2.0` 在目前 loss 中代表 `1 + 2 × overlap`，所以重疊像素的 BCE 權重為 3，再經總權重正規化。
