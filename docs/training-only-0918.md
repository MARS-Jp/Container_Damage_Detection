# 0918：主架構固定的訓練機制實驗

依最新要求，以已回退的 **0912** 作為基準，維持 YOLO11、AttentionResiduals、FSNetShuffle、CSAR 主架構。使用者允許推論端擴充，但本輪先採不需要新增推論模組的設計：七版的完整 backbone/head、參數形狀、推論計算圖均與 0912 相同，改動只在訓練階段。

這不是重新啟用結果不理想的 0915 shadow／boundary 架構；不使用 `ShadowIIMStem`、`Segment26MultiLabelBoundary`、NativeP2Proto 或 DCNv2。原推論預設仍是 `last0912.pt`，不會因新增實驗 YAML 自動切換。

## 七個新 YAML

所有檔案位於 `ultralytics/cfg/models/11_myself/`，共同前綴是 `yolo11-10csar-dr-trainonly-0918-`。

| YAML 後綴              | 成對原圖／陰影圖監督 | 一致性 | 實例邊界 | 新增重疊約束 |
| ---------------------- | :------------------: | :----: | :------: | :----------: |
| `baseline.yaml`        |          —           |   —    |    —     |      —       |
| `shadow-aug.yaml`      |          ✓           |   —    |    —     |      —       |
| `consistency.yaml`     |          ✓           |   ✓    |    —     |      —       |
| `boundary.yaml`        |          —           |   —    |    ✓     |      —       |
| `overlap.yaml`         |          —           |   —    |    —     |      ✓       |
| `shadow-boundary.yaml` |          ✓           |   ✓    |    ✓     |      —       |
| `full.yaml`            |          ✓           |   ✓    |    ✓     |      ✓       |

這是**獨立機制與組合實驗**，不是每一列都累加上一列。`consistency` 與 `shadow-aug` 使用完全相同的雙視角監督，兩者只差一致性 loss；`shadow-boundary` 則是使用者優先提出的 Shadow Consistency + Boundary 組合。建議先比較 `baseline`、`shadow-aug`、`consistency`、`boundary`、`shadow-boundary`，再比較額外 overlap 與 full；此順序不代表預先知道哪版最好。

原 0912 的 YAML 名稱是 `yolo11-10csar-dr-msat-0911.yaml`。七版保留其中所有 47 層及原有訓練目標，因此 baseline 的「無」是**沒有本輪新增機制**，不是移除舊 MSATMultiLabel 或原共存加權。

## 新增機制

### 1. Shadow Augmentation：幾何與標註不變

對每張已經完成原有資料增強的 RGB 影像，產生隨機方向、軟邊的半平面陰影。三個 RGB 通道使用相同比例衰減，不移動像素，不改 mask／bbox／heatmap／seedmap，也不將真凹洞改成背景。

- 最深衰減比例從 `[0.15, 0.45]` 取樣，亮度乘數最低約 0.55，不是把區域塗黑。
- `probability=1.0`、`softness=0.15` 是本輪固定起點，不是已驗證最優值。
- 原圖與陰影圖串接成 `2B` 個 view，進行一次共享模型 forward，再分開計算完整 GT loss 並取平均：`0.5 × (L_original + L_shadow)`。
- 不修改 batch 內原始影像和標註；BatchNorm 一次 forward 處理兩種 view，不維護另一套 teacher 網路。

CLI `batch=8` 表示 8 張原始圖片，但啟用陰影時模型會處理 16 個 view。相較 baseline，影像曝光數、BatchNorm 統計及訓練計算量都會改變，不能宣稱只有多一個 loss、所有訓練因素完全相同。`shadow-aug` 是辨別「增強本身」與「一致性約束」的必要對照。

### 2. Illumination Consistency：不複製原圖的錯誤漏檢

對現有五尺度 MSAT 類別 logits 的 sigmoid 機率做一致性約束，不比較 prototype 的任意基底，也不嘗試對齊 NMS 後數量不一致的 instances。

原圖輸出作為 stop-gradient teacher，陰影圖作為 student。只有 teacher 信心至少 0.7，且判斷與 GT 的獨立類別 mask 一致的位置，才納入平方機率差。前景與背景分別正規化，再平均有有效像素的組別，避免背景數量主導。

teacher 對 GT 凹洞給出高信心背景時，該位置不會因一致性被迫繼續漏檢；兩個 view 仍有完整 GT 監督。teacher 與 student 是同一模型的兩個 view，不是 EMA teacher，也不是本次另外實作的 contrastive learning。初期 teacher 若不夠可靠，一致性項可能為零，原監督仍正常運作。

固定起點：`consistency.gain=0.2`。跨尺度平均後加至 `sem_loss`；沒有任何有效位置時回傳可反傳的零。

### 3. Boundary-aware Loss：直接監督真正的實例遮罩

不新增邊界 head。沿用 YOLO 的正樣本指派，由現有 mask coefficients 與 prototypes 組合出 matched-instance mask logits：

```text
L_boundary = 0.5 × (soft-boundary Dice + GT-boundary-band mask BCE)
```

soft boundary 使用可微分的 3×3 膨脹減侵蝕，GT 邊界也從**每個實例** mask 產生。先計算邊界，再在外擴一個 mask pixel 的匹配 bbox 中比較，避免先裁切而製造假邊界。邊界帶 BCE 使初期接近平坦的預測仍有學習訊號。圖框採 replicate padding，不將均勻表面的外框人工當成損傷邊界。

這會直接更新主要 mask coefficients、prototypes 及其上游特徵，並非只訓練一張與最終遮罩無關的 edge map。同類別相接實例仍各自監督；D／R 重疊也保持獨立。此項適用於所有標註損傷類別，沒有只把 D／H 寫死。

固定起點：`boundary.gain=0.1`。這個 gain 作用於每個 matched-mask 的額外 loss，然後沿用原正 anchor 數量正規化及 `hyp.box` 的 segmentation gain，合計至 `seg_loss`。它不是以影像梯度或陰影邊緣當真值；標註若過粗或漏標，邊界監督也可能強化錯誤。

### 4. Additional Overlap Constraint：原有加權之外的真交集目標

0912 已使用獨立 mask、多標籤 BCE／Dice，並對多類別共存像素加權；原 `cooccurrence_weight=2.0` 對應 `1 + 2 = 3` 的像素 BCE 權重。本次所有版本保留它，不能將新 overlap 版宣稱為首次允許 D／R 重疊。

新增項另外監督一個類別對的 joint-probability **score**：`p_D × p_R`，其 GT 是兩類原始 mask 的真正交集。這個乘積是本次採用的 score，不宣稱是校準過的聯合機率估計。

- 先在目前 GT mask 原解析度（通常 stride 4）計算交集與聯集，再做 presence pooling，避免相鄰但不重疊的 D／R 因縮小到同一格而變成假交集。
- 只對含真交集的圖片啟用，在該類別對聯集內監督交集／非交集，分別正規化兩種區域。
- 沒有真交集的圖片新增 loss 精確為零；D-only 樣本不會被要求長出 R。
- 用穩定 joint-logit `a + b - logsumexp(0, a, b)` 搭配 BCE，而不是不穩定的機率連乘後取 log。

固定起點：`overlap.gain=0.2`，五尺度平均後加至 `sem_loss`。`pairs: [[D, R]]` 依資料集的 names 解析，不固定假設 D=2、R=5；若資料集名稱不同，請改成實際名稱或正確的整數 ID，找不到類別會明確報錯。

## 不採用的機制

- 不使用「damage 與 shadow 一重疊就懲罰」：真凹洞本來就可能在陰影中，這會與找回陰影 D 的目標衝突；目前也沒有 shadow-mask 標註。
- 沒有額外加 depth head、contrastive embedding、hard-negative 自動標註或推論形狀過濾。這些不是本輪七個 YAML 的隱藏功能，尤其不能把模型未預測到的損傷直接當負樣本。
- 不修改正常浪板／接縫的標籤，也不加入要求 D 必須伴隨 R 的 gate。

## 正常 YOLO 指令即可訓練

先同步七個 YAML，以及這次新增／修改的三個程式檔：

```text
ultralytics/utils/training_aux_ops.py       # 新增：無參數的陰影／邊界／一致性／交集運算
ultralytics/utils/training_aux_0918.py      # 新增：設定檢查、loss 與雙視角訓練流程
ultralytics/nn/tasks.py                    # 修改：依 YAML 選擇訓練期機制；舊 YAML 路徑不變
```

不需要另執行 `.py`。但只複製 YAML 到未更新的程式可能仍只跑原 loss，不能視為啟用本輪機制。初始化 criterion 時會印出 `0918 training-only constraints: ...`，列出 shadow 與各 gain；請確認實際使用這個專案的 Python 環境。

在 Linux 訓練主機的專案根目錄執行（資料／權重路徑請替換）：

```bash
yolo segment train \
  model=ultralytics/cfg/models/11_myself/yolo11-10csar-dr-trainonly-0918-shadow-boundary.yaml \
  pretrained=/path/to/last0912.pt \
  data=/path/to/data.yaml \
  epochs=250 batch=8 imgsz=640 \
  optimizer=auto amp=False overlap_mask=False mask_ratio=4 \
  compile=False resume=False name=yolo11-dr-0918-shadow-boundary
```

其他實驗只需換 YAML 後綴及 `name`，不要共同覆寫同一輸出目錄。這些原訓練參數沿用先前 checkpoint 的起點，不是本輪改良來源；請保持實際 optimizer、學習率、資料切分、seed、epoch、batch／累積與增強條件一致，並記錄雙視角的額外曝光及計算成本。`optimizer=auto` 可能隨總步數選擇不同 optimizer，不應用短跑直接當正式對照。

必要限制：

- `overlap_mask=False`：必須保留獨立實例，原 0912 與本輪皆需要。
- `compile=False`：雙視角訓練在 `model(batch)` 中組合 view，不能先以 compile 路徑預算單一 view 的 predictions；不相容時會明確報錯。
- `resume=False`：提供的 `last0912.pt` 沒有完整續訓狀態。七個實驗都應從同一份 0912 初始化，避免從前一個實驗接著訓練造成混淆。
- `amp=False`：沿用先前設定；局部運算有 FP32 保護，但尚未完成 CUDA AMP／多 GPU 長訓練驗證。
- 陰影版本處理 `2B` 個 view，顯存可能明顯增加。普通單視角的 AutoBatch 記憶體估計不能直接當成成對訓練的保證；先以明確 batch 做短程檢查，若調整則所有相關比較都須記錄。

新增 loss gain、陰影強度、confidence 門檻本身仍然是超參數。論文可寫「不以變更原 optimizer／LR／batch／epochs 為改良方式」，不能寫成本方法完全沒有超參數。

## 推論與驗證

訓練完成後用該次 run 的 `best.pt`／`last.pt` 正常推論，不能只載入原 `last0912.pt` 然後期待新訓練機制已生效。驗證與推論不產生合成陰影、不跑雙視角，也不新增 boundary head 或後處理。

七版的參數及 state key/shape 與 0912 一樣；nano、nc=8 都是 **11,927,937 個參數、2102 個 state tensors**。新增訓練策略不改變推論圖，不需要測試時移除某個分支。仍未量測 CUDA 實際延遲，不以此宣稱所有硬體／精度設定下毫秒數必然完全一樣。

loss 欄位維持七項：訓練 `seg_loss` 可含 boundary，`sem_loss` 可含 overlap／consistency，其餘原有項目保留。**validation loss 使用原始 baseline 目標**，不含本輪新正規化項；因此不同配置的 training loss 不適合直接比大小，主要應比較固定驗證條件下的 D/R 指標。

## 測試與效果界線

本機 CPU／PyTorch 2.6.0：本輪 40 項測試與前版 31 項迴歸測試，合計 **71 項通過**。

新增測試位於 `tests/test_training_aux_0918.py`，涵蓋七個 YAML 的圖結構／state shapes、完整 loss/backward、全關閉時精確比對 baseline、驗證無陰影、GT 不被修改、空 masks、EMA deepcopy、FP16 局部運算、teacher 梯度隔離、相鄰假交集、D-only，以及設定錯字／類別名稱檢查。

實際使用 `last0912.pt` 嚴格載入七版並逐項核對，2102 個 state tensors 全數吻合；也確認一般 `YOLO(YAML)` 判定為 segment、標準 `SegmentationTrainer.get_model` 可建立新配置且選取新 criterion，以及 checkpoint 記憶體內序列化／重新載入後可做有限值推論。這不是完整 trainer epoch／資料載入／CUDA 的端到端測試。

重跑本輪與前版迴歸測試：

```bash
python -m pytest tests/test_training_aux_0918.py tests/test_shadow_dent.py tests/test_dent_boundary.py tests/test_msat_multilabel.py tests/test_yolo11_6csar_iim.py --noconftest -o addopts= -q
```

這些是 CPU／合成資料的程式驗證，不是改善召回率的證據。尚未在完整貨櫃資料集完成這七輪正式訓練。請分別比較陰影 D、亮部 D、D-only、D+R 的 recall／AP，同時查看 R precision、正常浪板／純陰影的誤報；不要只有整體 mAP 或少量疊圖。

## 研究來源與實作界線

- [Araslanov & Roth，CVPR 2021](https://arxiv.org/abs/2105.00097)：分割中的增強一致性提供概念依據；本輪是有 GT 的 paired-view regularization，不是該論文的完整 domain-adaptation 方法。
- [Bokhovkin & Burnaev，2019](https://arxiv.org/abs/1905.07852) 與 [Kervadec 等，MIDL 2019](https://proceedings.mlr.press/v102/kervadec19a.html)：邊界與區域損失互補的研究背景；本次 morphology Dice＋邊界帶 BCE 不是原 distance-map loss 的重現。
- [Albumentations RandomShadow 文件](https://albumentations.ai/docs/api-reference/albumentations/augmentations/pixel/weather/)：影像層級陰影增強的參考。本輪用純 PyTorch 實作，不新增 Albumentations 相依。

這些來源不代表已證實能改善本資料集，也不足以單憑組合名稱宣稱論文新穎性或保證可發表。
