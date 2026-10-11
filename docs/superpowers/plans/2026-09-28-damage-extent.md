# Damage Extent Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 提供以完整損壞範圍為目標的訓練 loss、標記核對及六組依序編號 YAML。

**Architecture:** 保留目前 6-CSAR semantic-token 的推論圖，重用既有陰影／一致性／邊界訓練管線。新增直接作用於 matched instance mask 的範圍 loss，搭配唯讀標記核對與一致的訓練 recipes。

**Tech Stack:** Python、PyTorch、Ultralytics 本機 fork、PyYAML、NumPy、SciPy、pytest。

**Spec:** `docs/superpowers/specs/2026-09-28-damage-extent.md`

## Global Constraints

- 保留 yolo11-6csar-semantic-token.yaml 的 backbone、head、P3/P2/P4/P5 順序與推論參數形狀。
- 保留獨立 D/R 實例及原多標籤監督；overlap_mask=False、mask_ratio=4、compile=False、resume=False。
- 不覆寫原始影像、XML、現有分割標記、權重與 tools/predict-2.py。
- XML 框不自動轉成精細遮罩；範圍 loss 需要已核對的實例 mask。
- 新 loss 僅作用於訓練；舊 YAML 預設關閉；驗證保持原 loss。
- 不新增外部依賴，不自動啟動完整訓練，不修改既有推論門檻。

執行方式：使用者已要求直接更改，故本任務內逐項實作，不再次要求執行許可。本機沒有 executing-plans/subagent-driven-development，按本計畫自行驗證；不派遣子代理。保留修改供使用者檢視，不在 .git 唯讀權限下要求額外提交許可。

---

## 檔案責任

追加需求：`tools/predict_seg_by_class.py` 預設 `hide_labels=True`，原 `--hide-labels` 相容，另提供互斥的 `--show-scores` 明確開啟顯示；不修改 confidence 門檻。以原圖重跑一次確認無數值的輸出。

| 路徑                                                               | 責任                         |
| ------------------------------------------------------------------ | ---------------------------- |
| `ultralytics/utils/damage_extent.py`                               | 無參數的範圍 loss            |
| `ultralytics/utils/training_aux_0918.py`                           | 設定解析及接入真正 mask loss |
| `ultralytics/cfg/models/11_myself/yolo11-6csar-extent-0928-*.yaml` | 六個完整模型定義             |
| `ultralytics/cfg/experiments/damage-extent-0928/*.yaml`            | 六個配套訓練 recipe          |
| `tools/audit_damage_extent.py`                                     | 唯讀 XML／polygon 範圍核對   |
| `tests/test_damage_extent.py`                                      | loss 與六組模型整合測試      |
| `tests/test_damage_extent_audit.py`                                | 標記核對測試                 |
| `docs/damage-extent-0928.md`                                       | 操作、公式、訓練順序與限制   |

### Task 1: 範圍 loss 與既有 criterion 整合

**Files:** Create `ultralytics/utils/damage_extent.py`, `tests/test_damage_extent.py`; Modify `ultralytics/utils/training_aux_0918.py` 的 `parse_training_aux`, `TrainingAuxSegmentationLoss.__init__`, `single_mask_loss`。

**Interfaces:**

- Consumes: `logits`, `target` 為 `[N,H,W]`；`boxes` 為 mask pixel 座標 `[N,4]`。
- Produces: `instance_extent_loss(logits, target, boxes, fn_weight=0.7, margin=1.0) -> Tensor`，回傳對實例加總的 FP32 scalar。空 foreground 回傳可微零，不保留 state。

- [x] **Step 1: 寫入梯度與行為測試**

```python
target = torch.zeros(1, 8, 8)
target[:, 2:6, 2:6] = 1
box = torch.tensor([[2.0, 2.0, 6.0, 6.0]])
good = torch.where(target.bool(), 8.0, -8.0)
partial = torch.full_like(target, -8.0)
partial[:, 3:5, 3:5] = 8
assert instance_extent_loss(good, target, box) < instance_extent_loss(partial, target, box)
assert instance_extent_loss(good, target, box) < instance_extent_loss(torch.full_like(target, 8.0), target, box)
missed = torch.full_like(target, -20.0, requires_grad=True)
instance_extent_loss(missed, target, box).backward()
assert missed.grad[target.bool()].sum() < 0
```

- [x] **Step 2: 確認測試因新函式不存在而失敗**

Run: `.venv-shadow-check/Scripts/python.exe -m pytest tests/test_damage_extent.py --noconftest -o addopts= -q`。

- [x] **Step 3: 實作數值核心及接線**

用 `torch.arange` 建立與外擴 boxes 相交的 ROI，避免 crop_mask 在 CPU 原地修改輸入；FP32 關閉 autocast。核心：

```python
p = logits.float().sigmoid()
y = target.detach().float()
positive = roi * y
negative = roi * (1 - y)
tp = (p * positive).sum((1, 2))
fn = ((1 - p) * positive).sum((1, 2))
fp = (p * negative).sum((1, 2))
tversky = 1 - (tp + 1e-6) / (tp + fn_weight * fn + (1 - fn_weight) * fp + 1e-6)
bce = F.binary_cross_entropy_with_logits(logits.float(), y, reduction="none")
pos_n, neg_n = positive.sum((1, 2)), negative.sum((1, 2))
balanced = (bce * positive).sum((1, 2)) / pos_n.clamp_min(1) + (bce * negative).sum((1, 2)) / neg_n.clamp_min(1)
balanced = balanced / (pos_n.gt(0).float() + neg_n.gt(0).float()).clamp_min(1)
return (0.5 * (tversky + balanced) * pos_n.gt(0).float()).sum()
```

驗證輸入維度／shape/device，fn_weight 為有限值且嚴格在 (0,1)，margin 為有限非負值。parser 增加 `extent: {gain: 0.0, fn_weight: 0.7, margin: 1.0}`，gain 非負，無效／錯字設定 ValueError。`single_mask_loss` 原 loss 照算，只在 `_training_terms` 且 extent gain 非零時，以 `einsum('in,nhw->ihw', pred.float(), proto.float())` 加上 extent 項；原 boundary 計算保持不變。

- [x] **Step 4: 驗證主遮罩梯度、關閉等價及 validation 不啟用**

同一 raw prediction 分別交給原 criterion 與新 criterion；額外 seg_loss 必須 > 0，其他 loss slots 相同；對 coefficients/prototypes 求梯度需有限非零。原 0918 測試一併執行。

### Task 2: 六組 YAML、recipe 與整合驗證

**Files:** Create 上述六組 models/recipes；Extend `tests/test_damage_extent.py`。

**Interfaces:**

- Consumes: 原 semantic-token YAML、Task 1 的 `training_aux.extent`。
- Produces: 後綴 `01-baseline`, `02-shadow`, `03-consistency`, `04-extent`, `05-boundary`, `06-detail960`；recipe 使用同後綴。

- [x] **Step 1: 加入每組 graph/state、完整 backward、空 batch 與 recipe 解析測試**

```python
base = YAML.load("ultralytics/cfg/models/11_myself/yolo11-6csar-semantic-token.yaml")
for suffix in SUFFIXES:
    cfg = YAML.load(f"ultralytics/cfg/models/11_myself/yolo11-6csar-extent-0928-{suffix}.yaml")
    assert cfg["backbone"] == base["backbone"]
    assert cfg["head"] == base["head"]
    recipe = get_cfg(f"ultralytics/cfg/experiments/damage-extent-0928/{suffix}.yaml")
    assert recipe.overlap_mask is False and recipe.mask_ratio == 4
```

`SUFFIXES` 在測試中定義為上述六個字串 tuple；真實 forward/backward 沿用 `tests/test_damage_semantic.py` 的 overlapping_batch 形狀，另建 nc=8、D=2/R=5 的 batch。

- [x] **Step 2: 建立完整 YAML 定義並滿足整合測試**

每個 model 複製原檔全文，插入 `scale: n` 與以下差異：

| 後綴           | shadow.enabled | consistency.gain | extent.gain | boundary.gain | recipe imgsz |
| -------------- | -------------- | ---------------: | ----------: | ------------: | -----------: |
| 01-baseline    | false          |              0.0 |         0.0 |           0.0 |          640 |
| 02-shadow      | true           |              0.0 |         0.0 |           0.0 |          640 |
| 03-consistency | true           |              0.2 |         0.0 |           0.0 |          640 |
| 04-extent      | true           |              0.2 |         0.2 |           0.0 |          640 |
| 05-boundary    | true           |              0.2 |         0.2 |          0.05 |          640 |
| 06-detail960   | true           |              0.2 |         0.2 |           0.0 |          960 |

所有組 shadow 的 strength=[0.15,0.45]、probability=1.0、softness=0.15；consistency.confidence=0.7；extent.fn_weight=0.7、margin=1.0；overlap.gain=0.0、pairs=[[D,R]]。recipe 使用規格中的共同參數，`model` 指向對應 YAML；`name` 為 `damage-extent-0928-` 加後綴；`project=runs/damage_extent_0928`；不寫死 data 或 pretrained。

- [x] **Step 3: 執行所有模型整合測試與 last0926 嚴格載入**

Run: `.venv-shadow-check/Scripts/python.exe -m pytest tests/test_damage_extent.py tests/test_training_aux_0918.py tests/test_damage_semantic.py --noconftest -o addopts= -q`。

以 `SegmentationModel(cfg,nc=8)` 檢查與真實 last0926 的 `state_dict` strict=True；01 對原模型確認 loss 及 eval predictions 相同；所有組檢查 deepcopy/EMA 與 non-square eval。

### Task 3: 唯讀標記核對與交付文件

**Files:** Create `tools/audit_damage_extent.py`, `tests/test_damage_extent_audit.py`, `docs/damage-extent-0928.md`。

**Interfaces:**

- Consumes: `audit(xml_path: Path, labels_path: Path, names: list[str], min_iou=0.1, min_extent=0.75) -> dict`。
- Produces: JSON：XML counts、polygon counts、matches（xml_id/label_line/bbox_iou/width_ratio/height_ratio/review_extent）、unmatched_xml、unmatched_polygons；CLI `--xml --labels --names --out`，只寫 out。

- [x] **Step 1: 寫小型 XML/polygon 測試**

```python
xml = "<annotation><size><width>100</width><height>100</height></size><object><name>D</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>50</ymax></bndbox></object></annotation>"
labels = "0 0.2 0.2 0.3 0.2 0.3 0.3 0.2 0.3\n"
# bbox IoU=0.0625，因此預設應列未對應；min_iou=0.01 時 matched 但 review_extent=true。
```

另測同類兩個 GT 與單一 polygon 只匹配一次，5-column detection row／非有限或越界 polygon／未知 XML 類別均明確拒絕。

- [x] **Step 2: 實作解析與指派**

XML 僅接受正影像尺寸、圖內有效 xyxy、names 中的類別。polygon 一行需 `1+2K` 值、K≥3、ID 合法整數、normalized coordinates∈[0,1]。乘上 `[width,height]` 算外接框。按類別以 `scipy.optimize.linear_sum_assignment` 在含 dummy 行列的矩陣上最大化有效 IoU 總和（低於 min_iou 權重 0），最後只保留達門檻的配對。width_ratio/height_ratio=polygon_bbox_span/XML_span；低於 min_extent 標 review，不改 mask。

- [x] **Step 3: 跑實際新 XML 核對與文件指令檢查**

```powershell
& '.venv-shadow-check/Scripts/python.exe' 'tools/audit_damage_extent.py' --xml 'C:/Users/USER/Desktop/0928-1/CBHU0708054-A.xml' --labels 'C:/Users/USER/Desktop/碩論/dataset0608/dataset0608/dataset/labels/train/B/CBHU0708054-A.txt' --names B C D H O R X U --out 'runs/dent_diagnosis_0928_relabel/extent_audit.json'
```

文件列出六組 model 與 recipe、完整 loss 公式、原 gain/anchor normalization、標記前提、共同 checkpoint 的比較方式、05/06 分支、960成本與所有可重現測試。訓練入口為 `yolo segment train cfg=<recipe> data=<實際已核對的資料 YAML> pretrained=<實際既有權重>`，兩個外部輸入由操作者提供；不虛構已修訂資料集。

- [x] **Step 4: 自我檢查與最終 diff**

Run: `git diff --check`；檢查規格每條均有實作／測試；查找本計畫的未定義函式或實作佔位文字並修正；核對 `tools/predict-2.py` 未被本次修改。記錄通過數與尚需重新訓練的限制。

## 完成記錄（2026-09-28）

- 首次新 loss 測試因模組尚未建立而失敗；實作後 loss／模型／既有訓練流程的 69 項測試通過，標記核對的 8 項測試通過，共 77 項。
- 六個模型與六個訓練 recipe 已完成；全部八類模型可 strict=True 載入真實 last0926.pt，參數量均為 6,302,676。
- 已執行實際新 XML／本機舊 polygon 比對：30 個對應、64 個 XML 物件未對應、25 個對應需複核範圍。沒有改寫任一標記來源。
- 信心分數預設隱藏已實際重跑原圖確認；保留 confidence=0.25 與原有預測結果。
- docs/damage-extent-0928.md 已包含公式、標記前提、完整指令、01–04 主線、05/06 對照及預期影響。
- 自我檢查：規格各項均已對應實作／驗證；函式名稱與參數一致，無待實作佔位文字；git diff --check 通過。tools/predict-2.py 僅保留開始前已存在的 last0926.pt 路徑修改。
- 此任務完成的是訓練修改與可執行設定；尚未啟動完整訓練，實際資料需先核對完整 polygon，不能將 XML 框當成已完成的精細遮罩。
