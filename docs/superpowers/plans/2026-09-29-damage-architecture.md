# Damage Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 保留六個 CSAR 主幹，新增亮度、淺層與 P2 解碼旁路，交付四組真正有架構差異的完整 YAML 及共同訓練 recipes。

**Architecture:** 對入口 Conv 與原型解碼器以繼承方式保留既有參數與計算，旁路以有界殘差相加；淺層融合重用專案的 DentDetailFusion。所有尺度、主要層索引及 head 原輸出契約維持不變。

**Tech Stack:** 本地 Ultralytics fork、Python、PyTorch、PyYAML、pytest。

**Spec:** `docs/superpowers/specs/2026-09-29-damage-architecture.md`

## Global Constraints

- 原 AttentionResiduals、FSNetShuffle、六個 CSAR、四個 DamageSemanticAttention 的定義、連接與索引保持原樣。
- 保留原入口 Conv 的計算與參數名稱；新增亮度細節殘差旁路，不替換為 IIMStem。
- 保留 P3/P2/P4/P5 次序、八類覆寫、stride 4 遮罩、獨立 D/R 多標籤與原偵測分支。
- 新增模組保留所有基底 state_dict 的 key 與形狀，新增權重由訓練學習。
- 四組訓練 recipe 的 optimizer、loss、640 輸入與資料前提相同，不啟用 training_aux；overlap_mask=False、mask_ratio=4。
- 不覆寫 0928 YAML、原圖、標記或權重；保留照片預設不顯示分數。
- 不自動啟動完整訓練；不能宣稱未經資料集驗證的準確率改善。

使用者已明確要求直接修改與產生 YAML，按計畫在此任務內實作，不再次詢問執行方式；本機前次已確認沒有 executing-plans 技能，自行執行與自我檢查，不派遣子代理。不自動提交。

## 檔案責任

| 檔案                                                             | 責任                                                              |
| ---------------------------------------------------------------- | ----------------------------------------------------------------- |
| `ultralytics/nn/modules/damage_architecture.py`                  | LuminanceResidualStem、DualPathP2Proto、Segment26MultiLabelExtent |
| `ultralytics/nn/modules/__init__.py`                             | 匯出新模組                                                        |
| `ultralytics/nn/tasks.py`                                        | YAML parser 註冊，保留已有 criterion 判斷                         |
| `tests/test_damage_architecture.py`                              | 新增路徑、共同權重、真實訓練／推論、fuse 與保存測試               |
| `ultralytics/cfg/models/11_myself/yolo11-6csar-arch-0929-*.yaml` | 00-baseline、01-luma、02-shallow、03-dualproto 完整模型           |
| `ultralytics/cfg/experiments/damage-arch-0929/*.yaml`            | 四組相同訓練條件，model/name 對應各架構                           |
| `docs/damage-architecture-0929.md`                               | 局部架構圖、設定差異、原理、訓練指令、驗證結果                    |

### Task 1：入口與原型旁路

**Interfaces:**

`LuminanceResidualStem(c1,c2,k=3,s=2,detail_scale=0.1)`：B×3×H×W → 原 Conv 輸出形狀；c1≠3 拒絕，gain 須有限；forward_fuse 使用已融合 Conv 同樣加回旁路。

`DualPathP2Proto(ch,c_=256,c2=32,nc=80,detail_scale=0.1)`：恰好四個 P3/P2/P4/P5 特徵 → 與 Proto26MultiLabel 同形同型的 train/eval 回傳；P2 為 P3 的兩倍解析度。不保留 graph tensor 在 self。

`Segment26MultiLabelExtent(nc=80,nm=32,npr=256,multilabel_gain=0.5,cooccurrence_weight=2.0,detail_scale=0.1,reg_max=16,end2end=False,ch=())`：恰好 10 路輸入，前四特徵、中間四語意、末二淺層。繼承 Segment26MultiLabelShadow；以 DualPathP2Proto 取代 proto 類別，所有舊權重 key 保持相容。

- [x] Step 1：先加入關閉等價、梯度與非法輸入測試，確認未建模組時匯入失敗。

```python
old = Conv(3, 16, 3, 2).eval()
new = LuminanceResidualStem(3, 16, detail_scale=0).eval()
new.load_state_dict(old.state_dict(), strict=False)
x = torch.rand(2, 3, 32, 48)
torch.testing.assert_close(new(x), old(x), rtol=0, atol=0)
```

Run: `.venv-shadow-check/Scripts/python.exe -m pytest tests/test_damage_architecture.py --noconftest -o addopts= -q`。

- [x] Step 2：實作兩條旁路，重用亮度訊號與既有淺層融合，新增參數限定命名於 detail/native_*。

入口核心（base 由 Conv.forward 或 Conv.forward_fuse 算出）：

```python
detail = self.detail_stem(self.luminance_detail(x))
gate = self.detail_gate(torch.cat((base, detail), 1)).sigmoid()
return base + self.detail_scale.tanh() * gate * detail
```

proto 核心（original = super().forward(x)，base 為其中 prototypes）：

```python
detail = self.native_p2(x[1])
context = F.interpolate(self.native_context(x[2]), size=detail.shape[-2:], mode="bilinear", align_corners=False)
gate = self.native_gate(torch.cat((detail, context), 1)).sigmoid()
delta = self.native_output(self.native_refine(detail + context))
prototypes = base + self.native_scale.tanh() * gate * delta
return (prototypes, original[1]) if isinstance(original, tuple) else prototypes
```

所有 gate 的 weight/bias 初始為零，sigmoid 初始為 0.5，gain=0.1 提供立即可學梯度。保留原 prototype aux 輸出。

- [x] Step 3：新增入口類到 parser base_modules、新 head 到三個 segmentation/legacy 集合，模組 exports 同步更新。新 head 是 Segment26MultiLabel 子類，沿用原多標籤 criterion；不擴大 training_aux 的 head 接受範圍。

### Task 2：四組完整 YAML 與共同訓練設定

- [x] Step 1：以 `yolo11-6csar-semantic-token.yaml` 為基底產生四個檔，保留 layer 0–33 索引。

```python
cfg01["backbone"][0] = [-1, 1, "LuminanceResidualStem", [64, 3, 2, 0.1]]
cfg02["head"][-1] = [
    [22, 25, 28, 31, 23, 26, 29, 32, 6, 3],
    1,
    "Segment26MultiLabelShadow",
    ["nc", 32, 256, 0.5, 2.0, 0.1],
]
cfg03["head"][-1] = [
    [22, 25, 28, 31, 23, 26, 29, 32, 6, 3],
    1,
    "Segment26MultiLabelExtent",
    ["nc", 32, 256, 0.5, 2.0, 0.1],
]
```

cfg01/02/03 分別深複製前一版；cfg00 為原基底。每份完整檔附註新增路徑與相同訓練條件。

- [x] Step 2：複製 0928 的 01-baseline 訓練設定，僅更新 model、name、project 為 damage_arch_0929 系列。四組均為 640、epoch 200、batch 4、AdamW、lr0 0.0003、seed 928；不指定 data/pretrained、不啟用 training_aux。

- [x] Step 3：測試相同主要 YAML 圖、差異模組存在、舊 state 名稱與形狀完整保留、四組 recipe 除識別欄位外完全相等。

```python
assert cfg["backbone"][1:] == base_cfg["backbone"][1:]
assert cfg["head"][:-1] == base_cfg["head"][:-1]
assert all(k in new_state and v.shape == new_state[k].shape for k, v in old_state.items())
incompatible = model.load_state_dict(old_state, strict=False)
assert not incompatible.unexpected_keys
```

### Task 3：實際相容性、訓練契約與交付

- [x] Step 1：以 test_damage_semantic 的 overlapping_batch 類型建立 nc=8、D=2/R=5 的 128×128 低亮度樣本。四版皆須 loss.sum().backward() 有限，新旁路梯度非零。03 另測空物件及 s scale、nc=3 覆寫。

```python
model.args = get_cfg(overrides={"overlap_mask": False})
loss, items = model(batch)
assert torch.isfinite(loss).all()
loss.sum().backward()
ModelEMA(model)
```

- [x] Step 2：03 檢查 128×192 eval、FP16 eval、export tensor 契約與 checkpoint 重載；比較 fuse 前後輸出，並確認入口 gate 仍影響 fused 輸出。新模組 zero gain 的整個模型須回到基底。

- [x] Step 3：以真實 last0926.pt 對照所有 key/shape，記錄載入覆蓋率、參數量、trainable additions。不是 strict=True 的全新模型載入；預期僅新增分支 missing_keys。實際以 640 輸入做 forward，檢查輸出有限及遮罩解析度。

- [x] Step 4：執行新測試及相關回歸：`tests/test_damage_architecture.py tests/test_damage_semantic.py tests/test_shadow_dent.py tests/test_damage_extent.py tests/test_training_aux_0918.py`；不啟動完整訓練。

- [x] Step 5：寫 docs/damage-architecture-0929.md，說明三處局部圖差異、與 0928 的區別、權重初始化、相同 recipe 的 CLI 指令、推論額外成本及尚未量測的召回改善。自查規格與計畫覆蓋、無佔位文字、所有連結存在；`git diff --check` 通過後交付。

## 執行與自我檢查記錄（2026-09-29）

- 新測試先因 damage_architecture 尚未存在而失敗；實作後 14 項通過，與相關舊流程合併測試共 93 項通過。
- 四組完整模型與訓練 YAML 已建立，主要層與舊 state key/shape 保留。
- 真實 last0926.pt 的 999/999 個 state entries 均同名同形載入，且逐值核對一致。新 BatchNorm 的計數器由框架初始化為 0，其餘新增項按新分支初始化。
- 640 輸入四組 forward 均有限，prototype 皆為 [1,32,160,160]；參數量依序為 6,302,676／6,303,638／6,378,586／6,394,044。
- zero gain、非正方形、n/s 類別覆寫、空標記、fuse、FP16、checkpoint、EMA 及 export tensor 契約均有覆蓋。未實際執行 ONNX/TensorRT 匯出。
- 規格／計畫／程式介面一致；新增三條路徑均到達實際預測；詳細文件包含模型／訓練檔連結及 CLI。
- 保留既有 0928 檔與隱藏分數設定，沒有修改資料集或權重、沒有啟動完整訓練。
