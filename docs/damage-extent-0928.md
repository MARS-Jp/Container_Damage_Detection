# 0928：陰影漏檢與完整損壞範圍的訓練修改

本次已建立 **6 個完整模型 YAML、6 個配套訓練 YAML**，新增直接監督最終實例遮罩的範圍 loss，並讓推論照片預設不顯示信心分數。六組模型皆可嚴格載入現有 `last0926.pt`；目前完成程式與相容性驗證，尚未執行這六組的完整訓練，因此沒有新的準確率或改善幅度可宣稱。

此輪依據為重新標記的 `C:/Users/USER/Desktop/0928-1/CBHU0708054-A.xml`。目標是學到**影像中可判讀且有標記的完整損壞範圍**；照片完全看不到的表面不能憑空恢復為可靠的損壞遮罩。

## 1. 這張照片反映的兩種問題

新 XML 包含 25 個 D（凹損）與 69 個 R（鏽蝕）。以現有權重、`imgsz=640`、`conf=0.25`、`iou=0.3` 的核對結果為例：

| 現象                          | 此例證據                                                               | 本次對應處理                                 |
| ----------------------------- | ---------------------------------------------------------------------- | -------------------------------------------- |
| 有找到一小塊，但範圍不完整    | 新 D16 的框為 `[29,435,75,480]`，既有預測只有約 12×17；框 IoU 約 0.101 | 完整實例標記、範圍 loss                      |
| 陰影位置的凹損沒有被抓到      | 新 D24、D25 附近的原始 D 分數約為 0.0178、0.0043，遠低於 0.25          | 陰影成對訓練、可靠位置的一致性監督、補齊正例 |
| 相鄰或重疊的 D/R 容易互相干擾 | 25 個 D 框中有 19 個與至少一個 R 框相交                                | 保留獨立實例遮罩及既有多標籤分支             |

框相交只代表候選範圍相交，不能據此判定兩類損壞的像素一定重疊。提高解析度及切片推論的試跑沒有穩定找回 D24、D25，因此本次把訓練標記與監督方式列為主軸。單純把門檻降到極低，也可能同時帶入大量背景誤報。

## 2. 六組 YAML 與使用順序

每組都有兩種 YAML：**模型檔**包含完整 backbone、head 與 `training_aux`；**訓練檔**包含模型路徑、解析度、epoch、batch、optimizer 等執行設定。訓練時讀取下表右側的訓練檔。

| 順序 | 完整模型 YAML                                                                                      | 配套訓練 YAML                                                                           | 該組要回答的問題                           |
| ---- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------ |
| 01   | [01-baseline](../ultralytics/cfg/models/11_myself/yolo11-6csar-extent-0928-01-baseline.yaml)       | [01-baseline](../ultralytics/cfg/experiments/damage-extent-0928/01-baseline.yaml)       | 用已核對標記與共同參數重新建立基準         |
| 02   | [02-shadow](../ultralytics/cfg/models/11_myself/yolo11-6csar-extent-0928-02-shadow.yaml)           | [02-shadow](../ultralytics/cfg/experiments/damage-extent-0928/02-shadow.yaml)           | 加入陰影訓練能否改善暗處漏檢？             |
| 03   | [03-consistency](../ultralytics/cfg/models/11_myself/yolo11-6csar-extent-0928-03-consistency.yaml) | [03-consistency](../ultralytics/cfg/experiments/damage-extent-0928/03-consistency.yaml) | 陰影前後的可靠語意能否更穩定？             |
| 04   | [04-extent](../ultralytics/cfg/models/11_myself/yolo11-6csar-extent-0928-04-extent.yaml)           | [04-extent](../ultralytics/cfg/experiments/damage-extent-0928/04-extent.yaml)           | 最終實例遮罩是否較完整，同時控制多塗背景？ |
| 05   | [05-boundary](../ultralytics/cfg/models/11_myself/yolo11-6csar-extent-0928-05-boundary.yaml)       | [05-boundary](../ultralytics/cfg/experiments/damage-extent-0928/05-boundary.yaml)       | 在 04 上加邊界監督，是否改善精確輪廓？     |
| 06   | [06-detail960](../ultralytics/cfg/models/11_myself/yolo11-6csar-extent-0928-06-detail960.yaml)     | [06-detail960](../ultralytics/cfg/experiments/damage-extent-0928/06-detail960.yaml)     | 在 04 上改為 960 訓練，小凹損是否受益？    |

| 設定               |   01 |   02 |   03 |   04 |   05 |   06 |
| ------------------ | ---: | ---: | ---: | ---: | ---: | ---: |
| 陰影成對訓練       | 關閉 | 啟用 | 啟用 | 啟用 | 啟用 | 啟用 |
| `consistency.gain` |    0 |    0 |  0.2 |  0.2 |  0.2 |  0.2 |
| `extent.gain`      |    0 |    0 |    0 |  0.2 |  0.2 |  0.2 |
| `boundary.gain`    |    0 |    0 |    0 |    0 | 0.05 |    0 |
| 配套訓練檔 `imgsz` |  640 |  640 |  640 |  640 |  640 |  960 |

先比較 01→02→03→04，再分別比較 **04 與 05、04 與 06**。05 和 06 是獨立對照，06 沒有包含 05 的邊界項。

為了分辨各項修改的效果，六組均從**同一份 `last0926.pt`** 出發，使用相同 train/val/test 切分與標記版本；編號不表示每一組必須載入前一組的 `best.pt`。若逐次接續權重，訓練時間與已學到的內容也會改變，就不能將改善單獨歸因於新增項目。

06 的推論圖與 04 相同，960 由配套訓練檔設定；只載入名稱包含 `detail960` 的模型檔不會自動把輸入變為 960。所有共同參數都是待驗證的起點，並非已找到的最佳組合。

## 3. 標記是啟動訓練前的必要輸入

新 XML 提供矩形框；本專案的分割訓練需要逐實例 polygon。應把凹損可見的完整表面畫成 D polygon，不能只圈最暗的小塊，也不能把鏽蝕自動當成凹損。若同一位置同時有 D 和 R，分別保留兩個可相交的 polygon。YOLO 分割標記的一列為類別 ID 與多個正規化頂點，格式可參照 [Ultralytics 分割資料說明](https://docs.ultralytics.com/datasets/segment/)。

此次找到的本機舊 polygon 為：

```text
C:/Users/USER/Desktop/碩論/dataset0608/dataset0608/dataset/labels/train/B/CBHU0708054-A.txt
```

它只有 D=12、R=19；與新 XML 按同類框、IoU≥0.1 做一對一比對，得到 30 組對應、64 個 XML 物件未對應（D=14、R=50），25 組已對應物件的寬或高比例低於 0.75，另有 1 個舊 polygon 未對應。這是**幾何核對結果**，不能直接等同 64 個錯誤標記；小遮罩、標記拆分方式及範圍差異都可能影響配對。遠端訓練實際讀取的標記是否就是這份舊檔，仍未確認。

新增的核對工具只讀取原檔，將報告寫入另一個 JSON：

```powershell
python tools/audit_damage_extent.py `
  --xml 'C:/Users/USER/Desktop/0928-1/CBHU0708054-A.xml' `
  --labels 'C:/Users/USER/Desktop/碩論/dataset0608/dataset0608/dataset/labels/train/B/CBHU0708054-A.txt' `
  --names B C D H O R X U `
  --out 'runs/dent_diagnosis_0928_relabel/extent_audit.json'
```

複核時把 `--labels` 改為真正準備拿來訓練的 polygon 檔。報告的 `width_ratio`、`height_ratio` 是 polygon 外接框寬高除以 XML 框寬高，並非遮罩召回率；此工具不能代替逐像素的標記檢查。不要把 XML 方框填滿後當成精細遮罩，否則模型會被教導把框內正常背景也畫成損壞。

若沿用目前八類權重，資料集的類別順序應維持：

```yaml
names:
    0: B
    1: C
    2: D
    3: H
    4: O
    5: R
    6: X
    7: U
```

模型檔的 `nc: 80` 沿用原定義，trainer 會以資料 YAML 的八類覆寫。已用 `nc=8` 檢查全部參數相容；類別數相同但順序不同，仍會造成類別語意錯位。設定 `overlap_mask: false` 讓每個實例保有自己的遮罩；`mask_ratio: 4` 對應目前遮罩解析度。

## 4. 模型保留的部分與新增監督位置

推論仍使用原本的 AttentionResiduals → FSNetShuffle → 六個 CSAR → DamageSemanticAttention → Segment26MultiLabel。P3/P2/P4/P5 的輸入順序與 stride 4 的遮罩原型保持一致。四個尺度的語意分支仍以獨立 sigmoid 支援多類損壞；學習式 damage tokens 依類別訓練，沒有額外文字編碼器。

新 loss 接在**實際用來輸出分割結果的實例遮罩**，不是只約束旁邊的語意熱圖。程式位於 [damage_extent.py](../ultralytics/utils/damage_extent.py)；由 [training_aux_0918.py](../ultralytics/utils/training_aux_0918.py) 的 `single_mask_loss` 接入。六組推論參數均為 6,302,676；同解析度推論不增加本次訓練 loss 的運算。

### 4.1 從原型到實例遮罩

對第 i 個已指派到 GT 的正樣本，令 cᵢₖ 為模型輸出的 mask coefficient，Pₖ(u,v) 為第 k 個共享 prototype：

\[
z_i(u,v)=\sum_{k=1}^{32}c_{ik}P_k(u,v),\qquad
p_i(u,v)=\sigma(z_i)=\frac{1}{1+e^{-z_i}}.
\]

z 是未壓縮的 logit，p 是像素屬於該實例的機率。新增項從 p 或 z 計算，再經反向傳播更新 coefficients、prototypes 及前面的影像特徵。因此，修正的是模型畫出的損壞區域。此處的 i 是正樣本指派，一個 GT 實例可能對應多個正樣本；後續沿用正樣本總數正規化。

### 4.2 計算區域 R：看見損壞，也看見附近背景

令 GT 框在 mask 座標為 (x₁,y₁,x₂,y₂)，m=1，像素中心為 (u+0.5,v+0.5)：

\[
R_i(u,v)=\mathbf1[x_1-m\leq u+0.5<x_2+m]
\mathbf1[y_1-m\leq v+0.5<y_2+m].
\]

R 只決定在哪裡算新增 loss。外擴 1 個 mask 像素讓它看見邊界附近的背景，避免只看到框內正例。`mask_ratio=4` 時，相當於約 4 個網路輸入像素，映回原圖的寬度還取決於縮放比例。**GT 遮罩沒有膨脹，推論輸出也沒有被自動擴框或填滿。**

### 4.3 TP、FN、FP：把漏掉與多畫分開計算

令 yᵢ(u,v) 為人工標記的 0/1 遮罩，以下加總都在該實例的像素上：

\[
TP_i=\sum R_i p_i y_i,\qquad
FN_i=\sum R_i(1-p_i)y_i,\qquad
FP_i=\sum R_i p_i(1-y_i).
\]

這些是可微的軟計數，並非把 p 切成 0/1 後才計算。FN 在模型對真實損壞給出低機率時增加；FP 在模型把正常背景當成損壞時增加。例如完整凹損有 100 個標記像素，只對其中 30 個給高機率，FN 就會明顯偏大；全部塗滿雖然能減少 FN，卻會增加周圍的 FP。

### 4.4 Tversky：提高遺漏範圍的代價

\[
L_{T,i}=1-\frac{TP_i+\epsilon}
{TP_i+0.7FN_i+0.3FP_i+\epsilon},\qquad\epsilon=10^{-6}.
\]

0.7 與 0.3 決定分母中 FN、FP 的相對權重，使本次設定更偏向補回遺漏前景。這不表示召回率會提高 70%，也不代表兩類像素梯度在所有情況下都固定為 7:3。ε 避免零分母。Tversky 以不對稱錯誤權重調節分割取捨的想法來自 [Tversky loss 論文](https://arxiv.org/abs/1706.05721)；本專案採用的數值仍需用貨櫃資料驗證。

它的作用依賴標記：若 y 只包含一小塊凹損，完整凹損其餘部分仍被視為背景，再好的 loss 也會受到錯誤監督。因此不能用調大 `fn_weight` 取代標記修正。

### 4.5 平衡 BCE：給嚴重漏掉的前景恢復梯度

逐像素二元交叉熵為：

\[
\ell(z,y)=-y\log\sigma(z)-(1-y)\log(1-\sigma(z)).
\]

實作使用 `binary_cross_entropy_with_logits` 的穩定形式，避免直接對接近 0 的機率取 log。令 N₊=ΣRy、N₋=ΣR(1−y)，g=1[N₊>0]+1[N₋>0]：

\[
L_{B,i}=\frac{1}{\max(g,1)}\left[
\frac{\sum R_i y_i\ell(z_i,y_i)}{\max(N_+,1)}+
\frac{\sum R_i(1-y_i)\ell(z_i,y_i)}{\max(N_-,1)}\right].
\]

前景與背景分別平均，再平均有出現的群組；大量背景不會只靠數量壓過小凹損。對單一像素，BCE 對 z 的導數為 `p−y`。若真實前景 y=1，但 p 接近 0，導數接近 −1，梯度下降會把 z 往上推。這補強 Tversky 經 sigmoid 回傳時，在機率極端飽和區域可能較弱的梯度。

同理，背景 y=0 但 p 高時，導數為正，會降低 z。新增項同時處理「少畫」與「多畫」，不是無條件把遮罩放大。

### 4.6 合併到實際 seg loss

\[
L_{E,i}=\tfrac12(L_{T,i}+L_{B,i})\mathbf1[N_+>0],
\qquad
\Delta L_{seg}=\frac{\texttt{hyp.box}}{N_{pos}}\,
0.2\sum_i L_{E,i}.
\]

空前景的新增項為可微零；沒有任何正樣本時，沿用原 criterion 的空目標處理，不執行除以零。`N_pos` 是該批次正樣本總數；現有框架以 `hyp.box` 同時縮放分割項，回傳訓練 loss 時還有原本的 batch 尺度處理。0.2 是新增項增益，不是取代原本的 mask BCE。原 box、class、DFL、多標籤等監督也保留。

extent 目前作用於**所有已標記類別的 matched masks**，不是只有 D。它不能直接替一個完全沒有正樣本指派的凹損產生新候選；這類漏檢仍需要正確標記、陰影正例與原偵測分支的監督共同改善。

## 5. 陰影、一致性及重疊辨識如何配合

### 5.1 同一張圖的原始視圖與陰影視圖

令 x,y 為 −1 到 1 的正規化影像座標，θ 為隨機方向、b 為偏移，s=0.15 為柔化寬度：

\[
M(x,y)=\sigma\left(\frac{y\sin\theta+x\cos\theta-b}{s}\right),
\quad a\sim U(0.15,0.45),\quad I'=I(1-aM).
\]

M 產生柔和半平面陰影，同一係數作用於 RGB 三通道；a 是亮度衰減比例。它改變可見對比，不改變像素位置，因此原 polygon 可以用在 I 與 I′。每張圖的陰影套用機率為 1；原始視圖仍保留。陰影最深處約保留原亮度的 55%～85%，但不保證涵蓋真實場景所有光照。

02～06 把原始與陰影視圖合併成一個 batch 做 forward，兩邊各自使用同一 GT 計算監督並平均：

\[
L_{sup}=\tfrac12\{L(I,Y)+L(I',Y)\}.
\]

因此 `batch=4` 會讓這個 forward 看見 8 個視圖。相對 01，影像運算量與中間特徵記憶體會增加；此成本只在訓練。02 與 03 的成對方式相同，便於單獨比較一致性項。

### 5.2 一致性只使用符合 GT 的可靠位置

在每個尺度的語意分支，令 q=stop_gradient(σ(z_clean))、p′=σ(z_shadow)。正例選擇條件為 `Y=1 且 q≥0.7`，負例為 `Y=0 且 q≤0.3`：

\[
L_{cons}=\operatorname{BalancedMean}_{eligible}(p'-q)^2.
\]

此處 BalancedMean 先分別平均合格正例與負例，再平均存在的群組；沒有合格位置時該項為零。四個尺度平均後，以 `consistency.gain=0.2` 加入語意 loss。stop_gradient 使這一項只推動陰影分支去接近原始視圖；模型仍共用參數，原始視圖也繼續接受 GT 監督，沒有第二個常駐教師模型。

若原始視圖漏掉真正的凹損，該位置不會被當成「可靠負例」教給陰影分支，因為負例還必須滿足 GT=0。但沒有標記的真實凹損仍可能被當作背景，這再次說明補齊資料的重要性。

### 5.3 D 與 R 保有獨立的存在機率

既有多標籤分支使用 p_D=σ(z_D)、p_R=σ(z_R)，兩者不必加總為 1；因此同一位置可以同時預測凹損與鏽蝕。既有語意 BCE 對 GT 多類共現像素的權重為：

\[
w(u,v)=1+2\mathbf1\left[\sum_cY_c(u,v)>1\right].
\]

也就是這些像素在該項 loss 中的權重為 3，其餘為 1；既有語意 gain=0.5 保留。本次所有 YAML 的額外 `training_aux.overlap.gain=0`，因為新 XML 只有框，尚不足以新增真實像素交集監督。這不會關掉模型原本的多標籤共現能力，也不強迫每個凹損都要有鏽蝕。

### 5.4 05 的邊界對照

沿用既有 3×3 可微形態邊界 `E(p)=max₃×₃(p)−min₃×₃(p)`，使用 replicate padding。邊界差異為：

\[
L_{edge}=1-\frac{2\sum E(p)E(y)+\epsilon}
{\sum E(p)+\sum E(y)+\epsilon}.
\]

另加 GT 邊界帶上平均的 BCE，兩項各占一半，以 0.05 的增益加入 seg loss，沿用正樣本正規化。形態差分在輪廓附近較大，Dice 形式衡量邊界的對齊；邊界帶 BCE 為初始機率近乎平坦時提供梯度。若完整範圍已正確、輪廓仍粗糙，這一組才有明確比較價值；錯誤或很粗糙的 polygon 邊界也可能被強化。

## 6. 實際訓練指令

請在本專案根目錄、已安裝此 fork 的 Python/GPU 環境執行。先確認 `python -c "import ultralytics; print(ultralytics.__file__)"` 指向這份程式碼；`yolo` 也應來自同一環境。外部原版套件不包含此專案的 CSAR、語意模組與新增 criterion。

以下 PowerShell 指令會要求輸入**已核對且包含 polygon 的資料 YAML 路徑**；它不建立或假裝已經存在修訂後的資料集。若在遠端訓練，權重與資料路徑應改成該主機的實際位置。`cfg` 與 CLI 覆寫方式可參照 [Ultralytics 訓練設定說明](https://docs.ultralytics.com/usage/cfg/)，本次新增 `training_aux` 由本地 fork 實作。

```powershell
$damageData = Read-Host '請輸入已核對完整實例遮罩的 data.yaml 路徑'
$damageWeights = 'C:/Users/USER/Downloads/last0926.pt'
$damageCfg = 'ultralytics/cfg/experiments/damage-extent-0928'
if (-not (Test-Path -LiteralPath $damageData -PathType Leaf)) { throw '找不到 data.yaml' }
if (-not (Test-Path -LiteralPath $damageWeights -PathType Leaf)) { throw '找不到起始權重' }

# 先建立基準，再依序比較三項訓練修改；每組都從同一權重出發。
foreach ($damageStage in @('01-baseline', '02-shadow', '03-consistency', '04-extent')) {
    yolo segment train "cfg=$damageCfg/$damageStage.yaml" "data=$damageData" "pretrained=$damageWeights" device=0
    if ($LASTEXITCODE -ne 0) { throw "訓練失敗：$damageStage" }
}
```

完成 01～04 並確認標記品質後，執行兩個獨立對照：

```powershell
yolo segment train "cfg=$damageCfg/05-boundary.yaml" "data=$damageData" "pretrained=$damageWeights" device=0
yolo segment train "cfg=$damageCfg/06-detail960.yaml" "data=$damageData" "pretrained=$damageWeights" device=0
```

使用 `pretrained=<權重路徑>`，讓 trainer 先按資料類別數建立新 YAML 模型，再載入 checkpoint；`resume=false` 表示重新開始這組訓練而非恢復舊 optimizer 狀態。輸出位於 `runs/damage_extent_0928/damage-extent-0928-組名/`，重複執行時框架可能加上流水號。

共同設定為 epochs=200、patience=0、batch=4、seed=928、AdamW、lr0=0.0003、cos_lr=true，並關閉 mosaic、mixup、copy_paste。保留較溫和的 scale=0.2、translate=0.05、hsv_v=0.2；amp=false、compile=false 用於共同驗證起點。它們與先前 0926 訓練參數不必然相同，因此 01 必須作為這次對照基準。相同 seed 也不保證跨硬體完全一致。

960×960 的像素數為 640×640 的 2.25 倍；這是像素量比例，實際時間與顯存需量測。若 06 需要降低 batch，公平對照時也要重新安排相同 batch 的 04，或明確記錄差異。本輪沒有新增線上切片裁切：裁到一半的實例需要同步處理 polygon 與保留規則，否則會混入與完整範圍目標不一致的監督。

訓練主機需要本次的 `ultralytics/utils/damage_extent.py`、修改後的 `training_aux_0918.py`、六個模型檔與六個訓練檔，以及原本已包含語意模組的完整專案。`audit_damage_extent.py` 用來複核資料；`predict_seg_by_class.py` 用來套用顯示設定。

## 7. 如何判定有改善

每組使用固定的驗證與測試資料，並分別統計陰影／非陰影、D/R 共現／單一類別。這張圖片位於目前的 train 路徑，適合作為除錯案例；若仍用它訓練，就不能把它當成獨立測試證據。

1. **完全漏檢**：在固定 confidence 與 IoU 規則下，以一對一配對計算 D 的召回率和精確率，另報陰影 D 子集。維持相同評估門檻，才可比較模型。
2. **範圍完整度**：對有精細 GT 的遮罩計算 `mask recall=TP/(TP+FN)`、`mask precision=TP/(TP+FP)` 與 IoU。漏掉整個實例也必須列入漏檢統計，不能只統計成功配對的遮罩。
3. **多畫代價**：核對新伸出的遮罩是否落在正常板面、貨櫃立柱或僅有陰影的區域。單看遮罩面積增加無法判定改善。
4. **重疊辨識**：在確實有兩類像素標記的區域分別核對 D 與 R，檢查是否仍只輸出其中一類；不能只用框相交數取代。

extent gain 或 FN 權重過大，可能增加背景誤報；邊界項可能追隨不準確的標記；人工陰影也不等同真實金屬反光。應以以上結果選擇 04、05 或 06，無法先宣稱最後一個編號一定最好。

## 8. 照片不顯示信心分數

[predict_seg_by_class.py](../tools/predict_seg_by_class.py) 已改為預設 `hide_labels=True`，`tools/predict-2.py` 呼叫它時會沿用此行為。照片保留遮罩與輪廓；程式仍計算 confidence 並用原門檻篩選。需要暫時檢查分數時才加 `--show-scores`；舊的 `--hide-labels` 選項仍可使用。

已用現有 `last0926.pt` 重跑這張原圖，確認畫面沒有信心分數。範例使用 `conf=0.25`、`iou=0.3`，仍是 13 個 D、22 個 R，並非完成新訓練後的改善結果：

[現有權重的無分數輸出](../runs/damage_extent_0928/no_scores/000000_CBHU0708054-A/CBHU0708054-A__all.jpg)

```powershell
python tools/predict_seg_by_class.py `
  --model 'C:/Users/USER/Downloads/last0926.pt' `
  --source 'C:/Users/USER/Desktop/已標記/images/train/B/CBHU0708054-A.JPG' `
  --output 'runs/damage_extent_0928/no_scores' `
  --imgsz 640 --conf 0.25 --iou 0.3 --device cpu --retina-masks
```

## 9. 已完成的驗證

- loss／模型／舊訓練流程回歸測試 69 項通過，標記核對工具測試 8 項通過，共 **77 項**。
- 全部六組 YAML 完成真實 forward/backward、非正方形 eval、EMA 與設定解析檢查；01 與基底的原 loss／eval 結果一致。
- 新 loss 對完整遮罩、局部遮罩、塗滿背景、空目標、飽和漏檢、FP16 與 GT 不被修改等條件均有測試；新增梯度可到達 mask coefficients 和 prototypes。
- 對實際 `last0926.pt` 逐組執行 `state_dict` 的 `strict=True` 載入，六組皆通過。
- 驗證階段不啟用 extent／boundary／consistency 等本次訓練附加項；舊 YAML 沒有設定 extent 時，預設增益為 0。
- 未啟動完整 GPU 訓練，未改寫原 XML、原圖、舊 polygon 或權重。程式驗證通過並不等同資料集上的召回率已提高。

可重現的測試指令：

```powershell
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
& '.venv-shadow-check/Scripts/python.exe' -m pytest `
  tests/test_damage_extent.py tests/test_training_aux_0918.py tests/test_damage_semantic.py `
  tests/test_damage_extent_audit.py --noconftest -o addopts= -q
```

實作規格與逐項計畫另存於 [spec](superpowers/specs/2026-09-28-damage-extent.md) 與 [plan](superpowers/plans/2026-09-28-damage-extent.md)。
