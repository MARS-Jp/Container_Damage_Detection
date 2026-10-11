# YOLO11-6CSAR 語意 token：陰影區域與重疊損傷的架構、公式及影響

說明對象：`yolo11-6csar-semantic-token.yaml`，以 2026-09-27 工作區實作為準。

## 1. 先釐清這版要辨識什麼

這版讓每個損傷類別以自己的 token 查詢視覺特徵，產生各自的空間 response，再把語意資訊加回 YOLO 特徵。它直接支援的是**同一區域可同時具有 Rust 與 Dent 等損傷標籤**。對陰影的研究問題則是：**陰影中的損傷能否仍被辨識，以及純陰影是否會被誤判為損傷**。

目前 YAML 沒有陰影專用分類器、照明不變 stem、陰影增強設定、原圖／陰影圖一致性 loss 或邊界 loss。資料集若沒有 Shadow 類別，就不會額外輸出 Shadow；即使新增該類別，也仍需陰影標註來訓練。下文把陰影當成成像干擾，並明確區分「程式已實作」與「機制推測、待實驗驗證」。

重疊與陰影容易同時發生，但問題不同：前者需要保留多類別真值，後者需要辨認照明改變後仍有效的形狀和材質線索。多標籤 loss 本身不會移除陰影。

## 2. YAML 各段修改與實際資料流

### 2.1 讀懂一列設定

```yaml
- [18, 1, DamageSemanticAttention, [nc, 4, 128, 0.001]]
```

| 欄位                      | 此列意義           | 在模型中的影響                              |
| ------------------------- | ------------------ | ------------------------------------------- |
| `18`                      | 讀取第 18 層輸出   | 使用第二階段 CSAR 的 P3 特徵                |
| `1`                       | 建立一次模組       | 並非疊四層 attention                        |
| `DamageSemanticAttention` | 新增的語意模組     | 回傳強化特徵與多標籤 logits                 |
| `nc`                      | 資料集類別數       | 每個尺度建立 nc 個 token，順序對應 class ID |
| `4`                       | attention heads 數 | 128 維分成四組，每組 32 維                  |
| `128`                     | embedding 維度     | 四個尺度使用同維度，但參數各自獨立          |
| `0.001`                   | 初始殘差係數 α     | 初期少量加入語意分支，α 隨訓練更新          |

### 2.2 保留的參數與第 0–20 層

`nc: 80` 是建模預設，訓練器會依資料集改成實際類別數，並非只能辨識 80 種損傷。n/s/m/l/x 的深度與寬度設定沿用基礎模型，沒有新增尺度。n 的 `[0.50, 0.25, 1024]` 分別為深度倍率、寬度倍率與通道上限；一般通道換算為

\[
C_{\mathrm{actual}}=8\left\lceil\frac{\min(C_{\mathrm{yaml}},C_{\max})\,w}{8}\right\rceil.
\tag{1}
\]

例如 YAML 的 512 通道，在 n 尺度變為 128。重複次數大於 1 的層按 `max(round(repeats × depth), 1)` 縮放。這些設定控制模型容量與成本；更寬或更深不保證陰影或重疊辨識更好。

`end2end: False` 明確指定目前 `Segment26MultiLabel` 支援的非 end-to-end 模式。此 head 不支援 end2end=True，推論仍沿用現有後處理流程。

| YAML 層                    | 模組                                                | 是否改動                 | 對陰影／重疊問題的角色                               |
| -------------------------- | --------------------------------------------------- | ------------------------ | ---------------------------------------------------- |
| 0–14                       | Conv、AttentionResiduals、FSNetShuffle、SPPF、C2PSA | 保留                     | 提取局部紋理、形狀與上下文，並未新增陰影約束         |
| 15–17                      | 第一階段 CSAR                                       | 保留                     | 將 P2/P3/P4/P5 融合成 P3/P4/P5                       |
| 18–20                      | 第二階段 CSAR                                       | 保留                     | 再以 P2 與第一階段特徵融合                           |
| 21、24、27、30             | DamageSemanticAttention                             | 新增                     | 各損傷 token 查詢視覺特徵，回傳語意殘差與獨立 logits |
| 22/23、25/26、28/29、31/32 | Index                                               | 新增                     | 分開路由強化特徵與 logits，不新增可學習參數          |
| 33                         | Segment26MultiLabel                                 | 替換原第 21 層 Segment26 | 保留框與 mask 預測，加入多標籤監督介面               |

### 2.3 四個語意分支的對應關係

以下通道數以 **n 尺度、nc=3** 為例；640×640 只是用來解釋尺寸，並非訓練結果。

| 視覺來源                   | 新模組 | 強化特徵／logits 的 Index 層 | stride | 特徵尺寸 | 來源通道 |
| -------------------------- | -----: | ---------------------------- | -----: | -------- | -------: |
| 第 18 層 CSAR              |     21 | 22／23                       |      8 | 80×80    |      128 |
| 第 2 層 AttentionResiduals |     24 | 25／26                       |      4 | 160×160  |       64 |
| 第 19 層 CSAR              |     27 | 28／29                       |     16 | 40×40    |      128 |
| 第 20 層 CSAR              |     30 | 31／32                       |     32 | 20×20    |      256 |

P2 分支來自第 2 層，**沒有先通過 CSAR**。這個淺層捷徑與四尺度 head 在原始 YAML 中就已存在，不能將其歸為本版新增功能。高解析度特徵可能保留凹陷輪廓和弱紋理，也可能保留更多陰影邊界與背景雜訊。

`Index` 的 `[-1, 0]` 表示沿用來源特徵通道數，取 tuple 的第 0 項。`[nc, 1]` 表示輸出 nc 通道，取第 1 項 logits；這裡參數中的 `-1` 並不是額外一條影像支路。

最後一列為：

```yaml
- [[22, 25, 28, 31, 23, 26, 29, 32], 1, Segment26MultiLabel, [nc, 32, 256, 0.5, 2.0]]
```

前四個輸入是 P3/P2/P4/P5 強化特徵，後四個是相同順序的 logits。`32` 是 mask prototype 數，`256` 是 prototype 中間通道設定（n 尺度實際為 64），`0.5` 是多標籤 loss gain，`2.0` 是重疊像素額外權重。保留 P3 作為第一個輸入後，prototype 上採樣得到 stride 4 的 mask。

`Segment26MultiLabel` 使用既有 `Proto26MultiLabel`，不再使用原先由單一 `sem_masks` 圖提供的互斥語意分支。heatmap、seedmap 輔助分支仍保留。原 YOLO 分類本來就使用 sigmoid/BCE，因此這次的重點是類別 token、特徵回饋與可重疊的 dense targets，不能描述成「把整個 YOLO 的 softmax 換成 sigmoid」。

## 3. 未改動的前段公式，如何影響後段辨識

### 3.1 AttentionResiduals：在歷史特徵狀態之間選擇

對同一空間位置 j，各歷史狀態為 \(x_{s,j}\)，通道數為 C：

\[
\hat x_{s,j}=\frac{x_{s,j}}{\sqrt{C^{-1}\sum_c x_{s,j,c}^{2}+10^{-6}}},\quad
\beta_{s,j}=\frac{e^{q^\top\hat x_{s,j}}}{\sum_t e^{q^\top\hat x_{t,j}}},\quad
y_j=\sum_s\beta_{s,j}x_{s,j}.
\tag{2}
\]

先以 RMS 正規化評分用的特徵，再用可學習 q 對歷史狀態評分，softmax 沿「狀態」維度計算，最後混合原始值。q 初始化為零，因此多個狀態初期等權；只有一個狀態時直接回傳。

在影像辨識中，這種混合讓模型選擇不同處理深度的資訊。本專案可藉此保留早期紋理或後期形狀，但 q 不是 Rust/Dent token，沒有逐類別保護機制。RMS 正規化也不等於物理上的去陰影。

### 3.2 FSNetShuffle：通道交換

\[
Y=\operatorname{Conv}_{3\times3}\!\left(\operatorname{Shuffle}_g\!\left[
\operatorname{Resize}(P_1X_1),\ldots,\operatorname{Resize}(P_mX_m)
\right]\right).
\tag{3}
\]

\(P_i\) 為 1×1 Conv 投影，各來源先分配輸出通道、對齊尺寸、串接，再分組交換通道，最後用卷積融合。Shuffle 的核心是 `[g, C/g]` reshape 後轉置成 `[C/g, g]`，不創造新像素資訊。

這個模型的四組 FSNetShuffle 都接收同尺度的兩個來源，例如 `[1,2]`。雖然模組支援跨尺度輸入，此處主要交換 Conv 與 AttentionResiduals 特徵。第 9、12 層的 target=2 會由程式取 `2 % 2 = 0`，指向第一個輸入，並非不存在的第三個輸入；兩來源尺寸相同，所以輸出尺寸不變。

交換後的卷積有機會結合材質與局部結構，對暗處輪廓可能有幫助。但此處沒有亮度不變性公式，也沒有要求不同損傷必須分開保存。

### 3.3 CSAR：在尺度間選擇證據

\[
a_{s,h,j}=\operatorname{softmax}_{s}\left(\frac{q_{h,j}^{\top}k_{s,h,j}}{\sqrt{d_k}}\right),\quad
U_j=\operatorname{Concat}_h\sum_s a_{s,h,j}v_{s,h,j},
\tag{4}
\]

\[
Y=P_o\{U+\operatorname{DWConv}_{3\times3}(V_{\mathrm{target}})\}
+\operatorname{Shortcut}(X_{\mathrm{target}}).
\tag{5}
\]

各來源先 nearest resize 到目標尺度。Q 來自目標尺度，K/V 來自所有來源，softmax 沿「來源尺度」計算。深度卷積補入目標尺度的局部空間資訊，shortcut 保留目標特徵。

以 n 模型的 128 輸出通道為例，4 heads 各有 32 維 value，`attn_ratio=0.5` 使 key/query 每頭為 16 維。這與新增語意模組的每頭 32 維不同。

CSAR 可同時讀取小尺度細節與大尺度上下文；在陰影中，這可能幫助判斷暗邊界是否符合損傷形狀。不過其尺度權重總和為 1，且沒有 Rust/Dent 專用 query，因此不保證兩類線索都保留。這是新增類別查詢的動機之一，並非已證明 CSAR 一定會壓制某類損傷。

## 4. DamageSemanticAttention：每一步的公式與作用

以下省略 batch 下標。輸入 \(X\in\mathbb R^{C\times H\times W}\)，令 \(N=HW\)、類別數 K、embedding 維度 \(D=128\)、heads 數 \(H_a=4\)、每頭維度 \(d=32\)。K 此處表示類別數，attention 的 key 矩陣另寫成 \(K_v\)。

### 4.1 1×1 投影：把不同通道數轉為共同的 128 維

\[
U_j=W_pX_j+b_p,\qquad U\in\mathbb R^{N\times128}.
\tag{6}
\]

對每個特徵位置做通道線性組合，不混合鄰近位置。flatten/transpose 只把 `[C,H,W]` 重排為 `[HW,D]`，不改變位置的對應關係。

一般影像模型用投影對齊不同表示空間。本專案將 P2 的 64 通道擴展、P5 的 256 通道壓縮為 128，以便進行同維度查詢；各尺度投影權重沒有共享。壓縮可能丟失細節，擴展也不等於恢復影像中原本不存在的資訊。

### 4.2 LayerNorm：穩定特徵尺度

\[
\mu_j=\frac1D\sum_r U_{j,r},\quad
v_j=\frac1D\sum_r(U_{j,r}-\mu_j)^2,\quad
Z_{j,r}=\gamma_r\frac{U_{j,r}-\mu_j}{\sqrt{v_j+\epsilon}}+\beta_r.
\tag{7}
\]

這裡對每個位置的 128 個通道正規化，\(\epsilon=10^{-5}\) 為目前 PyTorch LayerNorm 預設。\(\gamma,\beta\) 可學習，控制正規化後的尺度與位移。它不依 batch 的其他影像計算統計量。[Layer Normalization](https://arxiv.org/abs/1607.06450) 提供其一般原理；此處正規化軸以程式的 `LayerNorm(embed_channels)` 為準。

若通道向量整體受到正比例縮放，在忽略 epsilon 時，正規化可抵銷這種尺度變化。但真實陰影還會改變局部對比、色彩、雜訊與前段非線性反應，且投影含 bias，所以不能由這個公式推出整個模型對陰影不變。

### 4.3 類別 token：每類一個可訓練向量

\[
T\in\mathbb R^{K\times128},\qquad
T\ \text{以標準差設定為 }0.02\text{ 的截斷常態初始化}.
\tag{8}
\]

Rust、Dent、Hole 是對 token 列的類別解釋，不是送進模型的文字。每個尺度各有一組 T。四 heads 是不同投影子空間，不代表四個損傷類別，也沒有指定「某 head 專看陰影」。token 的類別意義由對應通道的多標籤監督逐步建立。

### 4.4 Q/K/V：分開「要找什麼」「哪裡相符」「取回什麼」

\[
Q=W_Q\operatorname{LN}(T)+b_Q,\quad
K_v=W_KZ+b_K,\quad V=W_VZ+b_V.
\tag{9}
\]

程式再把 Q 切成 `[4,K,32]`，K/V 切成 `[4,N,32]`。這些切分只重排資料。每個類別 query 都能查詢同一張 feature map 的所有位置。此新增分支沒有額外的座標 embedding，位置資訊主要來自前段 CNN 特徵與後續逐像素對應。

### 4.5 除以 √32：控制點積幅度

\[
S_{h,k,j}=\frac{Q_{h,k}^{\top}K_{v,h,j}}{\sqrt{32}}.
\tag{10}
\]

若 q/k 分量近似獨立且方差為 1，32 項乘積之和的方差約為 32，除以 √32 後回到約 1。這是縮放 attention 分數的量級分析，不是對實際已訓練特徵分布的保證。它可降低 softmax 過早尖銳化的風險。[Scaled Dot-Product Attention](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS1) 是此計算的原始參考。

### 4.6 空間 softmax：每類各自選位置

\[
A_{h,k,j}=\frac{\exp(S_{h,k,j})}{\sum_{t=1}^N\exp(S_{h,k,t})},\qquad
\sum_j A_{h,k,j}=1.
\tag{11}
\]

分母涵蓋影像位置，**不涵蓋其他損傷類別**。Rust 的 attention 分布不會因 Dent 也高分而被強制降低。程式以 float32 計算 softmax，再轉回 value 的資料型別，以改善混合精度下的數值穩定性。

然而同一 Rust query 的不同位置仍會競爭。大面積亮處鏽蝕可能占據較多 attention，使暗處小鏽蝕貢獻下降；全圖 query 不會自動保證對每個 instance 都公平。各類別也仍共享視覺特徵與投影，不能聲稱已完全消除特徵干擾。

### 4.7 取回上下文，再更新 token

\[
C_k=\operatorname{Concat}_{h=1}^4\left(\sum_j A_{h,k,j}V_{h,j}\right),\qquad
T_1=T+W_OC+b_O.
\tag{12}
\]

attention 把全圖視覺證據加權成每個類別的 128 維上下文，再加到原 token。T 保留可學習類別先驗，C 帶入此張影像的資訊。這可讓相同 Dent token 因影像內容而調整，但也可能學到資料集中的背景或照明偏差。

### 4.8 FFN 與 GELU：非線性組合類別證據

\[
\operatorname{FFN}(u)=W_2\operatorname{GELU}(W_1\operatorname{LN}(u)+b_1)+b_2,
\tag{13}
\]

\[
\operatorname{GELU}(a)=a\Phi(a),\qquad
T'=\operatorname{LN}\{T_1+\operatorname{FFN}(T_1)\}.
\tag{14}
\]

\(\Phi\) 是標準常態累積分布。GELU 是平滑的非線性門控，並非單純把所有負數截為零。FFN 先由 128 擴展至 256，再回到 128，分別處理每個 token，沒有直接沿類別軸進行 token self-attention。它提供更複雜的特徵組合能力，但不等於自動形成「陰影判斷規則」。公式與當前 `nn.GELU()` 的精確模式一致。

### 4.9 逐像素 logits：將語意重新對回位置

\[
\ell_{k,j}=\frac{(W_{rq}T'_k+b_{rq})^\top(W_{rk}Z_j+b_{rk})}{\sqrt{128}}+b_k.
\tag{15}
\]

更新後的類別 token 與每個位置的視覺向量比對，得到 `[K,H,W]` logits。這次是在完整 128 維空間計算，因此除以 √128，不是 √32。\(b_k\) 可學習各類別整體的反應偏置。

logit 是尚未轉換為 0–1 的分數，也不是保證校準的信心值。它有空間位置，但尚未分開同類別的不同 instances；實例分離仍交由 YOLO 框與 mask 分支。

### 4.10 獨立 sigmoid：同一位置允許多個高分

\[
R_{k,j}=\sigma(\ell_{k,j})=\frac1{1+e^{-\ell_{k,j}}},\qquad
\frac{\partial R}{\partial\ell}=R(1-R).
\tag{16}
\]

沒有 \(\sum_k R_{k,j}=1\) 的限制。假設某重疊位置的 logits 為 Rust=2、Dent=1.5，則 response 約為 0.881、0.818，可同時為高分。這是示意計算，不是訓練結果。

sigmoid 的輸出彼此不做類別正規化，但參數、輸入與 loss 仍然有耦合。當 logits 絕對值很大時，sigmoid 導數變小。訓練直接使用 logits 版 BCE，可避免先 sigmoid 再取 log 的數值不穩定。

### 4.11 語意特徵回饋：用多類別 response 混合 token

\[
G_j=\frac1K\sum_{k=1}^K R_{k,j}(W_sT'_k+b_s),\qquad
\hat X=X+\alpha\operatorname{Conv}_{1\times1}(G),\quad \alpha_0=0.001.
\tag{17}
\]

Rust 與 Dent 都可對同一像素的語意特徵提供貢獻。分母 K 是類別數，**不是各 response 的總和**，所以這不是跨類別 softmax；除以 K 是控制混合量級，也可能在類別很多、只有少數類別出現時稀釋更新。

1×1 輸出投影把 128 維映回原 C 通道，α 控制加入原特徵的幅度。此 α 是可學習的自由標量，沒有 sigmoid/tanh 限制，可增大、變小或變成負值。初始 0.001 使分支接近小幅修正，但不能保證整個 head 與原模型數值完全一致。

在固定一張影像的 token values 下，\(G=R^TV_s/K\) 的矩陣秩最多為 K；這是額外提供的類別上下文，不能獨自承擔所有細節，因此保留 X 很重要。殘差的通用動機可參考 [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)，本專案的 α 與語意混合方式則以程式為準。

## 5. 重疊監督：從 instance masks 到 loss

### 5.1 multi-hot target：保留每個類別的正標籤

對影像 b、類別 k、位置 j：

\[
Y_{b,k,j}=\max_{i:\,image(i)=b,\,class(i)=k}\widetilde M_{i,j}.
\tag{18}
\]

同類別多個 mask 取聯集，不同類別分別保留。Rust 和 Dent 重疊時目標是 `[1,1,0]`，不需要二選一。\(\widetilde M\) 是已對齊該尺度的 instance mask。

`overlap_mask=False` 保留獨立 mask 張量。設為 True 會合併成 instance ID 圖，讓單一像素難以同時保存兩個 instance，因此此多標籤 loss 明確拒絕該設定。

### 5.2 mask 尺寸對齊：max pooling 的意義與代價

\[
\widetilde M_i(j)=\max_{u\in\Omega_j}M_i(u).
\tag{19}
\]

縮小 mask 時用 adaptive max pooling，只要對應區域中有前景，就保留正標籤；放大時用 nearest interpolation。這能避免小損傷在低解析度標籤中完全消失。

但相鄰且原本不相交的 Rust/Dent，可能分別在同一個粗網格單元內出現，使兩類都被標成 1。因此此版的「重疊加權」指**該尺度 multi-hot target 的共存**，不必然等於原始高解析度 mask 的精確交集。它也不是另一些 0918 YAML 的真實 D∩R 約束。

### 5.3 BCE：每個類別分別判斷存在與否

\[
\operatorname{BCE}(\ell,y)=-y\log\sigma(\ell)-(1-y)\log[1-\sigma(\ell)].
\tag{20}
\]

其等價穩定形式為

\[
\max(\ell,0)-\ell y+\log(1+e^{-|\ell|}),\qquad
\frac{\partial\operatorname{BCE}}{\partial\ell}=\sigma(\ell)-y.
\tag{21}
\]

對 y=1 且 response 偏低的類別，梯度下降會增加 logit；對 y=0 卻高分的純陰影誤報，會降低 logit。能否學到這個區分，取決於資料是否包含標註正確的純陰影負例和陰影內損傷正例。

示意：logits `[2,1.5]` 對正確目標 `[1,1]` 的平均 BCE 為 **0.1642**；若把 Dent 錯誤覆寫為 0，目標 `[1,0]` 的平均 BCE 變為 **0.9142**，而 Dent 的梯度方向會轉為壓低分數。這說明保存重疊標籤比只新增一個 attention 模組更根本。

### 5.4 重疊權重：提高共存區域在 BCE 中的相對比重

\[
O_{b,j}=\mathbf1\!\left[\sum_kY_{b,k,j}>1\right],\quad
w_{b,j}=1+2O_{b,j},
\tag{22}
\]

\[
L_{\mathrm{wBCE}}=
\frac{\sum_{b,k,j}w_{b,j}\operatorname{BCE}(\ell_{b,k,j},Y_{b,k,j})}
{K\sum_{b,j}w_{b,j}}.
\tag{23}
\]

一般位置權重為 1，共存位置為 3。這是**同一尺度內 BCE 項的相對權重 3:1**，不是所有 loss 或最終梯度一律增為三倍。分母隨權重和改變，而每個共存位置的所有類別通道都被加權，包含應維持為負的其他損傷類別。

若重疊標註有誤或粗尺度產生過多共存單元，這個設計也會放大錯誤監督。提高 `cooccurrence_weight` 需同時觀察非重疊區 precision。

### 5.5 Dice：評估每一類的區域吻合

\[
L_{\mathrm{Dice}}=
\frac1{BK}\sum_{b,k}\left[
1-\frac{2\sum_jR_{b,k,j}Y_{b,k,j}+1}
{\sum_jR_{b,k,j}+\sum_jY_{b,k,j}+1}
\right].
\tag{24}
\]

分子衡量預測與真值的交集，分母包含兩者面積，平滑常數為 1。這是專案 `MultiChannelDiceLoss` 的實際形式，分母沒有平方。逐類別計算可讓小面積類別也進入平均，但不能保證解決所有類別不平衡。

本版只有 BCE 使用式（22）的重疊權重，Dice 沒有額外乘該權重。空類別仍會因預測出不必要的前景受到懲罰。

### 5.6 四尺度 loss 與總目標

\[
L_{\mathrm{aux}}=\frac{0.5}{4}\sum_{s\in\{P3,P2,P4,P5\}}
\left(0.5L_{\mathrm{wBCE}}^{(s)}+0.5L_{\mathrm{Dice}}^{(s)}\right).
\tag{25}
\]

最外層 0.5 是 YAML 的 `multilabel_gain`；內層兩個 0.5 是 BCE 與 Dice 的混合比例。每個尺度先各自正規化，再等權平均，因此 P2 雖有更多像素，也不會只因像素數較多而在這個平均中占更高權重。

總目標可概括為

\[
L=L_{\mathrm{box}}+L_{\mathrm{mask}}+L_{\mathrm{cls}}+L_{\mathrm{DFL}}
+L_{\mathrm{heat}}+L_{\mathrm{seed}}+L_{\mathrm{aux}},
\tag{26}
\]

其中前面各項表示已套用各自 gain 的既有 loss；heat/seed 項依對應訓練輸出與目標啟用。多標籤項累加在報表的 `sem_loss` 槽位。程式回傳的訓練 loss 向量另外乘 batch size，不能把每個日誌數值直接視為未加權的式（26）項。

### 5.7 為何主 YOLO 任務也會訓練語意 token

\[
\frac{\partial L}{\partial T}=
\frac{\partial L_{\mathrm{main}}}{\partial\hat X}
\frac{\partial\hat X}{\partial T}
+\frac{\partial L_{\mathrm{aux}}}{\partial\ell}
\frac{\partial\ell}{\partial T}.
\tag{27}
\]

第一條路徑穿過語意殘差，讓 token 為框與 mask 任務提供有用特徵；第二條路徑直接監督每類 response。α 初始很小會縮小第一條路徑的梯度，但第二條不依賴 α。這解釋了為何此分支既能影響主要預測，也有自身的學習訊號。

## 6. 一個連貫的影像辨識過程

假設影像中有「亮處鏽蝕」「陰影內 Rust+Dent 重疊」「純陰影」三種區域。前段卷積、AttentionResiduals、FSNetShuffle 與 CSAR 先建立特徵，P2 捷徑保留較高解析度資訊。

各尺度的 Rust/Dent tokens 分別查詢視覺位置。例如已縮放分數 `[2,1,0]` 的 Rust attention 約為 `[0.665,0.245,0.090]`；Dent 分數 `[0.5,1.5,0]` 的 attention 約為 `[0.231,0.629,0.140]`。兩個 query 可偏好不同區域，也可都讀取重疊區。這些只是計算示例，實際模型的空間位置遠多於三個。

取回影像上下文後，更新過的 tokens 再逐像素產生 logits。重疊位置的 `[1,1,0]` 真值同時要求 Rust 與 Dent 高分，純陰影位置的全零損傷真值要求它們低分。四尺度 BCE/Dice 更新 token 與視覺特徵，語意殘差則改變送入 YOLO head 的表示。

最終 YOLO 仍需預測框、類別與 instance masks。dense response 不會直接變成兩個最終 instances，標準 `Results` 也沒有直接附加四組 response maps。

## 7. 陰影與重疊：可以預期什麼，哪些仍需驗證

### 7.1 陰影下的損傷辨識

作為簡化的成像示意，可寫為

\[
I(j)=a(j)R(j)+\eta(j),\quad 0<a(j)\le1.
\tag{28}
\]

R 是表面反射相關訊號，a 是局部照明衰減，η 是雜訊；此模型只是說明陰影會改變可見訊號，不是現有程式的物理模型。當 a 很小，局部紋理可能低於雜訊，任何後段 attention 都無法保證找回遺失資訊。

本版可能有幫助的路徑是：P2 保存微弱輪廓，token 從上下文辨認暗區是否符合 Dent/Rust，LN 減少某些特徵量級變動。這些均是機制推測。反面影響包括注意力偏向亮處、大面積陰影邊界造成誤報，以及以背景作為損傷捷徑。

若要建立直接的陰影穩定性目標，後續可另做「同一影像的原圖／陰影增強圖」一致性實驗；這不是本 YAML 已有功能，本說明也沒有把該功能加進模型。

### 7.2 重疊損傷辨識

已實作的直接支援包括：每類 query、獨立 sigmoid、multi-hot 目標、共存像素 BCE 加權，以及語意特徵回饋。它們容許 Rust/Dent 在同一區域保留高 response，但不保證最終框和 masks 都正確。

還可能發生共享特徵干擾、instance assignment 對同一候選的分配限制、背景共現捷徑、粗尺度共存標籤擴張，以及後處理丟失第二個類別。增加共存權重可能提高 overlap recall，也可能降低非重疊區 precision，必須一起量測。

### 7.3 推論後處理仍有影響

目前 `DetectionPredictor` 呼叫 NMS 時沒有啟用 `multi_label=True`，NMS 預設每個候選框保留最高分標籤；validator 路徑則傳入 `multi_label=True`。不同候選仍可能分別輸出 Rust 與 Dent，但 dense response 同時高分，不等於同一候選一定輸出兩個標籤。

另外，class-aware NMS 與 `agnostic_nms=True` 對跨類別重疊候選的處理不同。評估時應比較 raw logits、dense maps 與後處理結果，固定 NMS 設定，避免把後處理差異誤認為 feature suppression。

## 8. 成本、可比較性與建議驗證

### 8.1 實際參數量與 attention 張量規模

在本機由現有 YAML 建立 **n 尺度、nc=3** 模型，計算所有可學習參數：

| 指標                   |              數值 | 解讀                                   |
| ---------------------- | ----------------: | -------------------------------------- |
| 原 6CSAR 模型          |         5,527,701 | 包含原 Segment26 語意分支              |
| 新 semantic-token 模型 |         6,296,866 | 新 token 分支及替換後 head             |
| 淨增加                 | 769,165，約 13.9% | 不等於 FPS 或顯存增加比例              |
| 四個新增語意模組合計   |           880,208 | 比淨增量大，因原單標籤語意分支同時移除 |

attention score 張量形狀為 `[B,4,K,HW]`，其儲存量按 \(O(BH_aKN)\) 成長，避免 visual self-attention 的 N×N 矩陣。這不代表整個模組只有這些成本，還包括影像投影、FFN、response、梯度與 optimizer state。

640×640 輸入的四尺度位置數總和為 34,000，其中 P2 為 25,600，約占 75.3%。B=1、K=3 時，四個 attention score 張量合計有 408,000 個元素，單以 float32 儲存約 1.56 MiB；這是單組張量的估算，不是模型峰值顯存，也不是實測 GPU 效能。

### 8.2 驗證應分開報告

| 評估子集           | 主要問題                 | 建議量測                                           |
| ------------------ | ------------------------ | -------------------------------------------------- |
| 陰影內損傷         | 是否漏掉暗處 Dent/Rust   | 各類別 instance recall、mask IoU                   |
| 純陰影、無損傷     | 是否把光照邊界當損傷     | 每張誤報數、損傷預測的 FP 比例                     |
| Rust+Dent 真實重疊 | 兩類是否都保留           | 各類 recall、兩類同時命中的比例、交集 ROI response |
| 非重疊單類損傷     | 是否錯誤學成固定共現     | precision、額外類別誤報                            |
| 亮部對照           | 暗處改善是否犧牲正常辨識 | 相同指標與 confidence 分布                         |

可在固定信心與 IoU 門檻後，定義重疊配對召回率：

\[
\operatorname{PairRecall}=\frac{\#\{\text{GT 重疊配對中兩個 instance 都正確匹配}\}}
{\#\{\text{GT 重疊配對}\}}.
\tag{29}
\]

陰影內召回率與一般 recall 一樣是 \(TP/(TP+FN)\)，但只在事先定義的陰影子集計算。陰影／GT 交集比例與配對規則必須先固定，不能看結果後再調整分組。

基礎 6CSAR 與新模型的差異同時包含 token、語意回饋和多標籤 head/loss。若要辨別原因，可再設相同多標籤 head/loss 但無 token 查詢的對照，或關閉語意回饋而保留 auxiliary 分支。這些是待建立的消融，不是目前已執行的實驗。

維持相同資料切分、模型尺度、seed、訓練輪數與後處理設定。從原權重 warm-start 時，第 0–20 層可依相同鍵值載入，而 head 索引由 21 改為 33，不能假設原 head 自動搬移。現有測試可證明程式建模、loss 與推論介面可運作，不能代替資料集的精度評估。

## 9. 程式來源與閱讀位置

數學推導以本專案實作為主，外部論文只用於一般概念。損傷 response、loss 權重與陰影效益並非那些論文在本資料集上的結論。

- [新 YAML](../ultralytics/cfg/models/11_myself/yolo11-6csar-semantic-token.yaml)：各層來源與參數。
- [原 YAML](../ultralytics/cfg/models/11_myself/yolo11-6csar.yaml)：比較保留的第 0–20 層與四尺度輸入。
- [damage_semantic.py](../ultralytics/nn/modules/damage_semantic.py)：式（6）至（17）。
- [block.py](../ultralytics/nn/modules/block.py)：AttentionResiduals2d、FSNetShuffle、CSAR、Proto26MultiLabel。
- [head.py](../ultralytics/nn/modules/head.py)：Segment26MultiLabel 與原 Detect 的 sigmoid。
- [loss.py](../ultralytics/utils/loss.py)：MultiChannelDiceLoss、v8MultiLabelSegmentationLoss，式（18）至（26）。
- [predict.py](../ultralytics/models/yolo/detect/predict.py)、[val.py](../ultralytics/models/yolo/detect/val.py)、[nms.py](../ultralytics/utils/nms.py)：推論／驗證的 multi_label 與 NMS 行為。
- [HackMD 第二部分](https://hackmd.io/8gre3-RoQsG02dfl_OER_Q)：使用者指定的語意 token 架構概念。

本文所有示意分數均為公式計算，所有陰影與精度影響均標示為待驗證機制。尚無此模型在實際資料集上的增益數據。
