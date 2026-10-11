# 四個 0929 YAML 的數學說明：陰影中的凹損與完整損壞範圍

本文逐式對照本專案的實際程式，說明 `00-baseline`、`01-luma`、`02-shallow`、`03-dualproto`。重點是：陰影如何削弱影像線索、各版在哪個階段保留或補充線索、梯度如何到達最終分割，以及可能增加哪些誤報。

這四版沒有輸出獨立的「陰影類別」，也沒有新增陰影標記監督。本文的「陰影辨識」指**辨識陰影中的損壞，並降低把純陰影當成凹損的機會**。公式能解釋運算機制，不能取代訓練後的精度驗證。本文所有手算數值均為示意，不是新模型的辨識結果。

## A. 四個版本與符號

| 版本 | 局部架構改動 | n 規模、八類參數量 | 相較 00 |
|---|---|---:|---:|
| 00-baseline | 原入口、原六個 CSAR、原語意分支與原型解碼 | 6,302,676 | 基準 |
| 01-luma | 00 加入亮度／相對對比／方向梯度入口旁路 | 6,303,638 | +962，約 0.015% |
| 02-shallow | 01 加入 P2/P3 淺層特徵直達預測頭 | 6,378,586 | +75,910，約 1.204% |
| 03-dualproto | 02 加入 P2 原生尺度解碼旁路 | 6,394,044 | +91,368，約 1.450% |

00 是共同基準，沒有新增陰影專用分支。01～03 的新增分支均在訓練與推論時運算。四版保留原 AttentionResiduals、FSNetShuffle、六個 CSAR、四個 DamageSemanticAttention，以及原框、分類、mask coefficient 路徑。四份訓練 recipe 的 loss 與 640 輸入等設定相同，没有啟用 0928 的 training_aux。

| 符號 | 含義與單位 |
|---|---|
| I、R、G、B | 網路輸入影像與 RGB 通道，通常縮放至 0～1 |
| u、v | 目前所討論張量的空間位置，不能直接視為原圖像素 |
| F、S、D、C | 學習式特徵，數值可正可負，不是亮度或機率 |
| σ(z) | sigmoid，將實數映射至 0～1 |
| tanh(z) | 雙曲正切，輸出介於 −1 與 1 |
| ⊙、[A,B] | 逐元素相乘、沿通道拼接 |
| Pₖ、cᵢₖ | 第 k 個共享遮罩原型、第 i 個實例的原型係數 |
| y、p、z | 真值 0/1 遮罩、預測機率、未經 sigmoid 的 logit |
| K=8、M=32 | 類別數、原型數，兩者不同 |

對 640×640 輸入，P2、P3、P4、P5 分別為 160²、80²、40²、20²。以下張量形狀省略 batch 維度。若原圖為 800×600，實際推論會縮放／補邊，特徵位置與原圖位置必須經座標映射。

## B. 共通基礎：陰影、卷積、門控與梯度

### B1. 為何亮度低不能直接等同凹損

用简化成像模型理解：

\[
I(u,v)\approx L(u,v)\rho(u,v)+\eta(u,v).
\tag{B1}
\]

L 是照明，ρ 是表面反射相關量，η 代表雜訊。這是說明用的近似式，未涵蓋所有金屬反光、相機響應與幾何因素，也不是本模型內部估計的量。陰影改變 L，鏽蝕可能改變反射與紋理，凹損又可能因表面方向改變而形成明暗變化；同一個低亮度值可能來自多種原因。

在一個小區域内，若陰影近似一致衰減 a，則 `I′≈aI，0<a<1`。局部亮暗差也約變成原来的 a 倍，弱線索更容易接近雜訊；噪聲不一定同步縮小，因此不能聲稱相對對比就能恢復已丟失訊息。

### B2. Conv、BN、SiLU 的各自作用

本專案大多數 `Conv` 模組包含以下三步：

\[
z_o(u,v)=\sum_{c,r,t}W_{o,c,r,t}X_c(su+r,sv+t)+b_o,
\tag{B2}
\]
\[
\widehat z=\gamma\frac{z-\mu_B}{\sqrt{\sigma_B^2+\epsilon_B}}+\beta,
\qquad \operatorname{SiLU}(z)=z\sigma(z).
\tag{B3}
\]

W 為學習式濾波器，s 為 stride；第一式是常見卷積簡寫，省略 padding 索引。實作中有 BN 的 Conv 不設卷積 bias，平移由 BN 的 β 提供。3×3 聚合鄰近紋理，1×1 在每個位置混合通道並調整通道數。stride=2 同時降低空間解析度，計算量下降，但非常細的形狀可能不再完整保留。

BN 在訓練時按通道統計 batch／空間上的均值與變異，推論使用累積統計；它不會逐個陰影區域估計照明。SiLU 提供非線性且保留部分負響應，讓網路可表達較複雜的明暗組合。它不等於光照校正。程式中 `act=False` 的 Conv 仍含 BN，只省去 SiLU。

### B3. 深度卷積與一般卷積

通道數相同時，本文分支中的 DWConv 按通道做空間卷積：

\[
D_c(u,v)=\sum_{r,t}w_{c,r,t}X_c(u+r,v+t).
\tag{B4}
\]

一般 C→C 的 k×k 卷積約需 k²C² 個權重，depthwise 约為 k²C，這裡省略 BN／bias。DWConv 本身不混合不同通道，前後的 1×1 投影補足通道混合。它讓本次的局部細節分支較輕量，但參數少不代表整體 FPS 一定按相同比例提高。

### B4. sigmoid、tanh 與殘差門控

\[
\sigma(z)=\frac1{1+e^{-z}},\quad \sigma'(z)=\sigma(z)(1-\sigma(z)),
\tag{B5}
\]
\[
\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}},\quad
\tanh'(z)=1-\tanh^2(z).
\tag{B6}
\]

sigmoid 把門控值限制到 0～1；tanh 讓可學習的旁路增益有界且可正可負。門控可抑制或放行細節，但沒有被標成陰影機率。當 sigmoid 或 tanh 飽和，導數會變小，更新可能變慢。

對殘差融合 `F′=F+tanh(α)G⊙D`，固定其餘依賴時：

\[
\frac{\partial F'}{\partial D}=\tanh(\alpha)G,\qquad
\frac{\partial F'}{\partial\alpha}=(1-\tanh^2\alpha)G\odot D.
\tag{B7}
\]

實際 G 也依賴特徵，完整梯度還有門控的鏈式項。原 F 的直接路徑提供一條恒等梯度通道，而新分支仍能學習。在這些新增模組中 α 初始 0.1，門控卷積初始為零，因此 `G=0.5`，初始相乘係數約 `0.5×tanh(0.1)=0.049834`。這不是「整體特徵只改變 5%」的保證，因為 D 與 F 的幅度可能不同。

殘差學習的背景可參考 [ResNet](https://arxiv.org/abs/1512.03385)。本專案使用門控與小增益是具體工程設計，該論文的精度數字不能當作本專案的結果。

## C. 00-baseline：陰影線索如何通過原主架構

### C1. AttentionResiduals：在不同深度狀態間分配權重

令 Xⱼ 為同一區塊內的第 j 個特徵狀態：

\[
\widetilde X_j=\frac{X_j}{\sqrt{\operatorname{mean}_c(X_j^2)+10^{-6}}},\quad
e_j=\sum_cq_c\widetilde X_{j,c},
\tag{C1}
\]
\[
a_j=\frac{e^{e_j}}{\sum_re^{e_r}},\qquad Z=\sum_ja_jX_j.
\tag{C2}
\]

第一步是按位置跨通道的 RMS 正規化，用於計算權重，減輕單純特徵幅度差異的影響。q 是可學習通道查詢；第二式 softmax 沿「深度狀態」正規化，權重總和為 1。用正規化後的特徵計權重，但最後加權的是原 Xⱼ。

q 初始為零，多個狀態初始等權；只有一個狀態時直接回傳。陰影中的低階邊緣可以透過早期狀態保留，但權重並沒有凹損／陰影語意標籤，仍需資料學習。RMS 正規化也不等同還原真實光照。

### C2. FSNetShuffle：投影、通道重排與局部整合

\[
U_j=\operatorname{Resize}(\operatorname{Conv}_{1\times1,j}(X_j)),\quad
F=\operatorname{Conv}_{3\times3}(\operatorname{Shuffle}_2([U_1,U_2])).
\tag{C3}
\]

投影把兩個來源分配到目標通道；拼接後以兩組做 reshape、transpose 再 flatten，讓不同來源的通道交錯。Shuffle 只是通道置換，不產生新的空間細節；後續卷積才學習組合。

本 YAML 的 FSNet 各輸入對是同一解析度的 Conv／AttentionResiduals 特徵，所以這些位置通常不用實際縮放。模組雖支援跨解析度處理，不能把這份 YAML 中每次 FSNet 都描述為 P2/P4 交換。真正跨尺度的融合主要由後續 CSAR 完成。

### C3. CSAR：同一位置比較不同來源尺度

先把來源特徵對齊到目標尺寸，每個 attention head 計算：

\[
Q=W_qF_t,\quad K_s=W_{k,s}\operatorname{Resize}(F_s),\quad
V_s=W_{v,s}\operatorname{Resize}(F_s),
\tag{C4}
\]
\[
e_s(u,v)=\frac{\langle Q(u,v),K_s(u,v)\rangle}{\sqrt{d_k}},\quad
a_s=\operatorname{softmax}_{s}(e_s),\quad Z=\sum_sa_sV_s,
\tag{C5}
\]
\[
F_{out}=\operatorname{Proj}(Z+\operatorname{DWConv}_{3\times3}(V_t))
       +\operatorname{Shortcut}(F_t).
\tag{C6}
\]

W 的簡寫包含程式中的 1×1 Conv／BN。這裡 softmax 沿「來源尺度」進行，不是沿損壞類別，也不是全圖所有像素。內積衡量目標與來源的一致性，√dₖ 限制維度變大時的分數幅度，空間 depthwise 分支補入局部關係。

n 規模、四個 heads 下，128 通道輸出的 dₖ=16，256 通道輸出的 dₖ=32。這不同於下一節語意 token 的固定 32 維 head。六個 CSAR 分成兩輪，每輪形成 P3/P4/P5。陰影中局部線索弱時，較大範圍的資訊可能有幫助；若深層特徵也把陰影當損壞，融合仍可能傳遞錯誤。

### C4. DamageSemanticAttention：類別 token 讀取影像

每一尺度先將視覺特徵投影到 128 維，拉平成空間序列 N=H×W。LayerNorm 作用於每個 token 的通道：

\[
\operatorname{LN}(x)=\gamma\frac{x-\operatorname{mean}(x)}
 {\sqrt{\operatorname{var}(x)+\epsilon_{LN}}}+\beta.
\tag{C7}
\]

LN 穩定通道間尺度，不負責估計陰影照明。八個可學習 damage tokens 依資料類別排列，沒有文字編碼器。四個 heads 各有 32 維：

\[
Q=W_q\operatorname{LN}(T),\quad K=W_kV,\quad U=W_vV,
\quad A=\operatorname{softmax}_{space}(QK^T/\sqrt{32}),
\tag{C8}
\]
\[
T_1=T+W_o\operatorname{ConcatHeads}(AU),\quad
T'=\operatorname{LN}\{T_1+\operatorname{FFN}(T_1)\}.
\tag{C9}
\]

A 對每個類別各自沿空間位置分配注意力。某些明亮位置的可靠凹損紋理可參與更新類別 token，接著影響整張特徵圖的回應；如果陰影中缺乏相關證據，注意力本身不能創造它。

FFN 是 LayerNorm、128→256 的線性投影、GELU、256→128 的投影。GELU 可寫成 `xΦ(x)`，Φ 為標準常態累積分布，提供平滑非線性；它不是額外的機率分類器。縮放點積注意力的背景見 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，本程式的類別 token 與視覺回饋依本地實作定義。

### C5. 獨立類別回應及回饋

\[
\ell_{c,p}=\langle W'_qT'_c,W'_kV_p\rangle/\sqrt{128}+b_c,
\quad r_{c,p}=\sigma(\ell_{c,p}),
\tag{C10}
\]
\[
S_p=\frac1{8}\sum_{c=1}^{8}r_{c,p}W_{sv}T'_c,
\qquad \widehat F=F+\gamma_s\operatorname{Conv}_{1\times1}(S).
\tag{C11}
\]

與 C8 的空間 softmax 不同，C10 對每類各自取 sigmoid，因此 D 和 R 可在同一像素都高分。除以 8 平均類別回饋，γₛ 初始 0.001，且是原語意分支的直接可學習係數，**此處沒有 tanh**。不能把它和 01～03 新分支的 tanh(0.1) 混為一談。

\widehat F 真正送入 YOLO head，語意分支不只是旁觀的熱圖。四個來源為 P3 layer 18、P2 layer 2、P4 layer 19、P5 layer 20；P2 本來就有直接淺層路徑，02 是再加入 FSNet layer 3 的另一來源。

### C6. 原 prototype 與最終 mask

令送入 head 的特徵順序為 P3/P2/P4/P5：

\[
Z=F_{P3}+\sum_{s\in\{P2,P4,P5\}}
 \operatorname{Resize}^{nearest}_{P3}(\operatorname{Conv}_{1\times1,s}(F_s)),
\tag{C12}
\]
\[
P_{old}=\operatorname{Conv}_{1\times1}\!\left[
 \operatorname{Conv}_{3\times3}\!\left(
 \operatorname{ConvTranspose}_{2,2}(\operatorname{Conv}_{3\times3}
 (\operatorname{Conv}_{3\times3,fuse}(Z)))\right)\right],
\tag{C13}
\]
\[
z_i(u,v)=\sum_{k=1}^{32}c_{ik}P_k(u,v),\qquad p_i=\sigma(z_i).
\tag{C14}
\]

C13 按程式依序是 feat_fuse、cv1、upsample、cv2、cv3，Conv 包含其 BN／SiLU。原路徑先在 80×80 融合，再用學習式轉置卷積放大到 160×160。P2 投影後會經 nearest 縮到 P3，未取樣到的局部位置可能流失；轉置卷積有能力學習重建樣式，但不保證找回所有原始細節。

32 張原型是共享的遮罩基底，不是 32 個類別，也不必是一張完整凹損圖。實例係數 cᵢₖ 將它們線性組合，再取 sigmoid。框的定位、分類與候選配對仍影響最終能否找到物件，不能只看原型畫得較細就宣稱凹損召回提高。

**00 的陰影作用與限制：**依靠現有多尺度、類別上下文與 P2 特徵處理低亮度形狀，没有新增相對對比或原生 P2 解碼旁路。它提供共同基準，讓後續版的差異可歸因於新增架構。

## D. 01-luma：從輸入保留陰影中的相對明暗

### D1. 亮度近似

\[
Y=0.299R+0.587G+0.114B.
\tag{D1}
\]

三通道收斂為一張無色彩明暗圖，綠色通道權重較大。此處直接作用於輸入 RGB，沒有額外做線性光反伽瑪校正，因此更精確的說法是灰階／luma 近似，不是物理光度量。保留原 RGB 主路也很重要：鏽蝕顏色仍由原路處理，不會只剩灰階。

### D2. 局部平均與邊界處理

\[
\mu_k(u,v)=\frac1{k^2}\sum_{(r,t)\in\Omega_k}Y(u+r,v+t),
\qquad k\in\{5,15\}.
\tag{D2}
\]

5×5 表達較近鄰的背景，15×15 表達較大範圍的背景，兩者皆在原網路輸入尺寸計算。圖邊使用 replicate padding，把邊界值延伸出去；因此均勻亮面或均勻暗面不會因補零而多出假的邊缘。

### D3. 有號相對對比

\[
q_k=\frac{Y-\mu_k}{\mu_k+\epsilon},\quad
C_k=\tanh(q_k),\qquad\epsilon=1/255.
\tag{D3}
\]

分子衡量比周圍亮或暗多少；保留正負號可區分亮側與暗側。分母把絕對差轉成相對周圍亮度的差，ε 避免除以零，tanh 限制極端值。ε 是加在分母中的穩定項，不是 `max(μ,ε)`，兩者運算不同。

若局部影像同時乘上 a，則：

\[
q'_k=\frac{a(Y-\mu_k)}{a\mu_k+\epsilon}
     =\frac{Y-\mu_k}{\mu_k+\epsilon/a}.
\tag{D4}
\]

當 μ 遠大於 ε/a，q′ 接近 q，表示均勻變暗後相對對比可能較穩定。當環境非常暗，ε/a 變大、相對訊號會縮小；這是抑制極暗噪聲的代價。若陰影邊界穿過 5×5／15×15 區域，a 在區域內不一致，以上近似不成立，純陰影邊界仍會產生高對比。

示意：原 Y=0.20、μ=0.24，變暗為四分之一後 Y′=0.05、μ′=0.06。絕對差由 −0.04 變成 −0.01，經 D3 的相對對比分別約 −0.163 與 −0.155。這僅說明算子的數值行為，沒有证明這個區域是凹損。

### D4. 水平／垂直梯度

\[
K_x=\frac18\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix},
\qquad K_y=\frac18\begin{bmatrix}-1&-2&-1\\0&0&0\\1&2&1\end{bmatrix},
\tag{D5}
\]
\[
g_x=K_x*Y,\quad g_y=K_y*Y,\quad
G_x=\tanh\!\left(\frac{g_x}{\mu_5+\epsilon}\right),\quad
G_y=\tanh\!\left(\frac{g_y}{\mu_5+\epsilon}\right).
\tag{D6}
\]

此處 * 表示程式的空間濾波運算。Sobel 類核對一方向求差，同時在另一方向平滑；1/8 控制響應尺度。保留有號 x/y 梯度，可以描述凹損附近明暗轉折的方向，並未合成梯度大小或推算深度。貨櫃溝槽與投影陰影同樣有方向梯度，需由後續上下文與標記判別。

### D5. 五通道細節與 RGB 主路融合

\[
E=[Y,C_5,C_{15},G_x,G_y],\quad
D=\operatorname{DWConv}_{3\times3}(\operatorname{Conv}_{3\times3,s=2}(E)),
\tag{D7}
\]
\[
F=\operatorname{Conv}_{3\times3,s=2}(I),\quad
G=\sigma(W_g[F,D]+b_g),\quad
F'=F+\tanh(\alpha)G\odot D.
\tag{D8}
\]

n 規模下 F、D 均為 16×320×320，G 為 1×320×320，對所有通道共用同位置的門控。Wg 是 1×1 卷積，能根據原特徵與細節學習位置選擇。D8 與 B7 一起說明新分支如何獲得梯度，原 Conv 的同名權重仍可載入。

局部統計與濾波在 FP32 計算後轉回輸入 dtype，降低黑暗處除法在低精度中的數值風險。這是實作的數值處理，不會增加額外影像資訊。alpha 學習後也可為負，旁路因此可修正或抵消某些訊號。

**01 的影響：**讓低亮度區的相對變化更明確地進入主幹，僅增加 962 個可學習參數。可能同時放大暗處噪聲、陰影界線及板面溝槽；固定局部平均等運算不算入參數量，但仍消耗時間。

## E. 02-shallow：讓局部形狀直接到達預測頭

### E1. 原路與新增路的精確位置

02 繼承 01，head 從八個輸入增至十個。每一個新增融合只處理同解析度的兩個特徵：

| 融合尺度 | 原 head 特徵 | 新增淺層來源 | n 規模來源通道 | 融合後通道與尺寸 |
|---|---|---|---:|---|
| P3 | layer 22 | FSNet layer 6 | 64 | 128×80×80 |
| P2 | layer 25 | FSNet layer 3 | 32 | 64×160×160 |

原 P2 特徵本來就來自 layer 2 的語意分支。新增 layer 3 提供 FSNet 混合後的另一特徵來源；不能描述成原本完全沒有 P2，或所有 P2 原本都經過多輪 CSAR。

### E2. 投影與雙尺度差分

\[
S=\operatorname{Conv}_{1\times1}(S_{raw}),\quad
\Delta_3=S-\operatorname{Avg}_3(S),\quad
\Delta_7=S-\operatorname{Avg}_7(S).
\tag{E1}
\]

1×1 Conv 對齊通道數。S 是經過網路與 BN/SiLU 的特徵，可能含正負數；此處不做 D3 的亮度除法。差分把局部平均成分扣掉，保留局部變化，均勻特徵區則接近零。

它可寫成 `(δ−Bₖ)*S`，δ 為恒等核、Bₖ 為平均核，其直流響應為零，具高通特性。高通也會保留噪聲與規律溝槽，所以還保留完整 S，一併交給後續學習。P2 上 3/7 格約跨 12/28 個輸入像素，P3 約跨 24/56 個輸入像素；這是格距換算，不是包含前面所有卷積的完整感受野。

### E3. 拼接、細節重組與門控

\[
U=[S,\Delta_3,\Delta_7],\quad
D=\operatorname{DWConv}_{3\times3}(\operatorname{Conv}_{1\times1}(U)),
\tag{E2}
\]
\[
G=\sigma(W_g[F,S]+b_g),\qquad
F'=F+\tanh(\alpha)G\odot D.
\tag{E3}
\]

U 通道數為 3C，1×1 將三種訊號學習式融合回 C 通道，DWConv 再整合局部空間。**E3 的 gate 使用 F 與 S，不是 F 與 D**，這與入口分支的實作不同。P3、P2 各有自己的融合模組與可學習增益。

兩個融合後的特徵同時送入框、分類、mask coefficients 與 prototype。語意 logits 仍来自原四個語意分支，不另造一組 shadow logits。梯度能沿新增連接返回 FSNet 的淺層特徵，提供較直接的損壞監督路徑。

### E4. 對陰影與完整範圍的作用

如果深層特徵偏向較強烈的鏽斑而忽略附近微弱凹損，淺層差異提供另一個機會，使模型在分類與定位前就看到局部形狀。它也可能增加只看局部紋理的誤報，因此原深層 F 保留，gate 根據原／淺層特徵學習取捨。

02 相較 01 增加 74,948 個參數，相較 00 累積增加 75,910。這是跨層連接與卷積模組的增加，不是更改 confidence 門檻。側向連接結合細節與語意的背景可參考 [FPN](https://arxiv.org/abs/1612.03144)，本次具體路由依上表本地實作。

## F. 03-dualproto：保留 P2 尺度，補充完整遮罩

### F1. 原型解碼的資訊瓶頸

原路徑 C12 在 P3 80×80 融合，P2 160×160 的訊號會先經 nearest 降採樣，再由原解碼器上採樣。採樣沒選到的細小線索可能流失；03 保留原路並另加一條在 160×160 直接運算的分支。

這個改動**沒有把最終 mask prototype 從 160 改成 320**，也沒有增加偵測尺度。它改變的是 P2 訊息抵達原型的路徑。

### F2. P2 細節與 P4 上下文的通道投影

\[
D=\operatorname{Conv}_{1\times1}(F'_{P2}),\quad
C_0=\operatorname{Conv}_{1\times1}(F_{P4}),\quad
C=\operatorname{Bilinear}_{160\times160}(C_0).
\tag{F1}
\]

F′P2 是 02 淺層融合後的特徵，P4 為原語意增強特徵。n 規模下 D、C 都是 64×160×160；P4 投影前為 128×40×40。YAML 的 `npr=256` 經 n 寬度比例後為 64，而最終輸出原型數仍為 32。

### F3. bilinear 的每個權重在做什麼

對映射到來源網格的連續座標 (x,y)，令左上格為 (x₀,y₀)，a=x−x₀，b=y−y₀：

\[
C(x,y)=(1-a)(1-b)C_{00}+a(1-b)C_{10}
       +(1-a)bC_{01}+abC_{11}.
\tag{F2}
\]

四個權重非負且相加為 1，以相鄰四格估計中間值。`align_corners=False` 採像素中心的映射，單一方向約為 `x_source=(x_output+0.5)×size_in/size_out−0.5`，邊界依框架處理。它讓低解析度上下文較平順地對齊 P2，但不會產生新的真實細節，細節來源仍是直接的 P2。

### F4. context 門控與原型殘差

\[
G=\sigma(W_g[D,C]+b_g),\quad
H=\operatorname{DWConv}_{3\times3}(D+C),\quad
\Delta P=W_oH+b_o,
\tag{F3}
\]
\[
P'=P_{old}+\tanh(\beta)G\odot\Delta P.
\tag{F4}
\]

D+C 讓每個高解析度位置同時接觸細節與較大範圍的語意。Wₒ 是 1×1 線性 Conv2d，將 64 通道變成 32 張原型殘差；它沒有額外 sigmoid，所以 ΔP 可正可負。G 為單通道空間 gate，廣播到 32 個原型，β 初始 0.1。

門控初始為常數 0.5，需訓練後才會形成內容相關的取捨。較大範圍上下文有機會抑制單一溝槽或純陰影邊界，但沒有「看到鏽蝕才允許凹損」的硬規則。上下文與定位結合的背景可參考 [U-Net](https://arxiv.org/abs/1505.04597)，本模型仍使用 YOLO 的原型與實例係數機制。

### F5. 新分支如何影響實例與得到梯度

\[
z_i=\sum_kc_{ik}P'_k
   =\sum_kc_{ik}P_{old,k}
    +\tanh(\beta)G\sum_kc_{ik}\Delta P_k,
\qquad p_i=\sigma(z_i).
\tag{F5}
\]

新分支對每個實例的修正還依賴該實例自己的係數 cᵢ，因此不會把同一個固定區域貼給所有凹損。對單一像素 BCE，`∂ℓ/∂zᵢ=pᵢ−yᵢ`，固定其餘依賴時：

\[
\frac{\partial\ell}{\partial\Delta P_k}
 =(p_i-y_i)c_{ik}\tanh(\beta)G.
\tag{F6}
\]

若漏掉真實損壞，p−y 為負，但 cᵢₖ 可正可負，因此不能直接說「每張原型都會變大」。訓練是共同更新原型、係數與特徵，使最終組合更符合 GT。若不小心把正常背景也標為損壞，同樣的梯度會教模型擴到錯誤區域。

03 相較 02 增加 15,458 個參數，相較 00 累積增加 91,368。原生 P2 分支在較大特徵圖上運算，增加的時間與顯存需要實測，不能只從 +1.45% 參數推算。

**03 的作用邊界：**優先補充分割範圍，並不直接新增一組偵測框或類別分數。共享特徵的訓練可能間接影響偵測，但一個完全沒有正樣本配對的凹損，不能只靠新 prototype 分支保證找回。

## G. 四版共用的监督：為何正確標記仍決定結果

### G1. 最終實例的 BCE

\[
\ell(z,y)=-y\log\sigma(z)-(1-y)\log(1-\sigma(z)),
\quad \frac{\partial\ell}{\partial z}=\sigma(z)-y.
\tag{G1}
\]

實作使用穩定的 logits 形式。對 y=1 的漏掉像素，負梯度促使 logit 上升；對 y=0 的誤塗像素，正梯度促使 logit 下降。此關係是所有新分支學習完整範圍的直接來源之一。

令 Bᵢ 為該正樣本的 GT 框，Aᵢ 為其佔輸入影像的面積比例，HₘWₘ 為 prototype 像素數：

\[
L_{mask}=\frac{\lambda_{seg}}{N_{pos}}
\sum_i\frac{1}{A_iH_mW_m}\sum_{u,v}\mathbf1[(u,v)\in B_i],\ell(z_i,y_i).
\tag{G2}
\]

這是本地 mask loss 的正樣本／框內平均概括式，λseg 使用既有 `hyp.box`，框的位置先映射至 prototype 尺度；框架另有 batch 尺度處理。空目標走原空樣本處理，沒有除以零。i 是正樣本指派，一個實例可對應多個正樣本。若標記只圈凹損局部，loss 就只會教這個局部範圍。

### G2. D/R 多標籤與重疊權重

各類 GT 是該類實例的逐像素聯集，兩類可以同時為 1；縮小語意監督圖時使用 max pooling 保留正例，但粗尺度上也可能把非常接近的物件合併到同一格。

\[
w=1+2\mathbf1[\sum_cY_c>1],\qquad
L_{wBCE}=\frac{\sum_{b,c,u,v}w\,\ell(z_c,Y_c)}{K\sum_{b,u,v}w}.
\tag{G3}
\]

多類共現位置在 BCE 中具有相對 3:1 權重，經分母正規化後不等於總梯度固定放大三倍。每張圖、每個類別的 Dice 項為：

\[
L_{Dice}=\operatorname{mean}_{b,c}\left[
1-\frac{2\sum_{u,v}p_cY_c+1}{\sum_{u,v}p_c+\sum_{u,v}Y_c+1}\right],
\tag{G4}
\]
\[
L_{aux}=0.5\operatorname{mean}_{4\ scales}
              (0.5L_{wBCE}+0.5L_{Dice}).
\tag{G5}
\]

G4 的平滑常數是本程式設定的 1，Dice 沒有另外套用 G3 的像素權重。它衡量前景重合，BCE 提供逐位置監督，二者合併為既有語意項。沒有真實陰影標記時，這些公式也不會自動產生陰影監督。

原框、分類、DFL、heatmap、seedmap 等項維持原設定，本文不把未修改的項說成新方法。0929 也沒有額外加入 0928 的 Tversky extent loss 或成對陰影一致性 loss。

## H. 陰影情境下的預期影響與驗證

| 情境 | 主要涉及的運算 | 可能幫助 | 需排查的副作用 |
|---|---|---|---|
| 暗處但仍有局部明暗差 | D3～D8 | 相對對比、方向變化可進入主幹 | 雜訊與陰影邊界也可能增強 |
| 深層特徵忽略細小凹損 | E1～E3 | 淺層線索直接影響分類、定位與遮罩 | 溝槽或污漬造成局部誤報 |
| 有偵測但只畫一小塊 | F1～F6 | 原生 P2 資訊參與遮罩組合 | 邊界外溢、錯標範圍被強化 |
| 凹損與鏽蝕共存 | C10、G3～G5 | 保留兩類獨立存在機率 | 粗尺度共現不一定是真實像素交集 |
| 完全黑暗或物理不可見 | B1、D4 | 沒有可靠的恢復保證 | 用上下文猜測可能造成假損壞 |

評估時區分陰影中的真正凹損與「純陰影但無損壞」負例。用固定 confidence／IoU 規則做實例一對一配對，另以有精細 GT 的遮罩計算：

\[
\operatorname{Recall}=TP/(TP+FN),\quad
\operatorname{Precision}=TP/(TP+FP),\quad
\operatorname{IoU}=TP/(TP+FN+FP).
\tag{H1}
\]

H1 可用於明確定義的像素或實例計數，但兩種單位不能混用。遮罩召回提高且精確率大跌，可能只是塗得更大。未偵測到的整個物件也必須列入漏檢，不能只評估成功配對者。分母為零時應採用事先約定的評估規則並揭露。

各版需使用相同資料、起始權重與訓練安排。新 XML 只有框，不能直接當作精細 mask；尤其新的 D24/D25 陰影位置與 D16 完整範圍，需要與實際訓練 polygon 一致。這張照片若仍在 train 集，只能作除錯案例，不是獨立精度證據。

## I. 程式與公式對照

| 程式 | 主要公式 |
|---|---|
| [block.py](../ultralytics/nn/modules/block.py)：AttentionResiduals、FSNetShuffle、CSAR、Proto26MultiLabel | C1～C6、C12～C13 |
| [damage_semantic.py](../ultralytics/nn/modules/damage_semantic.py) | C7～C11 |
| [shadow_dent.py](../ultralytics/nn/modules/shadow_dent.py)：LuminanceDetail、DentDetailFusion | D1～D7、E1～E3 |
| [damage_architecture.py](../ultralytics/nn/modules/damage_architecture.py) | D8、F1～F5 |
| [head.py](../ultralytics/nn/modules/head.py)、[loss.py](../ultralytics/utils/loss.py) | C14、G1～G5 |

原模型／配套訓練 YAML 與執行方式見 [0929 架構文件](damage-architecture-0929.md)。93 項程式測試與既有權重移植檢查只證明受測條件下可執行，不代表陰影召回已提高；新分支仍需訓練與資料集驗證。

## J. 每版兩頁簡報

四份簡報各有兩頁，沒有另加封面。頁面呈現主要運算與影響，講者備註收錄對應詳細說明；公式與表格為可編輯內容。

- [00-baseline：原架構與陰影線索](presentations/arch-0929-shadow/00-baseline-shadow-math-2pages.pptx)
- [01-luma：亮度細節入口旁路](presentations/arch-0929-shadow/01-luma-shadow-math-2pages.pptx)
- [02-shallow：淺層特徵直達頭部](presentations/arch-0929-shadow/02-shallow-shadow-math-2pages.pptx)
- [03-dualproto：P2 原生尺度解碼](presentations/arch-0929-shadow/03-dualproto-shadow-math-2pages.pptx)
