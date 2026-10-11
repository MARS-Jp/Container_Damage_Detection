# 0918 七版 YAML：以「陰影下的損傷辨識」為主軸的數學與影響說明

這份說明對應本專案實際的七個 **yolo11-10csar-dr-trainonly-0918-\*.yaml**，以及 training_aux_0918.py、training_aux_ops.py、loss.py 的實作。七版不是七種新推論網路：YOLO11 + IIMStem + AttentionResiduals + FSNetShuffle + 10 CSAR + 五尺度 MSATMultiLabel + Segment26MultiLabel 均保持 0912 基準模型的形狀與輸出。改變的是訓練資料視角與目標函數。以下的「陰影辨識」意指**辨識陰影中的凹洞等損傷，並降低對光照的敏感性**；目前沒有陰影類別、陰影標註、陰影偵測 head，也沒有直接輸出陰影 mask。

## 1. 共用符號與七版總覽

令輸入 RGB 影像為 \(x_b\in[0,1]^{3\times H\times W}\)，\(b=1,\ldots,B\)；獨立實例真值為 \(M_i\)，類別 \(c_i\)；\(\theta\) 為原 0912 模型的可訓練參數。\(D\) 表凹洞、\(R\) 表生鏽。\(L_0(x,M;\theta)\) 表原來完整的 box、seg、cls、dfl、semantic 等七槽損失，**包括既有的 MSAT 多標籤共存加權**，不是普通、沒有共存設計的 YOLO loss。五個 MSAT 尺度以 \(s=1,\ldots,5\) 表示。

為使公式與程式相符，定義 \(L_B(x)\) 為新增邊界項在正 anchor 數正規化、原 segmentation gain 等既有流程處理後，對單視角總損失的貢獻；\(L_O(x)\) 為五尺度真交集項的平均；\(L_C(x,\tilde x)\) 為五尺度一致性項的平均。邊界 gain 0.1 不是直接乘一個「全圖平均 boundary loss」：實作先加進每個匹配實例，再經既有的 anchor 正規化與 hyp.box。

| YAML 後綴       | 原圖／陰影雙視角 | \(L_C\) gain | \(L_B\) gain | \(L_O\) gain | 單次訓練目標的簡寫                                          |
| --------------- | :--------------: | -----------: | -----------: | -----------: | ----------------------------------------------------------- |
| baseline        |        否        |            0 |            0 |            0 | \(L_0(x)\)                                                  |
| shadow-aug      |        是        |            0 |            0 |            0 | \(\tfrac12[L_0(x)+L_0(\tilde x)]\)                          |
| consistency     |        是        |          0.2 |            0 |            0 | 雙視角監督 \(+0.2L_C\)                                      |
| boundary        |        否        |            0 |          0.1 |            0 | \(L_0(x)+0.1L_B(x)\)                                        |
| overlap         |        否        |            0 |            0 |          0.2 | \(L_0(x)+0.2L_O(x)\)                                        |
| shadow-boundary |        是        |          0.2 |          0.1 |            0 | \(\tfrac12[(L_0+0.1L_B)(x)+(L_0+0.1L_B)(\tilde x)]+0.2L_C\) |
| full            |        是        |          0.2 |          0.1 |          0.2 | 前式兩視角各加 \(0.2L_O\)，再加 \(0.2L_C\)                  |

這七版是**消融對照與組合**，不是上一列的結果接著訓練下一列。要公平比較，從同一份 0912 權重各自初始化。baseline 裏的「0」只代表本次新增機制關閉；原有共存加權仍開啟。

## 2. 原有基礎：獨立實例與多標籤共存，不是本次新增

同圖同類所有實例在位置 \(u\) 的類別真值為
\[
Y_{b,c}(u)=\max_{i:\,\mathrm{image}(i)=b,\;c_i=c}M_i(u)\in\{0,1\}.
\]
因此 \(Y_{b,D}(u)=Y_{b,R}(u)=1\) 是合法的，而不是強迫二選一。這需要訓練設定 **overlap_mask=False**，保留互相獨立的實例 mask；否則相疊位置可能被壓成單一 ID，並在第一次 loss 建立時報錯。標註本身若沒有同一像素 D/R 的兩層 mask，新增約束無從學習。

原 0912 多標籤項已用 \(O_b(u)=\mathbb1[\sum_cY_{b,c}(u)>1]\) 及
\[
w_b(u)=1+2O_b(u)
\]
加權 BCE，故多類共存像素的 BCE 權重是 3、非共存是 1。各尺度的原項是 \(0.5L_{\mathrm{weighted\,BCE}}+0.5L_{\mathrm{Dice}}\)，五尺度平均後以既有 gain 0.5 加入 semantic loss。一般影像辨識中，這讓稀少的類別交疊像素不易被海量背景淹沒；在本專案中，生鏽覆在凹洞上時，D 與 R 都可有前景訊號。但**它只提高該像素兩個單類預測的重要性，並未顯式要求兩者預測的乘積對準真交集**，因此本次 overlap 項仍有可測的額外研究問題。

## 3. 合成陰影：從光度變化而不是類別標籤建立對照

對每張圖獨立抽樣角度 \(\phi_b\sim U(0,2\pi)\)、位置偏移 \(o_b\sim U(-0.5,0.5)\)、最大衰減 \(a_b\sim U(0.15,0.45)\)。把像素座標正規化成 \(r(u),c(u)\in[-1,1]\)，其中 \(r\) 是列、\(c\) 是欄。令
\[
d_b(u)=r(u)\sin\phi_b+c(u)\cos\phi_b-o_b,\qquad
h_b(u)=\sigma\!\left(\frac{d_b(u)}{0.15}\right),\quad
\sigma(t)=\frac1{1+e^{-t}}.
\]
\(d=0\) 是隨機傾斜的明暗分界；sigmoid 使邊緣柔和。0.15 是**正規化座標中的過渡寬度**，不是 0.15 像素。再令 \(q_b\sim\mathrm{Bernoulli}(1.0)\)，陰影衰減圖及新圖為
\[
A_b(u)=q_ba_bh_b(u),\qquad
\tilde x_b(k,u)=\mathrm{clip}\bigl[x_b(k,u)(1-A_b(u)),0,1\bigr],\quad k\in\{R,G,B\}.
\]
同一位置的三色通道共用衰減，**不旋轉、不移動、不縮放**，bbox、mask、heatmap、seedmap 全沿用原標註。若 \(h\approx1\) 且 \(a=0.45\)，當地 RGB 變為約 55%；不是讓每個陰影像素都降低 45%，也不是消除真實圖中原有陰影。

訓練時把原圖 \(B\) 張與陰影圖 \(B\) 張串成 \(2B\) 個視角，只做一次共享模型 forward，再拆開各算完整 GT 損失：
\[
L_{\rm pair}=\frac{L_0(x,M)+L_0(\tilde x,M)}2.
\]
圖像辨識意義是令同一幾何破損在亮與暗的輸入中都被監督；本專案預期可能減少「陰影中的凹洞因低亮度漏檢」與「只憑暗度學會 D」的問題。但它只是一種光度變換，不能模擬任意物體陰影、色偏、反射與立體形變。因同時處理 \(2B\) 個 view，forward 記憶體、運算時間以及 BatchNorm 統計會變；即使 optimizer、LR、epoch 不變，也不能說控制了所有訓練因素。

## 4. 陰影一致性：只跟隨「高信心且符合 GT」的原圖輸出

對既有第 \(s\) 個 MSAT 類別 logit，以 \(t_{s,b,c,u}=\sigma(z^{\rm clean}_{s,b,c,u})\) 作原圖 teacher，\(p_{s,b,c,u}=\sigma(z^{\rm shadow}_{s,b,c,u})\) 作陰影 student。teacher 在這個項上使用 stop-gradient：\(\mathrm{sg}(t)\)。兩個視角仍分別接受 GT 監督；沒有第二份權重、EMA 模型或對比學習投影頭。

令解析度相同的類別真值為 \(Y_s\)，信心門檻 \(\tau=0.7\)。只在
\[
P_s=\{Y_s=1\;\land\;t_s\ge0.7\},\qquad
N_s=\{Y_s=0\;\land\;t_s\le0.3\}
\]
計算平方差 \(e_s=(p_s-\mathrm{sg}(t_s))^2\)。定義 \(\mu(e,A)=\sum_{u\in A}e(u)/\max(|A|,1)\)，\(K_s=\mathbb1[|P_s|>0]+\mathbb1[|N_s|>0]\)，則
\[
L_C^{(s)}=
\frac{\mu(e_s,P_s)+\mu(e_s,N_s)}{\max(K_s,1)},\qquad
L_C=\frac15\sum_{s=1}^{5}L_C^{(s)},\qquad
\Delta L_{\rm sem}=0.2L_C.
\]
\(P_s\) 與 \(N_s\) **各自先平均，再平均存在的組別**，背景多不會自動壓過稀少 D 前景；無合格像素時此項恰為零。一般辨識意義是光照擾動下的輸出穩定性；本專案中，若原圖對 D 判對且信心高，陰影圖 D 機率會被拉近原圖。若原圖對 D 明確漏檢、把它當背景，因與 \(Y=1\) 不符，該點不納入一致性，避免直接複製錯誤；但也代表這些**最困難的漏檢點沒有一致性梯度**，只能靠兩視角的 GT 監督改善。原圖錯把純陰影判為 D 若 GT=0，同樣不會成為此項的正向 teacher。

### 小例子

某 D 真值像素 \(Y=1\)：原圖 \(t=0.9\)、陰影 \(p=0.4\)，納入 \(P\)，單點平方差 \((0.4-0.9)^2=0.25\)。若原圖 \(t=0.1\) 而 GT 仍是 D，因 \(0.1<0.7\)，該點不納入；並非把陰影圖拉向 0.1。某無損傷像素 \(Y=0\)：原圖 \(t=0.2\)、陰影 \(p=0.8\)，納入 \(N\)，促使陰影誤報下降。實際 loss 仍按所有合格像素、前後景、五尺度歸一化，不能把單點 0.25 當作最後總 loss。

## 5. 實例邊界：監督幾何形狀，但不把陰影邊緣當損傷

對 YOLO 已匹配的第 \(i\) 個實例，用既有係數 \(a_{ik}\) 與 prototype \(P_k(u)\) 組合 mask logit：
\[
z_i(u)=\sum_k a_{ik}P_k(u),\qquad p_i(u)=\sigma(z_i(u)).
\]
沒有新增邊界 head。對圖 \(F\) 定義 \(3\times3\) 軟邊界算子
\[
\mathcal B(F)(u)=\max_{v\in{\cal N}_3(u)}F(v)-\min_{v\in{\cal N}_3(u)}F(v).
\]
它是局部膨脹減侵蝕（形態學梯度）：均勻區域接近 0，數值變化的邊緣較大。min/max 對選中的元素可反傳；邊界外用 replicate padding，避免憑空造出整張圖的外框邊界。注意它作用在**mask 機率與 GT mask**，不是直接對 RGB 的明暗梯度求邊；因此陰影線本身並不被標記為損傷邊界。

令 \(E_i^p=\mathcal B(p_i)\)、\(E_i^g=\mathcal B(M_i)\)，先算邊界再裁到「已匹配 bbox 各邊外擴一個 mask pixel」的區域 \(Q_i\)。第一項是邊界 Dice：
\[
L_{\rm eDice,i}=1-
\frac{2\sum_{u\in Q_i}E_i^p(u)E_i^g(u)+\varepsilon}
{\sum_{u\in Q_i}E_i^p(u)+\sum_{u\in Q_i}E_i^g(u)+\varepsilon},\quad
\varepsilon=10^{-6}.
\]
交集越貼合真邊界，loss 越小；其分母同時約束假邊與漏邊。第二項把二元交叉熵集中在 GT 邊界帶：
\[
L_{\rm band,i}=
\frac{\sum_{u\in Q_i}E_i^g(u)\,
\mathrm{BCEWithLogits}(z_i(u),M_i(u))}
{\max(\sum_{u\in Q_i}E_i^g(u),1)}.
\]
其權重不是純二值邊框而是 GT 形態梯度；若預測 mask 初期幾乎平坦、eDice 提供的訊號弱，logit-BCE 仍能直接提供方向。實際加在每個 matched-instance 損失上的是
\[
L_{\rm boundary,i}=\mathbb1\!\left[\sum_{Q_i}E_i^g>0\right]
\frac{L_{\rm eDice,i}+L_{\rm band,i}}2,\qquad
L_{\rm mask,i}^{\rm new}=L_{\rm mask,i}^{\rm old}+0.1L_{\rm boundary,i}.
\]
接著沿用原 segmentation loss 的正 anchor 正規化與 hyp.box。一般辨識意義是區域重疊之外強調輪廓位置；本專案可能使細小凹洞的 mask 形狀較貼近標註，尤其在陰影遮住局部對比時提供幾何目標。不過若 GT 外框粗糙、漏標或邊界本來不清楚，會強化標註誤差；也不保證能產生原本完全沒有 proposal 的 D 實例。此 loss 對所有有實例 mask 的類別生效，並非 D 專用。

## 6. 新增真 D／R 交集：監督「同一像素同時成立」

先在原生 GT mask 解析度求
\[
T_b(u)=Y_{b,D}(u)\land Y_{b,R}(u),\qquad
U_b(u)=Y_{b,D}(u)\lor Y_{b,R}(u).
\]
\(T\) 是真交集，\(U\) 限制觀察範圍。**先交集、後縮圖**：縮圖時對 \(T\) 與 \(U\) 各做 adaptive max-pool（放大則 nearest），保留細小陽性，避免把原本相鄰但沒有交集的 D 與 R 各自縮到同一格，捏造交疊。某影像只有在原生 \(T\) 至少有一個正像素才參與這個新增項；D-only 或相鄰無交集影像的此項為零，原本 \(L_0\) 仍照常訓練。

對現有多標籤 logit \(a=z_D\)、\(b=z_R\)，定義共同成立分數 \(q=\sigma(a)\sigma(b)\)。這是**用兩個 sigmoid 輸出的乘積構造的監督分數**，不是已校準的統計聯合機率。為數值穩定，使用其 logit：
\[
\mathrm{logit}(q)
=\log\frac{\sigma(a)\sigma(b)}{1-\sigma(a)\sigma(b)}
=a+b-\log(1+e^a+e^b)
=a+b-\mathrm{logsumexp}(0,a,b).
\]
例如 \(a=b=0\) 時，單類機率各 0.5、乘積 \(q=0.25\)，所以 joint logit 約為 \(-1.099\)，不是 0。真交集區要求 \(q\to1\)，聯集內「只有一類」區要求 \(q\to0\)。

在有真交集的圖上，\(P_s=T_s\)、\(N_s=U_s\land\neg T_s\)，令 \(e_s=\mathrm{BCEWithLogits}(\mathrm{logit}(q_s),T_s)\)。採與一致性相同的前後景分組平均 \(L_O^{(s)}=[\mu(e_s,P_s)+\mu(e_s,N_s)]/\max(K_s,1)\)，五尺度平均 \(L_O=\tfrac15\sum_sL_O^{(s)}\)，以 0.2 加入 semantic loss。一般辨識意義是模型學習「共現」位置，而不只學兩類各自看起來像什麼；本專案針對鏽蝕覆蓋凹洞，期望幫助 D、R 在真重疊區都保留訊號。這不是「有 R 才能預測 D」的 gate；D-only 仍完全合法。若真交集樣本太少、標註互相排斥或鏽區標註不準，新增 loss 會非常稀疏或學到錯誤關係，且 R 假陽性可能增加，需實測。

## 7. 小公式的梯度、尺度與失敗模式

### 7.1 sigmoid 為何適合柔邊陰影與多標籤

\(\sigma(t)=1/(1+e^{-t})\) 把任意實數壓到 \((0,1)\)。在合成陰影中，\(h=\sigma(d/0.15)\)，有
\[
\frac{\partial h}{\partial d}=\frac{h(1-h)}{0.15}.
\]
因此半平面分界 \(d=0\) 附近變化最大，遠離分界則平滑地趨近 0 或 1；不會像硬二值陰影生成不自然的銳利邊。0.15 越小，過渡越窄、越像硬陰影；越大，過渡越寬、越像漸層照明。這裡 \(d\) 由隨機取樣給出，**不是網路要優化的變數**，導數僅用來理解影像變化。對 MSAT 類別 logit 用 sigmoid，則 D、R 各自有 \((0,1)\) 的分數，可以在同一像素同時為高；若改用互斥 softmax，就不符合本專案 D／R 重疊的標註邏輯。

同色衰減的局部關係是 \(\partial\tilde x(k,u)/\partial x(k,u)=1-A(u)\)，忽略 clip 端點時介於 0.55 與 1；幾何位置完全未變，但暗部紋理對比同步降低。這正對應「凹洞仍在，視覺證據變弱」的難例，也解釋為何增強過強可能讓細微輪廓不可辨、反而傷害 D recall。

### 7.2 BCE 與平方差真正推向哪個方向

對 logit \(z\)、二值標籤 \(y\)，穩定二元交叉熵可寫成
\[
\mathrm{BCEWithLogits}(z,y)=\log(1+e^z)-yz,\qquad
\frac{\partial\,\mathrm{BCEWithLogits}}{\partial z}=\sigma(z)-y.
\]
所以在真 D 邊界像素 \(y=1\) 卻預測很低時，梯度為負，反向更新會拉高 D mask logit；在非 D 邊界像素 \(y=0\) 卻預測很高時，梯度為正，會壓低錯誤的 D mask。邊界帶 \(E^g(u)\) 僅決定哪些像素得到較大權重，**不是把暗色像素設成 D**。eDice 補充輪廓整體對齊，BCE 補充每像素的方向；兩者取平均以避免只做一種。

一致性中的單點平方差 \(e=(p-t)^2\) 對陰影 student logit \(z_s\) 的導數為
\[
\frac{\partial e}{\partial z_s}=2(p-t)p(1-p),\qquad p=\sigma(z_s).
\]
若陰影 D 分數 \(p\) 低於可信原圖 D 分數 \(t\)，梯度為負，會拉高陰影下的 D；若陰影在 GT 背景產生假 D，且原圖給出可信低 D，梯度為正，會壓低陰影假陽性。teacher 的 stop-gradient 使此項**不**因迎合陰影 student 而改動原圖輸出；原圖分支仍透過自己的原始 GT loss 更新共同權重。當 student logit 極端飽和時 \(p(1-p)\) 很小，這項梯度可能弱，完整 GT BCE 監督仍不可省。

分組平均的 \(\mu(e,P)\)、\(\mu(e,N)\) 把「有 D 的少數像素」與「無 D 的大量背景」各視為一組；兩組都存在時，各占約一半的該尺度新增 loss。但這不是重新抽樣資料，也不保證 D 與其他前景類別間完全平衡。

### 7.3 真交集的兩類梯度與潛在牽制

設 \(\ell=a+b-\log(1+e^a+e^b)\) 為 D/R 共同分數的 logit，則
\[
\frac{\partial\ell}{\partial a}
=1-\frac{e^a}{1+e^a+e^b}
=\frac{1+e^b}{1+e^a+e^b}>0,
\qquad
\frac{\partial\ell}{\partial b}
=\frac{1+e^a}{1+e^a+e^b}>0.
\]
再結合 BCE 對 \(\ell\) 的導數 \(\sigma(\ell)-T\)：真交集 \(T=1\) 時，若共同分數偏低，兩類 logit 均收到往上的訊號；聯集內只有 D 或只有 R 時 \(T=0\)，共同分數偏高會收到往下的訊號。這就是新增項同時牽動兩類的數學原因，也說明若真交集標註不準，可能把 R 或 D 推向錯誤方向，須同時看 R precision 與 D-only recall。沒有真交集的圖此項為零，因此不會憑此項把「D 必須有 R」寫入每張圖。

### 7.4 五尺度平均、gain 與原有損失槽

對五尺度的 \(L_C^{(s)}\) 與 \(L_O^{(s)}\) 先平均，再分別乘 0.2；不因尺度數從一變五，就直接把新增 loss 放大五倍。新增邊界項乘 0.1 後放入單實例 mask loss，經原正 anchor 數與原 segmentation gain 處理，所以其實際總梯度大小也受 batch 中正樣本數與原訓練設定影響。訓練顯示的 sem_loss 會含新一致性／交集，seg_loss 會含新邊界；不能只比較七版顯示數字的絕對大小來判斷模型好壞。0.2、0.1、陰影深度與 \(\tau=0.7\) 均是起始超參數，並非由這些數學式證明的最佳值。

## 8. 七版各自回答甚麼問題

1. **baseline**：重現 0912 核心圖與既有多標籤 loss 的對照。它沒有新的陰影視角；若 D 在陰影下漏檢，baseline 只靠原資料中的陰影例子學習。
2. **shadow-aug**：只加成對光度增強，回答「更多陰影曝光與 GT 監督本身是否有效」。不能把效果歸給一致性。
3. **consistency**：與 shadow-aug 相同的雙視角，額外 \(0.2L_C\)，回答「在曝光數一致下，輸出穩定約束有無額外收益」。可能提升陰影 D recall，也可能因可靠 teacher 太少而收益小。
4. **boundary**：只加 \(0.1L_B\)，回答「精確實例輪廓是否改善暗處細微凹洞」。沒有人工陰影，因此不能單獨宣稱實現光照不變。
5. **overlap**：只加 \(0.2L_O\)，回答「原有共存加權之外，真 D／R 聯合監督是否有用」。只影響含真交集圖，不能代表所有陰影場景。
6. **shadow-boundary**：雙視角 \(+0.2L_C+0.1L_B\)，是針對「暗處還要看形狀」最直接的組合；若 D 位在鏽蝕內，新 overlap 約束仍是關閉的，原本共存加權則繼續存在。
7. **full**：雙視角 \(+0.2L_C+0.1L_B+0.2L_O\)，檢驗三個問題是否互補。若較差，可能是資料標註、loss 尺度、梯度衝突或訓練成本問題，不能僅憑單張圖判定哪個機制造成。

## 9. 評估與研究結論應怎麼寫

- **主要終點**：把有可靠 GT 的 D 實例按「陰影內／亮部」、「有 R 重疊／無 R 重疊」分層，報 D recall、mask AP 或 IoU；同時報 R precision、正常浪板與純陰影上的 D 假陽性。若沒有陰影 GT，可先制定**人工分層標記規則**；不要把合成陰影的 attenuation map 冒稱真實陰影 GT。
- **控制條件**：七版相同 0912 checkpoint、資料切分、影像大小、optimizer／LR、batch 的原始圖片數、epoch、seed 與閾值；記錄雙視角實際曝光數、GPU 記憶體與時間。比較 consistency 對 shadow-aug 才能隔離一致性項；比較 full 對 shadow-boundary 才能隔離新增交集項。
- **loss 解讀**：訓練的 seg_loss 可能含 boundary、sem_loss 可能含 overlap/consistency，故七版 train loss 絕對值不可直接當優劣。驗證階段無合成陰影，也不加入本次新正規化項；應看共同定義的驗證指標。
- **範圍限制**：七版都不直接「辨識陰影類別」；也不做「損傷與陰影重疊就懲罰」，因真 D 本來可以在陰影內。合成陰影可能造成亮部 D 精度下降或正常暗紋假陽性，邊界項可能放大標註噪音，交集項可能提升 D 同時傷 R precision。這些都是可檢驗風險，**目前 CPU／合成測試只證明流程可運行，不是實際提升的證據**。

原始實作：ultralytics/utils/training_aux_ops.py、ultralytics/utils/training_aux_0918.py、ultralytics/utils/loss.py；配置：ultralytics/cfg/models/11_myself/yolo11-10csar-dr-trainonly-0918-*.yaml；訓練與限制：docs/training-only-0918.md。
