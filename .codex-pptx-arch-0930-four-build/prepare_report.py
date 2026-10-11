from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / "docs/damage-architecture-0929-shadow-math.md"
target = root / "docs/damage-architecture-0930-shadow-math-detailed.md"
content = source.read_text(encoding="utf-8")
for old, new in {
    "没有": "沒有",
    "简化": "簡化",
    "區域内": "區域內",
    "原来的": "原來的",
    "约為": "約為",
    "恒等": "恆等",
    "边缘": "邊緣",
    "邊缘": "邊緣",
    "证明": "證明",
    "来自": "來自",
    "监督": "監督",
    r"K=W_kV": r"K_v=W_kV",
    r"QK^T": r"QK_v^T",
    r"\mathbf1[(u,v)\in B_i],\ell": r"\mathbf1[(u,v)\in B_i]\,\ell",
}.items():
    content = content.replace(old, new)
content = content.replace(
    "## A. 四個版本與符號",
    "本版核對日期：2026-09-30。所有公式以本地 YAML 與 Python 實作為準；章節 B～G 是完整逐式說明，最後附四份各兩頁簡報。\n\n## A. 四個版本與符號",
)
content = content.replace(
    "A 對每個類別各自沿空間位置分配注意力。",
    "C8 的 V 是經 1×1 投影、flatten／transpose 與 C7 的 LayerNorm 後的視覺序列。Kᵥ 表示 key 矩陣，避免與類別數 K=8 混淆；程式中的 Linear 含 bias，式中 W 表示完整仿射投影。A 對每個類別各自沿空間位置分配注意力。",
)
marker = "殘差學習的背景可參考"
extra = r"""門控本身也需要學習。令 G=σ(z_g)，固定 D 及其他依賴時：

\[
\frac{\partial F'}{\partial z_g}
 =\tanh(\alpha)D\odot G(1-G).
\tag{B8}
\]

初始 G=0.5 時，G(1−G)=0.25，讓 gate 有更新機會。若 gate 後來接近 0 或 1，梯度可能較小；因此不能把初始中性門控當成已經學會分離陰影的證據。原路、細節路與 gate 共享上游時，完整反向傳播會把所有依賴項相加。

"""
content = content.replace(marker, extra + marker)
marker = "示意：原 Y=0.20"
extra = r"""若衰減係數 a 隨位置變化，連續近似的乘積微分為：

\[
\nabla(aY)=a\nabla Y+Y\nabla a.
\tag{D4b}
\]

第一項含原表面的明暗變化，第二項來自照明變化。即使表面沒有凹損，只要陰影邊界讓 ∇a 不為零，也會有梯度。因此 Sobel 或高通響應只是候選線索，後續上下文、正確標記與純陰影負例才決定是否當作損壞。這是分析用的連續近似，實際程式使用下節的離散濾波核。

"""
content = content.replace(marker, extra + marker)
marker = "## J. 每版兩頁簡報"
extra = r"""## J. YAML 參數、空間尺度與數學運算的對應

四個檔是逐步累加的架構對照。`00-baseline` 沒有新增分支，01 加入口細節，02 再加兩條淺層直達連接，03 再加 prototype 旁路。各版模型檔如下：

- [00-baseline.yaml](../ultralytics/cfg/models/11_myself/yolo11-6csar-arch-0929-00-baseline.yaml)
- [01-luma.yaml](../ultralytics/cfg/models/11_myself/yolo11-6csar-arch-0929-01-luma.yaml)
- [02-shallow.yaml](../ultralytics/cfg/models/11_myself/yolo11-6csar-arch-0929-02-shallow.yaml)
- [03-dualproto.yaml](../ultralytics/cfg/models/11_myself/yolo11-6csar-arch-0929-03-dualproto.yaml)

01 的 `[-1,1,LuminanceResidualStem,[64,3,2,0.1]]` 表示接前一輸入、建立一次新 stem、名義 64 通道、3×3 卷積、stride 2、初始 α=0.1。n 模型的通道按以下方式縮放：

\[
C_{actual}=8\left\lceil\frac{\min(C_{yaml},1024)\times0.25}{8}\right\rceil.
\tag{J1}
\]

因此入口實際為 16 通道，prototype 中間通道 256 變為 64。式中的向上取整讓通道保持 8 的倍數；它控制計算容量，不是陰影強度。類別數 `nc` 由資料集覆寫為 8，原型數 32 不按這個寬度比例縮放。一般重複次數大於 1 的模組使用 `max(round(n×0.5),1)`；此為四版共用的 n 深度設定。

02／03 的 head 輸入為 `[22,25,28,31,23,26,29,32,6,3]`。前四路是特徵、接著四路是類別 logits、末兩路是新增淺層來源。`[nc,32,256,0.5,2.0,0.1]` 的末三個數分別是原有語意 loss gain、原有共現額外權重與新增分支增益初值；前三者決定輸出類別與原型容量。

這四版沒有建立新 Shadow 類別，沒有將陰影像素作為另一個 GT mask。01 的 G、02 的 G、03 的 G 都是任務內部門控，其值不能解讀為經校準的陰影機率。新分支必須訓練，只有 YAML 不會使既有 last0926.pt 自動學到新功能。

### 手算範例：分割修正不等於全部原型變亮

只看一個像素，若原 logit z=−1，則 p=σ(−1)≈0.269。假設學習後的原型殘差經係數組合，令該像素的 logit 增加 0.2，則 p′=σ(−0.8)≈0.310。這表示機率改變，不代表已越過顯示遮罩的切分門檻，更不等於物件 confidence 變成 0.310。

對真值 y=1、p=0.1 的漏掉像素，BCE 的 logit 梯度是 −0.9。若某原型係數 cᵢₖ=−0.5、β=0.1、G=0.5，F6 的該原型梯度約為 +0.0224。梯度下降會降低這個原型值，乘上負係數後反而使最終 logit 提高。所有數值均為公式示例，不是實測改善。

### 三種數值應分開閱讀

| 數值 | 意義 | 是否直接等同另一個數值 |
|---|---|---|
| 內部 gate G | 控制新增特徵的相對貢獻 | 不是陰影分類機率 |
| 實例像素 pᵢ=σ(zᵢ) | 像素屬於某個實例的估計 | 不等於物件 confidence |
| 顯示／篩選用 confidence | 物件候選的分類分數 | 隱藏文字不改變篩選門檻 |

## K. 每版兩頁簡報"""
content = content.replace(marker, extra)
target.write_text(content, encoding="utf-8")
print(target)
