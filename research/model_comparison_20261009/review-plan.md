# Literature review plan

## Review contract

- Working title: 本研究 last0912/S08 與 SegFormer、LRFormer、SegMAN 的分數來源比較
- Review question: 使用者圖片中三篇研究的九項分數是否可核實，並可否與本機核查後的結果作同條件比較？
- Intended reader: 貨櫃損傷分割研究作者與審閱者
- Retrieval cutoff: 2026-10-09（Asia/Taipei）
- Unit of analysis: 三個指定模型家族的正式論文與圖片提供分數；本研究指定 checkpoint / 推論設定 / split 的評估結果
- Meaning of "all": 僅上述三個模型家族，不宣稱窮盡所有分割研究

## Scope

- Include: 正式論文、作者官方程式庫、使用者提供的表格影像、本機原始核查資料
- Exclude: 無法回溯來源的網頁成績；不以跨資料集分數宣稱模型優劣；不猜測紅線遮住的舊 Ours 列
- Date range: 指定論文發表年份 2021 至 2025，核查與本機評估截至 2026-10-09
- Languages: 英文來源，繁體中文說明
- Source types: NeurIPS/CVF/IEEE 論文；作者官方 GitHub；本機 JSON/CSV；使用者圖片
- Known access limits: 圖片未提供資料集、模型大小、切分、評估程式或分数出处；先提問、同時查證，不填造條件

## Comparison axes

1. 模型身分、發表年份、Venue 與正式論文來源
2. 九項分數及其來源（圖片提供、論文報告或本機實測）
3. 資料集、split、模型大小、類別與 GT 型態
4. 像素 micro、類別 macro、物件匹配等分數定義
5. 是否具備同條件比較的依據；缺失資訊、單圖與重複標記的限制

## Search map

| Topic or family | Candidate primary sources               | Status                                   |
| --------------- | --------------------------------------- | ---------------------------------------- |
| SegFormer       | NeurIPS 2021 正式論文                   | 已核對身分與 Table 1(a) 公開基準         |
| LRFormer        | IEEE TPAMI 2025 作者稿 / 作者官方程式庫 | 已修正圖片 TIP 誤植；已核對 Tables 3-5   |
| SegMAN          | CVPR 2025；arXiv v2 / 作者官方程式庫    | CVF 存取失敗；以版本化作者稿核對 Table 2 |

## Planned synthesis

- Central finding: 模型年份與 venue 已核對；圖片九指標的原始實驗未能核實，因此只製作來源明確的描述性並列表，不排名
- Main comparison: 以本研究 val（160 張有效 XML）為主要列，train 與合併資料另列
- Most important disclosure gap: 圖片分數的原始實驗與資料集未知
- Expected limitations: 本研究批次為方框區域像素評估，尚非全量真實分割；37 組照片標記衝突未修復；圖片未註明其他模型規模

## Deliverables

- Manuscript format: 比較表與核查說明（非完整論文；不建立無需求的 LaTeX 稿）
- Evidence tables: 九指標並列表、論文來源核對表、各 split 結果表
- Source ledger path: research/source-ledger.csv
- Claim-evidence matrix path: research/claim-evidence-matrix.csv
- Build and publication path: output/research_comparison_20261009/ 與 output/pdf/；不發布至外部
