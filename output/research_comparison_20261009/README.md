# 本研究與其他研究比較表

主表使用核查後的 val（160/174 張）；train 與合併資料另列。
本研究九項數值為方框區域像素代理指標，不能視為完整遮罩分割精度。
其他模型的九項值轉錄自使用者提供圖片；尚未核實資料集、規模、切分和評估方法。
模型身分另依原始論文／作者資料核對：LRFormer 為 IEEE TPAMI 2025，圖片 TIP 已修正。
因此主表沒有同條件排名、最佳值加粗或提升百分比。

- comparison.csv：主表及每列資料來源狀態，保留本機分數的原始精度。
- ours_splits.csv：train、val、train_and_val，含有效筆數與獨立照片數。
- published_benchmarks.csv：原始研究指定版本的 ADE20K / Cityscapes / COCO-Stuff mIoU（%）。
- comparison_table.png：PDF 第 1 頁的完整可分享圖片，附來源與評估限制。
- comparison_manifest.json：輸入 SHA-256、算式驗證、完整原始精度及來源狀態。

來源與證據對應保存在 research/model_comparison_20261009/research/。
這次只整理既有結果，沒有重新推論或訓練任何模型。
重新匯出：python tools/compare_research_metrics.py
PDF 使用 ReportLab；中文字型為 Windows Microsoft JhengHei。
PNG 由 Poppler 渲染 PDF 第 1 頁，應在 PDF 更新後重新產生。
