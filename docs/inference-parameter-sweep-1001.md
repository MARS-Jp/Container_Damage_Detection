# 第二輪推論參數比較

批次程式每次讀取 `tools/predict-2.py` 的模型、已啟用圖片、基準尺寸、信心度、IoU、裝置與 retina mask 設定。
`--suite focused` 選用第二輪的 10 組測試；`--compare` 會在推論完成後，自動生成全圖、D、R 及中下部放大比較圖。
結果依時間、版本、模型與來源分開存放，單組失敗會記錄並繼續。

```powershell
& "C:\Users\USER\anaconda3\envs\yolov11\python.exe" `
  "C:\Users\USER\Documents\iim_0905\Yolov11_AttnRes\tools\predict_parameter_sweep.py" `
  --suite focused --compare
```

| 版本 | 調整項目（其餘繼承基準） | 比較目的 |
|---|---|---|
| V01 baseline | 無 | 保留相同顯示方式的基準 |
| V02 size576 | imgsz=576 | 比較稍小尺度 |
| V03 size704 | imgsz=704 | 比較稍大尺度 |
| V04 size768 | imgsz=768 | 比較接近原圖寬度的尺度 |
| V05 nms085 | IoU=0.85 | 查看放寬同類 NMS 後的候選與重複覆蓋 |
| V06 d008_r020 | D=0.08，R=0.20 | 保留較低分 D，限制較低分 R |
| V07 d010_r025 | D=0.10，R=0.25 | 較保守的類別門檻組合 |
| V08 d010_r020_nms050 | D=0.10，R=0.20，IoU=0.50 | 配合更強的同類抑制 |
| V09 size704_d010_r020_nms050 | V08 + imgsz=704 | 比較尺寸與類別門檻的組合 |
| V10 conservative025_nms030 | conf=0.25，IoU=0.30 | 保留較高分且較少重複的候選 |

單獨執行少數版本可加 `--versions baseline d008_r020 size704_d010_r020_nms050`。
先檢查指令與路徑可加 `--dry-run`，不會推論或產生結果檔。
`--reference-run` 可指向前次測試資料夾；預設只加入前次 `imgsz960`，可用 `--reference-versions` 指定其他版本。

結果預設位於 `tools/runs/segment_by_class/parameter_tests_round2/run_時間/`：

- 各版本資料夾：類別圖、合併圖、參數、執行日誌；啟用比較時另有每張圖片的預測統計。
- `summary.csv`：每組執行狀態與耗時。
- `comparisons/`：5 種比較 PNG、`comparison_summary.csv`、說明及參數表。

`D=0.08 R=0.20` 是**保留預測的信心門檻**。推論先使用所有類別中最低的門檻，再依類別同步篩選框與遮罩；
其餘類別仍依全域門檻篩選。模型權重、遮罩二值化門檻及既有填色方式均未改動。
預設最多保留 300 個候選；這次所有版本均未達到上限。

覆蓋率使用同類遮罩的聯集計算，重疊部分只計一次；D/R 重疊仍各自保留。
覆蓋率或預測數量增加不等於準確率提升，仍須對照正確標註確認。
