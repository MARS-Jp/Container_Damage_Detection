# 1001：陰影推論參數調整

本輪使用使用者提供的 `last0912.pt`，比較局部尺寸、提亮、水平翻轉、分類門檻、視圖支持和候選合併規則。原圖基準沿用 predict-2 的 640／conf 0.15／NMS IoU 0.70。所有輸出維持原圖上的遮罩塗色、既有類別顏色和 0.35 透明度。

## 可直接執行的設定

```powershell
& 'C:\Users\USER\anaconda3\envs\yolov11\python.exe' -B tools/predict_tuned_shadow.py
```

程式從 `tools/shadow_tuned_parameters.json` 讀取固定的模型與圖片路徑；後續修改 predict-2 的模型清單，不會讓此參數組誤用另一個權重。每次建立新資料夾，保存 baseline、tuned 各 9 張圖片、原始二值遮罩、預測來源與本次參數。推論過程不讀 XML，也不使用參數檔中回測的 matched_ids、tp 等欄位。

此設定的左右陰影座標專用於 800×600 的 CBHU0708054-A。換圖片需要重新設定區域及驗證，不能直接把這張圖的座標用到其他照片。

| 參數 | D 凹陷 | R 生鏽 |
|---|---|---|
| 局部推論尺寸 | 960，方形 padding | 960，方形 padding |
| 局部 NMS IoU | 0.50 | 0.50 |
| 新增候選信心門檻 | 0.60 | 0.12 |
| 不同視圖支持數 | 至少 2 種 | 至少 1 種 |
| 視圖支持的框 IoU | 0.25 | 0.25 |
| 最小候選遮罩 | 12 像素 | 12 像素 |
| 合併框 IoU | 0.85 | 0.50 |
| 合併方式 | 保留基準預測，再加入候選 | 陰影區內保留較高分完整預測；區外基準保留 |

D 的 0.60 **只篩選局部新增候選**，原圖基準仍使用 0.15。R 的局部替換仍以完整預測為單位，不拼湊遮罩。

兩側各使用原圖、Gamma、CLAHE、水平翻轉四種視圖；Gamma 0.75、CLAHE clip 2.0、暗部亮度門檻 160。原圖加兩側四視圖，共 9 次模型推論。翻轉後的框與遮罩完整映回原座標。候選生成下限 0.005、max_det 1200，最終仍須通過上表 D/R 門檻；同視圖重複候選只計一次支持。

左右檢查區為 x24..432/y120..540 和 x512..680/y420..534；含周邊背景的推論裁切為 x0..472/y60..582 和 x440..720/y150..582。裁切位置固定，不來自 XML 損傷框。沿用既有橘色柱子 R 排除設定。

## 這張圖的回測結果

使用新 XML 的 25 D、69 R，同類別按信心排序一對一配對，**框 IoU 固定 0.50**。這是定位比對，不是分割邊界精度。

| 版本 | D 命中/25 | D 未匹配 | R 命中/69 | R 未匹配 | 左側 D/22 | 左側 R/43 | 右下 D/2 | 右下 R/9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q01 原圖基準 | 2 | 14 | 8 | 34 | 2 | 5 | 0 | 0 |
| Q02 僅調 D | 3 | 16 | 8 | 34 | 3 | 5 | 0 | 0 |
| Q03 僅調 R | 2 | 14 | 14 | 46 | 2 | 10 | 0 | 1 |
| Q04 D/R 分別調整 | 3 | 16 | 14 | 46 | 3 | 10 | 0 | 1 |
| Q05 召回優先診斷 | 5 | 208 | 15 | 137 | 5 | 11 | 0 | 1 |

Q04 保留了基準已匹配的所有 XML 物件，D/R F1 分別從 0.098/0.144 變成 0.136/0.217。未匹配項可能是誤檢、重複或定位偏差；不能把新增遮罩數量當成品質提升。Q05 的多餘遮罩過多，只保存為診斷版本。

原本固定以框 IoU 0.50 去重，會刪掉一個定位較準的高分 D 候選：它與舊 D 預測 IoU 約 0.82，舊預測未匹配 XML #64，新候選可匹配。D 改用合併門檻 0.85，並要求兩種視圖支持，才保留了這個候選。R 則在陰影區允許較高分候選更新舊定位，減少固定保留舊候選造成的誤差。

這張圖同時用於選參數和回測，結果不能視為未見圖片的驗證。XML 只有框，不能證明新增遮罩的像素輪廓全部正確。右側兩個陰影凹陷仍沒有達到固定 IoU 0.50 的匹配標準。

## 實驗與檔案

第一輪跑 4 組局部尺寸/IoU × 5 組門檻/支持，加原圖基準共 21 版（189 張結果）。局部尺寸包括 512、768、960 和 640；另比較方形 padding 及 NMS IoU 0.30。全圖改方形 padding 或水平翻轉沒有改善，因此最終保留原圖基準，僅調整局部候選。

第二輪從保存的模型候選篩選 2,016 組每類門檻／支持數／合併規則，輸出 Q01–Q05 各自資料夾。這一步只篩選已推論的候選，沒有重跑 2,016 次模型，也沒有用 XML 框補畫遮罩。

追加陰影專用測試使用更集中的左右裁切、Gamma 0.55、CLAHE clip 3.0、亮度門檻 180，並比較 960/NMS 0.70、1024/NMS 0.85、640/NMS 0.85。共 7 版（63 張結果），寬鬆 NMS 與強提亮增加了未匹配項。右下 D 的最佳原始候選框 IoU 約 0.479 和 0.439，仍未達 0.50；沒有為了增加命中改動評估門檻。

對追加候選再篩選 1,512 組參數，共輸出 5 個版本。其分類合併版為 D 命中 3／未匹配 15、R 命中 14／未匹配 55；D 少一個未匹配項，但 R 多 9 個，因此保留溫和提亮組作為可執行設定。召回診斷版 D 7／未匹配 163、R 16／未匹配 93，右下 D 仍為 0。兩輪共篩選 3,528 組候選選擇規則。

- [跨兩輪的全圖統整](../runs/segment_shadow_refined/run_20261001_195249_391216/comparisons/final_all.png)
- [跨兩輪的左右區域數據](../runs/segment_shadow_refined/run_20261001_195249_391216/comparisons/final_summary.csv)
- [原圖與 Q04 全圖比較](../runs/segment_shadow_refined/run_20261001_195249_391216/comparisons/selected_all.png)
- [左側陰影比較](../runs/segment_shadow_refined/run_20261001_195249_391216/comparisons/selected_left.png)
- [右下陰影比較](../runs/segment_shadow_refined/run_20261001_195249_391216/comparisons/selected_right.png)
- [Q01–Q05 全版本](../runs/segment_shadow_refined/run_20261001_195249_391216/comparisons/all.png)
- [2,016 組參數篩選表](../runs/segment_shadow_refined/run_20261001_195249_391216/parameter_screen.csv)
- [追加陰影專用測試](../runs/segment_shadow_focused_tuning/run_20261001_200517_806674/README.md)
- [追加候選的分類合併比較](../runs/segment_shadow_focused_refined/run_20261001_200922_181928/README.md)
- [實際重跑的純塗色結果](../runs/segment_shadow_tuned/run_20261001_200119_891338/tuned/CBHU0708054-A__all.jpg)

實際推論已核對：Q04 的 79 個完整預測（19 D、60 R）和所有二值遮罩像素，與保存候選選出的 Q04 完全相同。模型與原圖 hash 保持不變；新流程只修改推論與候選選擇。

保留的 24 個基準預測（所有 D，以及檢查區外的 R）逐框／逐像素未改動；實際重跑的 JPEG 也與 Q04 完全相同。16 個相關單元測試通過，包括翻轉映回、視圖計票、區外保護、篩選與推論一致性及 XML 一對一配對。詳見結果目錄的 verification.json。

重跑第一輪與追加測試：

```powershell
& 'C:\Users\USER\anaconda3\envs\yolov11\python.exe' -B tools/tune_shadow_parameters.py --model 'C:\Users\USER\Downloads\last0912.pt'
& 'C:\Users\USER\anaconda3\envs\yolov11\python.exe' -B tools/tune_shadow_parameters.py --config tools/shadow_focused_profiles.json --model 'C:\Users\USER\Downloads\last0912.pt' --output runs/segment_shadow_focused_tuning
```

從已保存的候選重新篩選：

```powershell
& 'C:\Users\USER\anaconda3\envs\yolov11\python.exe' -B tools/refine_shadow_parameters.py --run-dir runs/segment_shadow_parameter_tuning/run_20261001_140446_557440
```

預設會更新 `tools/shadow_tuned_parameters.json`；比較其他實驗時加 `--no-update-preset`，讓已選定的執行設定保持原樣。每次仍保存該實驗自己的 selected_preset.json。
