from pathlib import Path
import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'tools/runs/nine_metric_reports/report_20261008_222744_713784/train_and_val/statistics_charts'
OUT=ROOT/'output/pdf'; OUT.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('Chinese','C:/Windows/Fonts/msjh.ttc',subfontIndex=0))
W,H=landscape(A4); margin=36
p=OUT/'nine_metrics_report_20261008.pdf'
c=canvas.Canvas(str(p),pagesize=(W,H)); c.setTitle('九項指標統計報告'); c.setAuthor('')
style=ParagraphStyle('body',fontName='Chinese',fontSize=11,leading=18,textColor=HexColor('#203247'))
page=0
def begin(title,sub=''):
    global page
    page+=1
    c.setFillColor(HexColor('#173f69')); c.setFont('Chinese',21); c.drawString(margin,H-48,title)
    if sub:
        c.setFont('Chinese',10); c.setFillColor(HexColor('#596a7d')); c.drawString(margin,H-70,sub)
    c.setStrokeColor(HexColor('#dce3eb')); c.line(margin,30,W-margin,30)
    c.setFont('Chinese',9); c.setFillColor(HexColor('#596a7d'))
    c.drawString(margin,17,'01_last0912 / S08_diagnostic_only / boxes'); c.drawRightString(W-margin,17,str(page))
def para(text,y,width=None):
    obj=Paragraph(text,style); _,h=obj.wrap(width or W-2*margin,H); obj.drawOn(c,margin,y-h); return y-h-12
def table(rows,y,widths=None,size=10):
    widths=widths or [(W-2*margin)/len(rows[0])]*len(rows[0])
    t=Table(rows,colWidths=widths,repeatRows=1)
    t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Chinese'),('FONTSIZE',(0,0),(-1,-1),size),('BACKGROUND',(0,0),(-1,0),HexColor('#214b75')),('TEXTCOLOR',(0,0),(-1,0),white),('TEXTCOLOR',(0,1),(-1,-1),HexColor('#203247')),('ROWBACKGROUNDS',(0,1),(-1,-1),[white,HexColor('#f0f5fa')]),('ALIGN',(0,0),(-1,-1),'RIGHT'),('ALIGN',(0,0),(0,-1),'LEFT'),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),('LINEBELOW',(0,0),(-1,0),.6,HexColor('#214b75'))]))
    _,h=t.wrap(W-2*margin,H); assert y-h>40,(page,y,h); t.drawOn(c,margin,y-h); return y-h-18
def image(name):
    path=SRC/name; iw,ih=Image.open(path).size
    scale=min((W-2*margin)/iw,(H-135)/ih)
    ww,hh=iw*scale,ih*scale
    c.drawImage(str(path),(W-ww)/2,45+(H-135-hh)/2,width=ww,height=hh)
def pct(v): return 'NA' if pd.isna(v) else f'{v*100:.2f}%'
s=pd.read_csv(SRC/'overall_pooled.csv'); cls=pd.read_csv(SRC/'class_pooled.csv'); stats=pd.read_csv(SRC/'image_descriptive_statistics.csv'); cov=pd.read_csv(SRC/'evaluation_coverage.csv')
metrics=['Accuracy','Precision','Recall','Dice','IoU','mIoU','Mean Dice','FPR','FNR']
begin('九項指標統計報告','依 image_metrics.csv 與 image_class_metrics.csv 整理')
y=para('共 1,050 張圖片，其中 616 張可評估、434 張未評估。以下指標僅使用可評估圖片；NA 保留為無法計算，不補零。',H-92)
rows=[['資料組','總圖片','可評估','未評估','涵蓋率']]+[[r.group,str(r.total),str(r.evaluated),str(r.not_evaluated),pct(r.coverage)] for r in cov.itertuples()]
y=table(rows,y)
rows=[['指標','train','val','train+val']]+[[m]+[pct(v) for v in s[m]] for m in metrics]
y=table(rows,y,size=10)
c.showPage()
for title,name,sub in [('九項指標比較','nine_metrics.png','七項指標採合併像素計數；mIoU、Mean Dice 採合併後類別平均'),('逐類指標熱圖','class_heatmap.png','沿用來源類別代碼；NA 與 0 分開呈現'),('逐圖片指標分布','image_distribution.png','箱體為第 25-75 百分位，中線為中位數；每張有效圖片等權')]:
    begin(title,sub); image(name); c.showPage()
for group in ['train','val','train+val']:
    begin(f'逐類合併指標：{group}','N 真值 = 有真值正像素的圖片數；Precision / Recall 等欄位以百分比呈現')
    part=cls[cls.group.eq(group)]
    rows=[['類別','N 真值','Accuracy','Precision','Recall','Dice','IoU','FPR','FNR']]
    for _,r in part.iterrows(): rows.append([r['class'],str(int(r.truth_positive_images))]+[pct(r[m]) for m in ['Accuracy','Precision','Recall','Dice','IoU','FPR','FNR']])
    y=table(rows,H-94,size=10)
    y=para('各類別先加總 TP、FP、FN、TN 後計算。來源逐類 CSV 的 mIoU 與 Mean Dice 重複記錄所屬圖片的平均值，因此不將它們當成該類別的指標。',y)
    c.showPage()
for group in ['train','val','train+val']:
    begin(f'逐圖片描述統計：{group}','各圖片等權平均；有效 N 依各指標可計算的圖片數決定；標準差採 ddof=1')
    part=stats[stats.group.eq(group)]
    rows=[['指標','有效 N','NA N','平均','標準差','中位數','最小','Q25','Q75','最大']]
    for _,r in part.iterrows(): rows.append([r.metric,str(int(r.valid_n)),str(int(r.NA_n))]+[pct(r[m]) for m in ['mean','std_sample','median','min','q25','q75','max']])
    widths=[100,52,45]+[(W-2*margin-197)/7]*7
    y=table(rows,H-94,widths=widths,size=9)
    para('這裡的平均值直接取自逐圖片數值，與主表「先加總計數再計算」的結果不同。標準差與四分位數描述圖片間差異。',y)
    c.showPage()
begin('計算方法與評估限制')
y=H-90
for text in ['評估基礎為 boxes：將預測框與標記框覆蓋區域轉為各類別的獨立二元遮罩，以像素計算 TP、FP、FN、TN。這些結果屬於框區域評估，不代表精細分割遮罩評估。',
'主表：先分別加總各類別計數，再合併類別計算七項像素指標。mIoU 與 Mean Dice 為加總後各類別 IoU 與 Dice 的等權平均，排除分母為零的類別。train+val 直接合併計數，不平均 train 與 val 的成績。',
'Accuracy = (TP+TN)/(TP+FP+FN+TN)；Precision = TP/(TP+FP)；Recall = TP/(TP+FN)。',
'Dice = 2TP/(2TP+FP+FN)；IoU = TP/(TP+FP+FN)；FPR = FP/(FP+TN)；FNR = FN/(TP+FN)。分母為零時標為 NA。',
'各類別皆為獨立二元遮罩，因此 TN 包含各類別的負像素。Accuracy 容易受到大量負像素影響，需搭配 Recall、Dice 與 IoU 判讀。',
'本批資料僅包含模型 01_last0912 與設定 S08_diagnostic_only。來源程式將此設定標示為診斷設定。類別名稱對照未提供，因此保留 B、C、D、H、O、R、X、U。',
'來源：report_20261008_222744_713784 / train_and_val 中的 image_metrics.csv 與 image_class_metrics.csv。已核對有效圖片的逐类計數加總與逐圖片計數一致，且圖片／類別識別鍵未重複。']:
    y=para(text,y)
c.showPage()
begin('未評估圖片原因','未評估圖片不納入指標統計')
reasons=pd.read_csv(SRC/'not_evaluated_reasons.csv')
labels={'missing_annotation':'找不到對應 XML','ambiguous_annotation':'找到多份 XML，無法確定標記','invalid_annotation':'XML 圖片名稱與來源不一致'}
rows=[['資料組','原因','圖片數']]+[[r.split,labels.get(r.reason_category,r.reason_category),str(r.images)] for r in reasons.itertuples()]
table(rows,H-94,widths=[120,500,W-2*margin-620])
c.save()
print(p); print('Pages:',page)
