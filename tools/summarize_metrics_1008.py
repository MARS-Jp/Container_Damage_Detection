from pathlib import Path
import sys, json
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pixel_metrics import summarize_counts, METRIC_COLUMNS, COUNT_COLUMNS

BASE = Path(__file__).resolve().parent / 'runs/nine_metric_reports/report_20261008_222744_713784/train_and_val'
OUT = BASE / 'statistics_charts'
OUT.mkdir(exist_ok=True)
a = pd.read_csv(BASE / 'image_metrics.csv')
b = pd.read_csv(BASE / 'image_class_metrics.csv')
keys = ['model','version','split','image_id']
assert not a.duplicated(keys).any()
assert not b.duplicated(keys+['class']).any()
valid = a[a.pixel_status.eq('ok')]
cv = b[b.pixel_status.eq('ok')]
check = cv.groupby(keys)[list(COUNT_COLUMNS)].sum().sort_index()
assert np.array_equal(check.to_numpy(), valid.set_index(keys)[list(COUNT_COLUMNS)].sort_index().to_numpy())
groups = {'train':['train'], 'val':['val'], 'train+val':['train','val']}
summary, classes, stats, coverage = [], [], [], []
for group,splits in groups.items():
    allrows=a[a.split.isin(splits)]; rows=valid[valid.split.isin(splits)]; cr=cv[cv.split.isin(splits)]
    counts={c:{k:int(v) for k,v in part[list(COUNT_COLUMNS)].sum().items()} for c,part in cr.groupby('class')}
    overall, byclass=summarize_counts(counts)
    summary.append({'group':group,'images':len(allrows),'evaluated':len(rows),'not_evaluated':len(allrows)-len(rows),**overall})
    for c,v in byclass.items():
        part=cr[cr['class'].eq(c)]
        classes.append({'group':group,'class':c,'evaluated_images':len(part),'truth_positive_images':int((part.TP+part.FN>0).sum()),'pred_positive_images':int((part.TP+part.FP>0).sum()),**v})
    for m in METRIC_COLUMNS:
        v=rows[m].dropna()
        stats.append({'group':group,'metric':m,'valid_n':len(v),'NA_n':len(rows)-len(v),'mean':v.mean(),'std_sample':v.std(),'median':v.median(),'min':v.min(),'q25':v.quantile(.25),'q75':v.quantile(.75),'max':v.max()})
    coverage.append({'group':group,'total':len(allrows),'evaluated':len(rows),'not_evaluated':len(allrows)-len(rows),'coverage':len(rows)/len(allrows)})
s=pd.DataFrame(summary); c=pd.DataFrame(classes); st=pd.DataFrame(stats); cov=pd.DataFrame(coverage)
for name,df in [('overall_pooled',s),('class_pooled',c),('image_descriptive_statistics',st),('evaluation_coverage',cov)]:
    df.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
missing=a[a.pixel_status.ne('ok')].copy()
missing['reason_category']=missing.pixel_reason.str.extract(r'(missing_annotation|ambiguous_annotation|invalid_annotation)',expand=False).fillna('other')
reasons=missing.groupby(['split','reason_category'],dropna=False).size().reset_index(name='images')
reasons.to_csv(OUT/'not_evaluated_reasons.csv',index=False,encoding='utf-8-sig')

fontpath='C:/Windows/Fonts/msjh.ttc'
def font(size): return ImageFont.truetype(fontpath,size)
def canvas(title,w=1800,h=1250):
    img=Image.new('RGB',(w,h),'white'); d=ImageDraw.Draw(img); d.text((40,24),title,font=font(32),fill='#173f69'); return img,d
colors=['#2463a0','#e18c35','#278776']
img,d=canvas('九項指標：框區域像素評估（合併計數；mIoU / Mean Dice 為類別平均）')
for k,m in enumerate(METRIC_COLUMNS):
    x=45+(k%3)*590; y=100+(k//3)*375
    d.text((x,y),m+(' ↓' if m in ['FPR','FNR'] else ' ↑'),font=font(26),fill='#203247')
    vals=s[m].to_numpy()*100; top=max(vals)*1.25 if m=='FPR' else 110
    for tick in range(5):
        yy=y+290-tick*60; d.line((x+70,yy,x+530,yy),fill='#dce3eb'); d.text((x,yy-13),f'{top*tick/4:.1f}%',font=font(18),fill='#667788')
    for j,v in enumerate(vals):
        xx=x+115+j*150; yy=y+290-v/top*240
        d.rectangle((xx,yy,xx+85,y+290),fill=colors[j]); d.text((xx-5,yy-28),f'{v:.2f}%',font=font(20),fill='#203247'); d.text((xx-8,y+305),s.group.iloc[j],font=font(20),fill='#203247')
img.save(OUT/'nine_metrics.png')
img,d=canvas('各類別合併像素指標（百分比；類別代碼沿用來源）',1800,800)
for k,group in enumerate(groups):
    x=35+k*590; d.text((x,85),group,font=font(28),fill='#173f69')
    part=c[c.group.eq(group)].set_index('class').reindex(['B','C','D','H','O','R','X','U'])
    data=part[['Precision','Recall','Dice','IoU']].to_numpy()*100
    for j,label in enumerate(['Precision','Recall','Dice','IoU']): d.text((x+65+j*125,135),label,font=font(20),fill='#203247')
    for i,label in enumerate(part.index):
        yy=175+i*65; d.text((x+10,yy+15),label,font=font(23),fill='#203247')
        for j in range(4):
            v=data[i,j]; shade=0 if np.isnan(v) else v/100
            fill=tuple(int(lo+(hi-lo)*shade) for lo,hi in zip((237,246,249),(29,81,130)))
            xx=x+65+j*125; d.rectangle((xx,yy,xx+120,yy+60),fill=fill)
            d.text((xx+20,yy+15),'NA' if np.isnan(v) else f'{v:.1f}',font=font(23),fill='white' if v>65 else '#18283b')
img.save(OUT/'class_heatmap.png')
img,d=canvas('逐圖片分布：箱體為第 25–75 百分位，中線為中位數；每張有效圖片等權',1800,700)
for k,m in enumerate(['Dice','IoU','Recall']):
    x=40+k*590; d.text((x+180,90),m,font=font(28),fill='#173f69')
    for tick in range(6):
        yy=570-tick*80; d.line((x+80,yy,x+530,yy),fill='#dce3eb'); d.text((x+10,yy-10),f'{tick*20}%',font=font(20),fill='#667788')
    for j,g in enumerate(['train','val']):
        v=valid[valid.split.eq(g)][m].dropna().to_numpy()*100; q1,med,q3=np.quantile(v,[.25,.5,.75]); iqr=q3-q1
        lo=v[v>=q1-1.5*iqr].min(); hi=v[v<=q3+1.5*iqr].max(); xx=x+200+j*220
        py=lambda n:570-n*4
        d.line((xx,py(lo),xx,py(hi)),fill=colors[j],width=3)
        d.rectangle((xx-50,py(q3),xx+50,py(q1)),fill='#edf4fa',outline=colors[j],width=3)
        d.line((xx-50,py(med),xx+50,py(med)),fill='#c25d17',width=4)
        for vv in [lo,hi]: d.line((xx-25,py(vv),xx+25,py(vv)),fill=colors[j],width=3)
        for vv in v[(v<lo)|(v>hi)]: d.ellipse((xx-3,py(vv)-3,xx+3,py(vv)+3),outline=colors[j])
        d.text((xx-30,590),g,font=font(24),fill='#203247')
img.save(OUT/'image_distribution.png')

def table(df,percent_cols):
    d=df.copy()
    for col in percent_cols:
        d[col]=d[col].map(lambda v:'NA' if pd.isna(v) else f'{v*100:.2f}%')
    return d.to_html(index=False,border=0,escape=True)
notes='''<p>模型：01_last0912；設定：S08_diagnostic_only（來源程式標示為診斷設定）。評估基礎為 boxes：以預測框與標記框覆蓋的區域計算像素指標，不代表精細分割遮罩評估。</p>
<p>主表先在各類別加總 TP、FP、FN、TN，再合併類別計算 Accuracy、Precision、Recall、Dice、IoU、FPR、FNR。mIoU 與 Mean Dice 為加總後各類別 IoU 與 Dice 的等權平均，排除分母為零的類別。train+val 直接合併計數。各類別皆為獨立二元遮罩，因此 TN 包含各類別的負像素，Accuracy 可能因負像素多而很高。</p>
<p>描述統計直接平均 image_metrics.csv 的有效逐圖片數值，NA 不補零；不同指標可能有不同有效樣本數。標準差採樣本標準差（ddof=1）。image_class_metrics.csv 的 mIoU、Mean Dice 是所屬圖片的平均值，不能當作個別類別表現，逐類表只呈現七項可獨立計算的指標。未評估圖片另列，不納入任何成績計算。</p>'''
html='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>九項指標統計</title><style>body{font:16px 'Microsoft JhengHei',sans-serif;color:#203247;background:#f3f6fa;margin:0}main{max-width:1400px;margin:auto;padding:32px}h1,h2{color:#173f69}section{background:white;padding:24px;margin:22px 0;border-radius:10px;overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap;font-size:14px}th{background:#214b75;color:white}th,td{padding:11px;text-align:right;border-bottom:1px solid #dce3eb}img{width:100%;height:auto}p{line-height:1.8}a{color:#2463a0}</style><main><h1>九項指標統計與圖表</h1>'''+notes
html+='<section><h2>評估涵蓋率</h2>'+table(cov,['coverage'])+'<h2>未評估原因</h2>'+table(reasons,[])+'</section>'
html+='<section><h2>整體指標（合併計數）</h2>'+table(s[['group','images','evaluated','not_evaluated',*METRIC_COLUMNS]],METRIC_COLUMNS)+'<img src="nine_metrics.png"></section>'
html+='<section><h2>逐類指標</h2><img src="class_heatmap.png">'+table(c[['group','class','evaluated_images','truth_positive_images','pred_positive_images','Accuracy','Precision','Recall','Dice','IoU','FPR','FNR']],['Accuracy','Precision','Recall','Dice','IoU','FPR','FNR'])+'</section>'
html+='<section><h2>逐圖片描述統計</h2><img src="image_distribution.png">'+table(st,['mean','std_sample','median','min','q25','q75','max'])+'</section>'
html+='<section><h2>計算定義與資料來源</h2><p>Accuracy=(TP+TN)/(TP+FP+FN+TN)；Precision=TP/(TP+FP)；Recall=TP/(TP+FN)；Dice=2TP/(2TP+FP+FN)；IoU=TP/(TP+FP+FN)；FPR=FP/(FP+TN)；FNR=FN/(TP+FN)。分母為零時記為 NA。</p><p>來源：上一層 image_metrics.csv、image_class_metrics.csv。類別代碼未附名稱對照，保留原代碼。</p><p>CSV 下載：'+ ' ／ '.join(f'<a href="{n}.csv">{n}</a>' for n in ['overall_pooled','class_pooled','image_descriptive_statistics','evaluation_coverage','not_evaluated_reasons'])+'</p></section></main></html>'
(OUT/'report.html').write_text(html,encoding='utf-8')
print(s[['group','evaluated',*METRIC_COLUMNS]].to_string(index=False))
print('Verified: image/class confusion counts reconcile; no duplicate keys.')
print('Output:',OUT)
