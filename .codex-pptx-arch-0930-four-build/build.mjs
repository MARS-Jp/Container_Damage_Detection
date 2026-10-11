import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const ROOT='C:/Users/USER/Documents/iim_0905/Yolov11_AttnRes';
const TMP=path.join(ROOT,'.codex-pptx-arch-0930-four-build');
const SKILL='C:/Users/USER/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const {resolvePresentationFont}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const FONT=resolvePresentationFont({fontFamily:'Noto Sans TC'});
const INK='#153447', ACCENT='#007B80', GRAY='#495C69', WHITE='#FFFFFF', PALE='#F1F6F7';
const report=await fs.readFile(path.join(ROOT,'docs/damage-architecture-0930-shadow-math-detailed.md'),'utf8');
const section=(a,b)=>report.slice(report.indexOf(a),b?report.indexOf(b):undefined);
const basics=section('## A.','## C.');
const commonEnd=section('## G.','## K.');
const sources='\n\n程式來源：ultralytics/nn/modules/{damage_architecture.py,shadow_dent.py,damage_semantic.py,block.py,head.py}；ultralytics/utils/loss.py；四個 yolo11-6csar-arch-0929-*.yaml。以 2026-09-30 工作區程式為準。\n原理參考：https://arxiv.org/abs/1706.03762；https://arxiv.org/abs/1512.03385；https://arxiv.org/abs/1612.03144；https://arxiv.org/abs/1505.04597。文獻只用於原理背景，未移用其效能數字。';
function text(s,name,value,x,y,w,h,size=26,color=INK,bold=false){
  const shape=s.shapes.add({geometry:'textbox',name,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  shape.text=value;
  shape.text.style={typeface:FONT,fontSize:size,color,bold,autoFit:'none'};
  return shape;
}
function table(s,name,values,x,y,w,h,widths,size=24){
  const tab=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});
  tab.borders.assign({style:'solid',fill:'#CDDBE0',width:0.8});
  for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
    const cell=tab.getCell(r,c);
    cell.fill=r===0?INK:(r%2===0?PALE:WHITE);
    cell.text.style={typeface:FONT,fontSize:size,color:r===0?WHITE:INK,bold:r===0||c===0};
  }
  return tab;
}
function page(p,title,subtitle,footer,notes){
  const s=p.slides.add(); s.background.fill=WHITE;
  text(s,'title',title,60,32,1160,72,45,INK,true);
  text(s,'subtitle',subtitle,62,111,1156,48,26,GRAY);
  text(s,'footer',footer,62,663,1156,38,22.7,GRAY);
  s.speakerNotes.textFrame.setText(basics+'\n\n'+notes+'\n\n'+commonEnd+sources);
  return s;
}
function h(s,name,value,x,y,w=545){return text(s,name,value,x,y,w,43,29,ACCENT,true);}
function eq(s,name,value,x,y,w=545,size=29){return text(s,name,value,x,y,w,50,size,INK,true);}
const decks=[];

{
  const p=Presentation.create({slideSize:{width:1280,height:720}});
  const notes=section('## C.','## D.');
  let s=page(p,'00 基準架構的陰影線索','原主幹與四個語意分支，作為後續架構的共同對照',
    '陰影辨識目標：找到暗處損壞，並控制純陰影誤報。目前沒有獨立 Shadow 類別。',notes);
  const image=await fs.readFile(path.join(ROOT,'runs/damage_extent_0928/no_scores/000000_CBHU0708054-A/CBHU0708054-A__all.jpg'));
  s.images.add({blob:new Uint8Array(image),contentType:'image/jpeg',alt:'既有 last0926.pt 的無信心數字推論圖，非新架構的訓練結果',fit:'contain',position:{left:62,top:184,width:480,height:360}});
  text(s,'photo-caption','既有 last0926.pt 推論，已隱藏信心分數\n橘色為 D，綠色為 R。此圖用於說明問題。',62,556,495,72,23,GRAY);
  h(s,'imaging-heading','低亮度仍可能包含形狀線索',601,178,610);
  eq(s,'imaging-formula','I ≈ Lρ + η',601,226,610,34);
  text(s,'imaging-meaning','照明 L 與表面反射 ρ 共同影響影像。\n暗區可能來自陰影、形變或材質。',601,280,610,78,26,GRAY);
  h(s,'residual-heading','跨深度狀態的 AttentionResiduals',601,366,610);
  eq(s,'residual-formula','aⱼ = softmax狀態(qᵀ RMS(Xⱼ))',601,413,610,28);
  h(s,'csar-heading','跨來源尺度的 CSAR',601,472,610);
  eq(s,'csar-formula','aₛ = softmax尺度(Q·Kₛ / √dₖ)',601,520,610,29);
  text(s,'csar-meaning','保留目標特徵，再融合尺度與局部線索。\n權重不是經校準的陰影機率。',601,576,610,72,25,GRAY);
  s.speakerNotes.textFrame.setText(basics+'\n\n'+notes+'\n\n'+commonEnd+sources+'\n影像來源：runs/damage_extent_0928/no_scores/000000_CBHU0708054-A/CBHU0708054-A__all.jpg，使用者提供之同一貨櫃影像的既有權重輸出。橘色 D，綠色 R，沒有繪製信心數字。');

  s=page(p,'00 類別查詢與最終實例遮罩','空間注意力、各類獨立回應及原 prototype 解碼',
    '基準參數 6,302,676。四版共用原分割 loss，沒有啟用 0928 的 training_aux。',notes);
  h(s,'token-heading','八個類別 token，各自讀取影像',62,181,560);
  eq(s,'token-attention','A = softmax空間(QKᵥᵀ / √32)',62,233,560,29);
  text(s,'token-attention-note','四個 heads 各 32 維，更新 token T′。',62,290,560,46,25,GRAY);
  eq(s,'token-response','r = σ(Q′K′ᵀ / √128 + b)',62,337,560,30);
  text(s,'token-response-note','D、R 各自取 sigmoid，可同位置共存。',62,392,560,43,25,GRAY);
  h(s,'mask-heading','原型由 P3 融合後上採樣',675,181,543);
  eq(s,'mask-combination','zᵢ = Σₖ cᵢₖPₖ，pᵢ = σ(zᵢ)',675,234,543,29);
  text(s,'mask-note','32 張共享原型，配合每個實例的係數。\nP2 160² 在原路徑先縮至 P3 80²，\n再由轉置卷積產生 160² 原型。',675,296,543,127,26,GRAY);
  table(s,'baseline-effects',[
    ['運算','對陰影凹損的可能作用','仍需驗證的限制'],
    ['多尺度與 token','利用局部形狀及類別上下文','注意力不能補出不可見資訊'],
    ['pᵢ = σ(zᵢ)','將共享原型組合成實例遮罩','仍受候選框、配對與標記影響'],
  ],62,462,1156,160,[260,448,448],25);
  decks.push({name:'00-baseline',p,tables:[2]});
}

{
  const p=Presentation.create({slideSize:{width:1280,height:720}});
  const notes=section('## D.','## E.');
  let s=page(p,'01 亮度細節入口旁路','保留原 RGB Conv，新增五通道明暗與方向訊號',
    '新增 962 個參數。局部統計以 FP32 計算；訊號可用性仍受曝光與雜訊限制。',notes);
  h(s,'five-heading','五通道的來源',62,180,570);
  table(s,'five-signals',[
    ['通道','定義'],
    ['Y','0.299R + 0.587G + 0.114B'],
    ['C₅、C₁₅','tanh((Y−μₖ)/(μₖ+ε))'],
    ['Gₓ','tanh((Kₓ*Y)/(μ₅+ε))'],
    ['Gᵧ','tanh((Kᵧ*Y)/(μ₅+ε))'],
  ],62,231,567,275,[151,416],24);
  text(s,'five-note','μₖ：5×5 或 15×15 局部平均\nKₓ、Kᵧ：有號 Sobel 類方向濾波核\nε = 1/255，圖邊採 replicate padding',62,524,580,117,25,GRAY);
  h(s,'stem-heading','原訊號與新細節相加',681,180,537);
  eq(s,'five-stack','E = [Y, C₅, C₁₅, Gₓ, Gᵧ]',681,232,537,29);
  text(s,'detail-map','E 經 Conv 與 DWConv 形成 D。\nF 是原 RGB Conv 的輸出。',681,290,537,78,26,GRAY);
  eq(s,'stem-gate','G = σ(Wg[F,D] + b)',681,381,537,30);
  eq(s,'stem-residual','F′ = F + tanh(α) G ⊙ D',681,444,537,29);
  text(s,'stem-init','α 初始 0.1，G 初始 0.5。\n初始旁路係數約為 0.0498，\n不代表整體特徵固定只改變 5%。',681,510,537,121,26,GRAY);

  s=page(p,'01 相對對比在陰影中的作用','均勻變暗時可保留部分相對差異，陰影界線仍會產生響應',
    '數值為公式示例，並非訓練結果。門控 G 表示特徵取捨，不是陰影機率。',notes);
  h(s,'example-heading','四分之一亮度的手算示例',62,181,570);
  table(s,'contrast-example',[
    ['數值','原亮度','變暗後'],
    ['Y','0.20','0.05'],
    ['μ','0.24','0.06'],
    ['Y−μ','−0.04','−0.01'],
    ['tanh((Y−μ)/(μ+ε))','−0.163','−0.155'],
  ],62,232,568,280,[310,129,129],23.5);
  text(s,'contrast-result','絕對差縮為 1/4，相對對比仍接近。\n極暗時 ε 的影響變大，訊號也會縮小。',62,534,572,84,25,GRAY);
  h(s,'shadow-formula-heading','局部衰減 a 的影響',686,181,532);
  eq(s,'relative-formula','q′ = (Y−μ) / (μ+ε/a)',686,233,532,31);
  text(s,'relative-meaning','只有鄰域內 a 近似一致，\n且 μ 遠大於 ε/a，才近似保留對比。',686,295,532,82,26,GRAY);
  eq(s,'shadow-edge-formula','∇(aY) = a∇Y + Y∇a',686,400,532,31);
  text(s,'edge-meaning','a∇Y：原表面的明暗變化\nY∇a：照明變化也會形成邊界\n需用上下文與純陰影負例控制誤報。',686,460,532,129,26,GRAY);
  decks.push({name:'01-luma',p,tables:[1,2]});
}

{
  const p=Presentation.create({slideSize:{width:1280,height:720}});
  const notes=section('## D.','## F.');
  let s=page(p,'02 淺層特徵直達預測頭','繼承 01 入口旁路，再加入 FSNet 的 P3 與 P2 來源',
    '02 相較 01 新增 74,948 個參數。新增特徵影響框、分類、係數與原型。',notes);
  h(s,'routes-heading','兩條新增連接',62,181,576);
  table(s,'route-table',[
    ['尺度','原 head 特徵','新增 FSNet'],
    ['P3','layer 22','layer 6'],
    ['P2','layer 25','layer 3'],
  ],62,234,566,196,[112,229,225],25);
  text(s,'routes-note','P3 融合後：128×80×80\nP2 融合後：64×160×160\n以上為 n 規模、640 輸入。',62,459,570,120,26,GRAY);
  h(s,'difference-heading','投影後保留局部差異',680,181,538);
  eq(s,'shallow-project','S = Conv₁×₁(Sraw)',680,232,538,31);
  eq(s,'difference3','Δ₃ = S − Avg₃(S)',680,303,538,31);
  eq(s,'difference7','Δ₇ = S − Avg₇(S)',680,365,538,31);
  text(s,'difference-meaning','S 是學習式特徵，可正可負。\n此處扣掉局部平均，不做亮度除法。\n均勻特徵的差分接近零。',680,432,538,128,26,GRAY);
  text(s,'p2-existing','原模型已含 P2。此版再加入 layer 3 的 FSNet 特徵，保留原語意路徑。',62,606,1155,43,25,ACCENT,true);

  s=page(p,'02 細節門控與純陰影誤報','保留完整淺層特徵 S，同時學習兩種局部差分的用途',
    '相較基準累積增加 75,910 個參數。陰影凹損召回與純陰影誤報須共同評估。',notes);
  h(s,'fuse-heading','淺層融合的實際順序',62,181,576);
  eq(s,'stack','U = [S, Δ₃, Δ₇]',62,233,576,31);
  eq(s,'refine','D = DWConv₃×₃(Conv₁×₁(U))',62,297,576,28);
  eq(s,'gate','G = σ(Wg[F,S] + b)',62,367,576,31);
  eq(s,'residual','F′ = F + tanh(α) G ⊙ D',62,435,576,30);
  text(s,'gate-input-note','此處 gate 讀取 F 與 S。\nP3、P2 各有自己的模組與增益。\n新增連接讓梯度直接返回淺層。',62,503,576,129,26,GRAY);
  h(s,'effect-heading','不同訊號都可能通過差分',686,181,532);
  table(s,'difference-effects',[
    ['區域特徵','差分與辨識影響'],
    ['均勻特徵','局部差分接近零'],
    ['微弱形狀變化','可能保留凹損線索'],
    ['純陰影界線','也可能產生強響應'],
    ['規律溝槽／雜訊','需原特徵與 gate 判別'],
  ],686,235,532,306,[204,328],25);
  text(s,'effect-note','高通線索不能單獨判定損壞。\n訓練標記與陰影負例決定學到的取捨。',686,561,532,82,25,GRAY);
  decks.push({name:'02-shallow',p,tables:[1,2]});
}

{
  const p=Presentation.create({slideSize:{width:1280,height:720}});
  const notes=section('## D.','## G.');
  let s=page(p,'03 原生 P2 的雙路遮罩解碼','繼承 02，保留原解碼器並加入直接處理 P2 的旁路',
    '03 相較 02 新增 15,458 個參數。最終原型仍為 32×160×160，沒有新增偵測尺度。',notes);
  h(s,'decoder-heading','同一輸出解析度，兩種資訊路徑',62,181,587);
  table(s,'decoder-paths',[
    ['路徑','主要過程'],
    ['原路','P3 80² 融合，再放大至 160²'],
    ['新增路','P2 160² 直接投影與局部運算'],
  ],62,233,583,198,[124,459],24.5);
  text(s,'bottleneck-note','原路徑中的 P2 會先縮小至 P3。\n新旁路讓 P2 細節保有原生網格，\n再與原 prototype 殘差相加。',62,461,583,127,26,GRAY);
  h(s,'context-heading','P2 細節與 P4 上下文',696,181,522);
  eq(s,'native-p2','D = Conv₁×₁(F′P2)',696,233,522,31);
  eq(s,'native-p4','C₀ = Conv₁×₁(FP4)',696,302,522,31);
  eq(s,'resize-context','C = Bilinear₁₆₀×₁₆₀(C₀)',696,371,522,29);
  text(s,'native-shapes','n 規模的 D、C 皆為 64×160×160。\nP4 原始尺寸是 128×40×40。\nBilinear 對齊上下文，不創造真實細節。',696,442,522,137,25,GRAY);
  text(s,'native-meaning','細節來自直接的 P2。較大範圍的 P4 context 提供另一種判讀依據。',62,607,1155,43,25,ACCENT,true);

  s=page(p,'03 原型殘差與完整凹損範圍','新增分支經最終實例 mask loss 學習，保留原型與係數的組合方式',
    '完整版相較基準新增約 1.45% 參數。尚未重訓，沒有準確率或 FPS 改善數據。',notes);
  h(s,'proto-heading','門控及可正可負的原型修正',62,181,583);
  eq(s,'proto-gate','G = σ(Wg[D,C] + b)',62,232,583,31);
  eq(s,'proto-delta','ΔP = Wo DWConv(D+C) + bo',62,295,583,28);
  eq(s,'proto-residual','P′ = Pold + tanh(β) G ⊙ ΔP',62,358,583,28);
  text(s,'proto-gate-note','G 為單通道 gate，廣播至 32 個原型。\nβ 初始 0.1，保留原路與新增路的梯度。',62,419,583,82,25,GRAY);
  h(s,'instance-heading','不同實例使用不同原型係數',696,181,522);
  eq(s,'instance-formula','zᵢ = Σₖ cᵢₖP′ₖ，pᵢ = σ(zᵢ)',696,232,522,28);
  eq(s,'bce-derivative','∂ℓ/∂zᵢ = pᵢ − yᵢ',696,295,522,31);
  text(s,'gradient-meaning','真實損壞漏掉時，負梯度推高 logit。\n係數 cᵢₖ 可正可負，因此各原型\n不必全部變亮，最終組合才是目標。',696,359,522,130,25,GRAY);
  table(s,'proto-effects',[
    ['可能作用','影響邊界與驗證重點'],
    ['補充已匹配實例的範圍','完整 polygon 必須正確，避免背景外溢'],
    ['直接監督 prototype 旁路','無法保證找回完全沒有候選框的凹損'],
  ],62,521,1156,129,[340,816],24);
  decks.push({name:'03-dualproto',p,tables:[1,2]});
}

const manifest=[];
for(const {name,p,tables} of decks){
  const dir=path.join(TMP,name); await fs.mkdir(dir,{recursive:true});
  const candidatePath=path.join(dir,'candidate.pptx');
  await (await PresentationFile.exportPptx(p)).save(candidatePath);
  for(let i=0;i<2;i++){
    const slide=p.slides.getItem(i);
    const png=await p.export({slide,format:'png',scale:1});
    await fs.writeFile(path.join(dir,`draft-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
    await fs.writeFile(path.join(dir,`draft-${i+1}.layout.json`),await (await slide.export({format:'layout'})).text());
  }
  manifest.push({name,candidatePath,tables,font:FONT});
  console.log(`Draft ready: ${name}`);
}
await fs.writeFile(path.join(TMP,'manifest.json'),JSON.stringify(manifest,null,2));
