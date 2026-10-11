import fs from 'node:fs/promises';
import { Presentation } from '@oai/artifact-tool';
import { resolvePresentationFont } from 'file:///C:/Users/USER/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations/container_tools/artifact_tool_utils.mjs';
const font=resolvePresentationFont({fontFamily:'Noto Sans TC'});
console.log('FONT '+font);
const p=Presentation.create({slideSize:{width:1280,height:720}});
const s=p.slides.add();s.background.fill='#FFFFFF';
const t=s.shapes.add({geometry:'textbox',position:{left:60,top:60,width:1120,height:100},fill:'none',line:{fill:'none',width:0}});
t.text='陰影中的凹損　σ(z)　μ₅　ΔP　Σₖ cᵢₖPₖ';t.text.style={typeface:font,fontSize:40,color:'#142735'};
try {
  const png=await p.export({slide:s,format:'png',scale:1});
  await fs.writeFile(new URL('./smoke.png',import.meta.url),new Uint8Array(await png.arrayBuffer()));
  console.log('Rendered PNG');
} catch(e) { console.log('RENDER ERROR '+e.stack); }
