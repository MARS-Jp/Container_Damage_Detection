import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { PresentationFile, FileBlob } from '@oai/artifact-tool';
const ROOT='C:/Users/USER/Documents/iim_0905/Yolov11_AttnRes';
const TMP=path.join(ROOT,'.codex-pptx-arch-0930-four-build');
const SKILL='C:/Users/USER/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const PYTHON='C:/Users/USER/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const manifest=JSON.parse(await fs.readFile(path.join(TMP,'manifest.json'),'utf8'));
const results=[];
for(const item of manifest){
  const finalPath=path.join(ROOT,'docs/presentations/arch-0929-shadow',`${item.name}-shadow-math-2pages.pptx`);
  const result=await finalizePresentation({
    explicitTotalSlideCount:2,
    workspaceDir:ROOT,candidatePath:item.candidatePath,finalPath,
    pythonExecutable:PYTHON,
    integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
    layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
    layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',
      ...item.tables.flatMap(n=>['--require-native-table-slide',String(n)])],
    requiredNativeTableOwnerSlides:item.tables,requiredNativeChartOwnerSlides:[],
    fontPolicy:{basis:'design',families:[item.font]},verifyArtifactToolImport:true,
    receiptPath:path.join(TMP,item.name,'validation.json'),
  });
  const deck=await PresentationFile.importPptx(await FileBlob.load(finalPath));
  for(let i=0;i<2;i++){
    const slide=deck.slides.getItem(i);
    const png=await deck.export({slide,format:'png',scale:1});
    await fs.writeFile(path.join(TMP,item.name,`final-${i+1}.png`),new Uint8Array(await png.arrayBuffer()));
  }
  results.push({name:item.name,finalPath,result});
  console.log(`Finalized and rendered: ${item.name}`);
}
await fs.writeFile(path.join(TMP,'results.json'),JSON.stringify(results,null,2));
