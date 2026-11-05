from pathlib import Path
import json,sys,collections,numpy as np
sys.path.insert(0,str(Path('.p2-stage').resolve()));from analyze_detect_d00 import match;from compare_detect_resolution import ious
root=Path('mower_obstrast_check/runs');out=root/'D10_D02_hardaug_ft60/analysis_20261104';result={}
for tag,date in [('D02','20261029'),('D10','20261104')]:
 folder=root/f'{tag.lower()}full_{date}';counts=collections.Counter();cases=[]
 for rec in json.loads((folder/'input_manifest.json').read_text()):
  with np.load(folder/'predictions'/Path(rec['file']).with_suffix('.npz')) as z:gt,gc,pred,pc,conf=[z[k] for k in ['gt_boxes','gt_cls','boxes','cls','conf']]
  gc=gc.astype(int);pc=pc.astype(int);sel=conf>=.25;pred=pred[sel];pc=pc[sel];conf=conf[sel];pairs=match(gt,gc,pred,pc);used=set(pairs[:,1]);ov=ious(gt,pred)
  for j in range(len(pred)):
   if j in used or not len(gt):continue
   i=int(ov[:,j].argmax())
   if ov[i,j]>=.5 and gc[i]!=pc[j]:
    key=f'{gc[i]}->{pc[j]}';counts[key]+=1
    if key in ['4->3','7->3','3->4']:cases.append({'file':rec['file'],'confusion':key,'confidence':float(conf[j])})
 result[tag]={'counts':dict(counts),'target_cases':cases}
(out/'full_confusion_comparison.json').write_text(json.dumps(result,indent=2));print({k:v['counts'] for k,v in result.items()})
