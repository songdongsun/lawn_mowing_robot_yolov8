import json,collections
from pathlib import Path
import numpy as np
from analyze_detect_d00 import match,NAMES
from compare_detect_resolution import ious

R=Path(__file__).resolve().parents[1]/'mower_obstrast_check/runs'
out=R/'D10_D02_hardaug_ft60/analysis_20261104'
result={}
for scope in ['full','test140']:
    folder=R/f'd10{scope}_20261104';records=json.loads((folder/'input_manifest.json').read_text())
    misses=[collections.Counter() for _ in range(8)];small=[collections.Counter() for _ in range(8)];fps=[collections.Counter() for _ in range(8)];quality=[collections.Counter() for _ in range(8)]
    for rec in records:
        with np.load(folder/'predictions'/Path(rec['file']).with_suffix('.npz')) as z:
            gt,gc,pred,pc,conf=[z[k] for k in ['gt_boxes','gt_cls','boxes','cls','conf']]
        gc=gc.astype(int);pc=pc.astype(int);sel=conf>=.25;pp=pred[sel];cc=pc[sel];pairs=match(gt,gc,pp,cc)
        hit=set(pairs[:,0]);used=set(pairs[:,1]);ov=ious(gt,pred);ov25=ov[:,sel]
        sizes=((gt[:,2:]-gt[:,:2])*640/max(rec['shape'])).prod(1)
        for i,j in pairs:
            quality[gc[i]]['matched_iou_50']+=1
            if ov25[i,j]>=.75:quality[gc[i]]['matched_iou_75']+=1
        for i,c in enumerate(gc):
            if i in hit:continue
            same=pc==c;good=same&(ov[i]>=.5)
            if (good&(conf>=.25)).any():reason='matching_competition'
            elif good.any():reason='low_confidence'
            elif ((pc!=c)&(ov[i]>=.5)&(conf>=.25)).any():reason='wrong_class_overlap'
            elif (same&(ov[i]>=.1)).any():reason='localization_0_1_to_0_5'
            else:reason='no_same_class_candidate_iou_0_1'
            misses[c][reason]+=1
            if sizes[i]<32**2:small[c][reason]+=1
        for j,c in enumerate(cc):
            if j in used:continue
            v=ov25[:,j];same=gc==c
            if (same&(v>=.5)).any():reason='duplicate_or_matching'
            elif ((gc!=c)&(v>=.5)).any():reason='wrong_class_overlap'
            elif (same&(v>=.1)).any():reason='partial_localization'
            else:reason='no_same_class_overlap_0_1'
            fps[c][reason]+=1
    result[scope]={name:dict(misses=dict(misses[c]),small_misses=dict(small[c]),false_positives=dict(fps[c]),matched_iou=dict(quality[c])) for c,name in enumerate(NAMES)}
(out/'error_decomposition.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
