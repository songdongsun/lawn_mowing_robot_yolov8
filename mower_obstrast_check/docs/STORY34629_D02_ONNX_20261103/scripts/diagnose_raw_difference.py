"""Locate every >0.01px candidate difference without changing acceptance gates."""
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
ROOT = next(p for p in OUT.parents if (p/'ultralytics').is_dir())
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

torch.set_num_threads(6)
model = YOLO(OUT/'best.pt').model.float().eval().fuse()
so = ort.SessionOptions()
so.intra_op_num_threads=6
so.inter_op_num_threads=1
session=ort.InferenceSession(str(OUT/'best.onnx'),so,providers=['CPUExecutionProvider'])
summary=json.loads((OUT/'comparison/summary.json').read_text())
manifest=json.loads((OUT/'comparison/input_manifest.json').read_text())
failed={(r['split'],r['file']) for r in summary['failed_images']}
rows=[]
for item in manifest:
    p=ROOT/item['image']
    if (item['split'],p.name) not in failed:
        continue
    image=cv2.imread(str(p))
    padded=LetterBox(new_shape=(640,640),auto=False)(image=image)
    x=np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255
    with torch.inference_mode():
        a=model(torch.from_numpy(x))[0].numpy()
        b=session.run(None,{'images':x})[0]
    box_delta=np.max(np.abs(a[0,:4]-b[0,:4]),axis=0)
    scores=np.maximum(a[0,4:].max(axis=0),b[0,4:].max(axis=0))
    bad=box_delta>.01
    kept=scores>=.25
    rows.append(dict(split=item['split'],file=p.name,exceeding_candidates=int(bad.sum()),
                     max_score_of_exceeding_candidates=float(scores[bad].max()) if bad.any() else None,
                     exceeding_candidates_above_conf=int((bad&kept).sum()),
                     max_box_delta_above_conf=float(box_delta[kept].max()) if kept.any() else 0))
result=dict(images=len(rows),candidates=sum(r['exceeding_candidates'] for r in rows),
            candidates_above_conf=sum(r['exceeding_candidates_above_conf'] for r in rows),
            max_score_of_exceeding_candidates=max(r['max_score_of_exceeding_candidates'] for r in rows if r['max_score_of_exceeding_candidates'] is not None),
            max_box_delta_above_conf=max(r['max_box_delta_above_conf'] for r in rows),rows=rows)
(OUT/'comparison/raw_difference_diagnosis.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
