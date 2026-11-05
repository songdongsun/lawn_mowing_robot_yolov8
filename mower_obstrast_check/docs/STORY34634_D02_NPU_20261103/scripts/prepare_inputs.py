"""Freeze training-only calibration and the existing 140-image test input tensors."""
import hashlib
import json
from pathlib import Path
import random
import sys

OUT = Path(__file__).resolve().parents[1]
ROOT = next(p for p in OUT.parents if (p/'ultralytics').is_dir())
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from ultralytics.data.augment import LetterBox

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

train = ROOT/'mower_obstrast_check/data/yolo_check'
previous = OUT.parent/'STORY34629_D02_ONNX_20261103'
test = [r for r in json.loads((previous/'comparison/input_manifest.json').read_text(encoding='utf-8')) if r['split']=='synthetic140']
val = json.loads((OUT.parent/'训练与评估/final_D02_20261103/evaluation/D02/full/input_manifest.json').read_text(encoding='utf-8'))
excluded = {r['image_sha256'] for r in test+val}
candidates=[[] for _ in range(8)]
for label in sorted((train/'labels/train').glob('*.txt')):
    areas={}
    for line in label.read_text().splitlines():
        cls,x,y,w,h=map(float,line.split())
        areas[int(cls)]=min(areas.get(int(cls),1.),w*h)
    for cls,area in areas.items(): candidates[cls].append((area,label))
image_paths={p.stem:p for p in (train/'images/train').iterdir()}
rng=random.Random(42)
chosen={}
used=set(excluded)
for cls,items in enumerate(candidates):
    items.sort()
    for quartile in range(4):
        pool=items[len(items)*quartile//4:len(items)*(quartile+1)//4]
        rng.shuffle(pool)
        count=0
        for area,label in pool:
            p=image_paths[label.stem]
            digest=sha(p)
            if digest in used: continue
            used.add(digest)
            chosen[p]=(label,cls,quartile,area,digest)
            count+=1
            if count==4: break
        assert count==4, (cls,quartile)
calib_dir=OUT/'calibration/images'
calib_dir.mkdir(exist_ok=True)
records=[]
for i,(p,(label,cls,q,area,digest)) in enumerate(chosen.items()):
    image=cv2.imread(str(p))
    padded=LetterBox(new_shape=(640,640),auto=False)(image=image)
    dst=calib_dir/f'{i:03d}.png'
    assert cv2.imwrite(str(dst),padded)
    records.append(dict(id=f'{i:03d}',source=str(p.relative_to(ROOT)),source_sha256=digest,
        label=str(label.relative_to(ROOT)),label_sha256=sha(label),selection_class=cls,
        class_area_quartile=q,min_class_area_fraction=area,gray_mean=float(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).mean()),
        prepared_sha256=sha(dst)))
(OUT/'calibration/manifest.json').write_text(json.dumps(dict(seed=42,method='16 unique training images/class; four per class-relative bounding-box area quartile; SHA256 excludes frozen val and test images',images=records),ensure_ascii=False,indent=2),encoding='utf-8')
input_dir=OUT/'board/inputs'
input_dir.mkdir(exist_ok=True)
rows=[]
for i,r in enumerate(test):
    p=ROOT/r['image'];assert sha(p)==r['image_sha256']
    image=cv2.imread(str(p));h,w=image.shape[:2]
    dst=input_dir/f'{i:03d}.png'
    assert cv2.imwrite(str(dst),LetterBox(new_shape=(640,640),auto=False)(image=image))
    scale=min(640/w,640/h)
    # Match Ultralytics scale_boxes inverse convention, including rounding.
    left=round((640-w*scale)/2-.1);top=round((640-h*scale)/2-.1)
    rows.append(dict(r,id=f'{i:03d}',width=w,height=h,scale=scale,left=left,top=top,prepared_sha256=sha(dst)))
(OUT/'board/input_manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'board/input_list.txt').write_text(''.join(f"{r['id']} {r['width']} {r['height']} {r['scale']:.12g} {r['left']} {r['top']}\n" for r in rows))
print(f'Calibration {len(records)} training-only images; test {len(rows)} images',flush=True)
