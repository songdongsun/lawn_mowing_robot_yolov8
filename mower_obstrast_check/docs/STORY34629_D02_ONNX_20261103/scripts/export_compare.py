"""Reproduce STORY34629 static-square FP32 export and paired inference audit."""
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
ROOT = next(p for p in OUT.parents if (p / 'ultralytics').is_dir())
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes
from reference_match import match

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

torch.set_num_threads(6)
source = ROOT / 'mower_obstrast_check/docs/训练与评估/final_D02_20261103/best.pt'
assert sha(source) == '388902aeff6d18f3b8313fef38a5ad548a4099abc73ec7d256fe7b4798339d5f'
shutil.copy2(source, OUT / 'best.pt')
options = dict(format='onnx', imgsz=640, batch=1, opset=12, simplify=False,
               dynamic=False, half=False, nms=False, device='cpu')
onnx_path = Path(YOLO(OUT / 'best.pt').export(**options))
graph = onnx.load(onnx_path)
onnx.checker.check_model(graph)
so = ort.SessionOptions()
so.intra_op_num_threads = 6
so.inter_op_num_threads = 1
session = ort.InferenceSession(str(onnx_path), so, providers=['CPUExecutionProvider'])
model = YOLO(OUT / 'best.pt').model.float().eval().fuse()
names = model.names
save('export_config.json', dict(options=options, weight_sha256=sha(source), onnx_sha256=sha(onnx_path),
     code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
     versions=dict(python=sys.version, torch=torch.__version__, ultralytics=ultralytics.__version__,
                   onnx=onnx.__version__, onnxruntime=ort.__version__, numpy=np.__version__),
     inputs=[dict(name=x.name, shape=x.shape, type=x.type) for x in session.get_inputs()],
     outputs=[dict(name=x.name, shape=x.shape, type=x.type) for x in session.get_outputs()],
     ops=sorted(set(x.op_type for x in graph.graph.node)), names=names,
     inference=dict(conf=.25, iou=.7, max_det=300, agnostic_nms=False, multi_label=False,
                    letterbox_auto=False, input='RGB NCHW float32 /255', threads=6),
     gates=dict(raw_box_abs_px=.01, raw_score_abs=.0001, matched_box_abs_original_px=.1,
                matched_score_abs=.0001, unmatched_boxes=0)))

data = ROOT / 'mower_obstrast_check/data'
val = sorted((data / 'yolo_check/images/val').glob('*'))
selected = set()
for cls in range(8):
    candidates = []
    for p in val:
        label = data / 'yolo_check/labels/val' / (p.stem + '.txt')
        if cls in [int(s.split()[0]) for s in label.read_text().splitlines() if s.strip()]:
            candidates.append(p)
    selected.update(candidates[:20])
test = sorted((data / 'test_sets/synthetic_8cls_140_20261030/images/test').glob('*'))
assert len(test) == 140
paths = [('val_stratified', p) for p in sorted(selected)] + [('synthetic140', p) for p in test]
manifest, rows, all_outputs = [], [], []
counts = {s: np.zeros((8, 7), dtype=int) for s in ['val_stratified', 'synthetic140']}
visualized = set()
for i, (split, p) in enumerate(paths):
    image = cv2.imread(str(p))
    assert image is not None, p
    label = Path(str(p).replace('/images/', '/labels/')).with_suffix('.txt')
    gt = np.array([list(map(float, line.split())) for line in label.read_text().splitlines() if line.strip()], dtype=np.float32).reshape(-1, 5)
    gc = gt[:, 0].astype(int)
    h, w = image.shape[:2]
    gb = np.column_stack(((gt[:, 1:3] - gt[:, 3:5]/2)*[w,h], (gt[:, 1:3] + gt[:, 3:5]/2)*[w,h]))
    padded = LetterBox(new_shape=(640, 640), auto=False)(image=image)
    x = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
    xt = torch.from_numpy(x)
    if i == 0:
        with torch.inference_mode():
            for _ in range(3):
                model(xt)
                session.run(None, {'images': x})
    with torch.inference_mode():
        t = time.perf_counter()
        raw = model(xt)[0].numpy()
        pt_ms = (time.perf_counter()-t)*1000
        t = time.perf_counter()
        other = session.run(None, {'images': x})[0]
        onnx_ms = (time.perf_counter()-t)*1000
    assert raw.shape == other.shape == (1, 12, 8400)
    assert np.isfinite(raw).all() and np.isfinite(other).all()
    delta = np.abs(raw-other)
    predictions = []
    for a in (raw, other):
        pred = non_max_suppression(torch.from_numpy(a.copy()), conf_thres=.25, iou_thres=.7,
                                   max_det=300, nc=8, multi_label=False, agnostic=False)[0]
        pred[:, :4] = scale_boxes((640,640), pred[:, :4], image.shape[:2])
        predictions.append(pred.numpy())
    pt, ox = predictions
    pairs = match(pt[:, :4], pt[:, 5], ox[:, :4], ox[:, 5])
    box_error = float(np.max(np.abs(pt[pairs[:,0], :4]-ox[pairs[:,1], :4]))) if len(pairs) else 0.
    score_error = float(np.max(np.abs(pt[pairs[:,0], 4]-ox[pairs[:,1], 4]))) if len(pairs) else 0.
    unmatched = len(pt)+len(ox)-2*len(pairs)
    passed = delta[:,:4].max() <= .01 and delta[:,4:].max() <= .0001 and box_error <= .1 and score_error <= .0001 and unmatched == 0
    rows.append(dict(split=split, file=p.name, pt_boxes=len(pt), onnx_boxes=len(ox), matched=len(pairs),
                     unmatched=unmatched, raw_box_abs=float(delta[:,:4].max()), raw_score_abs=float(delta[:,4:].max()),
                     box_abs_original_px=box_error, score_abs=score_error, pt_ms=pt_ms, onnx_ms=onnx_ms, passed=bool(passed)))
    for cls in range(8):
        counts[split][cls, 0] += int(sum(gc == cls))
    for backend, pred in enumerate(predictions):
        matched_gt = match(gb, gc, pred[:, :4], pred[:, 5])
        for cls in range(8):
            tp = int(sum(gc[matched_gt[:,0]] == cls))
            counts[split][cls, 1+backend*3:4+backend*3] += [tp, int(sum(pred[:,5] == cls))-tp, int(sum(gc==cls))-tp]
    manifest.append(dict(split=split, image=str(p.relative_to(ROOT)), image_sha256=sha(p),
                         label=str(label.relative_to(ROOT)), label_sha256=sha(label), shape=[h,w], gt_classes=gc.tolist()))
    all_outputs.append(dict(split=split, file=p.name, pt=pt.tolist(), onnx=ox.tolist(), matched_pairs=pairs.tolist()))
    new_classes = set(gc.tolist()) - visualized
    if new_classes or not passed:
        panels = []
        for title, pred in zip(['PyTorch', 'ONNX Runtime'], predictions):
            canvas = image.copy()
            for x1,y1,x2,y2,score,cls in pred:
                cv2.rectangle(canvas, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
                cv2.putText(canvas, f'{names[int(cls)]} {score:.4f}', (int(x1),max(25,int(y1))), cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
            canvas = cv2.resize(canvas, (640, max(100,round(h*640/w))))
            cv2.putText(canvas, title, (12,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            panels.append(canvas)
        cv2.imwrite(str(OUT / 'comparison' / f'{split}_{p.stem}.jpg'), np.hstack(panels))
        visualized.update(gc.tolist())
    if (i+1)%20 == 0:
        print(f'Paired inference {i+1}/{len(paths)}', flush=True)
save('comparison/input_manifest.json', manifest)
save('comparison/predictions.json', all_outputs)
with (OUT/'comparison/per_image.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
save('comparison/summary.json', dict(images=len(rows), split_images={s:sum(r['split']==s for r in rows) for s in counts},
     passed_images=sum(r['passed'] for r in rows), failed_images=[r for r in rows if not r['passed']],
     total_pt_boxes=sum(r['pt_boxes'] for r in rows), total_onnx_boxes=sum(r['onnx_boxes'] for r in rows),
     maxima={k:max(r[k] for r in rows) for k in ['raw_box_abs','raw_score_abs','box_abs_original_px','score_abs','unmatched']},
     cpu_forward_ms={k:dict(p50=float(np.percentile([r[k] for r in rows],50)),p95=float(np.percentile([r[k] for r in rows],95))) for k in ['pt_ms','onnx_ms']},
     class_counts_columns=['GT','PT_TP','PT_FP','PT_FN','ONNX_TP','ONNX_FP','ONNX_FN'],
     class_counts={s:c.tolist() for s,c in counts.items()}))
print((OUT/'comparison/summary.json').read_text(), flush=True)
