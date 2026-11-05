import csv
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.data.utils import img2label_paths
from ultralytics.utils import ops
from ultralytics.utils.metrics import smooth

RUN = ROOT / 'mower_obstrast_check/runs/D00_ultralytics_default'
parser = argparse.ArgumentParser()
parser.add_argument('--split', default='val')
parser.add_argument('--imgsz', type=int, default=640)
parser.add_argument('--out', type=Path, default=RUN / 'analysis_20261028')
parser.add_argument('--data', type=Path, default=ROOT / 'mower_obstrast_check/data/yolo_check/data.yaml')
parser.add_argument('--weights', type=Path, default=RUN / 'weights/best.pt')
args = parser.parse_args()
OUT = args.out
OUT.mkdir(exist_ok=True)
(OUT / 'predictions').mkdir(exist_ok=True)
records = []

class AuditValidator(DetectionValidator):
    def update_metrics(self, preds, batch):
        for si, pred in enumerate(preds):
            prepared = self._prepare_batch(si, batch)
            path = Path(prepared['im_file'])
            label = Path(img2label_paths([str(path)])[0])
            boxes = ops.scale_boxes(prepared['imgsz'], prepared['bboxes'].clone(), prepared['ori_shape'], ratio_pad=prepared['ratio_pad'])
            native = self.scale_preds(pred, prepared)
            np.savez_compressed(OUT / 'predictions' / (path.stem + '.npz'),
                gt_boxes=boxes.cpu().numpy(), gt_cls=prepared['cls'].cpu().numpy(),
                boxes=native['bboxes'].cpu().numpy(), cls=native['cls'].cpu().numpy(), conf=native['conf'].cpu().numpy())
            records.append(dict(file=path.name, shape=list(prepared['ori_shape']),
                image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), label_sha256=hashlib.sha256(label.read_bytes()).hexdigest()))
        super().update_metrics(preds, batch)

if __name__ == '__main__':
    torch.set_num_threads(6)
    model = YOLO(str(args.weights))
    checkpoint = dict(epoch=model.ckpt.get('epoch'), best_fitness=model.ckpt.get('best_fitness'),
                      train_args=model.ckpt.get('train_args'), train_metrics=model.ckpt.get('train_metrics'),
                      version=model.ckpt.get('version'), names=model.names,
                      weight_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest())
    (OUT / 'checkpoint.json').write_text(json.dumps(checkpoint, indent=2, default=str), encoding='utf-8')
    started = time.perf_counter()
    result = model.val(validator=AuditValidator, data=str(args.data), split=args.split,
                       imgsz=args.imgsz, batch=32, device='cpu', workers=0, conf=0.001, iou=0.7,
                       max_det=300, half=False, rect=True, plots=True, verbose=True,
                       project=str(OUT), name='validation', exist_ok=True)
    rows = result.summary(decimals=10)
    x = result.box.px
    index = int(smooth(result.box.f1_curve.mean(0), 0.1).argmax())
    summary = dict(per_class=rows, overall=result.results_dict, best_f1_confidence=float(x[index]),
                   names=result.names, config=dict(imgsz=args.imgsz,batch=32,conf=0.001,iou=0.7,max_det=300,rect=True,half=False),
                   elapsed_seconds=time.perf_counter()-started, speed_ms_per_image=result.speed,
                   versions=dict(torch=torch.__version__,ultralytics=ultralytics.__version__), samples=len(records))
    (OUT / 'metrics.json').write_text(json.dumps(summary, indent=2, default=float), encoding='utf-8')
    (OUT / 'input_manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    with (OUT / 'per_class_metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(OUT / 'curves.npz', confidence=x, precision=result.box.p_curve,
                       recall=result.box.r_curve, f1=result.box.f1_curve, ap=result.box.all_ap)
    print(json.dumps(summary, indent=2, default=float))
