"""Paired PT input-shape evaluation against existing grass polygons."""
import csv
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/pt_input_shapes_20261013'
DATA = Path('/mnt/e/cpp_project/lawn_mowing_robot_yolov8/mower_grass_seg/data/yolo_seg')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(pred, gt):
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.)


def panel(image, gt, pred, title):
    view = image.copy()
    if pred is None:
        regions = [(gt, (0, 200, 0))]
    else:
        regions = [(pred & gt, (0, 200, 0)), (pred & ~gt, (0, 0, 255)),
                   (~pred & gt, (255, 0, 0))]
    for region, color in regions:
        view[region] = (view[region]*.5 + np.asarray(color)*.5).astype(np.uint8)
    banner = np.full((32, image.shape[1], 3), 255, np.uint8)
    cv2.putText(banner, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1)
    return np.concatenate([banner, view], axis=0)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--iou', type=float, default=.45)
    parser.add_argument('--weights', type=Path, default=ROOT/'models/mower_yolov8n_seg.pt')
    args = parser.parse_args()
    OUT = args.out
    for folder in ['comparisons', 'masks']:
        (OUT/folder).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    weight = args.weights
    model = YOLO(str(weight))
    samples = json.loads((ROOT/'results/consistency_train25_val25_20261013/summary.json').read_text())['samples']
    rows, provenance = [], []
    for sample in samples:
        sid = sample['id']
        filename = sample['file'].split('__', 1)[1]
        path = DATA/'images'/sample['source_split']/filename
        annotation = DATA/'labels'/sample['source_split']/Path(filename).with_suffix('.txt')
        assert sha(path) == sample['sha256'], 'Image changed since paired experiment'
        image = cv2.imread(str(path))
        assert image.shape == (480, 640, 3)
        gt = np.zeros((480, 640), np.uint8)
        for line in annotation.read_text().splitlines():
            values = np.asarray([float(v) for v in line.split()])
            assert values[0] == 0 and len(values) >= 7 and (len(values)-1) % 2 == 0
            assert np.isfinite(values).all() and ((values[1:] >= 0) & (values[1:] <= 1)).all()
            points = np.rint(values[1:].reshape(-1, 2)*[640, 480]).astype(np.int32)
            cv2.fillPoly(gt, [points], 1)
        gt = gt.astype(bool)
        panels = [panel(image, gt, None, f'{sid} annotation | green=grass')]
        preds = {}
        for mode, inp in [('640x480', image), ('640x640', cv2.copyMakeBorder(
                image, 80, 80, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114)))]:
            result = model.predict(inp, imgsz=640, conf=.25, iou=args.iou, retina_masks=True,
                                   device='cpu', verbose=False)[0]
            # Assert the actual installed predictor's tensor, not only its requested imgsz.
            actual = model.predictor.preprocess([inp]).cpu().numpy()
            expected = inp[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)/255
            assert np.array_equal(actual, expected), f'Unexpected preprocessing: {mode}'
            pred = np.zeros(inp.shape[:2], bool) if result.masks is None else (
                result.masks.data.cpu().numpy() > .5).any(axis=0)
            if mode == '640x640':
                pred = pred[80:560]
            assert pred.shape == gt.shape
            stats = metrics(pred, gt)
            rows.append(dict(id=sid, file=filename, split=sample['source_split'], mode=mode,
                             detections=len(result.boxes), **stats))
            panels.append(panel(image, gt, pred, f'PT {mode} | green=TP red=FP blue=FN'))
            assert cv2.imwrite(str(OUT/'masks'/f'{sid}_{mode}.png'), pred.astype(np.uint8)*255)
            preds[mode] = stats
        assert cv2.imwrite(str(OUT/'masks'/f'{sid}_gt.png'),gt.astype(np.uint8)*255)
        assert cv2.imwrite(str(OUT/'comparisons'/f'{sid}.jpg'),np.concatenate(panels,axis=1))
        provenance.append(dict(id=sid, file=filename, split=sample['source_split'],
                               image_sha256=sha(path), label_sha256=sha(annotation)))
        print(sid, {m: round(s['iou'], 5) for m,s in preds.items()}, flush=True)
    aggregates = {}
    for split in ['all', 'train', 'val']:
        aggregates[split] = {}
        for mode in ['640x480', '640x640']:
            group = [r for r in rows if r['mode'] == mode and (split == 'all' or r['split'] == split)]
            totals = {k: sum(r[k] for r in group) for k in ['tp', 'fp', 'fn', 'tn']}
            tp,fp,fn = totals['tp'],totals['fp'],totals['fn']
            aggregates[split][mode] = dict(count=len(group), **totals,
                mean_iou=float(np.mean([r['iou'] for r in group])), min_iou=min(r['iou'] for r in group),
                micro_iou=tp/(tp+fp+fn), micro_precision=tp/(tp+fp), micro_recall=tp/(tp+fn))
    pairs = []
    for sample in samples:
        a,b = [next(r for r in rows if r['id']==sample['id'] and r['mode']==mode)
               for mode in ['640x480','640x640']]
        pairs.append(dict(id=a['id'], file=a['file'], split=a['split'],
                          iou_rect_minus_square=a['iou']-b['iou'],
                          fp_rect_minus_square=a['fp']-b['fp'], fn_rect_minus_square=a['fn']-b['fn']))
    with (OUT/'per_image.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = dict(model_path=str(weight.resolve()), model_sha256=sha(weight), versions=dict(torch=torch.__version__,ultralytics=ultralytics.__version__,opencv=cv2.__version__),
        method=f'Native PT CPU, conf=.25, NMS={args.iou}, retina_masks=True; actual RGB /255 tensors verified. Width x height in mode names.',
        annotation='Union of class-0 YOLO polygons, normalized coordinates times width/height rounded with np.rint and rasterized with cv2.fillPoly.',
        limitation='Current annotations, not independently reviewed ground truth; 25 train + 25 val, not independent field evaluation. Pixel metrics, not instance mAP.',
        aggregates=aggregates, rect_iou_better=sum(p['iou_rect_minus_square']>1e-10 for p in pairs),
        rect_iou_worse=sum(p['iou_rect_minus_square']< -1e-10 for p in pairs),
        pairs=pairs, samples=provenance)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(aggregates,indent=2))
    print('rect better/worse:',report['rect_iou_better'],report['rect_iou_worse'])


if __name__ == '__main__':
    main()
