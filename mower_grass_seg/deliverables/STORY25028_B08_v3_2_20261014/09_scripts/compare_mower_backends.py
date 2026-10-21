"""Paired output consistency, not ground-truth accuracy. No model conversion."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def prepare(image):
    h, w = image.shape[:2]
    scale = min(640 / w, 640 / h)
    nw, nh = round(w * scale), round(h * scale)
    left, top = (640 - nw) // 2, (640 - nh) // 2
    padded = cv2.copyMakeBorder(cv2.resize(image, (nw, nh)), top, 640-nh-top,
                               left, 640-nw-left, cv2.BORDER_CONSTANT,
                               value=(114, 114, 114))
    return padded, (h, w, scale, nw, nh, left, top)


def decode(pred, proto, geom, outward_crop=False):
    assert pred.shape == (37, 8400) and proto.shape == (32, 160, 160)
    assert np.isfinite(pred).all() and np.isfinite(proto).all()
    h, w, scale, nw, nh, left, top = geom
    boxes, anchors = [], []
    for i in np.flatnonzero(pred[4] >= .25):
        cx, cy, bw, bh = pred[:4, i]
        x1, x2 = np.clip((np.array([cx-bw/2, cx+bw/2])-left)/scale, 0, w if outward_crop else w-1)
        y1, y2 = np.clip((np.array([cy-bh/2, cy+bh/2])-top)/scale, 0, h if outward_crop else h-1)
        if x2-x1 > 1 and y2-y1 > 1:
            boxes.append([float(x1), float(y1), float(x2-x1), float(y2-y1)])
            anchors.append(int(i))
    # Same >= confidence and > IoU suppression boundaries as the C++ decoder.
    order = sorted(range(len(boxes)), key=lambda k: -float(pred[4, anchors[k]]))
    keep = []
    while order:
        i, *rest = order
        keep.append(i)
        a = boxes[i]
        order = []
        for j in rest:
            b = boxes[j]
            inter = max(0, min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0])) * max(
                0, min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
            union = a[2]*a[3]+b[2]*b[3]-inter
            if inter / union <= .45:
                order.append(j)
    mask = np.zeros((h, w), dtype=bool)
    detections = []
    for k in keep:
        # Accumulate in channel order, matching the deployed C++ mask calculation.
        logits = np.zeros((160, 160), dtype=np.float32)
        for channel in range(32):
            logits += pred[5+channel, anchors[k]] * proto[channel]
        prob = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
        prob = cv2.resize(prob, (640, 640))[top:top+nh, left:left+nw]
        binary = cv2.resize(prob, (w, h)) > .5
        x, y, bw, bh = map(int, boxes[k])
        if outward_crop:
            bx, by, bwidth, bheight = boxes[k]
            x, y = int(np.floor(bx)), int(np.floor(by))
            bw = min(w, int(np.ceil(bx+bwidth))) - x
            bh = min(h, int(np.ceil(by+bheight))) - y
        mask[y:y+bh, x:x+bw] |= binary[y:y+bh, x:x+bw]
        detections.append(dict(score=float(pred[4, anchors[k]]), box_xywh=boxes[k]))
    return mask, dict(candidates=len(boxes), detections=detections,
                      max_score=float(pred[4].max()), foreground_pixels=int(mask.sum()))


def measure(a, b):
    intersection = int((a & b).sum())
    union = int((a | b).sum())
    total = int(a.sum()+b.sum())
    return dict(mask_iou=intersection/union if union else 1.,
                dice=2*intersection/total if total else 1.,
                intersection=intersection, union=union, both_empty=not union,
                disagreement_pixels=int((a != b).sum()))


def visualize(image, mask, title):
    view = image.copy()
    view[mask] = (view[mask]*.5 + np.array([0, 0, 255])*.5).astype(np.uint8)
    view = cv2.resize(view, (480, 360))
    cv2.rectangle(view, (0, 0), (480, 30), (255, 255, 255), -1)
    cv2.putText(view, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1)
    return view


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--npu-only', action='store_true', help='Use existing PT/ONNX caches')
    parser.add_argument('--pt', type=Path, default=ROOT/'models/mower_yolov8n_seg.pt')
    parser.add_argument('--onnx', type=Path, default=ROOT/'models/mower_yolov8n_seg.onnx')
    parser.add_argument('--nb', type=Path, default=ROOT/'outputs/mower_uint8_hybrid_nbg_unify/network_binary.nb')
    parser.add_argument('--outward-crop', action='store_true')
    args = parser.parse_args()
    out = args.out
    for folder in ['inputs', 'reference', 'masks', 'comparisons', 'outputs']:
        (out/folder).mkdir(parents=True, exist_ok=True)
    files = sorted(args.data.glob('*.jpg'))
    if not files:
        raise ValueError('No JPG images')
    with (args.data/'manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        sources = {row['sample']: row for row in csv.DictReader(f)}
    assert set(sources) == {p.name for p in files}, 'Manifest mismatch'
    models = dict(pytorch=args.pt, onnx=args.onnx, npu=args.nb)
    hashes = {k: sha(v) for k, v in models.items()}
    if args.npu_only:
        checksums = {}
        for line in (out/'board_checksums.txt').read_text().splitlines():
            fields = line.split()
            if len(fields) == 2 and len(fields[0]) == 64:
                checksums[fields[1]] = fields[0]
        assert checksums['network_binary.nb'] == hashes['npu'], 'Wrong board model'
        for idx in range(len(files)):
            for relative in [f'inputs/{idx:03d}.png', f'outputs/{idx:03d}_0.f32',
                             f'outputs/{idx:03d}_1.f32']:
                assert sha(out/relative) == checksums[relative], f'Board checksum mismatch: {relative}'
        previous = json.loads((out/'summary.json').read_text())
        assert previous['model_sha256'] == hashes, 'Model changed since reference run'
        assert previous.get('outward_crop', False) == args.outward_crop, 'Crop mode changed'
        old_samples = {s['file']: s['sha256'] for s in previous['samples']}
        assert old_samples == {p.name: sha(p) for p in files}, 'Dataset changed'
    else:
        import torch
        import onnxruntime as ort
        import ultralytics
        torch.set_num_threads(4)
        model = ultralytics.YOLO(str(models['pytorch'])).model.float().eval()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        session = ort.InferenceSession(str(models['onnx']), options,
                                       providers=['CPUExecutionProvider'])
        write_json(out/'versions.json', dict(python=platform.python_version(),
                   torch=torch.__version__, onnxruntime=ort.__version__,
                   ultralytics=ultralytics.__version__, opencv=cv2.__version__, numpy=np.__version__))
    rows, samples = [], []
    for idx, path in enumerate(files):
        sid = f'{idx:03d}'
        image = cv2.imread(str(path))
        assert image is not None, path
        padded, geom = prepare(image)
        assert cv2.imwrite(str(out/'inputs'/f'{sid}.png'), padded)
        sample = dict(id=sid, file=path.name, sha256=sha(path), **sources[path.name])
        samples.append(sample)
        cache = out/'reference'/f'{sid}.npz'
        if not args.npu_only:
            tensor = padded[:, :, ::-1].transpose(2, 0, 1)[None].copy().astype(np.float32)/255
            with torch.inference_mode():
                raw = model(torch.from_numpy(tensor))
            if isinstance(raw[0], (tuple, list)):
                pt = (raw[0][0][0].numpy(), raw[0][1][0].numpy())
            else:
                pt = (raw[0][0].numpy(), raw[1][-1][0].numpy())
            onnx_values = session.run(None, {session.get_inputs()[0].name: tensor})
            byshape = {tuple(v.shape): v for v in onnx_values}
            onnx = (byshape[(1, 37, 8400)][0], byshape[(1, 32, 160, 160)][0])
            np.savez_compressed(cache, pt0=pt[0], pt1=pt[1], onnx0=onnx[0], onnx1=onnx[1])
        with np.load(cache) as values:
            arrays = {name: (values[prefix+'0'], values[prefix+'1']) for name, prefix in
                      [('pytorch', 'pt'), ('onnx', 'onnx')]}
        npu_files = [out/'outputs'/f'{sid}_{k}.f32' for k in range(2)]
        if all(p.exists() for p in npu_files):
            arrays['npu'] = (np.fromfile(npu_files[0], dtype='<f4').reshape(37, 8400),
                             np.fromfile(npu_files[1], dtype='<f4').reshape(32, 160, 160))
        decoded = {k: decode(*v, geom, outward_crop=args.outward_crop) for k, v in arrays.items()}
        for name, (mask, _) in decoded.items():
            assert cv2.imwrite(str(out/'masks'/f'{sid}_{name}.png'), mask.astype(np.uint8)*255)
        details = {k: v[1] for k, v in decoded.items()}
        for a, b in [('pytorch', 'onnx'), ('pytorch', 'npu'), ('onnx', 'npu')]:
            if b not in decoded:
                continue
            stats = measure(decoded[a][0], decoded[b][0])
            row = dict(id=sid, file=path.name, split=sample['source_split'], pair=f'{a}_vs_{b}',
                       **stats, detections_a=len(decoded[a][1]['detections']),
                       detections_b=len(decoded[b][1]['detections']),
                       score_max_abs=float(np.abs(arrays[a][0][4]-arrays[b][0][4]).max()),
                       proto_mae=float(np.abs(arrays[a][1]-arrays[b][1]).mean()))
            rows.append(row)
        panels = [visualize(image, np.zeros(image.shape[:2], bool), f'{sid} original')]
        panels += [visualize(image, value[0], name) for name, value in decoded.items()]
        assert cv2.imwrite(str(out/'comparisons'/f'{sid}.jpg'), np.concatenate(panels, axis=1))
        write_json(out/'reference'/f'{sid}_detections.json', details)
        print(f'{sid} {path.name}: ' + ', '.join(f'{k}={len(v[1]["detections"])}' for k,v in decoded.items()), flush=True)
    (out/'input_list.txt').write_text(''.join(f'{s["id"]}\n' for s in samples))
    aggregates = {}
    for pair in sorted({r['pair'] for r in rows}):
        group = [r for r in rows if r['pair'] == pair]
        active = [r for r in group if not r['both_empty']]
        aggregates[pair] = dict(count=len(group), mean_mask_iou=float(np.mean([r['mask_iou'] for r in group])),
            mean_dice=float(np.mean([r['dice'] for r in group])),
            median_mask_iou=float(np.median([r['mask_iou'] for r in group])),
            min_mask_iou=min(r['mask_iou'] for r in group), both_empty=sum(r['both_empty'] for r in group),
            nonempty_mean_iou=float(np.mean([r['mask_iou'] for r in active])) if active else None,
            global_mask_iou=sum(r['intersection'] for r in group)/sum(r['union'] for r in group) if active else None,
            detection_count_equal=sum(r['detections_a'] == r['detections_b'] for r in group),
            worst_samples=[r['id'] for r in sorted(group, key=lambda r:r['mask_iou'])[:5]])
        aggregates[pair]['by_split'] = {
            split: dict(count=sum(r['split'] == split for r in group),
                        mean_mask_iou=float(np.mean([r['mask_iou'] for r in group if r['split'] == split])),
                        min_mask_iou=min(r['mask_iou'] for r in group if r['split'] == split))
            for split in sorted({r['split'] for r in group})}
    with (out/'per_image.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(purpose='Output consistency, not ground-truth accuracy or independent field validation',
        acceptance='No error tolerance agreed; metrics only, not an acceptance verdict',
        preprocessing='640 letterbox, padding 114, BGR PNG to NPU preproc; RGB float32 /255 to PT/ONNX',
        postprocessing='Shared decoder: conf>=0.25, box NMS IoU>0.45, mask>0.5; union of grass instances',
        nms_ties='Equal-score tie order may differ from C++ std::sort; shared decoder isolates backend changes',
        outward_crop=args.outward_crop,
        npu_samples=sum(r['pair']=='pytorch_vs_npu' for r in rows), model_sha256=hashes,
        aggregates=aggregates, samples=samples)
    write_json(out/'summary.json', report)
    print(json.dumps(aggregates, indent=2))


if __name__ == '__main__':
    main()
