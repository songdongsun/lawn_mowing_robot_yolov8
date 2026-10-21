"""Summarize real-board consistency and compare masks with fixed annotations."""
import csv
import argparse
import json
from pathlib import Path
import re
import shutil

import cv2
import numpy as np

from evaluate_mower_pt_input_shapes import metrics, panel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/b08_v3_consistency_20261014'
GT = ROOT/'results/b08_v3_input_shapes_20261014'
WORK = ROOT/'work/b08_v3_20261014'


def main():
    global OUT, GT, WORK
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--gt', type=Path, default=GT)
    parser.add_argument('--work', type=Path, default=WORK)
    parser.add_argument('--name', default='b08_v3')
    args = parser.parse_args()
    OUT, GT, WORK = args.out, args.gt, args.work
    summary = json.loads((OUT/'summary.json').read_text())
    assert summary['npu_samples'] == 50
    assert (OUT/'deployment_before.txt').read_bytes() == (OUT/'deployment_after.txt').read_bytes()
    ground = json.loads((GT/'summary.json').read_text())
    assert summary['model_sha256']['pytorch'] == ground['model_sha256']
    (OUT/'vs_annotation').mkdir(exist_ok=True)
    rows = []
    for sample in summary['samples']:
        sid = sample['id']
        source = next(s for s in ground['samples'] if s['id'] == sid)
        assert source['image_sha256'] == sample['sha256']
        gt = cv2.imread(str(GT/'masks'/f'{sid}_gt.png'), 0) > 0
        im = cv2.imread(str(Path('/mnt/e/cpp_project/lawn_mowing_robot_yolov8/mower_grass_seg/data/test_train25_val25_b07_v4')/sample['file']))
        views = [panel(im, gt, None, f'{sid} annotation')]
        for backend in ['pytorch', 'onnx', 'npu']:
            mask = cv2.imread(str(OUT/'masks'/f'{sid}_{backend}.png'), 0) > 0
            rows.append(dict(id=sid, file=sample['file'], split=sample['source_split'], backend=backend,
                             **metrics(mask, gt)))
            views.append(panel(im, gt, mask, f'{backend} | green=TP red=FP blue=FN'))
        assert cv2.imwrite(str(OUT/'vs_annotation'/f'{sid}.jpg'), np.concatenate(views, axis=1))
    with (OUT/'vs_annotation/per_image.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    aggregates = {}
    for backend in ['pytorch', 'onnx', 'npu']:
        aggregates[backend] = {}
        for split in ['all', 'train', 'val']:
            group = [r for r in rows if r['backend'] == backend and (split == 'all' or r['split'] == split)]
            totals = {k:sum(r[k] for r in group) for k in ['tp', 'fp', 'fn']}
            aggregates[backend][split] = dict(count=len(group), mean_iou=float(np.mean([r['iou'] for r in group])),
                micro_iou=totals['tp']/sum(totals.values()), **totals)
    log = (OUT/'board_run.log').read_text()
    times = [float(v) for v in re.findall(r'inference\s*:\s*([\d.]+)\s*ms', log)]
    assert len(times) == 50 and len(re.findall(r'DUMP_OK', log)) == 50
    report = dict(annotations=aggregates, inference_ms=dict(count=len(times), mean=float(np.mean(times)), min=min(times), max=max(times)),
                  focus=[r for r in rows if r['id'] in ['018', '042', '044']])
    (OUT/'accuracy_and_timing.json').write_text(json.dumps(report, indent=2))
    bundle = OUT/'deliverables'
    bundle.mkdir(exist_ok=True)
    for name in ['comparisons', 'masks', 'vs_annotation']:
        shutil.copytree(OUT/name, bundle/name, dirs_exist_ok=True)
    (bundle/'reference').mkdir(exist_ok=True)
    for path in (OUT/'reference').glob('*_detections.json'):
        shutil.copyfile(path, bundle/'reference'/path.name)
    for name in ['summary.json', 'per_image.csv', 'versions.json', 'board_run.log', 'board_checksums.txt',
                 'deployment_before.txt', 'deployment_after.txt', 'accuracy_and_timing.json']:
        shutil.copyfile(OUT/name, bundle/name)
    (bundle/'models').mkdir(exist_ok=True)
    for name in [args.name+suffix for suffix in ['.pt', '.onnx', '.quantize', '_inputmeta.yml',
                 '_postprocess_file.yml']] + ['provenance.json']:
        shutil.copyfile(WORK/name, bundle/'models'/name)
    shutil.copyfile(WORK/'export_nbg_unify/network_binary.nb', bundle/'models/network_binary.nb')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
