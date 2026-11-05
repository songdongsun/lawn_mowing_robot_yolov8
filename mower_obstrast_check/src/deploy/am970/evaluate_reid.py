"""Compare the same recorded NPU detections with/without appearance, using official TrackEval."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--trial', required=True)
parser.add_argument('--gt', required=True)
parser.add_argument('--trackeval', required=True)
args = parser.parse_args()
root = Path(args.trial).resolve()
sys.path.insert(0, str(Path(args.trackeval).resolve()))
import trackeval

seq = 'MOT17-09'
gt_dir = root/'evaluation/gt'/seq/'gt'
gt_dir.mkdir(parents=True, exist_ok=True)
shutil.copyfile(args.gt, gt_dir/'gt.txt')
for name in ['baseline', 'reid']:
    dest = root/'evaluation/trackers'/name/'data'
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/(seq+'.txt')).open('w') as f:
        for line in (root/(name+'.jsonl')).read_text().splitlines():
            row = json.loads(line)
            for t in row['tracks']:
                if t['class_id'] != 0 or t['state'] != 'tracked' or not t['observed']:
                    continue
                box = np.asarray(t['xyxy'])*2  # Official GT is 1920x1080, preview video is 960x540.
                box[[0, 2]] = np.clip(box[[0, 2]], 0, 1920)
                box[[1, 3]] = np.clip(box[[1, 3]], 0, 1080)
                x1, y1, x2, y2 = box
                if x2 > x1 and y2 > y1:
                    f.write(f"{row['frame']},{t['track_id']},{x1:.6f},{y1:.6f},{x2-x1:.6f},{y2-y1:.6f},{t['confidence']:.6f},-1,-1,-1\n")
dataset = trackeval.datasets.MotChallenge2DBox(dict(GT_FOLDER=str(root/'evaluation/gt'),
    TRACKERS_FOLDER=str(root/'evaluation/trackers'), TRACKERS_TO_EVAL=['baseline', 'reid'],
    SKIP_SPLIT_FOL=True, SEQ_INFO={seq:525}, PRINT_CONFIG=False, DO_PREPROC=True))
summary = {}
for name in ['baseline', 'reid']:
    data = dataset.get_preprocessed_seq_data(dataset.get_raw_seq_data(name, seq), 'pedestrian')
    metrics = {}
    for metric in [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]:
        result = metric.eval_sequence(data)
        metrics[metric.get_name()] = {k: np.asarray(v).tolist() for k, v in result.items()}
    (root/(name+'_metrics.json')).write_text(json.dumps(metrics,indent=2))
    summary[name] = {k: float(np.mean(metrics['HOTA'][k]))*100 for k in ['HOTA','DetA','AssA']}
    summary[name].update({k: metrics['CLEAR'][k] for k in ['IDSW','Frag','CLR_TP','CLR_FP','CLR_FN']})
    summary[name]['MOTA'] = metrics['CLEAR']['MOTA']*100
    summary[name]['IDF1'] = metrics['Identity']['IDF1']*100
(root/'comparison.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
