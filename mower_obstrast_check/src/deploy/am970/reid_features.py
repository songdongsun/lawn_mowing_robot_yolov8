"""Extract person descriptors for an unchanged recorded detection sequence (host CPU experiment)."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import openvino as ov

parser = argparse.ArgumentParser()
parser.add_argument('--video', required=True)
parser.add_argument('--detections', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--frames', type=int, required=True)
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
compiled = ov.Core().compile_model(args.model, 'CPU', {'INFERENCE_NUM_THREADS': 2, 'NUM_STREAMS': 1})
capture = cv2.VideoCapture(args.video)
fps = capture.get(cv2.CAP_PROP_FPS)
durations, counts = [], []
start = time.perf_counter()
with open(args.detections) as source, (out/'baseline.txt').open('w') as baseline, (out/'features.txt').open('w') as features:
    for index in range(args.frames):
        line = source.readline()
        tokens = line.split()
        if not tokens:
            raise ValueError('Incomplete detections')
        frame, stamp, count = int(tokens[0]), float(tokens[1]), int(tokens[2])
        assert frame == index+1 and len(tokens) == 3+6*count
        assert abs(stamp-index*1000/fps) < .01
        ok, image = capture.read()
        if not ok:
            raise ValueError('Incomplete video')
        baseline.write(line)
        fields = tokens[:3]
        descriptors = 0
        a = time.perf_counter()
        for offset in range(count):
            detection = tokens[3+offset*6:9+offset*6]
            cls, score, x, y, w, h = map(float, detection)
            embedding = np.empty(0, dtype=np.float32)
            # Keep all detections. Only sufficiently large/high-score person crops supply appearance.
            if cls == 0 and score >= .25 and w >= 24 and h >= 64:
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(image.shape[1], int(np.ceil(x+w))), min(image.shape[0], int(np.ceil(y+h)))
                if x2 > x1 and y2 > y1:
                    crop = cv2.resize(image[y1:y2, x1:x2], (128, 256))
                    tensor = crop.transpose(2, 0, 1)[None].astype(np.float32)
                    embedding = np.asarray(compiled([tensor])[compiled.output(0)]).reshape(-1)
                    assert embedding.size == 256 and np.isfinite(embedding).all()
                    norm = np.linalg.norm(embedding)
                    if norm <= 1e-12:
                        raise ValueError('Invalid descriptor norm')
                    embedding = embedding/norm
                    descriptors += 1
            fields.extend(detection)
            fields.append(str(embedding.size))
            fields.extend(format(float(v), '.9g') for v in embedding)
        durations.append((time.perf_counter()-a)*1000)
        counts.append(descriptors)
        features.write(' '.join(fields)+'\n')
        if frame % 100 == 0:
            print('features', frame, 'mean_ms', round(float(np.mean(durations)), 2), flush=True)
capture.release()
summary = dict(frames=args.frames, descriptors=sum(counts), extraction_ms=dict(mean=float(np.mean(durations)),
               p50=float(np.percentile(durations,50)), p95=float(np.percentile(durations,95))),
               total_seconds=time.perf_counter()-start, runtime='OpenVINO '+ov.__version__, device='host CPU, 2 threads',
               preprocessing='BGR float32 0..255 NCHW 1x3x256x128, resize, model-internal normalization',
               model_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
               video_sha256=hashlib.sha256(Path(args.video).read_bytes()).hexdigest(),
               parameters=dict(min_score=.25, min_width=24, min_height=64, cosine_gate=.4, iou_cost_gate=.8,
                               iou_weight=.5, appearance_weight=.5, feature_ema=.9))
(out/'extraction.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
