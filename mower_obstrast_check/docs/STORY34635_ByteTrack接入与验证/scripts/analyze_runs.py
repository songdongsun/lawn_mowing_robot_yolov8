"""Summarize recorded tracker states, not ground-truth identity metrics."""
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
summaries = []
for folder in sorted(root.glob('run*')):
    if not (folder / 'tracks.jsonl').exists():
        continue
    rows = [json.loads(s) for s in (folder / 'tracks.jsonl').read_text().splitlines()]
    frames = [r for r in rows if 'frame' in r]
    by_id = defaultdict(list)
    for r in frames:
        for t in r['tracks']:
            by_id[t['track_id']].append((r, t))
    table = []
    for identity, records in sorted(by_id.items()):
        states = Counter(t['state'] for _, t in records)
        recovery = sum(a[1]['state'] == 'lost' and b[1]['state'] == 'tracked' for a, b in zip(records, records[1:]))
        low = sum(t['observed'] and t['confidence'] < .25 for _, t in records)
        longest = consecutive = 0
        last_frame = -2
        for r, t in records:
            if t['state'] == 'tracked':
                consecutive = consecutive + 1 if r['frame'] == last_frame + 1 else 1
                last_frame = r['frame']
                longest = max(longest, consecutive)
            else:
                consecutive = 0
        table.append(dict(track_id=identity, class_id=records[0][1]['class_id'], first_frame=records[0][0]['frame'],
                          last_frame=records[-1][0]['frame'], new=states['new'], tracked=states['tracked'],
                          lost=states['lost'], removed=states['removed'], low_score_observations=low,
                          lost_to_tracked=recovery, max_consecutive_tracked_frames=longest))
    if table:
        with (folder / 'track_statistics.csv').open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    elapsed = float(re.search(r'elapsed_s=([\d.]+)', (folder / 'run.log').read_text()).group(1))
    summary = dict(run=folder.name, frames=len(frames), elapsed_s=elapsed, throughput_fps=len(frames)/elapsed,
                   unique_track_ids=len(table), confirmed_ids=sum(t['tracked'] > 0 for t in table),
                   unconfirmed_ids=sum(t['tracked'] == 0 for t in table),
                   frames_with_observed_tracks=sum(any(t['observed'] for t in r['tracks']) for r in frames),
                   low_score_observations=sum(t['low_score_observations'] for t in table),
                   lost_to_tracked_transitions=sum(t['lost_to_tracked'] for t in table),
                   max_consecutive_tracked_frames=max((t['max_consecutive_tracked_frames'] for t in table), default=0),
                   final_status=rows[-1].get('status'))
    for key in ['inference_ms', 'tracking_ms']:
        values = [r[key] for r in frames]
        summary[key] = dict(mean=float(np.mean(values)), p50=float(np.percentile(values, 50)),
                            p95=float(np.percentile(values, 95)), maximum=float(max(values)))
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    summaries.append(summary)
    chosen = sorted(table, key=lambda x: x['tracked'], reverse=True)[:12]
    fig, ax = plt.subplots(figsize=(12, 5))
    origin = frames[0]['tracking_timestamp_ms']
    for i, item in enumerate(chosen):
        records = by_id[item['track_id']]
        for state, color in [('tracked', '#17864b'), ('lost', '#d38921'), ('new', '#3486d2')]:
            times = [(r['tracking_timestamp_ms']-origin)/1000 for r, t in records if t['state'] == state]
            ax.scatter(times, [i]*len(times), s=9, color=color, label=state if i == 0 else None)
    ax.set_yticks(range(len(chosen)), ['ID '+str(t['track_id']) for t in chosen])
    ax.set_xlabel('Elapsed capture time (s)'); ax.set_title(folder.name + ': longest tracks (not GT identities)')
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(folder/'track_timeline.png', dpi=150); plt.close(fig)
    previews = sorted((folder/'live_preview').glob('*.jpg'))
    if previews:
        frame_lookup = {r['frame']: r for r in frames}
        samples = [(frame_lookup[int(p.stem)]['tracking_timestamp_ms']/1000, p) for p in previews if int(p.stem) in frame_lookup]
        writer = cv2.VideoWriter(str(folder/'preview_sample.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 5, (640,480))
        if not writer.isOpened():
            raise RuntimeError('Video encoder unavailable')
        index = 0
        for stamp in np.arange(samples[0][0], samples[-1][0], .2):
            while index+1 < len(samples) and samples[index+1][0] <= stamp:
                index += 1
            image = cv2.imdecode(np.fromfile(samples[index][1], dtype=np.uint8), cv2.IMREAD_COLOR)
            writer.write(image)
        writer.release()
(root/'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
print(json.dumps(summaries, indent=2))
