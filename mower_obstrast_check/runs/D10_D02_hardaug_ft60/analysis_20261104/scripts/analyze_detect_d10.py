import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RUN = ROOT / 'mower_obstrast_check/runs/D00_ultralytics_default'
OUT = RUN / 'analysis_20261028'
DATA = ROOT / 'mower_obstrast_check/data/yolo_check'
NAMES = ['person', 'cat', 'dog', 'frisbee', 'sports ball', 'teddy bear', 'garden hose', 'rock']

def match(gt, gc, pred, pc):
    if len(gt) == 0 or len(pred) == 0:
        return np.zeros((0, 2), dtype=int)
    inter = np.maximum(0, np.minimum(gt[:, None, 2:], pred[None, :, 2:]) - np.maximum(gt[:, None, :2], pred[None, :, :2])).prod(2)
    area_gt = (gt[:, 2:] - gt[:, :2]).prod(1)
    area_pred = (pred[:, 2:] - pred[:, :2]).prod(1)
    iou = inter / (area_gt[:, None] + area_pred[None, :] - inter + 1e-7)
    pairs = np.argwhere((iou >= .5) & (gc[:, None] == pc[None, :]))
    if len(pairs) > 1:
        pairs = pairs[np.argsort(iou[pairs[:, 0], pairs[:, 1]])[::-1]]
        pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
        pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
    return pairs

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    RUN, OUT = args.run, args.out
    with (RUN / 'results.csv').open() as f:
        epochs = [{k.strip(): float(v) for k,v in r.items()} for r in csv.DictReader(f)]
    best = max(epochs, key=lambda r:r['metrics/mAP50-95(B)'])
    training = dict(epochs=len(epochs), best=best, last=epochs[-1], snapshots={str(i): epochs[i-1] for i in [1,4,8,15,len(epochs)] if i<=len(epochs)},
                    elapsed_hours=epochs[-1]['time']/3600)
    (OUT / 'training_analysis.json').write_text(json.dumps(training, indent=2))
    fig, axes = plt.subplots(2,2,figsize=(13,8))
    xs = [r['epoch'] for r in epochs]
    for key in ['metrics/mAP50(B)','metrics/mAP50-95(B)']:
        axes[0,0].plot(xs,[r[key]*100 for r in epochs],label=key)
    for key in ['metrics/precision(B)','metrics/recall(B)']:
        axes[0,1].plot(xs,[r[key]*100 for r in epochs],label=key)
    for ax, loss in zip(axes[1], ['box_loss','cls_loss']):
        for split in ['train','val']:
            ax.plot(xs,[r[split+'/'+loss] for r in epochs],label=split+'/'+loss)
    for ax in axes.flat:
        ax.axvline(best['epoch'],color='green',linestyle='--',label='best epoch')
        ax.axvline(1, color='gray', linestyle='--', alpha=0.6, label='Mosaic disabled throughout')
        ax.grid(alpha=.2); ax.legend(fontsize=8); ax.set_xlabel('Epoch')
    fig.tight_layout(); fig.savefig(OUT/'training_trends.png',dpi=160); plt.close(fig)
    stats={}
    for split in ['train','val']:
        boxes=np.zeros(8,int); images=np.zeros(8,int); empty=0
        label_paths = ([DATA/line.strip().replace('images/','labels/').replace('./','') for line in (DATA/'train_hardaug2x_20261104.txt').read_text().splitlines()] if split=='train' else list((DATA/'labels'/split).glob('*.txt')))
        label_paths = [path.with_suffix('.txt') for path in label_paths]
        for path in label_paths:
            labels=[int(line.split()[0]) for line in path.read_text().splitlines() if line.strip()]
            empty += not labels
            for cls in labels: boxes[cls]+=1
            for cls in set(labels): images[cls]+=1
        stats[split]=dict(boxes=boxes.tolist(),images=images.tolist(),empty=empty,unique_images=len(set(label_paths)))
    (OUT/'dataset_counts.json').write_text(json.dumps(stats,indent=2))
    metric=json.loads((OUT/'metrics.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(15,5))
    pos=np.arange(8)
    for i,key in enumerate(['Box-P','Box-R','mAP50','mAP50-95']):
        axes[0].bar(pos+(i-1.5)*.2,[r[key]*100 for r in metric['per_class']],.2,label=key)
    axes[0].set_xticks(pos,NAMES,rotation=35,ha='right'); axes[0].set_ylim(0,105);axes[0].legend();axes[0].set_ylabel('%')
    axes[1].bar(pos-.2,stats['train']['boxes'],.4,label='train boxes')
    axes[1].bar(pos+.2,stats['val']['boxes'],.4,label='val boxes')
    axes[1].set_yscale('log');axes[1].set_xticks(pos,NAMES,rotation=35,ha='right');axes[1].legend();axes[1].set_ylabel('Box count (log scale)')
    fig.tight_layout();fig.savefig(OUT/'class_metrics_and_support.png',dpi=160);plt.close(fig)
    thresholds=[.1,.25,.5]
    counts={t:np.zeros((8,3),int) for t in thresholds}
    sizes=np.zeros((8,3,2),int)
    images=[]
    manifest=json.loads((OUT/'input_manifest.json').read_text())
    for record in manifest:
        path=OUT/'predictions'/Path(record['file']).with_suffix('.npz')
        with np.load(path) as z:
            gt,gc,pred,pc,conf=[z[k] for k in ['gt_boxes','gt_cls','boxes','cls','conf']]
        gc=gc.astype(int);pc=pc.astype(int)
        h,w=record['shape'];wh=(gt[:,2:]-gt[:,:2])*640/max(h,w)
        bucket=np.digitize(wh.prod(1),[32**2,96**2])
        for t in thresholds:
            selected=conf>=t; pairs=match(gt,gc,pred[selected],pc[selected])
            for c in range(8):
                tp=int((gc[pairs[:,0]]==c).sum())
                counts[t][c]+=[tp,int((pc[selected]==c).sum())-tp,int((gc==c).sum())-tp]
            if t==.25:
                hit=np.zeros(len(gt),bool);hit[pairs[:,0]]=True
                for c,b in zip(gc,bucket):sizes[c,b,0]+=1
                for c,b in zip(gc[hit],bucket[hit]):sizes[c,b,1]+=1
                for c in np.unique(gc):
                    n=int((gc==c).sum());tp=int(((gc==c)&hit).sum())
                    images.append(dict(file=record['file'],cls=int(c),gt=n,tp=tp,fn=n-tp,
                                       small=int(((gc==c)&(bucket==0)).sum())))
    rows=[]
    for t,values in counts.items():
        for c,(tp,fp,fn) in enumerate(values):
            rows.append(dict(conf=t,cls=c,name=NAMES[c],tp=int(tp),fp=int(fp),fn=int(fn),
                             precision=float(tp/(tp+fp)) if tp+fp else 0,recall=float(tp/(tp+fn)) if tp+fn else 0))
    diagnostic=dict(thresholds=rows,size_counts=sizes.tolist(),size_definition='GT box area after isotropic scaling max(h,w) to 640; small<32^2, medium<96^2, large>=96^2. Diagnostic bins, not COCO APs/APm/APl.',
                    matching='Class-aware IoU>=0.5, greedy IoU one-to-one; cached post-NMS predictions; confidence filtering after same NMS IoU=0.7.')
    (OUT/'diagnostics.json').write_text(json.dumps(diagnostic,indent=2))
    with (OUT/'image_diagnostics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=images[0]);writer.writeheader();writer.writerows(images)
    (OUT/'cases').mkdir(exist_ok=True)
    chosen=[]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    for c in range(8):
        candidates=[r for r in images if r['cls']==c]
        # Most missed objects first; retain a positive hose example because it has no misses.
        row=max(candidates,key=lambda r:(r['fn'],r['gt']))
        record=next(r for r in manifest if r['file']==row['file'])
        with np.load(OUT/'predictions'/Path(row['file']).with_suffix('.npz')) as z:
            gt,gc,pred,pc,conf=[z[k] for k in ['gt_boxes','gt_cls','boxes','cls','conf']]
        im=Image.open(DATA/'images/val'/row['file']).convert('RGB')
        selected=(pc==c)&(conf>=.25)
        panels=[]
        for title,boxes,scores,color in [('GT '+NAMES[c],gt[gc==c],None,'#28d268'),('PRED conf>=0.25',pred[selected],conf[selected],'#ff643c')]:
            canvas=im.copy();draw=ImageDraw.Draw(canvas)
            for i,box in enumerate(boxes):
                draw.rectangle(tuple(box),outline=color,width=3)
                if scores is not None:draw.text((float(box[0]),max(0,float(box[1])-22)),f'{scores[i]:.2f}',fill=color,font=font)
            canvas.thumbnail((800,600))
            panel=Image.new('RGB',(800,640),'white');panel.paste(canvas,(0,40));ImageDraw.Draw(panel).text((10,8),title,fill='black',font=font);panels.append(panel)
        merged=Image.new('RGB',(1600,640),'white');merged.paste(panels[0],(0,0));merged.paste(panels[1],(800,0))
        filename=f'{c}_{NAMES[c].replace(" ","_")}.jpg';merged.save(OUT/'cases'/filename)
        chosen.append(dict(**row,image=filename))
    (OUT/'cases/index.json').write_text(json.dumps(chosen,indent=2))
    print(json.dumps(dict(training=training,dataset=stats,diagnostics=diagnostic,cases=chosen),indent=2))
