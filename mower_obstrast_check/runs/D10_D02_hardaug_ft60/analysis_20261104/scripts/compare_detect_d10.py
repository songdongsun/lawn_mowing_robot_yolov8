import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analyze_detect_d00 import match,NAMES
import numpy as np
import torch
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import DetMetrics
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont

RUNS=ROOT/'mower_obstrast_check/runs'
RUN=RUNS/'D10_D02_hardaug_ft60'
OUT=RUN/'analysis_20261104'
OUT.mkdir(exist_ok=True)
DATA=ROOT/'mower_obstrast_check/data/yolo_check'
FOLDERS={'D02_old':RUNS/'d02old_20261029','D10_old':RUNS/'d10old_20261104','D02_full':RUNS/'d02full_20261029','D10_full':RUNS/'d10full_20261104'}
old_manifest=json.loads((FOLDERS['D02_old']/'input_manifest.json').read_text())
old_files={r['file'] for r in old_manifest}
manifests={k:json.loads((p/'input_manifest.json').read_text()) for k,p in FOLDERS.items()}
assert {r['file']:r for r in manifests['D02_old']}=={r['file']:r for r in manifests['D10_old']}
assert {r['file']:r for r in manifests['D02_full']}=={r['file']:r for r in manifests['D10_full']}
assert len(manifests['D10_full'])==2938 and len(old_files)==2801
for r in old_manifest:assert r==next(x for x in manifests['D10_full'] if x['file']==r['file'])
validator=object.__new__(DetectionValidator)
validator.iouv=torch.linspace(.5,.95,10);validator.niou=10
summaries={};image_rows={};cache={}

def new_group():
    return dict(n=0,totals={str(t):np.zeros((8,3),int) for t in [.1,.25,.5]},sizes=np.zeros((8,3,2),int),metric=DetMetrics(names=dict(enumerate(NAMES))))

for tag,folder in FOLDERS.items():
    groups={'all':new_group()}
    if tag.endswith('full'):groups.update(new_rock=new_group(),new_hose=new_group())
    image_rows[tag]=[];cache[tag]={}
    for record in manifests[tag]:
        filename=record['file']
        with np.load(folder/'predictions'/Path(filename).with_suffix('.npz')) as z:
            gt,gc,pred,pc,conf=[z[k] for k in ['gt_boxes','gt_cls','boxes','cls','conf']]
        gc=gc.astype(int);pc=pc.astype(int)
        bucket=np.digitize(((gt[:,2:]-gt[:,:2])*640/max(record['shape'])).prod(1),[32**2,96**2])
        selection=['all']
        if tag.endswith('full') and filename not in old_files:
            selection.append('new_hose' if filename.startswith('ai_garden_hose') else 'new_rock')
        stat=dict(tp=validator._process_batch({'bboxes':torch.from_numpy(pred),'cls':torch.from_numpy(pc)}, {'bboxes':torch.from_numpy(gt),'cls':torch.from_numpy(gc)})['tp'],conf=conf,pred_cls=pc,target_cls=gc,target_img=np.unique(gc))
        vals={};hit25=None
        for threshold in [.1,.25,.5]:
            sel=conf>=threshold;pairs=match(gt,gc,pred[sel],pc[sel]);hit=np.zeros(len(gt),bool);hit[pairs[:,0]]=True
            vals[str(threshold)]=np.array([[int(((gc==c)&hit).sum()),int((pc[sel]==c).sum())-int(((gc==c)&hit).sum()),int(((gc==c)&~hit).sum())] for c in range(8)])
            if threshold==.25:hit25=hit
        sizes=np.zeros((8,3,2),int)
        for c,b,h in zip(gc,bucket,hit25):sizes[c,b]+=[1,int(h)]
        for key in selection:
            group=groups[key];group['n']+=1;group['sizes']+=sizes
            for t,v in vals.items():group['totals'][t]+=v
            for k,v in stat.items():group['metric'].stats[k].append(v)
        cache[tag][filename]=dict(gt=gt,gc=gc,pred=pred,pc=pc,conf=conf,hit=hit25,bucket=bucket)
        for c in np.unique(gc):
            tp,fp,fn=vals['0.25'][c]
            image_rows[tag].append(dict(file=filename,cls=int(c),gt=int((gc==c).sum()),tp=int(tp),fp=int(fp),fn=int(fn),small=int(((gc==c)&(bucket==0)).sum())))
    summaries[tag]={}
    for key,g in groups.items():
        g['metric'].process()
        summaries[tag][key]=dict(images=g['n'],totals={k:v.tolist() for k,v in g['totals'].items()},sizes=g['sizes'].tolist(),metrics=g['metric'].summary(decimals=10),overall=g['metric'].results_dict)
    # Cache-based AP must reproduce direct validator AP to rounding/numeric tolerance.
    actual=json.loads((folder/'metrics.json').read_text())['overall']['metrics/mAP50-95(B)']
    assert abs(summaries[tag]['all']['overall']['metrics/mAP50-95(B)']-actual)<.001,(tag,actual)
    print(tag,summaries[tag]['all']['overall'],flush=True)

assert summaries['D10_full']['new_rock']['images']==42 and summaries['D10_full']['new_hose']['images']==95
paired=[];transitions=np.zeros((8,3,4),int)
for record in old_manifest:
    filename=record['file'];a=cache['D02_old'][filename];b=cache['D10_old'][filename]
    assert np.array_equal(a['gc'],b['gc']) and np.allclose(a['gt'],b['gt'],atol=.002)
    for c,bin_,h0,h1 in zip(a['gc'],a['bucket'],a['hit'],b['hit']):
        kind=0 if h0 and h1 else 1 if h1 else 2 if h0 else 3
        transitions[c,bin_,kind]+=1
    for c in np.unique(a['gc']):
        mask=a['gc']==c
        paired.append(dict(file=filename,cls=int(c),gt=int(mask.sum()),tp0=int((mask&a['hit']).sum()),tp1=int((mask&b['hit']).sum()),recovered=int((mask&~a['hit']&b['hit']).sum()),lost=int((mask&a['hit']&~b['hit']).sum())))
(OUT/'comparison.json').write_text(json.dumps(dict(groups=summaries,transitions=transitions.tolist(),transition_order=['both_hit','recovered_D10','lost_D10','both_missed'],fixed_input_hashes_verified=True),indent=2,default=lambda v:v.item()),encoding='utf-8')
with (OUT/'paired_images.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=paired[0]);w.writeheader();w.writerows(paired)
for tag,rows in image_rows.items():
    with (OUT/(tag+'_image_diagnostics.csv')).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

fig,axes=plt.subplots(1,3,figsize=(17,5));x=np.arange(8)
for tag,off in [('D02_old',-.2),('D10_old',.2)]:
    g=summaries[tag]['all'];sizes=np.array(g['sizes']);t=np.array(g['totals']['0.25'])
    axes[0].bar(x+off,np.divide(sizes[:,0,1],sizes[:,0,0],out=np.zeros(8),where=sizes[:,0,0]>0)*100,.4,label=tag)
    axes[1].bar(x+off,t[:,0]/(t[:,0]+t[:,2])*100,.4,label=tag)
    axes[2].bar(x+off,t[:,1],.4,label=tag)
for ax,title in zip(axes,['Small-object recall: original 2801','All-size recall @ conf=.25','False-positive boxes @ conf=.25']):
    ax.set_title(title);ax.set_xticks(x,NAMES,rotation=40,ha='right');ax.legend();ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(OUT/'fixed_comparison.png',dpi=160);plt.close(fig)

cases=[]
old_cases=json.loads((RUNS/'D02_CA_200ep_3gpu/analysis_20261029/cases/index.json').read_text())
for c in range(8):
    oldcase=next(r for r in old_cases if r['cls']==c)
    cases.append(dict(cls=c,file=oldcase['file'],tag='previous_hard',pool='old'))
    group=[r for r in paired if r['cls']==c]
    for tag,key in [('gain',lambda r:r['tp1']-r['tp0']),('regression',lambda r:r['tp0']-r['tp1'])]:
        r=max(group,key=key)
        if key(r)>0:cases.append(dict(cls=c,file=r['file'],tag=tag,pool='old'))
for prefix,c in [('obstructions_rock_',7),('ai_garden_hose',6)]:
    group=[r for r in image_rows['D10_full'] if r['cls']==c and r['file'].startswith(prefix)]
    for tag,key in [('new_hard',lambda r:r['fn']),('new_fp',lambda r:r['fp'])]:
        r=max(group,key=key);cases.append(dict(cls=c,file=r['file'],tag=tag,pool='full'))
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
(OUT/'cases').mkdir(exist_ok=True)
for row in cases:
    c=row['cls'];file=row['file'];pool=row['pool']
    a=cache['D02_'+pool][file];b=cache['D10_'+pool][file]
    im=Image.open(DATA/'images/val'/file).convert('RGB');canvas=Image.new('RGB',(1920,520),'white')
    for col,(title,z,is_gt) in enumerate([('GT',a,True),('D02 conf>=.25',a,False),('D10 conf>=.25',b,False)]):
        sel=z['gc']==c if is_gt else (z['pc']==c)&(z['conf']>=.25)
        boxes=z['gt'][sel] if is_gt else z['pred'][sel];scores=None if is_gt else z['conf'][sel]
        view=im.copy();draw=ImageDraw.Draw(view)
        for i,box in enumerate(boxes):
            draw.rectangle(tuple(box),outline='#22cc55' if is_gt else '#ff6138',width=3)
            if scores is not None:draw.text((float(box[0]),max(0,float(box[1])-18)),f'{scores[i]:.2f}',fill='#ff6138',font=font)
        view.thumbnail((640,480));canvas.paste(view,(col*640,40));ImageDraw.Draw(canvas).text((col*640+8,8),NAMES[c]+' '+title,fill='black',font=font)
    row['image']=f'{c}_{row["tag"]}.jpg';canvas.save(OUT/'cases'/row['image'])
    row['gt']=int((a['gc']==c).sum())
    for name,z in [('D02',a),('D10',b)]:
        tp=int(((z['gc']==c)&z['hit']).sum());preds=int(((z['pc']==c)&(z['conf']>=.25)).sum())
        row[name]=dict(tp=tp,fp=preds-tp,fn=row['gt']-tp)
(OUT/'cases/index.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
print('Comparison, source groups, fixed thresholds, size buckets and cases written.',flush=True)
