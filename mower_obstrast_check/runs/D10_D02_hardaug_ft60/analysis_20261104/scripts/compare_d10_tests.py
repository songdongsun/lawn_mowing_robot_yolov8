from pathlib import Path
import json,sys,hashlib,collections
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.p2-stage'))
from analyze_detect_d00 import match
from compare_detect_resolution import ious
B=ROOT/'mower_obstrast_check';R=B/'runs';OUT=R/'D10_D02_hardaug_ft60/analysis_20261104'
OUT.mkdir(exist_ok=True);(OUT/'test_cases').mkdir(exist_ok=True)
J=lambda p:json.loads(p.read_text(encoding='utf-8'))
source={x['name']:x for x in J(B/'data/test_sets/synthetic_8cls_140_20261030/manifest.json')['files']}
names=['person','cat','dog','frisbee','sports ball','teddy bear','garden hose','rock']
summaries={};cache={};rows=[]
for tag,date in [('D02','20261031'),('D05','20261101'),('D10','20261104')]:
    summaries[tag]={};cache[tag]={}
    for n in [140,90,50]:
        folder=R/f'{tag.lower()}test{n}_{date}';records=J(folder/'input_manifest.json');assert len(records)==n
        count={str(t):np.zeros((8,3),int) for t in [.1,.25,.5]};sizes=np.zeros((8,3,2),int);confusions=collections.Counter()
        for rec in records:
            file=rec['file'];assert rec['image_sha256']==source[file]['image_sha256'] and rec['label_sha256']==source[file]['label_sha256']
            with np.load(folder/'predictions'/Path(file).with_suffix('.npz')) as z:
                gt,gc,pred,pc,conf=[z[k] for k in ['gt_boxes','gt_cls','boxes','cls','conf']]
            gc=gc.astype(int);pc=pc.astype(int);bucket=np.digitize(((gt[:,2:]-gt[:,:2])*640/max(rec['shape'])).prod(1),[32**2,96**2])
            for t in [.1,.25,.5]:
                sel=conf>=t;pr=pred[sel];cc=pc[sel];pairs=match(gt,gc,pr,cc);hit=np.zeros(len(gt),bool);hit[pairs[:,0]]=True
                for c in range(8):
                    tp=int(((gc==c)&hit).sum());fp=int((cc==c).sum())-tp;fn=int(((gc==c)&~hit).sum());count[str(t)][c]+=[tp,fp,fn]
                    if n==140 and t==.25:rows.append(dict(model=tag,file=file,cls=c,tp=tp,fp=fp,fn=fn,gt=tp+fn))
                if t==.25:
                    for c,b,h in zip(gc,bucket,hit):sizes[c,b]+=[1,int(h)]
                    used=set(pairs[:,1]);ov=ious(gt,pr)
                    for j in range(len(pr)):
                        if j in used or len(gt)==0:continue
                        i=int(ov[:,j].argmax())
                        if ov[i,j]>=.5 and gc[i]!=cc[j]:confusions[f'{names[gc[i]]}->{names[cc[j]]}']+=1
                    if n==140:cache[tag][file]=dict(gt=gt,gc=gc,pred=pr,pc=cc,conf=conf[sel],hit=hit)
        summaries[tag][str(n)]=dict(metrics=J(folder/'metrics.json'),totals={t:x.tolist() for t,x in count.items()},sizes=sizes.tolist(),confusion_boxes=dict(confusions))
cases=[]
for c in range(8):
    group=[r for r in rows if r['model']=='D10' and r['cls']==c and r['gt']]
    for kind,key in [('miss',lambda r:r['fn']),('fp',lambda r:r['fp'])]:
        row=max(group,key=key)
        if kind=='fp' and key(row)==0:continue
        cases.append(dict(cls=c,file=row['file'],kind=kind))
for c,file in [(6,'testmix_015.png'),(7,'test90_c7_010.png'),(4,'test90_c4_001.png')]:cases.append(dict(cls=c,file=file,kind='previous_case'))
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
for row in cases:
    c=row['cls'];file=row['file'];im=Image.open(B/'data/test_sets/synthetic_8cls_140_20261030/images/test'/file).convert('RGB')
    canvas=Image.new('RGB',(2560,520),'white')
    for col,tag in enumerate(['GT','D02','D05','D10']):
        z=cache['D10' if tag=='GT' else tag][file];view=im.copy();draw=ImageDraw.Draw(view)
        boxes=z['gt'] if tag=='GT' else z['pred'];cls=z['gc'] if tag=='GT' else z['pc']
        for i,(box,cc) in enumerate(zip(boxes,cls)):
            color='#22cc55' if tag=='GT' else '#ff6138';draw.rectangle(tuple(box),outline=color,width=2)
            draw.text((float(box[0]),max(0,float(box[1])-17)),names[cc]+('' if tag=='GT' else f' {z["conf"][i]:.2f}'),fill=color,font=font)
        view.thumbnail((640,480));canvas.paste(view,(col*640,40));ImageDraw.Draw(canvas).text((col*640+8,8),tag+('' if tag=='GT' else ' conf>=.25'),fill='black',font=font)
        if tag!='GT':
            tp=int(((z['gc']==c)&z['hit']).sum());row[tag]=dict(tp=tp,fp=int((z['pc']==c).sum())-tp,fn=int((z['gc']==c).sum())-tp)
    row['image']=f'{c}_{row["kind"]}.jpg';canvas.save(OUT/'test_cases'/row['image'])
(OUT/'test_comparison.json').write_text(json.dumps(dict(groups=summaries,cases=cases,per_image=rows),indent=2),encoding='utf-8')
print('Test metrics, fixed thresholds, size buckets and cases ready')
for tag in summaries:print(tag,summaries[tag]['140']['metrics']['overall'],summaries[tag]['140']['totals']['0.25'])
