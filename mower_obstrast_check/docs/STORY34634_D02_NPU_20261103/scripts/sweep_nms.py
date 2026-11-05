"""Compare NMS thresholds on unchanged raw outputs, tracking individual lost/recovered GTs."""
import json
from pathlib import Path
import sys
OUT=Path(__file__).resolve().parents[1]
ROOT=next(p for p in OUT.parents if (p/'ultralytics').is_dir())
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(OUT.parent/'STORY34629_D02_ONNX_20261103/scripts'))
import cv2
import numpy as np
import torch
import torchvision
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes
from reference_match import match

torch.set_num_threads(4)
mode=sys.argv[1]
if mode=='test140':
    records=json.loads((OUT/'board/input_manifest.json').read_text(encoding='utf-8'))
    tags=['ONNX','uint8_hybrid','fp16']
else:
    assert mode in ['val154','val2938']
    if mode=='val154':
        records=[r for r in json.loads((OUT.parent/'STORY34629_D02_ONNX_20261103/comparison/input_manifest.json').read_text(encoding='utf-8')) if r['split']=='val_stratified']
    else:
        data=ROOT/'mower_obstrast_check/data/yolo_check'
        records=[dict(image=str(p.relative_to(ROOT)),label=str((data/'labels/val'/(p.stem+'.txt')).relative_to(ROOT))) for p in sorted((data/'images/val').glob('*'))]
        assert len(records)==2938
    tags=['ONNX']
    import onnxruntime as ort
    so=ort.SessionOptions();so.intra_op_num_threads=4
    session=ort.InferenceSession(str(OUT.parent/'STORY34629_D02_ONNX_20261103/best.onnx'),so,providers=['CPUExecutionProvider'])
thresholds=[.7,.6,.5,.45,.4]
profiles=[(str(t),t,.25) for t in thresholds]+[('0.6_person0.3',.6,.3),('0.6_person0.35',.6,.35),('0.6_containment',.6,.25)]
if mode=='val2938':profiles=[('0.7',.7,.25),('0.6',.6,.25)]
result={tag:{name:dict(totals=np.zeros((8,3),int),lost=np.zeros(8,int),recovered=np.zeros(8,int),cases=[]) for name,_,_ in profiles} for tag in tags}
raw_cache=OUT/'optimization_20261103/val_raw';raw_cache.mkdir(exist_ok=True)
for i,r in enumerate(records):
    p=ROOT/r['image'];label=ROOT/r['label']
    image=cv2.imread(str(p));h,w=image.shape[:2]
    gt=np.array([list(map(float,l.split())) for l in label.read_text().splitlines() if l.strip()],np.float32).reshape(-1,5)
    gc=gt[:,0].astype(int);gb=np.column_stack(((gt[:,1:3]-gt[:,3:5]/2)*[w,h],(gt[:,1:3]+gt[:,3:5]/2)*[w,h]))
    if mode.startswith('val'):
        cached=raw_cache/(p.stem+'.npz')
        if cached.exists():raws={'ONNX':np.load(cached)['onnx']}
        else:
            padded=LetterBox(new_shape=(640,640),auto=False)(image=image)
            x=np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255
            raws={'ONNX':session.run(None,{'images':x})[0]}
            if mode=='val154':np.savez_compressed(cached,onnx=raws['ONNX'])
    else:
        raws={'ONNX':np.load(OUT/'evaluation/reference_raw'/f"{r['id']}.npz")['onnx']}
        for tag,folder in [('uint8_hybrid','outputs'),('fp16','fp16/outputs')]:
            raws[tag]=np.fromfile(OUT/'board'/folder/f"{r['id']}.f32",np.float32).reshape(1,12,8400)
    for tag,raw in raws.items():
        for name,t,person_conf in profiles:
            pred=non_max_suppression(torch.from_numpy(raw.copy()),conf_thres=.25,iou_thres=t,max_det=300,nc=8,multi_label=False)[0]
            pred[:,:4]=scale_boxes((640,640),pred[:,:4],(h,w));pred=pred.numpy()
            pred=pred[(pred[:,5]!=0)|(pred[:,4]>=person_conf)]
            if name=='0.6_containment':
                kept=[]
                for box in pred:
                    area=(box[2]-box[0])*(box[3]-box[1])
                    redundant=False
                    for other in kept:
                        intersection=np.maximum(0,np.minimum(box[2:4],other[2:4])-np.maximum(box[:2],other[:2])).prod()
                        if box[5]==other[5] and box[4]<=other[4]*.5 and intersection/(area+1e-7)>=.95:
                            redundant=True;break
                    if not redundant:kept.append(box)
                pred=np.array(kept,dtype=np.float32).reshape(-1,6)
            pairs=match(gb,gc,pred[:,:4],pred[:,5]);hit=np.zeros(len(gc),bool);hit[pairs[:,0]]=True
            if name=='0.7':base=hit.copy()
            group=result[tag][name]
            for cls in range(8):
                tp=int(((gc==cls)&hit).sum());group['totals'][cls]+=[tp,int((pred[:,5]==cls).sum())-tp,int(((gc==cls)&~hit).sum())]
                group['lost'][cls]+=int(((gc==cls)&base&~hit).sum())
                group['recovered'][cls]+=int(((gc==cls)&~base&hit).sum())
            if (base!=hit).any() or i==0:
                group['cases'].append(dict(file=p.name,lost_classes=gc[base&~hit].tolist(),recovered_classes=gc[~base&hit].tolist(),boxes=pred.tolist()))
    if (i+1)%40==0:print(mode,i+1,flush=True)
target=OUT/'optimization_20261103'/f'nms_sweep_{mode}.json'
target.write_text(json.dumps(dict(images=len(records),conf=.25,nms_backend='torchvision '+torchvision.__version__,results=result),indent=2,default=lambda x:x.tolist()))
for tag,groups in result.items():
    for t,g in groups.items():print(tag,t,g['totals'].sum(axis=0).tolist(),'lost',g['lost'].tolist(),'recover',g['recovered'].tolist(),flush=True)
