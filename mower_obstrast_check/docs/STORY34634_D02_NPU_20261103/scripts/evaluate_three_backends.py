"""Same-input static audit. Run 'reference' first, then 'npu' after board download."""
import csv
import hashlib
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
import torchvision  # Explicit CPU NMS backend; stable score-tie ordering matches board C++.
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import DetMetrics
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes
from reference_match import match

torch.set_num_threads(6)
manifest=json.loads((OUT/'board/input_manifest.json').read_text(encoding='utf-8'))
names=['person','cat','dog','frisbee','sports ball','teddy bear','garden hose','rock']
mode=sys.argv[1]
variant=sys.argv[2] if len(sys.argv)>2 else 'uint8_hybrid'
assert variant in ['uint8_hybrid','fp16','head_tail_fp16','attention_tail_fp16','attention_tail_kl','p5_class_fp16','coco80_kl','coco80_fp16','coco80_int8']
coco=variant in ['coco80_kl','coco80_fp16','coco80_int8']
if coco:
    names=list(json.loads((OUT/'history_coco80/export_config.json').read_text())['names'].values())
nc=len(names)
board_output=OUT/'board'/('outputs' if variant=='uint8_hybrid' else variant+'/outputs')
evaluation=OUT/('evaluation' if variant in ['uint8_hybrid','fp16'] else 'variants/'+variant)
if coco:
    evaluation=OUT/'variants/coco80_int8' if variant=='coco80_int8' else OUT/'history_coco80/evaluation'
    evaluation.mkdir(exist_ok=True)
tags=['PT','ONNX'] if mode=='reference' else ['NPU']
metrics={t:DetMetrics(names=dict(enumerate(names))) for t in tags}
totals={t:np.zeros((nc,3),int) for t in tags}
sizes={t:np.zeros((nc,3,2),int) for t in tags}
predictions={t:[] for t in tags}
validator=object.__new__(DetectionValidator)
validator.iouv=torch.linspace(.5,.95,10);validator.niou=10
cache=OUT/('history_coco80/reference_raw' if coco else 'evaluation/reference_raw');cache.mkdir(exist_ok=True)
if mode=='reference':
    import onnxruntime as ort
    source=OUT/'history_coco80' if coco else OUT.parent/'STORY34629_D02_ONNX_20261103'
    stem='yolov8n' if coco else 'best'
    model=YOLO(source/(stem+'.pt')).model.float().eval().fuse()
    so=ort.SessionOptions();so.intra_op_num_threads=6;so.inter_op_num_threads=1
    session=ort.InferenceSession(str(source/(stem+'.onnx')),so,providers=['CPUExecutionProvider'])
rows=[]
for i,r in enumerate(manifest):
    p=ROOT/r['image'];label=ROOT/r['label']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==r['image_sha256']
    assert hashlib.sha256(label.read_bytes()).hexdigest()==r['label_sha256']
    im_path=OUT/'board/inputs'/f"{r['id']}.png"
    assert hashlib.sha256(im_path.read_bytes()).hexdigest()==r['prepared_sha256']
    image=cv2.imread(str(im_path))
    gt=np.array([list(map(float,l.split())) for l in label.read_text().splitlines() if l.strip()],np.float32).reshape(-1,5)
    if coco:
        gt=gt[gt[:,0]<6]
        gt[:,0]=np.array([0,15,16,29,32,77])[gt[:,0].astype(int)]
    gc=gt[:,0].astype(int);w,h=r['width'],r['height']
    gb=np.column_stack(((gt[:,1:3]-gt[:,3:5]/2)*[w,h],(gt[:,1:3]+gt[:,3:5]/2)*[w,h])).astype(np.float32)
    bucket=np.digitize(((gb[:,2:]-gb[:,:2])*640/max(w,h)).prod(1),[32**2,96**2])
    if mode=='reference':
        x=np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255
        with torch.inference_mode(): pt=model(torch.from_numpy(x))[0].numpy()
        ox=session.run(None,{'images':x})[0]
        raws={'PT':pt,'ONNX':ox}
        np.savez_compressed(cache/f"{r['id']}.npz",onnx=ox)
    else:
        raw=np.fromfile(board_output/f"{r['id']}.f32",np.float32)
        assert raw.size==(nc+4)*8400 and np.isfinite(raw).all()
        raws={'NPU':raw.reshape(1,nc+4,8400)}
    row=dict(id=r['id'],file=p.name)
    for tag,raw in raws.items():
        low=non_max_suppression(torch.from_numpy(raw.copy()),conf_thres=.001,iou_thres=.7,max_det=300,nc=nc,multi_label=True)[0]
        low[:,:4]=scale_boxes((640,640),low[:,:4],(h,w))
        stat=dict(tp=validator._process_batch({'bboxes':low[:,:4],'cls':low[:,5]},
                  {'bboxes':torch.from_numpy(gb),'cls':torch.from_numpy(gc)})['tp'],conf=low[:,4].numpy(),
                  pred_cls=low[:,5].numpy(),target_cls=gc,target_img=np.unique(gc))
        for k,v in stat.items():metrics[tag].stats[k].append(v)
        pred=non_max_suppression(torch.from_numpy(raw.copy()),conf_thres=.25,iou_thres=.7,max_det=300,nc=nc,multi_label=False)[0]
        pred[:,:4]=scale_boxes((640,640),pred[:,:4],(h,w));pred=pred.numpy()
        pairs=match(gb,gc,pred[:,:4],pred[:,5]);hit=np.zeros(len(gc),bool);hit[pairs[:,0]]=True
        for cls in range(nc):
            tp=int(((gc==cls)&hit).sum());totals[tag][cls]+=[tp,int((pred[:,5]==cls).sum())-tp,int(((gc==cls)&~hit).sum())]
        for cls,b,ok in zip(gc,bucket,hit):sizes[tag][cls,b]+=[1,int(ok)]
        predictions[tag].append(dict(id=r['id'],file=p.name,boxes=pred.tolist(),gt_boxes=gb.tolist(),gt_classes=gc.tolist(),gt_hit=hit.tolist()))
        row[tag+'_boxes']=len(pred)
        if mode=='npu':
            lines=(board_output/f"{r['id']}.txt").read_text().splitlines()
            board=np.array([list(map(float,l.split())) for l in lines],np.float32).reshape(-1,6)
            pairs=match(pred[:,:4],pred[:,5],board[:,:4],board[:,5])
            row['board_postprocess_unmatched']=len(pred)+len(board)-2*len(pairs)
            row['board_postprocess_max_coord_error']=float(np.max(np.abs(pred[pairs[:,0],:4]-board[pairs[:,1],:4]))) if len(pairs) else 0
            ox=np.load(cache/f"{r['id']}.npz")['onnx']
            row['raw_box_abs']=float(np.abs(raw[:,:4]-ox[:,:4]).max())
            row['raw_score_abs']=float(np.abs(raw[:,4:]-ox[:,4:]).max())
    rows.append(row)
    if (i+1)%20==0:print(mode,i+1,flush=True)
for tag in tags:
    metrics[tag].process()
    result=dict(images=len(manifest),scope='six shared labelled classes only; not COCO80 benchmark; hose/rock excluded' if coco else 'project eight classes',nms_backend='torchvision '+torchvision.__version__,overall=metrics[tag].results_dict,classes=metrics[tag].summary(decimals=8),
                totals_columns=['TP','FP','FN'],totals_conf025=totals[tag].tolist(),
                size_bins='area after scaling longest side to 640: small<32^2; medium<96^2; large>=96^2',
                size_counts_gt_hit=sizes[tag].tolist(),ap_config=dict(conf=.001,iou=.7,max_det=300,multi_label=True),
                deployment_config=dict(conf=.25,iou=.7,max_det=300,multi_label=False))
    if coco:
        supported={0,15,16,29,32,77}
        result['unassessed_detection_counts']={names[i]:int(totals[tag][i,1]) for i in range(nc) if i not in supported and totals[tag][i,1]}
        result['totals_conf025']=[r if i in supported else None for i,r in enumerate(result['totals_conf025'])]
        result['size_counts_gt_hit']=[r if i in supported else None for i,r in enumerate(result['size_counts_gt_hit'])]
    (evaluation/f'{tag}_metrics.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item()))
    (evaluation/f'{tag}_predictions.json').write_text(json.dumps(predictions[tag],indent=2))
    print(tag,result['overall'],flush=True)
with (evaluation/f'{mode}_per_image.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
