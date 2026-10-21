"""Diagnostic only: reuse the frozen shared decoder; never change training or deployment."""
import csv
import importlib.util
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import ultralytics

OUT = Path(__file__).resolve().parent
GRASS = OUT.parents[1]
old = GRASS/'deliverables/STORY25028_B08_v3_2_20261014/09_scripts/compare_mower_backends.py'
spec = importlib.util.spec_from_file_location('shared_decoder', old)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
for folder in ('originals', 'inputs', 'outputs', 'reference', 'masks', 'comparisons', 'ros_saved'):
    (OUT/folder).mkdir(exist_ok=True)
pt = GRASS/'runs/segment/B08_v3_2/weights/best.pt'
onnx = GRASS/'data/b08_v3_2_consistency_20261014/models/b08_v3_2.onnx'
assert shared.sha(pt) == 'ab8e640f665208ad80857a29786b4111d17774b7bb36773d83967d2afebe20ff'
assert shared.sha(onnx) == '05ef0fe144942c2c2a7f29d26c85506729eee959a57cd2dd56f84b901af70089'
if (OUT/'board_checksums.txt').exists():
    for line in (OUT/'board_checksums.txt').read_text().splitlines():
        checksum, relative = line.split()
        path = (GRASS/'data/b08_v3_2_consistency_20261014/models/network_binary.nb'
                if relative == 'network_binary.nb' else OUT/relative)
        assert shared.sha(path) == checksum, relative
figures = json.loads((GRASS/'STORY25034_20261020_assets/figure_sources.json').read_text(encoding='utf-8'))[1:]
torch.set_num_threads(4)
model = ultralytics.YOLO(str(pt)).model.float().eval()
native = ultralytics.YOLO(str(pt))
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
session = ort.InferenceSession(str(onnx), opts, providers=['CPUExecutionProvider'])
rows = []
details = []
for idx, figure in enumerate(figures):
    sid = f'{idx:03d}'
    source = GRASS/f'STORY25034_20261020_assets/fig{figure["figure"]}_original.png'
    assert shared.sha(source) == figure['original_sha256']
    shutil.copy2(source, OUT/'originals'/f'{sid}.png')
    # Convert the recorded Windows path to WSL only for reading the original saved Mask.
    maskpath = figure['mask_path'].replace('\\','/')
    if str(OUT).startswith('/mnt/'):
        maskpath = '/mnt/'+maskpath[0].lower()+maskpath[2:]
    assert shared.sha(maskpath) == figure['mask_sha256']
    shutil.copy2(maskpath, OUT/'ros_saved'/f'{sid}.png')
    image = cv2.imread(str(source))
    padded, geom = shared.prepare(image)
    cv2.imwrite(str(OUT/'inputs'/f'{sid}.png'), padded)
    tensor = padded[:,:,::-1].transpose(2,0,1)[None].copy().astype(np.float32)/255
    with torch.inference_mode():
        raw = model(torch.from_numpy(tensor))
    if isinstance(raw[0], (tuple,list)):
        pt_values = (raw[0][0][0].numpy(), raw[0][1][0].numpy())
    else:
        pt_values = (raw[0][0].numpy(), raw[1][-1][0].numpy())
    values = session.run(None, {session.get_inputs()[0].name:tensor})
    shapes = {tuple(v.shape):v for v in values}
    arrays = dict(pt_shared=pt_values,
                  onnx_shared=(shapes[(1,37,8400)][0], shapes[(1,32,160,160)][0]))
    np.savez_compressed(OUT/'reference'/f'{sid}.npz', pt0=pt_values[0],pt1=pt_values[1],
                        onnx0=arrays['onnx_shared'][0],onnx1=arrays['onnx_shared'][1])
    if (OUT/'outputs'/f'{sid}_0.f32').exists():
        arrays['npu_shared']=(np.fromfile(OUT/'outputs'/f'{sid}_0.f32',dtype='<f4').reshape(37,8400),
                              np.fromfile(OUT/'outputs'/f'{sid}_1.f32',dtype='<f4').reshape(32,160,160))
    masks = {}; info={}
    for name, pair in arrays.items():
        masks[name], info[name] = shared.decode(*pair,geom,outward_crop=True)
    for threshold in (.45,.7):
        name=f'pt_native_nms{threshold}'
        result=native.predict(padded,imgsz=640,conf=.25,iou=threshold,rect=False,
                              retina_masks=True,device='cpu',verbose=False)[0]
        actual=native.predictor.preprocess([padded]).cpu().numpy()
        assert np.array_equal(actual,tensor)
        mask=np.zeros((640,640),bool) if result.masks is None else (
            result.masks.data.cpu().numpy()>.5).any(axis=0)
        masks[name]=mask[80:560,:]
        info[name]=dict(detections=result.boxes.data.cpu().tolist(),
                        foreground_pixels=int(masks[name].sum()))
    masks['ros_saved']=cv2.imread(str(OUT/'ros_saved'/f'{sid}.png'),0)>0
    static=OUT/'board_static/masks'/f'{sid}.png.png'
    if static.exists(): masks['npu_static']=cv2.imread(str(static),0)>0
    for name,mask in masks.items():
        assert mask.shape==(480,640)
        cv2.imwrite(str(OUT/'masks'/f'{sid}_{name}.png'),mask.astype(np.uint8)*255)
        info.setdefault(name,{})['foreground_pixels']=int(mask.sum())
    for a,b in [('pt_shared','onnx_shared'),('pt_shared','ros_saved'),
                ('pt_native_nms0.45','pt_shared'),('pt_native_nms0.7','pt_native_nms0.45'),
                ('pt_shared','npu_shared'),('npu_shared','ros_saved'),
                ('npu_static','ros_saved'),('npu_shared','npu_static')]:
        if a in masks and b in masks:
            rows.append(dict(id=sid,case=figure['figure'],pair=f'{a}_vs_{b}',**shared.measure(masks[a],masks[b])))
    names=['pt_native_nms0.7','pt_shared','onnx_shared','npu_shared','npu_static','ros_saved']
    panels=[shared.visualize(image,np.zeros((480,640),bool),sid+' original')]
    panels += [shared.visualize(image,masks[n],n) for n in names if n in masks]
    # Two rows, consistent panel size; no changes to source images.
    while len(panels)%3: panels.append(np.full_like(panels[0],240))
    cv2.imwrite(str(OUT/'comparisons'/f'{sid}.jpg'),
                np.concatenate([np.concatenate(panels[j:j+3],axis=1) for j in range(0,len(panels),3)],axis=0))
    details.append(dict(id=sid,source=figure,backends=info))
    print(sid,json.dumps(info),flush=True)
shared.write_json(OUT/'details.json',details)
shared.write_json(OUT/'summary.json',dict(purpose='backend consistency, NOT ground-truth accuracy',
    policy='640 square letterbox 114; BGR uint8 NPU; RGB/255 PT/ONNX; shared conf .25 NMS .45 mask .5 outward crop',
    native_policy='retina_masks=True, rect=False; NMS .45 and historical .7 kept separate',
    models=dict(pt=shared.sha(pt),onnx=shared.sha(onnx)),decoder_sha256=shared.sha(old),
    versions=dict(torch=torch.__version__,ultralytics=ultralytics.__version__,
                  onnxruntime=ort.__version__,opencv=cv2.__version__),pairs=rows))
with (OUT/'per_case.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
(OUT/'input_list.txt').write_text('000\n001\n002\n')
print(json.dumps(rows,indent=2))
