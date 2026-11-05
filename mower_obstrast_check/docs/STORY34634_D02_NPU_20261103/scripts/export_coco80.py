"""Export the unchanged official COCO checkpoint for the standalone fallback audit."""
import hashlib
import json
from pathlib import Path
import sys

OUT=Path(__file__).resolve().parents[1]/'history_coco80'
ROOT=next(p for p in OUT.parents if (p/'ultralytics').is_dir())
sys.path.insert(0,str(ROOT))
import onnx
import torch
import ultralytics
from ultralytics import YOLO

torch.set_num_threads(6)
model=YOLO(OUT/'yolov8n.pt')
assert len(model.names)==80
options=dict(format='onnx',imgsz=640,batch=1,opset=12,simplify=False,dynamic=False,half=False,nms=False,device='cpu')
path=Path(model.export(**options))
graph=onnx.load(path);onnx.checker.check_model(graph)
(OUT/'export_config.json').write_text(json.dumps(dict(options=options,names=model.names,
    input_shape=[1,3,640,640],output_shape=[1,84,8400],
    versions=dict(torch=torch.__version__,ultralytics=ultralytics.__version__,onnx=onnx.__version__),
    pt_sha256=hashlib.sha256((OUT/'yolov8n.pt').read_bytes()).hexdigest(),
    onnx_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2))
