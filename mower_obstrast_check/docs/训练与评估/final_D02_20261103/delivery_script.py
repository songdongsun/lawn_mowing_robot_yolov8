from pathlib import Path
import json,hashlib,shutil,time,sys,platform,subprocess,csv
R=next(p for p in Path(__file__).resolve().parents if (p/"pyproject.toml").exists() and (p/"mower_obstrast_check").is_dir());sys.path.insert(0,str(R))
import torch,numpy as np,yaml
from ultralytics import YOLO,__version__
B=R/'mower_obstrast_check/runs'; O=R/'mower_obstrast_check/docs/训练与评估/final_D02_20261103';O.mkdir(exist_ok=True)
J=lambda p:json.loads(p.read_text(encoding='utf-8'))
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def cp(a,b):b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b);assert H(a)==H(b)
def table(h,rs):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rs])
P=lambda v:f'{100*v:.3f}'
runs={'D01':'D01_data_expanded_baseline_2gpu','D02':'D02_CA_200ep_3gpu','D04':'D04_CA_AdamW_400ep_3gpu','D05':'D05_CA_AdamW_200ep_noearly_3gpu','D06':'D06_CA_AdamW_300ep_scale03_mosaic05_gpu47','D07':'D07_P2_CA_AdamW_300ep_scale03_mosaic05_3gpu','D08':'D08_CA_AdamW_300ep_cls075_augstrong_gpu67','D09':'D09_P2_CA_AdamW_300ep_cls075_augstrong_gpu23'}
dates={'D01':'20261029','D02':'20261029','D04':'20261031','D05':'20261101','D06':'20261102','D07':'20261102','D08':'20261103','D09':'20261103'}
scopes=['full','old','test140','test90','test50'];metrics={}
for tag,d in dates.items():
 metrics[tag]={}
 for scope in scopes:
  date={'D01':'20261030','D02':'20261031'}.get(tag,d) if scope.startswith('test') else d
  src=B/f'{tag.lower()}{scope}_{date}';metrics[tag][scope]=J(src/'metrics.json')
  ref=B/f'd09{scope}_20261103'
  assert metrics[tag][scope]['config']==J(ref/'metrics.json')['config']
  assert {x['file']:x for x in J(src/'input_manifest.json')}=={x['file']:x for x in J(ref/'input_manifest.json')}
  cp(src/'metrics.json',O/'evaluation'/tag/scope/'metrics.json')
  if tag=='D02':
   for f in ['input_manifest.json','checkpoint.json','per_class_metrics.csv']:cp(src/f,O/'evaluation'/tag/scope/f)
src=B/runs['D02']
cp(src/'weights/best.pt',O/'best.pt');assert H(O/'best.pt')==J(B/'d02full_20261029/checkpoint.json')['weight_sha256']
for f in ['args.yaml','results.csv','server_train.log','results.png','confusion_matrix_normalized.png','download_manifest_20261029.json','data_verification.json']:
 cp(src/f,O/'training'/f)
cp(src/'weights/last.pt',O/'training/last.pt')
shutil.copytree(src/'analysis_20261029/cases',O/'failure_cases/validation',dirs_exist_ok=True)
latest=B/runs['D09']/'analysis_20261103'
shutil.copytree(latest/'test_cases',O/'failure_cases/synthetic',dirs_exist_ok=True)
cp(latest/'test_comparison.json',O/'failure_cases/test_comparison.json')
cp(B/'d02full_20261029/diagnostics.json',O/'evaluation/D02/diagnostics.json')
cfg=dict(model='mower_obstrast_check/docs/训练与评估/final_D02_20261103/best.pt',imgsz=640,conf=.25,iou=.7,max_det=300,device='cpu',half=False,rect=True,batch=1,augment=False,agnostic_nms=False,save=True,save_txt=True,save_conf=True,project='mower_obstrast_check/docs/训练与评估/final_D02_20261103/inference',name='predict',exist_ok=False)
(O/'predict.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
torch.set_num_threads(6)
models={tag:YOLO(str(B/n/'weights/best.pt')).model.float().eval().fuse(verbose=False) for tag,n in runs.items()}
times={t:[] for t in runs};tensor=torch.zeros(1,3,640,640)
with torch.inference_mode():
 for rd in range(3):
  for tag in list(runs)[::(-1 if rd%2 else 1)]:
   for _ in range(3):models[tag](tensor)
   for _ in range(10):
    t=time.perf_counter();models[tag](tensor);times[tag].append((time.perf_counter()-t)*1000)
bench=dict(method='Same process CPU FP32, 6 threads, batch1, 640 square zero tensor; fused forward only. Three alternating rounds, each 3 warmups and 10 measured forwards. Excludes decode/preprocess/NMS/ROS2. Not NPU or end-to-end.',results={t:dict(mean_ms=float(np.mean(v)),p50_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),samples=len(v),raw_ms=v) for t,v in times.items()})
write(O/'cpu_benchmark.json',bench);print('BENCHMARK DONE',flush=True)
model=YOLO(str(O/'best.pt'));checks=[];(O/'smoke').mkdir(exist_ok=True)
for i in range(8):
 path=R/f'mower_obstrast_check/data/test_sets/synthetic_8cls_140_20261030/images/test/test90_c{i}_001.png'
 kw={k:v for k,v in cfg.items() if k not in ['model','project','name','exist_ok']};kw.update(save=False,save_txt=False,save_conf=False,verbose=False)
 result=model.predict(source=str(path),**kw)[0];boxes=result.boxes.data.cpu().numpy()
 assert np.isfinite(boxes).all() and all(0<=int(r[5])<8 and 0<=r[4]<=1 and r[2]>=r[0] and r[3]>=r[1] for r in boxes)
 result.save(filename=str(O/'smoke'/path.name));cp(path,O/'smoke/originals'/path.name)
 checks.append(dict(source=str(path.relative_to(R)),image_sha256=H(path),shape=list(result.orig_shape),detections=[dict(class_id=int(r[5]),class_name=result.names[int(r[5])],confidence=float(r[4]),xyxy=r[:4].tolist()) for r in boxes]))
assert sum(len(x['detections']) for x in checks)>0
write(O/'smoke/results.json',checks)
env=dict(python=sys.version,torch=torch.__version__,ultralytics=__version__,platform=platform.platform(),processor=platform.processor(),threads=torch.get_num_threads(),repository_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),weight_sha256=H(O/'best.pt'),names=model.names,note='Repository head records packaging environment, not asserted to be original training commit. Training runtime configuration is in original logs and checkpoint.')
write(O/'environment.json',env)
names=['人员','猫','狗','飞盘','运动球','玩具熊','软管','石块'];diag=J(O/'evaluation/D02/diagnostics.json');tc=J(O/'failure_cases/test_comparison.json')['groups']['D02']['140']
selection=(B/'训练日志.md').read_text(encoding='utf-8').split('## 最终模型选型：2026-11-03停止继续训练，推荐D02',1)[1]
# Rebase original log links for the independent delivery folder.
import re
selection=re.sub(r'(!?\[[^\]]*\]\()([^\)]+)(\))',lambda x:x[1]+'../'+x[2]+x[3],selection)
lines=['# 障碍物检测第一版模型评估与选型交付报告','','交付日期：2026-11-03。最终推荐D02第194轮best.pt，用户已确认停止继续训练。本任务的模型训练与评估交付已完成；板端部署、量化及实车验收另行开展。','','## 任务目标与完成情况','','完成已有基线及候选的统一复评，选定第一版权重和初始推理参数，归档配置、日志、指标、失败样本及使用说明。当前交付是项目内可复现的PT模型包，加载依赖本项目的通道注意力实现。','','## 验收对应关系','',table(['验收项','交付证据与结果'],[
['模型加载与结构化输出','本次从交付目录best.pt重新加载，8张图片推理完成，输出类别ID、名称、置信度及xyxy，见smoke/results.json与标注图；不要求每图都检出或检出即正确。'],
['每类与重点目标指标','五套范围8类P/R/AP与样本数，固定阈值TP/FP/FN和尺寸分档，见evaluation与本报告。'],
['权重和配置可追溯','best.pt、predict.yaml、training/args.yaml、CSV、日志、哈希清单与environment.json。'],
['优化无收益时保留基线','按同口径结果选择D02而非最新实验；D01基线完整AP59.968%、旧集59.587%、合成81.243%，D02分别60.467%、59.838%、85.109%。提升幅度和局限如实保留。'],
['问题与未完成项明确','石块/小球漏检、软管误检、小人员召回取舍；D03缺少完整离线评估，不用于最终排名；未开展NPU与实车验证。'],
['交付范围','模型训练、离线评估与选型。CPU耗时不能代表板端帧率、ROS2延迟或安全停车能力。']]),'',
'## 统一复评与最终选型',selection,'','## D02各类别最终结果','']
for scope in scopes:
 m=metrics['D02'][scope];lines += [f'### {scope}','',table(['类别','含该类图片','GT框','P%','R%','AP50%','AP50-95%'],[[names[i],int(x['Images']),int(x['Instances']),P(x['Box-P']),P(x['Box-R']),P(x['mAP50']),P(x['mAP50-95'])] for i,x in enumerate(m['per_class'])]),'']
lines+=['## 重点目标召回与失败样本','','固定conf0.25、同类IoU≥0.5一对一匹配；AP表P/R为F1报告点，两者不可混用。']
for scope,arr in [('完整验证',[[x[k] for k in ['tp','fp','fn']] for x in diag['thresholds'] if x['conf']==.25]),('合成140',tc['totals']['0.25'])]:
 rows=[]
 for title,ids in [('人员',[0]),('猫狗',[1,2]),('低矮类别：飞盘/球/熊/软管/石块',[3,4,5,6,7])]:
  a=np.array(arr)[ids].sum(0);rows.append([title,*a,P(a[0]/(a[0]+a[2]))])
 lines += ['',scope,'',table(['分组','TP','FP','FN','召回%'],rows)]
lines+=['','低矮类别分组不等同尺寸小目标；小目标按图像长边缩到640后GT面积<32²定义，详见选型小目标表。','']
cases=J(O/'failure_cases/validation/index.json')
for i in [0,4,6,7]:
 c=next(x for x in cases if x['cls']==i);lines += [f'### {names[i]}典型验证失败样本',f'原图`{c["file"]}`；D02结果{c["D02"]}。拼图为GT、D01、D02，conf0.25。',f'![失败对照](failure_cases/validation/{c["image"]})','']
lines+=['合成混合场景对照见failure_cases/synthetic（GT、D02、D05、D09），对应原图名和计数在failure_cases/test_comparison.json。完整逐图预测仍保存在原runs评估缓存；本交付包保存指标、输入哈希清单、代表图和日志，不复制整个数据集。','','## 推理耗时','','本次同一进程、CPU FP32、6线程、batch1、640×640零张量，模型融合后交替测3轮，每轮预热3次、计时10次；只测模型前向，不含读图、预处理、NMS。各30次，原始时序见cpu_benchmark.json。此项用于结构成本参考，不能替代板端测试。','',table(['候选','平均ms','P50 ms','P95 ms'],[[t,*[f"{bench['results'][t][k]:.3f}" for k in ['mean_ms','p50_ms','p95_ms']]] for t in runs]),'','## 后续工作与已知限制','','1. 固定D02，下一阶段进行ONNX/NPU转换与同图一致性、量化精度、板端耗时和稳定性测试。\n2. 使用真实双目相机独立场景检查远处人员、小球与石块，以及软管/飞盘混淆，再确定业务阈值；合成140已用于多轮诊断，不代表真实泛化。\n3. D09小目标召回较强的结果作为保留对照，当前不再训练。\n4. 当前初始conf0.25不是业务召回验收线；本任务不承诺实车安全停车或达到未定义业务指标。','','## 交付文件','','[推荐权重](best.pt) · [推理配置](predict.yaml) · [使用说明](README.md) · [原始训练配置](training/args.yaml) · [训练曲线](training/results.png) · [本次推理输出](smoke/results.json) · [环境](environment.json) · [文件哈希](SHA256SUMS.json)','']
(O/'模型评估与选型交付报告.md').write_text('\n\n'.join(lines),encoding='utf-8')
readme='''# 障碍物检测第一版 D02 使用说明

推荐权重为best.pt，第194轮，8类，YOLOv8n三尺度检测头加P3通道注意力。输入640，初始conf0.25、NMS IoU0.7、max_det300。详细结果见[交付报告](模型评估与选型交付报告.md)。

## 运行

在本项目根目录，使用能导入本地ultralytics的Python环境运行。Windows已验证环境为D:/itsoft/conda/envs/grass_seg/python.exe。模型包含C2fChannelAttention，不能假设原版PyPI包能直接加载；保留当前项目代码。环境及仓库版本见environment.json，模型内原始train_args见evaluation/D02/full/checkpoint.json。CPU FP32为本次已验证配置。

```python
from pathlib import Path
import yaml
from ultralytics import YOLO

cfg = yaml.safe_load(Path("mower_obstrast_check/docs/训练与评估/final_D02_20261103/predict.yaml").read_text())
model = YOLO(cfg.pop("model"))
results = model.predict(source="你的图片或图片目录", **cfg)
for result in results:
    for box in result.boxes:
        print(int(box.cls.item()), float(box.conf.item()), box.xyxy[0].tolist())
```

请替换source。predict.yaml中的路径相对项目根目录；移动交付目录后相应修改model/project。CUDA环境可把device改为实际GPU编号，但本次CPU计时不代表该设备性能。输出标注图片及YOLO文本保存到inference/predict或自动递增目录，文本附带置信度；Python接口的xyxy为原图像素坐标。

## 类别

0 person 人员；1 cat 猫；2 dog 狗；3 frisbee 飞盘；4 sports ball 运动球；5 teddy bear 玩具熊；6 garden hose 软管；7 rock 石块。

## 交付范围

training保留best来源的配置、last、CSV与完整日志；evaluation保留8个候选五套范围的指标及D02输入清单；failure_cases为失败对照；smoke保存本次8图推理输出。cpu_benchmark.json只代表CPU前向成本。SHA256SUMS.json覆盖交付文件，不包含它自身。

conf0.25是固定离线口径下的初始使用值，尚未针对真实双目场景优化。石块、小球及远处人员漏检和软管误检仍存在。板端转换、量化、ROS2以及实车验收属于后续任务。
'''
(O/'README.md').write_text(readme,encoding='utf-8')
cp(Path(__file__),O/'delivery_script.py')
write(O/'SHA256SUMS.json',{p.relative_to(O).as_posix():H(p) for p in sorted(O.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json'})
print('DELIVERY',O,'SMOKE boxes',sum(len(x['detections']) for x in checks),flush=True)
