"""Summarize measured precision candidates without replacing the selected model."""
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from docx import Document
from docx.shared import Inches

OUT = Path(__file__).resolve().parents[1]
ROOT = next(p for p in OUT.parents if (p / 'ultralytics').is_dir())
DEST = OUT / 'optimization_20261103'
def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

variants = ['uint8_hybrid', 'head_tail_fp16', 'attention_tail_fp16', 'fp16']
metrics = {'ONNX': read(OUT / 'evaluation/ONNX_metrics.json')}
metrics.update({v: read(OUT / 'variants' / v / 'NPU_metrics.json') for v in variants})
lines = ['# D02 后处理与混合量化精度专项复核', '', '日期：2026-11-03。原交付报告的补充记录；未替换根目录正式模型，未修改板端默认 NMS。', '',
'## 结论', '',
'注意力内部与检测头尾部采用 FP16 的候选版提高了整体精度，但石头漏检仍未恢复到 ONNX 水平，暂不作为满足精度验收的正式推荐版本。全 FP16 仍只作为静态精度参照，速度不满足实时要求。', '',
'## 统一口径', '',
'固定合成测试集 140 张、231 个标注框；640×640 方形 letterbox，填充值 114。AP 使用 conf=0.001、分类 NMS IoU=0.7、multi_label=true；部署统计使用 conf=0.25、IoU=0.7、multi_label=false、max_det=300，同类别 IoU≥0.5 匹配 GT。两种统计口径不可混淆。', '',
'128 张校准图全部来自训练集，沿用原校准清单，未使用验证集或测试集校准。该测试集已参与部署优化分析，不能再称为未接触的最终盲测集。', '',
'## 板端实测', '', '| 版本 | AP50 (%) | AP50-95 (%) | 推理 P50 / P95 (ms) |', '|---|---:|---:|---:|']
for tag, m in metrics.items():
    timing = '—'
    if tag != 'ONNX':
        board = OUT / 'board' if tag == 'uint8_hybrid' else OUT / 'board' / tag
        rows = list(csv.DictReader((board / 'outputs/timing.csv').open()))[5:]
        values = [float(r['inference_ms']) for r in rows]
        timing = f'{np.percentile(values,50):.3f} / {np.percentile(values,95):.3f}'
    o = m['overall']
    lines.append(f"| {tag} | {100*o['metrics/mAP50(B)']:.3f} | {100*o['metrics/mAP50-95(B)']:.3f} | {timing} |")
lines += ['', '耗时剔除前 5 张预热，只计板端推理，不含图像读取、主机缩放、传输与保存，不等于端到端帧率。新候选均完成板端 140 张推理，退出码为 0。', '',
'uint8_hybrid 是 UINT8 与 FP16 混合，不能表述为全 INT8。head_tail_fp16 保护最终预测输出、DFL、Sigmoid 与解码等尾部；attention_tail_fp16 进一步保护注意力内部计算，在模块输出保留量化边界。主干特征仍为整数计算。', '',
'## 分类别 AP50-95 (%)', '', '| 类别 | ONNX | 原混合 | 尾部 FP16 | 注意力内部+尾部 FP16 | 全 FP16 |', '|---|---:|---:|---:|---:|---:|']
for i, c in enumerate(metrics['ONNX']['classes']):
    vals = [100*metrics[t]['classes'][i]['mAP50-95'] for t in ['ONNX']+variants]
    lines.append('| '+c['Class']+' | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
lines += ['', '## 部署阈值下 TP / FP / FN', '', '| 类别 | ONNX | 原混合 | 尾部 FP16 | 注意力内部+尾部 FP16 | 全 FP16 |', '|---|---|---|---|---|---|']
for i, c in enumerate(metrics['ONNX']['classes']):
    lines.append('| '+c['Class']+' | '+' | '.join('/'.join(map(str,metrics[t]['totals_conf025'][i])) for t in ['ONNX']+variants)+' |')
lines += ['', '## 重复框：原因和取舍', '',
'脚部样本 test90_c0_001.png 在 ONNX 上已经存在两个 person 框，分数约 0.849 和 0.257。框间 IoU=0.4522，小框约 99.79% 被大框包含。原 NMS=0.7 按规则保留两框，不是板端重复解析输出。提高精度也不保证自动消除模型原有重复框。', '',
'![同一图片的 NMS 对比](feet_nms_comparison.jpg)', '',
'| 数据范围 | NMS=0.7 TP/FP/FN | NMS=0.6 TP/FP/FN |', '|---|---|---|']
for split in ['test140','val154','val2938']:
    data = read(DEST / f'nms_sweep_{split}.json')['results']['ONNX']
    vals = ['/'.join(map(str,np.array(data[k]['totals']).sum(axis=0))) for k in ['0.7','0.6']]
    lines.append(f'| {split} | {vals[0]} | {vals[1]} |')
lines += ['', '完整验证集 2938 张：降低到 0.6 减少 268 个 FP，但 TP 净减少 15 个；丢失 15 个人员和 1 个玩具熊，找回 1 个人员。因此不能描述为无损去重。', '',
'0.45 可消除示例脚部重复框，但在 154 张验证子集丢失 2 个人员和 1 个玩具熊；额外包含关系过滤也会丢失真实人员。上述方案均未设为正式默认。完整验证集本次只验证了 0.7/0.6，不把子集结论冒充全量结论。', '',
'## 逐图对比', '', '![脚部及量化漏检样本](precision_comparison.jpg)', '',
'图中框使用相同部署阈值。所有预测及 GT 命中记录保存在 evaluation/ 与 variants/*/NPU_predictions.json，可按图片名追溯。', '',
'## 工程校验与限制', '',
'新混合版的板端 C++ 后处理与主机相同原始输出的 Python 后处理逐图核对，匹配数及坐标差记录于各 variant/npu_per_image.csv。转换失败的算子类型日志保留：部分 FP16 激活不能直接连接 UINT8 权重卷积；通过保护 DFL 权重、保留注意力出口整数边界完成编译。开发机曾因内存压力中断转换，最终成功产物与板端完整结果独立保存，失败结果未纳入指标。', '',
'## 下一步建议', '',
'1. 不继续盲目降低全局 NMS 阈值；保留 0.7 作为当前追溯基准。若业务决定接受召回损失再减少误报，应单独验收人员遮挡、交叠场景。',
'2. 当前最佳快速候选 attention_tail_fp16 相对 ONNX 仍下降约 1.29 个 AP50-95 百分点，未达到暂定下降≤1个百分点的目标；石头 TP 仍为 4/27，ONNX 为 7/27，不据此声称漏检问题解决。',
'3. 下一项优先实验是量化校准算法与敏感特征层分析：在训练校准数据上比较当前 normal 与 KL 校准，并逐层比较浮点/量化特征误差，优先保护造成小目标信息损失的层。该实验尚未完成，不能提前承诺收益。',
'4. 若上述仍不能达到精度目标，再扩大浮点保护范围，并实测速度；全 FP16 约 1.4 秒/张已证明存在明显性能代价。此次不重新训练 D02，不改变模型类别或数据划分。', '',
'## 产物', '', '候选模型、转换配置和日志：../variants/head_tail_fp16/ 与 ../variants/attention_tail_fp16/；原始板端输出：../board/ 对应目录；NMS 统计：本目录 nms_sweep_*.json。模型哈希统一收录 ../SHA256SUMS.json。']
manifest = read(OUT/'board/input_manifest.json')
base = cv2.imread(str(ROOT/manifest[0]['image']))
sweep = read(DEST/'nms_sweep_test140.json')['results']['ONNX']
def panel(image, boxes, title):
    canvas = cv2.resize(image, (400,300))
    sx,sy = 400/image.shape[1],300/image.shape[0]
    for index,(x1,y1,x2,y2,score,cls) in enumerate(boxes):
        a,b = (int(x1*sx),int(y1*sy)),(int(x2*sx),int(y2*sy))
        cv2.rectangle(canvas,a,b,(0,220,255),2)
        origin=(a[0],min(290,max(15,a[1]+15)+16*index))
        label=f'{int(cls)} {score:.2f}'
        cv2.putText(canvas,label,origin,0,.45,(0,0,0),3)
        cv2.putText(canvas,label,origin,0,.45,(255,255,255),1)
    canvas = cv2.copyMakeBorder(canvas,30,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
    cv2.putText(canvas,title,(8,21),0,.55,(0,0,0),1)
    return canvas
cv2.imwrite(str(DEST/'feet_nms_comparison.jpg'),np.hstack([panel(base,sweep[k]['cases'][0]['boxes'],'ONNX NMS '+k) for k in ['0.7','0.6','0.45']]))
pred = {'ONNX':read(OUT/'evaluation/ONNX_predictions.json')}
pred.update({v:read(OUT/'variants'/v/'NPU_predictions.json') for v in variants})
lost = next((i for i in range(140) if any(a and not b for a,b in zip(pred['ONNX'][i]['gt_hit'],pred['attention_tail_fp16'][i]['gt_hit']))),0)
panels=[]
for i in dict.fromkeys([0,lost]):
    im=cv2.imread(str(ROOT/manifest[i]['image']))
    panels.append(np.hstack([panel(im,pred[t][i]['boxes'],manifest[i]['id']+' '+t) for t in ['ONNX','uint8_hybrid','attention_tail_fp16','fp16']]))
cv2.imwrite(str(DEST/'precision_comparison.jpg'),np.vstack(panels))
diagnosis=DEST/'rock_raw_diagnosis.json'
if diagnosis.exists():
    lines += ['', '## 追加：石头漏检原始输出定位', '',
        '针对 ONNX 检出、attention_tail_fp16 漏检的三个 GT，在 NMS 前选取与该 GT 的 IoU≥0.5 的候选中石头分数最高者。此处比较各端最佳匹配候选，不假设候选索引完全相同。', '',
        '| 图片 | ONNX 石头分数 | 原混合 | 注意力内部+尾部 FP16 | 全 FP16 | 新混合候选 IoU / 最高分类 |',
        '|---|---:|---:|---:|---:|---|']
    for record in read(diagnosis):
        b=record['backends'];current=b['attention_tail_fp16']
        lines.append('| '+record['file']+' | '+' | '.join(f"{b[t]['rock_score']:.6f}" for t in ['ONNX','uint8_hybrid','attention_tail_fp16','fp16'])+f" | {current['iou']:.4f} / {current['top_class']} |")
    lines += ['', '三个目标均有位置准确的候选框（IoU>0.93），但石头分数低于 0.25，故主要是分类分数退化，发生在 NMS 之前。test90_c7_005.png 的上述候选最高类别变为飞碟（3）。降低 NMS IoU 无法恢复这些已被置信度筛选排除的候选。具体内部敏感层尚未定位，不能仅凭输出将责任归于某一层。']
kl_path=OUT/'variants/attention_tail_kl/NPU_metrics.json'
if kl_path.exists():
    kl=read(kl_path)
    rows=list(csv.DictReader((OUT/'board/attention_tail_kl/outputs/timing.csv').open()))[5:]
    values=[float(r['inference_ms']) for r in rows]
    lines += ['', '## 追加：KL 校准对照实测', '',
        '使用相同 128 张训练校准图片、相同 FP16 保护策略和板端后处理，仅将校准算法改为 kl_divergence；新模型位于 ../variants/attention_tail_kl/，没有替换正式模型。', '',
        f"AP50={100*kl['overall']['metrics/mAP50(B)']:.3f}%，AP50-95={100*kl['overall']['metrics/mAP50-95(B)']:.3f}%；推理 P50={np.percentile(values,50):.3f}ms，P95={np.percentile(values,95):.3f}ms。", '',
        '| 类别 | normal AP50-95 (%) | KL AP50-95 (%) | normal TP/FP/FN | KL TP/FP/FN |', '|---|---:|---:|---|---|']
    normal=metrics['attention_tail_fp16']
    for i,c in enumerate(kl['classes']):
        lines.append(f"| {c['Class']} | {100*normal['classes'][i]['mAP50-95']:.2f} | {100*c['mAP50-95']:.2f} | "+'/'.join(map(str,normal['totals_conf025'][i]))+' | '+'/'.join(map(str,kl['totals_conf025'][i]))+' |')
path=DEST/'后处理与混合量化精度复核.md'
if kl_path.exists():
    lines[lines.index('## 结论')+2]='本轮新增 KL 校准与 P5 分类分支 FP16 两项实测。KL 是当前更好的快速部署候选：AP50-95=83.42%，推理中位耗时约45.5ms，石头恢复至6/27；仍有新增漏检，尚未替换正式归档模型。P5 分支保护未恢复石头，且新增人员漏检，不推荐。前文保留前轮对照记录，最新结论见本节和追加实测。'
    lines=[s.replace('## 下一步建议','## 前轮建议及本轮落实').replace('当前最佳快速候选 attention_tail_fp16','前轮快速候选 attention_tail_fp16').replace('该实验尚未完成，不能提前承诺收益。','本轮已完成 KL 对照与 P5 分支保护实验，结果见追加章节；完整逐层敏感度扫描仍未完成。') for s in lines]
    p5=read(OUT/'variants/p5_class_fp16/NPU_metrics.json')
    rows=list(csv.DictReader((OUT/'board/p5_class_fp16/outputs/timing.csv').open()))[5:]
    values=[float(r['inference_ms']) for r in rows]
    lines += ['', '## 追加：P5 分类分支 FP16 对照', '',
        '在 normal 校准及 attention_tail_fp16 基础上，保护 P5 分类分支内部激活与后续卷积权重；第一层卷积仍保留整数输入/整数权重边界。该实验并非全网络浮点，不能据此排除更上游特征层的量化误差。', '',
        f"AP50-95={100*p5['overall']['metrics/mAP50-95(B)']:.3f}%，推理 P50={np.percentile(values,50):.3f}ms、P95={np.percentile(values,95):.3f}ms。", '',
        '| 类别 | P5 分支 FP16 AP50-95 (%) | TP/FP/FN |', '|---|---:|---|']
    for i,c in enumerate(p5['classes']):
        lines.append(f"| {c['Class']} | {100*c['mAP50-95']:.2f} | "+'/'.join(map(str,p5['totals_conf025'][i]))+' |')
    lines += ['', '该候选石头仍只检出4/27，且相比 ONNX 新增人员漏检。不推荐以它替换 KL 或原始模型。', '',
        '## 本轮逐图恢复情况', '',
        '| 图片 | ONNX 石头分数 | normal 混合 | KL 混合 | P5 分类保护 |', '|---|---:|---:|---:|---:|']
    for r in read(diagnosis):
        b=r['backends']
        lines.append('| '+r['file']+' | '+' | '.join(f"{b[t]['rock_score']:.6f}" for t in ['ONNX','attention_tail_fp16','attention_tail_kl','p5_class_fp16'])+' |')
    lines += ['', '![三个石头及脚部对比](kl_p5_comparison.jpg)', '',
        'KL 找回 test90_c7_003.png 和 testmix_044.png 中的两个石头。test90_c7_005.png 最佳匹配候选的石头分数仍仅约0.082；最终输出包含飞碟误检，未恢复正确识别。', '',
        '脚部样本 KL 输出2个人员框，normal 注意力混合版为5个、ONNX为2个。量化带来的额外框减少，但原模型的重复框仍未解决；本轮没有降低 NMS 或置信度阈值。', '',
        '相对 ONNX，KL 逐 GT 核对共丢失3个（猫、石头、球各1个），找回5个（狗1个、球4个）。软管与 ONNX 的 GT 命中相同，但比上一版 normal 混合少检1个。详见 candidate_gt_changes_vs_onnx.json，不能以总检出数增加代替逐图一致性。', '',
        '## 当前建议与未完成项', '',
        '优先保留 attention_tail_kl 作为快速部署候选：相对原混合量化 AP50-95 提高约1.70个百分点，相对 ONNX 下降约0.65个百分点。140张静态结果满足暂定整体 AP 下降≤1个百分点，但分类别并非全部恢复，且耗时仅为推理时间，不能代替端到端实时验收。根目录正式归档仍保持全 FP16 静态参考。', '',
        '下一步应先在固定真实验证样本和板端连续图像上复核 KL 候选，重点检查新增猫、石头漏检；如继续优化校准集，应从已有训练数据选取更多独立场景，排查同源 copy/增强图片的相关性，再做受控比较，不使用测试图校准。若石头分类仍明显退化，再定位更上游特征层，而不是盲目扩大整段 FP16。', '',
        '128张校准原图、标签和预处理图片的哈希复核全部通过（calibration_integrity_check.json）。KL与normal最终均有42个FP16参数条目且集合相同（kl_policy_check.json），支持算法对照的可比性。初次KL混合阶段因发现输出保护需要显式恢复而主动中止；修正后完整重跑并成功上板，非板端推理失败。']
    extra={t:read(OUT/'variants'/t/'NPU_predictions.json') for t in ['attention_tail_kl','p5_class_fp16']}
    compare={**pred,**extra}
    panels=[]
    for i in [0,82,84,133]:
        im=cv2.imread(str(ROOT/manifest[i]['image']))
        panels.append(np.hstack([panel(im,compare[t][i]['boxes'],manifest[i]['id']+' '+t) for t in ['ONNX','attention_tail_kl','p5_class_fp16','fp16']]))
    cv2.imwrite(str(DEST/'kl_p5_comparison.jpg'),np.vstack(panels))
path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
doc=Document()
table=None
for line in lines:
    if line.startswith('|'):
        cells=[v.strip() for v in line.strip('|').split('|')]
        if all(set(v)<=set('-: ') for v in cells):
            continue
        if table is None:
            table=doc.add_table(rows=0,cols=len(cells))
            table.style='Table Grid'
        for cell,value in zip(table.add_row().cells,cells):
            cell.text=value
        continue
    table=None
    if line.startswith('!['):
        doc.add_picture(str(DEST/line.split('](')[1][:-1]),width=Inches(6.3))
    elif line.startswith('#'):
        doc.add_heading(line.lstrip('# '),level=min(3,len(line)-len(line.lstrip('#'))))
    elif line:
        doc.add_paragraph(line)
doc.save(DEST/'后处理与混合量化精度复核.docx')
hashes={}
for p in sorted(OUT.rglob('*')):
    if not p.is_file() or p.name=='SHA256SUMS.json' or '__pycache__' in p.parts:
        continue
    digest=hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    hashes[p.relative_to(OUT).as_posix()]=digest.hexdigest()
(OUT/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
print(path)
