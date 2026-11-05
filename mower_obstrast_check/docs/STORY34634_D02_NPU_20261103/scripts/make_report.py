"""Publish only the selected D02 KL model; COCO80 experiments are historical."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from docx import Document
from docx.shared import Inches

OUT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
main=OUT/'delivery_D02_KL';selection=read(OUT/'selected_model.json')
assert selection['selected']=='attention_tail_kl' and selection['backup'] is None
m=read(main/'NPU_metrics.json');ref=read(OUT/'evaluation/ONNX_metrics.json')
rows=list(csv.DictReader((OUT/'board/attention_tail_kl/outputs/timing.csv').open()))[5:]
t=np.percentile([float(r['inference_ms']) for r in rows],[50,95])
lines=['# STORY #34634：D02 KL 八类模型交付报告','',
'日期：2026-11-03。当前唯一交付模型为 D02 KL 混合量化版。用户已取消80类备选，相关方案仅作历史记录，不再作为备选或推荐。','',
'## 1. 交付入口','',
'[正式交付目录](delivery_D02_KL/README.md)：best.pt、best.onnx、network_binary.nb、推理配置、板端程序、指标及140张对比图。根目录network_binary.nb与该目录模型相同。','',
'模型为D02 YOLOv8n＋P3通道注意力，UINT8＋FP16混合精度、KL校准。类别ID：0人员、1猫、2狗、3飞碟、4运动球、5玩具熊、6软管、7石头。','',
'输入[1,3,640,640]，输出[1,12,8400]；ONNX opset12、batch1、静态尺寸、不含NMS。部署conf严格大于0.25、分类NMS IoU=0.7、max_det=300。','',
f"NPU SHA256：`{selection['nb_sha256']}`。PT/ONNX来源及哈希见selected_model.json和delivery_D02_KL/export_config.json。",'',
'## 2. 评估口径与结果','',
'固定140张合成图片、231个标注框。640×640等比例letterbox，补边114。AP使用conf=0.001、multi_label=true；单点统计使用conf=0.25、multi_label=false。同类IoU≥0.5一对一匹配GT。与之前rect=true训练指标不可直接相减。','',
'| 后端 | 八类AP50 (%) | 八类AP50-95 (%) |','|---|---:|---:|']
for name,data in [('ONNX',ref),('D02 KL NPU',m)]:
    o=data['overall'];lines.append(f"| {name} | {100*o['metrics/mAP50(B)']:.3f} | {100*o['metrics/mAP50-95(B)']:.3f} |")
lines+=['','| 类别 | GT | ONNX AP50-95 (%) | KL AP50-95 (%) | KL P/R@0.25 (%) | TP/FP/FN |','|---|---:|---:|---:|---|---|']
for i,c in enumerate(m['classes']):
    tp,fp,fn=m['totals_conf025'][i]
    lines.append(f"| {c['Class']} | {c['Instances']} | {100*ref['classes'][i]['mAP50-95']:.2f} | {100*c['mAP50-95']:.2f} | {100*tp/max(1,tp+fp):.2f}/{100*tp/max(1,tp+fn):.2f} | {tp}/{fp}/{fn} |")
lines+=['', '同六个共有类别比较：D02 ONNX为92.58%，D02 KL为91.21%；原版YOLOv8n ONNX/FP16约88.82%，原版KL约84.44%。六类平均与八类平均不可混比。该结果支持本轮选择D02，结论限于当前测试集。','',
'## 3. 已知问题','',
'人员47/47无漏检，6个FP均位于真实人体部位，主要是重复框或定位问题。石头检出6/27，ONNX为7/27。相对ONNX逐GT新增漏检3个（猫、石头、球各1），恢复5个（狗1、球4）；整体AP接近不代表逐图完全等价。','',
'![人员重复框及定位问题](optimization_20261103/kl_person_false_positives.jpg)','',
'降低NMS阈值存在人员召回损失，本轮保留0.7。详细过程见[历史精度复核](optimization_20261103/后处理与混合量化精度复核.md)，其中当时选型描述已由当前报告替代。','',
'## 4. 性能与一致性','',
f'板端推理P50/P95为{t[0]:.3f}/{t[1]:.3f}ms，剔除前5张、统计后135张。不包含读图、主机缩放、网络、保存或ROS2通信，不等于端到端FPS。','',
'同一NPU输出的C++和Python后处理未匹配框0、最大坐标差0。交付程序另行复跑140张，280个原始输出和框文件与原KL结果哈希一致，见delivery_D02_KL/runner_regression.json。140张PT/ONNX/NPU并排图位于delivery_D02_KL/visualizations/。','',
'## 5. 实时摄像头推理验证','',
'2026-11-03，在192.168.8.11开发板使用当前交付D02 KL混合量化模型接入USB摄像头（/dev/video0，MJPEG 640×480，配置采集30 FPS），实时完成图像预处理、NPU推理、检测后处理及画面叠框。以下为真实摄像头实时推理留图，不是140张测试集静态推理图。','',
'| 指标 | 实测结果与口径 |','|---|---|',
'| 实时处理帧率 | 约14.69 FPS，按累计处理帧数/运行时间统计，包含采集、预处理、推理、后处理及周期性预览保存 |',
'| NPU平均推理耗时 | 约45.70 ms/帧，依据运行日志；不包含前后处理和网页显示 |',
'| 图1 / 图2单帧NPU耗时 | 46.68 / 45.73 ms |',
'| 网页预览刷新 | 约5 FPS，与模型处理帧率不同 |',
'| 本次运行 | 共处理4324帧，收到停止指令后正常退出，退出码0 |','',
'![实时摄像头图1：近距离手部触发person检测](camera_live/capture_20261103_163959.jpg)','',
'图1：第4167帧，近距离伸手场景输出person框，置信度约0.73。该图证明实时相机链路能够输出结果，也暴露局部人体触发人员框的情况，不能据此认定完整人员检测准确。','',
'![实时摄像头图2：遮挡人员及重复检测框](camera_live/capture_20261103_164010.jpg)','',
'图2：第4320帧，人员被椅背等遮挡，输出两个person框，置信度约0.69和0.60，仍存在重复框/局部框问题。此处如实保留结果，不作为已解决重复检测的证明。','',
'本次仅验证实时摄像头推理可运行及短时处理性能，未标注该现场视频，未计算现场Precision/Recall，也未完成长期稳定性或显示端端到端延迟测量。JPEG与JSON分别发布，保存时检查JSON稳定，但不保证严格逐帧同步。全部6张留图、JSON及日志见[camera_live运行记录](camera_live/摄像头运行记录.md)。摄像头及预览服务已按要求停止。','',
'## 6. 校准、复现与交付边界','',
'128张项目训练校准图，按八类各选16张并覆盖目标面积分位；未使用验证集或测试图校准，原图、标签、预处理图哈希已核对。测试集已参与部署优化，不再视为未接触盲测。','',
'工具链：Acuity6.33.31、VivanteIDE5.10.1、VIP9000NANODI_PLUS_PID0X1000000B。转换配置和日志见variants/attention_tail_kl/；校准清单见calibration/；板端原始结果见board/attention_tail_kl/和board/delivery_D02_KL/。','',
'[运行说明](运行说明.md)。完整性清单：SHA256SUMS.json。当前交付包含模型转换、固定图片静态验证及基础实时USB摄像头推理验证；现场精度评估、ROS2及时间戳传递联调、ByteTrack、端到端延迟、持续稳定性和实车验收仍待后续完成。','',
'80类历史资料已归档至history_coco80/，variants/及board/中的coco80目录也仅为试验记录。historical_coco80_comparison.md/docx保留取消前的对照，不属于当前交付方案。','']
path=OUT/'STORY34634_交付报告.md';path.write_text('\n'.join(lines),encoding='utf-8')
doc=Document();table=None
for line in lines:
    if line.startswith('|'):
        cells=[v.strip() for v in line.strip('|').split('|')]
        if all(set(v)<=set('-: ') for v in cells):continue
        if table is None:table=doc.add_table(rows=0,cols=len(cells));table.style='Table Grid'
        for cell,value in zip(table.add_row().cells,cells):cell.text=value
        continue
    table=None
    if line.startswith('!['):doc.add_picture(str(OUT/line.split('](')[1][:-1]),width=Inches(6.3))
    elif line.startswith('#'):doc.add_heading(line.lstrip('# '),level=min(3,len(line)-len(line.lstrip('#'))))
    elif line:doc.add_paragraph(line)
doc.save(path.with_suffix('.docx'))
(OUT/'delivery_summary.json').write_text(json.dumps(dict(selected=selection,backup=None,inference_ms=dict(p50=t[0],p95=t[1]),history='history_coco80'),indent=2))
hashes={}
for p in sorted(OUT.rglob('*')):
    if not p.is_file() or p.name=='SHA256SUMS.json' or '__pycache__' in p.parts:continue
    digest=hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    hashes[p.relative_to(OUT).as_posix()]=digest.hexdigest()
(OUT/'SHA256SUMS.json').write_text(json.dumps(hashes,indent=2))
print('Sole D02 KL delivery report complete')
