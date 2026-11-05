from pathlib import Path
import json,csv,shutil,hashlib
import numpy as np
import matplotlib.pyplot as plt
R=Path.cwd();B=R/'mower_obstrast_check/runs';run=B/'D10_D02_hardaug_ft60';O=run/'analysis_20261104';J=lambda p:json.loads(p.read_text(encoding='utf-8'))
P=lambda v:f'{100*v:.3f}'
def table(h,rows):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
names=['人员','猫','狗','飞盘','运动球','玩具熊','软管','石头'];scopes=['full','old','test140','test90','test50'];titles=['完整验证集2938张','旧固定验证集2801张','合成测试140张','原90张测试','混合50张测试']
folders={tag:{scope:B/f'{tag.lower()}{scope}_{("20261031" if scope.startswith("test") else "20261029") if tag=="D02" else "20261104"}' for scope in scopes} for tag in ['D02','D10']}
m={tag:{s:J(p/'metrics.json') for s,p in folders[tag].items()} for tag in folders}
for scope in scopes:
 assert {r['file']:r for r in J(folders['D02'][scope]/'input_manifest.json')}=={r['file']:r for r in J(folders['D10'][scope]/'input_manifest.json')}
 assert m['D02'][scope]['config']==m['D10'][scope]['config']
g=J(O/'comparison.json')['groups'];t=J(O/'test_comparison.json')['groups'];stats=J(B/'d10full_20261104/dataset_counts.json');train=J(B/'d10full_20261104/training_analysis.json');best=train['best'];last=train['last'];sha=hashlib.sha256((run/'weights/best.pt').read_bytes()).hexdigest()
for name in ['training_trends.png','class_metrics_and_support.png','training_analysis.json','dataset_counts.json','diagnostics.json']:shutil.copy2(B/'d10full_20261104'/name,O/name)
(O/'scripts').mkdir(exist_ok=True)
for name in ['evaluate_detect_d00.py','run_d10_analysis.py','analyze_detect_d10.py','compare_detect_d10.py','compare_d10_tests.py','diagnose_d10_errors.py','d10_confusion_comparison.py','write_d10_report.py']:shutil.copy2(R/'.p2-stage'/name,O/'scripts'/name)
def group(tag,scope):return g[tag+'_'+scope]['all'] if scope in ['full','old'] else t[tag][scope[4:]]
def triple(v):return '/'.join(map(str,v))
def recall(v):return P(v[0]/(v[0]+v[2])) if v[0]+v[2] else 'N/A'
lines=['# D10：D02难例增强微调结果与同口径对比','','日期：2026-11-04。实验：`D10_D02_hardaug_ft60`。','','## 1. 结论','','**本轮正常早停，未形成替换D02的综合收益，继续保留D02为当前交付模型。**',
f'计划60轮、patience15；完成23轮，最佳第8轮。训练CSV最佳AP50-95为{P(best["metrics/mAP50-95(B)"])}%，最后一轮{P(last["metrics/mAP50-95(B)"])}%。日志明确记录连续15轮未改善后EarlyStopping，不是断网、抢卡或人工停止。',
f'固定CPU FP32复评：完整验证AP50-95 {P(m["D02"]["full"]["overall"]["metrics/mAP50-95(B)"])}% → {P(m["D10"]["full"]["overall"]["metrics/mAP50-95(B)"])}%；合成140张 {P(m["D02"]["test140"]["overall"]["metrics/mAP50-95(B)"])}% → {P(m["D10"]["test140"]["overall"]["metrics/mAP50-95(B)"])}%。',
'conf0.25下，140张测试中球命中仍为13/26；球→飞盘重叠误分类框7→6，石头→飞盘仍为1；石头命中8/27→6/27。软管FP16→4，是明确局部改善，但软管TP25→24，同时有召回损失。不能把局部误检下降写成整体验收提升。','','## 2. 权重、配置、时长和数据','','[best.pt](../weights/best.pt) · [last.pt](../weights/last.pt) · [args.yaml](../args.yaml) · [results.csv](../results.csv) · [服务器训练日志](../server_train.log)',f'best SHA256：`{sha}`。已逐文件核验服务器与本地21个训练产物，见[下载校验](../download_manifest_20261104.json)。当前验证集5876个图像/标签文件SHA256一致，见[数据校验](../data_verification.json)。',
'从D02 best.pt初始化，继承P3通道注意力；模型参数全部迁移357/357，不是从原始yolov8n重新训练。resume=false，新优化器和新学习率周期。GPU1、7双卡，总batch32、每卡16，imgsz640，workers4/进程，AMP=true，cache=false，seed42。',
'MuSGD，lr0=0.0005，lrf=0.1，cos_lr=true，momentum0.9，weight_decay0.0005；warmup_epochs3、warmup_bias_lr0。mosaic0、close_mosaic0、scale0.1、translate0.05，fliplr0.5；其余参数以args.yaml为准。',
f'训练CSV累计{last["time"]:.1f}秒，约{last["time"]/3600:.3f}小时（2小时51分42秒）；日志四舍五入为2.862小时。最佳第8轮时累计{best["time"]:.2f}秒（约{best["time"]/60:.1f}分钟）。这些是训练循环与轮间验证时间，不包含本次下载和独立复评，也不是两张卡的GPU小时总和。',
'**分组学习率不能只看lr0。** 当前trainer对MuSGD分类分支cv3使用3倍学习率；第8轮lr/pg0=0.00145517、lr/pg1=0.000485056，第23轮分别为0.00109955、0.000366516。早停时未走到计划60轮的末端学习率。这解释了日志中的不同分组数值，但单次实验不能把退化唯一归因于优化器或学习率。',
'数据：`data_hardaug2x_20261104.yaml`、`train_hardaug2x_20261104.txt`。上一轮已修正4份训练标签、排除27张待复核图片；对22张确认难例各生成一张轻量增强图，使用原图＋增强图。训练名单共66839张文件，对应66817张独立原图来源，214085个标注框。增强图占训练文件约0.033%，选中的22张原图加22张增强图共占约0.066%；并不是整套训练集翻倍。',
'验证集2938张/9099框保持不变；140张合成测试集保持不变，未参与这次微调。人员、猫狗、飞盘、球和玩具熊仍有大量COCO来源，合成测试与真实草坪视角不能互相替代。',
table(['类别','实际训练图片数','实际训练框数','验证图片数','验证框数'],[[names[i],stats['train']['images'][i],stats['train']['boxes'][i],stats['val']['images'][i],stats['val']['boxes'][i]] for i in range(8)]),
'每类图片数会因多类别同图而重复计数，不能将各行图片数相加当作数据集总图数。训练统计按实际train列表，包含hard_aug，未误用物理train目录总量。服务器仓库HEAD记录的是本次核验时状态，不冒充未单独记录的启动时提交。','','## 3. 统一评估口径','','best.pt，CPU FP32，640、batch32、rect=True、conf0.001收集AP曲线，NMS IoU0.7、max_det300。固定操作点另按conf0.25、同类IoU≥0.5的一对一匹配统计TP/FP/FN。P/R指标表来自最佳平均F1工作点，不等于固定conf0.25。',
'五个范围均逐图核验D02与D10的图片和标签哈希，且验证配置一致。主表AP取直接validator输出；缓存预测的重算AP存在小幅浮点差异，未拿它替换主表。训练AMP日志与本地FP32结果分开列出。本报告比较的是PT模型，不将它直接与板端NPU或非rect预处理指标混为一谈。',
table(['范围','D02 AP50%','D10 AP50%','D02 AP50-95%','D10 AP50-95%','差值pp'],[[title,P(m['D02'][s]['overall']['metrics/mAP50(B)']),P(m['D10'][s]['overall']['metrics/mAP50(B)']),P(m['D02'][s]['overall']['metrics/mAP50-95(B)']),P(m['D10'][s]['overall']['metrics/mAP50-95(B)']),f"{100*(m['D10'][s]['overall']['metrics/mAP50-95(B)']-m['D02'][s]['overall']['metrics/mAP50-95(B)']):+.3f}"] for s,title in zip(scopes,titles)]),
'','','## 4. 训练趋势','','![训练曲线](training_trends.png)',
f'第8轮后继续15轮没有刷新最佳。train box {best["train/box_loss"]:.5f}→{last["train/box_loss"]:.5f}、train cls {best["train/cls_loss"]:.5f}→{last["train/cls_loss"]:.5f}；val box {best["val/box_loss"]:.5f}→{last["val/box_loss"]:.5f}、val cls {best["val/cls_loss"]:.5f}→{last["val/cls_loss"]:.5f}。训练损失继续下降，但没有转成整体AP收益。不能只因没有跑满60轮就判断学习不够，也不能凭一次早停证明再训练绝对无收益。']
for s,title in zip(scopes,titles):
 mm=m['D10'][s];a=np.array(group('D02',s)['totals']['0.25']);bb=np.array(group('D10',s)['totals']['0.25'])
 lines+=['',f'## {title}：八类最终结果','',f'D10最佳平均F1阈值约{mm["best_f1_confidence"]:.4f}；下面P/R按此曲线工作点。',table(['类别','图片数','框数','P%','R%','AP50%','D10 AP50-95%','D02 AP50-95%','差值pp'],[[names[i],int(x['Images']),int(x['Instances']),P(x['Box-P']),P(x['Box-R']),P(x['mAP50']),P(x['mAP50-95']),P(m['D02'][s]['per_class'][i]['mAP50-95']),f"{100*(x['mAP50-95']-m['D02'][s]['per_class'][i]['mAP50-95']):+.3f}"] for i,x in enumerate(mm['per_class'])]),'',
 table(['类别','D02 TP/FP/FN','D10 TP/FP/FN','D02 R@0.25%','D10 R@0.25%'],[[names[i],triple(a[i]),triple(bb[i]),recall(a[i]),recall(bb[i])] for i in range(8)])]
 if s in ['full','test140']:
  lines+=['',table(['重点分组','D02 TP/FP/FN','D10 TP/FP/FN','D10召回%'],[[label,triple(a[ix].sum(0)),triple(bb[ix].sum(0)),recall(bb[ix].sum(0))] for label,ix in [('人员',[0]),('猫狗',[1,2]),('低矮类别',[3,4,5,6,7])]]),'',table(['类别','conf0.1 TP/FP/FN','conf0.25','conf0.5'],[[names[i],*[triple(group('D10',s)['totals'][th][i]) for th in ['0.1','0.25','0.5']]] for i in range(8)])]
lines+=['','## 小目标与新增来源','','小目标定义：GT等比缩放到长边640后面积<32²。低矮类别不等同于尺寸小目标；以下均为conf0.25诊断召回，不是COCO APs。',table(['类别','D02旧集小目标命中/GT','D10旧集小目标命中/GT'],[[names[i],f"{g['D02_old']['all']['sizes'][i][0][1]}/{g['D02_old']['all']['sizes'][i][0][0]}",f"{g['D10_old']['all']['sizes'][i][0][1]}/{g['D10_old']['all']['sizes'][i][0][0]}"] for i in range(8)]),'','![固定操作点及小目标](fixed_comparison.png)']
for source,title in [('new_rock','新增岩石42张'),('new_hose','新增AI软管95张')]:
 lines+=['',f'### {title}',table(['类别','D02 TP/FP/FN','D10 TP/FP/FN'],[[names[i],triple(g['D02_full'][source]['totals']['0.25'][i]),triple(g['D10_full'][source]['totals']['0.25'][i])] for i in range(8)])]
lines+=['','## 主要混淆和失败样本','','完整验证集按“未匹配预测与异类GT IoU≥0.5”的诊断口径，两组球→飞盘、石头→飞盘均为0，飞盘→球均为3。因此该验证集上的整体AP对现场/合成图的目标混淆未必敏感，不能据验证集零事件宣称实际问题消失。',table(['140张测试误分类框','D02','D10'],[[k,t['D02']['140']['confusion_boxes'].get(k,0),t['D10']['140']['confusion_boxes'].get(k,0)] for k in sorted(set(t['D02']['140']['confusion_boxes'])|set(t['D10']['140']['confusion_boxes']))]),
'以上是框级诊断计数，不是独立图片数，也不是唯一原因判断。错误分解见[error_decomposition.json](error_decomposition.json)，验证集目标混淆清单见[full_confusion_comparison.json](full_confusion_comparison.json)。',
'![球漏检对照：GT、D02、D05、D10](test_cases/4_miss.jpg)','','![石头漏检对照](test_cases/7_miss.jpg)','','![软管误检对照](test_cases/6_fp.jpg)',
'更多验证集的收益、退化和历史难例见[cases/index.json](cases/index.json)，测试图原文件名与类别、TP/FP/FN见[test_comparison.json](test_comparison.json)。图中D05仅作为历史参考，本轮主要决策对照是D02。']
last_folder=B/'d10lasttest140_20261104'
if (last_folder/'metrics.json').exists():
 lm=J(last_folder/'metrics.json');lines+=['','## last.pt补充核查','','为避免整体AP选出的best遗漏后续难例收益，另对第23轮last执行相同140张测试。',table(['权重','AP50%','AP50-95%'],[[name,P(x['overall']['metrics/mAP50(B)']),P(x['overall']['metrics/mAP50-95(B)'])] for name,x in [('D02 best',m['D02']['test140']),('D10 best第8轮',m['D10']['test140']),('D10 last第23轮',lm)]]),'补充结果只用于诊断；不能在未完整复评last的其余范围时把它宣称为替代交付模型。']
lines+=['','## 下一步建议','','1. 保留D02交付与板端模型，归档D10，不因软管局部收益整体替换。',
'2. 不直接扩大epochs重跑。先将球/石头→飞盘的真实独立失败图做固定专项验证，区分分类错误、无候选和置信度不足，再选择微调检查点；现有普通验证集并未覆盖出明显同类混淆事件。',
'3. 如果继续微调，先设计明确的难例采样比例与普通样本回放比例。当前新增22张增强图只占整库约0.033%，整库训练并不等于足够强调目标问题；但不能据此断言必须大幅过采样，以免破坏其他类别。',
'4. 下一轮还需明确分类分支实际学习率，而不是只写lr0。固定其他条件后再做对照；本轮同时涉及数据清洗、增强、初始化和学习率周期，不能将所有变化归因于某一个因素。',
'5. 球、石头独立测试样本仍较少（分别26、27个GT框），单张改变会明显影响召回。结论限于当前数据，不将合成140张当作实车泛化证明。',
'本次未启动新训练、未修改已交付D02、未转换或部署D10 NPU模型。','']
text='\n'.join(lines);(O/'D10微调结果分析.md').write_text(text,encoding='utf-8')
summary={'training':train,'overall':{tag:{s:m[tag][s]['overall'] for s in scopes} for tag in m},'recommendation':'retain D02; D10 not a comprehensive improvement','best_sha256':sha,'all_five_scope_input_hashes_equal':True}
(O/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
log=B/'训练日志.md';content=log.read_text(encoding='utf-8');heading='## D10 难例增强微调复评（2026-11-04）';assert heading not in content
entry='\n\n'+heading+'\n\n[完整分析报告](D10_D02_hardaug_ft60/analysis_20261104/D10微调结果分析.md)\n\n'+text.replace('# D10：D02难例增强微调结果与同口径对比','### D10详细记录').replace('## ','### ')
# Preserve links when the same report is appended to the root training log.
import re
base='D10_D02_hardaug_ft60/analysis_20261104/'
entry=re.sub(r'\]\(([^)]+)\)',lambda x:']('+ (x.group(1) if x.group(1).startswith(('http','D10_D02_')) else base+x.group(1)) +')',entry)
log.write_text(content+entry,encoding='utf-8');print('REPORT AND TRAINING LOG WRITTEN')
