"""AM970 Acuity uint8 backbone + FP16 coordinate decoding conversion."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import yaml

OUT=Path(__file__).resolve().parents[1]
TOOLS=Path('/root/cpp_project/am970/model/tools')
WORK=Path('/root/cpp_project/am970/model/work/d02_detect_20261103')
mode=sys.argv[1] if len(sys.argv)>1 else 'baseline'
if mode in ['head_tail_fp16','attention_tail_fp16','attention_tail_kl','p5_class_fp16','coco80_kl','coco80_int8']:
    base=WORK.parent/'d02_coco80_kl_20261103' if mode=='coco80_int8' else WORK
    WORK=base.parent/('d02_'+mode+'_20261103')
    WORK.mkdir(exist_ok=True)
    if mode!='coco80_kl' and sys.argv[2:] not in [['export'],['hybrid']]:
        for name in ['d02.json','d02.data','d02_inputmeta.yml','d02_postprocess_file.yml','d02.quantize']:
            if mode!='coco80_int8' or name!='d02.quantize':
                shutil.copy2(base/name,WORK/name)
PY=TOOLS/'conda-acuity310/bin/python'
ACUITY=TOOLS/'acuity-toolkit-whl-6.33.31/bin/pegasus.py'
def run(stage,args):
    print(stage,flush=True)
    (WORK/(stage+'.command.json')).write_text(json.dumps([str(PY),str(ACUITY),*args],indent=2))
    with (WORK/(stage+'.log')).open('w') as log:
        subprocess.run([str(PY),str(ACUITY),*args],cwd=WORK,stdout=log,stderr=subprocess.STDOUT,check=True)

if mode=='coco80_fp16':
    WORK=WORK.parent/'d02_coco80_kl_20261103'
WORK.mkdir(exist_ok=True)
if mode=='coco80_kl' and sys.argv[2:] not in [['export'],['hybrid']]:
    shutil.copy2(OUT/'history_coco80/yolov8n.onnx',WORK/'d02.onnx')
    run('import',['import','onnx','--model','d02.onnx','--inputs','images','--input-size-list','3,640,640','--outputs','output0'])
    graph=json.loads((WORK/'d02.json').read_text())['Layers']
    meta=yaml.safe_load((base/'d02_inputmeta.yml').read_text())
    meta['input_meta']['databases'][0]['ports'][0]['lid']=next(k for k,v in graph.items() if v['op']=='input')
    (WORK/'d02_inputmeta.yml').write_text(yaml.safe_dump(meta,sort_keys=False))
    post=yaml.safe_load((base/'d02_postprocess_file.yml').read_text())
    post['postprocess']['app_postprocs'][0]['lid']=next(k for k,v in graph.items() if v['op']=='output')
    (WORK/'d02_postprocess_file.yml').write_text(yaml.safe_dump(post,sort_keys=False))
if len(sys.argv)==1:
    shutil.copy2(OUT.parent/'STORY34629_D02_ONNX_20261103/best.onnx',WORK/'d02.onnx')
    run('import',['import','onnx','--model','d02.onnx','--inputs','images','--input-size-list','3,640,640','--outputs','output0'])
layers=json.loads((WORK/'d02.json').read_text())['Layers']
if mode in ['head_tail_fp16','attention_tail_fp16','attention_tail_kl','p5_class_fp16','coco80_kl','coco80_int8']:
    if mode=='coco80_int8' and sys.argv[2:]!=['export']:
        run('quantize_int8',['quantize','--model','d02.json','--model-data','d02.data',
            '--with-input-meta','d02_inputmeta.yml','--quantizer','symmetric_affine','--qtype','int8',
            '--iterations','128','--algorithm','kl_divergence','--rebuild-all'])
    if mode in ['attention_tail_kl','coco80_kl'] and sys.argv[2:] not in [['export'],['hybrid']]:
        run('quantize_kl',['quantize','--model','d02.json','--model-data','d02.data',
            '--with-input-meta','d02_inputmeta.yml','--quantizer','asymmetric_affine','--qtype','uint8',
            '--iterations','128','--algorithm','kl_divergence','--rebuild'])
    q=yaml.safe_load((WORK/'d02.quantize').read_text())
    original={'quantize_parameters':{}} if mode in ['coco80_kl','coco80_int8'] else yaml.safe_load((base/'d02.quantize').read_text())
    for tensor,parameter in original['quantize_parameters'].items():
        if parameter.get('qtype')=='float16':
            q['quantize_parameters'][tensor]=parameter
    overrides={}
    for key,layer in layers.items():
        name=layer['name']
        head=name.startswith('model.22/') and (
            ('/cv2.' not in name and '/cv3.' not in name) or name.endswith('.2/Conv_output_0'))
        head=head or (mode=='coco80_kl' and name in ['output0','attach_output0/out0'])
        if mode=='coco80_int8':head=False
        # Keep the attention product uint8 at the boundary to downstream uint8 convolutions.
        attention=mode in ['attention_tail_fp16','attention_tail_kl','p5_class_fp16'] and '/attention/' in name and layer['op']!='multiply'
        p5=mode=='p5_class_fp16' and 'model.22/cv3.2/' in key
        tensor='@'+key+':out0'
        if (head or attention or p5) and tensor in q['quantize_parameters']:
            q['quantize_parameters'][tensor]=dict(qtype='float16',quantizer='float16')
            overrides[tensor]=layer
            for parameter in ['weight','bias']:
                param='@'+key+':'+parameter
                if ('/dfl/' in name or '/attention/' in name or (p5 and '/cv3.2.0/conv/' not in name)) and param in q['quantize_parameters']:
                    q['quantize_parameters'][param]=dict(qtype='float16',quantizer='float16')
    (WORK/'d02.quantize').write_text(yaml.safe_dump(q,sort_keys=False))
    (WORK/'precision_policy.json').write_text(json.dumps(dict(variant=mode,
        base='Signed INT8 full graph without FP16 overrides' if mode=='coco80_int8' else 'Original uint8 backbone + FP16 coordinate chain',overrides=overrides),indent=2))
    if mode=='coco80_int8':
        counts={}
        for key,value in q['quantize_parameters'].items():
            dtype=value.get('qtype');counts[dtype]=counts.get(dtype,0)+1
        (WORK/'precision_audit.json').write_text(json.dumps(dict(qtypes=counts,overrides=overrides,requested='signed INT8 weights/activations; INT32 bias; no FP16 overrides'),indent=2))
        assert not overrides
    elif sys.argv[2:]!=['export']:
        run('quantize_hybrid',['quantize','--model','d02.json','--model-data','d02.data',
            '--with-input-meta','d02_inputmeta.yml','--quantizer','asymmetric_affine','--qtype','uint8',
            '--iterations','128','--model-quantize','d02.quantize','--hybrid',
            '--algorithm','kl_divergence' if mode in ['attention_tail_kl','coco80_kl'] else 'normal'])
    else:
        json.loads((WORK/'d02.quantize.json').read_text())
    run('export',['export','ovxlib','--model','d02.json' if mode=='coco80_int8' else 'd02.quantize.json','--model-data','d02.data',
        '--model-quantize','d02.quantize','--with-input-meta','d02_inputmeta.yml',
        '--postprocess-file','d02_postprocess_file.yml','--dtype','quantized','--pack-nbg-unify',
        '--optimize','VIP9000NANODI_PLUS_PID0X1000000B','--viv-sdk',
        str(TOOLS/'VivanteIDE5.10.1/VivanteIDE5.10.1/cmdtools'),'--output-path','export/d02'])
    target=OUT/'variants'/mode;target.mkdir(parents=True,exist_ok=True)
    shutil.copy2(WORK/'export_nbg_unify/network_binary.nb',target/'network_binary.nb')
    for p in WORK.iterdir():
        if p.is_file() and p.suffix in ['.json','.yml','.quantize','.log']:
            shutil.copy2(p,target/p.name)
    print(mode+' complete',flush=True)
    sys.exit(0)
if mode in ['fp16','coco80_fp16']:
    run('export_fp16',['export','ovxlib','--model','d02.json','--model-data','d02.data',
        '--with-input-meta','d02_inputmeta.yml','--postprocess-file','d02_postprocess_file.yml',
        '--dtype','float16','--pack-nbg-unify','--optimize','VIP9000NANODI_PLUS_PID0X1000000B',
        '--viv-sdk',str(TOOLS/'VivanteIDE5.10.1/VivanteIDE5.10.1/cmdtools'),'--output-path','fp16_export/d02'])
    target=OUT/'variants'/('coco80_fp16' if mode=='coco80_fp16' else 'fp16');target.mkdir(parents=True,exist_ok=True)
    shutil.copy2(WORK/'fp16_export_nbg_unify/network_binary.nb',target/'network_binary.nb')
    for name in ['export_fp16.log','export_fp16.command.json']:
        shutil.copy2(WORK/name,(target if mode=='coco80_fp16' else OUT/'conversion')/name)
    print('FP16 NBG complete',flush=True)
    sys.exit(0)
calib=WORK/'calibration'
shutil.copytree(OUT/'calibration/images',calib,dirs_exist_ok=True)
(WORK/'dataset.txt').write_text(''.join(str(p)+'\n' for p in sorted(calib.glob('*.png'))))
assert len(list(calib.glob('*.png')))==128
template=Path('/root/cpp_project/am970/model/work/b08_cfg_ca_field_v22_20261028')
meta=yaml.safe_load((template/'b08_cfg_ca_field_v22_inputmeta.yml').read_text())
meta['input_meta']['databases'][0]['path']=str(WORK/'dataset.txt')
inputs=[k for k,v in layers.items() if v['op']=='input'];assert len(inputs)==1
meta['input_meta']['databases'][0]['ports'][0]['lid']=inputs[0]
(WORK/'d02_inputmeta.yml').write_text(yaml.safe_dump(meta,sort_keys=False))
post=yaml.safe_load((template/'b08_cfg_ca_field_v22_postprocess_file.yml').read_text())
post['postprocess']['app_postprocs']=post['postprocess']['app_postprocs'][:1]
assert layers[post['postprocess']['app_postprocs'][0]['lid']]['op']=='output'
(WORK/'d02_postprocess_file.yml').write_text(yaml.safe_dump(post,sort_keys=False))
names=['output0','attach_output0/out0','model.22/Mul_2_output_0','model.22/Concat_2_output_0',
       'model.22/Div_1_output_0','model.22/Sub_1_output_0','model.22/Add_2_output_0',
       'model.22/Add_1_output_0','model.22/Sub_output_0']
policy={}
for name in names:
    nodes=[k for k,v in layers.items() if v['name']==name]
    assert len(nodes)==1,(name,nodes)
    policy['@'+nodes[0]+':out0']=layers[nodes[0]]
(WORK/'precision_policy.json').write_text(json.dumps(dict(reason='Decoded coordinates use pixel-scale ranges; preserve coordinate arithmetic and mixed coordinate/class output in FP16. Selected by D02 graph node names and recorded dependencies, not copied segmentation node IDs.',overrides=policy),indent=2))
common=['--model','d02.json','--model-data','d02.data','--with-input-meta','d02_inputmeta.yml',
        '--quantizer','asymmetric_affine','--qtype','uint8','--iterations','128']
if len(sys.argv)==1:
    run('quantize_initial',['quantize',*common,'--algorithm','normal','--rebuild'])
    q=yaml.safe_load((WORK/'d02.quantize').read_text())
    for node in policy:
        assert node in q['quantize_parameters'],node
        q['quantize_parameters'][node]=dict(qtype='float16',quantizer='float16')
    (WORK/'d02.quantize').write_text(yaml.safe_dump(q,sort_keys=False))
    run('quantize_hybrid',['quantize',*common,'--model-quantize','d02.quantize','--hybrid'])
else:
    assert sys.argv[1:]==['export']
    json.loads((WORK/'d02.quantize.json').read_text())
run('export',['export','ovxlib','--model','d02.quantize.json','--model-data','d02.data',
    '--model-quantize','d02.quantize','--with-input-meta','d02_inputmeta.yml',
    '--postprocess-file','d02_postprocess_file.yml','--dtype','quantized','--pack-nbg-unify',
    '--optimize','VIP9000NANODI_PLUS_PID0X1000000B','--viv-sdk',
    str(TOOLS/'VivanteIDE5.10.1/VivanteIDE5.10.1/cmdtools'),'--output-path','export/d02'])
shutil.copy2(WORK/'export_nbg_unify/network_binary.nb',OUT/'network_binary.nb')
variant=OUT/'variants/uint8_hybrid';variant.mkdir(parents=True,exist_ok=True)
shutil.copy2(OUT/'network_binary.nb',variant/'network_binary.nb')
for p in WORK.iterdir():
    if p.is_file() and p.suffix in ('.json','.yml','.quantize','.log','.txt'):
        shutil.copy2(p,OUT/'conversion'/p.name)
(OUT/'conversion/model_hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [WORK/'d02.onnx',OUT/'network_binary.nb']},indent=2))
print('NBG export complete',flush=True)
