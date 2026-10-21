"""Isolated single-coordinate-node mixed precision experiment."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--chain', action='store_true')
args = parser.parse_args()

root = Path(__file__).resolve().parents[1]
base = 'mower_coord_chain_fp16' if args.chain else 'mower_coord_fp16'
work = root/'work'/base
work.mkdir(exist_ok=False)
source = root/'models/mower_yolov8n_seg'
for suffix in ['.json', '.data', '_postprocess_file.yml']:
    shutil.copyfile(str(source)+suffix, work/(base+suffix))
meta = yaml.safe_load(Path(str(source)+'_inputmeta.yml').read_text())
for database in meta['input_meta']['databases']:
    for port in database['ports']:
        port['preprocess']['scale'] = [1/255]*3
        port['preprocess']['preproc_node_params']['add_preproc_node'] = True
(work/(base+'_inputmeta.yml')).write_text(yaml.safe_dump(meta, sort_keys=False))
quant = yaml.safe_load(Path(str(source)+'.quantize').read_text())
overrides = ['@output0_3:out0', '@attach_output0/out0_0:out0',
             '@model.22/Mul_2_output_0_6:out0']
if args.chain:
    overrides += ['@model.22/'+node+':out0' for node in [
        'Concat_5_output_0_11', 'Div_1_output_0_18', 'Sub_1_output_0_19',
        'Add_2_output_0_24', 'Add_1_output_0_26', 'Sub_output_0_27']]
for name in overrides:
    assert name in quant['quantize_parameters'], name
    quant['quantize_parameters'][name] = dict(qtype='float16', quantizer='float16')
(work/(base+'.quantize')).write_text(yaml.safe_dump(quant, sort_keys=False))
(work/'experiment.json').write_text(json.dumps(dict(overrides=overrides,
    baseline_sha256=hashlib.sha256((root/'outputs/mower_uint8_hybrid_nbg_unify/network_binary.nb').read_bytes()).hexdigest()), indent=2))
python = root/'tools/conda-acuity310/bin/python'
pegasus = root/'tools/acuity-toolkit-whl-6.33.31/bin/pegasus.py'
def run(stage, args):
    print(stage, flush=True)
    with (work/(stage+'.log')).open('w') as log:
        subprocess.run([str(python), str(pegasus), *args], cwd=work,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
run('quantize', ['quantize', '--model', base+'.json', '--model-data', base+'.data',
    '--model-quantize', base+'.quantize', '--with-input-meta', base+'_inputmeta.yml',
    '--quantizer', 'asymmetric_affine', '--qtype', 'uint8', '--hybrid', '--iterations', '100'])
run('export', ['export', 'ovxlib', '--model', base+'.quantize.json', '--model-data', base+'.data',
    '--model-quantize', base+'.quantize', '--with-input-meta', base+'_inputmeta.yml',
    '--postprocess-file', base+'_postprocess_file.yml', '--dtype', 'quantized', '--pack-nbg-unify',
    '--optimize', 'VIP9000NANODI_PLUS_PID0X1000000B', '--viv-sdk',
    str(root/'tools/VivanteIDE5.10.1/VivanteIDE5.10.1/cmdtools'), '--output-path', 'export/'+base])
assert (work/'export_nbg_unify/network_binary.nb').is_file()
print(work/'export_nbg_unify/network_binary.nb', flush=True)
