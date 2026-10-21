"""Build B08_v3 in isolation, retaining the verified coordinate FP16 policy."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import argparse

import yaml
import export_mower_onnx

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/'work/b08_v3_20261014'
NAME = 'b08_v3'
SOURCE = Path('/mnt/e/cpp_project/lawn_mowing_robot_yolov8/mower_grass_seg/runs/segment/B08_v3/weights/best.pt')


def run(stage, args):
    print(stage, flush=True)
    with (WORK/(stage+'.log')).open('w') as log:
        subprocess.run([str(ROOT/'tools/conda-acuity310/bin/python'),
                        str(ROOT/'tools/acuity-toolkit-whl-6.33.31/bin/pegasus.py'), *args],
                       cwd=WORK, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    global WORK, NAME, SOURCE
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--work', type=Path, default=WORK)
    parser.add_argument('--name', default=NAME)
    parser.add_argument('--resume-imported', action='store_true')
    args = parser.parse_args()
    WORK, NAME, SOURCE = args.work.resolve(), args.name, args.source.resolve()
    if not NAME.replace('_', '').isalnum():
        parser.error('name must contain only letters, digits and underscores')
    if not args.resume_imported:
        WORK.mkdir(exist_ok=False)
        shutil.copyfile(SOURCE, WORK/(NAME+'.pt'))
        export_mower_onnx.WEIGHT = WORK/(NAME+'.pt')
        export_mower_onnx.ONNX = WORK/(NAME+'.onnx')
        export_mower_onnx.main()
        run('import', ['import', 'onnx', '--model', str(WORK/(NAME+'.onnx')),
            '--inputs', 'images', '--input-size-list', '3, 640, 640', '--outputs', 'output0 output1'])
    assert SOURCE.read_bytes() == (WORK/(NAME+'.pt')).read_bytes()
    old = json.loads((ROOT/'models/mower_yolov8n_seg.json').read_text())
    new = json.loads((WORK/(NAME+'.json')).read_text())
    assert old['Layers'] == new['Layers'], 'Graph topology changed; review precision policy'
    for suffix in ['_inputmeta.yml', '_postprocess_file.yml']:
        shutil.copyfile(ROOT/('models/mower_yolov8n_seg'+suffix), WORK/(NAME+suffix))
    # Quantize fresh weights first; only precision selections are reused below.
    common = ['--model', NAME+'.json', '--model-data', NAME+'.data',
              '--with-input-meta', NAME+'_inputmeta.yml', '--quantizer', 'asymmetric_affine',
              '--qtype', 'uint8', '--iterations', '100']
    run('quantize_initial', ['quantize', *common, '--algorithm', 'normal', '--rebuild'])
    quant_path = WORK/(NAME+'.quantize')
    quant = yaml.safe_load(quant_path.read_text())
    policy = json.loads((ROOT/'work/mower_coord_chain_fp16/experiment.json').read_text())['overrides']
    for node in policy:
        assert node in quant['quantize_parameters'], node
        quant['quantize_parameters'][node] = dict(qtype='float16', quantizer='float16')
    quant_path.write_text(yaml.safe_dump(quant, sort_keys=False))
    run('quantize_hybrid', ['quantize', *common, '--model-quantize', NAME+'.quantize', '--hybrid'])
    run('export', ['export', 'ovxlib', '--model', NAME+'.quantize.json', '--model-data', NAME+'.data',
        '--model-quantize', NAME+'.quantize', '--with-input-meta', NAME+'_inputmeta.yml',
        '--postprocess-file', NAME+'_postprocess_file.yml', '--dtype', 'quantized', '--pack-nbg-unify',
        '--optimize', 'VIP9000NANODI_PLUS_PID0X1000000B', '--viv-sdk',
        str(ROOT/'tools/VivanteIDE5.10.1/VivanteIDE5.10.1/cmdtools'), '--output-path', 'export/'+NAME])
    nb = WORK/'export_nbg_unify/network_binary.nb'
    assert nb.is_file()
    files = [WORK/(NAME+'.pt'), WORK/(NAME+'.onnx'), nb, ROOT/'calib_mower/dataset.txt']
    files += [Path(p) for p in files[-1].read_text().splitlines()]
    report = dict(source=str(SOURCE), overrides=policy,
                  sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (WORK/'provenance.json').write_text(json.dumps(report, indent=2))
    print(nb, flush=True)


if __name__ == '__main__':
    main()
