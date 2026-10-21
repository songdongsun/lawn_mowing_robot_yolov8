"""Prepare fixed NPU inputs and render returned board tensors; no PC inference."""
import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1] / 'data/test'
REFERENCE = Path('//wsl.localhost/Ubuntu-22.04/root/cpp_project/am970/deliverables/STORY25028_B08_v3_2_20261014/09_scripts/compare_mower_backends.py')


def main():
    global ROOT, SOURCE
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'render'])
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT, SOURCE = args.output, args.source
    ROOT.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('reference', REFERENCE)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    if args.mode == 'prepare':
        (ROOT / 'inputs').mkdir(exist_ok=True)
        manifest = []
        for index, path in enumerate(sorted(SOURCE.glob('*.jpg'))):
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(str(path))
            prepared, geometry = reference.prepare(image)
            sid = f'{index:03d}'
            target = ROOT / 'inputs' / (sid + '.png')
            assert cv2.imwrite(str(target), prepared)
            manifest.append(dict(id=sid, source=str(path), geometry=geometry,
                                 source_sha256=reference.sha(path), input_sha256=reference.sha(target)))
        (ROOT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        (ROOT / 'input_list.txt').write_bytes(''.join(item['id'] + '\n' for item in manifest).encode('ascii'))
        (ROOT / 'smoke_list.txt').write_bytes((manifest[0]['id'] + '\n').encode('ascii'))
        print(f'Prepared {len(manifest)} images', flush=True)
        return
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    if (ROOT / 'board_inputs.sha256').exists():
        for line in (ROOT / 'board_inputs.sha256').read_text().splitlines():
            digest, relative = line.split(maxsplit=1)
            if relative == 'network_binary.nb':
                assert digest == '9f162de626472adaac621eb5ffde0fff4dc3e7daf6089ec1810c4141023bf898'
            else:
                assert reference.sha(ROOT / relative) == digest, relative
    for line in (ROOT / 'board_outputs.sha256').read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        assert reference.sha(ROOT / relative) == digest, relative
    for directory in ['masks', 'overlays', 'comparisons']:
        (ROOT / directory).mkdir(exist_ok=True)
    results = []
    for item in manifest:
        sid = item['id']
        image = cv2.imread(item['source'])
        assert reference.sha(item['source']) == item['source_sha256']
        prediction = np.fromfile(ROOT / 'outputs' / f'{sid}_0.f32', dtype='<f4').reshape(37, 8400)
        proto = np.fromfile(ROOT / 'outputs' / f'{sid}_1.f32', dtype='<f4').reshape(32, 160, 160)
        mask, stats = reference.decode(prediction, proto, item['geometry'], outward_crop=True)
        stem = Path(item['source']).stem
        assert cv2.imwrite(str(ROOT / 'masks' / (stem + '.png')), mask.astype(np.uint8) * 255)
        overlay = image.copy()
        overlay[mask] = (overlay[mask] * .5 + np.array([0, 0, 255]) * .5).astype(np.uint8)
        assert cv2.imwrite(str(ROOT / 'overlays' / (stem + '.jpg')), overlay)
        h, w = image.shape[:2]
        size = (960, round(h * 960 / w))
        comparison = np.hstack([cv2.resize(image, size), cv2.resize(overlay, size)])
        assert cv2.imwrite(str(ROOT / 'comparisons' / (stem + '.jpg')), comparison)
        results.append(dict(source=item['source'], **stats))
        print(f'Rendered {sid}: {stem}', flush=True)
    (ROOT / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    (ROOT / '审核说明.txt').write_text(
        f'模型：B08_v3_2；真实 AM970 NPU 静态推理，共 {len(manifest)} 张。\n'
        '模型 SHA256：9f162de626472adaac621eb5ffde0fff4dc3e7daf6089ec1810c4141023bf898\n'
        'comparisons：左侧原图，右侧红色草坪叠加，便于人工审核。\n'
        'overlays：原尺寸叠加图；masks：原尺寸二值 Mask，白色为草坪。\n'
        'inputs：等比例缩放并填充至640的NPU输入；outputs：板端原始输出。\n'
        '预处理和后处理复用交付包，NPU在板端运行，后处理在Windows执行。\n'
        '原始测试图片不修改。无人工真值，结果不能作为精度验收结论。\n'
        '本次不是ROS2实时推理或实时性能验收。\n', encoding='utf-8')


if __name__ == '__main__':
    main()
