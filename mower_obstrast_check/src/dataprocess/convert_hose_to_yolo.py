"""Convert the two reviewed hose subsets, retaining all available co-occurring labels."""

import argparse
import ast
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def box_line(class_id, xyxy):
    """Clip normalized corners and serialize a five-column detection label."""
    if not all(math.isfinite(v) for v in xyxy):
        raise ValueError(f"Nonfinite box: {xyxy}")
    x1, y1, x2, y2 = [min(1, max(0, v)) for v in xyxy]
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Empty box: {xyxy}")
    return str(class_id) + " " + " ".join(
        f"{v:.10f}" for v in ((x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1)
    )


def convert(root):
    review = root / "按类别人工审核"
    output = root / "整合数据"
    category_dirs = [review / "软管_garden hose", review / "软管_garden_hose"]
    for directory in category_dirs:
        if not directory.is_dir() or (output / directory.name).exists():
            raise ValueError(f"Source missing or destination already exists: {directory.name}")
    for category in category_dirs:
        source = next(p for p in category.iterdir() if p.is_dir())
        original = root / source.name
        destination = output / category.name / source.name
        by_image = defaultdict(list)
        image_info = {}
        if "LVIS" in source.name:
            annotation_file = original / "selected_original_annotations.json"
            data = json.loads(annotation_file.read_text(encoding="utf-8"))
            selected = {int(p.stem) for p in source.rglob("*.jpg")}
            for annotation in data["annotations"]:
                if annotation["image_id"] in selected:
                    by_image[annotation["image_id"]].append(annotation)
            image_info = {i["id"]: i for i in data["images"] if i["id"] in selected}
            used = {a["category_id"] for items in by_image.values() for a in items}
            classes = {c["id"]: c["name"] for c in data["categories"] if c["id"] in used}
        else:
            annotation_file = original / "data.yaml"
            names = ast.literal_eval(next(
                line.split(":", 1)[1] for line in annotation_file.read_text().splitlines()
                if line.startswith("names:")
            ))
            used = {int(line.split()[0]) for p in source.rglob("*.txt")
                    for line in p.read_text().splitlines() if line.strip()}
            classes = {i: names[i] for i in sorted(used)}
        mapping = {old: new for new, old in enumerate(sorted(classes))}
        report = {"source": str(original), "class_mapping": [
            {"original_id": old, "yolo_id": new, "name": classes[old]} for old, new in mapping.items()
        ], "splits": {}}
        yaml = [f"path: {json.dumps(destination.as_posix(), ensure_ascii=False)}"]
        for split in sorted(p for p in source.iterdir() if p.is_dir()):
            counts = Counter()
            images = sorted((split / "images").glob("*.jpg"))
            target = destination / split.name
            (target / "images").mkdir(parents=True)
            (target / "labels").mkdir()
            trace = []
            for image in images:
                lines = []
                if image_info:
                    info = image_info[int(image.stem)]
                    for annotation in by_image[info["id"]]:
                        x, y, w, h = annotation["bbox"]
                        old = annotation["category_id"]
                        lines.append(box_line(mapping[old], (x/info["width"], y/info["height"],
                                                             (x+w)/info["width"], (y+h)/info["height"])))
                        counts[classes[old]] += 1
                        trace.append(annotation)
                else:
                    label = split / "原始标签" / f"{image.stem}.txt"
                    for line in label.read_text().splitlines():
                        fields = line.split()
                        if not fields:
                            continue
                        old = int(fields[0])
                        coords = list(map(float, fields[1:]))
                        if len(coords) == 4:
                            x, y, w, h = coords
                            corners = (x-w/2, y-h/2, x+w/2, y+h/2)
                        elif len(coords) >= 6 and len(coords) % 2 == 0:
                            corners = (min(coords[::2]), min(coords[1::2]), max(coords[::2]), max(coords[1::2]))
                        else:
                            raise ValueError(f"Invalid label: {label}")
                        lines.append(box_line(mapping[old], corners))
                        counts[classes[old]] += 1
                    (target / "原始标签").mkdir(exist_ok=True)
                    shutil.copy2(label, target / "原始标签" / label.name)
                copied = target / "images" / image.name
                shutil.copy2(image, copied)
                assert hashlib.sha256(image.read_bytes()).digest() == hashlib.sha256(copied.read_bytes()).digest()
                (target / "labels" / f"{image.stem}.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
            if image_info:
                (target / "同图原始标注.json").write_text(json.dumps({
                    "categories": [{"id": k, "name": v} for k, v in classes.items()],
                    "images": list(image_info.values()), "annotations": trace,
                }, ensure_ascii=False), encoding="utf-8")
            report["splits"][split.name] = {"images": len(images), "boxes_by_class": dict(counts)}
            key = "train" if split.name.endswith("_train") else "test" if split.name.endswith("_test") else "val"
            yaml.append(f"{key}: {json.dumps(split.name + '/images', ensure_ascii=False)}")
        yaml.append("names:")
        yaml.extend(f"  {mapping[old]}: {json.dumps(name)}" for old, name in classes.items())
        (destination / "data.yaml").write_text("\n".join(yaml)+"\n", encoding="utf-8")
        (destination / "转换记录.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (destination / "使用说明.txt").write_text(
            "仅转换软管图片子集。名称不变，ID在本来源内连续重编，从0开始；映射见转换记录.json。\n"
            "同图全部已有标注均保留，包括规范外类别，尚未完成最终类别筛选。\n"
            "多边形取外接框，原检测框按原坐标转换；输出为五列归一化YOLO检测标签。\n"
            "images为独立副本，labels为转换结果；原始标签/同图原始标注用于追溯。\n"
            "两来源类别配置独立，不可直接与COCO或彼此混用ID。合并训练前需统一映射。\n"
            "LVIS仅有验证集，因此其data.yaml是类别和路径说明，不可直接启动训练。\n"
            "没有补造训练集，没有自动补充原始数据中缺失的标注；仍需人工查漏标。\n",
            encoding="utf-8-sig",
        )
        print(json.dumps(report["splits"], ensure_ascii=True), flush=True)


def convert_selected(root, shovel=False):
    """Export historical six subsets, or the approved shovel subset with active IDs."""
    names = ['person', 'cat', 'dog', 'frisbee', 'sports ball', 'teddy bear',
             'spade', 'watering-cane', 'hoe', 'wheel-barrow', 'pruning-saw', 'garden hose', 'rock']
    categories = ['铁锹_spade', '浇水壶_watering-cane', '锄头_hoe',
                  '手推车_wheel-barrow', '修枝锯_pruning-saw', '软管_garden hose']
    output = root / '整合数据' / '六类_YOLO检测_13类ID_20261016'
    if shovel:
        active_config = Path(__file__).resolve().parents[2] / 'data/yolo_check/data.yaml'
        names = [json.loads(line.split(':', 1)[1]) for line in
                 active_config.read_text(encoding='utf-8').splitlines() if line.startswith('  ')]
        if len(names) != 12 or names[6] != 'spade_hoe':
            raise ValueError('Expected active 12-class configuration with spade_hoe at ID 6')
        categories = ['铲子_shovel']
        output = root / '整合数据' / '铲子_shovel_YOLO检测_12类ID_20261016'
    if output.exists():
        raise FileExistsError(output)
    config = root / '按类别人工审核' / categories[0] / 'Obstructions.v1i.yolov8' / '原始类别配置_仅供查阅.yaml'
    source_names = ast.literal_eval(next(line.split(':', 1)[1] for line in
        config.read_text(encoding='utf-8-sig').splitlines() if line.startswith('names:')))
    mapping = {i: names.index(name.lower() if name == 'Rock' else name)
               for i, name in enumerate(source_names) if name in names or name == 'Rock'}
    if shovel:
        mapping.update({i: 6 for i, name in enumerate(source_names) if name in ('shovel', 'spade', 'hoe')})
    jobs, reports = [], {}
    for category in categories:
        source = root / ('整合数据' if category == '软管_garden hose' else '按类别人工审核') / category / 'Obstructions.v1i.yolov8'
        if not source.is_dir():
            raise FileNotFoundError(source)
        report = {'source': str(source), 'splits': {}, 'excluded_classes': {}, 'polygons': 0, 'boxes': 0}
        excluded = Counter()
        for split_dir in sorted(p for p in source.iterdir() if p.is_dir()):
            split = {'训练集_train': 'train', '验证集_valid': 'val', '测试集_test': 'test'}[split_dir.name]
            images = sorted((split_dir / 'images').glob('*'))
            labels = split_dir / '原始标签'
            image_stems = {p.stem for p in images}
            label_stems = {p.stem for p in labels.glob('*.txt')}
            if not images or image_stems - label_stems or (not shovel and image_stems != label_stems):
                raise ValueError(f'Image/label mismatch: {split_dir}')
            counts = Counter()
            for image in images:
                original = labels / (image.stem + '.txt')
                converted = []
                for row in original.read_text(encoding='utf-8-sig').splitlines():
                    if not row.strip():
                        continue
                    tokens = row.split()
                    old = int(tokens[0])
                    if not 0 <= old < len(source_names):
                        raise ValueError(f'Unknown source ID: {original}: {row}')
                    if old not in mapping:
                        excluded[source_names[old]] += 1
                        continue
                    values = list(map(float, tokens[1:]))
                    if not all(math.isfinite(v) for v in values):
                        raise ValueError(f'Nonfinite coordinates: {original}')
                    if len(values) == 4:
                        x, y, w, h = values
                        corners = (x-w/2, y-h/2, x+w/2, y+h/2)
                        report['boxes'] += 1
                    elif len(values) >= 6 and len(values) % 2 == 0:
                        corners = (min(values[::2]), min(values[1::2]), max(values[::2]), max(values[1::2]))
                        report['polygons'] += 1
                    else:
                        raise ValueError(f'Invalid label: {original}: {row}')
                    line = box_line(mapping[old], corners)
                    _, x, y, w, h = map(float, line.split())
                    actual = (x-w/2, y-h/2, x+w/2, y+h/2)
                    if w <= 0 or h <= 0 or any(abs(a-min(1, max(0, b))) > 1e-9 for a, b in zip(actual, corners)):
                        raise ValueError(f'Box roundtrip failed: {original}')
                    converted.append(line)
                    counts[names[mapping[old]]] += 1
                jobs.append((category, split, image, original, '\n'.join(converted) + ('\n' if converted else '')))
            report['splits'][split] = {'images': len(images), 'boxes': sum(counts.values()), 'classes': dict(counts),
                                      'orphan_original_labels': [str(labels / (stem + '.txt')) for stem in sorted(label_stems-image_stems)]}
        report['excluded_classes'] = dict(excluded)
        reports[category] = report
    for category, split, image, original, content in jobs:
        target = output / category
        for folder in ('images', 'labels', '原始标签'):
            (target / folder / split).mkdir(parents=True, exist_ok=True)
        copied = target / 'images' / split / image.name
        shutil.copy2(image, copied)
        if hashlib.sha256(image.read_bytes()).digest() != hashlib.sha256(copied.read_bytes()).digest():
            raise ValueError(f'Copy verification failed: {image}')
        shutil.copy2(original, target / '原始标签' / split / original.name)
        label = target / 'labels' / split / original.name
        label.write_text(content, encoding='utf-8')
        if label.read_text(encoding='utf-8') != content:
            raise ValueError(f'Label write verification failed: {label}')
    for category, report in reports.items():
        target = output / category
        for split, stats in report['splits'].items():
            for orphan in stats['orphan_original_labels']:
                shutil.copy2(orphan, target / '原始标签' / split / Path(orphan).name)
        shutil.copy2(config, target / '原始类别配置_仅供查阅.yaml')
        lines = [f'path: {json.dumps(target.as_posix(), ensure_ascii=False)}']
        lines += [f'{split}: images/{split}' for split in report['splits']]
        lines += [f'nc: {len(names)}', 'names:'] + [f'  {i}: {json.dumps(name)}' for i, name in enumerate(names)]
        (target / 'data.yaml').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    report = {'names': names, 'source_names': source_names, 'mapping': mapping, 'datasets': reports,
              'note': '仅转换已有标注，未自动补标；原始标签保留。分别输出，未合并到训练数据集或改变集合划分。'}
    (output / 'conversion_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'datasets': reports}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--selected-six', action='store_true')
    selection.add_argument('--shovel', action='store_true')
    args = parser.parse_args()
    if args.shovel:
        convert_selected(args.root.resolve(), shovel=True)
    else:
        (convert_selected if args.selected_six else convert)(args.root.resolve())
