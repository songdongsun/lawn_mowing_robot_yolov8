"""Build a separate, traceable review set; never modify the active dataset."""
import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

from convert_hose_to_yolo import box_line


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_boxes(path):
    rows = [list(map(float, line.split())) for line in path.read_text().splitlines() if line.strip()]
    for c, x, y, w, h in rows:
        assert c == int(c) and 0 <= c < 12 and 0 < w <= 1 and 0 < h <= 1
        assert -1e-6 <= x-w/2 < x+w/2 <= 1.000001
        assert -1e-6 <= y-h/2 < y+h/2 <= 1.000001
    return rows


def polygons(path, size):
    masks = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        parts = list(map(float, line.split()))
        if not parts:
            continue
        assert len(parts) >= 7 and len(parts) % 2 == 1, path
        points = np.array(parts[1:]).reshape(-1, 2)
        assert np.isfinite(points).all() and points.min() >= 0 and points.max() <= 1
        mask = np.zeros((size[1], size[0]), np.uint8)
        cv2.fillPoly(mask, [np.rint(points * np.array(size)).astype(np.int32)], 255)
        masks.append(Image.fromarray(mask))
    return masks


def refine_mask(image, polygon, method='holes'):
    """Separate foreground from exterior background colors, including enclosed holes."""
    inside = np.asarray(polygon) > 0
    assert inside.any() and (~inside).any(), 'Mask needs foreground and background samples.'
    trimap = np.where(inside, cv2.GC_PR_FGD, cv2.GC_BGD).astype(np.uint8)
    cv2.setRNGSeed(20261016)
    cv2.grabCut(np.asarray(image), trimap, None, np.zeros((1, 65)), np.zeros((1, 65)),
                5, cv2.GC_INIT_WITH_MASK)
    removed = inside & ((trimap == cv2.GC_BGD) | (trimap == cv2.GC_PR_BGD))
    if method == 'foreground':
        return Image.fromarray((inside & ~removed).astype(np.uint8) * 255)
    assert method == 'holes'
    # Only remove enclosed background islands. Exterior-connected white blades,
    # frames and highlights are not holes and must retain their original mask.
    count, components = cv2.connectedComponents(removed.astype(np.uint8), connectivity=8)
    boundary = inside & (cv2.erode(inside.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0)
    foreground = inside.copy()
    for component in range(1, count):
        island = components == component
        if island.sum() >= 9 and not np.any(island & boundary):
            foreground[island] = False
    return Image.fromarray(foreground.astype(np.uint8) * 255)


def merge_review(root, source):
    """Import only generated samples from an explicitly approved review manifest."""
    report_path = root/'merge_augmentation_v1_report.json'
    assert not report_path.exists(), 'This batch already has a merge report.'
    manifest_path = source/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert Path(manifest['source_dataset']).resolve() == root
    yaml_before = (root/'data.yaml').read_bytes()
    for c, name in {7: 'watering-cane', 8: 'wheel-barrow', 9: 'pruning-saw', 10: 'garden hose'}.items():
        assert f'  {c}: "{name}"' in yaml_before.decode('utf-8')
    jobs, skipped, hashes = [], [], set()
    for sample in manifest['samples']:
        image, label = ((source/sample[key]).resolve() for key in ('image', 'label'))
        assert image.is_relative_to(source) and label.is_relative_to(source)
        assert sha(image) == sample['output_sha256']
        rows = read_boxes(label)
        assert rows and all(int(row[0]) == sample['class_id'] for row in rows)
        original = Path(sample['source_image']).resolve()
        assert original.parent == root/'images/train' and sha(original) == sample['source_sha256']
        if sample['mode'] == 'original':
            assert sha(image) == sha(original)
            assert sha(label) == sha(root/'labels/train'/f'{original.stem}.txt')
            skipped.append(str(image))
            continue
        assert sample['mode'] in ('photometric_flip', 'grass_composite')
        name = 'augv1_' + image.name
        assert sha(image) not in hashes, 'Duplicate generated images require review.'
        hashes.add(sha(image))
        for split in ('train', 'val'):
            assert not (root/'images'/split/name).exists()
            assert not (root/'labels'/split/f'{Path(name).stem}.txt').exists()
        jobs.append({'source_image': str(image), 'source_label': str(label), 'name': name,
                     'sha256': sha(image), 'label_sha256': sha(label), 'class_id': sample['class_id'],
                     'mode': sample['mode'], 'boxes': len(rows), 'foreground_source': str(original)})
    assert len(jobs) == 362 and len(skipped) == 118
    sizes = {Path(j['source_image']).stat().st_size for j in jobs}
    for split in ('train', 'val'):
        for image in (root/'images'/split).iterdir():
            if image.stat().st_size in sizes:
                assert sha(image) not in hashes, f'Duplicate in active {split}: {image}'

    def stats():
        result = {}
        for split in ('train', 'val'):
            images = list((root/'images'/split).glob('*'))
            labels = list((root/'labels'/split).glob('*.txt'))
            assert {p.stem for p in images} == {p.stem for p in labels}
            boxes, class_images = Counter(), Counter()
            for label in labels:
                ids = [int(row[0]) for row in read_boxes(label)]
                boxes.update(ids)
                class_images.update(set(ids))
            result[split] = {'images': len(images), 'boxes': sum(boxes.values()),
                             'class_images': dict(class_images), 'class_boxes': dict(boxes)}
        return result

    val_hashes = {str(p): sha(p) for folder in ('images', 'labels') for p in (root/folder/'val').iterdir()}
    before = stats()
    report = {'source': str(source), 'source_manifest_sha256': sha(manifest_path), 'before': before,
              'skipped_originals': skipped, 'files': jobs, 'status': 'planned',
              'note': 'First version explicitly selected by user. Known cutout/background defects retained; not a quality acceptance.'}
    with report_path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    for job in jobs:
        for field, folder, name in [('source_image', 'images', job['name']),
                                    ('source_label', 'labels', Path(job['name']).stem+'.txt')]:
            destination = root/folder/'train'/name
            with Path(job[field]).open('rb') as src, destination.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            assert sha(destination) == job['sha256' if folder == 'images' else 'label_sha256']
    after = stats()
    assert after['train']['images'] == before['train']['images'] + len(jobs)
    assert after['train']['boxes'] == before['train']['boxes'] + sum(j['boxes'] for j in jobs)
    assert before['val'] == after['val'] and all(sha(Path(p)) == digest for p, digest in val_hashes.items())
    assert (root/'data.yaml').read_bytes() == yaml_before
    report.update(after=after, status='complete', validation_unchanged=True,
                  added_images=len(jobs), added_boxes=sum(j['boxes'] for j in jobs))
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'added_images': len(jobs), 'skipped_originals': len(skipped), 'after': after}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--grass', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--merge-review', type=Path, help='Merge an approved first-version review into train only.')
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--cutout-selection', type=Path, help='Reviewed source objects and per-object masking methods.')
    args = parser.parse_args()
    if args.merge_review:
        merge_review(args.dataset.resolve(), args.merge_review.resolve())
        return
    if not args.grass or not args.output:
        parser.error('--grass and --output are required when generating a review')
    root, grass, out = args.dataset.resolve(), args.grass.resolve(), args.output.resolve()
    assert not out.exists(), 'Output must be new; never overwrite an earlier review.'
    rng = random.Random(args.seed)
    names = {7: 'watering-cane', 8: 'wheel-barrow', 9: 'pruning-saw', 10: 'garden hose'}
    approved = {}
    tiers = {}
    if args.cutout_selection:
        selection = json.loads(args.cutout_selection.read_text(encoding='utf-8'))
        previous = json.loads(Path(selection['review_manifest']).read_text(encoding='utf-8'))
        for field, tier in (('selection', 'high'), ('secondary_selection', 'secondary')):
            for key, method in selection.get(field, {}).items():
                sample, = [s for s in previous['samples'] if Path(s['image']).stem == f'{key}_grass_composite']
                source_key = (sample['source_image'], tuple(sample['object_crop']))
                assert source_key not in approved
                approved[source_key] = method
                tiers[source_key] = tier
    origins = {j['name']: Path(j['image']) for j in json.loads((root/'merge_six_report.json').read_text(encoding='utf-8'))['files'].values()}
    groups = {c: [] for c in names}
    for label in sorted((root/'labels/train').glob('*.txt')):
        rows = read_boxes(label)
        for c in names.keys() & {int(row[0]) for row in rows}:
            assert {int(row[0]) for row in rows} == {c}, 'Review multiclass sources separately.'
            image = root/'images/train'/f'{label.stem}.jpg'
            assert image.is_file()
            groups[c].append((image, label, rows))
    # Exclude byte-identical validation/test images from both foregrounds and backgrounds.
    forbidden = set()
    for dataset in (root, grass):
        for split in ('val', 'test'):
            for p in (dataset/'images'/split).glob('*'):
                if p.is_file():
                    forbidden.add(sha(p))
    for rows in groups.values():
        assert rows and len(rows) < 90
        for image, _, _ in rows:
            assert sha(image) not in forbidden, f'Validation duplicate: {image}'
    backgrounds = []
    background_hashes = []
    grass_paths = sorted((grass/'images/train').glob('*'))
    rng.shuffle(grass_paths)
    for image in grass_paths:
        label = grass/'labels/train'/f'{image.stem}.txt'
        if not label.is_file() or sha(image) in forbidden:
            continue
        with Image.open(image) as im:
            im = im.convert('RGB')
            if min(im.size) < 192:
                continue
            raw = [line.split() for line in label.read_text().splitlines() if line.strip()]
            assert all(int(row[0]) == 0 for row in raw)
            masks = polygons(label, im.size)
            if not masks:
                continue
            mask = np.maximum.reduce([np.asarray(m) for m in masks])
            for _ in range(60):
                side = rng.randint(192, min(512, *im.size))
                x, y = rng.randint(0, im.width-side), rng.randint(0, im.height-side)
                if np.all(mask[y:y+side, x:x+side] == 255):
                    thumbnail = np.asarray(im.crop((x, y, x+side, y+side)).convert('L').resize((17, 16)))
                    fingerprint = thumbnail[:, 1:] > thumbnail[:, :-1]
                    if any(np.count_nonzero(fingerprint != old) < 40 for old in background_hashes):
                        continue
                    backgrounds.append((image, (x, y, x+side, y+side)))
                    background_hashes.append(fingerprint)
                    break
        if len(backgrounds) == 120:
            break
    assert len(backgrounds) >= 30, f'Only {len(backgrounds)} clean grass crops found.'
    out.mkdir(parents=True)
    if args.cutout_selection:
        shutil.copy2(args.cutout_selection, out/'cutout_selection.json')
    manifest = {'seed': args.seed, 'target_per_class': 120, 'source_dataset': str(root),
                'grass_dataset': str(grass), 'background_pool': len(backgrounds),
                'cutout_selection': str(args.cutout_selection), 'status': 'review_only', 'samples': [], 'counts': {}}
    print(f'Diverse grass background crops: {len(backgrounds)}', flush=True)
    for c, name in names.items():
        folder = out/f'{c}_{name.replace(" ", "_")}'
        for sub in ('images/train', 'labels/train', 'preview', 'contact_sheets', 'masks', 'cutouts'):
            (folder/sub).mkdir(parents=True)
        sources = groups[c]
        # Every source gets comparable exposure; existing sources include augmentation pairs.
        schedule = sources.copy()
        if approved:
            schedule = [s for s in sources if any(key[0] == str(s[0]) and tiers[key] == 'high' for key in approved)]
            assert schedule
        rng.shuffle(schedule)
        cutouts = []
        for image, label, rows in sources:
            if approved and not any(key[0] == str(image) for key in approved):
                continue
            original = origins[image.name]
            annotation = original.parents[2]/'原始标签'/original.parent.name/f'{original.stem}.txt'
            assert sha(image) == sha(original)
            with Image.open(image) as im:
                im = im.convert('RGB')
                masks = polygons(annotation, im.size)
                assert len(masks) == len(rows)
                for object_index, (mask, row) in enumerate(zip(masks, rows)):
                    bounds = mask.getbbox()
                    assert bounds is not None
                    x, y, w, h = row[1:]
                    expected = np.array([x-w/2, y-h/2, x+w/2, y+h/2])*np.array([im.width, im.height]*2)
                    assert np.max(np.abs(np.array(bounds)-expected)) <= 3, 'Mask/box mismatch'
                    if min(bounds[2]-bounds[0], bounds[3]-bounds[1]) < 10:
                        continue
                    if approved and (str(image), bounds) not in approved:
                        continue
                    mask = refine_mask(im, mask, approved.get((str(image), bounds), 'holes'))
                    mask.save(folder/'masks'/f'{image.stem}_{object_index}.png')
                    transparent = im.convert('RGBA')
                    transparent.putalpha(mask)
                    transparent.crop(bounds).save(folder/'cutouts'/f'{image.stem}_{object_index}.png')
                    cutouts.append((image, annotation, im.crop(bounds), mask.crop(bounds), bounds))
        assert cutouts, f'No usable polygons for {name}'
        rng.shuffle(cutouts)
        high_cutouts = [item for item in cutouts if tiers.get((str(item[0]), item[4]), 'high') == 'high']
        secondary_cutouts = [item for item in cutouts if tiers.get((str(item[0]), item[4])) == 'secondary']
        assert high_cutouts
        if 'secondary' in tiers.values():
            assert secondary_cutouts, f'Missing reviewed secondary sources for {name}'
        counts = {'original': len(sources), 'photometric_flip': 90-len(sources), 'grass_composite': 30}
        counts['composite_quality'] = {'high': 24 if secondary_cutouts else 30,
                                       'secondary': 6 if secondary_cutouts else 0}
        for index in range(120):
            record = {'class_id': c, 'class_name': name}
            if index < 90:
                image, label, boxes = sources[index] if index < len(sources) else schedule[(index-len(sources)) % len(schedule)]
                im = Image.open(image).convert('RGB')
                record.update(source_image=str(image), source_label=str(label), source_sha256=sha(image))
                if index < len(sources):
                    mode = 'original'
                else:
                    mode = 'photometric_flip'
                    flip = bool((index-len(sources)) % 2)
                    brightness, contrast = rng.uniform(.78, 1.22), rng.uniform(.85, 1.15)
                    im = ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(brightness)).enhance(contrast)
                    if flip:
                        im = ImageOps.mirror(im)
                        boxes = [[cl, 1-x, y, w, h] for cl, x, y, w, h in boxes]
                    record.update(flip=flip, brightness=brightness, contrast=contrast)
            else:
                mode = 'grass_composite'
                quality = 'secondary' if secondary_cutouts and index >= 114 else 'high'
                pool = secondary_cutouts if quality == 'secondary' else high_cutouts
                image, annotation, obj, mask, bounds = pool[(index-(114 if quality == 'secondary' else 90)) % len(pool)]
                record['quality_tier'] = quality
                background, crop = backgrounds[((index-90)*4+c-7) % len(backgrounds)]
                im = Image.open(background).convert('RGB').crop(crop).resize((640, 640), Image.Resampling.LANCZOS)
                factor = rng.uniform(.80, 1.15)
                obj = ImageEnhance.Brightness(obj).enhance(factor)
                flip = rng.choice((True, False))
                if flip:
                    obj, mask = ImageOps.mirror(obj), ImageOps.mirror(mask)
                long_side = rng.randint(160, 350)
                scale = long_side/max(obj.size)
                size = tuple(max(4, round(s*scale)) for s in obj.size)
                obj = obj.resize(size, Image.Resampling.LANCZOS)
                mask = mask.resize(size, Image.Resampling.NEAREST)
                x, y = rng.randint(25, 615-size[0]), rng.randint(25, 615-size[1])
                # Feather inward only: never blend the original background back into holes.
                hard = np.asarray(mask)
                alpha = Image.fromarray(np.minimum(hard, cv2.GaussianBlur(hard, (3, 3), .5)))
                im.paste(obj, (x, y), alpha)
                left, top, right, bottom = mask.getbbox()
                coords = ((x+left)/640, (y+top)/640, (x+right)/640, (y+bottom)/640)
                boxes = [list(map(float, box_line(c, coords).split()))]
                record.update(mask_method=approved.get((str(image), bounds), 'holes'),
                              source_image=str(image), source_polygon=str(annotation), source_sha256=sha(image),
                              background=str(background), background_sha256=sha(background), grass_crop=crop,
                              object_crop=bounds, paste_xy=[x, y], object_size=size, flip=flip, brightness=factor)
            stem = f'{c}_{index:03d}_{mode}'
            if mode == 'grass_composite':
                stem += f'_{quality}'
            ip, lp = folder/'images/train'/f'{stem}.jpg', folder/'labels/train'/f'{stem}.txt'
            if mode == 'original':
                shutil.copy2(image, ip)
                shutil.copy2(label, lp)
                assert sha(image) == sha(ip) and sha(label) == sha(lp)
            else:
                im.save(ip, quality=95)
                lines = [box_line(int(cl), (x-w/2, y-h/2, x+w/2, y+h/2)) for cl, x, y, w, h in boxes]
                lp.write_text('\n'.join(lines)+'\n', encoding='utf-8')
            checked = read_boxes(lp)
            preview = Image.open(ip).convert('RGB')
            draw = ImageDraw.Draw(preview)
            for cl, x, y, w, h in checked:
                rect = [(x-w/2)*preview.width, (y-h/2)*preview.height, (x+w/2)*preview.width, (y+h/2)*preview.height]
                draw.rectangle(rect, outline='red', width=2)
                draw.text((rect[0], max(0, rect[1]-12)), f'{int(cl)} {name}', fill='red', stroke_width=1, stroke_fill='white')
            preview.save(folder/'preview'/f'{stem}.jpg', quality=90)
            record.update(mode=mode, image=str(ip.relative_to(out)), label=str(lp.relative_to(out)), output_sha256=sha(ip))
            manifest['samples'].append(record)
        previews = sorted((folder/'preview').glob('*.jpg'))
        for page in range(4):
            sheet = Image.new('RGB', (1200, 1100), 'white')
            for slot, path in enumerate(previews[page*30:(page+1)*30]):
                thumb = Image.open(path).convert('RGB')
                thumb.thumbnail((235, 155))
                x, y = (slot % 5)*240, (slot//5)*183
                sheet.paste(thumb, (x, y))
                ImageDraw.Draw(sheet).text((x, y+156), path.stem, fill='black')
            sheet.save(folder/'contact_sheets'/f'{page+1}.jpg', quality=90)
        manifest['counts'][name] = counts
        print(name, counts, flush=True)
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'审核说明.txt').write_text(
        '每类120张，包含现有训练原图；不是全部新增。original不可再次重复合并。\n'
        '仅供人工审核，未合并训练集，未改验证集；类别ID与当前12类配置一致。\n'
        'preview为逐图带框预览，contact_sheets为总览，images/train与labels/train为候选数据。\n'
        '合成只取训练侧草坪多边形内部区域；目标外轮廓内使用GrabCut细分前景，允许内部孔洞透明。\n'
        'masks保存细化后的源图尺寸掩膜；自动分割仍须检查白色物体、反光、细线是否误删。\n'
        'cutouts为透明物体素材；启用人工精选清单时，新增增强和合成仅使用清单内素材。\n'
        '已排除与验证/测试图片字节完全重复的素材；不能保证视频近邻帧或视觉近重复已排除。\n'
        '草坪标签不是障碍物全标注：请检查背景漏标、目标轮廓、内部孔洞、比例、透视、接地和边缘。\n'
        '不自动修复原始漏标；原图增强保留全部已有检测框，合成只标贴入的目标。\n'
        '来源、参数和哈希详见manifest.json；原始同源增强对不算独立场景。\n', encoding='utf-8')
    assert len(manifest['samples']) == 480
    print(out, flush=True)


if __name__ == '__main__':
    main()
