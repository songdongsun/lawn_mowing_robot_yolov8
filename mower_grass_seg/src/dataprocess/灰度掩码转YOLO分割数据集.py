# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""将VOC灰度语义标签转换为YOLO分割多边形数据集。"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image


SPLITS = ("train", "val")
ALLOWED_MASK_VALUES = {0, 1, 2}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root / "data/original/VOCdevkit/VOC2007",
        help="VOC2007原始数据目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data/yolo_seg",
        help="YOLO分割数据输出目录",
    )
    parser.add_argument("--grass-value", type=int, default=1, help="灰度标签中的草坪像素值")
    parser.add_argument("--ignore-value", type=int, default=2, help="灰度标签中的忽略像素值")
    parser.add_argument("--min-contour-area", type=float, default=1.0, help="保留轮廓的最小像素面积")
    parser.add_argument("--quality-iou", type=float, default=0.98, help="低于该回填IoU的样本进入质量告警")
    parser.add_argument("--preview-count", type=int, default=20, help="每个集合生成的常规预览数量")
    return parser.parse_args()


def read_split_ids(split_file: Path) -> list[str]:
    ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    duplicates = [sample_id for sample_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"{split_file}存在重复样本ID: {duplicates[:10]}")
    return ids


def exclude_fully_ignored_samples(
    source: Path, sample_ids: list[str], ignore_value: int
) -> tuple[list[str], list[str]]:
    """Exclude samples that contain no trainable pixels from YOLO output."""
    kept = []
    excluded = []
    mask_root = source / "SegmentationClass"
    for sample_id in sample_ids:
        mask_path = mask_root / f"{sample_id}.png"
        mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is not None and np.all(mask == ignore_value):
            excluded.append(sample_id)
        else:
            kept.append(sample_id)
    return kept, excluded


def merge_holes(outer: np.ndarray, holes: list[np.ndarray]) -> np.ndarray:
    """用零宽桥连接外轮廓和孔洞，使单条YOLO多边形保留内部非草坪区域。"""
    polygon = outer.reshape(-1, 2)
    for hole in holes:
        hole = hole.reshape(-1, 2)
        distances = ((polygon[:, None, :] - hole[None, :, :]) ** 2).sum(axis=2)
        polygon_index, hole_index = np.unravel_index(np.argmin(distances), distances.shape)
        hole = np.roll(hole, -hole_index, axis=0)
        hole_path = np.concatenate((hole, hole[:1]), axis=0)
        polygon = np.concatenate(
            (polygon[: polygon_index + 1], hole_path, polygon[polygon_index :]), axis=0
        )
    return polygon


def find_contours(binary_mask: np.ndarray, min_area: float) -> tuple[list[np.ndarray], int]:
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return [], 0

    accepted: list[np.ndarray] = []
    skipped = 0
    hierarchy = hierarchy[0]
    outer_indexes = [index for index, item in enumerate(hierarchy) if item[3] == -1]
    outer_indexes.sort(key=lambda index: cv2.boundingRect(contours[index])[:2])
    for outer_index in outer_indexes:
        outer = contours[outer_index]
        if len(outer) < 3 or cv2.contourArea(outer) < min_area:
            skipped += 1
            continue

        holes = []
        child_index = hierarchy[outer_index][2]
        while child_index != -1:
            if len(contours[child_index]) >= 3:
                holes.append(contours[child_index])
            child_index = hierarchy[child_index][0]
        accepted.append(merge_holes(outer, holes))
    return accepted, skipped


def write_yolo_label(label_path: Path, contours: list[np.ndarray], width: int, height: int) -> None:
    lines = []
    for contour in contours:
        coordinates = []
        for x, y in contour:
            coordinates.extend((f"{x / width:.6f}", f"{y / height:.6f}"))
        lines.append("0 " + " ".join(coordinates))
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def calculate_reconstruction(
    mask: np.ndarray, contours: list[np.ndarray], grass_value: int, ignore_value: int
) -> tuple[np.ndarray, float]:
    reconstructed = np.zeros(mask.shape, dtype=np.uint8)
    if contours:
        cv2.fillPoly(reconstructed, [contour.astype(np.int32) for contour in contours], 1)

    valid = mask != ignore_value
    ground_truth = mask == grass_value
    intersection = np.logical_and(reconstructed == 1, ground_truth)[valid].sum()
    union = np.logical_or(reconstructed == 1, ground_truth)[valid].sum()
    iou = 1.0 if union == 0 else float(intersection / union)
    return reconstructed, iou


def save_preview(
    image_path: Path,
    mask: np.ndarray,
    reconstructed: np.ndarray,
    output_path: Path,
    grass_value: int,
    ignore_value: int,
) -> None:
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取原图: {image_path}")

    overlay = image.copy()
    overlay[mask == grass_value] = (0, 180, 0)
    overlay[mask == ignore_value] = (0, 0, 220)
    valid = mask != ignore_value
    overlay[np.logical_and(reconstructed == 1, mask != grass_value) & valid] = (180, 0, 180)
    overlay[np.logical_and(reconstructed == 0, mask == grass_value)] = (255, 80, 0)
    result = cv2.addWeighted(image, 0.55, overlay, 0.45, 0)
    boundaries, _ = cv2.findContours(reconstructed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, boundaries, -1, (0, 255, 255), 2)
    ok, encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError(f"无法编码预览图: {output_path}")
    encoded.tofile(output_path)


def process_split(
    split: str,
    sample_ids: list[str],
    source: Path,
    output: Path,
    args: argparse.Namespace,
) -> tuple[dict, list[dict], list[str]]:
    source_images = source / "JPEGImages"
    source_masks = source / "SegmentationClass"
    output_images = output / "images" / split
    output_labels = output / "labels" / split
    output_previews = output / "previews" / split
    for directory in (output_images, output_labels, output_previews):
        directory.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    errors: list[str] = []
    preview_written = 0
    low_iou_preview_written = 0
    stats = {
        "samples": len(sample_ids),
        "converted": 0,
        "empty_grass_samples": 0,
        "polygon_count": 0,
        "skipped_contours": 0,
        "ignore_samples": 0,
        "low_iou_samples": 0,
        "pixel_counts": {"0": 0, "1": 0, "2": 0},
    }

    for index, sample_id in enumerate(sample_ids, start=1):
        image_path = source_images / f"{sample_id}.jpg"
        mask_path = source_masks / f"{sample_id}.png"
        try:
            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f"缺少原图或标签: image={image_path.is_file()}, mask={mask_path.is_file()}")

            with Image.open(image_path) as image:
                image_size = image.size
            mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError("灰度标签无法读取")
            height, width = mask.shape
            if image_size != (width, height):
                raise ValueError(f"图片和标签尺寸不一致: image={image_size}, mask={(width, height)}")

            unique_values, counts = np.unique(mask, return_counts=True)
            unexpected = set(map(int, unique_values)) - ALLOWED_MASK_VALUES
            if unexpected:
                raise ValueError(f"存在未定义的标签值: {sorted(unexpected)}")
            for value, count in zip(unique_values, counts):
                stats["pixel_counts"][str(int(value))] += int(count)

            binary_mask = (mask == args.grass_value).astype(np.uint8)
            contours, skipped = find_contours(binary_mask, args.min_contour_area)
            reconstructed, reconstruction_iou = calculate_reconstruction(
                mask, contours, args.grass_value, args.ignore_value
            )

            destination_image = output_images / image_path.name
            destination_label = output_labels / f"{sample_id}.txt"
            shutil.copy2(image_path, destination_image)
            write_yolo_label(destination_label, contours, width, height)

            has_ignore = bool(np.any(mask == args.ignore_value))
            low_iou = reconstruction_iou < args.quality_iou
            if not contours:
                stats["empty_grass_samples"] += 1
            if has_ignore:
                stats["ignore_samples"] += 1
            if low_iou:
                stats["low_iou_samples"] += 1
            stats["polygon_count"] += len(contours)
            stats["skipped_contours"] += skipped
            stats["converted"] += 1

            record = {
                "id": sample_id,
                "width": width,
                "height": height,
                "polygons": len(contours),
                "skipped_contours": skipped,
                "has_ignore": has_ignore,
                "reconstruction_iou": round(reconstruction_iou, 6),
            }
            records.append(record)

            save_regular_preview = preview_written < args.preview_count
            save_low_iou_preview = low_iou and low_iou_preview_written < args.preview_count
            if save_regular_preview or save_low_iou_preview:
                prefix = "low_iou" if save_low_iou_preview else "sample"
                preview_name = f"{prefix}_{sample_id}_iou_{reconstruction_iou:.4f}.jpg"
                save_preview(
                    image_path,
                    mask,
                    reconstructed,
                    output_previews / preview_name,
                    args.grass_value,
                    args.ignore_value,
                )
                if save_regular_preview:
                    preview_written += 1
                if save_low_iou_preview:
                    low_iou_preview_written += 1
        except Exception as exc:
            errors.append(f"{split}/{sample_id}: {exc}")

        if index % 250 == 0 or index == len(sample_ids):
            print(f"[{split}] {index}/{len(sample_ids)}")

    return stats, records, errors


def verify_output(output: Path, expected_ids: dict[str, list[str]]) -> list[str]:
    errors = []
    for split, sample_ids in expected_ids.items():
        expected = set(sample_ids)
        image_ids = {path.stem for path in (output / "images" / split).glob("*.jpg")}
        label_ids = {path.stem for path in (output / "labels" / split).glob("*.txt")}
        if image_ids != expected:
            errors.append(f"{split}图片集合不一致: missing={len(expected - image_ids)}, extra={len(image_ids - expected)}")
        if label_ids != expected:
            errors.append(f"{split}标签集合不一致: missing={len(expected - label_ids)}, extra={len(label_ids - expected)}")
    return errors


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    split_root = source / "ImageSets" / "Segmentation"
    metadata_root = output / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)

    source_split_ids = {split: read_split_ids(split_root / f"{split}.txt") for split in SPLITS}
    overlap = set(source_split_ids["train"]) & set(source_split_ids["val"])
    if overlap:
        raise ValueError(f"训练集和验证集存在重叠: {sorted(overlap)[:10]}")

    split_ids = {}
    excluded_ids = {}
    for split in SPLITS:
        split_ids[split], excluded_ids[split] = exclude_fully_ignored_samples(
            source, source_split_ids[split], args.ignore_value
        )

    for split in SPLITS:
        for image in (output / "images" / split).glob("*.jpg"):
            image.unlink()
        for label in (output / "labels" / split).glob("*.txt"):
            label.unlink()
        for preview in (output / "previews" / split).glob("*.jpg"):
            preview.unlink()

    report = {
        "source": str(source),
        "output": str(output),
        "class_mapping": {"source_0": "background", "source_1": "grass -> yolo class 0", "source_2": "ignore"},
        "settings": {
            "min_contour_area": args.min_contour_area,
            "quality_iou": args.quality_iou,
            "preview_count": args.preview_count,
        },
        "excluded_fully_ignored": excluded_ids,
        "splits": {},
    }
    all_records = {}
    all_errors = []

    for split in SPLITS:
        source_split = split_root / f"{split}.txt"
        shutil.copy2(source_split, metadata_root / f"source_{split}.txt")
        (metadata_root / f"effective_{split}.txt").write_text(
            "\n".join(split_ids[split]) + "\n", encoding="utf-8"
        )
        stats, records, errors = process_split(split, split_ids[split], source, output, args)
        stats["source_samples"] = len(source_split_ids[split])
        stats["excluded_fully_ignored_samples"] = len(excluded_ids[split])
        report["splits"][split] = stats
        all_records[split] = sorted(records, key=lambda item: item["reconstruction_iou"])
        all_errors.extend(errors)

    all_errors.extend(verify_output(output, split_ids))
    (metadata_root / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (metadata_root / "sample_quality.json").write_text(
        json.dumps(all_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (metadata_root / "conversion_errors.txt").write_text(
        "\n".join(all_errors) + ("\n" if all_errors else ""), encoding="utf-8"
    )
    (output / "grass.yaml").write_text(
        yaml.safe_dump(
            {"train": "images/train", "val": "images/val", "names": {0: "grass"}},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    if all_errors:
        raise RuntimeError(f"转换完成但发现{len(all_errors)}个错误，请检查conversion_errors.txt")

    print(json.dumps(report["splits"], ensure_ascii=False, indent=2))
    print(f"转换完成: {output}")


if __name__ == "__main__":
    main()
