# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""将YOLO分割多边形以闭合线框绘制到原图上，供人工检查。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


SPLITS = ("train", "val")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root / "data/yolo_seg",
        help="转换后的YOLO分割数据集目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data/yolo_seg_label_lines",
        help="线框检查图输出目录",
    )
    parser.add_argument("--thickness", type=int, default=2, help="线框粗细")
    return parser.parse_args()


def read_image(image_path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    return image


def read_polygons(label_path: Path, width: int, height: int) -> list[np.ndarray]:
    polygons = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        tokens = line.split()
        if tokens[0] != "0" or len(tokens) < 7 or (len(tokens) - 1) % 2:
            raise ValueError(f"{label_path.name}:{line_number} 标签格式错误")

        coordinates = np.asarray(tokens[1:], dtype=np.float32).reshape(-1, 2)
        if np.any(coordinates < 0) or np.any(coordinates > 1):
            raise ValueError(f"{label_path.name}:{line_number} 存在越界坐标")
        coordinates[:, 0] *= width
        coordinates[:, 1] *= height
        polygons.append(np.rint(coordinates).astype(np.int32))
    return polygons


def write_image(output_path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError(f"无法编码图片: {output_path}")
    encoded.tofile(output_path)


def process_split(split: str, dataset: Path, output: Path, thickness: int) -> tuple[dict, list[str]]:
    image_root = dataset / "images" / split
    label_root = dataset / "labels" / split
    output_root = output / split
    output_root.mkdir(parents=True, exist_ok=True)
    for stale_preview in output_root.glob("*.jpg"):
        stale_preview.unlink()

    image_paths = sorted(image_root.glob("*.jpg"))
    errors = []
    stats = {"images": len(image_paths), "rendered": 0, "empty_labels": 0, "polygons": 0}

    for index, image_path in enumerate(image_paths, start=1):
        label_path = label_root / f"{image_path.stem}.txt"
        try:
            if not label_path.is_file():
                raise FileNotFoundError(f"缺少标签: {label_path}")
            image = read_image(image_path)
            height, width = image.shape[:2]
            polygons = read_polygons(label_path, width, height)
            if polygons:
                cv2.polylines(image, polygons, isClosed=True, color=(0, 255, 255), thickness=thickness)
            else:
                stats["empty_labels"] += 1

            write_image(output_root / image_path.name, image)
            stats["polygons"] += len(polygons)
            stats["rendered"] += 1
        except Exception as exc:
            errors.append(f"{split}/{image_path.name}: {exc}")

        if index % 250 == 0 or index == len(image_paths):
            print(f"[{split}] {index}/{len(image_paths)}")

    return stats, errors


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    report = {
        "dataset": str(dataset),
        "output": str(output),
        "drawing": "黄色闭合线框，不填充Mask",
        "thickness": args.thickness,
        "splits": {},
    }
    all_errors = []

    for split in SPLITS:
        stats, errors = process_split(split, dataset, output, args.thickness)
        report["splits"][split] = stats
        all_errors.extend(errors)

    (output / "visualization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "visualization_errors.txt").write_text(
        "\n".join(all_errors) + ("\n" if all_errors else ""), encoding="utf-8"
    )
    if all_errors:
        raise RuntimeError(f"线框图生成完成但发现{len(all_errors)}个错误")

    print(json.dumps(report["splits"], ensure_ascii=False, indent=2))
    print(f"线框检查图生成完成: {output}")


if __name__ == "__main__":
    main()
