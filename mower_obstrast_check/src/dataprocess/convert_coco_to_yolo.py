"""Convert the downloaded COCO subset to project YOLO boxes without modifying sources."""

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path


NAMES = ["person", "cat", "dog", "frisbee", "sports ball", "teddy bear"]
CATEGORY_MAP = {1: 0, 17: 1, 18: 2, 34: 3, 37: 4, 88: 5}


def convert(source, output):
    """Preserve official splits; quarantine crowd images and reject invalid boxes."""
    source, output = source.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output must be empty (no automatic overwrite): {output}")
    report = {
        "source": str(source),
        "names": NAMES,
        "category_map": CATEGORY_MAP,
        "image_storage": "hardlinks; do not edit image contents in place",
        "crowd_policy": "exclude entire images containing target-category crowd annotations",
        "splits": {},
    }
    for split, official in (("train", "train2017"), ("val", "val2017")):
        annotation_path = source / "annotations" / f"instances_{official}.json"
        annotation_bytes = annotation_path.read_bytes()
        digest = hashlib.sha256(annotation_bytes).hexdigest()
        data = json.loads(annotation_bytes)
        del annotation_bytes
        by_image = defaultdict(list)
        for annotation in data["annotations"]:
            if annotation["category_id"] in CATEGORY_MAP:
                by_image[annotation["image_id"]].append(annotation)
        image_info = {item["file_name"]: item for item in data["images"]}
        del data
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        counts, class_images = Counter(), Counter()
        exclusions = []
        stats = Counter()
        paths = sorted((source / "images" / official).glob("*.jpg"))
        if not paths:
            raise ValueError(f"No source JPG images for {official}")
        for index, path in enumerate(paths, 1):
            info = image_info[path.name]
            annotations = by_image[info["id"]]
            reason = ""
            if any(a.get("iscrowd", 0) for a in annotations):
                reason = "target_crowd"
            elif not annotations:
                reason = "no_selected_annotations"
            elif any(a["bbox"][2] <= 0 or a["bbox"][3] <= 0 for a in annotations):
                reason = "invalid_box_dimensions"
            if reason:
                exclusions.append({"file_name": path.name, "reason": reason})
                stats[reason] += 1
                continue
            width, height = info["width"], info["height"]
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image dimensions: {path}")
            lines = {}
            for annotation in annotations:
                x, y, w, h = annotation["bbox"]
                if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
                    raise ValueError(f"Invalid bbox: annotation {annotation['id']}")
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(width, x + w), min(height, y + h)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"Box outside image: annotation {annotation['id']}")
                stats["clipped_boxes"] += (x1, y1, x2, y2) != (x, y, x + w, y + h)
                cls = CATEGORY_MAP[annotation["category_id"]]
                values = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                          (x2 - x1) / width, (y2 - y1) / height)
                line = f"{cls} " + " ".join(f"{value:.10f}" for value in values)
                stats["duplicate_boxes_removed"] += line in lines
                lines[line] = cls
            os.link(path, image_dir / path.name)
            (label_dir / f"{path.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            counts.update(lines.values())
            class_images.update(set(lines.values()))
            stats["output_images"] += 1
            if index % 5000 == 0:
                print(f"{split}: scanned {index}/{len(paths)}, exported {stats['output_images']}", flush=True)
        if hashlib.sha256(annotation_path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source annotations changed during conversion: {annotation_path}")
        stats["source_images"] = len(paths)
        report["splits"][split] = {
            **stats,
            "annotation_sha256": digest,
            "boxes": {name: counts[i] for i, name in enumerate(NAMES)},
            "class_images": {name: class_images[i] for i, name in enumerate(NAMES)},
        }
        (output / f"excluded_{split}.json").write_text(
            json.dumps(exclusions, indent=2), encoding="utf-8"
        )
        print(f"{split}: {dict(stats)}, boxes={sum(counts.values())}", flush=True)
    yaml = f"path: {json.dumps(output.as_posix(), ensure_ascii=False)}\ntrain: images/train\nval: images/val\nnames:\n"
    yaml += "".join(f"  {i}: {name}\n" for i, name in enumerate(NAMES))
    (output / "data.yaml").write_text(yaml, encoding="utf-8")
    (output / "conversion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "README.txt").write_text(
        "YOLO detection: class_id x_center y_center width height (normalized).\n"
        "Official COCO train/val split preserved; no test split created.\n"
        "Images are HARDLINKS to coco_part, not independent copies. Do not edit pixels in place.\n"
        "Deleting a link does not delete the source image; copying this dataset makes independent files.\n"
        "Target crowd images are excluded entirely, listed in excluded_train/val.json; sources retained.\n"
        "Out-of-bounds boxes clipped; identical output boxes deduplicated. Invalid boxes stop conversion.\n"
        "Six original category names retained: person, cat, dog, frisbee, sports ball, teddy bear.\n"
        "Original COCO IDs 1,17,18,34,37,88 mapped to contiguous IDs 0,1,2,3,4,5.\n"
        "Images containing zero/negative-size target boxes are excluded and recorded.\n"
        "Format conversion is NOT completion of manual cleaning.\n"
        "Review and supplement annotations before final training. No downloads or image resizing performed.\n"
        "data.yaml contains an absolute path; update path when moving the dataset.\n"
        "Conversion refuses nonempty outputs. See conversion_report.json for split/class counts.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=project / "data/coco_part/raw/coco2017")
    parser.add_argument("--output", type=Path, default=project / "data/yolo_check")
    args = parser.parse_args()
    convert(args.source, args.output)
