"""Evaluate a fixed 50-image set with YOLO instance and merged semantic masks."""

from __future__ import annotations

import csv
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "data/test_train25_val25_b07_v4"
WEIGHTS = ROOT / "runs/segment/B07_v4/weights/best.pt"
OUTPUT = ROOT / "runs/segment/B07_v4_test50_eval"
SIZES = (512, 640, 960)
THRESHOLDS = (0.25, 0.40, 0.60)
RECOMMENDED_SIZE = 640


def ground_truth(label: Path, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize YOLO segmentation polygons at original image resolution."""
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for line in label.read_text(encoding="utf-8").splitlines():
        values = np.fromstring(line, sep=" ")
        if len(values) < 7 or int(values[0]) != 0 or (len(values) - 1) % 2:
            raise ValueError(f"Invalid grass polygon in {label}")
        points = values[1:].reshape(-1, 2) * (width, height)
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    return mask.astype(bool)


def scores(pred: np.ndarray, truth: np.ndarray) -> dict[str, int | float]:
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    tn = int(np.count_nonzero(~pred & ~truth))
    grass_union = tp + fp + fn
    background_union = tn + fp + fn
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "grass_iou": tp / grass_union if grass_union else float("nan"),
        "non_grass_iou": tn / background_union if background_union else float("nan"),
    }


def main() -> None:
    rows = list(csv.DictReader((SAMPLES / "manifest.csv").open(encoding="utf-8-sig", newline="")))
    if len(rows) != 50 or len({row["sample"] for row in rows}) != 50:
        raise ValueError("Expected 50 unique test images")
    image_dir = OUTPUT / "dataset/images/test"
    label_dir = OUTPUT / "dataset/labels/test"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        prefix = f"{row['source_split']}__"
        if not row["sample"].startswith(prefix):
            raise ValueError(f"Unexpected sample name: {row['sample']}")
        original_name = row["sample"][len(prefix):]
        source = ROOT / "data/yolo_seg/images" / row["source_split"] / original_name
        label = ROOT / "data/yolo_seg/labels" / row["source_split"] / f"{source.stem}.txt"
        if not source.is_file() or not label.is_file() or not (SAMPLES / row["sample"]).is_file():
            raise FileNotFoundError(source if not source.is_file() else label)
        shutil.copy2(SAMPLES / row["sample"], image_dir / row["sample"])
        shutil.copy2(label, label_dir / f"{Path(row['sample']).stem}.txt")
    dataset = OUTPUT / "dataset/test.yaml"
    dataset.write_text(yaml.safe_dump({"path": str((OUTPUT / "dataset").resolve()), "train": "images/test",
                                       "val": "images/test", "test": "images/test", "names": {0: "grass"}}), encoding="utf-8")

    model = YOLO(WEIGHTS)
    yolo = []
    for size in SIZES:
        metric = model.val(data=dataset, split="test", imgsz=size, conf=0.001, batch=1, device="cpu", workers=0,
                           plots=False, verbose=False, project=OUTPUT, name=f"yolo_metrics_{size}", exist_ok=True)
        yolo.append({"imgsz": size, "mask_precision": float(metric.seg.mp), "mask_recall": float(metric.seg.mr),
                     "mask_map50": float(metric.seg.map50), "mask_map50_95": float(metric.seg.map)})
    with (OUTPUT / "yolo_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=yolo[0].keys())
        writer.writeheader()
        writer.writerows(yolo)

    detail = []
    for size in SIZES:
        for row in rows:
            name = row["sample"]
            original = cv2.imread(str(image_dir / name))
            if original is None:
                raise ValueError(f"Cannot read {name}")
            height, width = original.shape[:2]
            truth = ground_truth(label_dir / f"{Path(name).stem}.txt", (height, width))
            result = model.predict(source=original, imgsz=size, conf=0.01, device="cpu", retina_masks=True, verbose=False)[0]
            instances = []
            if result.masks is not None:
                for confidence, mask in zip(result.boxes.conf.cpu().numpy(), result.masks.data.cpu().numpy()):
                    binary = mask.astype(np.uint8)
                    if binary.shape != (height, width):
                        binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
                    instances.append((float(confidence), binary.astype(bool)))
            for threshold in THRESHOLDS:
                pred = np.zeros_like(truth)
                for confidence, mask in instances:
                    if confidence >= threshold:
                        pred |= mask
                detail.append({"image": name, "imgsz": size, "conf": threshold,
                               "instances": sum(confidence >= threshold for confidence, _ in instances),
                               **scores(pred, truth)})
                if size == RECOMMENDED_SIZE and threshold == 0.25:
                    mask_dir = OUTPUT / "semantic_masks"
                    mask_dir.mkdir(exist_ok=True)
                    cv2.imwrite(str(mask_dir / f"{Path(name).stem}.png"), pred.astype(np.uint8) * 255)
                    preview_dir = OUTPUT / "previews"
                    preview_dir.mkdir(exist_ok=True)
                    preview = original.copy()
                    preview[pred] = (preview[pred] * 0.55 + np.array([255, 80, 0]) * 0.45).astype(np.uint8)
                    contours, _ = cv2.findContours(truth.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(preview, contours, -1, (0, 255, 255), 2)
                    cv2.imwrite(str(preview_dir / name), preview)
        print(f"Completed imgsz={size}", flush=True)

    with (OUTPUT / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail[0].keys())
        writer.writeheader()
        writer.writerows(detail)
    summary = []
    for size in SIZES:
        for threshold in THRESHOLDS:
            subset = [row for row in detail if row["imgsz"] == size and row["conf"] == threshold]
            tp, fp, fn, tn = (sum(row[key] for row in subset) for key in ("tp", "fp", "fn", "tn"))
            grass_iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
            non_grass_iou = tn / (tn + fp + fn) if tn + fp + fn else float("nan")
            summary.append({"imgsz": size, "conf": threshold, "images": len(subset), "tp": tp, "fp": fp,
                            "fn": fn, "tn": tn, "grass_iou": grass_iou, "non_grass_iou": non_grass_iou,
                            "miou": (grass_iou + non_grass_iou) / 2})
    with (OUTPUT / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    baseline = sorted((row for row in detail if row["imgsz"] == RECOMMENDED_SIZE and row["conf"] == 0.25),
                      key=lambda row: row["grass_iou"])
    with (OUTPUT / "failure_candidates.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=baseline[0].keys())
        writer.writeheader()
        writer.writerows(baseline)
    for page in range(2):
        canvas = np.full((5 * 256, 5 * 320, 3), 255, dtype=np.uint8)
        for slot, row in enumerate(baseline[page * 25:(page + 1) * 25]):
            thumbnail = cv2.resize(cv2.imread(str(OUTPUT / "previews" / row["image"])), (320, 240))
            y, x = divmod(slot, 5)
            canvas[y * 256:y * 256 + 240, x * 320:x * 320 + 320] = thumbnail
            stem_parts = Path(row["image"]).stem.split("_")
            code = stem_parts[-2] if stem_parts[-1] == "China" else stem_parts[-1]
            cv2.putText(canvas, f"{code}  IoU {row['grass_iou']:.3f}", (x * 320 + 4, y * 256 + 253),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1)
        cv2.imwrite(str(OUTPUT / f"contact_sheet_{page + 1}.jpg"), canvas)
    digest = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    (OUTPUT / "model_sha256.txt").write_text(f"{digest}  {WEIGHTS.name}\n", encoding="utf-8")
    log = {"run_utc": datetime.now(timezone.utc).isoformat(), "model_sha256": digest,
           "manifest_sha256": hashlib.sha256((SAMPLES / "manifest.csv").read_bytes()).hexdigest(),
           "test_images": len(rows), "device": "cpu", "sizes": list(SIZES), "thresholds": list(THRESHOLDS),
           "recommended_size": RECOMMENDED_SIZE, "recommended_conf": 0.25,
           "ap_conf": 0.001, "predict_conf": 0.01, "ultralytics": ultralytics.__version__,
           "torch": str(torch.__version__), "yolo_metrics": yolo, "semantic_summary": summary}
    (OUTPUT / "evaluation_log.yaml").write_text(yaml.safe_dump(log, allow_unicode=True), encoding="utf-8")
    print(f"YOLO: {yolo}\nSemantic: {summary}\nOutput: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
