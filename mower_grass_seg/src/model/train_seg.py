# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Train the grass segmentation model."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CONFIG = PROJECT_ROOT / "data/yolo_seg/grass.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8n-seg.pt", help="Model checkpoint or YAML file")
    parser.add_argument("--device", help="Training device, e.g. 2 or 2,5")
    parser.add_argument("--name", help="Output run name")
    parser.add_argument("--epochs", type=int, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, help="Early stopping patience")
    parser.add_argument("--imgsz", type=int, help="Training image size")
    parser.add_argument("--batch", type=int, help="Batch size")
    parser.add_argument("--workers", type=int, help="Data loader workers")
    parser.add_argument("--lr0", type=float, help="Initial learning rate")
    parser.add_argument("--mosaic", type=float, help="Mosaic probability")
    parser.add_argument("--close-mosaic", type=int, help="Final epochs without Mosaic")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--smoke", action="store_true", help="Run a short training smoke test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = {
        "data": DATA_CONFIG,
        "epochs": 200,
        "patience": 20,
        "imgsz": 640,
        "batch": 16,
        "workers": 4,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "momentum": 0.9,
        "warmup_bias_lr": 0.0,
        "amp": False,
        "mosaic": 0.3,
        "close_mosaic": 120,
        "seed": 0,
        "project": PROJECT_ROOT / "runs/segment",
        "name": "B08_v1",
    }
    if args.device:
        settings["device"] = args.device
    for key in ("name", "epochs", "patience", "imgsz", "batch", "workers", "lr0", "mosaic", "close_mosaic", "seed"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value
    if args.smoke:
        settings.update(epochs=1, imgsz=640, batch=2, workers=0, fraction=0.02, val=False, plots=False, name="smoke")

    YOLO(args.model).train(**settings)


if __name__ == "__main__":
    main()
