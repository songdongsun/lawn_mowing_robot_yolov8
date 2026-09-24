# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Train the grass segmentation model."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CONFIG = PROJECT_ROOT / "data/yolo_seg/grass.yaml"
B03_BEST = PROJECT_ROOT / "runs/segment/B03/weights/best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=B03_BEST, help="Model checkpoint or YAML file")
    parser.add_argument("--device", help="Training device, e.g. 2 or 2,5")
    parser.add_argument("--smoke", action="store_true", help="Run a short training smoke test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = {
        "data": DATA_CONFIG,
        "epochs": 40,
        "patience": 15,
        "imgsz": 640,
        "batch": 16,
        "workers": 4,
        "optimizer": "AdamW",
        "lr0": 0.0001,
        "momentum": 0.9,
        "warmup_bias_lr": 0.0,
        "amp": False,
        "mosaic": 0.0,
        "close_mosaic": 0,
        "project": PROJECT_ROOT / "runs/segment",
        "name": "B04_A",
    }
    if args.device:
        settings["device"] = args.device
    if args.smoke:
        settings.update(epochs=1, imgsz=640, batch=2, workers=0, fraction=0.02, val=False, plots=False, name="smoke")

    YOLO(args.model).train(**settings)


if __name__ == "__main__":
    main()
