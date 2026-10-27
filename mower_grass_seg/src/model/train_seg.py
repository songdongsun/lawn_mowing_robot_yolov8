# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Train the grass segmentation model."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT.parent))

from ultralytics import YOLO  # noqa: E402

DATA_CONFIG = PROJECT_ROOT / "data/yolo_seg/grass.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8n-seg.pt", help="Model checkpoint or YAML file")
    parser.add_argument("--data", type=Path, help="Dataset YAML file")
    parser.add_argument("--channel-attention", action="store_true", help="Add channel attention to YOLOv8-seg P3 fusion")
    parser.add_argument("--contrast-augmentation", action="store_true", help="Add contrast-only jitter (0.65-1.35, p=0.3)")
    parser.add_argument("--device", help="Training device, e.g. 2 or 2,5")
    parser.add_argument("--name", help="Output run name")
    parser.add_argument("--epochs", type=int, help="Maximum training epochs")
    parser.add_argument("--patience", type=int, help="Early stopping patience")
    parser.add_argument("--imgsz", type=int, help="Training image size")
    parser.add_argument("--batch", type=int, help="Batch size")
    parser.add_argument("--workers", type=int, help="Data loader workers")
    parser.add_argument("--lr0", type=float, help="Initial learning rate")
    parser.add_argument("--translate", type=float, help="Translation fraction (+/-)")
    parser.add_argument("--scale", type=float, help="Scale gain (+/-)")
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
        "resume": False,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.05,
        "scale": 0.2,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "bgr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 0,
        "seed": 0,
        "project": PROJECT_ROOT / "runs/segment",
        "name": "B08_v1",
    }
    if args.device:
        settings["device"] = args.device
    for key, value in vars(args).items():
        if key in settings and value is not None:
            settings[key] = value
    if args.smoke:
        settings.update(epochs=1, imgsz=640, batch=2, workers=0, fraction=0.02, val=False, plots=False, name="smoke")
    if args.contrast_augmentation:
        import albumentations as A

        from ultralytics.data.augment import Albumentations

        settings["augmentations"] = [
            *Albumentations().transform.transforms,
            A.ColorJitter(brightness=0, contrast=(0.65, 1.35), saturation=0, hue=0, p=0.3),
        ]

    model = YOLO(args.model)
    if args.channel_attention:
        config = deepcopy(model.model.yaml)
        layers = config["backbone"] + config["head"]
        layers[layers[-1][0][0]][2] = "C2fChannelAttention"
        attention_model = type(model.model)(config, verbose=False)
        attention_model.load(model.model)
        model.model = attention_model
    model.train(**settings)


if __name__ == "__main__":
    main()
