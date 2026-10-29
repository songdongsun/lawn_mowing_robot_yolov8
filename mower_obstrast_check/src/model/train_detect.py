"""Command-line obstacle detection training; no training starts on import."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8n.pt", help="Detection checkpoint or model YAML")
    parser.add_argument("--channel-attention", action="store_true", help="Add channel attention to the P3 C2f fusion")
    parser.add_argument("--data", type=Path, help="Custom dataset YAML; its paths are used unchanged")
    parser.add_argument("--data-root", type=Path, help="Dataset root override; contains images/ and labels/")
    parser.add_argument("--smoke", action="store_true", help="One epoch on a small training fraction")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration without writing or training")
    for key in ("device", "name", "optimizer", "project"):
        parser.add_argument(f"--{key}")
    for key in ("epochs", "patience", "imgsz", "batch", "workers", "close-mosaic", "seed"):
        parser.add_argument(f"--{key}", type=int)
    for key in ("lr0", "lrf", "mosaic", "fraction"):
        parser.add_argument(f"--{key}", type=float)
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                        help="Other Ultralytics options, e.g. amp=true cache=false fliplr=0.5")
    args = parser.parse_args(argv)
    settings = dict(epochs=200, patience=20, imgsz=640, batch=16, workers=4,
                    optimizer="AdamW", lr0=0.001, momentum=0.9, warmup_bias_lr=0.0,
                    amp=False, mosaic=0.3, close_mosaic=10, seed=0,
                    project=str(PROJECT_ROOT / "runs/detect"), name="D01", exist_ok=False)
    if args.smoke:
        settings.update(epochs=1, batch=2, workers=0, fraction=0.002,
                        mosaic=0.0, close_mosaic=0, plots=False, name="smoke")
    controls = {"model", "channel_attention", "data", "data_root", "smoke", "dry_run", "set"}
    settings.update({k: v for k, v in vars(args).items() if k not in controls and v is not None})
    for item in args.set:
        key, separator, value = item.partition("=")
        if not separator or not value or key in {"model", "data", "task", "mode"}:
            parser.error("--set expects KEY=VALUE; model/data use their own options; task is detect")
        settings[key] = yaml.safe_load(value)
    if args.channel_attention and settings.get("resume"):
        parser.error("Use --model last.pt --set resume=true without --channel-attention to resume")
    return args, settings


def main():
    args, settings = parse_args()
    source = (args.data or PROJECT_ROOT / "data/yolo_check/data.yaml").resolve()
    dataset = yaml.safe_load(source.read_text(encoding="utf-8"))
    # Relocate the project's Windows YAML without changing the original dataset config.
    relocate = args.data is None or args.data_root is not None
    if relocate:
        dataset["path"] = str((args.data_root or source.parent).resolve())
    settings["data"] = str(source)
    print(yaml.safe_dump(dict(model=args.model, channel_attention=args.channel_attention,
                             train=settings, dataset=dataset), allow_unicode=True), flush=True)
    if args.dry_run:
        return
    from ultralytics import YOLO

    model = YOLO(args.model)
    if model.task != "detect":
        raise ValueError("Use a detection model, not a segmentation checkpoint")
    if relocate or args.channel_attention:
        # Unique persistent YAML keeps each run reproducible and concurrent runs independent.
        from tempfile import NamedTemporaryFile

        config_dir = PROJECT_ROOT / "logs/train_configs"
        config_dir.mkdir(parents=True, exist_ok=True)
    if args.channel_attention:
        config = deepcopy(model.model.yaml)
        layers = config["backbone"] + config["head"]
        p3 = layers[-1][0][0]
        if layers[p3][2] not in {"C2f", "C2fChannelAttention"}:
            raise ValueError("--channel-attention requires a C2f P3 fusion layer")
        layers[p3][2] = "C2fChannelAttention"
        config.pop("yaml_file", None)
        # DDP rebuilds from args.model: persist the modified architecture for every rank.
        with NamedTemporaryFile(mode="w", suffix=".yaml", prefix="detect_ca_", dir=config_dir,
                                encoding="utf-8", delete=False) as stream:
            yaml.safe_dump(config, stream, allow_unicode=True)
        if model.ckpt:
            settings.setdefault("pretrained", str(Path(model.ckpt_path).resolve()))
        model = YOLO(stream.name, task="detect")
    if relocate:
        with NamedTemporaryFile(mode="w", suffix=".yaml", prefix="detect_", dir=config_dir,
                                encoding="utf-8", delete=False) as stream:
            yaml.safe_dump(dataset, stream, allow_unicode=True)
            settings["data"] = stream.name
    model.train(**settings)


if __name__ == "__main__":
    main()
