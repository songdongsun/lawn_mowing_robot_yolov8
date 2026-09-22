# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL = PROJECT_ROOT / "runs/segment/yolov8n_seg/weights/best.pt"
SOURCE = PROJECT_ROOT / "data/test"
OUTPUT = PROJECT_ROOT / "runs/segment"

def main() -> None:
    """Run B01 inference on the local test images."""
    image_count = instance_count = 0
    results = YOLO(MODEL).predict(
        source=SOURCE, imgsz=640, conf=0.25, save=True, stream=True, project=OUTPUT, name="B01_park_test", exist_ok=True
    )
    for result in results:
        image_count += 1
        instance_count += len(result.boxes)
    print(f"Processed {image_count} images with {instance_count} grass instances.")
    print(f"Saved visualizations to: {(OUTPUT / 'B01_park_test').resolve()}")

if __name__ == "__main__":
    main()
