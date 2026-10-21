from pathlib import Path
import hashlib

import onnx
from ultralytics import YOLO


ROOT = Path("/root/cpp_project/am970/model")
WEIGHT = ROOT / "models" / "mower_yolov8n_seg.pt"
ONNX = ROOT / "models" / "mower_yolov8n_seg.onnx"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not WEIGHT.exists():
        raise FileNotFoundError(WEIGHT)

    model = YOLO(str(WEIGHT))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=640,
            opset=12,
            simplify=False,
            dynamic=False,
            half=False,
            nms=False,
            batch=1,
        )
    )
    if exported.resolve() != ONNX.resolve():
        exported.replace(ONNX)

    graph = onnx.load(str(ONNX))
    print("pt", WEIGHT, WEIGHT.stat().st_size, sha256(WEIGHT))
    print("onnx", ONNX, ONNX.stat().st_size, sha256(ONNX))
    print("ir_version", graph.ir_version)
    print("opset", [(o.domain or "ai.onnx", o.version) for o in graph.opset_import])
    print("inputs", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in graph.graph.input])
    print("outputs", [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in graph.graph.output])
    print("nodes", len(graph.graph.node))


if __name__ == "__main__":
    main()
