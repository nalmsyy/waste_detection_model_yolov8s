from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model YOLOv8 ke format lain seperti ONNX/TFLite.")
    parser.add_argument("--model", default="runs/detect/deteksi_sampah_5kelas/weights/best.pt")
    parser.add_argument("--format", default="onnx", help="onnx, tflite, openvino, engine, dll")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.model).exists():
        raise SystemExit(f"Model tidak ditemukan: {args.model}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    output = model.export(format=args.format, imgsz=args.imgsz)
    print(f"Export selesai: {output}")


if __name__ == "__main__":
    main()
