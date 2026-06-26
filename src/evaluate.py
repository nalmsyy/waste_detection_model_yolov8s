from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluasi model YOLOv8 hasil training.")
    parser.add_argument("--model", default="runs/detect/deteksi_sampah_5kelas/weights/best.pt")
    parser.add_argument("--data", default="datasets/sampah_5kelas/data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.model).exists():
        raise SystemExit(f"Model tidak ditemukan: {args.model}")
    if not Path(args.data).exists():
        raise SystemExit(f"data.yaml tidak ditemukan: {args.data}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    metrics = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, conf=args.conf, iou=args.iou, plots=True)
    print("Evaluasi selesai.")
    print(metrics)


if __name__ == "__main__":
    main()
