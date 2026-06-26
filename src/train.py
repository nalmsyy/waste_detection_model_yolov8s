from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training YOLOv8 detection untuk deteksi sampah 5 kelas.")
    parser.add_argument("--data", default="datasets/sampah_5kelas/data.yaml", help="Path data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained model YOLOv8: yolov8n.pt/yolov8s.pt/dll")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="0 untuk GPU pertama, cpu untuk CPU, kosongkan untuk auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="deteksi_sampah_5kelas")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"data.yaml tidak ditemukan: {data_path}\nJalankan download dan merge dataset dulu.")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics belum terinstall. Jalankan: pip install -r requirements.txt") from exc

    model = YOLO(args.model)
    kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "patience": args.patience,
        "exist_ok": args.exist_ok,
        "plots": True,
    }
    if args.device:
        kwargs["device"] = args.device

    print("Mulai training YOLOv8...")
    print(kwargs)
    model.train(**kwargs)
    print("Training selesai.")
    print(f"Model terbaik biasanya ada di: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
