from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prediksi gambar/folder/video menggunakan model YOLOv8.")
    parser.add_argument("--model", default="runs/detect/deteksi_sampah_5kelas/weights/best.pt")
    parser.add_argument("--source", required=True, help="Path gambar, folder, video, atau URL")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--project", default="outputs/predict")
    parser.add_argument("--name", default="hasil_prediksi")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.model).exists():
        raise SystemExit(f"Model tidak ditemukan: {args.model}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=args.save,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )
    print(f"Prediksi selesai. Jumlah hasil: {len(results)}")
    print(f"Output tersimpan di: {args.project}/{args.name}")


if __name__ == "__main__":
    main()
