from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download dataset Roboflow Universe/Workspace dalam format YOLOv8."
    )
    parser.add_argument("--workspace", default="internship-rpdlx", help="Nama workspace Roboflow")
    parser.add_argument("--project", default="deteksi-sampah-r4w18", help="Nama project Roboflow")
    parser.add_argument("--version", type=int, default=5, help="Version dataset Roboflow")
    parser.add_argument("--format", default="yolov8", help="Format download Roboflow")
    parser.add_argument("--output", default="datasets/rf_deteksi_sampah_raw", help="Folder output dataset")
    parser.add_argument("--api-key", default=None, help="API key Roboflow. Bisa juga lewat env ROBOFLOW_API_KEY")
    parser.add_argument("--overwrite", action="store_true", help="Hapus folder output jika sudah ada")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.getenv("ROBOFLOW_API_KEY")

    if not api_key:
        raise SystemExit(
            "API key Roboflow belum ada.\n"
            "PowerShell: $env:ROBOFLOW_API_KEY=\"ISI_API_KEY_KAMU\"\n"
            "CMD       : set ROBOFLOW_API_KEY=ISI_API_KEY_KAMU\n"
            "Atau tambahkan argumen --api-key ISI_API_KEY_KAMU"
        )

    output_dir = Path(args.output)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise SystemExit("Library roboflow belum terinstall. Jalankan: pip install -r requirements.txt") from exc

    print("Menghubungkan ke Roboflow...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)

    print(f"Download dataset: workspace={args.workspace}, project={args.project}, version={args.version}")
    print(f"Format: {args.format}")
    print(f"Output: {output_dir.resolve()}")

    dataset = version.download(args.format, location=str(output_dir.resolve()), overwrite=args.overwrite)
    print("Selesai download.")
    print(f"Lokasi dataset: {dataset.location}")
    print("Cek data.yaml, lalu lanjut merge class ke 5 kelas.")


if __name__ == "__main__":
    main()
