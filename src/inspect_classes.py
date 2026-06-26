from __future__ import annotations

import argparse
from pathlib import Path

from common import load_yaml, parse_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lihat daftar class pada data.yaml YOLO.")
    parser.add_argument("--data", default="datasets/rf_deteksi_sampah_raw/data.yaml", help="Path ke data.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    data = load_yaml(data_path)
    names = parse_names(data.get("names", []))

    print(f"Data YAML: {data_path}")
    print(f"Jumlah class: {len(names)}")
    print("Daftar class:")
    for idx, name in enumerate(names):
        print(f"  {idx}: {name}")


if __name__ == "__main__":
    main()
