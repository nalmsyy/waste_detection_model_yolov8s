from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from common import SPLITS, count_images, count_labels, load_yaml, parse_names, resolve_dataset_root, split_image_label_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validasi cepat struktur dataset YOLO detection.")
    parser.add_argument("--data", default="datasets/sampah_5kelas/data.yaml", help="Path ke data.yaml")
    parser.add_argument("--show-label-stats", action="store_true", help="Hitung distribusi class dari file label")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    data = load_yaml(data_path)
    root = resolve_dataset_root(data_path)
    names = parse_names(data.get("names", []))

    print("=" * 70)
    print("CEK DATASET YOLO")
    print("=" * 70)
    print(f"data.yaml : {data_path.resolve()}")
    print(f"root      : {root}")
    print(f"classes   : {len(names)}")
    for idx, name in enumerate(names):
        print(f"  {idx}: {name}")

    print("\nSplit dataset:")
    total_images = 0
    total_labels = 0
    found_any = False

    for split in SPLITS:
        image_dir, label_dir = split_image_label_dirs(root, split)
        img_count = count_images(image_dir)
        lbl_count = count_labels(label_dir)
        if img_count == 0 and lbl_count == 0:
            continue
        found_any = True
        total_images += img_count
        total_labels += lbl_count
        status = "OK" if img_count == lbl_count else "CEK"
        print(f"  {split:<5} | images: {img_count:<6} | labels: {lbl_count:<6} | {status}")

    if not found_any:
        print("  Tidak ada split yang ditemukan. Pastikan struktur folder train/valid/test benar.")

    print(f"\nTotal images: {total_images}")
    print(f"Total labels: {total_labels}")

    if args.show_label_stats:
        print("\nDistribusi label per class:")
        counter = Counter()
        invalid = 0
        for split in SPLITS:
            _, label_dir = split_image_label_dirs(root, split)
            if not label_dir.exists():
                continue
            for txt in label_dir.rglob("*.txt"):
                for line in txt.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cls_id = int(float(parts[0]))
                    except ValueError:
                        invalid += 1
                        continue
                    if cls_id < 0 or cls_id >= len(names):
                        invalid += 1
                    else:
                        counter[cls_id] += 1

        for idx, name in enumerate(names):
            print(f"  {idx:<2} {name:<12}: {counter[idx]}")
        if invalid:
            print(f"  Invalid label lines: {invalid}")

    print("\nSelesai.")


if __name__ == "__main__":
    main()
