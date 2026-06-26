from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from common import (
    IMAGE_EXTS,
    SPLITS,
    copy_file,
    load_yaml,
    normalize_name,
    parse_names,
    save_yaml,
    split_image_label_dirs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge/remap class dataset Roboflow YOLO menjadi 5 kelas final: plastik, kertas, logam, organik, lainnya."
    )
    parser.add_argument("--raw", default="datasets/rf_deteksi_sampah_raw", help="Folder dataset YOLOv8 asli dari Roboflow")
    parser.add_argument("--out", default="datasets/sampah_5kelas", help="Folder output dataset 5 kelas")
    parser.add_argument("--mapping", default="configs/class_mapping.yaml", help="File mapping YAML")
    parser.add_argument("--overwrite", action="store_true", help="Hapus output lama sebelum proses")
    parser.add_argument(
        "--keep-empty-labels",
        action="store_true",
        help="Simpan gambar walau semua labelnya diskip. Default: tetap simpan label kosong untuk background images.",
    )
    return parser.parse_args()


def find_data_yaml(root: Path) -> Path:
    candidates = [root / "data.yaml", root / "data.yml"]
    for path in candidates:
        if path.exists():
            return path
    found = list(root.rglob("data.yaml")) + list(root.rglob("data.yml"))
    if not found:
        raise FileNotFoundError(f"data.yaml tidak ditemukan di {root}")
    return found[0]


def build_mapping(raw_names: List[str], mapping_yaml: dict) -> tuple[Dict[int, Optional[int]], List[str], Counter]:
    final_names = mapping_yaml.get("final_names") or ["plastik", "kertas", "logam", "organik", "lainnya"]
    final_lookup = {normalize_name(name): idx for idx, name in enumerate(final_names)}
    raw_to_target_names = {
        normalize_name(k): normalize_name(v)
        for k, v in (mapping_yaml.get("mapping") or {}).items()
    }

    class_map: Dict[int, Optional[int]] = {}
    action_counter = Counter()

    for old_id, old_name in enumerate(raw_names):
        old_norm = normalize_name(old_name)
        target_name = raw_to_target_names.get(old_norm)

        if target_name is None:
            # Fallback: jika nama class raw sudah sama dengan final_names.
            target_id = final_lookup.get(old_norm)
            if target_id is not None:
                class_map[old_id] = target_id
                action_counter["kept_same_name"] += 1
            else:
                class_map[old_id] = None
                action_counter["unmapped_skipped"] += 1
            continue

        if target_name == "__skip__":
            class_map[old_id] = None
            action_counter["configured_skipped"] += 1
            continue

        target_id = final_lookup.get(target_name)
        if target_id is None:
            class_map[old_id] = None
            action_counter["invalid_mapping_skipped"] += 1
        else:
            class_map[old_id] = target_id
            action_counter["mapped"] += 1

    return class_map, final_names, action_counter


def convert_label_file(src_label: Path, dst_label: Path, class_map: Dict[int, Optional[int]], stats: Counter) -> None:
    output_lines: List[str] = []

    if src_label.exists():
        for line in src_label.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                if parts:
                    stats["invalid_lines"] += 1
                continue
            try:
                old_id = int(float(parts[0]))
            except ValueError:
                stats["invalid_lines"] += 1
                continue

            new_id = class_map.get(old_id)
            if new_id is None:
                stats["boxes_skipped"] += 1
                continue

            output_lines.append(" ".join([str(new_id), *parts[1:]]))
            stats[f"class_{new_id}"] += 1
            stats["boxes_kept"] += 1
    else:
        stats["missing_label_files"] += 1

    dst_label.parent.mkdir(parents=True, exist_ok=True)
    dst_label.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")
    if output_lines:
        stats["label_files_with_boxes"] += 1
    else:
        stats["empty_label_files"] += 1


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw)
    out_root = Path(args.out)
    mapping_path = Path(args.mapping)

    if not raw_root.exists():
        raise SystemExit(f"Folder dataset raw tidak ditemukan: {raw_root}")

    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    data_yaml_path = find_data_yaml(raw_root)
    data = load_yaml(data_yaml_path)
    raw_names = parse_names(data.get("names", []))
    mapping_yaml = load_yaml(mapping_path)
    class_map, final_names, action_counter = build_mapping(raw_names, mapping_yaml)

    print("=" * 70)
    print("MERGE CLASS KE 5 KELAS")
    print("=" * 70)
    print(f"Dataset raw : {raw_root.resolve()}")
    print(f"data.yaml   : {data_yaml_path.resolve()}")
    print(f"Output      : {out_root.resolve()}")
    print("\nMapping class:")
    for old_id, old_name in enumerate(raw_names):
        new_id = class_map.get(old_id)
        if new_id is None:
            print(f"  {old_id:<2} {old_name:<40} -> SKIP")
        else:
            print(f"  {old_id:<2} {old_name:<40} -> {new_id}: {final_names[new_id]}")

    stats = Counter()

    for split in SPLITS:
        src_img_dir, src_lbl_dir = split_image_label_dirs(raw_root, split)
        if not src_img_dir.exists():
            continue

        # Roboflow biasanya memakai valid, tetapi data.yaml final untuk Ultralytics tetap bisa memakai valid.
        dst_split = split
        dst_img_dir = out_root / dst_split / "images"
        dst_lbl_dir = out_root / dst_split / "labels"

        images = [p for p in src_img_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        print(f"\nMemproses split {split}: {len(images)} gambar")

        for img_path in tqdm(images, desc=f"{split}"):
            rel = img_path.relative_to(src_img_dir)
            dst_img = dst_img_dir / rel
            copy_file(img_path, dst_img, overwrite=True)

            src_label = src_lbl_dir / rel.with_suffix(".txt")
            dst_label = dst_lbl_dir / rel.with_suffix(".txt")
            convert_label_file(src_label, dst_label, class_map, stats)
            stats[f"images_{split}"] += 1

    final_data = {
        "path": str(out_root.resolve()).replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images" if (out_root / "valid" / "images").exists() else "val/images",
        "test": "test/images" if (out_root / "test" / "images").exists() else "",
        "names": {idx: name for idx, name in enumerate(final_names)},
    }
    save_yaml(final_data, out_root / "data.yaml")

    print("\nRingkasan konfigurasi mapping:")
    for k, v in action_counter.items():
        print(f"  {k}: {v}")

    print("\nRingkasan label:")
    print(f"  boxes_kept      : {stats['boxes_kept']}")
    print(f"  boxes_skipped   : {stats['boxes_skipped']}")
    print(f"  invalid_lines   : {stats['invalid_lines']}")
    print(f"  missing_labels  : {stats['missing_label_files']}")
    for idx, name in enumerate(final_names):
        print(f"  {idx} {name:<10}: {stats[f'class_{idx}']}")

    print(f"\nSelesai. data.yaml final: {out_root / 'data.yaml'}")
    print("Lanjut cek dataset:")
    print(f"  python src/check_dataset.py --data {out_root / 'data.yaml'} --show-label-stats")


if __name__ == "__main__":
    main()
