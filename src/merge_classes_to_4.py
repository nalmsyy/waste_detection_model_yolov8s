import argparse
import shutil
import yaml
from pathlib import Path


FINAL_CLASSES = {
    0: "plastik",
    1: "kertas",
    2: "logam",
    3: "lainnya",
}


def normalize_name(name: str) -> str:
    return (
        str(name)
        .lower()
        .strip()
        .replace("-", " ")
        .replace("_", " ")
        .replace("  ", " ")
    )


def map_class_to_4(class_name: str):
    """
    Mapping class dataset Roboflow ke 4 kelas final:
    0 plastik
    1 kertas
    2 logam
    3 lainnya
    """
    name = normalize_name(class_name)

    # Buang class aneh dari metadata/teks Roboflow kalau memang ada di data.yaml
    skip_keywords = [
        "annotate",
        "create datasets",
        "understand",
        "unstructured",
        "image data",
    ]

    for keyword in skip_keywords:
        if keyword in name:
            return None

    # Plastik
    if name in [
        "plastik",
        "plastic",
        "botol plastik",
        "bottle",
        "plastic bottle",
    ]:
        return 0

    # Kertas
    if name in [
        "kertas",
        "paper",
        "kardus",
        "cardboard",
        "karton",
        "koran",
        "newspaper",
    ]:
        return 1

    # Logam
    if name in [
        "logam",
        "metal",
        "kaleng",
        "can",
        "alumunium foil",
        "aluminium foil",
        "aluminum foil",
        "foil",
    ]:
        return 2

    # Sisanya masuk lainnya
    # Termasuk organik, kaca, gabus, kemasan, dan class lain.
    return 3


def load_yaml(yaml_path: Path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(yaml_path: Path, data: dict):
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def get_names_dict(data_yaml: dict):
    names = data_yaml.get("names", {})

    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}

    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}

    raise ValueError("Format names di data.yaml tidak dikenali.")


def convert_label_file(src_label: Path, dst_label: Path, old_id_to_new_id: dict):
    """
    Mengubah label YOLO lama ke label 4 kelas.
    Format label:
    class_id x_center y_center width height
    """
    new_lines = []

    if src_label.exists():
        with open(src_label, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            try:
                old_cls_id = int(float(parts[0]))
            except ValueError:
                continue

            if old_cls_id not in old_id_to_new_id:
                continue

            new_cls_id = old_id_to_new_id[old_cls_id]

            # Kalau None, label dibuang
            if new_cls_id is None:
                continue

            parts[0] = str(new_cls_id)
            new_lines.append(" ".join(parts[:5]))

    dst_label.parent.mkdir(parents=True, exist_ok=True)

    # Tetap buat file label, walaupun kosong.
    # File kosong artinya background/no object.
    with open(dst_label, "w", encoding="utf-8") as f:
        if new_lines:
            f.write("\n".join(new_lines) + "\n")


def copy_split(source_root: Path, target_root: Path, split: str, old_id_to_new_id: dict):
    src_images_dir = source_root / split / "images"
    src_labels_dir = source_root / split / "labels"

    dst_images_dir = target_root / split / "images"
    dst_labels_dir = target_root / split / "labels"

    if not src_images_dir.exists():
        print(f"[SKIP] Split tidak ditemukan: {split}")
        return

    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
        image_paths.extend(src_images_dir.glob(ext))

    copied = 0

    for img_path in image_paths:
        dst_img_path = dst_images_dir / img_path.name
        shutil.copy2(img_path, dst_img_path)

        src_label_path = src_labels_dir / f"{img_path.stem}.txt"
        dst_label_path = dst_labels_dir / f"{img_path.stem}.txt"

        convert_label_file(src_label_path, dst_label_path, old_id_to_new_id)

        copied += 1

    print(f"[{split}] copied images: {copied}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge dataset Roboflow menjadi 4 kelas: plastik, kertas, logam, lainnya"
    )

    parser.add_argument(
        "--source",
        default="datasets/rf_deteksi_sampah_raw",
        help="Folder dataset Roboflow utama",
    )

    parser.add_argument(
        "--target",
        default="datasets/sampah_4kelas",
        help="Folder output dataset 4 kelas",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Hapus folder target jika sudah ada",
    )

    args = parser.parse_args()

    source_root = Path(args.source)
    target_root = Path(args.target)

    data_yaml_path = source_root / "data.yaml"

    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml tidak ditemukan: {data_yaml_path}")

    if target_root.exists():
        if args.overwrite:
            shutil.rmtree(target_root)
        else:
            raise FileExistsError(
                f"Folder target sudah ada: {target_root}\n"
                f"Gunakan --overwrite untuk menimpa."
            )

    data_yaml = load_yaml(data_yaml_path)
    old_names = get_names_dict(data_yaml)

    print("Mapping class lama ke 4 kelas:")
    old_id_to_new_id = {}

    for old_id, old_name in old_names.items():
        new_id = map_class_to_4(old_name)
        old_id_to_new_id[old_id] = new_id

        if new_id is None:
            print(f"{old_id}: {old_name} -> SKIP")
        else:
            print(f"{old_id}: {old_name} -> {new_id}: {FINAL_CLASSES[new_id]}")

    for split in ["train", "valid", "test"]:
        copy_split(source_root, target_root, split, old_id_to_new_id)

    final_yaml = {
        "path": str(target_root.resolve()).replace("\\", "/"),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": FINAL_CLASSES,
    }

    save_yaml(target_root / "data.yaml", final_yaml)

    print("\nSelesai membuat dataset 4 kelas.")
    print(f"Output: {target_root}")
    print("Cek dataset dengan:")
    print("python src/check_dataset.py --data datasets/sampah_4kelas/data.yaml --show-label-stats")


if __name__ == "__main__":
    main()