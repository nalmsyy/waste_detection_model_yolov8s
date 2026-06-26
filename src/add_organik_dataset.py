import argparse
import shutil
from pathlib import Path


FINAL_ORGANIK_CLASS_ID = 3


def rewrite_label_to_organik(src_label_path: Path, dst_label_path: Path):
    """
    Mengubah semua class id pada label YOLO menjadi class organik.
    Format YOLO:
    class_id x_center y_center width height
    """
    if not src_label_path.exists():
        return

    new_lines = []

    with open(src_label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) < 5:
            continue

        # Ganti class id menjadi 3 = organik
        parts[0] = str(FINAL_ORGANIK_CLASS_ID)
        new_lines.append(" ".join(parts[:5]))

    if new_lines:
        dst_label_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")


def copy_split(src_root: Path, dst_root: Path, split: str):
    src_images_dir = src_root / split / "images"
    src_labels_dir = src_root / split / "labels"

    dst_images_dir = dst_root / split / "images"
    dst_labels_dir = dst_root / split / "labels"

    if not src_images_dir.exists():
        print(f"[SKIP] Folder tidak ditemukan: {src_images_dir}")
        return

    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
        image_paths.extend(src_images_dir.glob(ext))

    copied = 0
    skipped = 0

    for img_path in image_paths:
        src_label_path = src_labels_dir / f"{img_path.stem}.txt"

        if not src_label_path.exists():
            skipped += 1
            continue

        # Prefix agar nama file tidak bentrok dengan dataset utama
        new_name = f"organik_extra_{img_path.name}"
        dst_img_path = dst_images_dir / new_name
        dst_label_path = dst_labels_dir / f"{Path(new_name).stem}.txt"

        shutil.copy2(img_path, dst_img_path)
        rewrite_label_to_organik(src_label_path, dst_label_path)

        copied += 1

    print(f"[{split}] copied: {copied}, skipped tanpa label: {skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--organik-root",
        default="datasets/rf_organik_raw",
        help="Folder dataset organik hasil download Roboflow",
    )
    parser.add_argument(
        "--target-root",
        default="datasets/sampah_5kelas",
        help="Folder dataset final 5 kelas",
    )
    args = parser.parse_args()

    organik_root = Path(args.organik_root)
    target_root = Path(args.target_root)

    if not organik_root.exists():
        raise FileNotFoundError(f"Dataset organik tidak ditemukan: {organik_root}")

    if not target_root.exists():
        raise FileNotFoundError(
            f"Dataset target belum ada: {target_root}\n"
            f"Jalankan dulu: python src/merge_classes_to_5.py --overwrite"
        )

    for split in ["train", "valid", "test"]:
        copy_split(organik_root, target_root, split)

    print("\nSelesai menambahkan dataset organik.")
    print("Class organik disimpan sebagai ID 3.")
    print("Lanjut cek dataset:")
    print("python src/check_dataset.py --data datasets/sampah_5kelas/data.yaml --show-label-stats")


if __name__ == "__main__":
    main()