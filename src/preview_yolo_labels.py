import cv2
import os
import random
import argparse
from pathlib import Path
import yaml

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def yolo_to_xyxy(label, img_w, img_h):
    cls_id, x_center, y_center, w, h = label
    x_center *= img_w
    y_center *= img_h
    w *= img_w
    h *= img_h

    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)

    return int(cls_id), x1, y1, x2, y2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/rf_deteksi_sampah_raw/data.yaml")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    data = load_yaml(args.data)
    dataset_root = Path(args.data).parent

    names = data.get("names", {})
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}

    image_dir = dataset_root / args.split / "images"
    label_dir = dataset_root / args.split / "labels"

    if not image_dir.exists():
        print(f"Folder gambar tidak ditemukan: {image_dir}")
        return

    if not label_dir.exists():
        print(f"Folder label tidak ditemukan: {label_dir}")
        return

    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpeg"))

    if not image_paths:
        print("Tidak ada gambar ditemukan.")
        return

    random.shuffle(image_paths)

    for img_path in image_paths[:args.count]:
        label_path = label_dir / f"{img_path.stem}.txt"

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_h, img_w = img.shape[:2]

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                label = list(map(float, parts))
                cls_id, x1, y1, x2, y2 = yolo_to_xyxy(label, img_w, img_h)

                class_name = names.get(cls_id, str(cls_id))

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    class_name,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("Preview YOLO Label", img)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()