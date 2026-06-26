from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ["train", "valid", "val", "test"]


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def normalize_name(name: object) -> str:
    return str(name).strip().lower().replace("_", " ").replace("-", " ")


def parse_names(names_obj) -> List[str]:
    if isinstance(names_obj, dict):
        parsed: List[Tuple[int, str]] = []
        for key, value in names_obj.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            parsed.append((idx, str(value)))
        return [name for _, name in sorted(parsed, key=lambda item: item[0])]

    if isinstance(names_obj, list):
        return [str(x) for x in names_obj]

    raise ValueError("Format 'names' pada data.yaml tidak dikenali. Harus list atau dict.")


def resolve_dataset_root(data_yaml: str | Path) -> Path:
    data_yaml = Path(data_yaml)
    data = load_yaml(data_yaml)
    raw_path = data.get("path")
    if raw_path:
        root = Path(raw_path)
        if not root.is_absolute():
            root = (data_yaml.parent / root).resolve()
        return root
    return data_yaml.parent.resolve()


def split_image_label_dirs(root: Path, split: str) -> Tuple[Path, Path]:
    """Mendukung dua struktur umum Roboflow/YOLO:
    1) root/train/images dan root/train/labels
    2) root/images/train dan root/labels/train
    """
    first_images = root / split / "images"
    first_labels = root / split / "labels"
    if first_images.exists():
        return first_images, first_labels

    second_images = root / "images" / split
    second_labels = root / "labels" / split
    return second_images, second_labels


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def count_labels(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*.txt"))


def copy_file(src: Path, dst: Path, overwrite: bool = False) -> None:
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
