"""Generate aligned Caltech101 Edge files from an existing RGB train/test split."""

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms

from preprocess_rgb2edge import sobel_edge_extraction


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def collect_split(rgb_root: Path):
    plan = {}
    class_names = None
    split_keys = {}

    for split in ("train", "test"):
        split_dir = rgb_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing RGB split: {split_dir}")

        classes = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
        if not classes or any(name.lower() == "faces" for name in classes):
            raise ValueError(f"Invalid Caltech101 classes in {split_dir}")
        if class_names is None:
            class_names = classes
        elif classes != class_names:
            raise ValueError("RGB train and test must contain the same classes")

        samples = []
        keys = set()
        for class_name in classes:
            class_dir = split_dir / class_name
            files = sorted(path for path in class_dir.iterdir()
                           if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
            if not files:
                raise ValueError(f"No RGB images in {class_dir}")
            for source in files:
                key = (class_name, source.stem)
                if key in keys:
                    raise ValueError(f"Duplicate class/stem in {split}: {key}")
                keys.add(key)
                samples.append((source, class_name))
        plan[split] = samples
        split_keys[split] = keys

    overlap = split_keys["train"] & split_keys["test"]
    if overlap:
        raise ValueError(f"RGB train/test share class/stem: {next(iter(overlap))}")
    return class_names, plan, split_keys


def verify_dvs(dvs_root: Path, split_keys):
    for split in ("train", "test"):
        split_dir = dvs_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing DVS split: {split_dir}")
        dvs_keys = {(path.parent.name, path.stem)
                    for path in split_dir.glob("*/*.pt") if path.is_file()}
        if dvs_keys != split_keys[split]:
            missing = split_keys[split] - dvs_keys
            extra = dvs_keys - split_keys[split]
            raise ValueError(f"{split} RGB/DVS filenames differ: "
                             f"missing DVS={len(missing)}, extra DVS={len(extra)}")


def generate(rgb_root: Path, out_root: Path, img_size: int = 48,
             dvs_root: Optional[Path] = None):
    rgb_root = rgb_root.resolve(strict=True)
    out_root = out_root.resolve()
    if out_root.exists():
        raise FileExistsError(f"Output already exists: {out_root}")
    if out_root == rgb_root or rgb_root in out_root.parents:
        raise ValueError("Output must be outside the RGB directory")
    if img_size <= 0:
        raise ValueError("img_size must be positive")

    class_names, plan, split_keys = collect_split(rgb_root)
    if dvs_root is not None:
        verify_dvs(dvs_root.resolve(strict=True), split_keys)

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    out_root.mkdir(parents=True)
    with (out_root / "manifest.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["rgb_path", "split", "class", "filename", "edge_path", "label"])
        for split in ("train", "test"):
            for source, class_name in plan[split]:
                destination = out_root / split / class_name / f"{source.stem}.pt"
                destination.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(source) as image:
                    rgb = transform(image.convert("RGB"))
                edge = sobel_edge_extraction(rgb)
                label = class_to_idx[class_name]
                torch.save((edge, torch.tensor(label)), destination)
                writer.writerow([str(source), split, class_name, source.name,
                                 str(destination), label])
            print(f"{split}: {len(plan[split])} Edge files")

    with (out_root / "label_map.json").open("w", encoding="utf-8") as fp:
        json.dump(class_to_idx, fp, ensure_ascii=False, indent=2)
    print(f"Created {out_root}: {len(class_names)} classes, img_size={img_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb_root", type=Path, required=True,
                        help="Pre-split Caltech101 RGB root with train/ and test/")
    parser.add_argument("--out_root", type=Path, required=True,
                        help="New Edge output root")
    parser.add_argument("--img_size", type=int, default=48)
    parser.add_argument("--dvs_root", type=Path, default=None,
                        help="Optional synthetic DVS root to verify one-to-one filenames")
    args = parser.parse_args()
    generate(args.rgb_root, args.out_root, args.img_size, args.dvs_root)
