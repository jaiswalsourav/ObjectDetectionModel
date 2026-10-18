"""Split labeled images into train/val and write data.yaml for YOLO.

Expected input (YOLO format, e.g. exported from Label Studio / Roboflow / labelImg):
    raw/images/*.jpg|png
    raw/labels/*.txt   (one line per box: class_id cx cy w h, normalized 0-1)

Output:
    dataset/images/{train,val}, dataset/labels/{train,val}, dataset/data.yaml
"""
import random
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
RAW = ROOT / "raw"
OUT = ROOT / "dataset"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    images = sorted(p for p in (RAW / "images").iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"No images found in {RAW / 'images'}")

    missing = [p.name for p in images if not (RAW / "labels" / f"{p.stem}.txt").exists()]
    if missing:
        print(f"Warning: {len(missing)} images have no label file (treated as background): {missing[:5]}...")

    random.seed(42)
    random.shuffle(images)
    n_val = max(1, int(len(images) * cfg["val_ratio"]))
    splits = {"val": images[:n_val], "train": images[n_val:]}

    if OUT.exists():
        shutil.rmtree(OUT)
    for split, files in splits.items():
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)
        for img in files:
            shutil.copy2(img, OUT / "images" / split / img.name)
            label = RAW / "labels" / f"{img.stem}.txt"
            if label.exists():
                shutil.copy2(label, OUT / "labels" / split / label.name)

    data = {
        "path": str(OUT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(cfg["classes"])},
    }
    (OUT / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"train: {len(splits['train'])}  val: {len(splits['val'])}  -> {OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
