"""Fine-tune a pretrained YOLO model on dataset/data.yaml."""
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).parent


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    model = YOLO(cfg["model"])
    model.train(
        data=str(ROOT / "dataset" / "data.yaml"),
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        epochs=cfg["epochs"],
        patience=cfg["patience"],
        workers=2,          # Windows: keep dataloader workers low
        project=str(ROOT / "runs"),
        name="train",
        exist_ok=True,
    )
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.3f}  mAP50-95: {metrics.box.map:.3f}")
    print(f"Best weights: {ROOT / 'runs' / 'train' / 'weights' / 'best.pt'}")


if __name__ == "__main__":  # required on Windows for multiprocessing dataloaders
    main()
