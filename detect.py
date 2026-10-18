"""Run the trained model on an image, folder, video file, or webcam.

Examples:
    python detect.py --source photo.jpg
    python detect.py --source test_images/
    python detect.py --source video.mp4
    python detect.py --source 0          # webcam
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).parent
DEFAULT_WEIGHTS = ROOT / "runs" / "train" / "weights" / "best.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="image/folder/video path, or webcam index like 0")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--conf", type=float, default=0.25, help="minimum confidence")
    args = ap.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(args.weights)

    # stream=True yields results one frame at a time (low memory for video/webcam)
    for result in model.predict(source, conf=args.conf, show=True, save=True, stream=True,
                                project=str(ROOT / "runs"), name="detect", exist_ok=True):
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"{result.names[int(box.cls)]}: {float(box.conf):.2f} "
                  f"[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")


if __name__ == "__main__":
    main()
