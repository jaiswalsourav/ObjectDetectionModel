# Object Detection with YOLO

Train a custom object detector with [Ultralytics YOLO](https://docs.ultralytics.com) and use it on images, videos or a webcam.

The current model detects **5 flower types**: daisy, dandelion, rose, sunflower and tulip. To detect other objects, swap in a different dataset (see [Train on your own objects](#train-on-your-own-objects)).

---

## Quick start

```powershell
.venv\Scripts\activate
python detect.py --source flower.jpg      # an image
python detect.py --source 0               # webcam (press q to quit)
```

---

## Project structure

```
ObjectDetection/
├── config.yaml              # class names + training settings  ← edit this
├── prepare_dataset.py       # splits raw/ into dataset/ (train/val)
├── train.py                 # trains the model
├── detect.py                # runs the trained model
├── requirements.txt         # Python packages
├── README.md
├── .gitignore
│
│   ── Everything below is NOT in git (see .gitignore) ──
│
├── .venv/                   # Python virtual environment
├── yolo11n.pt               # pretrained base model (auto-downloaded, training starts from it)
├── weights/
│   └── yolo26n.pt           # auto-downloaded by Ultralytics for a GPU check (AMP), not needed by you
│
├── raw/                     # YOUR INPUT DATA
│   ├── images/              #   photos (.jpg/.png), directly in this folder
│   └── labels/              #   YOLO .txt label files, same names as the images
│
├── dataset/                 # created by prepare_dataset.py (rebuilt each run, don't edit)
│   ├── data.yaml            #   dataset config used by train.py
│   ├── images/
│   │   ├── train/           #   80% of photos
│   │   └── val/             #   20% of photos
│   └── labels/
│       ├── train/
│       ├── val/
│       ├── train.cache      #   speed-up cache made by Ultralytics (safe to delete)
│       └── val.cache
│
├── download/                # optional: extracted dataset zips before moving into raw/
│
└── runs/                    # OUTPUTS
    ├── train_log.txt        #   console log of the last training run
    ├── train/               #   created by train.py
    │   ├── weights/
    │   │   ├── best.pt      #   ← THE TRAINED MODEL, use this
    │   │   └── last.pt      #     last epoch (only for resuming training)
    │   ├── results.csv      #     metrics per epoch
    │   ├── results.png      #     loss / mAP charts (written at the end)
    │   ├── confusion_matrix.png
    │   ├── labels.jpg       #     class balance and box-size overview
    │   ├── train_batch*.jpg #     sample training images with labels
    │   └── args.yaml        #     exact settings used
    └── detect/              #   created by detect.py (images/videos with boxes)
```

---

## 1. Setup

Requires Python 3.10+ and, optionally, an NVIDIA GPU.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Check that the GPU is detected:

```powershell
python -c "import torch; print(torch.cuda.is_available())"   # True = GPU, False = CPU only
```

> **Low disk space?** The CUDA build of PyTorch needs about 5 GB. Add `--no-cache-dir` to the `pip install` commands, or run `pip cache purge` afterwards.

---

## 2. Get labeled data

The model learns from images where every object has a box drawn around it.

### Option A: Download a labeled dataset (easiest)

1. Find a dataset on [Roboflow Universe](https://universe.roboflow.com) or [Kaggle](https://www.kaggle.com/datasets).
2. Download it in **YOLOv11** or **YOLOv8** format. Other formats such as COCO JSON or Pascal VOC won't work.
3. Copy all the images into `raw/images/` and all the `.txt` files into `raw/labels/`. If the download is split into `train/`, `valid/` and `test/` folders, merge them all.
4. Copy the class names from the dataset's `data.yaml` into `config.yaml`, **in the same order**.

### Option B: Label your own photos

1. Put your photos in `raw/images/`.
2. Draw boxes with [Roboflow](https://roboflow.com) (in the browser) or [Label Studio](https://labelstud.io).
3. Export in **YOLO** format and put the `.txt` files in `raw/labels/`.

Tips for good results:
- 100+ images per class (more is better)
- varied backgrounds, lighting, angles and distances
- a few images with no objects in them

### Label file format

Each image has a `.txt` file with the same name, with one line per object:

```
<class_id> <x_center> <y_center> <width> <height>
0 0.512 0.430 0.210 0.180
```

Coordinates are fractions of the image size (0–1). `class_id` is the item's position in the `classes` list in `config.yaml`, counting from 0.

---

## 3. Configure

Edit [config.yaml](config.yaml):

```yaml
classes:            # must match the class ids in your label files
  - daisy           # 0
  - dandelion       # 1
  - rose            # 2
  - sunflower       # 3
  - tulip           # 4

model: yolo11n.pt   # n = nano (fast). Try yolo11s.pt for more accuracy
imgsz: 320          # training image size
batch: 16           # lower this if you get "CUDA out of memory"
epochs: 100
patience: 20        # stop early if no improvement for 20 epochs
val_ratio: 0.2      # 20% of photos held out for validation
```

---

## 4. Train

```powershell
python prepare_dataset.py
python train.py
```

- `prepare_dataset.py` splits the data 80/20 into training and validation sets. Augmented copies of the same photo (Roboflow's `_aug_N` / `.rf.` files) always go into the same set, so the validation score stays honest.
- `train.py` fine-tunes a pretrained YOLO model and saves the best version to `runs/train/weights/best.pt`.
- Progress charts and sample predictions are saved in `runs/train/` (`results.png`, `confusion_matrix.png`, …).

**Reading the score:** `mAP50` is the main accuracy metric, where 1.0 is perfect. Above 0.7 is good for most uses.

To resume a training run that was interrupted:

```powershell
yolo train resume model=runs/train/weights/last.pt
```

---

## 5. Detect

```powershell
python detect.py --source photo.jpg            # single image
python detect.py --source my_photos/           # folder of images
python detect.py --source video.mp4            # video file
python detect.py --source 0                    # webcam
python detect.py --source photo.jpg --conf 0.5 # only show confident detections
python detect.py --source photo.jpg --weights other_model.pt
```

Results with boxes drawn on them are saved to `runs/detect/`. Each detection is also printed:

```
rose: 0.91 [34, 50, 210, 288]
```

---

## Use the model in another project

Copy **`runs/train/weights/best.pt`**. It is the only file you need, and it includes the class names.

### Python

```powershell
pip install ultralytics
```

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model("photo.jpg", conf=0.25)

for box in results[0].boxes:
    name = results[0].names[int(box.cls)]     # "rose"
    conf = float(box.conf)                    # 0.91
    x1, y1, x2, y2 = box.xyxy[0].tolist()     # box corners in pixels
    print(name, conf, x1, y1, x2, y2)
```

### Without Python / PyTorch

Export the model to another format:

| Target | Command | Output |
|---|---|---|
| ONNX (C#, C++, Java, JS, OpenCV) | `yolo export model=best.pt format=onnx imgsz=320` | `best.onnx` |
| Android / mobile | `yolo export model=best.pt format=tflite imgsz=320` | `best_saved_model/*.tflite` |
| Intel CPU (OpenVINO) | `yolo export model=best.pt format=openvino imgsz=320` | `best_openvino_model/` |

Always export with the same `imgsz` used for training.

---

## Train on your own objects

1. Empty `raw/images/` and `raw/labels/`, then add your new data (see [step 2](#2-get-labeled-data)).
2. Update `classes` in `config.yaml`.
3. Set `imgsz` to roughly match your image size (`640` is the usual default).
4. Run `python prepare_dataset.py` and then `python train.py`.

Each training run overwrites `runs/train/`, so copy `best.pt` somewhere safe first if you want to keep the old model.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Lower `batch` (16 → 8 → 4) or `imgsz` in `config.yaml` |
| `torch.cuda.is_available()` is `False` | Reinstall PyTorch with the `--index-url .../cu128` command from Setup |
| `No space left on device` | Run `pip cache purge` and use `--no-cache-dir` |
| `No images found in raw/images` | Check that the images are directly inside `raw/images/`, not in a subfolder |
| Detections have the wrong names | The order of `classes` in `config.yaml` doesn't match the label files |
| Low accuracy | Add more varied images, train longer, or use `yolo11s.pt` |
| Webcam doesn't open | Try `--source 1`, or close other apps that use the camera |

---

## Dataset credit

Flower data: [Detection of Flowers](https://universe.roboflow.com/data-geqjj/detection-of-flowers-uj9zz/dataset/6) on Roboflow Universe, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
