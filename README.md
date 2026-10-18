# Custom Object Detection (YOLO)

## 1. Setup
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # should print True
```
If no PyTorch wheel exists for your Python version, install Python 3.12 and create the venv with `py -3.12 -m venv .venv`.

## 2. Label your images
1. Put your photos in `raw/images/`.
2. Draw bounding boxes with a labeling tool, e.g. [Label Studio](https://labelstud.io) (`pip install label-studio`, then `label-studio`) or Roboflow (web).
3. Export in **YOLO** format and put the `.txt` files in `raw/labels/` (same filenames as the images).
4. Edit `classes` in `config.yaml` so it matches the class order used by the labeling tool (`classes.txt`).

Tips: aim for 100+ images per class, with varied backgrounds, lighting, angles and distances. Include some images with no objects in them.

## 3. Train
```powershell
python prepare_dataset.py
python train.py
```
Best weights are saved to `runs/train/weights/best.pt`. If you get a CUDA out-of-memory error, lower `batch` or `imgsz` in `config.yaml`.

## 4. Detect
```powershell
python detect.py --source path\to\image.jpg
python detect.py --source 0        # webcam
```
