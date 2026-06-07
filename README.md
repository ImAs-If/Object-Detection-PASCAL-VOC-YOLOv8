# 🎯 Object Detection on PASCAL VOC 2012 using YOLOv8s

An end-to-end object detection pipeline that automatically converts **PASCAL VOC 2012 XML annotations** into **YOLO format**, organizes the dataset into the required directory structure, and trains a high-performance **YOLOv8s** model using the Ultralytics framework.

---

## 📌 Overview

This project simplifies the process of training a YOLOv8 object detection model on the PASCAL VOC dataset by providing:

* Automatic VOC XML → YOLO TXT annotation conversion
* Dataset restructuring for Ultralytics YOLO
* YOLOv8s model training and validation
* Inference and prediction utilities
* Performance visualization and evaluation

---

## 🗂️ Dataset

**Dataset:** PASCAL VOC 2012

### Classes (20 Object Categories)

```python
[
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
```

### Dataset Statistics

| Split      |     Images |
| ---------- | ---------: |
| Training   |      5,717 |
| Validation |      5,823 |
| **Total**  | **11,540** |

---

## 📂 Project Structure

After conversion, the dataset is automatically organized as:

```text
yolo_voc/
├── train/
│   ├── images/
│   └── labels/
│
└── val/
    ├── images/
    └── labels/
```

```text
train/images  → 5,717 images
train/labels  → 5,717 annotation files

val/images    → 5,823 images
val/labels    → 5,823 annotation files
```

---

## ⚙️ Requirements

### Software

* Python 3.10+
* PyTorch
* Ultralytics YOLOv8
* OpenCV
* Matplotlib
* Pandas

### Hardware (Recommended)

* NVIDIA Tesla T4 GPU or higher
* CUDA-enabled environment

---

## 🛠️ Installation

```bash
pip install -q ultralytics opencv-python matplotlib pandas torch
```

---

## 💻 Configuration

```python
class Config:
    VOC_INPUT = "/path/to/VOC2012_train_val"
    WORK_DIR = "./working"
    OUTPUT_DIR = f"{WORK_DIR}/yolo_voc"

    # Training Hyperparameters
    EPOCHS = 30
    BATCH_SIZE = 16
    IMG_SIZE = 640
    MODEL = "yolov8s.pt"

    # VOC Classes
    CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
```

---

## 🚀 Training Pipeline

### Step 1: Convert VOC Annotations to YOLO Format

```python
convert_voc_to_yolo(
    config.VOC_INPUT,
    config.OUTPUT_DIR,
    config.CLASSES
)
```

### Step 2: Load YOLOv8s

```python
from ultralytics import YOLO

model = YOLO(config.MODEL)
```

### Step 3: Train the Model

```python
results = model.train(
    data=f"{config.WORK_DIR}/voc.yaml",
    epochs=config.EPOCHS,
    imgsz=config.IMG_SIZE,
    batch=config.BATCH_SIZE,
    device=0
)
```

---

## 🔍 Inference

### Load the Trained Model

```python
from ultralytics import YOLO

model = YOLO("best.pt")
```

### Predict on an Image

```python
results = model("image.jpg")

results[0].show()
results[0].save(filename="result.jpg")
```

### Set a Confidence Threshold

```python
results = model("image.jpg", conf=0.5)
```

### Extract Bounding Box Information

```python
for r in results:
    for box in r.boxes:
        print(
            f"Class: {int(box.cls[0])} | "
            f"Confidence: {float(box.conf[0]):.2f} | "
            f"BBox: {box.xyxy[0].tolist()}"
        )
```

---

## 📊 Training Output

The training process generates:

* Precision Curve
* Recall Curve
* mAP@50
* mAP@50-95
* Loss Curves
* Validation Metrics
* Prediction Visualizations

These results can be found in the generated `runs/detect/train/` directory.

---

## 🎯 Features

✅ Automatic VOC XML to YOLO TXT conversion

✅ YOLOv8-compatible dataset structure

✅ GPU-accelerated training

✅ Easy inference and prediction

✅ Support for all 20 VOC classes

✅ Performance visualization

✅ Reproducible training pipeline

---

## 📈 Model

| Model   | Input Size | Framework          |
| ------- | ---------- | ------------------ |
| YOLOv8s | 640×640    | Ultralytics YOLOv8 |

---

## 🤝 Acknowledgments

* Ultralytics for the YOLOv8 training framework.
* PASCAL VOC Community for the benchmark object detection dataset.
* PyTorch for deep learning support.

---

## 📜 License

This project is intended for educational and research purposes.

Please follow the licensing terms of the PASCAL VOC dataset and Ultralytics YOLO framework when using this project.

---

## 👨‍💻 Author

**Md. Asif Ahmed**

Bachelor of Computer Science (BCS)

Interested in:

* Machine Learning
* Computer Vision
* Deep Learning
* AI Research
