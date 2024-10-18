# Training Your Own Model with YOLOv8

In this section, we'll guide you through the process of training your own object detection model using YOLOv8. By training your own model, you can detect custom objects tailored to your specific application.

## Prerequisites

Before you start, ensure you have the following:

- **Python 3.7 or later**
- **pip** (Python package manager)
- **Ultralytics YOLOv8** installed (`pip install ultralytics`)
- **Basic understanding of command-line operations**
- **A dataset** prepared in the correct format (we'll cover this)

## Step 1: Obtain Your Dataset

You can find datasets on many different websites or you can create your own.

Datasets:
- [roboflow universe](https://universe.roboflow.com)


# Training Your Own Model with YOLOv8
In this section, we'll guide you through the process of training your own object detection model using YOLOv8. By training your own model, you can detect custom objects tailored to your specific application.

## Prerequisites

Before you start, ensure you have the following:

- **Python 3.7 or later**
- **pip** (Python package manager)
- **Ultralytics YOLOv8** installed (`pip install ultralytics`)
- **Basic understanding of command-line operations**
- **A dataset** prepared in the correct format (we'll cover this)

## Step 1: Obtain Your Dataset

You can find datasets on many different websites or you can create your own.

Datasets:
- [roboflow universe](https://roboflow.com/annotate)
- [kaggle](https://www.kaggle.com/)

If you want to create your own you will need to obtain images you want to train on and annotate them.

Annotation Websites:
- [cvat](https://www.cvat.ai/post/yolo)
- [roboflow](https://roboflow.com/annotate)

## Step 2: Organize Your Dataset

YOLOv8 expects datasets in a specific format. You'll need to organize your images and annotations accordingly.

### 2.1. Organize Your Images
- **Training Images**: Place all your training images in a folder (Common to have around 80% of your images for training), e.g., `images/train/`
- **Validation Images**: Place all your validation images in a folder (Common to have around 20% of your images for validation), e.g., `images/val/`

### 2.2. Create Annotations in YOLO Format
For each image, create a corresponding text file containing annotations in YOLO format:

- Each line in the annotation file represents one object.

- The format is:

    ```bash
    <class_id> <x_center> <y_center> <width> <height>
    ```
    All coordinates are normalized between 0 and 1, relative to the image width and height.

Example annotation (for one object):
```bash
0 0.492 0.450 0.200 0.300
```

- 0: Class ID (integer starting from 0)
- 0.492: X center coordinate
- 0.450: Y center coordinate
- 0.200: Width of the bounding box
- 0.300: Height of the bounding box

### 2.3. Directory Structure

Your dataset directory should look like this:

```bash
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image3.jpg
│       ├── image4.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    └── val/
        ├── image3.txt
        ├── image4.txt
        └── ...
```

### 2.4. Create a Data Configuration File

Create a YAML file (e.g., data.yaml) that describes your dataset:

```bash
path: datasets  # root dataset directory
train: images/train  # training images (relative to 'path')
val: images/val  # validation images (relative to 'path')

nc: 2  # number of classes
names: ['class1', 'class2']  # class names
```

- nc: Number of classes in your dataset.
- names: List of class names.

Save this file as `data.yaml` in your project directory.

## Step 3: Train the Model

Use the `yolo` command-line tool to start training or make a python file.

### Basic Training Command

```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

- `data=data.yaml`: Path to your data configuration file.
- `model=yolov8n.pt`: Pre-trained model to start with (nano version).
- `epochs`=100: Number of training epochs.
- `imgsz`=640: Image size for training.

```python
from ultralytics import YOLO

# Load YOLOv8n (nano) model
model = YOLO('yolov8n.pt')  # Pre-trained model

# Train the model
model.train(
    data='path/to/data.yaml',  # Path to dataset (in YOLO format)
    epochs=100,                # Number of epochs to train
    batch=16,                  # Batch size
    seed=1,
    imgsz=640,                 # Image size for training
    project='runs/train',       # Directory to save training runs
    name='yolov8n_custom'      # Name for this specific run
)

# Perform inference on an image
results = model('path/to/image.jpg')

# Display results
results.show()
```

### Additional Training Options

- Batch Size: Number of images per batch. Larger batch sizes require more VRAM (Different from normal RAM). For more information on batch size: [ultralytics - batch size](https://www.ultralytics.com/glossary/batch-size) 
```bash
batch=32
```

- Device Selection: Choose CPU or GPU.
```bash
device=0  # Use GPU 0
device='cpu'  # Use CPU
```

### Full Example Command
```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=32 device=0
```

## Step 4: Run Inference with Your Trained Model

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source='path/to/your/test/image.jpg'
```

Predictions are saved in `runs/detect/predict/`.


w.com/)
- [kaggle](https://www.kaggle.com/)

If you want to create your own you will need to obtain images you want to train on and annotate them.

Annotation Websites:
- [cvat](https://www.cvat.ai/post/yolo)
- [roboflow](https://roboflow.com/annotate)

## Step 2: Organize Your Dataset

YOLOv8 expects datasets in a specific format. You'll need to organize your images and annotations accordingly.

### 2.1. Organize Your Images
- **Training Images**: Place all your training images in a folder (Common to have around 80% of your images for training), e.g., `images/train/`
- **Validation Images**: Place all your validation images in a folder (Common to have around 20% of your images for validation), e.g., `images/val/`

### 2.2. Create Annotations in YOLO Format
For each image, create a corresponding text file containing annotations in YOLO format:

- Each line in the annotation file represents one object.

- The format is:

    ```bash
    <class_id> <x_center> <y_center> <width> <height>
    ```
    All coordinates are normalized between 0 and 1, relative to the image width and height.

Example annotation (for one object):
```bash
0 0.492 0.450 0.200 0.300
```

- 0: Class ID (integer starting from 0)
- 0.492: X center coordinate
- 0.450: Y center coordinate
- 0.200: Width of the bounding box
- 0.300: Height of the bounding box

### 2.3. Directory Structure

Your dataset directory should look like this:

```bash
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image3.jpg
│       ├── image4.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    └── val/
        ├── image3.txt
        ├── image4.txt
        └── ...
```

### 2.4. Create a Data Configuration File

Create a YAML file (e.g., data.yaml) that describes your dataset:

```bash
path: datasets  # root dataset directory
train: images/train  # training images (relative to 'path')
val: images/val  # validation images (relative to 'path')

nc: 2  # number of classes
names: ['class1', 'class2']  # class names
```

- nc: Number of classes in your dataset.
- names: List of class names.

Save this file as `data.yaml` in your project directory.

## Step 3: Train the Model

Use the `yolo` command-line tool to start training or make a python file.

### Basic Training Command

```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

- `data=data.yaml`: Path to your data configuration file.
- `model=yolov8n.pt`: Pre-trained model to start with (nano version).
- `epochs`=100: Number of training epochs.
- `imgsz`=640: Image size for training.

```python
from ultralytics import YOLO

# Load YOLOv8n (nano) model
model = YOLO('yolov8n.pt')  # Pre-trained model

# Train the model
model.train(
    data='path/to/data.yaml',  # Path to dataset (in YOLO format)
    epochs=100,                # Number of epochs to train
    batch=16,                  # Batch size
    seed=1,
    imgsz=640,                 # Image size for training
    project='runs/train',       # Directory to save training runs
    name='yolov8n_custom'      # Name for this specific run
)

# Perform inference on an image
results = model('path/to/image.jpg')

# Display results
results.show()
```

### Additional Training Options

- Batch Size: Number of images per batch. Larger batch sizes require more VRAM (Different from normal RAM). For more information on batch size: [ultralytics - batch size](https://www.ultralytics.com/glossary/batch-size) 
```bash
batch=32
```

- Device Selection: Choose CPU or GPU.
```bash
device=0  # Use GPU 0
device='cpu'  # Use CPU
```

### Full Example Command
```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=32 device=0
```

## Step 4: Run Inference with Your Trained Model

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source='path/to/your/test/image.jpg'
```

Predictions are saved in `runs/detect/predict/`.


