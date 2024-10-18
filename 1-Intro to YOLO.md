# Welcome to the Black Knight project!

This page will introduce you to one of the tools used in the Black Knight project.

## What is YOLO?

**YOLO** (You Only Look Once) is a real-time object detection algorithm known for its speed and accuracy, capable of detecting multiple objects in images or video streams. It has gained popularity due to its efficiency in performing object detection tasks in applications like autonomous driving, surveillance, and industrial automation.

**Ultralytics** is a company that specializes in artificial intelligence (AI) and computer vision, particularly in advancing the YOLO family of models. They have developed and maintain **YOLOv5** and **YOLOv8**, which are open-source models optimized for ease of use, high performance, and seamless integration with various platforms.

https://docs.ultralytics.com/

---

## Getting Started with YOLOv8n

In this guide, we'll help you get **YOLOv8n** up and running so you can try it out for yourself. **YOLOv8n** is the nano version of YOLOv8, optimized for speed and resource efficiency, making it ideal for beginners and applications where computational resources are limited.

### Prerequisites

Before you begin, make sure you have the following installed:

- **Python 3.7 or later** - https://www.python.org/downloads/
- **pip** (Python package manager)
- **Basic understanding of command-line operations**

### Step 1: Install Ultralytics YOLOv8

You can install YOLOv8 directly using `pip`:

```
!pip install ultralytics
```

### Step 2: Verify the Installation

```
!yolo
```

This should display the help information for the YOLO command-line interface.

### Step 3: Run YOLOv8n on a Sample Image

The following code will download the `yolov8n.pt` model (~6MB), the source image, and create a `runs/` directory in your current working directory.

```bash
!yolo detect predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```

This command does the following:

 - `detect predict`: Runs the detection task in prediction mode.
 - `model=yolov8n.pt`: Specifies the pre-trained YOLOv8n model weights to use.
 - `source='https://ultralytics.com/images/bus.jpg'`: Specifies the input image for object detection.

After running the command, the output image with detected objects will be saved in the runs/detect/predict directory. Open the image to see the results.

## Exploring Other Modes in YOLOv8

YOLOv8 supports multiple tasks and operations beyond object detection. You can experiment with these parameters and tasks:

`conf=0.25`: confidence - How confident do you want the model to be for it to show you the detection?

`device=cpu/gpu`: Do you want to use you cpu or gpu? (A dedicated gpu is usually faster)

`show=True`: Do you want to see the video or do you want to just see the command line list the classes?

`classes=0,1,2,3`: What classes do you want to detect for? (Just people `classes=0`)

https://docs.ultralytics.com/modes/predict/#inference-arguments

### Tasks:

1. **Detection (`detect`)**

   Object detection involves identifying and locating objects within an image or video. YOLOv8 excels at real-time object detection tasks.

2. **Classification (`classify`)**

   Image classification assigns a label or category to an entire image. YOLOv8 can be used for image classification tasks using its classification models.

   ```
   !yolo classify predict model=yolov8n-cls.pt source=/path/to/img/
   ```

3. **Segmentation (`segment`)**

   Instance segmentation not only detects objects but also creates a pixel-wise mask for each detected object, allowing you to distinguish individual objects within the same class.

   ```
   !yolo segment predict model=yolov8n-seg.pt source=0 show=True
   ```

4. **Pose Estimation (`pose`)**

   Pose estimation predicts the pose of a person or object, typically represented as keypoints. YOLOv8 can perform pose estimation tasks.

   ```
    !yolo pose predict model=yolov8n-pose.pt source=0 show=True
   ```

5. **Oriented Bounding Boxes (`obb`)**

   Similar to detect, but with angled boxes for higher accuracy in an image

   ```
   !yolo obb predict model=yolov8n-obb.pt source=0 show=True
   ```
