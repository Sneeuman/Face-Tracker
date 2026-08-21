# YOLO + DeepFace Webcam Detector

A real-time object detection system that uses **YOLOv8** to detect people and animals through a live webcam feed, then runs **DeepFace** on any detected person to estimate their age. Detected objects are boxed and labeled directly on the video stream as it plays.

## Overview

This script opens a webcam feed, runs each frame through a YOLOv8 model, and filters detections down to a set of "living things" (people and common animals). Every detection gets a bounding box and a label. When the detected object is a person, the cropped region is passed to DeepFace for an age estimate, which is displayed alongside the box.

## Features

- Real-time detection via YOLOv8 (`yolov8n.pt`)
- Bounding boxes and class labels drawn live on the video feed
- Age estimation for detected people using DeepFace
- Configurable confidence threshold for detections
- Clean exit via the `Esc` key

## How It Works

1. **Camera setup** — Opens the default webcam (index `0`) and sets the capture resolution to 1280x720.
2. **Frame loop** — Continuously reads frames from the camera while the stream is open.
3. **Detection** — Each frame is passed to the YOLOv8 model with a confidence threshold of `0.3`.
4. **Filtering** — Detected classes are checked against a predefined list of living things: `person`, `dog`, `cat`, `bird`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe`.
5. **Age estimation** — If the detected class is `person`, the bounding box region is cropped and sent to DeepFace's `analyze()` function to estimate age.
6. **Rendering** — Bounding boxes and labels (and age, for people) are drawn on the frame using OpenCV.
7. **Display** — The annotated frame is resized and shown in a live window titled `Webcam feed`.
8. **Exit** — The loop runs until the `Esc` key is pressed, at which point the camera is released and all windows are closed.

## Requirements

```bash
pip install opencv-python
pip install ultralytics
pip install deepface
```

You'll also need the YOLOv8 nano weights file (`yolov8n.pt`), which `ultralytics` will download automatically on first run if it isn't already present in your working directory.

## Usage

```bash
python detector.py
```

- Make sure a webcam is connected and accessible at index `0`. If you're using an external camera, update the `cv2.VideoCapture()` index accordingly.
- Press **Esc** at any time to close the feed and release the camera.

## Configuration

| Parameter | Location | Default | Description |
|---|---|---|---|
| Camera index | `cv2.VideoCapture(0)` | `0` | Which camera device to use |
| Frame width | `CAP_PROP_FRAME_WIDTH` | `1280` | Capture resolution width |
| Frame height | `CAP_PROP_FRAME_HEIGHT` | `720` | Capture resolution height |
| Confidence threshold | `model(frame, conf=0.3)` | `0.3` | Minimum confidence for a detection to be shown |
| Display size | `cv2.resize()` | `1200x720` | Size of the output display window |

## Tech Stack

- **OpenCV** — video capture, frame processing, and rendering
- **Ultralytics YOLOv8** — object detection
- **DeepFace** — facial analysis / age estimation
