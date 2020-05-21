# Keras-TF2-YOLOv3
Keras  with Tensorflow2 - YOLOv3 - Best on GPU 
Tested with 

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K).


---

## Quick Start

1. Download YOLOv3 weights from [YOLO website](http://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model.
3. Modify VideoCamera source   
   1. (cv2.VideoCapture("rtsp://x.x.x.x)) - IP Camera
   2.  (cv2.VideoCapture(0) - Webcam
4. Run File

```
wget https://pjreddie.com/media/files/yolov3.weights
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
python realtime-single-cam.py
python realtime-multi-cam.py
```

For Tiny YOLOv3, just do in a same but alter just specify model path and anchor path with --model model_file and --anchors anchor_file.

---

## Things to know

1. OpenCV VideoGet function is in a thread for improved performance
2. OpenCV imshow in thread was not behaving
3. Keras model predict in thread did not show performance improvement
4. Python Queues seem to generate less latency then simple procedural while loop


---

## Some issues to know

1. Parameterized more of the code
2. Clean up and modularize files. 
3. The test environment is
    - Python 3.5.
    - Keras 2.3.1
    - tensorflow 2.1.0-rc1
    - GPU GTX 1060