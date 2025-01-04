# Real-Time Object Tracking System

## Overview
The **Real-Time Object Tracking System** is a Python-based project that integrates object detection and tracking to monitor objects in real-time using a webcam or video feed. It leverages the YOLOv3 deep learning model for detection and OpenCV trackers for tracking.

---

## Features
- Real-time object detection using YOLOv3.
- Object tracking using OpenCV trackers (e.g., CSRT, KCF).
- Configurable parameters via `config.yaml`.
- Modular and extendable code structure.

---

## Project Structure
object-tracking-system/
│
├── data/                 
│   ├── coco.names        
│   └── yolov3.weights    
│
├── src/                  
│   ├── detection.py      
│   ├── tracking.py       
│   └── main.py           
│
├── config.yaml           
├── requirements.txt      
└── README.md             

You need to download yolov3.weights Dataset Or you can use your own dataset.
