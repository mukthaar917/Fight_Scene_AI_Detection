AI-Based CCTV Violence & Fight Scene Detection

An automated, real-time surveillance pipeline built with YOLOv8 and OpenCV designed to detect altercations and violence in outdoor CCTV feeds while minimizing false positives from normal pedestrian activity.


Features

Altercation Detection: Custom-weighted YOLOv8 model combined with a secondary tracker to accurately flag fighting individuals.
False-Positive Suppression: Excludes regular pedestrians, cleaning staff, and swaying background objects (banners/flags).
Clean UI / Overlay:
  - Top-right non-intrusive warning alert (Warning: Violence/ Fight Detected).
  - Embedded CCTV timestamp removal via letterbox cropping.
  - Synchronized, dynamic running timestamp overlay at the bottom.
GPU Acceleration: NVIDIA CUDA-enabled inference pipeline tested on RTX 3050 Laptop GPU.

Tech Stack

- Language: Python 3.12
- Framework: Ultralytics YOLOv8
- Computer Vision: OpenCV (cv2)
- Acceleration: PyTorch (CUDA 12.1)

Getting Started

1. Clone the Repository

git clone [https://github.com/mukthaar917/Fight_Scene_AI_Detection.git](https://github.com/mukthaar917/Fight_Scene_AI_Detection.git)
cd Fight_Scene_AI_Detection