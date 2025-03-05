#!/usr/bin/env python3

from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("/home/roth/ros2_ws/yolov8n.pt")

# Export the model to OpenVINO format
model.export(format="openvino")

print("Model successfully converted to OpenVINO format!")

