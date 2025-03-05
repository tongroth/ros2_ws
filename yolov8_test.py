#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ultralytics import YOLO
import cv2
import os

class YOLOv8OpenVINO(Node):
    def __init__(self):
        super().__init__('yolov8_openvino_test_node')

        # Paths
        model_path = "/home/roth/ros2_ws/yolov8n_openvino_model"  # Change to OpenVINO exported model
        image_path = "/home/roth/ros2_ws/bus.jpg"

        # Check files
        if not os.path.exists(image_path):
            self.get_logger().error(f"Image not found: {image_path}")
            return

        self.get_logger().info("Loading YOLOv8 OpenVINO model...")
        self.model = YOLO(model_path)

        self.get_logger().info(f"Processing image: {image_path}")
        image = cv2.imread(image_path)

        # Run YOLOv8 inference
        results = self.model(image)

        # Display results
        for result in results:
            result.show()

        self.get_logger().info("YOLOv8 OpenVINO inference complete.")

def main(args=None):
    rclpy.init(args=args)
    node = YOLOv8OpenVINO()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
