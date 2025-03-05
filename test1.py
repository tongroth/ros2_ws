#!/usr/bin/env python3

"""
A simple ROS 2 node that subscribes to /cam_1/color/image_raw and /cam_1/depth/image_rect_raw,
performs human detection using the Ultralytics YOLO model, and overlays on each detected bounding box
the distance (in meters) between the robot and the detected human as well as the angle (in degrees)
between the image center and the bounding box centroid.
"""

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"  # Ensure proper X11 usage for cv2.imshow

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from typing import Tuple, Optional

# ---- Import the Ultralytics YOLO model ----
from ultralytics import YOLO


def gen_bbox_human(img: np.ndarray, model: YOLO, acc: float = 0.8) -> np.ndarray:
    """
    Use the Ultralytics YOLO model to detect humans in an image.

    Args:
        img (np.ndarray): Input image (BGR or RGB).
        model (YOLO): Pre-loaded YOLO model.
        acc (float): Confidence threshold.

    Returns:
        np.ndarray: Array of bounding boxes [x1, y1, x2, y2]. Returns an empty array if no detection.
    """
    results = model.predict(
        img,
        conf=acc,
        classes=[0],  # Class 0 for human (COCO)
        save=False,
        save_txt=False,
        show_conf=False
    )
    if len(results) > 0 and len(results[0].boxes) > 0:
        bbox = results[0].boxes
        pos_res = bbox.xyxy.cpu().numpy()  # shape: Nx4 (or more columns; we use first 4)
        if pos_res.shape[1] > 4:
            pos_res = pos_res[:, :4]
    else:
        pos_res = np.array([])
    return pos_res


def draw_rect(
    img: np.ndarray,
    pos_res: np.ndarray,
    if_centroid: bool = True,
    if_coords: bool = True,
    distance: Optional[float] = None,
    angle: Optional[float] = None
) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    """
    Draw bounding boxes and optionally centroids on the image. If distance and angle are provided,
    overlay them as text near the bounding box.

    Args:
        img (np.ndarray): Image in BGR.
        pos_res (np.ndarray): Nx4 bounding boxes.
        if_centroid (bool): Whether to draw centroid circles.
        if_coords (bool): Whether to display coordinate text.
        distance (Optional[float]): Distance in meters.
        angle (Optional[float]): Angle in degrees.

    Returns:
        Tuple containing the updated image and the centroid (of the last detection) or None.
    """
    color = (255, 0, 255)  # Purple
    thickness = 2
    centroid_res = None

    for pos in pos_res:
        x1, y1, x2, y2 = map(int, pos)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        centroid_x = int((x1 + x2) / 2)
        centroid_y = int((y1 + y2) / 2)
        centroid_res = (centroid_x, centroid_y)

        if if_centroid:
            cv2.circle(img, (centroid_x, centroid_y), 5, color, -1)

        if if_coords:
            coord_text = f"({centroid_x}, {centroid_y})"
            cv2.putText(img, coord_text, (centroid_x, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if distance is not None and angle is not None:
            info_text = f"Dist: {distance:.2f} m, Angle: {angle:.1f} deg"
            cv2.putText(img, info_text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return img, centroid_res


class HumanObjectDetector(Node):
    def __init__(self):
        super().__init__('human_object_detector')
        # Subscribe to the color image topic
        self.color_subscription = self.create_subscription(
            Image,
            '/cam_1/color/image_raw',
            self.image_callback,
            10
        )
        # Subscribe to the depth image topic
        self.depth_subscription = self.create_subscription(
            Image,
            '/cam_1/depth/image_rect_raw',
            self.depth_callback,
            10
        )
        self.bridge = CvBridge()
        self.depth_image = None

        # Initialize YOLO model (ensure that "yolo11n.pt" is in your working directory or provide full path)
        self.model = YOLO("yolo11n.pt", verbose=False)
        self.get_logger().info("YOLO-based Human Detector Initialized.")

    def depth_callback(self, msg: Image):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Run YOLO detection on the color image
        pos_res = gen_bbox_human(cv_image, self.model, acc=0.6)

        distance = None
        angle = None
        centroid = None

        if pos_res.size != 0:
            # For simplicity, use the first detected bounding box
            x1, y1, x2, y2 = pos_res[0]
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            # Compute distance using the depth image at the centroid
            if self.depth_image is not None:
                try:
                    depth_value = self.depth_image[centroid[1], centroid[0]]
                    if depth_value > 0:
                        distance = depth_value / 1000.0  # Convert mm to m
                except Exception as e:
                    self.get_logger().error(f"Error reading depth at centroid: {e}")

            # Compute horizontal angle relative to the image center
            image_width = cv_image.shape[1]
            center_x = image_width / 2.0
            horizontal_fov = 60.0  # Assumed horizontal field of view in degrees
            angle = ((centroid[0] - center_x) / center_x) * (horizontal_fov / 2.0)

            self.get_logger().info(f"Detected centroid: {centroid}, Distance: {distance}, Angle: {angle}")

        # Draw bounding boxes with centroid and overlay distance/angle info if available
        cv_image, _ = draw_rect(cv_image, pos_res, if_centroid=True, if_coords=True, distance=distance, angle=angle)

        # Display the processed image in an OpenCV window
        cv2.imshow("YOLO Human Detection", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = HumanObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

