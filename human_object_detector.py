#!/usr/bin/env python3

"""
A simple ROS 2 node that subscribes to /cam_1/color/image_raw and performs
human detection using the Ultralytics YOLO model.
"""

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"  # Ensures proper X11 usage for cv2.imshow in some environments

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from typing import Tuple, Optional

# ---- Add the Ultralytics YOLO import ----
from ultralytics import YOLO


def gen_bbox_human(img: np.ndarray, model: YOLO, acc: float = 0.8) -> np.ndarray:
    """
    Use the Ultralytics YOLO model to detect humans in an image.

    Args:
        img (np.ndarray): (H, W, 3) BGR or RGB image (OpenCV format).
        model (YOLO): Pre-loaded YOLO model for inference.
        acc (float): Confidence threshold.

    Returns:
        np.ndarray: Array of bounding boxes in [x1, y1, x2, y2] format.
                    If no detections, returns an empty array.
    """
    # Run YOLO inference
    results = model.predict(
        img,
        conf=acc,           # confidence threshold
        classes=[0],        # Only detect humans ('class 0' in COCO)
        save=False,
        save_txt=False,
        show_conf=False
    )

    # Extract bounding boxes (in xyxy format)
    if len(results) > 0 and len(results[0].boxes) > 0:
        bbox = results[0].boxes
        pos_res = bbox.xyxy.cpu().numpy()  # shape: Nx4 or Nx6 (x1,y1,x2,y2,conf,class)
        # If it has more columns, slice the first 4
        if pos_res.shape[1] > 4:
            pos_res = pos_res[:, :4]
    else:
        pos_res = np.array([])

    return pos_res


def draw_rect(
    img: np.ndarray,
    pos_res: np.ndarray,
    if_centroid: bool = True,
    if_coords: bool = True
) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    """
    Draw bounding boxes (and optionally centroids) for each detection.

    Args:
        img (np.ndarray): (H, W, 3) image in BGR format.
        pos_res (np.ndarray): Nx4 array of bounding boxes [x1, y1, x2, y2].
        if_centroid (bool): Whether to draw centroid circles.
        if_coords (bool): Whether to draw text with (x, y) coordinates.

    Returns:
        Tuple[np.ndarray, Optional[Tuple[int, int]]]:
            - Updated image with bounding boxes.
            - The last centroid found (centroid_x, centroid_y) or None if no detections.
    """
    color = (255, 0, 255)  # Purple
    thickness = 2
    centroid_res = None

    for pos in pos_res:
        x1, y1, x2, y2 = pos

        start_point = (int(np.ceil(x1)), int(np.ceil(y1)))
        end_point = (int(np.ceil(x2)), int(np.ceil(y2)))

        cv2.rectangle(img, start_point, end_point, color, thickness)

        # Compute centroid
        centroid_x = int(np.ceil((x1 + x2) / 2))
        centroid_y = int(np.ceil((y1 + y2) / 2))
        centroid_res = (centroid_x, centroid_y)

        if if_centroid:
            cv2.circle(img, (centroid_x, centroid_y), radius=5, color=color, thickness=-1)

        if if_coords:
            text = f"({centroid_x}, {centroid_y})"
            text_position = (int(np.ceil(centroid_x)), int(np.ceil(y1) - 10))
            cv2.putText(
                img, text, text_position,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=color, thickness=1, lineType=cv2.LINE_AA
            )

    return img, centroid_res


class HumanObjectDetector(Node):
    def __init__(self):
        super().__init__('human_object_detector')
        # Subscription
        self.subscription = self.create_subscription(
            Image,
            '/cam_1/color/image_raw',
            self.image_callback,
            10
        )
        self.subscription  # prevent unused variable warning

        # ROS <-> OpenCV image bridge
        self.bridge = CvBridge()

        # ---- Initialize YOLO model here ----
        self.model = YOLO("yolo11n.pt", verbose=False)  # Make sure "yolo11n.pt" is accessible
        self.get_logger().info("YOLO-based Human Detector Initialized.")

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV (BGR) image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to convert image: {str(e)}")
            return

        # Detect humans using YOLO
        pos_res = gen_bbox_human(cv_image, self.model, acc=0.6)

        # Draw bounding boxes (and optionally centroids)
        cv_image, centroid_res = draw_rect(cv_image, pos_res, if_centroid=True, if_coords=False)

        # For demonstration, you can log the centroid if detected
        if centroid_res:
            cx, cy = centroid_res
            self.get_logger().info(f"Detected human centroid at: ({cx}, {cy})")

        # Display image in an OpenCV window
        cv2.imshow("YOLO Human Detection", cv_image)
        cv2.waitKey(1)  # Required to update the OpenCV window


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

