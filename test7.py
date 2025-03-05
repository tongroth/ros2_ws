#!/usr/bin/env python3
"""
A ROS 2 node that subscribes to /cam_1/color/image_raw and /scan,
detects a human using the Ultralytics YOLO11 model, uses LIDAR to measure
the distance to the detected human, and computes the lateral offset from the image.
Two PID controllers are used to generate TwistStamped commands for a mecanum drive robot
to follow the human at a fixed distance of 1.5 meters.
The node overlays the measured distance on the detection.
"""

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"  # Ensures proper X11 usage for cv2.imshow

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import TwistStamped
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from typing import Tuple, Optional

from ultralytics import YOLO

# ---------------------------
# Simple PID Controller Class
# ---------------------------
class PID:
    def __init__(self, Kp: float, Ki: float, Kd: float, setpoint: float = 0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.last_error = None

    def update(self, measurement: float, dt: float) -> float:
        error = self.setpoint - measurement
        self.integral += error * dt
        derivative = 0.0 if self.last_error is None else (error - self.last_error) / dt
        self.last_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

# ---------------------------
# YOLO Detection Functions
# ---------------------------
def gen_bbox_human(img: np.ndarray, model: YOLO, acc: float = 0.8) -> np.ndarray:
    results = model.predict(
        img,
        conf=acc,
        classes=[0],  # Detect only humans (COCO class 0)
        save=False,
        save_txt=False,
        show_conf=False
    )
    if len(results) > 0 and len(results[0].boxes) > 0:
        bbox = results[0].boxes
        pos_res = bbox.xyxy.cpu().numpy()  # Nx4 bounding boxes [x1, y1, x2, y2]
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
    angle: Optional[float] = None  # kept for potential future use
) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    """
    Draw bounding boxes on the image. Overlays the measured distance (in meters) on the box.
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
        if distance is not None:
            info_text = f"Distance: {distance:.2f} m"
            cv2.putText(img, info_text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img, centroid_res

# ---------------------------
# Human Follower Node for Mecanum Drive using YOLO11 & LIDAR
# ---------------------------
class HumanFollower(Node):
    def __init__(self):
        super().__init__('human_follower')
        # Subscribe to color image and LaserScan topics
        self.color_sub = self.create_subscription(
            Image, '/cam_1/color/image_raw', self.image_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        # Publisher for TwistStamped commands (for mecanum drive)
        self.cmd_pub = self.create_publisher(TwistStamped, '/mecanum_drive_controller/cmd_vel', 10)
        self.bridge = CvBridge()
        self.laser_scan = None
        self.last_time = self.get_clock().now()

        # Initialize YOLO11 model (make sure "yolo11n.pt" is accessible)
        self.model = YOLO("yolo11n.pt", verbose=False)
        self.get_logger().info("Human Follower node initialized with YOLO11.")

        # Desired following parameters
        self.desired_distance = 1.5         # meters (desired following distance)
        self.desired_lateral_offset = 0.0     # centered

        # PID controllers for distance and lateral (centering) control
        self.pid_distance = PID(Kp=0.5, Ki=0.0, Kd=0.1, setpoint=self.desired_distance)
        self.pid_lateral = PID(Kp=0.5, Ki=0.0, Kd=0.05, setpoint=self.desired_lateral_offset)

        # Threshold (in pixels) for considering the human roughly centered
        self.center_threshold = 50

    def laser_callback(self, msg: LaserScan):
        self.laser_scan = msg

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Resize image if necessary; here we use 640x480
            cv_image = cv2.resize(cv_image, (640, 480))
        except CvBridgeError as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        # Run YOLO11 detection on the color image to get bounding boxes
        pos_res = gen_bbox_human(cv_image, self.model, acc=0.6)

        measured_distance = None
        measured_lateral = None
        centroid = None
        error_angle_rad = None

        if pos_res.size != 0:
            # Use the first detected bounding box
            x1, y1, x2, y2 = pos_res[0]
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            # Compute horizontal error (in pixels) from the image center
            image_width = cv_image.shape[1]
            center_x = image_width / 2.0
            pixel_error = centroid[0] - center_x

            # Convert pixel error to an angle (radians); assume a horizontal FOV of 60°
            horizontal_fov_deg = 60.0
            horizontal_fov_rad = np.deg2rad(horizontal_fov_deg)
            angle_per_pixel = horizontal_fov_rad / image_width
            error_angle_rad = pixel_error * angle_per_pixel

            # Use the latest LIDAR scan to obtain the distance measurement at the detection angle.
            if self.laser_scan is not None:
                index = int((error_angle_rad - self.laser_scan.angle_min) / self.laser_scan.angle_increment)
                if 0 <= index < len(self.laser_scan.ranges):
                    measured_distance = self.laser_scan.ranges[index]
                    if measured_distance == float('Inf') or measured_distance == 0.0:
                        measured_distance = None

            # For small angles, approximate lateral offset (in meters)
            if measured_distance is not None:
                measured_lateral = measured_distance * error_angle_rad

            self.get_logger().info(f"Detected centroid: {centroid}, LIDAR distance: {measured_distance}, Lateral error: {measured_lateral}")

            # Compute PID outputs if a valid distance measurement is available
            linear_control = -self.pid_distance.update(measured_distance, dt) if measured_distance is not None else 0.0
            lateral_control = -self.pid_lateral.update(measured_lateral, dt) if measured_lateral is not None else 0.0

            # Optionally clamp the control outputs (here we use limits)
            max_linear = 0.5   # m/s
            max_lateral = 1.0  # m/s
            linear_cmd = max(-max_linear, min(max_linear, linear_control))
            lateral_cmd = max(-max_lateral, min(max_lateral, lateral_control))

            self.get_logger().info(f"PID outputs: Linear {linear_cmd:.2f} m/s, Lateral {lateral_cmd:.2f} m/s")

            # Publish TwistStamped command for mecanum drive
            twist_msg = TwistStamped()
            twist_msg.header.stamp = current_time.to_msg()
            twist_msg.twist.linear.x = linear_cmd
            twist_msg.twist.linear.y = lateral_cmd
            twist_msg.twist.linear.z = 0.0
            twist_msg.twist.angular.x = 0.0
            twist_msg.twist.angular.y = 0.0
            twist_msg.twist.angular.z = 0.0
            self.cmd_pub.publish(twist_msg)
        else:
            # No human detected: rotate slowly to search
            self.get_logger().info("No human detected: rotating to search...")
            twist_msg = TwistStamped()
            twist_msg.header.stamp = current_time.to_msg()
            twist_msg.twist.linear.x = 0.0
            twist_msg.twist.linear.y = 0.0
            twist_msg.twist.linear.z = 0.0
            twist_msg.twist.angular.x = 0.0
            twist_msg.twist.angular.y = 0.0
            twist_msg.twist.angular.z = 0.5  # rad/s
            self.cmd_pub.publish(twist_msg)

        # For visualization: overlay the bounding box and measured distance
        displayed_angle = np.rad2deg(error_angle_rad) if error_angle_rad is not None else None
        cv_image, _ = draw_rect(cv_image, pos_res, if_centroid=True, if_coords=True,
                                distance=measured_distance, angle=displayed_angle)
        cv2.imshow("Human Follower Detection", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = HumanFollower()
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
