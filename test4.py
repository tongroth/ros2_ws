#!/usr/bin/env python3
"""
A ROS 2 node that subscribes to /cam_1/color/image_raw and /cam_1/depth/image_rect_raw,
detects a human using the Ultralytics YOLO model, computes the distance (in meters) and
lateral offset (converted from a horizontal pixel error) to the detected human, and uses PID controllers 
to compute TwistStamped commands for a mecanum drive robot to follow the human.
The node overlays the distance (in meters) on the detected bounding box.
"""

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"  # Ensures proper X11 usage for cv2.imshow

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
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
    Draw bounding boxes on the image. Overlays the distance (in meters) on the box.
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
        # Display only the distance on the bounding box (in meters)
        if distance is not None:
            info_text = f"Distance: {distance:.2f} m"
            cv2.putText(img, info_text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img, centroid_res

# ---------------------------
# Human Follower Node for Mecanum Drive
# ---------------------------
class HumanFollower(Node):
    def __init__(self):
        super().__init__('human_follower')
        # Subscriptions for color and depth images
        self.color_sub = self.create_subscription(
            Image,
            '/cam_1/color/image_raw',
            self.image_callback,
            10
        )
        self.depth_sub = self.create_subscription(
            Image,
            '/cam_1/detections',
            self.depth_callback,
            10
        )
        # Publisher for TwistStamped commands for the mecanum drive
        self.cmd_pub = self.create_publisher(TwistStamped, '/mecanum_drive_controller/cmd_vel', 10)
        self.bridge = CvBridge()
        self.depth_image = None
        self.last_time = self.get_clock().now()

        # Initialize YOLO model (ensure that "yolo11n.pt" is accessible)
        self.model = YOLO("yolo11n.pt", verbose=False)
        self.get_logger().info("Human Follower node initialized with YOLO.")

        # Desired following parameters (setpoints)
        self.desired_distance = 1.0         # meters (desired following distance)
        self.desired_lateral_offset = 0.0     # meters (desired lateral position: centered)

        # PID controllers:
        self.pid_distance = PID(Kp=0.5, Ki=0.0, Kd=0.1, setpoint=self.desired_distance)
        self.pid_lateral = PID(Kp=0.5, Ki=0.0, Kd=0.05, setpoint=self.desired_lateral_offset)

    def depth_callback(self, msg: Image):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        except CvBridgeError as e:
            self.get_logger().error(f"Depth conversion error: {e}")

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        # Run YOLO detection on the color image
        pos_res = gen_bbox_human(cv_image, self.model, acc=0.6)

        measured_distance = None
        measured_lateral = None
        centroid = None
        error_angle_rad = None

        if pos_res.size != 0:
            # Use the first detected bounding box
            x1, y1, x2, y2 = pos_res[0]
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            # Compute distance using the depth image at the centroid
            if self.depth_image is not None:
                try:
                    depth_value = self.depth_image[centroid[1], centroid[0]]
                    if depth_value > 0:
                        measured_distance = depth_value / 1000.0  # Convert mm to m
                except Exception as e:
                    self.get_logger().error(f"Error reading depth at centroid: {e}")

            # Compute lateral error: difference between centroid and image center (in pixels)
            image_width = cv_image.shape[1]
            center_x = image_width / 2.0
            pixel_error = centroid[0] - center_x

            # Convert pixel error to an angle in radians (assume horizontal FOV of 60°)
            horizontal_fov_deg = 60.0
            horizontal_fov_rad = np.deg2rad(horizontal_fov_deg)
            angle_per_pixel = horizontal_fov_rad / image_width
            error_angle_rad = pixel_error * angle_per_pixel

            # For small angles, lateral offset (in meters) is approximately: distance * angle (in radians)
            if measured_distance is not None:
                measured_lateral = measured_distance * error_angle_rad

            self.get_logger().info(f"Detected centroid: {centroid}, Distance: {measured_distance}, Lateral error: {measured_lateral}")

            # Compute PID control outputs
            linear_control = -self.pid_distance.update(measured_distance, dt) if measured_distance is not None else 0.0
            lateral_control = -self.pid_lateral.update(measured_lateral, dt) if measured_lateral is not None else 0.0

            # Limit commands to safe maximums.
            max_linear = 0.5   # m/s maximum forward velocity
            max_lateral = 1.0  # m/s maximum lateral velocity
            linear_cmd = max(max_linear, min(max_linear, linear_control))
            lateral_cmd = max(-max_lateral, min(max_lateral, lateral_control))

            self.get_logger().info(f"PID Output: Linear {linear_cmd:.2f} m/s, Lateral {lateral_cmd:.2f} m/s")

            # Publish TwistStamped command for mecanum drive (with linear.x and linear.y commands)
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
            self.get_logger().info("No human detected: Rotating to search...")
            twist_msg = TwistStamped()
            twist_msg.header.stamp = current_time.to_msg()
            twist_msg.twist.linear.x = 0.0
            twist_msg.twist.linear.y = 0.0
            twist_msg.twist.linear.z = 0.0
            twist_msg.twist.angular.x = 0.0
            twist_msg.twist.angular.y = 0.0
            twist_msg.twist.angular.z = 1.0  # rad/s search rotation
            self.cmd_pub.publish(twist_msg)

        # For display, convert error angle from radians to degrees (if available)
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
