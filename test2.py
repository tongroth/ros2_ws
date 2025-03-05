#!/usr/bin/env python3
"""
A ROS 2 node that subscribes to /cam_1/color/image_raw and /cam_1/depth/image_rect_raw,
detects a human using the Ultralytics YOLO model, computes the distance (in meters) and horizontal angle 
(in degrees, relative to the image center) to the detected human, and uses PID controllers to generate
TwistStamped commands so that the robot follows the human. If no human is detected, the robot rotates slowly to search.
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
        pos_res = bbox.xyxy.cpu().numpy()  # Expected shape: Nx4
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
            info_text = f"Dist: {distance:.2f} m, Angle: {angle:.1f}°"
            cv2.putText(img, info_text, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img, centroid_res

# ---------------------------
# Human Follower Node
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
            '/cam_1/depth/image_rect_raw',
            self.depth_callback,
            10
        )
        # Publisher for TwistStamped commands
        self.cmd_pub = self.create_publisher(TwistStamped, '/mecanum_drive_controller/cmd_vel', 10)
        self.bridge = CvBridge()
        self.depth_image = None
        self.last_time = self.get_clock().now()

        # Initialize YOLO model (update model path as necessary)
        self.model = YOLO("yolo11n.pt", verbose=False)
        self.get_logger().info("Human Follower node initialized with YOLO.")

        # Desired following parameters (setpoints)
        self.desired_distance = 1.0  # meters
        self.desired_angle = 0.0     # degrees (centered)

        # PID controllers for distance and angle
        self.pid_distance = PID(Kp=0.5, Ki=0.0, Kd=0.1, setpoint=self.desired_distance)
        self.pid_angle = PID(Kp=0.02, Ki=0.0, Kd=0.005, setpoint=self.desired_angle)

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
        measured_angle = None
        centroid = None

        if pos_res.size != 0:
            # Use the first detected bounding box for control
            x1, y1, x2, y2 = pos_res[0]
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            # Compute distance from the depth image at the centroid
            if self.depth_image is not None:
                try:
                    depth_value = self.depth_image[centroid[1], centroid[0]]
                    if depth_value > 0:
                        measured_distance = depth_value / 1000.0  # mm to m
                except Exception as e:
                    self.get_logger().error(f"Error reading depth at centroid: {e}")

            # Compute horizontal angle relative to the image center
            image_width = cv_image.shape[1]
            center_x = image_width / 2.0
            horizontal_fov = 60.0  # assumed horizontal FOV in degrees
            measured_angle = ((centroid[0] - center_x) / center_x) * (horizontal_fov / 2.0)

            self.get_logger().info(f"Detected centroid: {centroid}, Distance: {measured_distance}, Angle: {measured_angle}")

            # Compute PID control outputs
            # If no valid measurement, use current setpoint (this should not happen if detection is valid)
            control_distance = -self.pid_distance.update(measured_distance, dt) if measured_distance is not None else 0.0
            control_angle = -self.pid_angle.update(measured_angle, dt) if measured_angle is not None else 0.0

            # Limit commands to safe maximums
            max_linear = 0.5   # m/s
            max_angular = 1.0  # rad/s
            linear_cmd = max(max_linear, min(max_linear, control_distance))
            angular_cmd = max(-max_angular, min(max_angular, control_angle * np.pi / 180.0))

            self.get_logger().info(f"PID Output: Linear {linear_cmd:.2f} m/s, Angular {angular_cmd:.2f} rad/s")

            # Publish the computed command
            twist_msg = TwistStamped()
            twist_msg.header.stamp = current_time.to_msg()
            twist_msg.twist.linear.x = linear_cmd
            twist_msg.twist.linear.y = 0.0
            twist_msg.twist.linear.z = 0.0
            twist_msg.twist.angular.x = 0.0
            twist_msg.twist.angular.y = 0.0
            twist_msg.twist.angular.z = angular_cmd
            self.cmd_pub.publish(twist_msg)
        else:
            # No detection: rotate slowly to search for a human
            self.get_logger().info("No human detected: Rotating to search...")
            twist_msg = TwistStamped()
            twist_msg.header.stamp = current_time.to_msg()
            twist_msg.twist.linear.x = 0.0
            twist_msg.twist.linear.y = 0.0
            twist_msg.twist.linear.z = 0.0
            twist_msg.twist.angular.x = 0.0
            twist_msg.twist.angular.y = 0.0
            twist_msg.twist.angular.z = 0.2  # rad/s for search rotation
            self.cmd_pub.publish(twist_msg)

        # Draw bounding boxes with overlays on the image
        cv_image, _ = draw_rect(cv_image, pos_res, if_centroid=True, if_coords=True,
                                distance=measured_distance, angle=measured_angle)
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
