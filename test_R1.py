#!/usr/bin/env python3
"""
A ROS 2 node that subscribes to /cam_1/color/image_raw and /cam_1/depth/image_rect_raw.
When the camera detects a human using the Ultralytics YOLO model, it reinitializes an OpenCV CSRT tracker to track that human.
If the human leaves the frame (no detection and tracker fails), the robot rotates in search mode.
The node reads the depth image to compute distance and lateral error and uses PID controllers to produce TwistStamped commands
for a mecanum drive robot so that it focuses on (follows) the detected human.
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
# YOLO Detection Function
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
        pos_res = bbox.xyxy.cpu().numpy()  # Nx4 bounding boxes: [x1, y1, x2, y2]
        if pos_res.shape[1] > 4:
            pos_res = pos_res[:, :4]
    else:
        pos_res = np.array([])
    return pos_res

# ---------------------------
# Visualization Function
# ---------------------------
def draw_rect(
    img: np.ndarray,
    pos_res: np.ndarray,
    if_centroid: bool = True,
    if_coords: bool = True,
    distance: Optional[float] = None,
    angle: Optional[float] = None
) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    """
    Draws bounding boxes on the image and overlays distance information.
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
# Human Follower Node with Focused Tracking
# ---------------------------
class HumanFollower(Node):
    def __init__(self):
        super().__init__('human_follower')
        # Subscriptions for color and depth images.
        self.color_sub = self.create_subscription(Image, '/cam_1/color/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/cam_1/depth/image_rect_raw', self.depth_callback, 10)
        # Publisher for TwistStamped commands.
        self.cmd_pub = self.create_publisher(TwistStamped, '/mecanum_drive_controller/cmd_vel', 10)
        self.bridge = CvBridge()
        self.depth_image = None
        self.last_time = self.get_clock().now()

        # Load YOLO model (ensure "yolo11n.pt" is accessible).
        self.model = YOLO("yolo11n.pt", verbose=False)
        self.get_logger().info("Human Follower node initialized with YOLO.")

        # Following parameters.
        self.desired_distance = 1.0         # Desired following distance (meters).
        self.desired_lateral_offset = 0.0     # Desired lateral offset (centered).

        # PID controllers.
        self.pid_distance = PID(Kp=0.5, Ki=0.0, Kd=0.1, setpoint=self.desired_distance)
        self.pid_lateral = PID(Kp=0.5, Ki=0.0, Kd=0.05, setpoint=self.desired_lateral_offset)

        # --- Tracking Variables ---
        self.tracker = None
        self.tracker_initialized = False
        self.tracker_type = "CSRT"  # Using CSRT tracker.

    def init_tracker(self, frame, bbox):
        """
        Initialize (or reinitialize) the tracker.
        bbox must be in (x, y, w, h) format.
        """
        try:
            if self.tracker_type == "CSRT":
                if hasattr(cv2, 'legacy'):
                    self.tracker = cv2.legacy.TrackerCSRT_create()
                else:
                    self.tracker = cv2.TrackerCSRT_create()
            else:
                if hasattr(cv2, 'legacy'):
                    self.tracker = cv2.legacy.TrackerKCF_create()
                else:
                    self.tracker = cv2.TrackerKCF_create()
            self.tracker.init(frame, bbox)
            self.tracker_initialized = True
            self.get_logger().info("Tracker re-initialized with new detection.")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize tracker: {e}")
            self.tracker_initialized = False

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

        # Run YOLO detection on the current image.
        pos_res = gen_bbox_human(cv_image, self.model, acc=0.6)

        bbox_used = None  # [x1, y1, x2, y2] from detection or tracker.
        tracker_used = False

        if pos_res.size != 0:
            # When a human is detected, always reinitialize the tracker.
            x1, y1, x2, y2 = pos_res[0]
            bbox_used = [x1, y1, x2, y2]
            bbox_tracker = (x1, y1, x2 - x1, y2 - y1)  # Format for tracker.
            self.init_tracker(cv_image, bbox_tracker)
        elif self.tracker_initialized:
            # If no detection is available, try to update the tracker.
            ok, bbox_tracker = self.tracker.update(cv_image)
            if ok:
                x, y, w, h = bbox_tracker
                bbox_used = [x, y, x + w, y + h]
                tracker_used = True
            else:
                self.get_logger().info("Tracker lost. Searching for human...")
                self.tracker_initialized = False

        measured_distance = None
        measured_lateral = None
        centroid = None
        error_angle_rad = None

        if bbox_used is not None:
            # Compute the centroid.
            x1, y1, x2, y2 = bbox_used
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            # Retrieve depth at the centroid.
            if self.depth_image is not None:
                try:
                    depth_value = self.depth_image[centroid[1], centroid[0]]
                    if depth_value > 0:
                        measured_distance = depth_value / 1000.0  # Convert from mm to m.
                except Exception as e:
                    self.get_logger().error(f"Error reading depth at centroid: {e}")

            # Compute lateral error (difference between centroid x and image center).
            image_width = cv_image.shape[1]
            center_x = image_width / 2.0
            pixel_error = centroid[0] - center_x

            # Convert pixel error to angle (assuming horizontal FOV of 60°).
            horizontal_fov_deg = 60.0
            horizontal_fov_rad = np.deg2rad(horizontal_fov_deg)
            angle_per_pixel = horizontal_fov_rad / image_width
            error_angle_rad = pixel_error * angle_per_pixel

            if measured_distance is not None:
                # Approximate lateral offset (in meters).
                measured_lateral = measured_distance * error_angle_rad

            self.get_logger().info(
                f"{'Tracked' if tracker_used else 'Detected'} human at centroid: {centroid}, "
                f"Distance: {measured_distance}, Lateral error: {measured_lateral}"
            )

            # Compute PID outputs if measurements are valid.
            if measured_distance is not None and measured_lateral is not None:
                linear_control = -self.pid_distance.update(measured_distance, dt)
                lateral_control = -self.pid_lateral.update(measured_lateral, dt)

                max_linear = 0.5   # m/s (forward)
                max_lateral = 1.0  # m/s (sideways)
                linear_cmd = np.clip(linear_control, -max_linear, max_linear)
                lateral_cmd = np.clip(lateral_control, -max_lateral, max_lateral)

                self.get_logger().info(f"PID Output: Linear {linear_cmd:.2f} m/s, Lateral {lateral_cmd:.2f} m/s")

                # Publish the drive command.
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
            # No detection and no tracking available: search mode.
            self.get_logger().info("No human detected: Rotating to search...")
            twist_msg = TwistStamped()
            twist_msg.header.stamp = current_time.to_msg()
            twist_msg.twist.linear.x = 0.0
            twist_msg.twist.linear.y = 0.0
            twist_msg.twist.linear.z = 0.0
            twist_msg.twist.angular.x = 0.0
            twist_msg.twist.angular.y = 0.0
            twist_msg.twist.angular.z = 1.0  # Rotate (rad/s)
            self.cmd_pub.publish(twist_msg)

        displayed_angle = np.rad2deg(error_angle_rad) if error_angle_rad is not None else None

        # Draw the current bounding box.
        if bbox_used is not None:
            pos_res_to_draw = np.array([bbox_used])
        else:
            pos_res_to_draw = np.array([])
        cv_image, _ = draw_rect(cv_image, pos_res_to_draw, if_centroid=True, if_coords=True,
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
