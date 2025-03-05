#!/usr/bin/env python3
"""
A combined ROS 2 node that subscribes to:
  - /cam_1/color/image_raw
  - /cam_1/depth/image_rect_raw
  - /scan (LiDAR)

It uses the Ultralytics YOLO model for human detection and a depth image to compute the
distance (in meters) and lateral offset (in meters) of the detected human relative to the image center.
In parallel, LiDAR data is clustered and fused via a Kalman filter to refine the camera’s angle measurement.
Three controllers (two PID controllers for distance and lateral offset, plus a proportional controller for rotation)
compute TwistStamped commands for a mecanum drive robot.
If no human is detected, the node enters search mode (rotating in place).
"""

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"  # Ensure proper X11 usage for cv2.imshow

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import TwistStamped
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import math
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
def gen_bbox_human(img: np.ndarray, model: YOLO, acc: float = 0.6) -> np.ndarray:
    results = model.predict(
        img,
        conf=acc,
        classes=[0],  # detect only humans (COCO class 0)
        save=False,
        save_txt=False,
        show_conf=False,
        verbose=False
    )
    if results and len(results[0].boxes) > 0:
        bbox = results[0].boxes
        pos_res = bbox.xyxy.cpu().numpy()  # shape: Nx4 [x1, y1, x2, y2]
        if pos_res.shape[1] > 4:
            pos_res = pos_res[:, :4]
    else:
        pos_res = np.array([])
    return pos_res

def draw_rect(
    img: np.ndarray,
    pos_res: np.ndarray,
    centroid: Optional[Tuple[int, int]] = None,
    info_text: Optional[str] = None,
    box_color: Tuple[int, int, int] = (0, 0, 255)
) -> np.ndarray:
    thickness = 2
    for pos in pos_res:
        x1, y1, x2, y2 = map(int, pos)
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)
    if centroid is not None:
        cv2.circle(img, centroid, 5, box_color, -1)
    if info_text is not None:
        cv2.putText(img, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img

# ---------------------------
# Combined Human Follower Node
# ---------------------------
class HumanFollowerCombined(Node):
    def __init__(self):
        super().__init__('human_follower_combined')
        # Subscriptions: color image, depth image, and LiDAR scan
        self.color_sub = self.create_subscription(Image, '/cam_1/color/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/cam_1/detections', self.depth_callback, 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        # Publisher for mecanum drive commands
        self.cmd_pub = self.create_publisher(TwistStamped, '/mecanum_drive_controller/cmd_vel', 10)
        self.bridge = CvBridge()

        # Load YOLO model (ensure the model file is in the proper location)
        self.model = YOLO("yolo11n.pt", verbose=False)
        self.get_logger().info("YOLO model loaded.")

        # Camera parameters (adjust as needed)
        self.image_width = 848  # example value
        self.image_height = 480
        self.horizontal_fov = 60.0  # degrees; adjust based on your camera

        # Desired setpoints
        self.desired_distance = 1.0       # meters
        self.desired_lateral = 0.0        # ideally, the human is centered in lateral direction
        self.desired_angle = 0.0          # desired angle error (zero if human is centered)

        # PID controllers for distance and lateral offset (and optionally for rotation)
        self.pid_distance = PID(Kp=0.5, Ki=0.0, Kd=0.1, setpoint=self.desired_distance)
        self.pid_lateral  = PID(Kp=0.5, Ki=0.0, Kd=0.05, setpoint=self.desired_lateral)
        # For rotation, here we use a simple proportional control.
        self.kp_rotation = 0.02

        # Kalman filter variables for angle refinement (state: [angle; angular_rate] in degrees)
        self.kf_x = np.array([[0.0], [0.0]])
        self.kf_P = np.eye(2) * 1.0
        self.last_kf_time = None
        self.refined_angle = None  # from LiDAR and KF
        self.camera_angle = None   # from camera detection

        # Latest LiDAR scan message (for clustering and fallback)
        self.latest_scan = None

        # For smoothing the linear velocity command
        self.filtered_linear_x = 0.0

        # For fallback when no valid detection: search mode (rotate in place)
        self.search_angular_speed = 0.2  # rad/s

        # Visualization window
        cv2.namedWindow("Human Follower Combined", cv2.WINDOW_NORMAL)

    # ---------------
    # LiDAR Handling
    # ---------------
    def lidar_callback(self, msg: LaserScan):
        self.latest_scan = msg
        current_time = self.get_clock().now().nanoseconds / 1e9  # seconds

        clusters = self.cluster_lidar(msg)
        if self.camera_angle is not None and clusters:
            # Choose the cluster with mean angle closest to the camera measurement.
            best_cluster = min(clusters, key=lambda c: abs(c[0] - self.camera_angle))
            measured_angle = best_cluster[0]  # in degrees
            self.update_kalman(measured_angle, current_time)

    def cluster_lidar(self, scan_msg: LaserScan):
        """
        Groups adjacent LiDAR points into clusters based on range differences.
        Returns a list of tuples: (mean_angle, mean_range, list_of_indices)
        where mean_angle is in degrees.
        """
        clusters = []
        current_cluster = []
        for i, r in enumerate(scan_msg.ranges):
            if r == float('inf') or r == 0.0:
                continue
            if not current_cluster:
                current_cluster.append(i)
            else:
                prev = scan_msg.ranges[current_cluster[-1]]
                if abs(r - prev) < 0.2:  # threshold; adjust as needed
                    current_cluster.append(i)
                else:
                    if len(current_cluster) >= 3:
                        clusters.append(current_cluster)
                    current_cluster = [i]
        if current_cluster and len(current_cluster) >= 3:
            clusters.append(current_cluster)

        cluster_data = []
        for cluster in clusters:
            angles = []
            ranges = []
            for idx in cluster:
                a = scan_msg.angle_min + idx * scan_msg.angle_increment  # in radians
                angles.append(np.rad2deg(a))
                ranges.append(scan_msg.ranges[idx])
            mean_angle = np.mean(angles)
            mean_range = np.mean(ranges)
            cluster_data.append((mean_angle, mean_range, cluster))
        return cluster_data

    def update_kalman(self, z, current_time):
        """
        Update the Kalman filter with measurement z (angle in degrees) using a constant velocity model.
        """
        if self.last_kf_time is None:
            self.last_kf_time = current_time
            self.kf_x[0, 0] = z
            self.refined_angle = z
            return
        dt = current_time - self.last_kf_time
        self.last_kf_time = current_time

        # Predict
        A = np.array([[1, dt],
                      [0, 1]])
        Q = np.array([[0.1, 0],
                      [0, 0.1]])
        self.kf_x = A.dot(self.kf_x)
        self.kf_P = A.dot(self.kf_P).dot(A.T) + Q

        # Update
        H = np.array([[1, 0]])
        R = np.array([[1.0]])
        z = np.array([[z]])
        y = z - H.dot(self.kf_x)
        S = H.dot(self.kf_P).dot(H.T) + R
        K = self.kf_P.dot(H.T).dot(np.linalg.inv(S))
        self.kf_x = self.kf_x + K.dot(y)
        self.kf_P = (np.eye(2) - K.dot(H)).dot(self.kf_P)
        self.refined_angle = self.kf_x[0, 0]

    # ---------------
    # Depth Image Handling
    # ---------------
    def depth_callback(self, msg: Image):
        try:
            # Depth image in 16UC1 (millimeters)
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        except CvBridgeError as e:
            self.get_logger().error(f"Depth conversion error: {e}")
            self.depth_image = None

    # ---------------
    # Color Image Handling & Main Processing
    # ---------------
    def image_callback(self, msg: Image):
        try:
            color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        current_time = self.get_clock().now()
        dt = 0.1  # default dt (in seconds) if needed
        # (You could compute dt from a stored previous time if desired)

        # Run YOLO detection on the color image.
        pos_res = gen_bbox_human(color_image, self.model, acc=0.6)

        measured_distance = None
        measured_lateral = None  # in meters
        chosen_bbox = None
        centroid = None

        # If detection found and depth is available, use it:
        if pos_res.size != 0 and hasattr(self, 'depth_image') and self.depth_image is not None:
            best_distance = float('inf')
            for pos in pos_res:
                x1, y1, x2, y2 = map(int, pos)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                if 0 <= cx < self.depth_image.shape[1] and 0 <= cy < self.depth_image.shape[0]:
                    depth_value = self.depth_image[cy, cx]
                    if depth_value > 0:
                        d = depth_value / 1000.0  # convert mm to meters
                        if d < best_distance:
                            best_distance = d
                            chosen_bbox = pos
                            centroid = (cx, cy)
                            measured_distance = d
            # Compute lateral offset error based on the pixel error from image center.
            if centroid is not None and measured_distance is not None:
                center_x = self.image_width / 2.0
                pixel_error = centroid[0] - center_x
                horizontal_fov_rad = np.deg2rad(self.horizontal_fov)
                angle_per_pixel = horizontal_fov_rad / self.image_width
                error_angle_rad = pixel_error * angle_per_pixel
                measured_lateral = measured_distance * error_angle_rad  # approx lateral error (in m)

                # Compute initial camera angle (in degrees) from the centroid offset.
                self.camera_angle = np.rad2deg(error_angle_rad)
        else:
            # Fallback: if no valid detection from YOLO+depth, try using LiDAR directly.
            if self.latest_scan is not None:
                # Use the LiDAR’s minimum valid range as an approximation.
                ranges = np.array(self.latest_scan.ranges)
                valid = np.where((ranges > self.latest_scan.range_min) & (ranges < self.latest_scan.range_max))[0]
                if valid.size > 0:
                    idx = valid[np.argmin(ranges[valid])]
                    measured_distance = ranges[idx]
                    # Approximate lateral offset from LiDAR angle:
                    angle = self.latest_scan.angle_min + idx * self.latest_scan.angle_increment
                    measured_lateral = measured_distance * math.sin(angle)
                    # In this fallback, we set the camera angle to the LiDAR angle (in degrees).
                    self.camera_angle = math.degrees(angle)

        # Combine the camera angle and the KF-refined angle (if available)
        if self.camera_angle is not None and self.refined_angle is not None:
            control_angle = 0.5 * self.camera_angle + 0.5 * self.refined_angle
        elif self.refined_angle is not None:
            control_angle = self.refined_angle
        else:
            control_angle = self.camera_angle if self.camera_angle is not None else 0.0

        # Decide if we have a valid measurement for control.
        have_measurement = (measured_distance is not None and measured_lateral is not None)

        twist_msg = TwistStamped()
        twist_msg.header.stamp = current_time.to_msg()

        if have_measurement:
            # Use PID controllers for distance and lateral offset.
            linear_control = -self.pid_distance.update(measured_distance, dt)
            lateral_control = -self.pid_lateral.update(measured_lateral, dt)

            # Proportional controller for rotation (aim to have 0° error).
            angular_control = self.kp_rotation * (-control_angle)

            # Optionally, apply exponential filtering for smooth forward speed.
            alpha = 0.7
            self.filtered_linear_x = alpha * linear_control + (1 - alpha) * self.filtered_linear_x

            self.get_logger().info(
                f"Tracking: Dist = {measured_distance:.2f} m, Lateral = {measured_lateral:.2f} m, "
                f"CamAng = {self.camera_angle:.1f}°, RefinedAng = {self.refined_angle if self.refined_angle is not None else 0.0:.1f}°"
            )
            self.get_logger().info(
                f"PID: LinearX = {self.filtered_linear_x:.2f} m/s, Lateral = {lateral_control:.2f} m/s, Angular = {angular_control:.2f} rad/s"
            )

            twist_msg.twist.linear.x = np.clip(self.filtered_linear_x, 0.5, 0.5)
            twist_msg.twist.linear.y = np.clip(lateral_control, -1.0, 1.0)
            twist_msg.twist.angular.z = np.clip(angular_control, -1.0, 1.0)
        else:
            # No valid measurement: enter search mode (rotate slowly)
            self.get_logger().info("No human detected: entering search mode (rotating).")
            twist_msg.twist.linear.x = 0.0
            twist_msg.twist.linear.y = 0.0
            twist_msg.twist.angular.z = self.search_angular_speed

        self.cmd_pub.publish(twist_msg)

        # Prepare an overlay for visualization.
        info_text = ""
        if measured_distance is not None:
            info_text = f"D: {measured_distance:.2f} m, Lat: {measured_lateral:.2f} m, Ang: {control_angle:.1f}°"
        # Draw the chosen bounding box (if available) in green; others in red.
        if chosen_bbox is not None:
            color_image = draw_rect(color_image, pos_res, centroid=centroid, info_text=info_text, box_color=(0, 0, 255))
            # Highlight the chosen detection with a green rectangle.
            x1, y1, x2, y2 = map(int, chosen_bbox)
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            color_image = draw_rect(color_image, pos_res, info_text=info_text, box_color=(0, 0, 255))

        cv2.imshow("Human Follower Combined", color_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = HumanFollowerCombined()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
