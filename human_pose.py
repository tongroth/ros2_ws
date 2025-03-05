#!/usr/bin/env python3
"""
A ROS 2 node that detects a human using YOLO, uses LiDAR data for depth estimation,
computes the human's position relative to the camera optical frame, publishes it,
and then transforms the pose into a global frame (e.g., "odom") for navigation.
"""

import os
import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO

import tf2_ros
from tf2_geometry_msgs import do_transform_pose

class HumanPosePublisher(Node):
    def __init__(self):
        super().__init__('human_pose_publisher')

        # Declare parameters for camera frame and target (global) frame
        self.declare_parameter("camera_frame", "cam_1_depth_optical_frame")
        self.declare_parameter("target_frame", "odom")
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value
        self.target_frame = self.get_parameter("target_frame").get_parameter_value().string_value

        # Subscribe to the color image topic
        self.color_sub = self.create_subscription(
            Image,
            '/cam_1/color/image_raw',
            self.image_callback,
            10
        )
        # Subscribe to the LiDAR scan topic
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',  # Adjust topic name if necessary
            self.scan_callback,
            10
        )
        # Publisher for the human pose in camera frame
        self.pose_pub = self.create_publisher(PoseStamped, '/human_pose', 10)
        # Publisher for the transformed human pose in the global frame
        self.global_pose_pub = self.create_publisher(PoseStamped, '/human_global_pose', 10)

        self.bridge = CvBridge()
        self.scan = None  # Will store the latest LiDAR scan

        # Initialize the YOLO model (ensure "yolo11n.pt" is accessible)
        self.model = YOLO("yolo11n.pt", verbose=False)
        self.get_logger().info("Human Pose Publisher node initialized with YOLO and LiDAR.")

        # Initialize TF2 buffer and listener for transforming poses
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def scan_callback(self, msg: LaserScan):
        """Store the latest LiDAR scan."""
        self.scan = msg

    def image_callback(self, msg: Image):
        """
        Process the color image, detect a human, compute its pose using LiDAR depth data,
        and publish both camera and global frame poses.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        current_time = self.get_clock().now()

        # Run YOLO detection on the image to get bounding boxes (only human detections)
        pos_res = self.gen_bbox_human(cv_image)
        if pos_res.size == 0:
            self.get_logger().info("No human detected.")
            return

        # Use the first detected bounding box
        x1, y1, x2, y2 = pos_res[0]
        centroid_x = int((x1 + x2) / 2)
        centroid_y = int((y1 + y2) / 2)

        # Ensure a LiDAR scan has been received
        if self.scan is None:
            self.get_logger().warning("No LiDAR scan received yet.")
            return

        # Compute lateral error based on image pixel difference
        image_width = cv_image.shape[1]
        center_x = image_width / 2.0
        pixel_error = centroid_x - center_x

        # Convert pixel error to an angle in radians assuming a horizontal FOV (adjust as needed)
        horizontal_fov_rad = math.radians(87.0)
        angle_per_pixel = horizontal_fov_rad / image_width
        error_angle_rad = pixel_error * angle_per_pixel

        # Map the error angle to a LiDAR measurement index
        angle_min = self.scan.angle_min
        angle_max = self.scan.angle_min + len(self.scan.ranges) * self.scan.angle_increment
        if not (angle_min <= error_angle_rad <= angle_max):
            self.get_logger().warning("Detected angle is outside LiDAR's FOV.")
            return

        index = int((error_angle_rad - angle_min) / self.scan.angle_increment)
        measured_distance = self.scan.ranges[index]
        if math.isinf(measured_distance) or math.isnan(measured_distance) or measured_distance == 0.0:
            self.get_logger().warning("Invalid LiDAR range measurement.")
            return

        # Compute lateral offset in meters using tangent
        measured_lateral = measured_distance * math.tan(error_angle_rad)

        self.get_logger().info(
            f"Human detected: Distance {measured_distance:.2f} m, Lateral offset {measured_lateral:.2f} m"
        )

        # Create and populate the PoseStamped message in the camera frame
        pose_msg = PoseStamped()
        pose_msg.header.stamp = current_time.to_msg()
        pose_msg.header.frame_id = self.camera_frame

        # In the optical frame:
        #   - X: lateral offset (measured_lateral)
        #   - Y: vertical offset (assumed 0)
        #   - Z: forward distance (measured_distance)
        pose_msg.pose.position.x = measured_lateral
        pose_msg.pose.position.y = 0.0
        pose_msg.pose.position.z = measured_distance

        # Convert error_angle_rad to a quaternion (rotation around the vertical axis)
        qz = math.sin(error_angle_rad / 2.0)
        qw = math.cos(error_angle_rad / 2.0)
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw

        # Publish the pose in the camera frame
        self.pose_pub.publish(pose_msg)

        # Transform the pose to the target (global) frame and publish it
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.camera_frame,
                rclpy.time.Time()
            )
            global_pose = do_transform_pose(pose_msg, transform)
            global_pose.header.frame_id = self.target_frame
            self.global_pose_pub.publish(global_pose)
            self.get_logger().info(
                f"Published global human pose in '{self.target_frame}': x={global_pose.pose.position.x:.2f}, y={global_pose.pose.position.y:.2f}"
            )
        except tf2_ros.LookupException as e:
            self.get_logger().warning(f"Transform lookup exception: {e}")
        except tf2_ros.ExtrapolationException as e:
            self.get_logger().warning(f"Transform extrapolation exception: {e}")
        except Exception as e:
            self.get_logger().error(f"Error transforming pose: {e}")

    def gen_bbox_human(self, img: np.ndarray) -> np.ndarray:
        """
        Run YOLO detection on the input image and return detected bounding boxes.
        Only human detections (COCO class 0) are returned.
        """
        results = self.model.predict(img, conf=0.6, classes=[0], save=False)
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            # Keep only the first 4 values per box: [x1, y1, x2, y2]
            if boxes.ndim > 1 and boxes.shape[1] > 4:
                boxes = boxes[:, :4]
            return boxes
        return np.array([])

def main(args=None):
    rclpy.init(args=args)
    node = HumanPosePublisher()
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
