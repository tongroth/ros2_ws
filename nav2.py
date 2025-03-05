#!/usr/bin/env python3
"""
This ROS 2 node detects a human using camera topics and YOLO,
computes a waypoint (PoseStamped) for following the human, and:
  1. Publishes the waypoint on `/human_waypoint` (for visualization)
  2. Sends the waypoint as a goal to Navigation2 via the NavigateToPose action.
Additionally, the node subscribes to the point cloud topic
(`/cam_1/depth/color/points`) and uses it to compute a 3D bounding box,
which is published as a MarkerArray on `/human_markers` for RViz visualization.

If no valid depth measurement is available, a fallback distance is used.
"""

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"  # For cv2.imshow with X11

import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped, Quaternion, Point
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray

# Import the Navigation2 action (make sure nav2_msgs is installed)
from nav2_msgs.action import NavigateToPose

from ultralytics import YOLO
from typing import Tuple, Optional

# For converting PointCloud2 to a numpy array
import sensor_msgs_py.point_cloud2 as pc2

# ---------------------------
# Helper: Convert yaw to quaternion
# ---------------------------
def quaternion_from_yaw(yaw: float) -> Quaternion:
    """Return a quaternion from a yaw angle in radians."""
    qz = math.sin(yaw * 0.5)
    qw = math.cos(yaw * 0.5)
    return Quaternion(x=0.0, y=0.0, z=qz, w=qw)

# ---------------------------
# Helper: Convert organized PointCloud2 to a numpy array
# ---------------------------
def pointcloud2_to_array(cloud_msg: PointCloud2) -> np.ndarray:
    """
    Convert an organized PointCloud2 message into a numpy array of shape (height, width, 3).
    This assumes that the point cloud is organized.
    """
    # Read points (x, y, z) from the cloud message.
    gen = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=False)
    arr = np.array(list(gen), dtype=np.float32)
    if cloud_msg.height > 1:
        arr = arr.reshape((cloud_msg.height, cloud_msg.width, 3))
    else:
        # If not organized, return a flat array (won't be used for drawing bounding box)
        pass
    return arr

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
        # Subscribers for color image and point cloud
        self.color_sub = self.create_subscription(
            Image, '/cam_1/color/image_raw', self.image_callback, 10)
        # We still subscribe to a depth topic if needed, but here we also use the point cloud.
        self.pc_sub = self.create_subscription(
            PointCloud2, '/cam_1/depth/color/points', self.pc_callback, 10)
        self.bridge = CvBridge()
        self.pc_array = None  # Will hold the organized point cloud as a numpy array
        self.pc_frame = None  # Store the frame id from point cloud messages

        # Publisher for the computed waypoint (for visualization)
        self.waypoint_pub = self.create_publisher(PoseStamped, '/human_waypoint', 10)
        # Publisher for MarkerArray to display 3D bounding box in RViz
        self.marker_pub = self.create_publisher(MarkerArray, '/human_markers', 10)

        # Initialize YOLO model (ensure "yolo11n.pt" is accessible)
        self.model = YOLO("yolo11n.pt", verbose=False)
        self.get_logger().info("Human Follower node initialized with YOLO.")

        # Desired following parameters
        self.desired_distance = 1.0  # meters
        self.desired_lateral_offset = 0.0  # centered

        # Action client for Navigation2's NavigateToPose action
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self._current_goal_handle = None  # Store current goal (if any)

    def pc_callback(self, msg: PointCloud2):
        try:
            self.pc_array = pointcloud2_to_array(msg)
            self.pc_frame = msg.header.frame_id
        except Exception as e:
            self.get_logger().error(f"Error converting point cloud: {e}")

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        current_time = self.get_clock().now()
        # Run YOLO detection on the color image
        pos_res = gen_bbox_human(cv_image, self.model, acc=0.6)

        measured_distance = None
        measured_lateral = None  # Lateral offset (in meters)
        centroid = None

        if pos_res.size != 0:
            # Use the first detected bounding box
            x1, y1, x2, y2 = pos_res[0]
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            # Get depth from the point cloud (if available and organized)
            if self.pc_array is not None:
                # Ensure indices are within bounds
                height, width, _ = self.pc_array.shape
                u = np.clip(centroid[0], 0, width - 1)
                v = np.clip(centroid[1], 0, height - 1)
                pt = self.pc_array[v, u]
                if not np.isnan(pt[0]) and not np.isnan(pt[1]) and not np.isnan(pt[2]):
                    # Compute distance as Euclidean norm
                    measured_distance = math.sqrt(pt[0]**2 + pt[1]**2 + pt[2]**2)
            # Fallback if point cloud data not available or invalid:
            if measured_distance is None:
                measured_distance = self.desired_distance
                self.get_logger().warn(f"No valid depth from point cloud; using fallback: {measured_distance:.2f} m")

            # Compute lateral offset from image center (using color image)
            image_width = cv_image.shape[1]
            center_x = image_width / 2.0
            pixel_error = centroid[0] - center_x
            horizontal_fov_rad = np.deg2rad(60.0)  # Assumed FOV (in radians)
            angle_per_pixel = horizontal_fov_rad / image_width
            error_angle_rad = pixel_error * angle_per_pixel
            measured_lateral = measured_distance * error_angle_rad

            self.get_logger().info(
                f"Detected human at {centroid}, distance: {measured_distance:.2f} m, lateral error: {measured_lateral:.2f} m")

            # Compute waypoint (in the robot's base frame)
            # Here we simply use the measured distance and lateral offset to compute a target
            dx = measured_distance
            dy = measured_lateral
            human_distance = math.sqrt(dx**2 + dy**2)
            if human_distance > self.desired_distance:
                ratio = (human_distance - self.desired_distance) / human_distance
                goal_x = dx * ratio
                goal_y = dy * ratio
            else:
                goal_x = dx
                goal_y = dy

            yaw = math.atan2(dy, dx)
            quat = quaternion_from_yaw(yaw)

            waypoint = PoseStamped()
            waypoint.header.stamp = current_time.to_msg()
            # Use an appropriate frame; here we assume the waypoint is in the camera (or transformed later)
            waypoint.header.frame_id = self.pc_frame if self.pc_frame is not None else "cam_1_depth_frame"
            waypoint.pose.position.x = goal_x
            waypoint.pose.position.y = goal_y
            waypoint.pose.position.z = 0.0
            waypoint.pose.orientation = quat

            # Publish the waypoint for visualization
            self.waypoint_pub.publish(waypoint)
            self.get_logger().info(
                f"Waypoint published: x={goal_x:.2f}, y={goal_y:.2f}, yaw={math.degrees(yaw):.1f}°")

            # Send the waypoint as a navigation goal to Navigation2
            self.send_goal(waypoint)

            # ---------------------------
            # Create a 3D Bounding Box Marker from the PointCloud
            # ---------------------------
            # If the point cloud is organized, use the bounding box pixel coordinates to get 3D corners.
            if self.pc_array is not None:
                height, width, _ = self.pc_array.shape
                u1 = int(np.clip(x1, 0, width - 1))
                v1 = int(np.clip(y1, 0, height - 1))
                u2 = int(np.clip(x2, 0, width - 1))
                v2 = int(np.clip(y2, 0, height - 1))
                # Four corners in pixel coordinates: top-left, top-right, bottom-right, bottom-left
                corners = []
                for (u, v) in [(u1, v1), (u2, v1), (u2, v2), (u1, v2)]:
                    pt = self.pc_array[v, u]
                    # Create a Point message for each 3D corner
                    p = Point()
                    p.x, p.y, p.z = pt[0], pt[1], pt[2]
                    corners.append(p)

                # Create a marker to display the bounding box in RViz
                marker = Marker()
                marker.header.stamp = current_time.to_msg()
                marker.header.frame_id = self.pc_frame if self.pc_frame is not None else "cam_1_depth_frame"
                marker.ns = "human_bbox"
                marker.id = 0
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                marker.scale.x = 0.01  # line width
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                # Add the four corners and close the loop (repeat first point)
                marker.points = corners + [corners[0]]
                marker.lifetime.sec = 0  # persist until updated

                marker_array = MarkerArray()
                marker_array.markers.append(marker)
                self.marker_pub.publish(marker_array)
        else:
            # No human detected; cancel any active navigation goal
            self.get_logger().info("No human detected; canceling any active goal.")
            if self._current_goal_handle is not None:
                self._current_goal_handle.cancel_goal()
                self._current_goal_handle = None

        # Optionally, display the detection overlay on the image (2D)
        cv_image, _ = draw_rect(cv_image, pos_res, if_centroid=True, if_coords=True,
                                distance=measured_distance,
                                angle=np.rad2deg(error_angle_rad) if centroid is not None else None)
        cv2.imshow("Human Detection", cv_image)
        cv2.waitKey(1)

    def send_goal(self, waypoint: PoseStamped):
        """Send a navigation goal to Navigation2 using the NavigateToPose action."""
        if not self.nav_to_pose_client.server_is_ready():
            self.get_logger().info("Waiting for NavigateToPose action server...")
            self.nav_to_pose_client.wait_for_server()

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = waypoint

        if self._current_goal_handle is not None:
            self.get_logger().info("Canceling previous goal before sending new one.")
            self._current_goal_handle.cancel_goal()

        send_goal_future = self.nav_to_pose_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Navigation goal rejected.')
            return
        self.get_logger().info('Navigation goal accepted.')
        self._current_goal_handle = goal_handle
        goal_handle.get_result_async().add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        # Process feedback if needed
        pass

    def get_result_callback(self, future):
        self.get_logger().info("Navigation goal reached.")
        self._current_goal_handle = None

def main(args=None):
    rclpy.init(args=args)
    node = HumanFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Human Follower node...")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
