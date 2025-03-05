#!/usr/bin/env python3
"""
A simple ROS 2 node that subscribes to /cam_1/color/image_raw and performs
human detection using OpenCV’s HOG person detector. You can expand this node
to include additional object detection functionality.
"""
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class HumanObjectDetector(Node):
    def __init__(self):
        super().__init__('human_object_detector')
        self.subscription = self.create_subscription(
            Image,
            '/cam_1/color/image_raw',
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Create a CvBridge to convert ROS images to OpenCV format.
        self.bridge = CvBridge()

        # Set up the HOG descriptor/person detector.
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.get_logger().info("Human/Object Detector Initialized.")

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error('Failed to convert image: %s' % str(e))
            return

        # Detect humans in the image.
        # The detectMultiScale function returns bounding boxes and weights.
        boxes, weights = self.hog.detectMultiScale(cv_image, winStride=(8, 8))
        for (x, y, w, h) in boxes:
            # Draw a rectangle around each detected person.
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # (Optional) Here you could add additional object detection logic.
        # For example, run a deep learning detector for other objects and draw bounding boxes.

        # Display the image with detection boxes.
        cv2.imshow("Human and Object Detection", cv_image)
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

