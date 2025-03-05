#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class WaypointListener(Node):
    def __init__(self):
        super().__init__('waypoint_listener')
        self.subscription = self.create_subscription(
            PoseStamped,
            '/human_waypoint',
            self.listener_callback,
            10)
        self.get_logger().info("Subscribed to /human_waypoint")

    def listener_callback(self, msg: PoseStamped):
        self.get_logger().info(f"Received waypoint: {msg}")

def main(args=None):
    rclpy.init(args=args)
    node = WaypointListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Waypoint Listener node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

