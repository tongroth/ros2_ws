#!/usr/bin/env python3
"""
A simple bridge node that subscribes to the standard `/cmd_vel` topic (from Navigation2)
and converts the incoming Twist messages into TwistStamped messages, publishing them on
`/mecanum_drive_controller/cmd_vel` for a mecanum wheeled robot.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped

class CmdVelBridge(Node):
    def __init__(self):
        super().__init__('cmd_vel_bridge')
        # Subscribe to the navigation command velocity topic
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',            # Topic from Navigation2
            self.cmd_vel_callback,
            10)
        self.subscription  # Prevent unused variable warning

        # Publisher for the mecanum drive command velocity (TwistStamped)
        self.publisher = self.create_publisher(
            TwistStamped,
            '/mecanum_drive_controller/cmd_vel',  # Robot's expected topic
            10)

        self.get_logger().info("CmdVelBridge node has been started.")

    def cmd_vel_callback(self, twist_msg: Twist):
        """
        Callback that receives a Twist message from `/cmd_vel`, wraps it into a TwistStamped
        message with the current time, and publishes it on `/mecanum_drive_controller/cmd_vel`.
        """
        twist_stamped = TwistStamped()
        twist_stamped.header.stamp = self.get_clock().now().to_msg()
        twist_stamped.header.frame_id = 'map'  # Set frame_id if required by your robot
        twist_stamped.twist = twist_msg

        self.publisher.publish(twist_stamped)
        self.get_logger().debug(f"Bridged cmd_vel: {twist_msg}")

def main(args=None):
    rclpy.init(args=args)
    node = CmdVelBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

