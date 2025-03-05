#!/usr/bin/env python3
"""
A simple keyboard teleoperation node for a mecanum drive that publishes TwistStamped messages.
Use keys:
  • w/s: Increase/decrease forward (x) speed.
  • a/d: Increase/decrease lateral (y) speed.
  • q/e: Increase/decrease angular (z) velocity.
  • Space or k: Stop.
  • Ctrl-C: Quit.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
import sys
import select
import termios
import tty

# Define key bindings.
# Each key maps to a tuple: (x direction, y direction, angular z)
moveBindings = {
    'w': (1, 0, 0),   # forward
    's': (-1, 0, 0),  # backward
    'a': (0, 1, 0),   # left
    'd': (0, -1, 0),  # right
    'q': (0, 0, 1),   # rotate counter-clockwise
    'e': (0, 0, -1),  # rotate clockwise
}


def getKey(settings):
    """Capture a single key press (non-blocking)."""
    tty.setraw(sys.stdin.fileno())
    # Wait for up to 0.1 sec for a key press.
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


class TeleopMecanum(Node):
    def __init__(self):
        super().__init__('teleop_mecanum')
        self.publisher_ = self.create_publisher(
            TwistStamped, '/mecanum_drive_controller/cmd_vel', 10)
        # Set your base speeds (adjust as needed)
        self.linear_speed = 0.5  # base speed for x and y
        self.angular_speed = 1.0   # base speed for rotation

    def publish_cmd(self, x, y, angular):
        """Publish a TwistStamped message with current speeds."""
        msg = TwistStamped()
        # Use the node’s clock to set the header stamp.
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ''
        msg.twist.linear.x = x
        msg.twist.linear.y = y
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = angular
        self.publisher_.publish(msg)

    def run(self):
        settings = termios.tcgetattr(sys.stdin)
        print("Control Your Mecanum Robot!")
        print("Use keys:")
        print("  w/s : increase/decrease forward speed")
        print("  a/d : increase/decrease lateral speed")
        print("  q/e : rotate counter-clockwise/clockwise")
        print("  Space or k: Stop")
        print("  Ctrl-C to quit")
        try:
            while rclpy.ok():
                key = getKey(settings)
                if key == '\x03':  # Ctrl-C
                    break

                if key in moveBindings:
                    x = moveBindings[key][0] * self.linear_speed
                    y = moveBindings[key][1] * self.linear_speed
                    angular = moveBindings[key][2] * self.angular_speed
                elif key == ' ' or key == 'k':
                    # Stop motion when space or k is pressed.
                    x = 0.0
                    y = 0.0
                    angular = 0.0
                else:
                    # If no valid key, do not change command.
                    continue

                self.publish_cmd(x, y, angular)
        except Exception as e:
            self.get_logger().error(f"Error: {e}")
        finally:
            # On exit, publish zero velocities to stop the robot.
            self.publish_cmd(0.0, 0.0, 0.0)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


def main(args=None):
    rclpy.init(args=args)
    node = TeleopMecanum()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

