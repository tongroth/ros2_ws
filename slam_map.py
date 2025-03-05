#!/usr/bin/env python3
"""
Save Map Command Node

This node listens for keyboard input. When the user presses the 's' key,
it calls the slam_toolbox save_map service to save the current map.
"""

import sys
import select
import termios
import tty

import rclpy
from rclpy.node import Node

# Import the service type from slam_toolbox.
# Adjust the import if your installation uses a different package/service type.
from slam_toolbox.srv import SaveMap

def getKey(settings):
    """Capture a single key press (non-blocking)."""
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

class SaveMapCommand(Node):
    def __init__(self):
        super().__init__('save_map_command')
        # Create a client for the slam_toolbox save_map service.
        self.cli = self.create_client(SaveMap, '/slam_toolbox/save_map')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /slam_toolbox/save_map service...')
        self.get_logger().info("SaveMapCommand node started. Press 's' to save the map.")

    def send_save_map_request(self):
        """Send a request to save the map via slam_toolbox."""
        req = SaveMap.Request()
        # If your service has parameters, set them here.
        future = self.cli.call_async(req)
        future.add_done_callback(self.save_map_response_callback)

    def save_map_response_callback(self, future):
        try:
            response = future.result()
            if response:
                self.get_logger().info(f"Map saved successfully: {response}")
            else:
                self.get_logger().error("Failed to save map.")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SaveMapCommand()
    settings = termios.tcgetattr(sys.stdin)
    try:
        print("Press 's' to save the map, Ctrl-C to exit.")
        while rclpy.ok():
            key = getKey(settings)
            if key == '\x03':  # Ctrl-C
                break
            if key == 's':
                node.get_logger().info("Saving map...")
                node.send_save_map_request()
    except Exception as e:
        node.get_logger().error(f"Error: {e}")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

