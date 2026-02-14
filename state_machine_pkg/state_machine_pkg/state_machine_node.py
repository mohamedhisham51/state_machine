import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import numpy as np

from .gap_corner_detector import GapCornerDetector


class StateMachineNode(Node):

    def __init__(self):
        super().__init__("state_machine_node")

        # -----------------------------
        # Detector
        # -----------------------------
        self.detector = GapCornerDetector()

        # -----------------------------
        # State
        # -----------------------------
        self.state = "GAP_FOLLOW"

        # -----------------------------
        # ROS Interfaces
        # -----------------------------
        self.scan_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10,
        )

        self.drive_pub = self.create_publisher(
            AckermannDriveStamped,
            "/drive",
            10,
        )

        self.get_logger().info("State Machine Node Started")

    # ==================================================
    # Main Callback
    # ==================================================
    def scan_callback(self, msg: LaserScan):

        # Preprocess once
        proc_ranges, center_index = self.detector.preprocess_lidar(msg.ranges)

        result = self.detector.analyze(msg.ranges)

        self.update_state(result)

        steering, speed = self.compute_control(
            result,
            center_index
        )

        self.publish_drive(steering, speed)


    # ==================================================
    # State Logic
    # ==================================================
    def update_state(self, result):

        new_state = self.state  # default = no change

        if result["is_cornering"] or result["has_gap"]:
            new_state = "GAP_FOLLOW"
        else:
            new_state = "WALL_FOLLOW"

        # Log only if state changed
        if new_state != self.state:
            self.get_logger().info(f"STATE CHANGE: {self.state} → {new_state}")

        self.prev_state = self.state
        self.state = new_state


    # ==================================================
    # Control Logic
    # ==================================================
    def compute_control(self, result, center_index):

        if self.state == "GAP_FOLLOW":

            best_idx = result["best_gap_index"]

            if best_idx is None:
                return 0.0, 0.0

            angle = (best_idx - center_index) * self.detector.radians_per_elem

            speed = 1.0
            return float(angle), speed

        elif self.state == "WALL_FOLLOW":
            return 0.0, 1.5

        else:
            return 0.0, 0.0


    # ==================================================
    # Publisher
    # ==================================================
    def publish_drive(self, steering, speed):

        msg = AckermannDriveStamped()
        msg.drive.steering_angle = steering
        msg.drive.speed = speed

        self.drive_pub.publish(msg)


# ======================================================
# Main
# ======================================================
def main(args=None):
    rclpy.init(args=args)
    node = StateMachineNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
