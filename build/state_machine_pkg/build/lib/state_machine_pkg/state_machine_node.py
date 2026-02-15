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
        # Intial State
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

        #result = self.detector.analyze(msg.ranges)

        #self.update_state(result)

        self.state = self.classify_environment(proc_ranges, center_index)
        self.get_logger().info(f"current STATE: {self.state}")

        steering, speed = self.compute_control()

        self.publish_drive(steering, speed)


    def classify_environment(self, proc_ranges, center_index):
        """
        Returns one of:
        "STOP", "WALL_FOLLOW", "FOLLOW_GAP"
        """

        front_window = 20
        side_window = 80

        front = proc_ranges[center_index - front_window :
                            center_index + front_window]

        left = proc_ranges[:side_window]
        right = proc_ranges[-side_window:]

        front_mean = np.mean(front)
        left_mean = np.mean(left)
        right_mean = np.mean(right)

        # 🚨 STOP condition
        if front_mean < 0.5:
            return "STOP"

        # 🟢 Straight corridor
        if front_mean > 2.5 and abs(left_mean - right_mean) < 0.3:
            return "WALL_FOLLOW"

        # 🟡 Corner / obstacle
        return "FOLLOW_GAP"

    # ==================================================
    # State Logic
    # ==================================================
    def update_state(self, result):

        new_state = self.state  # default = no change

        if result["is_cornering"] or result["has_gap"]:
            new_state = "FOLLOW_GAP"
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
    def compute_control(self):

        if self.state == "FOLLOW_GAP":

            #best_idx = result["best_gap_index"]

            #if best_idx is None:
            #    return 0.0, 0.0

            #angle = (best_idx - center_index) * self.detector.radians_per_elem

            angle = 0.0

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
