#!/usr/bin/env python3
# --------------------------------------------------------------------------
# LandmarkLoggerNode                 -- saves all the data to disk for the vlm and report generator to use
#
# Subscribes to:
#   /localized_landmarks     (LocalizedLandmark)
#   /odometry/filtered       (nav_msgs/Odometry)
#
# dir structure:
#   ~/mission_data/
#     path.json                      -- robot path, updated every PATH_LOG_INTERVAL_S
#     landmarks/
#       landmark_1/
#         snapshot.jpg               -- cropped image of the detection
#         data.json                  -- position, camera, confidence, timestamp
#       landmark_2/
# --------------------------------------------------------------------------

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

from cv_bridge import CvBridge
import cv2

import os
import json
from mission_interfaces.msg import LocalizedLandmark

SAVE_DIR = os.path.expanduser("~/mission_data")

PATH_LOG_INTERVAL_S = 1.0

class LandmarkLoggerNode(Node):

    def __init__(self):
        super().__init__("landmark_logger_node")

        self.bridge = CvBridge()
        self.last_path_log_time = 0.0

        self.landmarks_dir = os.path.join(SAVE_DIR, "landmarks")
        os.makedirs(self.landmarks_dir, exist_ok=True)
        existing = [
            d for d in os.listdir(self.landmarks_dir)
            if os.path.isdir(os.path.join(self.landmarks_dir, d))
            and d.startswith("landmark_")
        ]

        self.landmark_count = len(existing)
        if self.landmark_count > 0:
            self.get_logger().info(
                f"Resuming -- found {self.landmark_count} existing landmark(s)."
            )

        self.path_file = os.path.join(SAVE_DIR, "path.json")
        
        # Load the path into memory ONCE at startup
        if os.path.exists(self.path_file):
            self.path_data = self._read_json(self.path_file)
            self.get_logger().info(f"Resuming path with {len(self.path_data)} points.")
        else:
            self.path_data = []
            self._write_json(self.path_file, self.path_data)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            Odometry,
            "/odometry/filtered",
            self._odom_callback,
            sensor_qos,
        )

       #SUBSCRIBE BLOCK
        self.create_subscription(
            LocalizedLandmark,
            "/localized_landmarks",
            self._landmark_callback,
            10,
        )

        self.get_logger().info("LandmarkLoggerNode ready.")

    def _odom_callback(self, msg: Odometry):

        now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if now - self.last_path_log_time < PATH_LOG_INTERVAL_S:
            return

        self.last_path_log_time = now

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Append to memory, then overwrite file (No reading!)
        self.path_data.append({
            "timestamp": now,
            "x": x,
            "y": y,
        })
        self._write_json(self.path_file, self.path_data)

    def _landmark_callback(self, loc):

        self.landmark_count += 1
        landmark_id = f"landmark_{self.landmark_count}"
        landmark_dir = os.path.join(self.landmarks_dir, landmark_id)
        os.makedirs(landmark_dir, exist_ok=True)

        self._save_snapshot(loc.snapshot, landmark_dir)

        timestamp = loc.stamp.sec + loc.stamp.nanosec * 1e-9
        data = {
            "id":         landmark_id,
            "timestamp":  timestamp,
            "camera":     loc.camera_name,
            "class":      loc.class_name,      # "unknown" until VLM identifies it
            "confidence": loc.confidence,
            "frame":      loc.frame_id,        # "odom" or "map"
            "position": {
                "x": loc.position.x,
                "y": loc.position.y,
                "z": loc.position.z,
            },
            "vlm_description":   "",
            "vlm_mars_relation": "",
        }
        self._write_json(os.path.join(landmark_dir, "data.json"), data)

        self.get_logger().info(
            f"Saved {landmark_id} -- "
            f"pos=({loc.position.x:.2f}, {loc.position.y:.2f}, {loc.position.z:.2f}) "
            f"camera={loc.camera_name}"
        )

    def _save_snapshot(self, snapshot_msg: Image, landmark_dir: str):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(snapshot_msg, "bgr8")
            path = os.path.join(landmark_dir, "snapshot.jpg")
            cv2.imwrite(path, cv_image)
        except Exception as e:
            self.get_logger().error(f"Failed to save snapshot: {e}")

    def _write_json(self, filepath: str, data):
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def _read_json(self, filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)

def main(args=None):
    rclpy.init(args=args)
    node = LandmarkLoggerNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
