# --------------------------------------------------------------------------
# LandmarkLoggerNode
#
# subscribes to:
#   /localized_landmarks        (mars_msgs/LocalizedLandmark)
#   /boundary/polygon           (geometry_msgs/PolygonStamped)  -- from boundary_builder_node
#   /odometry/filtered          (nav_msgs/Odometry)
#
# Filters:
#   1. Boundary polygon proximity -- rejects detections within BOUNDARY_MARKER_RADIUS_M
#           of any vertex of the boundary polygon. (to prevent detecting markers)
#   2. Camera time-window dedup -- one save per camera per N seconds.
#
# ros2 run <pkg> landmark_logger_node --ros-args -p clear_on_start:=true (for clearing on startup)
#
# dir structure:
#   ~/mission_data/
#     path.json
#     landmarks/
#       landmark_1/
#         snapshot.jpg
#         data.json
# --------------------------------------------------------------------------

import os
import json
import shutil
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from geometry_msgs.msg import PolygonStamped

from cv_bridge import CvBridge
import cv2

from mars_msgs.msg import LocalizedLandmark

SAVE_DIR = os.path.expanduser("~/mission_data")

PATH_LOG_INTERVAL_S = 1.0

BOUNDARY_MARKER_RADIUS_M = 1.0

# per-camera cooldown
CAMERA_DEDUP_WINDOW_S = 5.0


class LandmarkLoggerNode(Node):

    def __init__(self):
        super().__init__("landmark_logger_node")

        self.bridge = CvBridge()
        self.last_path_log_time = 0.0

        self.boundary_vertices: list[tuple[float, float]] = []

        self.last_saved_time: dict[str, float] = {}

        self.declare_parameter("clear_on_start", False)
        clear_on_start = self.get_parameter("clear_on_start").value

        self.landmarks_dir = os.path.join(SAVE_DIR, "landmarks")
        self.path_file     = os.path.join(SAVE_DIR, "path.json")

        if clear_on_start:
            self.get_logger().warn(
                "clear_on_start=True -- wiping all previous mission data."
            )
            if os.path.isdir(self.landmarks_dir):
                shutil.rmtree(self.landmarks_dir)
            if os.path.exists(self.path_file):
                os.remove(self.path_file)
            self.get_logger().info("Mission data cleared.")

        os.makedirs(self.landmarks_dir, exist_ok=True)
        if not os.path.exists(self.path_file):
            self._write_json(self.path_file, [])

        existing = [
            d for d in os.listdir(self.landmarks_dir)
            if os.path.isdir(os.path.join(self.landmarks_dir, d))
            and d.startswith("landmark_")
        ]
        self.landmark_count = len(existing)

        if clear_on_start:
            self.get_logger().info("Starting fresh. landmark_count=0")
        elif self.landmark_count > 0:
            self.get_logger().info(
                f"Resuming -- found {self.landmark_count} existing landmark(s)."
            )

        self.get_logger().info(f"Saving mission data to: {SAVE_DIR}")

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

        self.create_subscription(
            PolygonStamped,
            "/boundary/polygon",
            self._polygon_callback,
            10,
        )

        self.create_subscription(
            LocalizedLandmark,
            "/localized_landmarks",
            self._landmark_callback,
            10,
        )

        self.get_logger().info(
            f"LandmarkLoggerNode ready. "
            f"Boundary vertex filter: {BOUNDARY_MARKER_RADIUS_M}m, "
            f"camera dedup window: {CAMERA_DEDUP_WINDOW_S}s"
        )

    def _polygon_callback(self, msg: PolygonStamped):
        self.boundary_vertices = [
            (float(p.x), float(p.y))
            for p in msg.polygon.points
        ]

    def _odom_callback(self, msg: Odometry):
        now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if now - self.last_path_log_time < PATH_LOG_INTERVAL_S:
            return
        self.last_path_log_time = now
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        path = self._read_json(self.path_file)
        path.append({"timestamp": now, "x": x, "y": y})
        self._write_json(self.path_file, path)

    def _landmark_callback(self, loc: LocalizedLandmark):
        now = loc.stamp.sec + loc.stamp.nanosec * 1e-9
        lx, ly = loc.position.x, loc.position.y
        for (vx, vy) in self.boundary_vertices:
            dist = math.sqrt((lx - vx) ** 2 + (ly - vy) ** 2)
            if dist < BOUNDARY_MARKER_RADIUS_M:
                self.get_logger().debug(
                    f"[{loc.camera_name}] Rejected: near boundary vertex "
                    f"({vx:.2f}, {vy:.2f}), dist={dist:.2f}m"
                )
                return

        last_t = self.last_saved_time.get(loc.camera_name, 0.0)
        if now - last_t < CAMERA_DEDUP_WINDOW_S:
            self.get_logger().debug(
                f"[{loc.camera_name}] Dedup: {now - last_t:.1f}s since last save"
            )
            return

        self.last_saved_time[loc.camera_name] = now
        self.landmark_count += 1
        landmark_id  = f"landmark_{self.landmark_count}"
        landmark_dir = os.path.join(self.landmarks_dir, landmark_id)
        os.makedirs(landmark_dir, exist_ok=True)

        self._save_snapshot(loc.snapshot, landmark_dir)

        data = {
            "id":         landmark_id,
            "timestamp":  now,
            "camera":     loc.camera_name,
            "source":     loc.source,
            "class":      loc.class_name,
            "confidence": loc.confidence,
            "frame":      loc.frame_id,
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
            f"Saved {landmark_id} | camera={loc.camera_name} "
            f"source={loc.source} "
            f"pos=({loc.position.x:.2f}, {loc.position.y:.2f}, {loc.position.z:.2f})"
        )

    def _save_snapshot(self, snapshot_msg: Image, landmark_dir: str):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(snapshot_msg, "bgr8")
            cv2.imwrite(os.path.join(landmark_dir, "snapshot.jpg"), cv_image)
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
