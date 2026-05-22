# ------------------------------------------------------------------------
# LandmarkLocalizationNode
#
# for each detection:
#   1. transforms position_3d from the camera optical frame -> odom frame
#      using the TF tree.
#   2. deduplicates if a landmark of the same class already exists within
#      DEDUP_DISTANCE_M metres in odom frame, the detection is dropped.
#   3. publishes a LocalizedLandmark message on /localized_landmarks.
#
# topics subscribed:
#   /landmark_detections     (LandmarkDetection)
#
# topics published:
#   /localized_landmarks     (LocalizedLandmark)
# ------------------------------------------------------------------------

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Point, PointStamped

import tf2_ros
import tf2_geometry_msgs

import math

# min dist b/w two detections (in metres) to be considered the same landmark, might need tuning
DEDUP_DISTANCE_M = 1.0

# change to "map" once slam_toolbox is running and publishing a map frame
TARGET_FRAME = "odom"

CAMERA_OPTICAL_FRAMES = {
    "front": "front_cam_color_optical_frame",
    "left":  "left_cam_color_optical_frame",
    "rear":  "rear_cam_color_optical_frame",
    "right": "right_cam_color_optical_frame",
}


class LandmarkLocalizationNode(Node):

    def __init__(self):
        super().__init__("landmark_localization_node")

        # tf buffer + listener (from husarion tf tutorial)
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.known_landmarks = []

        # PUBLISH BLOCK (uncomment after pkg build)
        # from <pkg_name>.msg import LocalizedLandmark
        # self.pub = self.create_publisher(LocalizedLandmark, "/localized_landmarks", 10)

        # SUBSCRIBE BLOCK (uncomment after pkg build)
        # from <pkg_name>.msg import LandmarkDetection
        # self.sub = self.create_subscription(
        #     LandmarkDetection,
        #     "/landmark_detections",
        #     self._detection_callback,
        #     10,
        # )

        self.get_logger().info(
            f"LandmarkLocalizationNode ready. "
            f"Target frame: '{TARGET_FRAME}', "
            f"Dedup distance: {DEDUP_DISTANCE_M}m"
        )

    def _detection_callback(self, det):
        source_frame = CAMERA_OPTICAL_FRAMES.get(det.camera_name)
        if source_frame is None:
            self.get_logger().warn(
                f"Unknown camera name '{det.camera_name}', skipping."
            )
            return

        point_in_cam = PointStamped()
        point_in_cam.header.stamp = det.stamp
        point_in_cam.header.frame_id = source_frame
        point_in_cam.point = det.position_3d

        # transform the point from camera optical frame -> TARGET_FRAME
        try:
            point_in_odom = self.tf_buffer.transform(
                point_in_cam,
                TARGET_FRAME,
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
        except tf2_ros.LookupException as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return
        except tf2_ros.ExtrapolationException as e:
            self.get_logger().warn(f"TF extrapolation failed: {e}")
            return

        odom_pos = point_in_odom.point

        # deduplicate
        if self._is_duplicate(det.class_name, odom_pos):
            self.get_logger().debug(
                f"[{det.camera_name}] Duplicate {det.class_name} "
                f"at ({odom_pos.x:.2f}, {odom_pos.y:.2f}, {odom_pos.z:.2f}), skipped."
            )
            return

        self.known_landmarks.append({
            "class_name": det.class_name,
            "x": odom_pos.x,
            "y": odom_pos.y,
            "z": odom_pos.z,
        })

        # TEST BLOCK (straight to terminal)
        self.get_logger().info(
            f"\n"
            f"  [NEW LANDMARK #{len(self.known_landmarks)}]\n"
            f"  camera    : {det.camera_name}\n"
            f"  class     : {det.class_name}\n"
            f"  confidence: {det.confidence:.2f}\n"
            f"  frame     : {TARGET_FRAME}\n"
            f"  position  : x={odom_pos.x:.2f} y={odom_pos.y:.2f} z={odom_pos.z:.2f}\n"
            f"  total known landmarks: {len(self.known_landmarks)}"
        )
        # END TEST BLOCK

        # PUBLISH BLOCK (uncomment after pkg build)
        # loc = LocalizedLandmark()
        # loc.camera_name  = det.camera_name
        # loc.class_name   = det.class_name
        # loc.confidence   = det.confidence
        # loc.frame_id     = TARGET_FRAME
        # loc.position     = odom_pos
        # loc.snapshot     = det.snapshot
        # loc.stamp        = det.stamp
        # self.pub.publish(loc)

    def _is_duplicate(self, class_name: str, position: Point) -> bool:

        for known in self.known_landmarks:
            if known["class_name"] != class_name:
                continue

            dist = math.sqrt(
                (position.x - known["x"]) ** 2 +
                (position.y - known["y"]) ** 2 +
                (position.z - known["z"]) ** 2
            )

            if dist < DEDUP_DISTANCE_M:
                return True

        return False

def main(args=None):
    rclpy.init(args=args)
    node = LandmarkLocalizationNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
