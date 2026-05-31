# --------------------------------------------------------------------------
# LandmarkLocalizationNode
#
# Subscribes to:
#   /landmark_detections              (LandmarkDetection)
#   /cam/depth/color/points           (PointCloud2)
#
# For each detection from detection node:
#   1. looks up the latest pointcloud from the matching camera
#   2. extracts all points inside the bounding box
#   3. takes the centroid of valid points as the object's 3D position
#   4. transforms that position from the camera depth frame -> TARGET_FRAME
#   5. deduplicates within DEDUP_DISTANCE_M metres
#   6. publishes a LocalizedLandmark
#
# Publishes:
#   /localized_landmarks              (LocalizedLandmark)
# --------------------------------------------------------------------------

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point, PointStamped

import tf2_ros
import tf2_geometry_msgs

import math
import struct

DEDUP_DISTANCE_M = 1.0

# change to 'odom' if testing
TARGET_FRAME = "map"

CAMERA_CONFIG = {
    "front": {
        "pointcloud_topic": "/front_cam/depth/color/points",
        "frame_id":         "front_cam_depth_optical_frame",
    },
    "left": {
        "pointcloud_topic": "/left_cam/depth/color/points",
        "frame_id":         "left_cam_depth_optical_frame",
    },
    "rear": {
        "pointcloud_topic": "/rear_cam/depth/color/points",
        "frame_id":         "rear_cam_depth_optical_frame",
    },
    "right": {
        "pointcloud_topic": "/right_cam/depth/color/points",
        "frame_id":         "right_cam_depth_optical_frame",
    },
}

POINT_STEP   = 16   # bytes per point (x=4, y=4, z=4 + 4 padding typical for realsense)
X_OFFSET     = 0
Y_OFFSET     = 4
Z_OFFSET     = 8


class LandmarkLocalizationNode(Node):

    def __init__(self):
        super().__init__("landmark_localization_node")

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.latest_clouds = {cam: None for cam in CAMERA_CONFIG}

        self.known_landmarks = []

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        for cam_name, config in CAMERA_CONFIG.items():
            self.create_subscription(
                PointCloud2,
                config["pointcloud_topic"],
                self._make_cloud_callback(cam_name),
                sensor_qos,
            )
            self.get_logger().info(
                f"Subscribed to {config['pointcloud_topic']}"
            )

        """
        #SUBSCRIBE BLOCK
        from <pkg_name>.msg import LandmarkDetection
        self.create_subscription(
            LandmarkDetection,
            "/landmark_detections",
            self._detection_callback,
            10,
        )

        #PUBLISH BLOCK
        from <your_pkg>.msg import LocalizedLandmark
        self.pub = self.create_publisher(
            LocalizedLandmark, "/localized_landmarks", 10
        )
        """

        self.get_logger().info(
            f"LandmarkLocalizationNode ready. "
            f"Target frame: '{TARGET_FRAME}', "
            f"Dedup distance: {DEDUP_DISTANCE_M}m"
        )

    def _make_cloud_callback(self, cam_name: str):
        def callback(msg: PointCloud2):
            self.latest_clouds[cam_name] = msg
        return callback

    def _detection_callback(self, det):

        cloud_msg = self.latest_clouds.get(det.camera_name)
        if cloud_msg is None:
            self.get_logger().warn(
                f"[{det.camera_name}] No pointcloud received yet; skipping detection."
            )
            return

        centroid = self._extract_centroid(
            cloud_msg,
            int(det.bbox_x),
            int(det.bbox_y),
            int(det.bbox_x + det.bbox_width),
            int(det.bbox_y + det.bbox_height),
        )

        if centroid is None:
            self.get_logger().warn(
                f"[{det.camera_name}] No valid points in bbox for {det.class_name}; skipping."
            )
            return

        point_in_cam = PointStamped()
        point_in_cam.header.stamp    = cloud_msg.header.stamp
        point_in_cam.header.frame_id = cloud_msg.header.frame_id  # depth optical frame
        point_in_cam.point           = centroid

        try:
            point_in_target = self.tf_buffer.transform(
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

        final_pos = point_in_target.point

        if self._is_duplicate(det.class_name, final_pos):
            self.get_logger().debug(
                f"[{det.camera_name}] Duplicate {det.class_name} "
                f"at ({final_pos.x:.2f}, {final_pos.y:.2f}, {final_pos.z:.2f}); skipped."
            )
            return

        self.known_landmarks.append({
            "class_name": det.class_name,
            "x": final_pos.x,
            "y": final_pos.y,
            "z": final_pos.z,
        })

        # TEST BLOCK
        self.get_logger().info(
            f"\n"
            f"  [NEW LANDMARK #{len(self.known_landmarks)}]\n"
            f"  camera    : {det.camera_name}\n"
            f"  class     : {det.class_name}\n"
            f"  confidence: {det.confidence:.2f}\n"
            f"  frame     : {TARGET_FRAME}\n"
            f"  position  : x={final_pos.x:.2f} y={final_pos.y:.2f} z={final_pos.z:.2f}\n"
            f"  total known: {len(self.known_landmarks)}"
        )
        # END TEST BLOCK

        # PUBLISH BLOCK
        # loc = LocalizedLandmark()
        # loc.camera_name  = det.camera_name
        # loc.class_name   = det.class_name
        # loc.confidence   = det.confidence
        # loc.frame_id     = TARGET_FRAME
        # loc.position     = final_pos
        # loc.snapshot     = det.snapshot
        # loc.stamp        = det.stamp
        # self.pub.publish(loc)

    def _extract_centroid(
        self,
        cloud_msg: PointCloud2,
        x1: int, y1: int,
        x2: int, y2: int,
    ):

        width      = cloud_msg.width
        point_step = cloud_msg.point_step
        raw        = cloud_msg.data

        # clamp bbox to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(cloud_msg.height - 1, y2)

        sum_x = 0.0
        sum_y = 0.0
        sum_z = 0.0
        count = 0

        for row in range(y1, y2 + 1):
            for col in range(x1, x2 + 1):
                offset = (row * width + col) * point_step

                x, y, z = struct.unpack_from("fff", raw, offset)

                if not math.isfinite(x) or not math.isfinite(y) or not math.isfinite(z):
                    continue
                if z <= 0.0:
                    continue

                sum_x += x
                sum_y += y
                sum_z += z
                count += 1

        if count == 0:
            return None

        centroid = Point()
        centroid.x = sum_x / count
        centroid.y = sum_y / count
        centroid.z = sum_z / count

        return centroid

    def _is_duplicate(self, class_name: str, position: Point) -> bool:
        for known in self.known_landmarks:
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
