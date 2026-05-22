# --------------------------------------------------------------------------
# LandmarkDetectionNode
#
# subscribes to all 4 camera image + depth topics and runs yoloV11 on each frame
# for each detection, samples depth at the bounding box centre and estimates a 3D position in the camera frame
#
# topics subscribed:
#   /front_cam/color/image_raw        (sensor_msgs/Image)
#   /front_cam/depth/image_rect_raw   (sensor_msgs/Image)
#   /left_cam/color/image_raw         (sensor_msgs/Image)
#   /left_cam/depth/image_rect_raw    (sensor_msgs/Image)
#   /rear_cam/color/image_raw         (sensor_msgs/Image)
#   /rear_cam/depth/image_rect_raw    (sensor_msgs/Image)
#   /right_cam/color/image_raw        (sensor_msgs/Image)
#   /right_cam/depth/image_rect_raw   (sensor_msgs/Image)
#
# topics published:
#   /landmark_detections              (LandmarkDetection)
# --------------------------------------------------------------------------

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point

import cv2
from cv_bridge import CvBridge
import numpy as np
from ultralytics import YOLO
import message_filters

CONFIDENCE_THRESHOLD = 0.45

CAMERAS = {
    "front": "/front_cam",
    "left":  "/left_cam",
    "rear":  "/rear_cam",
    "right": "/right_cam",
}


class LandmarkDetectionNode(Node):

    def __init__(self):
        super().__init__("landmark_detection_node")

        self.bridge = CvBridge()
        self.model = YOLO("yolo11n.pt")     # swap to yolo11s.pt or yolo11m.pt for better accuracy if needed.
        self.get_logger().info("YOLOv11 model loaded.")

        # QoS profile matching the sensor data (same as husarion tutorials)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # PUBLISH BLOCK (uncomment after pkg build)
        # from <pkg_name>.msg import LandmarkDetection
        # self.detection_pub = self.create_publisher(
        #     LandmarkDetection, "/landmark_detections", 10
        # )

        for cam_name, topic_prefix in CAMERAS.items():
            color_topic = f"{topic_prefix}/color/image_raw"
            depth_topic = f"{topic_prefix}/depth/image_rect_raw"

            color_sub = message_filters.Subscriber(
                self, Image, color_topic, qos_profile=sensor_qos
            )
            depth_sub = message_filters.Subscriber(
                self, Image, depth_topic, qos_profile=sensor_qos
            )

            sync = message_filters.ApproximateTimeSynchronizer(
                [color_sub, depth_sub], queue_size=5, slop=0.1
            )
            sync.registerCallback(self._make_callback(cam_name))

            self.get_logger().info(
                f"Subscribed to {color_topic} and {depth_topic}"
            )

        self.get_logger().info("LandmarkDetectionNode ready, waiting for frames...")


    def _make_callback(self, cam_name: str):
        def callback(color_msg: Image, depth_msg: Image):
            self._process_frame(cam_name, color_msg, depth_msg)
        return callback

    def _process_frame(self, cam_name: str, color_msg: Image, depth_msg: Image):
        # convert ROS Image -> OpenCV (from husarion opencv tutorial)
        try:
            color_frame = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_frame = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )  # depth image is 32-bit float, each pixel = distance in metres
        except Exception as e:
            self.get_logger().error(f"[{cam_name}] cv_bridge error: {e}")
            return

        results = self.model(color_frame, verbose=False)

        for box in results[0].boxes:
            confidence = float(box.conf[0])
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            # pixel coordinates of bounding box corners
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

            class_name = self.model.names[int(box.cls[0])]

            # centre of the bounding box for depth sampling
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            depth_m    = self._sample_depth(depth_frame, cx, cy)
            position_3d = self._estimate_3d_position(cx, cy, depth_m, color_frame.shape)

            # crop the detected object out of the colour frame
            snapshot_crop = color_frame[
                max(0, y1) : min(color_frame.shape[0], y2),
                max(0, x1) : min(color_frame.shape[1], x2),
            ]
            snapshot_msg = self.bridge.cv2_to_imgmsg(snapshot_crop, "bgr8")

            # TEST BLOCK (print directly to terminal)
            self.get_logger().info(
                f"\n"
                f"  camera    : {cam_name}\n"
                f"  class     : {class_name}\n"
                f"  confidence: {confidence:.2f}\n"
                f"  bbox      : x={x1} y={y1} w={x2-x1} h={y2-y1}\n"
                f"  depth     : {depth_m:.2f} m\n"
                f"  position  : x={position_3d.x:.2f} y={position_3d.y:.2f} z={position_3d.z:.2f}"
            )
            # END TEST BLOCK

            # PUBLISH BLOCK (replace test w ts after pkg build)
            # det = LandmarkDetection()
            # det.camera_name  = cam_name
            # det.class_name   = class_name
            # det.confidence   = confidence
            # det.bbox_x       = float(x1)
            # det.bbox_y       = float(y1)
            # det.bbox_width   = float(x2 - x1)
            # det.bbox_height  = float(y2 - y1)
            # det.depth_m      = depth_m
            # det.position_3d  = position_3d
            # det.snapshot     = snapshot_msg
            # det.stamp        = color_msg.header.stamp
            # self.detection_pub.publish(det)

    def _sample_depth(self, depth_frame: np.ndarray, cx: int, cy: int, window: int = 5) -> float:
        h, w = depth_frame.shape[:2]

        x0 = max(0, cx - window)
        x1 = min(w, cx + window)
        y0 = max(0, cy - window)
        y1 = min(h, cy + window)

        patch = depth_frame[y0:y1, x0:x1].astype(np.float32)
        valid = patch[(patch > 0) & np.isfinite(patch)]

        if len(valid) == 0:
            return float("nan")

        return float(np.median(valid))

    def _estimate_3d_position(self, cx: int, cy: int, depth_m: float, frame_shape: tuple) -> Point:
        #rough 3d position in camera frame using a simple pinhole model.

        point = Point()

        if np.isnan(depth_m):
            return point  # returns (0, 0, 0) - caller can check depth_m for nan

        h, w = frame_shape[:2]

        fov_h = np.radians(90.0)
        fx    = (w / 2.0) / np.tan(fov_h / 2.0)
        fy    = fx
        ppx   = w / 2.0
        ppy   = h / 2.0

        point.x = float((cx - ppx) * depth_m / fx)
        point.y = float((cy - ppy) * depth_m / fy)
        point.z = float(depth_m)

        return point

def main(args=None):
    rclpy.init(args=args)
    node = LandmarkDetectionNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
