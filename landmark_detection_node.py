# --------------------------------------------------------------------------
# LandmarkDetectionNode
#
# two independent detectors run on every synchronised colour+depth frame:
#
#   1. YOLO-World: zero-shot object detector driven by text prompts
#
#   2. Background anomaly detector: builds a running average of what
#      normal terrain looks like per camera. Flags any region that
#      differs significantly from that average and is large enough to
#      be a physical landmark
#
# Subscribes:
#   /cam/color/image_raw          (sensor_msgs/Image)
#   /cam/depth/image_rect_raw     (sensor_msgs/Image)
#
# Publishes:
#   /landmark_detections          (LandmarkDetection)
# --------------------------------------------------------------------------

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point

import cv2
from cv_bridge import CvBridge
import numpy as np
from ultralytics import YOLOWorld
import message_filters

# tunable parameters
CONFIDENCE_THRESHOLD = 0.30
YOLO_PROMPTS = [
    "equipment",
    "tool",
    "container",
    "structure",
    "object",
    "box",
    "pole",
    "device",
]
# min contour area in pixels for the anomaly detector
# 1280x720 image = 921600 pixels total; 0.5% = 4600 pixels
MIN_CONTOUR_AREA = 4600

BACKGROUND_LEARNING_RATE = 0.01

ANOMALY_THRESHOLD = 40.0

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
        self.model = YOLOWorld("yolov8s-world.pt")
        self.model.set_classes(YOLO_PROMPTS)
        self.get_logger().info(f"YOLO-World loaded. Prompts: {YOLO_PROMPTS}")

        self.backgrounds = {cam: None for cam in CAMERAS}

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # PUBLISH BLOCK
        # from <pkg_name>.msg import LandmarkDetection
        # self.detection_pub = self.create_publisher(
        #      LandmarkDetection, "/landmark_detections", 10
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

        self.get_logger().info("LandmarkDetectionNode ready; waiting for frames...")

    def _make_callback(self, cam_name: str):
        def callback(color_msg: Image, depth_msg: Image):
            self._process_frame(cam_name, color_msg, depth_msg)
        return callback

    def _process_frame(self, cam_name: str, color_msg: Image, depth_msg: Image):

        try:
            color_frame = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_frame = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )
        except Exception as e:
            self.get_logger().error(f"[{cam_name}] cv_bridge error: {e}")
            return

        yolo_detections     = self._run_yolo(color_frame)
        anomaly_detections  = self._run_anomaly_detector(cam_name, color_frame)

        all_detections = self._merge_detections(yolo_detections, anomaly_detections)

        for (x1, y1, x2, y2, source) in all_detections:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            depth_m     = self._sample_depth(depth_frame, cx, cy)
            position_3d = self._estimate_3d_position(cx, cy, depth_m, color_frame.shape)

            snapshot_crop = color_frame[
                max(0, y1) : min(color_frame.shape[0], y2),
                max(0, x1) : min(color_frame.shape[1], x2),
            ]
            snapshot_msg = self.bridge.cv2_to_imgmsg(snapshot_crop, "bgr8")

            # TEST BLOCK
            self.get_logger().info(
                f"\n"
                f"  camera    : {cam_name}\n"
                f"  source    : {source}\n"
                f"  class     : unknown\n"
                f"  bbox      : x={x1} y={y1} w={x2-x1} h={y2-y1}\n"
                f"  depth     : {depth_m:.2f} m\n"
                f"  position  : x={position_3d.x:.2f} y={position_3d.y:.2f} z={position_3d.z:.2f}"
            )
            # END TEST BLOCK

            # PUBLISH BLOCK
            # det = LandmarkDetection()
            # det.camera_name  = cam_name
            # det.class_name   = "unknown"
            # det.confidence   = 1.0  # anomaly detector has no confidence score
            # det.bbox_x       = float(x1)
            # det.bbox_y       = float(y1)
            # det.bbox_width   = float(x2 - x1)
            # det.bbox_height  = float(y2 - y1)
            # det.depth_m      = depth_m
            # det.position_3d  = position_3d
            # det.snapshot     = snapshot_msg
            # det.stamp        = color_msg.header.stamp
            # self.detection_pub.publish(det)

    def _run_yolo(self, color_frame: np.ndarray) -> list:
        results = self.model(color_frame, verbose=False)
        detections = []

        for box in results[0].boxes:
            confidence = float(box.conf[0])
            if confidence < CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            detections.append((x1, y1, x2, y2, "yolo"))
        return detections

    def _run_anomaly_detector(self, cam_name: str, color_frame: np.ndarray) -> list:
        # converts to grayscale (lf intensity diff)
        gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if self.backgrounds[cam_name] is None:
            self.backgrounds[cam_name] = gray.copy()
            return []

        cv2.accumulateWeighted(gray, self.backgrounds[cam_name], BACKGROUND_LEARNING_RATE)

        diff = cv2.absdiff(gray, self.backgrounds[cam_name])

        _, mask = cv2.threshold(diff, ANOMALY_THRESHOLD, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask   = cv2.dilate(mask, kernel, iterations=2)
        mask   = cv2.erode(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            detections.append((x, y, x + w, y + h, "anomaly"))

        return detections

    def _merge_detections(self, yolo_detections: list, anomaly_detections: list) -> list:

        all_detections = list(yolo_detections)

        for a_box in anomaly_detections:
            overlaps = False
            for y_box in yolo_detections:
                if self._iou(a_box, y_box) > 0.3:
                    overlaps = True
                    break
            if not overlaps:
                all_detections.append(a_box)

        return all_detections

    def _iou(self, box_a: tuple, box_b: tuple) -> float:
        # computes intersection over union between two bounding boxes
        # each box is (x1, y1, x2, y2, source)
        ax1, ay1, ax2, ay2 = box_a[:4]
        bx1, by1, bx2, by2 = box_b[:4]

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union  = area_a + area_b - intersection

        if union == 0:
            return 0.0

        return intersection / union

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
        # rough pinhole estimate
        point = Point()
        if np.isnan(depth_m):
            return point
        h, w  = frame_shape[:2]
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
