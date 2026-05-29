"""
TBD: Include node description and usage instructions here.
Sources used:
https://github.com/JMU-ROBOTICS-VIVA/ros2_aruco/blob/main/ros2_aruco/ros2_aruco/aruco_node.py
https://github.com/AIRLab-POLIMI/ros2-aruco-pose-estimation/blob/main/aruco_pose_estimation/aruco_pose_estimation/pose_estimation.py

"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from aruco_msgs.msg import MarkerArray, Marker
from cv_bridge import CvBridge
import cv2
import numpy as np

class ArucoDetectionNode(Node):
    def __init__(self):
        super().__init__("aruco_detection_node")

        # Declare parameters (override specifics from YAML file for each camera)
        self.declare_parameters(
            namespace = '',
            parameters = [
                (
                    "camera_name",
                    "front",
                    ParameterDescriptor(
                        type = ParameterType.PARAMETER_STRING,
                        description = "Name of camera to subscribe to"
                    )
                ),
                (
                    "marker_size",
                    0.1,
                    ParameterDescriptor(
                        type = ParameterType.PARAMETER_DOUBLE,
                        description = "Size of aruco markers in meters"
                    )
                ),
                (
                    "aruco_dict",
                    "DICT_4X4_100",
                    ParameterDescriptor(
                        type = ParameterType.PARAMETER_STRING,
                        description = "Aruco dictionary being used"
                    )
                ),
                (
                    "image_topic",
                    "/cam/front/image_raw",
                    ParameterDescriptor(
                        type = ParameterType.PARAMETER_STRING,
                        description = "Topic to subscribe for camera images"
                    )
                ),
                (
                    "camera_info_topic",
                    "/cam/front/camera_info",
                    ParameterDescriptor(
                        type = ParameterType.PARAMETER_STRING,
                        description = "Topic to subscribe for camera info"
                    )
                )
            ]
        )

        # Get parameters
        camera = self.get_parameter("camera_name").value
        marker_size = self.get_parameter('marker_size').value
        aruco_dict_name = self.get_parameter('aruco_dict').value
        
        self.get_logger().info(f"Camera: {camera}")
        self.get_logger().info(f"Marker Size: {marker_size}")
        self.get_logger().info(f"Aruco Dictionary: {aruco_dict_name}")

        self.marker_size = marker_size
        self.bridge = CvBridge()
        self.intrinsic_matrix = None   # intrinsic matrix
        self.dist_coeffs = None   # distortion coefficients
        
        # Map of aruco dictionary names to OpenCV constants -- ADD MORE AS NEEDED, ALSO CHECK WHETHER AUTO-DETECTION IS NEEDED OR NOT
        arucodict_map = {
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100
        }

        aruco_dict = cv2.aruco.getPredefinedDictionary(arucodict_map[aruco_dict_name])
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, self.detector_params)

        
        # Set up subsscriptions to camera image and info topics 
        img_topic  = self.get_parameter('image_topic').value
        info_topic = self.get_parameter('camera_info_topic').value
        self.image_sub  = self.create_subscription(Image, img_topic, self.image_cb, 10)
        self.info_sub = self.create_subscription(CameraInfo, info_topic, self.info_cb, 10)

        # set up publisher
        self.pub = self.create_publisher(MarkerArray, f'/aruco/markers/{camera}', 10)
        self.get_logger().info(f'[aruco_detector/{camera}] is ready!!!')



    def info_cb(self, msg: CameraInfo):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs   = np.array(msg.d)

    def image_cb(self, msg: Image):
        if self.camera_matrix is None:
            return  # wait for calibration

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return

        out = MarkerArray()
        out.header = msg.header


        marker_length = self.marker_size
        half_len = marker_length / 2.0
        object_points = np.array([
            [-half_len,  half_len, 0],
            [ half_len,  half_len, 0],
            [ half_len, -half_len, 0],
            [-half_len, -half_len, 0]
        ], dtype=np.float32)
        
        for i, marker_id in enumerate(ids.flatten()):
            image_points = corners[i][0].astype(np.float32)
            success, rvec, tvec = cv2.solvePnP(
                object_points, image_points, self.camera_matrix, self.dist_coeffs
            )
            if not success:
                continue
            m = Marker()
            m.header  = msg.header
            m.id      = int(marker_id)
            m.pose.pose.position.x = float(tvec[0])
            m.pose.pose.position.y = float(tvec[1])
            m.pose.pose.position.z = float(tvec[2])
            rot_mat, _ = cv2.Rodrigues(rvec)
            m.pose.pose.orientation = self._rot_to_quat(rot_mat)
            out.markers.append(m)

        self.pub.publish(out)

    # Helper function to convert rotation matrix to quaternion
    # CHECK WHETHER SHOULD USE SCIPY OR SELF-IMPLEMENTATION INSTEAD OF ROS TF TRANSFORMATIONS
    @staticmethod
    def _rot_to_quat(R):
        import numpy as np
        from geometry_msgs.msg import Quaternion
        import tf_transformations

        matrix_4x4 = np.eye(4)
        matrix_4x4[0:3, 0:3] = R 

        raw_quat = tf_transformations.quaternion_from_matrix(matrix_4x4)

        return Quaternion(x=raw_quat[0], y=raw_quat[1], z=raw_quat[2], w=raw_quat[3])
    
def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
