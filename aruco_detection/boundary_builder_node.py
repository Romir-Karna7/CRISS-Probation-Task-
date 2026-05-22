import rclpy
from rclpy.node import Node
from aruco_msgs.msg import MarkerArray
from geometry_msgs.msg import PolygonStamped, Point32
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped

class BoundaryBuilderNode(Node):
    def __init__(self):
        super().__init__('boundary_builder')
        self.declare_parameter('map_frame', 'map')
        self.map_frame = self.get_parameter('map_frame').value

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(MarkerArray, '/aruco/detected_markers', self.markers_cb, 10)
        self.pub = self.create_publisher(PolygonStamped, '/boundary/polygon', 10)

        # id (x, y) in map frame
        self.known_markers: dict[int, tuple[float, float]] = {}

    def markers_cb(self, msg: MarkerArray):
        for m in msg.markers:
            # Transform pose into map frame
            ps = PoseStamped()
            ps.header = m.header
            ps.pose   = m.pose.pose
            try:
                ps_map = self.tf_buffer.transform(
                    ps,
                    self.map_frame,
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                self.known_markers[m.id] = (
                    ps_map.pose.position.x,
                    ps_map.pose.position.y,
                )
            except Exception as e:
                self.get_logger().warn(f'TF failed for marker {m.id}: {e}')

        self._publish_polygon()

    def _publish_polygon(self):
        if len(self.known_markers) < 3:
            return

        # sorting by descending ID — highest ID is at the back of robot,
        # then IDs decrease clockwise as given
        ordered = sorted(self.known_markers.items(), key=lambda kv: -kv[0])

        poly = PolygonStamped()
        poly.header.stamp    = self.get_clock().now().to_msg()
        poly.header.frame_id = self.map_frame

        for _, (x, y) in ordered:
            p = Point32()
            p.x, p.y = float(x), float(y)
            poly.polygon.points.append(p)

        self.pub.publish(poly)
        self.get_logger().info(
            f'Boundary polygon updated: {len(ordered)} markers', throttle_duration_sec=5.0)


def main():
    rclpy.init()
    rclpy.spin(BoundaryBuilderNode())
    rclpy.shutdown()