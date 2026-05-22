import rclpy
from rclpy.node import Node
from aruco_msgs.msg import MarkerArray, Marker
from collections import defaultdict
import time

CAMERAS = ['front', 'back', 'left', 'right']
DEDUP_WINDOW_SEC = 0.2   # merging detections within this window, eill change if necessary

class MarkerAggregatorNode(Node):
    def __init__(self):
        super().__init__('marker_aggregator')
        self._buffer: dict[int, tuple[float, Marker]] = {}  # id is (timestamp, marker)

        for cam in CAMERAS:
            self.create_subscription(
                MarkerArray, f'/aruco/markers/{cam}',
                lambda msg, c=cam: self.marker_cb(msg, c), 10)

        self.pub = self.create_publisher(
            MarkerArray, '/aruco/detected_markers', 10)

        # flushing at 10 Hz
        self.create_timer(0.1, self.flush)

    def marker_cb(self, msg: MarkerArray, camera: str):
        now = time.time()
        for m in msg.markers:
            mid = m.id
            if mid not in self._buffer:
                self._buffer[mid] = (now, m)
            else:
                _, existing = self._buffer[mid]
                # keeping closer (lower Z) detection
                if m.pose.pose.position.z < existing.pose.pose.position.z:
                    self._buffer[mid] = (now, m)

    def flush(self):
        now = time.time()
        out = MarkerArray()
        stale = []
        for mid, (ts, m) in self._buffer.items():
            if now - ts < DEDUP_WINDOW_SEC * 10:  # keeping for 2 secs
                out.markers.append(m)
            else:
                stale.append(mid)
        for mid in stale:
            del self._buffer[mid]
        if out.markers:
            self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(MarkerAggregatorNode())
    rclpy.shutdown()