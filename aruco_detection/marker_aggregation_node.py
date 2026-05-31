#!/usr/bin/env python3
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
            self.get_logger().info(f"Listening for detections on: /aruco/markers/{cam}")

        self.pub = self.create_publisher(
            MarkerArray, '/aruco/detected_markers', 10)

        # flushing at 10 Hz
        self.create_timer(0.1, self.flush)

    def marker_cb(self, msg: MarkerArray, camera: str):
        now = self.get_clock().now() # changed how to get time so that can match gazebo time
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
        now = self.get_clock().now()
        out = MarkerArray()
        stale = []
        for mid, (ts, m) in list(self._buffer.items()):
            time = (now - ts).nanoseconds / 1e9 # as the type of now - its a duration object so need to extract time from it 
            if time < DEDUP_WINDOW_SEC: 
            ## if now - ts < DEDUP_WINDOW_SEC * 10:  # keeping for 2 secs
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

if __name__ == '__main__':
    main()