#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient 
import math
from aruco_msgs.msg import MarkerArray # for aruco detection message type 
from geometry_msgs.msg import PoseStamped # as tf2 do transform pose requires geometry msg type 
from nav2_msgs.action import NavigateToPose #nav2 being the action server - contains the goal message type

from tf2_ros import Buffer, TransformListener  
import tf2_geometry_msgs

class ArucoNavigation(Node):
    def __init__(self):
        super().__init__('aruco_navigation')

        self.tf_buffer = Buffer() # to store transforms of last 10 seconds 
        self.tf_listener = TransformListener(self.tf_buffer, self) # start listening to transforms immediately

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose') #setting up the action client 
        
        self.subscription = self.create_subscription(  
            MarkerArray, '/aruco/detected_markers', self.aruco_callback, 10 ) # subscribing to aruco detection topic

        self.is_navigating = False  # in order to track if navigating is going on or not, to decide when to look for markers
        
        self.visited_ids = set() # list of visited ids so as to not go back to them (set instead of list to prevent duplicates) 
        self.last_target_id = float('inf') 
        
        self.get_logger().info("Aruco Navigation to begin. Waiting to detect markers...")

    def aruco_callback(self, msg):
        if not msg.markers or self.is_navigating:  # to avoid navigation if no markers or already navigating
            return

        valid_unvisited_markers = [m for m in msg.markers if m.id not in self.visited_ids] # filter out visited markers

        if not valid_unvisited_markers: # if looking at all visited ids, do nothing and wait for new detections
            return

        target_marker = None
        lower_id_markers = [m for m in valid_unvisited_markers if m.id < self.last_target_id] # filter out markers with IDs lower than the last target ID

        if lower_id_markers:
            target_marker = max(lower_id_markers, key=lambda m: m.id)
        else:
            return

        self.is_navigating = True
        self.last_target_id = target_marker.id
        self.get_logger().info(f"Next Target acquired. ID is {target_marker.id}.")

        self.send_goal(target_marker)

    def send_goal(self, marker):
        try: #added to avoid error when TF2 math or network fails, which was happening when the robot was moving and the transform from camera to map was not available for a short time
            transform = self.tf_buffer.lookup_transform(
                'map', 
                marker.header.frame_id, 
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.5) # Transform from camera to map (WAIT up to 1.5 seconds for it to exist)
            )

            raw_pose = PoseStamped()
            raw_pose.header = marker.header
            raw_pose.pose = marker.pose.pose

            standoff = 1.0 # changeble (how far from the markers we want to stop)
            if raw_pose.pose.position.z > (standoff + 0.2): # if marker is further than 1.2 subtract, stop at 1 m from it
                raw_pose.pose.position.z -= standoff
            else:
                raw_pose.pose.position.z = max(0.0, raw_pose.pose.position.z - 0.4) # if marker is closer than 1.2, stop at 0.4 m from it

            global_pose = tf2_geometry_msgs.do_transform_pose(raw_pose.pose, transform)

            if not self.nav_client.server_is_ready():
                    self.get_logger().warn("Nav2 Action Server is not ready yet! Aborting goal.") #needed to add this as program kept crashing
                    self.is_navigating = False
                    return

            goal_msg = NavigateToPose.Goal()
            goal_msg.pose.header.frame_id = 'map'
            goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
            goal_msg.pose.pose.position = global_pose.position
                
            goal_msg.pose.pose.orientation.x = 0.0 # setting orientation to make it stay in whatever direction it was facing early (no rotation)
            goal_msg.pose.pose.orientation.y = 0.0
            goal_msg.pose.pose.orientation.z = 0.0
            goal_msg.pose.pose.orientation.w = 1.0
                
            self.send_goal_future = self.nav_client.send_goal_async(goal_msg)
            self.send_goal_future.add_done_callback(self.goal_response_callback)

        except Exception as e:
            self.get_logger().error(f"TF2 Math or Network Failed: {str(e)}")
            self.is_navigating = False # Unlock the camera to try again!   
            
        
            

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Nav2 rejected the goal!')
            self.is_navigating = False
            return

        self.get_logger().info('Nav2 accepted the goal, driving towards it')
        self.get_result_future = goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Arrived at destination')
        
        self.visited_ids.add(self.last_target_id) # add the last target ID to the visited set
        self.is_navigating = False

def main(args=None):
    rclpy.init(args=args)
    node = ArucoNavigation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()