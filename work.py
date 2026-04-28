import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
import math

from tf_transformations import euler_from_quaternion

class PreventCliffs(Node):
    def __init__(self):
        super().__init__('prevent_cliff')
        
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.get_logger().info("Waiting for data from imu topic")

    def imu_callback(self, msg):
        q = msg.orientation
        quaternion_list = [q.x, q.y, q.z, q.w]

        (roll, pitch, yaw) = euler_from_quaternion(quaternion_list)

        pitch_deg = math.degrees(pitch)
        roll_deg = math.degrees(roll)

        if abs(pitch_deg) > 15.0 or abs(roll_deg) > 15.0:
            self.get_logger().warn(f"slope has been detected Pitch: {pitch_deg}, Roll: {roll_deg:}")
            
            self.get_logger().error("slope exceeded, had to stop")
            self.stop_robot()

    def stop_robot(self):
        stop_msg = Twist()
        stop_msg.linear.x = 0.0
        stop_msg.angular.z = 0.0
        self.cmd_pub.publish(stop_msg)
        self.get_logger().error("EMERGENCY STOP SENT")

def main(args=None):
    rclpy.init(args=args)
    node = PreventCliffs()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()