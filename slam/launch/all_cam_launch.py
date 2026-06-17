from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Common parameters for all nodes
    sim_time = {'use_sim_time': True} # to ensure gazebo time is used when running
    
    # 1. Front Camera Detector
    front_detector = Node(
        package='slam', # to look inside slam package
        executable='aruco_detection_node.py',# python script to run from inside slam package
        name='aruco_detector_front', # node name in ros node list
        parameters=[sim_time, {
            'camera_name': 'front',
            'image_topic': '/front_cam/color/image_raw',
            'camera_info_topic': '/front_cam/color/camera_info'
        }]
    )

    # # 2. Back Camera Detector (Maps '/rear_cam' to 'back')
    # back_detector = Node(
    #     package='slam',
    #     executable='aruco_detection_node.py',
    #     name='aruco_detector_back',
    #     parameters=[sim_time, {
    #         'camera_name': 'back',
    #         'image_topic': '/rear_cam/color/image_raw',
    #         'camera_info_topic': '/rear_cam/color/camera_info'
    #     }]
    # )

    # 3. Left Camera Detector
    left_detector = Node(
        package='slam',
        executable='aruco_detection_node.py',
        name='aruco_detector_left',
        parameters=[sim_time, {
            'camera_name': 'left',
            'image_topic': '/left_cam/color/image_raw',
            'camera_info_topic': '/left_cam/color/camera_info'
        }]
    )

    # 4. Right Camera Detector
    right_detector = Node(
        package='slam',
        executable='aruco_detection_node.py',
        name='aruco_detector_right',
        parameters=[sim_time, {
            'camera_name': 'right',
            'image_topic': '/right_cam/color/image_raw',
            'camera_info_topic': '/right_cam/color/camera_info'
        }]
    )

    # 5. The Aggregator Node
    aggregator_node = Node(
        package='slam',
        executable='marker_aggregation_node.py', 
        name='marker_aggregator',
        parameters=[sim_time]
    )

    # Launch all 5 nodes simultaneously
    return LaunchDescription([ #packages all 5 nodes into an array return to ros2 to launch together
        front_detector,
        # back_detector,
        left_detector,
        right_detector,
        aggregator_node
    ])