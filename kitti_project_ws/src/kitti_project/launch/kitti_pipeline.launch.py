from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='kitti_project',
            executable='semantic_node',
            name='semantic_node',
            output='screen'
        ),
        Node(
            package='kitti_project',
            executable='cloud_node',
            name='cloud_node',
            output='screen'
        ),
        Node(
            package='kitti_project',
            executable='bev_node',
            name='bev_node',
            output='screen'
        )
    ])
