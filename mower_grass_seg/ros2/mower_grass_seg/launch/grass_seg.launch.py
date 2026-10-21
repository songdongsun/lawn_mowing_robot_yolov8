from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = Path(get_package_share_directory('mower_grass_seg')) / 'config' / 'grass_seg.yaml'
    return LaunchDescription([
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera/camera_info'),
        DeclareLaunchArgument('mask_topic', default_value='/grass/mask'),
        Node(package='mower_grass_seg', executable='grass_seg_node', name='grass_seg_node',
             parameters=[str(config)], output='screen', remappings=[
                 ('image', LaunchConfiguration('image_topic')),
                 ('camera_info', LaunchConfiguration('camera_info_topic')),
                 ('mask', LaunchConfiguration('mask_topic')),
             ]),
    ])
