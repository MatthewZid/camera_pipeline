from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import RegisterEventHandler, LogInfo, TimerAction
from launch.event_handlers import OnProcessStart
from launch.actions import IncludeLaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # venv_path = os.path.join(get_package_share_directory('phd_semantic_nav'), '.venv/audioclip/bin/python') # for production
    venv_path = '/ros2_ws/src/camera_pipeline/.venv/yolo/bin/python'

    camera_node = Node(
        package='camera_pipeline',
        namespace='camera_pipeline',
        executable='yolo',
        name='yolo',
        output='screen',
        prefix=venv_path
    )
    
    return LaunchDescription([
        camera_node,
    ])