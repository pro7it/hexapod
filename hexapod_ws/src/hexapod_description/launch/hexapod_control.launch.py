from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    description_share = FindPackageShare('hexapod_description')

    xacro_file = PathJoinSubstitution([
        description_share, 'urdf', 'hexapod.urdf.xacro'
    ])

    world_file = PathJoinSubstitution([
        description_share, 'worlds', 'hexapod.world'
    ])

    robot_description = ParameterValue(
        Command(['xacro', ' ', xacro_file]),
        value_type=str,
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py',
            ])
        ),
        launch_arguments={
            'gz_args': ['-r ', world_file],
        }.items(),
    )

    state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': robot_description}],
        output='screen',
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'hexapod',
            '-x', '0',
            '-y', '0',
            '-z', '0.5',
        ],
        output='screen',
    )

    return LaunchDescription([
        gazebo,
        state_publisher,
        TimerAction(
            period=5.0,
            actions=[spawn_robot],
        ),
    ])
