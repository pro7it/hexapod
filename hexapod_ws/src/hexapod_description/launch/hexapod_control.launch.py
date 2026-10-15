from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit


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
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
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

    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen',
    )

    def spawner(*args):
        return Node(
            package='controller_manager',
            executable='spawner',
            arguments=[*args, '--controller-manager', '/controller_manager',
                    '--controller-manager-timeout', '60'],
            parameters=[{'use_sim_time': True}],
            output='screen',
        )

    joint_state_broadcaster = spawner('joint_state_broadcaster')
    joint_trajectory_controller = spawner('joint_trajectory_controller')
    joint_group_position_controller = spawner('joint_group_position_controller', '--inactive')

    after_spawn = RegisterEventHandler(
        OnProcessExit(target_action=spawn_robot, on_exit=[joint_state_broadcaster])
    )
    after_jsb = RegisterEventHandler(
        OnProcessExit(
            target_action=joint_state_broadcaster,
            on_exit=[joint_trajectory_controller, joint_group_position_controller],
        )
    )

    return LaunchDescription([
        gazebo,
        clock_bridge,
        state_publisher,
        TimerAction(period=5.0, actions=[spawn_robot]),
        after_spawn,
        after_jsb,
    ])
