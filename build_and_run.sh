rm -rf build/ install/ log/
colcon build --packages-select hexapod_description
source install/setup.bash
ros2 launch hexapod_description hexapod_view.launch.py