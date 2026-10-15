rm -rf build/ install/ log/
colcon build --packages-select hexapod_description
source install/setup.bash
GZ_SIM_RESOURCE_PATH="$PWD/src:$PWD/install/hexapod_description/share" ros2 launch hexapod_description hexapod_control.launch.py