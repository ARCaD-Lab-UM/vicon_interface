# vicon_interface

Vicon ROS2 Python Driver

# Install

1. Use rosdep to install the ROS dependencies: `rosdep install --from-paths ./ --ignore-src -r -y`
2. If step 1 did not install it: `sudo apt install ros-$ROS_DISTRO-tf-transformations` (required by [`tf_transformations`](https://github.com/DLu/tf_transformations))
3. Platform independent Python3 wrapper for Vicon Datastream SDK: `sudo apt install python3-pip`, then `pip install pyvicon-datastream` (install in system environment; on Ubuntu 24.04 the command is `pip install --break-system-packages pyvicon-datastream`)

Verify every dependency imports before launching:
```bash
source /opt/ros/$ROS_DISTRO/setup.bash
python3 -c "import rclpy, tf_transformations, pyvicon_datastream"
```

# Configure

Please update `vicon_tracker_ip` and `vicon_object_names` (list) to match the Vicon PC's IP and tracked objects. The node publishes each object under `vicon_pose/<name>`, `odometry/<name>`, `odometry/filtered/<name>`, and `vicon_latency/<name>`. Optionally provide `child_frame_ids` (list) to align per-object child frame IDs; otherwise each object defaults to `<name>/base_link`.

# Build

```bash
source /opt/ros/<ros_distro>/setup.bash
colcon build
```
Or only compile this package. From your `project_ws` run:
```bash
source /opt/ros/$ROS_DISTRO/setup.bash
colcon build --symlink-install --packages-select vicon_interface
```

# Run

```bash
source install/setup.bash
ros2 launch vicon_interface vicon_interface.launch.py
```
