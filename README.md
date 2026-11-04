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

Please update `vicon_tracker_ip` and `vicon_object_names` (list) to match the Vicon PC's IP and tracked objects. Optionally provide `child_frame_ids` (list) to align per-object child frame IDs; otherwise each object defaults to `<name>/base_link`.

The node publishes each object under:

- `vicon_pose/<name>`: the pose of the child frame in `parent_frame_id`.
- `odometry/<name>`: the same pose plus a twist in the child frame from a rolling-window finite difference. `header.stamp` is the capture time (back-dated by the Vicon latency), so a consumer can compensate the delay itself.
- `odometry/filtered/<name>`: a Kalman-filtered world position and child-frame velocity, published by a timer at `filtered_odom_dt`, already moved to the publish time (the Vicon latency is compensated) and stamped `now`. It pauses while the object has had no frame for `vicon_offline_timeout`. Tuning lives in the `kf_*` and `vicon_*` entries of `param/vicon_interface.param.yaml`, each commented with its unit.
- `vicon_latency/<name>`: the Vicon pipeline latency in seconds.
- `/tf`: `parent_frame_id -> <child frame>` per object when `publish_tf` is true.

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

# Test

The unit tests need no Vicon and no running node, only the sourced ROS environment for `tf_transformations`:

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
/usr/bin/python3 -m pytest test/test_state_estimators.py test/test_vicon_data_filter.py
```

`colcon test --packages-select vicon_interface` runs the same tests plus the ament linters, which still report the package's older style findings.
