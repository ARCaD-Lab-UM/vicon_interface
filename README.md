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

- `vicon_pose/<name>`: the pose of the child frame in `parent_frame_id` as Vicon gives it, its covariance from `kf_measurement_std`; `/tf` carries the same pose.
- `odometry/<name>`: the pose filter's pose and twist (child frame) at the capture time, with the filter's covariances. `twist_source: finite_difference` publishes Vicon's pose and a rolling-window finite difference instead. `header.stamp` is the capture time from the Vicon frame clock, so a consumer can compensate the delay itself. While the filter restarts, for `rolling_window_size` frames, it is not published.
- `odometry/filtered/<name>`: a Kalman-filtered pose and child-frame twist (constant-velocity or constant-acceleration model, `kf_motion_model`), published by a timer at `filtered_odom_dt`, already moved to the publish time (the Vicon latency is compensated) and stamped `now`. It pauses while the object has had no frame for `vicon_offline_timeout`. Tuning lives in the `kf_*` and `vicon_*` entries of `param/vicon_interface.param.yaml`, each commented with its unit.
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
/usr/bin/python3 -m pytest test/test_state_estimators.py test/test_pose_filter.py test/test_vicon_data_filter.py test/test_replay_vicon_log.py test/test_recordings.py
```

`colcon test --packages-select vicon_interface` runs the same tests plus the ament linters, which still report the package's older style findings.

# Tune

Set `raw_log_path` in `param/vicon_interface.param.yaml` (for example `/tmp/vicon_raw_%Y%m%d-%H%M%S.csv`) and run the node: it logs every frame as the SDK gave it. Record a stretch at rest and a stretch of motion, then replay the log from this directory, with ROS sourced:

```bash
/usr/bin/python3 replay_vicon_log.py /tmp/vicon_raw_<time>.csv --object g1 --static 0 20
/usr/bin/python3 replay_vicon_log.py data/2026-11-04_g1_static_dynamic/dynamic_markers.npz --object g1
```

It also reads a recording's `*_markers.npz` (`data/`, `--rate` for its frame rate). It feeds the frames to the node's estimators and prints how late the host picked them up. Then, for position, rotation, velocity and angular velocity, it scores the raw Vicon pose, the old host-clocked finite difference, the frame-clocked one and the pose filter with each motion model against zero-phase references (quadratic fits over ±12 frames): the error's RMS, 99.9th percentile and maximum, its frame-to-frame change (a jump, with the motion itself taken out) and, for the twists, the lag. `--skip T0 T1` leaves a stretch out, `--set key=value` overrides a parameter for a sweep, and `--static T0 T1` (seconds from the first frame) prints the rigid body's pose noise at rest, the values for `kf_measurement_std`.
