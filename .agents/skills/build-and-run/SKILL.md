---
name: build-and-run
description: Install the dependencies of vicon_interface, build it with colcon, launch the Vicon node and check that it publishes. Use for any build, rebuild, dependency, launch or "is it running" request, including after an OS or ROS distro upgrade. Not for changing what the node publishes.
---

# Build and run vicon_interface

Last verified: 2026-10-26 (Ubuntu 24.04, ROS 2 Jazzy)

## When to use

- First build on a machine, or after a ROS distro upgrade.
- `ros2 launch` cannot find the package, or the node dies on start with an import error.
- Checking that the node is up and at what rate it publishes.

## Preconditions

1. ROS sourced in this shell (agent shells do not inherit `~/.bashrc`):

   ```bash
   source /opt/ros/jazzy/setup.bash
   ```

2. Run colcon from the workspace root, the directory whose `src/` holds this repo
   (`~/Documents/arcad-ws` on the lab desktop), never from inside the package.
3. After a distro upgrade, the workspace's `build/`, `install/` and `log/` belong to the old distro.
   Ask the user, then delete them before the first build.

## Steps

### Dependencies (once per machine)

```bash
sudo rosdep init          # only if /etc/ros/rosdep/sources.list.d does not exist yet
rosdep update
rosdep install --from-paths src/vicon_interface --ignore-src -r -y
/usr/bin/python3 -m pip install --user --break-system-packages pyvicon-datastream
```

- `rosdep` resolves `tf_transformations` to `ros-jazzy-tf-transformations`.
- `pyvicon-datastream` compiles on install (minutes, network needed). It must be installed for
  `/usr/bin/python3`, the interpreter the installed node script uses. Drop `--break-system-packages`
  on 22.04. Details: `.agents/notes/ros2-jazzy-migration.md`.
- Anything with `sudo` is the user's to run.

### Build

```bash
colcon build --symlink-install --packages-select vicon_interface
source install/setup.bash
```

`--symlink-install` links `vicon_interface/*.py`, the launch file and `param/*.yaml` from `install/`
back to `src/`; editing them needs no rebuild. A new file or a new `console_scripts` entry does.

### Run and check

```bash
ros2 launch vicon_interface vicon_interface.launch.py
```

In a second shell with `install/setup.bash` sourced:

```bash
ros2 topic hz /odometry/<object>
ros2 topic echo /vicon_latency/<object> --once
```

`<object>` is an entry of `vicon_object_names` in `param/vicon_interface.param.yaml`. The node logs
`Connection to <ip> successful` on start. With no reachable tracker it logs `Connection to <ip> failed`
and exits.

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `No module named 'pyvicon_datastream'` | installed for another interpreter (a conda Python on `PATH`), or for the previous Python minor version (`~/.local/lib/python3.10` after moving to 24.04) | reinstall for `/usr/bin/python3` |
| `No module named 'tf_transformations'` | `ros-jazzy-tf-transformations` missing, or ROS not sourced | precondition 1, then `rosdep install` above |
| `install/` or `build/` from the previous distro | stale workspace | precondition 3 |
| `Connection to <ip> failed` | tracker unreachable, or wrong `vicon_tracker_ip` | fix the network or the YAML |
