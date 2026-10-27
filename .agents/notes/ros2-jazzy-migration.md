# ROS 2 Jazzy migration

Date: 2026-10-26
Status: current
Evidence: a clean `colcon build --symlink-install` of this package on Ubuntu 24.04.5 / ROS 2 Jazzy
(Python 3.12), `rosdep check`, the node launched against a stand-in for the Vicon SDK that replays a
moving rigid body, and the real SDK wrapper imported and asked to connect with no tracker reachable

Summary: no Jazzy-specific code change was needed beyond a deprecated logger call. The work is in the
environment: a Python 3.12 build of `pyvicon-datastream`, `tf_transformations` from apt, and a clean
workspace build. The package has not yet been run against a live tracker on Jazzy.

## Package changes

- `rclpy`'s `Logger.warn()` is deprecated in Jazzy; the node calls `warning()`, which Humble also has.
- `README.md` gives the 24.04 form of the `pip` command (`--break-system-packages`) and no longer names
  a distro in the `tf_transformations` apt package.

## What a 24.04 / Jazzy machine needs

- `ros-jazzy-tf-transformations` from apt, which `rosdep install` resolves from `package.xml`; it
  depends on `python3-transforms3d`.
- `pyvicon-datastream` 0.2.4 installed for the system interpreter `/usr/bin/python3`, the one the
  installed `vicon_stream_node` script runs under (`#!/usr/bin/python3`). A conda Python on `PATH` does
  not count. PyPI ships it as source only: `pip install` compiles Vicon DataStream SDK 1.11 with
  scikit-build, fetching CMake and Ninja into the build environment and Boost 1.81 from GitHub, and takes
  minutes. It needs a C++ toolchain (`build-essential`, `python3-dev`) and network access. The compiled
  module is bound to the Python minor version, so after an OS upgrade it must be reinstalled: a copy
  left in `~/.local/lib/python3.10` is invisible to Python 3.12.
- `rosdep init` and `rosdep update` once per machine. `rosdep check --from-paths . --ignore-src` run in
  the package then reports every dependency satisfied.

## Workspace

- `build/`, `install/` and `log/` written under Humble do not carry over. Delete them before the first
  Jazzy build.
- With `--symlink-install`, `install/` links the Python package, the launch file and `param/*.yaml` back
  to `src/`, so editing those needs no rebuild. A new file or a new console entry point does.
