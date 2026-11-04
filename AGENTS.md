# vicon_interface

ROS 2 (`ament_python`) driver for the Vicon DataStream SDK. Forked from `LeCAR-Lab/vicon_interface`
(CMU, F1TENTH origin) into `ARCaD-Lab-UM`, used with the Tron1 and now the Arcadian humanoid
(ARCaD Lab, University of Michigan). One node, `vicon_stream_node`, polls the tracker through the
`pyvicon-datastream` wrapper and publishes, per tracked rigid body, a pose, an odometry with body-frame
twist, a timer-driven Kalman-filtered odometry, the Vicon pipeline latency and a TF. Its consumer today
is `humanoid-control`'s `whole_body_inference`, whose `_gt` policy variants observe the base's linear
velocity and height.

## How this documentation is organised

- `README.md` is for humans: install, configure, build, run, test.
- `AGENTS.md` (this file) is the agent entry point: facts that apply to every session, the design
  decisions behind the code, and links to everything else. `CLAUDE.md` is a symlink to it.
- Procedures live in `.agents/skills/<name>/SKILL.md`. A command sequence may appear in `README.md` and
  in one skill, nowhere else.
- Notes live in `.agents/notes/`; only files whitelisted by that folder's `.gitignore` are tracked.
  Either kind can be stale: the code, the launch file and `param/*.yaml` win. A note is written only
  with the user's agreement and holds only settled facts someone will need again, in a few lines;
  measurements, guesses and the story of an investigation stay in the conversation.
- A number that exists in code or YAML is quoted in at most one document, with its source. Elsewhere, link.
- Before changing a documented fact, grep the repo for it and fix every copy.

## Session facts

- Source ROS first: `source /opt/ros/jazzy/setup.bash` (Jazzy since 2026-10-26, see the table below).
- This repo is one package inside a colcon workspace (`~/Documents/arcad-ws/src/vicon_interface` on the
  lab desktop). Run colcon from the workspace root:

  ```bash
  colcon build --symlink-install --packages-select vicon_interface
  ```

  `--symlink-install` links the Python, launch and `param/*.yaml` files back to the source tree: editing
  them needs no rebuild, a new file or console entry point does. Everything else: skill `build-and-run`.
- The installed node script runs `/usr/bin/python3`, so `pyvicon-datastream` and `tf_transformations`
  must be importable by the system interpreter, not by a conda Python on `PATH`.
- Dependencies are `package.xml` (rosdep: `rclpy`, the message packages, `tf2_ros`, `tf_transformations`)
  plus `pyvicon-datastream` from PyPI, which `setup.py` lists but rosdep cannot install.
- `test/` holds the ament lint stubs (copyright skipped, flake8, pep257) and two ROS-free pytest files
  for the Kalman filter and `ViconDataFilter`; the commands are in `README.md`.

## Layout

- `vicon_interface/vicon_stream_node.py` the node: parameters, the SDK polling loop, the publishers.
- `vicon_interface/vicon_data_buffer.py` `ViconDataBuffer`, the per-object rolling window that turns
  poses into velocities and applies the frame offset.
- `vicon_interface/vicon_data_filter.py` `ViconDataFilter(ViconDataBuffer)`, the same window plus the
  Kalman filter and watchdog behind `odometry/filtered/<obj>`; the node holds one per object.
- `vicon_interface/state_estimators/` and `vicon_interface/utils/daemon.py`: arc-bridge's
  `KalmanFilter`, `FloatingBaseLinearStateEstimator` (explicit `A`, no scipy) and `Daemon` ported as
  they are, plus the new `ViconLinearStateEstimator` (constant acceleration, position observed).
- `launch/vicon_interface.launch.py` starts the node with `param/vicon_interface.param.yaml`.
- `param/vicon_interface.param.yaml` the only configuration: tracker IP, object names, frames, offsets,
  rates, window size, state clipping, TF switch, Kalman filter and watchdog. Every number lives there.
- `.agents/` skills and notes. `.claude/skills` and `.codex/skills` are symlinks to `.agents/skills`.

## What the node does

- The poll loop is a plain `threading.Thread` running `timer_callback` (a `while True` loop paced by
  `poll_vicon_dt`), not a ROS timer. The filtered odometry is a ROS timer at `filtered_odom_dt` in its
  own callback group under a `MultiThreadedExecutor`. The two threads meet only inside the per-object
  `ViconDataFilter`, under its lock. Its getters return copies: the node writes clipped values into them.
- Each new frame (frame number changed) gives one `add_pose()` per object: mm to m, Vicon's XYZ Euler
  triple re-expressed as `sxyz`, the frame offset applied, then a finite-difference velocity across the
  whole rolling window, rotated into the child frame. The orientation rate is an Euler-angle rate, not
  ω. Wrap the angle difference, not the rate: wrapping the rate spiked `twist.angular` at ±180° yaw.
- `ViconDataFilter` adds a world-frame constant-acceleration Kalman filter (p, v, a per axis, white
  jerk, position measured) on the Vicon clock: per frame the poll thread predicts by the frame number
  step over the frame rate, with exact `A(dt)` and `Q(dt)`, then corrects. Host time never enters
  `dt`; a dropped frame is one longer step. The timer only predicts to its publish time (at most
  `kf_max_delay`) via the minimum host-minus-Vicon offset over `kf_clock_window`. After
  `vicon_offline_timeout` without a frame the `Daemon` stops publishing; the next frame resets the
  filter, which publishes again after `rolling_window_size` frames, also how it starts.
- Topics per object `<obj>` in `vicon_object_names`: `vicon_pose/<obj>` (PoseWithCovarianceStamped),
  `odometry/<obj>` (Odometry, twist in the child frame), `odometry/filtered/<obj>` (Odometry, the
  Kalman-filtered world position and velocity moved to the publish time, twist in the child frame),
  `vicon_latency/<obj>` (Float32, seconds, the SDK's `get_latency_total()`), and with `publish_tf` one TF
  `parent_frame_id -> child_frame_id` per object, all objects in one message per frame. The four
  publishers use QoS depth 1; TF uses the `TransformBroadcaster` default.
- States are clipped to `min_states` / `max_states` (12 entries: xyz, rpy, body v, body Euler rate)
  before publishing, on both odometry topics and the pose.
- A tracker that does not answer at start is fatal: the constructor raises and the process exits.

## Frame offsets, the one design decision worth knowing

`child_frame_offset_xyz` / `child_frame_offset_rpy_deg` express the pose of the frame to report, in the
tracked rigid body's own frame (`T_marker->child`). They are applied **per sample inside
`ViconDataBuffer.add_pose()`, upstream of the finite difference**.

That placement is the point: differentiating the *corrected* position gives the lever-arm term for free:

```
p_child(t) = p_marker(t) + R(t)·d
d/dt       = v_marker + Ṙ·d = v_marker + ω × (R·d)
```

So the published twist is the child frame's velocity with no angular rate needed anywhere. Correcting
*after* the differencing, which is what an offset applied downstream would do, fixes position but leaves
the velocity wrong by `ω × r`. Measured on the real buffer with a 0.25 m arm and a 10° / 1.4 Hz pitch:
0.384 m/s of false velocity, larger than the robot's own 0.3 m/s forward speed. **Do not move this
compensation downstream.**

The YAML's objects today (`g1`, `box`) sit below their rigid bodies (negative z) with axes turned about
180° in yaw, hence the yaw entries; if a rigid body is redefined, change these entries, nothing
downstream. Verified to machine precision against analytic truth for zero, yaw-90° and 10/-20/45°
offsets; with both offsets zero the output is bit-identical to the pre-offset code. Pitch near ±90°
degrades the Euler angles and the Euler-rate `twist.angular` with them.

## Things that bite consumers

- `odometry/<obj>` is the **child frame** (base_link), not the marker; old bags and scripts need updating.
- `odometry/<obj>` back-dates `header.stamp` by the Vicon pipeline latency, so `now - stamp` is a real
  end-to-end delay and latency compensation works. `odometry/filtered/<obj>` is compensated to its
  publish time and stamped `now()` by design, and pauses while the object is offline; a consumer that
  compensates latency itself uses the unfiltered one, as `humanoid-control` does
  (`modules/bridges/vicon_client`, `/odometry/arcadian`).
- Velocity is a finite difference across the whole rolling window (`rolling_window_size`, 5 in the
  param file at 250 Hz), so it carries about 8 ms of group delay relative to its own timestamp.

## Where to look next

| Task | Read |
|---|---|
| dependencies, build, launch, is it publishing | skill `build-and-run` |
| a new Ubuntu or ROS release | `.agents/notes/ros2-jazzy-migration.md` |
| what a consumer relies on | "Things that bite consumers" above, then `humanoid-control`'s `vicon_ros2_client.hpp` |

## Safety and irreversible actions

- The node only reads from the tracker; nothing here moves hardware. Its consumers do: do not change a
  topic name, frame, stamp or the meaning of the twist without checking `humanoid-control`.
- Anything that needs `sudo` (apt, `rosdep init`) is the user's to run. Give the exact command and wait.
  Do not work around a missing `sudo` with an alternative install.
- A workspace's `build/`, `install/` and `log/` belong to every package in it, not only this one; ask
  before deleting them.

## Commits

- Commit only when asked. Never add AI attribution or co-author lines to commits or pull requests.
  `.claude/settings.json` disables them for Claude Code; other tools follow the rule by hand.
- Message: a short imperative title ("Handle missing object in frame"), then at most a few short lines
  on what changed and why. No development notes or measurements.
- A PR says what was tested and which ROS parameters changed; a launch or parameter change comes with
  an example command.

## Code conventions

- Python, PEP 8, 4-space indent. Do not reformat lines you are not changing.
- `snake_case` functions and variables; instance attributes end with an underscore (`self.tracker_`,
  `self.lock_`), except in the classes ported from arc-bridge (`x`, `P`, `R`); `CapWords` classes.
- Parameters: `declare_parameter` with an explicit `rclpy.Parameter.Type` or a default, read once in
  the constructor. Per-object vectors are flat `3 * n` double arrays in `vicon_object_names` order,
  because ROS 2 parameters cannot nest arrays. Validate, log `fatal`, then raise.
- Logging through `self.get_logger()`, with `throttle_duration_sec` inside loops; `warning()`, not the
  deprecated `warn()`.
- Comments say what a line does (units, frame, sign). Rationale and "do not move this" warnings go in
  this file. Topic names stay `<kind>/<object>`; frame ids are parameters, never hard-coded.

## Writing conventions

English throughout. Short sentences. No em-dashes. Keep this file under 180 lines; longer material
goes to a skill or a note.

## Tool adapters

- Claude Code: `CLAUDE.md -> AGENTS.md` and `.claude/skills -> .agents/skills` are symlinks;
  `.claude/settings.json` turns off commit attribution. Codex reads `AGENTS.md` and `.agents/skills/`
  natively, `.codex/skills` mirrors the link; the stray `.codex` file it may drop is gitignored, the
  directory is not. Gemini CLI: `.gemini/settings.json` names `AGENTS.md` as the context file; skills
  come from `.agents/skills/`. A new skill is a new directory under `.agents/skills/`; no links to
  update.

## Maintenance

- Changing a topic, parameter, launch argument or command sequence means updating `README.md` and the
  skill that quotes it in the same change, and bumping the skill's `Last verified` line.
- Notes are append-only; add a dated update instead of rewriting a conclusion.
