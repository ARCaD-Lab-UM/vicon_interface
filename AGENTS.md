# vicon_interface

ROS 2 (`ament_python`) driver for the Vicon DataStream SDK. Forked from `LeCAR-Lab/vicon_interface`
(CMU, F1TENTH origin) into `ARCaD-Lab-UM`, used with the Tron1 and now the Arcadian humanoid
(ARCaD Lab, University of Michigan). One node, `vicon_stream_node`, polls the tracker through the
`pyvicon-datastream` wrapper and publishes, per tracked rigid body, a pose, an odometry with body-frame
twist, a timer-driven extrapolated odometry, the Vicon pipeline latency and a TF. Its consumer today is
`humanoid-control`'s `whole_body_inference`, whose `_gt` policy variants observe the base's linear
velocity and height.

## How this documentation is organised

- `README.md` is for humans: install, configure, build, run.
- `AGENTS.md` (this file) is the agent entry point: facts that apply to every session, the design
  decisions behind the code, and links to everything else. `CLAUDE.md` is a symlink to it.
- Procedures live in `.agents/skills/<name>/SKILL.md`. A command sequence may appear in `README.md` and
  in one skill, nowhere else.
- Notes live in `.agents/notes/`. Only the files whitelisted by that folder's `.gitignore` are tracked;
  the rest are local working notes. Either kind can be stale; the code, the launch file and
  `param/*.yaml` win. A note is written only with the user's agreement, holds only settled, verified
  facts that someone will need again, and says them in a few lines. Not every finding is a note:
  measurements, guesses and the story of an investigation stay in the conversation.
- A number that exists in code or YAML is quoted in at most one document, with its source. Elsewhere, link.
- Before changing a documented fact, grep the repo for it and fix every copy.

## Session facts

- Source ROS before anything: `source /opt/ros/jazzy/setup.bash`. The lab desktop moved from Humble to
  Jazzy on 2026-10-26; `.agents/notes/ros2-jazzy-migration.md` has what changed.
- This repo is one package inside a colcon workspace (`~/Documents/arcad-ws/src/vicon_interface` on the
  lab desktop). Run colcon from the workspace root:

  ```bash
  colcon build --symlink-install --packages-select vicon_interface
  ```

  `--symlink-install` links `vicon_interface/*.py`, the launch file and `param/*.yaml` from `install/`
  back to the source tree, so editing them needs no rebuild; a new file or console entry point does.
  Everything else: skill `build-and-run`.
- The installed node script runs `/usr/bin/python3`, so `pyvicon-datastream` and `tf_transformations`
  must be importable by the system interpreter, not by a conda Python on `PATH`.
- Dependencies are `package.xml` (rosdep: `rclpy`, the message packages, `tf2_ros`, `tf_transformations`)
  plus `pyvicon-datastream` from PyPI, which `setup.py` lists but rosdep cannot install.
- `test/` holds the ament lint stubs only (copyright skipped, flake8, pep257); no behavioural tests.

## Layout

- `vicon_interface/vicon_stream_node.py` the node: parameters, the SDK polling loop, the publishers.
- `vicon_interface/vicon_data_buffer.py` `ViconDataBuffer`, the per-object rolling window that turns
  poses into velocities and applies the frame offset; `ViconDataFilter`, a first-order low-pass variant
  that the node does not use.
- `launch/vicon_interface.launch.py` starts the node with `param/vicon_interface.param.yaml`. Its
  `vicon_mpc_state -> /vehicle_state` remapping matches no topic the node publishes.
- `param/vicon_interface.param.yaml` the only configuration: tracker IP, object names, frames, offsets,
  rates, window size, state clipping, TF switch. Every number lives there.
- `.agents/` skills and notes. `.claude/skills` and `.codex/skills` are symlinks to `.agents/skills`.

## What the node does

- The poll loop is a plain `threading.Thread` running `timer_callback` (a `while True` loop paced by
  `poll_vicon_dt`), not a ROS timer. The extrapolated odometry comes from a ROS timer at
  `filtered_odom_dt` in its own callback group under a `MultiThreadedExecutor`. The two threads meet
  only inside `ViconDataBuffer`, under its lock.
- Each new frame (frame number changed) gives one `add_pose()` per object: mm to m, Vicon's XYZ Euler
  triple re-expressed as `sxyz`, the frame offset applied, then a finite-difference velocity across the
  whole rolling window, rotated into the child frame. The orientation rate is an Euler-angle rate, not ω.
- Topics per object `<obj>` in `vicon_object_names`: `vicon_pose/<obj>` (PoseWithCovarianceStamped),
  `odometry/<obj>` (Odometry, twist in the child frame), `odometry/filtered/<obj>` (Odometry, the latest
  pose extrapolated by the latest velocity to the timer tick), `vicon_latency/<obj>` (Float32, seconds,
  the SDK's `get_latency_total()`), and with `publish_tf` one TF `parent_frame_id -> child_frame_id` per
  object, all objects in one message per frame. The four publishers use QoS depth 1; TF uses the
  `TransformBroadcaster` default.
- States are clipped to `min_states` / `max_states` (12 entries: xyz, rpy, body v, body Euler rate)
  before publishing, on both odometry topics and the pose.
- A tracker that does not answer at start is fatal: the constructor raises and the process exits.

## Frame offsets, the one design decision worth knowing

`child_frame_offset_xyz` / `child_frame_offset_rpy_deg` express the pose of the frame to report, in the
tracked rigid body's own frame (`T_marker->child`). They are applied **per sample inside
`ViconDataBuffer.add_pose()`, upstream of the finite difference**.

That placement is the whole point, not an implementation detail. Differentiating the *corrected*
position reproduces the lever-arm term for free:

```
p_child(t) = p_marker(t) + R(t)·d
d/dt       = v_marker + Ṙ·d = v_marker + ω × (R·d)
```

So the published twist is the child frame's velocity with no angular rate needed anywhere. Correcting
*after* the differencing, which is what an offset applied downstream would do, fixes position but leaves
the velocity wrong by `ω × r`. Measured on the real buffer with a 0.25 m arm and a 10° / 1.4 Hz pitch:
0.384 m/s of false velocity, larger than the robot's own 0.3 m/s forward speed. **Do not move this
compensation downstream.**

Current values are for `arcadian`: its Vicon rigid body is defined 0.25 m *above* base_link, so
base_link sits 0.25 m below it in the rigid body's frame and the z entry is **negative**. The rotation
is zero only because the rigid body axes happen to align with base_link; if a rigid body is ever
redefined rotated, put the correction in `child_frame_offset_rpy_deg` rather than anywhere downstream.

Verified to machine precision (position, body-frame velocity and orientation) against analytic truth
for zero, yaw-90°, and 10/-20/45° rotation offsets. With both offsets zero the node's output is
bit-identical to the pre-offset implementation.

Caveat: a rotation offset large enough to put the reported pitch near ±90° degrades the Euler
representation, and `twist.angular`, an Euler-angle rate rather than a true ω, would distort with it.

## Things that bite consumers

- `odometry/<obj>` means the **child frame** (base_link), not the marker. Bags and analysis scripts
  written against the old meaning need updating.
- `odometry/<obj>` back-dates `header.stamp` by the Vicon pipeline latency, so `now - stamp` is a real
  end-to-end delay and latency compensation works. `odometry/filtered/<obj>` stamps with `now()`
  instead; subscribing to it silently makes the measured delay zero. Consumers that compensate for
  latency must use the unfiltered one. `humanoid-control` does (`modules/bridges/vicon_client`,
  `/odometry/arcadian`).
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
  `self.lock_`); `CapWords` classes.
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
  `.claude/settings.json` turns off commit attribution.
- Codex: reads `AGENTS.md` and `.agents/skills/` natively; `.codex/skills` mirrors the link. The stray
  file named `.codex` that Codex may drop in the project is gitignored, the directory is not.
- Gemini CLI: `.gemini/settings.json` names `AGENTS.md` as the context file; skills come from
  `.agents/skills/`.
- A new skill is a new directory under `.agents/skills/`; no links to update.

## Maintenance

- Changing a topic, parameter, launch argument or command sequence means updating `README.md` and the
  skill that quotes it in the same change, and bumping the skill's `Last verified` line.
- Notes are append-only; add a dated update instead of rewriting a conclusion.
