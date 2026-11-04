# g1 static and dynamic recordings, 2026-11-04

Recorded while looking into the jumps in the linear velocity of `odometry/g1` and
`odometry/filtered/g1`. Vicon ran at 250 Hz. The node ran from commit 619a73e with the working-tree
param file, in which `child_frame_offset_rpy_deg` was zero for both objects. Every other parameter
was as committed.

The two recordings were taken back to back, about 40 s each:

- `static_*`: g1 held still, with a small disturbance around t = 19 to 21 s. box static.
- `dynamic_*`: g1 moved by hand, about 1.2 m of travel and 200 deg of yaw. box static.

All files are `np.savez_compressed` archives. Within one recording, the ROS and marker files share
the host `time.monotonic()` clock, so their rows can be aligned.

## `<recording>_ros.npz`

A subscriber on the same host as the node (QoS depth 1000). Its keys are `<obj>_raw`
(`odometry/<obj>`), `<obj>_filt` (`odometry/filtered/<obj>`), `<obj>_pose` (`vicon_pose/<obj>`) and
`<obj>_lat` (`vicon_latency/<obj>`), for `<obj>` in `g1` and `box`. The subscriber missed some
messages, so the receive rate is below the publish rate; use the marker file for frame numbers.

| Key | Columns |
|---|---|
| `_raw`, `_filt` (16) | receive `time.monotonic()` s, receive `time.time()` s, `header.stamp` s, position xyz m, orientation xyzw, `twist.linear` xyz m/s (child frame), `twist.angular` xyz as published |
| `_pose` (10) | receive monotonic s, receive wall s, `header.stamp` s, position xyz m, orientation xyzw |
| `_lat` (3) | receive monotonic s, receive wall s, latency s |

## `<recording>_markers.npz`

A second, read-only DataStream client (`pyvicon-datastream`, ServerPush, the same axis mapping as
the node), one row per g1 frame. `names` holds the marker names (`g11` to `g16`).

`rows` has 30 columns: host `time.monotonic()` s, host `time.time()` s, frame number, total latency
s, subject quality (NaN: not reported), segment global translation xyz mm, segment local rotation
Euler XYZ rad (as the node reads it), unlabeled marker count, then xyz mm for each marker in `names`
order.

These marker positions fit the segment pose with zero residual and were never reported occluded.
They come from the Tracker model, so they say nothing about marker visibility.
