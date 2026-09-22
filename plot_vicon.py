#!/usr/bin/env python3
"""Live matplotlib view of vicon_interface output.

Run in a second terminal while vicon_stream_node is already up:

    source /opt/ros/$ROS_DISTRO/setup.bash
    python3 plot_vicon.py

Left panel: both objects drawn together in 3D as coordinate frames -- a red/green/blue
arrow triad per object showing where each one sits and how it is oriented, with a short
trail of where it has been. The world frame sits at the origin with thin dashed arrows.

Right, one column per object, three stacked panels:

  position          published child frame, x / y / z as red / green / blue
  linear velocity   body frame, x / y / z as red / green / blue
  angular velocity  body-frame Euler rates, roll / pitch / yaw as red / green / blue

Everything plotted is exactly what the node publishes on `odometry/<obj>`; this script
does no pose maths of its own beyond turning the published quaternion into the arrows.

Object names are read from the running node's parameters, so this follows whatever is
in vicon_interface.param.yaml with no edits here.
"""

import argparse
import threading
from collections import deque

import numpy as np
import rclpy
import tf_transformations
from nav_msgs.msg import Odometry
from rcl_interfaces.srv import GetParameters
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# x / y / z  and  roll / pitch / yaw share this colour order.
AXIS_COLORS = ("tab:red", "tab:green", "tab:blue")
XYZ_LABELS = ("x", "y", "z")
RPY_LABELS = ("roll", "pitch", "yaw")

PARAM_NAMES = (
    "vicon_object_names",
    "child_frame_offset_xyz",
    "child_frame_offset_rpy_deg",
)


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def value_of(pv):
    """Unpack an rcl_interfaces ParameterValue, whatever its type."""
    return {
        8: lambda v: list(v.double_array_value),
        9: lambda v: list(v.string_array_value),
    }.get(pv.type, lambda v: None)(pv)


def fetch_node_params(node, target, timeout):
    """Read object names + offsets from the running vicon node. None if unreachable."""
    client = node.create_client(GetParameters, f"{target}/get_parameters")
    if not client.wait_for_service(timeout_sec=timeout):
        return None
    future = client.call_async(GetParameters.Request(names=list(PARAM_NAMES)))
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    result = future.result()
    if result is None or len(result.values) != len(PARAM_NAMES):
        return None
    return dict(zip(PARAM_NAMES, (value_of(v) for v in result.values)))


def discover_objects(node):
    """Fall back to whatever /odometry/<name> topics exist (ignoring the filtered ones)."""
    names = []
    for topic, _ in node.get_topic_names_and_types():
        if topic.startswith("/odometry/") and not topic.startswith("/odometry/filtered/"):
            names.append(topic.rsplit("/", 1)[1])
    return sorted(names)


class ObjectTrace:
    """Rolling history for one tracked object, filled from the subscription thread."""

    def __init__(self, name, offset_note, maxlen):
        self.name = name
        self.offset_note = offset_note
        self.lock = threading.Lock()
        self.t = deque(maxlen=maxlen)
        self.pos = deque(maxlen=maxlen)
        self.v_lin = deque(maxlen=maxlen)
        self.v_ang = deque(maxlen=maxlen)
        self.rot = None  # latest orientation, for the 3D triad

    def add(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        lin, ang = msg.twist.twist.linear, msg.twist.twist.angular
        R = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        with self.lock:
            self.t.append(stamp_to_sec(msg.header.stamp))
            self.pos.append([p.x, p.y, p.z])
            self.v_lin.append([lin.x, lin.y, lin.z])
            self.v_ang.append([ang.x, ang.y, ang.z])
            self.rot = R

    def snapshot(self):
        with self.lock:
            if not self.t:
                return None
            return (
                np.asarray(self.t),
                np.asarray(self.pos),
                np.asarray(self.v_lin),
                np.asarray(self.v_ang),
                self.rot,
            )


class PlotNode(Node):
    def __init__(self, traces):
        super().__init__("vicon_plot")
        for trace in traces.values():
            self.create_subscription(
                Odometry,
                f"odometry/{trace.name}",
                lambda msg, tr=trace: tr.add(msg),
                10,
            )


def build_figure(traces, window):
    n = len(traces)
    fig = plt.figure(figsize=(6.2 + 5.6 * n, 8.4))
    fig.canvas.manager.set_window_title("vicon_interface live signals")
    grid = fig.add_gridspec(3, n + 1, width_ratios=[1.5] + [1.0] * n)

    ax3d = fig.add_subplot(grid[:, 0], projection="3d")

    lines = {}
    for col, (name, trace) in enumerate(traces.items()):
        ax_pos = fig.add_subplot(grid[0, col + 1])
        ax_lin = fig.add_subplot(grid[1, col + 1], sharex=ax_pos)
        ax_ang = fig.add_subplot(grid[2, col + 1], sharex=ax_pos)

        ax_pos.set_title(f"{name}   [{trace.offset_note}]", fontsize=10)

        per_object = {"pos": [], "lin": [], "ang": []}
        for lbl, color in zip(XYZ_LABELS, AXIS_COLORS):
            per_object["pos"].append(ax_pos.plot([], [], color=color, lw=1.5, label=lbl)[0])
            per_object["lin"].append(ax_lin.plot([], [], color=color, lw=1.4, label=lbl)[0])
        for lbl, color in zip(RPY_LABELS, AXIS_COLORS):
            per_object["ang"].append(ax_ang.plot([], [], color=color, lw=1.4, label=lbl)[0])
        lines[name] = per_object

        ax_pos.set_ylabel("position [m]")
        ax_lin.set_ylabel("linear vel [m/s]\n(body frame)")
        ax_ang.set_ylabel("angular vel [rad/s]\n(Euler rates, body)")
        ax_ang.set_xlabel(f"time [s]  (rolling {window:.0f} s)")
        for ax in (ax_pos, ax_lin, ax_ang):
            ax.grid(alpha=0.3)
            ax.legend(loc="upper left", fontsize=8, ncol=3)

    fig.tight_layout()
    return fig, ax3d, lines


def draw_frames(ax3d, poses, axis_len, trails):
    """Redraw the shared 3D view: one arrow triad per object, plus its recent trail."""
    # Keep whatever rotation the user has dragged the view to.
    elev, azim = ax3d.elev, ax3d.azim
    ax3d.clear()

    pts = [np.zeros(3)]  # always keep the world origin in view

    # World frame: thin dashed triad at the origin.
    for i, color in enumerate(AXIS_COLORS):
        d = np.zeros(3)
        d[i] = axis_len * 0.7
        ax3d.quiver(0, 0, 0, *d, color=color, lw=1.0, ls=":", alpha=0.6, arrow_length_ratio=0.15)
    ax3d.text(0, 0, -axis_len * 0.35, "world", fontsize=8, color="0.35", ha="center")

    for name, (p, R) in poses.items():
        pts.append(p)
        for i, color in enumerate(AXIS_COLORS):
            d = R[:, i] * axis_len  # body x/y/z expressed in world
            ax3d.quiver(p[0], p[1], p[2], d[0], d[1], d[2],
                        color=color, lw=2.2, arrow_length_ratio=0.18)
        ax3d.text(p[0], p[1], p[2] + axis_len * 0.35, name, fontsize=9, weight="bold", ha="center")

        trail = trails.get(name)
        if trail is not None and len(trail) > 1:
            ax3d.plot(trail[:, 0], trail[:, 1], trail[:, 2], color="0.45", lw=0.9, alpha=0.7)
            pts.extend((trail[0], trail[-1]))

    # Equal aspect so the triads stay orthogonal: one cube around everything.
    pts = np.asarray(pts)
    center = (pts.max(axis=0) + pts.min(axis=0)) / 2.0
    span = max(float(np.max(pts.max(axis=0) - pts.min(axis=0))), axis_len * 4.0)
    half = span / 2.0 * 1.15
    ax3d.set_xlim(center[0] - half, center[0] + half)
    ax3d.set_ylim(center[1] - half, center[1] + half)
    ax3d.set_zlim(center[2] - half, center[2] + half)
    ax3d.set_box_aspect((1.0, 1.0, 1.0))

    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.set_title("objects in space  (arrows = body x/y/z)", fontsize=10)
    ax3d.view_init(elev=elev, azim=azim)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--node", default="/vicon_stream_node", help="node to read parameters from")
    ap.add_argument("--objects", nargs="*", default=None, help="override the object list")
    ap.add_argument("--window", type=float, default=10.0, help="rolling window in seconds")
    ap.add_argument("--trail", type=float, default=3.0, help="3D trail length in seconds")
    ap.add_argument("--axis-len", type=float, default=0.25, help="3D arrow length in metres")
    ap.add_argument("--rate", type=float, default=20.0, help="redraw rate in Hz")
    ap.add_argument("--timeout", type=float, default=3.0, help="parameter fetch timeout in seconds")
    args = ap.parse_args()

    rclpy.init()
    probe = rclpy.create_node("vicon_plot_probe")
    params = fetch_node_params(probe, args.node, args.timeout)

    if params and params["vicon_object_names"]:
        names = params["vicon_object_names"]
        off_xyz = params["child_frame_offset_xyz"] or []
    else:
        print(f"[warn] could not read parameters from {args.node}; is the node running?")
        names, off_xyz = [], []

    if args.objects:
        names = args.objects
    if not names:
        names = discover_objects(probe)
        print(f"[info] falling back to discovered topics: {names or 'none'}")
    if not names:
        print("[error] no objects to plot. Start vicon_stream_node, or pass --objects g1 box.")
        probe.destroy_node()
        rclpy.shutdown()
        return 1

    # Offsets are flat, 3 per object in vicon_object_names order. Shown as a label only --
    # the node has already applied them to everything published.
    if len(off_xyz) != 3 * len(names):
        off_xyz = [0.0] * (3 * len(names))

    maxlen = max(64, int(args.window * 200.0))
    traces = {}
    for i, name in enumerate(names):
        xyz = off_xyz[3 * i:3 * i + 3]
        note = f"offset xyz={list(xyz)} m applied" if any(xyz) else "no offset"
        traces[name] = ObjectTrace(name, note, maxlen)
    probe.destroy_node()

    print(f"[info] plotting {', '.join(names)} -- close the window to quit")
    node = PlotNode(traces)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    fig, ax3d, lines = build_figure(traces, args.window)

    def update(_frame):
        poses, trails = {}, {}
        for name, trace in traces.items():
            snap = trace.snapshot()
            if snap is None:
                continue
            t, pos, v_lin, v_ang, R = snap
            t_rel = t - t[-1]  # newest sample at 0, history to the left
            keep = t_rel >= -args.window
            t_rel = t_rel[keep]

            for key, data in (("pos", pos[keep]), ("lin", v_lin[keep]), ("ang", v_ang[keep])):
                for i, line in enumerate(lines[name][key]):
                    line.set_data(t_rel, data[:, i])

            if R is not None:
                poses[name] = (pos[-1], R)
            # Subsample the trail: the 3D panel is redrawn whole each frame.
            tr_keep = (t - t[-1]) >= -args.trail
            trails[name] = pos[tr_keep][::4]

        for ax in fig.get_axes():
            if ax is ax3d:
                continue
            ax.relim()
            ax.autoscale_view(scalex=False)
            ax.set_xlim(-args.window, 0.5)

        if poses:
            draw_frames(ax3d, poses, args.axis_len, trails)
        return []

    anim = FuncAnimation(  # noqa: F841 -- keep a reference or it is garbage collected
        fig, update, interval=max(10, int(1000.0 / args.rate)), cache_frame_data=False
    )

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the executor and let spin() return before the node goes away,
        # otherwise teardown aborts as the spin thread touches a destroyed node.
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
