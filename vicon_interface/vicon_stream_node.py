from dataclasses import dataclass
from threading import Thread
from typing import Any
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float32
from tf2_ros import TransformBroadcaster

from pyvicon_datastream import tools
import pyvicon_datastream as pv
import tf_transformations

import numpy as np

from vicon_interface.vicon_data_filter import ViconDataFilter


@dataclass
class TrackedObject:
    name: str
    child_frame_id: str
    buffer: ViconDataFilter
    pose_publisher: Any
    odom_publisher: Any
    filtered_odom_publisher: Any
    latency_publisher: Any


class ViconStreamNode(Node):
    def __init__(self):
        super().__init__("vicon_stream_node")
        self.viacon_tracker_ip_ = (
            self.declare_parameter("vicon_tracker_ip", rclpy.Parameter.Type.STRING)
            .get_parameter_value()
            .string_value
        )
        self.viacon_object_names_ = list(
            self.declare_parameter("vicon_object_names", rclpy.Parameter.Type.STRING_ARRAY)
            .get_parameter_value()
            .string_array_value
        )
        if not self.viacon_object_names_:
            self.get_logger().fatal("At least one Vicon object name must be provided via `vicon_object_names`.")
            raise ValueError("Missing Vicon object names parameter")
        self.min_states_ = np.array(
            self.declare_parameter("min_states", rclpy.Parameter.Type.DOUBLE_ARRAY)
            .get_parameter_value()
            .double_array_value
        )
        self.max_states_ = np.array(
            self.declare_parameter("max_states", rclpy.Parameter.Type.DOUBLE_ARRAY)
            .get_parameter_value()
            .double_array_value
        )
        self.parent_frame_id_ = (
            self.declare_parameter("parent_frame_id", "map")
            .get_parameter_value()
            .string_value
        )
        self.child_frame_ids_ = list(
            self.declare_parameter("child_frame_ids", rclpy.Parameter.Type.STRING_ARRAY)
            .get_parameter_value()
            .string_array_value
        )
        if self.child_frame_ids_ and len(self.child_frame_ids_) != len(self.viacon_object_names_):
            self.get_logger().fatal(f"`child_frame_ids` must match the length of `vicon_object_names` (expected {len(self.viacon_object_names_)}, got {len(self.child_frame_ids_)}).")
            raise ValueError("Mismatched child frame list length")
        self.poll_vicon_dt_ = (
            self.declare_parameter("poll_vicon_dt", 0.01)
            .get_parameter_value()
            .double_value
        )
        self.filtered_odom_dt_ = (
            self.declare_parameter("filtered_odom_dt", 0.02)
            .get_parameter_value()
            .double_value
        )
        self.rolling_window_size_ = (
            self.declare_parameter("rolling_window_size", 10)
            .get_parameter_value()
            .integer_value
        )
        self.publish_tf_ = (
            self.declare_parameter("publish_tf", True).get_parameter_value().bool_value
        )
        # Pose filter and watchdog behind both odometry topics; units in the param file.
        self.kf_motion_model_ = (
            self.declare_parameter("kf_motion_model", rclpy.Parameter.Type.STRING)
            .get_parameter_value()
            .string_value
        )
        if self.kf_motion_model_ not in ("cv", "ca"):
            self.get_logger().fatal(
                f"`kf_motion_model` must be \"cv\" or \"ca\"; got \"{self.kf_motion_model_}\".")
            raise ValueError("Bad parameter `kf_motion_model`")
        noise_psd = {m: self._read_double_array(f"kf_{m}_noise_psd", 2) for m in ("cv", "ca")}
        self.kf_noise_psd_ = noise_psd[self.kf_motion_model_]
        self.kf_measurement_std_ = self._read_double_array("kf_measurement_std", 2)
        std_p, std_theta = self.kf_measurement_std_  # m, rad
        self.pose_covariance_ = np.diag([std_p ** 2] * 3 + [std_theta ** 2] * 3).ravel().tolist()
        self.kf_initial_std_ = self._read_double_array("kf_initial_std", 4)
        self.kf_gate_ = self._read_double_array("kf_gate", 2)
        self.kf_rot_huber_k_ = self._read_double("kf_rot_huber_k")
        self.kf_max_rejections_ = (
            self.declare_parameter("kf_max_rejections", rclpy.Parameter.Type.INTEGER)
            .get_parameter_value()
            .integer_value
        )
        self.kf_max_delay_ = self._read_double("kf_max_delay")
        self.kf_clock_window_ = self._read_double("kf_clock_window")
        self.twist_source_ = (
            self.declare_parameter("twist_source", "filter").get_parameter_value().string_value
        )
        if self.twist_source_ not in ("filter", "finite_difference"):
            self.get_logger().fatal(
                "`twist_source` must be \"filter\" or \"finite_difference\"; "
                f"got \"{self.twist_source_}\".")
            raise ValueError("Bad parameter `twist_source`")
        self.vicon_offline_timeout_ = self._read_double("vicon_offline_timeout")
        self.vicon_online_jitter_time_ = self._read_double("vicon_online_jitter_time")

        # (T_marker->child): the child origin in marker coordinates, and the child orientation as sxyz Euler angles in degrees.
        n_obj = len(self.viacon_object_names_)
        self.child_frame_offsets_xyz_ = self._read_per_object_vec3(
            "child_frame_offset_xyz", n_obj
        )
        offsets_rpy_deg = self._read_per_object_vec3("child_frame_offset_rpy_deg", n_obj)
        self.child_frame_offsets_R_ = [
            tf_transformations.euler_matrix(*np.deg2rad(rpy), "sxyz")[:3, :3]
            for rpy in offsets_rpy_deg
        ]
        for name, xyz, rpy in zip(
            self.viacon_object_names_, self.child_frame_offsets_xyz_, offsets_rpy_deg
        ):
            if np.any(xyz) or np.any(rpy):
                self.get_logger().info(
                    f"`{name}` reported at offset xyz={xyz.tolist()} m, "
                    f"rpy={rpy.tolist()} deg from the tracked rigid body."
                )

        self.get_logger().info(
            f"Connecting to Viacon tracker at {self.viacon_tracker_ip_}..."
        )
        self.tracker_ = tools.ObjectTracker(self.viacon_tracker_ip_)
        if self.tracker_.is_connected:
            self.get_logger().info(
                f"Connection to {self.viacon_tracker_ip_} successful"
            )
        else:
            self.get_logger().fatal(f"Connection to {self.viacon_tracker_ip_} failed")
            raise Exception(f"Connection to {self.viacon_tracker_ip_} failed")

        self.tracked_objects_: dict[str, TrackedObject] = {}
        for idx, object_name in enumerate(self.viacon_object_names_):
            child_frame = (self.child_frame_ids_[idx] if idx < len(self.child_frame_ids_) else f"{object_name}/base_link")
            pose_topic = f"vicon_pose/{object_name}"
            odom_topic = f"odometry/{object_name}"
            filtered_odom_topic = f"odometry/filtered/{object_name}"
            latency_topic = f"vicon_latency/{object_name}"
            tracked_object = TrackedObject(
                name=object_name,
                child_frame_id=child_frame,
                buffer=ViconDataFilter(
                    6,
                    self.rolling_window_size_,
                    offset_xyz=self.child_frame_offsets_xyz_[idx],
                    offset_R=self.child_frame_offsets_R_[idx],
                    motion_model=self.kf_motion_model_,
                    noise_psd=self.kf_noise_psd_,
                    measurement_std=self.kf_measurement_std_,
                    initial_std=self.kf_initial_std_,
                    gate=self.kf_gate_,
                    rot_huber_k=self.kf_rot_huber_k_,
                    max_rejections=self.kf_max_rejections_,
                    max_delay=self.kf_max_delay_,
                    clock_window=self.kf_clock_window_,
                    offline_timeout=self.vicon_offline_timeout_,
                    online_jitter_time=self.vicon_online_jitter_time_,
                    owner_id=object_name,
                ),
                pose_publisher=self.create_publisher(PoseWithCovarianceStamped, pose_topic, 1),
                odom_publisher=self.create_publisher(Odometry, odom_topic, 1),
                filtered_odom_publisher=self.create_publisher(Odometry, filtered_odom_topic, 1),
                latency_publisher=self.create_publisher(Float32, latency_topic, 1),
            )
            self.tracked_objects_[object_name] = tracked_object

        self.last_frame_nums_ = {name: 0 for name in self.viacon_object_names_}

        if self.publish_tf_:
            self.tf_broadcaster_ = TransformBroadcaster(self)

        self.cb_group_poll_viacon_ = MutuallyExclusiveCallbackGroup()
        self.cb_group_filtered_odom_ = MutuallyExclusiveCallbackGroup()

        # self.timer_ = self.create_timer(self.poll_vicon_dt_, self.timer_callback, self.cb_group_poll_viacon_)
        self.poll_vicon_thread = Thread(target=self.timer_callback)
        self.poll_vicon_thread.start()

        self.filtered_odom_timer_ = self.create_timer(
            self.filtered_odom_dt_,
            self.filtered_odom_timer_callback,
            self.cb_group_filtered_odom_,
        )

    def _read_double_array(self, name, n):
        """Read a double array parameter that must have exactly n entries; fatal otherwise."""
        values = list(
            self.declare_parameter(name, rclpy.Parameter.Type.DOUBLE_ARRAY)
            .get_parameter_value()
            .double_array_value
        )
        if len(values) != n:
            self.get_logger().fatal(f"`{name}` must have {n} entries; got {len(values)}.")
            raise ValueError(f"Bad parameter `{name}`")
        return values

    def _read_double(self, name):
        return (
            self.declare_parameter(name, rclpy.Parameter.Type.DOUBLE)
            .get_parameter_value()
            .double_value
        )

    def _read_per_object_vec3(self, name, n_obj):
        """Read a flat 3*n_obj parameter as a list of 3-vectors; all-zero if unset or wrong size.
        Three entries per object, in vicon_object_names order (ROS 2 parameter cannot nest arrays).
        """
        values = list(
            self.declare_parameter(name, [0.0] * (3 * n_obj))
            .get_parameter_value()
            .double_array_value
        )
        if len(values) != 3 * n_obj:
            self.get_logger().warning(
                f"`{name}` must have {3 * n_obj} entries "
                f"({n_obj} objects x 3); got {len(values)}. Using zeros."
            )
            values = [0.0] * (3 * n_obj)
        return [np.array(values[3 * i:3 * i + 3]) for i in range(n_obj)]

    def timer_callback(self):
        while True:
            begin = time.monotonic()
            if not self.tracker_.is_connected:
                continue
            frame_result = self.tracker_.vicon_client.get_frame()
            end_get_frame_time = time.monotonic()
            # print(f"Time to get frame: {end_get_frame_time - begin:.6f}s")

            transforms_to_publish = []
            for object_name, tracked_object in self.tracked_objects_.items():
                if frame_result == pv.Result.Success:
                    latency = self.tracker_.vicon_client.get_latency_total()
                    framenumber = self.tracker_.vicon_client.get_frame_number()
                    frame_rate = self.tracker_.vicon_client.get_frame_rate()  # Hz
                    position = self.tracker_._get_object_position(object_name)
                    if latency is None or framenumber is None or not position:
                        self.get_logger().warning(f"Missing object `{object_name}` data in frame.",throttle_duration_sec=1.0)
                        continue
                else:
                    self.get_logger().warning(f"Cannot get the frame with `{object_name}`.", throttle_duration_sec=1.0)
                    continue

                grab_time = self.get_clock().now()
                latency = float(latency)
                frame_time = grab_time - Duration(seconds=latency)
                frame_stamp = frame_time.to_msg()
                frame_num = framenumber
                if frame_num == self.last_frame_nums_[object_name]:
                    continue
                self.last_frame_nums_[object_name] = frame_num
                # print(f"For object `{object_name}`, got frame {frame_num} at time {frame_stamp} with latency {latency:.4f}s.")
                try:
                    obj = position[0]
                    _, _, x, y, z, roll, pitch, yaw = obj
                    q = tf_transformations.quaternion_from_euler(roll, pitch, yaw, "rxyz")
                    roll, pitch, yaw = tf_transformations.euler_from_quaternion(q, "sxyz")
                    tracked_object.buffer.add_pose(
                        (x, y, z, roll, pitch, yaw), latency, frame_num, frame_rate
                    )
                except Exception as e:
                    self.get_logger().warning(
                        f"Vicon dropped a frame for `{object_name}`: {e}",
                        throttle_duration_sec=1.0,
                    )
                    continue  # publishing now would repeat the previous frame under a new stamp

                estimate, stamp = tracked_object.buffer.get_frame_state()
                if stamp is not None:  # capture time from the Vicon clock, free of host delays
                    age = max(time.monotonic() - stamp, 0.0)  # s
                    frame_stamp = (self.get_clock().now() - Duration(seconds=age)).to_msg()
                if self.twist_source_ == "filter":
                    publish_odom = estimate is not None  # not while the filter warms up
                    v_body = estimate.R.T @ estimate.v if publish_odom else np.zeros(3)  # m/s
                    ang_v_body = estimate.w if publish_odom else np.zeros(3)  # rad/s, child
                else:
                    v_body, ang_v_body, timestamp = tracked_object.buffer.get_latest_velocity()
                    publish_odom = timestamp != 0
                pose, timestamp = tracked_object.buffer.get_latest_pose()
                if timestamp == 0:
                    continue
                # vicon_pose and TF carry the Vicon pose; odometry/<obj> the filter's, at the frame
                x_r, y_r, z_r, roll_r, pitch_r, yaw_r = np.clip(
                    pose, self.min_states_[:6], self.max_states_[:6])
                q_r = tf_transformations.quaternion_from_euler(roll_r, pitch_r, yaw_r)
                odom_pose = pose
                if self.twist_source_ == "filter" and publish_odom:
                    m = np.eye(4)
                    m[:3, :3] = estimate.R
                    odom_pose = np.r_[estimate.p, tf_transformations.euler_from_matrix(m, "sxyz")]
                states = np.clip(np.r_[odom_pose, v_body, ang_v_body],
                                 self.min_states_, self.max_states_)
                x, y, z, roll, pitch, yaw = states[:6]
                v_body, ang_v_body = states[6:9], states[9:12]

                odom_msg = Odometry()
                odom_msg.header.stamp = frame_stamp
                odom_msg.header.frame_id = self.parent_frame_id_
                odom_msg.child_frame_id = tracked_object.child_frame_id
                odom_msg.pose.pose.position.x = x
                odom_msg.pose.pose.position.y = y
                odom_msg.pose.pose.position.z = z
                q = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
                odom_msg.pose.pose.orientation.x = q[0]
                odom_msg.pose.pose.orientation.y = q[1]
                odom_msg.pose.pose.orientation.z = q[2]
                odom_msg.pose.pose.orientation.w = q[3]
                odom_msg.twist.twist.linear.x = v_body[0]
                odom_msg.twist.twist.linear.y = v_body[1]
                odom_msg.twist.twist.linear.z = v_body[2]
                odom_msg.twist.twist.angular.x = ang_v_body[0]
                odom_msg.twist.twist.angular.y = ang_v_body[1]
                odom_msg.twist.twist.angular.z = ang_v_body[2]
                if self.twist_source_ == "filter" and publish_odom:
                    pose_covariance, twist_covariance = estimate.odometry_covariances()
                    odom_msg.pose.covariance = pose_covariance.ravel().tolist()
                    odom_msg.twist.covariance = twist_covariance.ravel().tolist()
                else:
                    odom_msg.pose.covariance = self.pose_covariance_  # the Vicon pose as measured
                if publish_odom:
                    tracked_object.odom_publisher.publish(odom_msg)

                pose_msg = PoseWithCovarianceStamped()
                pose_msg.pose.pose.position.x = x_r
                pose_msg.pose.pose.position.y = y_r
                pose_msg.pose.pose.position.z = z_r
                pose_msg.pose.pose.orientation.x = q_r[0]
                pose_msg.pose.pose.orientation.y = q_r[1]
                pose_msg.pose.pose.orientation.z = q_r[2]
                pose_msg.pose.pose.orientation.w = q_r[3]
                pose_msg.header.stamp = odom_msg.header.stamp
                pose_msg.pose.covariance = self.pose_covariance_
                pose_msg.header.frame_id = self.parent_frame_id_
                tracked_object.pose_publisher.publish(pose_msg)

                if self.publish_tf_:
                    transform = TransformStamped()
                    transform.header.stamp = odom_msg.header.stamp
                    transform.header.frame_id = self.parent_frame_id_
                    transform.child_frame_id = tracked_object.child_frame_id
                    transform.transform.translation.x = x_r
                    transform.transform.translation.y = y_r
                    transform.transform.translation.z = z_r
                    transform.transform.rotation.x = q_r[0]
                    transform.transform.rotation.y = q_r[1]
                    transform.transform.rotation.z = q_r[2]
                    transform.transform.rotation.w = q_r[3]
                    transforms_to_publish.append(transform)

                latency_msg = Float32()
                latency_msg.data = latency
                tracked_object.latency_publisher.publish(latency_msg)

            if self.publish_tf_ and transforms_to_publish:
                # Publish the full set of TF in a single message
                self.tf_broadcaster_.sendTransform(transforms_to_publish)

            end = time.monotonic()
            if end - begin < self.poll_vicon_dt_:
                time.sleep(self.poll_vicon_dt_ - (end - begin))

    def filtered_odom_timer_callback(self):
        for object_name, tracked_object in self.tracked_objects_.items():
            if not tracked_object.buffer.update():
                if tracked_object.buffer.is_offline():
                    self.get_logger().warning(
                        f"No Vicon frame for `{object_name}` within vicon_offline_timeout, "
                        "odometry/filtered paused.",
                        throttle_duration_sec=1.0,
                    )
                elif not tracked_object.buffer.has_frame_clock():
                    self.get_logger().warning(
                        f"Vicon reports no frame rate for `{object_name}`, "
                        "odometry/filtered paused.",
                        throttle_duration_sec=1.0,
                    )
                continue
            estimate, _ = tracked_object.buffer.get_filtered_state()  # moved to now
            if estimate is None:
                continue
            v_body = estimate.R.T @ estimate.v  # m/s, child frame
            ang_v_body = estimate.w  # rad/s, child frame
            m = np.eye(4)
            m[:3, :3] = estimate.R

            x, y, z = estimate.p
            roll, pitch, yaw = tf_transformations.euler_from_matrix(m, "sxyz")
            states = np.array(
                [
                    x,
                    y,
                    z,
                    roll,
                    pitch,
                    yaw,
                    v_body[0],
                    v_body[1],
                    v_body[2],
                    ang_v_body[0],
                    ang_v_body[1],
                    ang_v_body[2],
                ]
            )
            states = np.clip(states, self.min_states_, self.max_states_)
            (
                x,
                y,
                z,
                roll,
                pitch,
                yaw,
                v_body[0],
                v_body[1],
                v_body[2],
                ang_v_body[0],
                ang_v_body[1],
                ang_v_body[2],
            ) = states

            odom_msg = Odometry()
            odom_msg.header.stamp = self.get_clock().now().to_msg()
            odom_msg.header.frame_id = self.parent_frame_id_
            odom_msg.child_frame_id = tracked_object.child_frame_id
            odom_msg.pose.pose.position.x = x
            odom_msg.pose.pose.position.y = y
            odom_msg.pose.pose.position.z = z
            q = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
            odom_msg.pose.pose.orientation.x = q[0]
            odom_msg.pose.pose.orientation.y = q[1]
            odom_msg.pose.pose.orientation.z = q[2]
            odom_msg.pose.pose.orientation.w = q[3]
            odom_msg.twist.twist.linear.x = v_body[0]
            odom_msg.twist.twist.linear.y = v_body[1]
            odom_msg.twist.twist.linear.z = v_body[2]
            odom_msg.twist.twist.angular.x = ang_v_body[0]
            odom_msg.twist.twist.angular.y = ang_v_body[1]
            odom_msg.twist.twist.angular.z = ang_v_body[2]
            pose_covariance, twist_covariance = estimate.odometry_covariances()
            odom_msg.pose.covariance = pose_covariance.ravel().tolist()
            odom_msg.twist.covariance = twist_covariance.ravel().tolist()
            tracked_object.filtered_odom_publisher.publish(odom_msg)


def main(args=None):
    rclpy.init(args=args)
    vicon_stream_node = ViconStreamNode()
    executor = MultiThreadedExecutor()
    executor.add_node(vicon_stream_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    # vicon_stream_node.destroy_node()
    # rclpy.shutdown()


if __name__ == "__main__":
    main()
