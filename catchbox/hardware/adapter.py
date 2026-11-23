"""ROS node: move the box opening to requested positions, streaming joints through FrankaPy.

    python3 -m catchbox.hardware.adapter

The task space is the 0.2 m sphere around the calibrated origin, above
min_opening_z_m. Requested targets are clamped into it; IK goes to the nearest
reachable point with joints 1 and 4 (joint 6 keeps the box level). Every
streamed command is checked against the task space: a plan that would leave it
is held, and a command that leaves it stops the node.

Start with the robot at home (calibration --check1-home) and FrankaPy's control
PC server running. Motion follows adapter.enable_motion in configuration.yaml;
with it false the node plans and publishes what it would send, without moving.

Subscribes /catchbox_command_adapter/opening_target (geometry_msgs/PointStamped in panda_link0).
Publishes  /catchbox_command_adapter/joint_target  (sensor_msgs/JointState): every streamed command.
"""
import argparse
import threading
import time
from pathlib import Path
import numpy as np

from catchbox.hardware.calibration import BASE_LINK, DEFAULT_CONFIG, DEFAULT_MIN_Z, JOINT_NAMES, validate_config, vector
from catchbox.hardware.idle import select_target
from catchbox.hardware.ik import solve_restricted
from catchbox.hardware.limiter import JointLimiter
from catchbox.hardware.panda_kinematics import PandaKinematics

NODE = 'catchbox_command_adapter'
PLAN_HZ = 50
REPLAN_M = .001          # replan when the requested opening moves this far
HOME_TOLERANCE_RAD = .05  # IK holds joints 2, 3, 5, 7 at home, so start there
STREAM_TOLERANCE_M = .005


class TaskSpace:
    """The 0.2 m sphere around the origin, above the floor."""

    def __init__(self, origin, radius, min_z):
        self.origin, self.radius, self.min_z = np.asarray(origin, float), float(radius), float(min_z)
        if not np.isfinite(self.min_z) or self.origin[2] < self.min_z:
            raise ValueError('min_opening_z_m must be finite and below the calibration origin')

    def clamp(self, p):
        p = np.asarray(p, dtype=float).copy()
        delta = p - self.origin
        distance = np.linalg.norm(delta)
        if distance > self.radius:
            p = self.origin + delta * self.radius / distance
        p[2] = max(p[2], self.min_z)  # raising z toward the origin keeps p inside the sphere
        return p

    def contains(self, p, tolerance=0.0):
        return bool(np.linalg.norm(p - self.origin) <= self.radius + tolerance and p[2] >= self.min_z - tolerance)


class Adapter:
    def __init__(self, settings, arm):
        import rospy
        from geometry_msgs.msg import PointStamped
        from sensor_msgs.msg import JointState
        self.rospy, self.JointState = rospy, JointState
        cfg, ad = settings['calibration'], settings['adapter']
        offset, radius = validate_config(cfg)
        origin = vector(cfg.get('origin_xyz_m'), 3, 'origin_xyz_m')
        self.home = vector(cfg.get('home_joints_rad'), 7, 'home_joints_rad')
        self.space = TaskSpace(origin, radius, cfg.get('min_opening_z_m', DEFAULT_MIN_Z))
        self.kin = PandaKinematics(offset)
        if np.linalg.norm(self.kin.position(self.home) - origin) > .002:
            raise ValueError('origin_xyz_m does not match home_joints_rad; rerun --set-origin')
        self.lower, self.upper = self.kin.lower + .01, self.kin.upper - .01
        self.motion, self.rate_hz = bool(ad['enable_motion']), float(ad['rate_hz'])
        self.arm, self.lock = arm, threading.Lock()

        q = np.asarray(arm.get_joints(), dtype=float)
        if np.max(np.abs(q - self.home)) > HOME_TOLERANCE_RAD:
            raise ValueError('Robot is not at home; run calibration --check1-home first')
        self.limiter = JointLimiter(q, ad['max_joint_velocity'], ad['max_joint_acceleration'])
        self.joint_goal = q.copy()
        self.goal, self.goal_time, self.input_fault = None, time.monotonic(), False
        self.returning, self.planned, self.fault = False, None, None
        self.debug = rospy.Publisher('~joint_target', JointState, queue_size=1)
        self.sub = rospy.Subscriber('~opening_target', PointStamped, self.on_target, queue_size=1)

    def on_target(self, msg):
        p = np.array([msg.point.x, msg.point.y, msg.point.z])
        age = (self.rospy.Time.now() - msg.header.stamp).to_sec()
        bad = msg.header.frame_id != BASE_LINK or not np.isfinite(p).all() or msg.header.stamp.is_zero() or not 0 <= age <= .25
        with self.lock:
            if bad:
                self.goal, self.input_fault = None, True
            else:
                self.goal, self.goal_time, self.input_fault = self.space.clamp(p), time.monotonic(), False
        if bad:
            self.rospy.logwarn_throttle(2, 'Target has the wrong frame, a bad timestamp or non-finite values; ignoring')

    def stop(self, reason):
        with self.lock:
            if self.fault is None:
                self.fault = reason
            self.joint_goal = self.limiter.q.copy()
        self.rospy.logerr(f'Stopping: {reason}')

    def plan_loop(self):
        rate = self.rospy.Rate(PLAN_HZ)
        while not self.rospy.is_shutdown() and self.fault is None:
            try:
                self.plan_once()
            except Exception as e:  # stop rather than keep streaming an unknown plan
                self.stop(f'planning failed: {e}')
            rate.sleep()

    def plan_once(self):
        with self.lock:
            goal, goal_time, input_fault = self.goal, self.goal_time, self.input_fault
            q_cmd = self.limiter.q.copy()
        if self.motion and self.arm.is_skill_done(ignore_errors=False):
            return self.stop('FrankaPy dynamic skill ended; restart the adapter')
        mode, p, self.returning = select_target(self.space.origin, self.kin.position(q_cmd), goal,
                                                time.monotonic() - goal_time, 0.0, self.returning, input_fault)
        if mode == 'hold':
            with self.lock:
                self.joint_goal, self.planned = q_cmd, None
            return
        if self.planned is not None and np.linalg.norm(p - self.planned) < REPLAN_M:
            return
        q = solve_restricted(p, self.home, q_cmd, self.lower, self.upper, self.kin.position).joints
        reached = self.kin.position(q)
        with self.lock:
            if not self.space.contains(reached):
                self.rospy.logwarn_throttle(2, 'Nearest reachable point %s is outside the task space; holding',
                                            np.round(reached, 3).tolist())
                self.joint_goal, self.planned = q_cmd, None
            elif self.goal is goal or mode == 'return':  # skip plans for goals that have since changed
                self.joint_goal, self.planned = q, p

    def run(self):
        threading.Thread(target=self.plan_loop, daemon=True).start()
        stream = Stream(self.arm, self.limiter.q) if self.motion else None
        self.rospy.loginfo('Adapter running; motion %s', 'ENABLED' if self.motion else 'disabled (dry run)')
        dt, last = 1.0 / self.rate_hz, time.monotonic()
        rate = self.rospy.Rate(self.rate_hz)
        try:
            while not self.rospy.is_shutdown():
                now = time.monotonic()
                step_dt, last = min(now - last, 3 * dt), now
                with self.lock:
                    q = self.limiter.step(self.joint_goal, step_dt)
                if not self.space.contains(self.kin.position(q), STREAM_TOLERANCE_M):
                    self.stop('commanded opening left the task space')
                    break
                if stream is not None:
                    stream.send(q)
                msg = self.JointState()
                msg.header.stamp, msg.name, msg.position = self.rospy.Time.now(), JOINT_NAMES, q.tolist()
                self.debug.publish(msg)
                if self.fault is not None and np.max(np.abs(self.limiter.v)) < 1e-4:
                    break
                rate.sleep()
        finally:
            if stream is not None:
                stream.close()
        if self.fault is not None:
            raise SystemExit(f'Adapter stopped: {self.fault}')


class Stream:
    """FrankaPy dynamic joint skill: publish a new joint target every cycle."""

    SESSION_S = 3600.0

    def __init__(self, arm, q):
        import rospy
        from frankapy import FrankaConstants as FC
        from franka_interface_msgs.msg import SensorDataGroup
        self.rospy = rospy
        self.pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=10)
        arm.goto_joints(np.asarray(q).tolist(), duration=self.SESSION_S, dynamic=True,
                        buffer_time=10, ignore_errors=False)
        self.start, self.id = rospy.Time.now().to_time(), 0

    def send(self, q):
        from frankapy import SensorDataMessageType
        from frankapy.proto import JointPositionSensorMessage
        from frankapy.proto_utils import make_sensor_group_msg, sensor_proto2ros_msg
        self.id += 1
        proto = JointPositionSensorMessage(id=self.id, timestamp=self.rospy.Time.now().to_time() - self.start,
                                           joints=np.asarray(q).tolist())
        self.pub.publish(make_sensor_group_msg(trajectory_generator_sensor_msg=sensor_proto2ros_msg(
            proto, SensorDataMessageType.JOINT_POSITION)))

    def close(self):
        from frankapy import SensorDataMessageType
        from frankapy.proto import ShouldTerminateSensorMessage
        from frankapy.proto_utils import make_sensor_group_msg, sensor_proto2ros_msg
        proto = ShouldTerminateSensorMessage(timestamp=self.rospy.Time.now().to_time() - self.start,
                                             should_terminate=True)
        self.pub.publish(make_sensor_group_msg(termination_handler_sensor_msg=sensor_proto2ros_msg(
            proto, SensorDataMessageType.SHOULD_TERMINATE)))


def main():
    import rospy
    import yaml
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(rospy.myargv()[1:])
    settings = yaml.safe_load(args.config.read_text())
    rospy.init_node(NODE)
    from frankapy import FrankaArm
    Adapter(settings, FrankaArm(init_node=False)).run()


if __name__ == '__main__':
    main()
