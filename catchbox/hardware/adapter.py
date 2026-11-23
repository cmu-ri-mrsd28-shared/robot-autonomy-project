"""ROS node: turn box-opening targets into checked joint targets streamed through FrankaPy.

    python3 -m catchbox.hardware.adapter              # MoveIt collision screening required
    python3 -m catchbox.hardware.adapter --no-moveit  # bring-up only: sphere, floor and FrankaPy walls

Start with the robot at home (check1-home) and FrankaPy's control PC server
running. Motion follows adapter.enable_motion in configuration.yaml; with it
false the node plans and publishes what it would send, without moving.

Subscribes /catchbox_command_adapter/opening_target (geometry_msgs/PointStamped in panda_link0).
Publishes  /catchbox_command_adapter/joint_target  (sensor_msgs/JointState): every streamed command.
"""
import argparse
import threading
import time
from pathlib import Path
import numpy as np

from catchbox.hardware.calibration import (BASE_LINK, DEFAULT_CONFIG, DEFAULT_MIN_Z, JOINT_NAMES,
                                           MOVE_GROUP, WORKSPACE_RADIUS_M, validate_config, vector)
from catchbox.hardware.idle import select_target
from catchbox.hardware.ik import solve_restricted
from catchbox.hardware.limiter import JointLimiter
from catchbox.hardware.panda_kinematics import PandaKinematics

NODE = 'catchbox_command_adapter'
PLAN_HZ = 20
REPLAN_M = .001          # replan when the requested opening moves this far
HOME_TOLERANCE_RAD = .05
PATH_SAMPLES = 21


class Adapter:
    def __init__(self, settings, arm, use_moveit=True):
        import rospy
        from geometry_msgs.msg import PointStamped
        from sensor_msgs.msg import JointState
        self.rospy, self.JointState = rospy, JointState
        cfg, ad = settings['calibration'], settings['adapter']
        offset, self.radius = validate_config(cfg)
        self.origin = vector(cfg.get('origin_xyz_m'), 3, 'origin_xyz_m')
        self.home = vector(cfg.get('home_joints_rad'), 7, 'home_joints_rad')
        self.min_z = float(cfg.get('min_opening_z_m', DEFAULT_MIN_Z))
        if not np.isfinite(self.min_z) or self.origin[2] < self.min_z:
            raise ValueError('min_opening_z_m must be finite and below the calibration origin')
        self.kin = PandaKinematics(offset)
        if np.linalg.norm(self.kin.position(self.home) - self.origin) > .002:
            raise ValueError('origin_xyz_m does not match home_joints_rad; rerun --set-origin')
        self.lower, self.upper = self.kin.lower + .01, self.kin.upper - .01
        self.motion = bool(ad['enable_motion'])
        self.rate_hz = float(ad['rate_hz'])
        self.tracking_limit = float(ad['tracking_limit_rad'])
        self.arm, self.lock = arm, threading.Lock()

        q = np.asarray(arm.get_joints(), dtype=float)
        if np.max(np.abs(q - self.home)) > HOME_TOLERANCE_RAD:
            raise ValueError('Robot is not at home; run calibration --check1-home first')
        self.limiter = JointLimiter(q, ad['max_joint_velocity'], ad['max_joint_acceleration'])
        self.joint_goal = q.copy()
        self.goal, self.goal_time, self.input_fault = None, time.monotonic(), False
        self.returning, self.planned, self.fault = False, None, None

        self.check_state = None
        if use_moveit:
            from moveit_msgs.srv import GetStateValidity
            rospy.wait_for_service('/check_state_validity', timeout=10)
            self.check_state = rospy.ServiceProxy('/check_state_validity', GetStateValidity)
        self.debug = rospy.Publisher('~joint_target', JointState, queue_size=1)
        self.sub = rospy.Subscriber('~opening_target', PointStamped, self.on_target, queue_size=1)

    # Inputs -------------------------------------------------------------
    def on_target(self, msg):
        p = np.array([msg.point.x, msg.point.y, msg.point.z])
        age = (self.rospy.Time.now() - msg.header.stamp).to_sec()
        bad = (msg.header.frame_id != BASE_LINK or not np.isfinite(p).all() or msg.header.stamp.is_zero()
               or age < 0 or age > .25 or np.linalg.norm(p - self.origin) > self.radius + 1e-9 or p[2] < self.min_z)
        with self.lock:
            if bad:
                self.goal, self.input_fault = None, True
            else:
                self.goal, self.goal_time, self.input_fault = p, time.monotonic(), False
        if bad:
            self.rospy.logwarn_throttle(2, 'Invalid, stale, out-of-sphere or below-floor target; rejecting')

    # Checks -------------------------------------------------------------
    def state_valid(self, q):
        if np.any(q < self.lower) or np.any(q > self.upper):
            return False
        p = self.kin.position(q)
        if np.linalg.norm(p - self.origin) > self.radius + .001 or p[2] < self.min_z - .001:
            return False
        if self.arm.is_joints_in_collision_with_boxes(q.tolist()):
            return False
        if self.check_state is not None:
            from moveit_msgs.msg import RobotState
            from moveit_msgs.srv import GetStateValidityRequest
            req = GetStateValidityRequest()
            req.group_name, req.robot_state = MOVE_GROUP, RobotState()
            req.robot_state.is_diff = True
            req.robot_state.joint_state.name = JOINT_NAMES
            req.robot_state.joint_state.position = q.tolist()
            return bool(self.check_state(req).valid)
        return True

    def path_valid(self, start, end):
        # Sampled screening of the straight joint-space line, not a continuous guarantee.
        return all(self.state_valid(start + (end - start) * f) for f in np.linspace(0, 1, PATH_SAMPLES))

    def stop(self, reason):
        with self.lock:
            if self.fault is None:
                self.fault = reason
            self.joint_goal = self.limiter.q.copy()
        self.rospy.logerr(f'Stopping: {reason}')

    # Planning thread ----------------------------------------------------
    def plan_loop(self):
        rate = self.rospy.Rate(PLAN_HZ)
        while not self.rospy.is_shutdown() and self.fault is None:
            try:
                self.plan_once()
            except Exception as e:  # any failure stops motion rather than continuing blind
                self.stop(f'planning failed: {e}')
            rate.sleep()

    def plan_once(self):
        with self.lock:
            goal, goal_time, input_fault = self.goal, self.goal_time, self.input_fault
            q_cmd = self.limiter.q.copy()
        if self.motion:
            if self.arm.is_skill_done(ignore_errors=False):
                return self.stop('FrankaPy dynamic skill ended; restart the adapter')
            measured = np.asarray(self.arm.get_joints(), dtype=float)
            if np.max(np.abs(measured - q_cmd)) > self.tracking_limit:
                return self.stop('robot is not following the commanded joints')
        mode, p, self.returning = select_target(self.origin, self.kin.position(q_cmd), goal,
                                                time.monotonic() - goal_time, 0.0, self.returning, input_fault)
        if mode == 'hold':
            with self.lock:
                self.joint_goal, self.planned = q_cmd, None
            return
        if self.planned is not None and np.linalg.norm(p - self.planned) < REPLAN_M:
            return
        ik = solve_restricted(p, self.home, q_cmd, self.lower, self.upper, self.kin.position)
        if not ik.reachable:
            self.rospy.logwarn_throttle(2, 'Target unreachable under joint restrictions (error %.3f m)', ik.error_m)
            ok = False
        elif not self.path_valid(q_cmd, ik.joints):
            self.rospy.logwarn_throttle(2, 'Joint path failed collision, sphere or floor screening')
            ok = False
        else:
            ok = True
        with self.lock:
            # Never apply a plan for a goal that has since changed or gone stale.
            if ok and (self.goal is goal or mode == 'return'):
                self.joint_goal, self.planned = ik.joints, p
            elif not ok:
                self.joint_goal, self.planned = q_cmd, None

    # Streaming (main thread) --------------------------------------------
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
                p = self.kin.position(q)
                if np.linalg.norm(p - self.origin) > self.radius + .005 or p[2] < self.min_z - .005:
                    self.stop('commanded opening left the sphere or went below the floor')
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
        self.rospy, self.arm = rospy, arm
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
    parser.add_argument('--no-moveit', action='store_true',
                        help='Skip MoveIt collision screening (bring-up only; sphere, floor and FrankaPy walls still apply)')
    args = parser.parse_args(rospy.myargv()[1:])
    settings = yaml.safe_load(args.config.read_text())
    rospy.init_node(NODE)
    from frankapy import FrankaArm
    arm = FrankaArm(init_node=False)
    if args.no_moveit:
        rospy.logwarn('MoveIt collision screening is OFF')
    Adapter(settings, arm, use_moveit=not args.no_moveit).run()


if __name__ == '__main__':
    main()
