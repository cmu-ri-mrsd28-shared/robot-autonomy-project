"""ROS 1 camera node: publish the tracked ball for the catching model (model_node).

Runs the same vision pipeline as measure_ball.py (lock-on tracking of the orange
ball, catch box found by its tape stripes, "ball in box" detection) on the
RealSense, converts the ball into the Franka base frame with the camera
calibration, and publishes the three camera topics model_node.py subscribes to
(see docs/HARDWARE_ROS1.md):

  /catchbox/ball_state            nav_msgs/Odometry, frame panda_link0, stamped at
                                  capture time: ball position (m) and velocity (m/s).
                                  Published only while the ball is visible.
  /catchbox/ball_tracking_status  std_msgs/Int8, every frame:
                                  -1 no active ball, 0 hidden, 1 visible without
                                  velocity, 2 visible with valid velocity
  /catchbox/ball_in_box           std_msgs/Int8, every frame:
                                  -1 unknown, 0 outside, 1 inside / probably inside

Status and in-box are sent every camera frame (30 Hz), well inside model_node's
0.2 s freshness limit. The ball is "inside" only with camera evidence: seen in the
box, or it vanished into the box (measure_ball.BallLock); a ball that merely
disappeared is "hidden" with in-box unknown.

Run on the computer with the camera, with ROS 1 sourced, from the project folder:

  python3 -B scripts/ball_vision_node.py --ball orange \\
      --calibration calibration/camera_to_base.json --preview

Keep the ball out of view for the first 1.5 s while it learns the empty scene.
All measure_ball.py tracking options work here too (--fps, --sat-min, ...).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import measure_ball as mb  # noqa: E402

BASE_LINK = "panda_link0"   # = catchbox.hardware.calibration.BASE_LINK (model_node checks it)
NO_BALL, HIDDEN, VISIBLE, TRACKED = -1, 0, 1, 2
UNKNOWN, OUTSIDE, INSIDE = -1, 0, 1
MAX_CLOCK_SKEW_S = 0.5      # trust the camera's capture time only if it agrees with ours
GRAVITY = 9.81
# Velocity is estimated exactly like the simulated camera the model was trained with
# (scripts/steady_motion_sensing.py, CameraEstimator): a least-squares fit to the last
# 5 sightings, assuming free fall (z up in panda_link0), valid from 3 sightings, and
# restarted whenever the ball is not seen.
VELOCITY_SAMPLES = 5
MIN_VELOCITY_SAMPLES = 3
MAX_VELOCITY_GAP_S = 0.1    # a gap longer than this restarts the fit (the sim restarts on any)


class BallisticVelocity:
    """CameraEstimator's velocity fit, in the Franka base frame.

    For a ball in free fall this gives the velocity at the latest frame (no lag).
    Like the simulation, a ball held still reads a small downward velocity
    (about -0.6 m/s at 30 fps): the model was trained with the same behavior.
    """

    def __init__(self, np):
        self.np = np
        self.history = []

    def reset(self):
        self.history = []

    def update(self, t, position):
        np = self.np
        if self.history and not 0 < t - self.history[-1][0] <= MAX_VELOCITY_GAP_S:
            self.history = []
        self.history = (self.history + [(t, np.asarray(position, float))])[-VELOCITY_SAMPLES:]
        if len(self.history) < MIN_VELOCITY_SAMPLES:
            return None
        times = np.array([h[0] - t for h in self.history])
        xyz = np.array([h[1] for h in self.history])
        adjusted = xyz.copy()
        adjusted[:, 2] += 0.5 * GRAVITY * times**2
        centered = times - times.mean()
        return (centered[:, None] * (adjusted - adjusted.mean(axis=0))).sum(axis=0) / (
            centered @ centered)


def camera_status(lock, found, now):
    """(tracking_status, in_box) for this frame, from the tracker's state.

    lock: measure_ball.BallLock after this frame's update.
    found: the ball measured this frame (or None). The returned status is VISIBLE
    when found; the caller upgrades it to TRACKED once a velocity is available.
    """
    if found is not None and lock.state == "LOCKED":
        in_box = INSIDE if lock.in_box_3d(found.detection, found.position) else OUTSIDE
        return VISIBLE, in_box
    if lock.in_box_balls:
        # It went into the box (seen vanishing there): active, hidden, inside.
        return HIDDEN, INSIDE
    if lock.state in ("LOCKED", "LOST"):
        # The locked ball just disappeared somewhere else: still active, but hidden,
        # and there's no evidence it is in the box.
        return HIDDEN, UNKNOWN
    return NO_BALL, UNKNOWN   # searching, or only a would-be ball (LOCKING)


class BallOutput:
    """Turns tracker results into the values model_node expects (no ROS here)."""

    def __init__(self, np, camera_to_base):
        self.np = np
        self.camera_to_base = camera_to_base
        self.velocity = BallisticVelocity(np)
        self.ball_id = None

    def update(self, lock, found, now, captured_s):
        """Return dict(status, in_box, position, velocity) in the Franka base frame.

        position/velocity are None unless the ball is visible (velocity also needs
        3 sightings in a row). captured_s is the frame's capture time in seconds.
        """
        status, in_box = camera_status(lock, found, now)
        position = velocity = None
        if status == VISIBLE:
            if lock.ball_id != self.ball_id:
                self.velocity.reset()   # a new ball: don't mix its track with the last
                self.ball_id = lock.ball_id
            position = self.camera_to_base.point(self.np.asarray(found.position, float))
            fitted = self.velocity.update(captured_s, position)
            if fitted is not None and self.np.isfinite(fitted).all():
                status, velocity = TRACKED, fitted
        else:
            self.velocity.reset()
        return {"status": status, "in_box": in_box, "position": position, "velocity": velocity}


def capture_time_s(rs, frame, wall_now):
    """The frame's capture time on the wall clock (what rospy.Time uses), if known.

    With the RealSense "global time" domain the timestamp is host-synchronized
    ms. Otherwise (or if it disagrees with the wall clock) use arrival time.
    """
    try:
        global_domain = frame.get_frame_timestamp_domain() == rs.timestamp_domain.global_time
    except (AttributeError, RuntimeError):
        global_domain = False
    if global_domain:
        captured = frame.get_timestamp() / 1000.0
        if 0.0 <= wall_now - captured <= MAX_CLOCK_SKEW_S:
            return captured
    return wall_now


def fill_odometry(msg, stamp, position, velocity):
    """Fill a nav_msgs/Odometry the way model_node reads it."""
    msg.header.stamp = stamp
    msg.header.frame_id = BASE_LINK
    msg.child_frame_id = BASE_LINK
    p = msg.pose.pose.position
    p.x, p.y, p.z = (float(v) for v in position)
    msg.pose.pose.orientation.w = 1.0
    v = msg.twist.twist.linear
    v.x, v.y, v.z = (0.0, 0.0, 0.0) if velocity is None else (float(x) for x in velocity)
    return msg


class CameraTopics:
    """The node's three ROS publishers (also used by docker/vision_node_ros_test.py)."""

    def __init__(self, rospy):
        from nav_msgs.msg import Odometry
        from std_msgs.msg import Int8
        self.rospy, self.Odometry, self.Int8 = rospy, Odometry, Int8
        self.ball = rospy.Publisher("/catchbox/ball_state", Odometry, queue_size=1)
        self.status = rospy.Publisher("/catchbox/ball_tracking_status", Int8, queue_size=1)
        self.in_box = rospy.Publisher("/catchbox/ball_in_box", Int8, queue_size=1)

    def publish(self, out, captured_s):
        """Publish one frame's BallOutput.update() result, stamped at capture time."""
        if out["position"] is not None:
            self.ball.publish(fill_odometry(self.Odometry(), self.rospy.Time.from_sec(captured_s),
                                            out["position"], out["velocity"]))
        self.flags(out["status"], out["in_box"])

    def flags(self, status, in_box):
        self.status.publish(self.Int8(data=status))
        self.in_box.publish(self.Int8(data=in_box))


def main():
    import rospy

    parser = mb.build_parser(__doc__)
    parser.add_argument("--calibration", required=True,
                        help="Camera-to-Franka-base calibration JSON (camera_to_base.json)")
    parser.add_argument("--preview", action="store_true",
                        help="Also show the camera window (needs a desktop)")
    parser.set_defaults(ball="orange")
    args = parser.parse_args(rospy.myargv()[1:])
    mb.apply_settings(parser, args)

    import cv2
    import numpy as np
    import pyrealsense2 as rs
    from calibrate_camera import load_calibration
    from check_camera import camera_session, print_stream_profiles

    rospy.init_node("catchbox_ball_vision")
    topics = CameraTopics(rospy)

    with camera_session(rs, fps=args.fps) as (pipeline, profile):
        print_stream_profiles(rs, profile)
        serial = profile.get_device().get_info(rs.camera_info.serial_number)
        camera_to_base = load_calibration(args.calibration, serial)
        intrinsics = (profile.get_stream(rs.stream.color)
                      .as_video_stream_profile().get_intrinsics())
        align = rs.align(rs.stream.color)
        tracker = mb.VisionTracker(rs, cv2, np, intrinsics, args)
        output = BallOutput(np, camera_to_base)
        rospy.loginfo("Learning the empty scene: keep the ball OUT of view...")
        previous_numbers, previous_status = (None, None), None
        try:
            while not rospy.is_shutdown():
                frames = pipeline.wait_for_frames(5000)
                color, depth = frames.get_color_frame(), frames.get_depth_frame()
                if not (color and depth):
                    continue
                numbers = (color.get_frame_number(), depth.get_frame_number())
                if numbers[0] == previous_numbers[0] or numbers[1] == previous_numbers[1]:
                    continue
                previous_numbers = numbers
                now = time.monotonic()
                captured = capture_time_s(rs, color, time.time())
                image = np.asanyarray(color.get_data())

                if tracker.learning:
                    if tracker.learn(image):
                        rospy.loginfo("Empty scene learned. Put the ball in view.")
                    topics.flags(NO_BALL, UNKNOWN)
                    continue

                aligned = align.process(frames).get_depth_frame() or None
                tracker.detect(image, aligned, now)
                found, shown, _ = tracker.track(image, aligned, now)
                for text in tracker.lock.events:
                    rospy.loginfo("%s | ball in box: %s", text,
                                  "YES" if tracker.lock.in_box_balls else "NO")
                tracker.lock.events.clear()

                out = output.update(tracker.lock, found, now, captured)
                topics.publish(out, captured)
                if out["status"] != previous_status:
                    rospy.loginfo("tracking status %d, in box %d", out["status"], out["in_box"])
                    previous_status = out["status"]

                if args.preview:
                    names = {NO_BALL: "NO BALL", HIDDEN: "HIDDEN", VISIBLE: "VISIBLE",
                             TRACKED: "TRACKED"}
                    lines = [f"in box: {out['in_box']}  (-1 unknown, 0 out, 1 in)"]
                    if out["position"] is not None:
                        x, y, z = out["position"]
                        lines.append(f"base XYZ  {x:+.3f} {y:+.3f} {z:+.3f} m")
                    if out["velocity"] is not None:
                        vx, vy, vz = out["velocity"]
                        lines.append(f"velocity  {vx:+.2f} {vy:+.2f} {vz:+.2f} m/s")
                    key = mb.draw(cv2, np, image, shown, f"ROS: {names[out['status']]}",
                                  mb.GREEN if out["status"] > 0 else mb.YELLOW,
                                  f"status {out['status']}", lines, mb.GREEN,
                                  box=tracker.lock.box, box_label=tracker.box_info,
                                  button_text=None)
                    if key in (ord("q"), 27):
                        break
                    if key == ord("b"):
                        tracker.relearn()
        finally:
            # Leave the model with "no active ball" rather than a stale last state.
            try:
                topics.flags(NO_BALL, UNKNOWN)
            except Exception:  # noqa: BLE001 - ROS may already be shutting down
                pass
            if args.preview:
                cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
