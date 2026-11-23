"""Camera-to-arm bridge: predict where a thrown ball will reach the box and aim there.

In plain words, every camera frame this script:
  1. Finds the ball (track_ball.py detection) and converts it to robot base
     coordinates with the camera calibration.
  2. Fits the last ~0.2 s of positions with gravity included, giving position and
     velocity. No air drag is modeled.
  3. Predicts where the ball will come down through the box-opening height, and when.
  4. If that point is reachable and the prediction has settled, it is the target.
     By default the target is only printed (DRY RUN, the arm does not move).
     With --publish it is sent to Patrick's ROS 1 controller.

After each throw it prints where the ball REALLY crossed the opening height
versus what was predicted, so prediction quality can be checked with the arm off.

Examples:
  # Dry run: watch predictions, arm never moves
  python -B scripts/catch_bridge.py --calibration calibration/camera_to_base.json --preview
  # Also save every frame to a CSV for later analysis
  python -B scripts/catch_bridge.py --calibration calibration/camera_to_base.json --log throws.csv
  # Send targets to the controller (only after dry runs look right)
  python -B scripts/catch_bridge.py --calibration calibration/camera_to_base.json --publish

Controller contract (catchbox_control/franka_control.py on pbasq/update): a
geometry_msgs/PointStamped in frame panda_link0, stamped now, inside a 0.2 m
sphere around the recorded home opening and above min_opening_z_m. When targets
stop, the controller stops after 0.2 s and returns home after 3 s.
"""

import argparse
from collections import deque
import csv
from dataclasses import dataclass
import math
import os
import sys
import time

import numpy as np


GRAVITY = np.array([0.0, 0.0, -9.81])  # base frame, +z up (Franka mounted on a table)
# Patrick's recorded home opening and limits (pbasq/update configuration.yaml).
DEFAULT_ORIGIN = (0.4866, 0.0105, 0.2971)
DEFAULT_RADIUS_M = 0.20
DEFAULT_MIN_Z_M = 0.04
BASE_FRAME = "panda_link0"
TARGET_TOPIC = "/catchbox_command_adapter/opening_target"


# --------------------------------------------------------------------------
# Ball motion: least-squares fit with gravity (no camera or ROS needed)
# --------------------------------------------------------------------------

class BallisticFit:
    """Fit p(t) = p0 + v0 t + g t^2 / 2 to recent base-frame positions.

    A straight-line fit (as in track_ball.py) under-reads downward speed by
    about g * window / 2, roughly 0.7 m/s over 150 ms. Including gravity removes
    that bias, which matters a lot for where the ball comes down.
    """

    def __init__(self, window_s=0.20, max_gap_s=0.075, min_samples=4,
                 min_span_s=0.05, max_error_m=0.03):
        self.window_s, self.max_gap_s = window_s, max_gap_s
        self.min_samples, self.min_span_s, self.max_error_m = min_samples, min_span_s, max_error_m
        self.samples = deque()

    def reset(self):
        self.samples.clear()

    def update(self, t_s, position):
        """Add a sample. Return (position, velocity) at t_s, or None if not ready."""
        if self.samples and not 0 < t_s - self.samples[-1][0] <= self.max_gap_s:
            self.samples.clear()  # tracking gap or clock jump: start a new fit
        self.samples.append((t_s, np.asarray(position, float)))
        while t_s - self.samples[0][0] > self.window_s:
            self.samples.popleft()
        if len(self.samples) < self.min_samples or t_s - self.samples[0][0] < self.min_span_s:
            return None
        dt = np.array([s[0] - t_s for s in self.samples])  # <= 0, latest sample at 0
        pts = np.stack([s[1] for s in self.samples]) - 0.5 * np.outer(dt**2, GRAVITY)
        design = np.column_stack((np.ones_like(dt), dt))
        coef = np.linalg.lstsq(design, pts, rcond=None)[0]
        resid = pts - design @ coef
        if np.linalg.norm(resid, axis=1).max() > self.max_error_m:
            # A bounce or a false detection does not fit free flight.
            last = self.samples[-1]
            self.samples.clear()
            self.samples.append(last)
            return None
        return coef[0], coef[1]


def predict_crossing(position, velocity, height):
    """Where and when free flight next passes DOWN through z = height.

    Returns (point, seconds_from_now) or None if it never does.
    """
    g = -GRAVITY[2]
    a, b, c = -0.5 * g, velocity[2], position[2] - height
    disc = b * b - 4 * a * c
    if disc < 0:
        return None  # never gets up to that height
    t = (-b - math.sqrt(disc)) / (2 * a)  # later root: the descending crossing
    if t <= 0:
        return None  # already below and falling
    point = position + velocity * t + 0.5 * GRAVITY * t * t
    return point, t


# --------------------------------------------------------------------------
# Decision: when is a prediction good enough to aim at?
# --------------------------------------------------------------------------

@dataclass
class Decision:
    state: str                 # no ball | tracking | aiming | out of reach | too late | settling
    target: np.ndarray = None  # base-frame opening target when state == aiming
    time_to_arrival_s: float = None
    prediction: np.ndarray = None


class CatchPlanner:
    """Turn predictions into targets, only when they are reachable and settled."""

    def __init__(self, origin, radius, min_z, height, min_lead_s=0.03, max_lead_s=1.5,
                 settle_m=0.03, settle_frames=3):
        self.origin, self.radius, self.min_z = np.asarray(origin, float), radius, min_z
        self.height, self.min_lead_s, self.max_lead_s = height, min_lead_s, max_lead_s
        self.settle_m, self.settle_frames = settle_m, settle_frames
        self.recent = deque(maxlen=settle_frames)

    def reset(self):
        self.recent.clear()

    def decide(self, state):
        """state: (position, velocity) in base frame, or None."""
        if state is None:
            return Decision("tracking")
        hit = predict_crossing(state[0], state[1], self.height)
        if hit is None:
            self.recent.clear()
            return Decision("tracking")
        point, lead = hit
        if lead > self.max_lead_s:
            return Decision("tracking", time_to_arrival_s=lead, prediction=point)
        if lead < self.min_lead_s:
            return Decision("too late", time_to_arrival_s=lead, prediction=point)
        self.recent.append(point)
        if np.linalg.norm(point - self.origin) > self.radius or point[2] < self.min_z:
            return Decision("out of reach", time_to_arrival_s=lead, prediction=point)
        spread = (np.ptp(np.stack(self.recent), axis=0).max()
                  if len(self.recent) == self.settle_frames else math.inf)
        if spread > self.settle_m:
            return Decision("settling", time_to_arrival_s=lead, prediction=point)
        return Decision("aiming", point.copy(), lead, point)


# --------------------------------------------------------------------------
# Throw report: where did the ball really cross vs. what was predicted?
# --------------------------------------------------------------------------

class ThrowReport:
    """Collect one throw and compare predictions with the actual crossing."""

    def __init__(self, height, gap_s=0.3):
        self.height, self.gap_s = height, gap_s
        self.last_seen = None
        self.reset()

    def reset(self):
        self.prev = None
        self.predictions = []  # (time, predicted point, predicted crossing time)
        self.actual = None
        self.aimed = False

    def new_throw(self, t_s):
        """True when the ball reappears after a gap, i.e. a new throw starts."""
        fresh = self.last_seen is None or t_s - self.last_seen > self.gap_s
        self.last_seen = t_s
        return fresh

    def observe(self, t_s, position, decision):
        if decision.prediction is not None and decision.time_to_arrival_s is not None:
            self.predictions.append((t_s, decision.prediction, t_s + decision.time_to_arrival_s))
        self.aimed |= decision.state == "aiming"
        if self.actual is None and self.prev is not None:
            (t0, p0), (t1, p1) = self.prev, (t_s, position)
            if p0[2] >= self.height > p1[2]:
                u = (p0[2] - self.height) / (p0[2] - p1[2])
                self.actual = (t0 + u * (t1 - t0), p0 + u * (p1 - p0))
        self.prev = (t_s, position)

    def summary(self):
        if self.actual is None:
            return None
        t_cross, p_cross = self.actual
        lines = [f"THROW: ball crossed opening height at x={p_cross[0]:+.3f} "
                 f"y={p_cross[1]:+.3f} m"]
        for lead in (0.3, 0.2, 0.1):
            # Last prediction made at least `lead` seconds before the real crossing.
            early = [p for p in self.predictions if p[0] <= t_cross - lead]
            if early:
                _, point, t_pred = early[-1]
                err = 100 * np.linalg.norm((point - p_cross)[:2])
                lines.append(f"  prediction {1000 * lead:.0f} ms before: off by {err:.1f} cm "
                             f"| timing off by {1000 * (t_pred - t_cross):+.0f} ms")
        lines.append(f"  arm would have been aimed: {'yes' if self.aimed else 'NO'}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# ROS output (only with --publish)
# --------------------------------------------------------------------------

class RosTarget:
    def __init__(self, topic):
        import rospy
        from geometry_msgs.msg import PointStamped
        self.rospy, self.msg_type = rospy, PointStamped
        rospy.init_node("catchbox_catch_bridge", anonymous=True, disable_signals=True)
        self.pub = rospy.Publisher(topic, PointStamped, queue_size=1)

    def send(self, point):
        msg = self.msg_type()
        msg.header.stamp = self.rospy.Time.now()
        msg.header.frame_id = BASE_FRAME
        msg.point.x, msg.point.y, msg.point.z = (float(v) for v in point)
        self.pub.publish(msg)


# --------------------------------------------------------------------------
# Camera loop
# --------------------------------------------------------------------------

def run(rs, cv2, pipeline, profile, args):
    from calibrate_camera import load_calibration
    from track_ball import describe_detection, detect_ball, show_detection

    serial = profile.get_device().get_info(rs.camera_info.serial_number)
    calib = load_calibration(args.calibration, serial)
    height = args.height if args.height is not None else args.origin[2]
    fit = BallisticFit()
    planner = CatchPlanner(args.origin, args.radius, args.min_z, height)
    report = ThrowReport(height)
    sender = RosTarget(args.topic) if args.publish else None
    print(f"Opening height {height:.3f} m | reach: {100 * args.radius:.0f} cm around "
          f"({args.origin[0]:+.3f}, {args.origin[1]:+.3f}, {args.origin[2]:+.3f}) m | "
          f"{'PUBLISHING to ' + args.topic if sender else 'DRY RUN (arm will not move)'}")

    log = writer = None
    if args.log:
        log = open(args.log, "w", newline="")
        writer = csv.writer(log, lineterminator="\n")
        writer.writerow(["time_s", "latency_ms", "seen", "x", "y", "z", "vx", "vy", "vz",
                         "state", "target_x", "target_y", "target_z", "time_to_arrival_s"])

    align = rs.align(rs.stream.color)
    previous_gray = None
    last_color = last_depth = None
    last_print = -math.inf
    try:
        while True:
            frames = pipeline.wait_for_frames(2000)
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not (color and depth):
                continue
            numbers = (color.get_frame_number(), depth.get_frame_number())
            if numbers[0] == last_color or numbers[1] == last_depth:
                continue  # the SDK can repeat a frame; process each once
            last_color, last_depth = numbers

            image = np.asanyarray(color.get_data())
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ball = detect_ball(cv2, np, image, gray, previous_gray, args.brightness)
            previous_gray = gray
            t_s = color.get_timestamp() / 1000.0
            # With the global time domain, camera time tracks the host clock,
            # so this shows how old the frame is when we finish with it.
            latency_ms = 1000 * time.time() - color.get_timestamp()
            latency_ms = latency_ms if 0 <= latency_ms < 1000 else float("nan")
            cam_pos, status = describe_detection(
                rs, np, align, frames, color, depth, ball, args.max_skew_ms)

            if cam_pos is None:
                state, decision, pos = None, Decision("no ball"), None
            else:
                pos = calib.point(cam_pos)
                if report.new_throw(t_s):
                    done = report.summary()
                    if done:
                        print(done, flush=True)
                    report.reset()
                    fit.reset()
                    planner.reset()
                state = fit.update(t_s, pos)
                decision = planner.decide(state)
                report.observe(t_s, pos, decision)

            if sender is not None and decision.state == "aiming":
                sender.send(decision.target)

            if writer:
                v = state[1] if state is not None else (math.nan,) * 3
                p = pos if pos is not None else (math.nan,) * 3
                tgt = decision.target if decision.target is not None else (math.nan,) * 3
                tta = decision.time_to_arrival_s
                writer.writerow([f"{t_s:.4f}", f"{latency_ms:.1f}", int(pos is not None),
                                 *(f"{x:.4f}" for x in p), *(f"{x:.3f}" for x in v),
                                 decision.state, *(f"{x:.4f}" for x in tgt),
                                 "" if tta is None else f"{tta:.3f}"])

            text = decision.state
            if decision.prediction is not None:
                q = decision.prediction
                text += (f" | lands x={q[0]:+.3f} y={q[1]:+.3f} in "
                         f"{1000 * decision.time_to_arrival_s:.0f} ms")
            now = time.monotonic()
            if decision.state != "no ball" and (args.every_frame or now - last_print >= 0.1):
                print(f"t={t_s:.3f} | lag {latency_ms:.0f} ms | {text}", flush=True)
                last_print = now
            if args.preview and not show_detection(cv2, image, ball, pos is not None, text):
                break
    finally:
        done = report.summary()
        if done:
            print(done)
        if log:
            log.close()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration", required=True,
                        help="Camera-to-base JSON from calibrate_camera(_live).py")
    parser.add_argument("--publish", action="store_true",
                        help="Send targets to the ROS controller (default: dry run)")
    parser.add_argument("--topic", default=TARGET_TOPIC, help="PointStamped target topic")
    parser.add_argument("--origin", type=float, nargs=3, default=DEFAULT_ORIGIN,
                        metavar=("X", "Y", "Z"),
                        help="Home box-opening position, base frame (m); match configuration.yaml")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS_M,
                        help="Reach around --origin the controller accepts (m, default 0.2)")
    parser.add_argument("--min-z", type=float, default=DEFAULT_MIN_Z_M,
                        help="Lowest allowed target height (m); match min_opening_z_m")
    parser.add_argument("--height", type=float,
                        help="Catch height in base frame (m, default: origin z)")
    parser.add_argument("--fps", type=int, choices=(30, 60), default=30,
                        help="Camera rate (60 needs USB 3, default 30)")
    parser.add_argument("--brightness", type=int, default=170, help="Ball brightness threshold")
    parser.add_argument("--max-skew-ms", type=float, default=12.0,
                        help="Maximum RGB-depth timestamp difference (ms)")
    parser.add_argument("--preview", action="store_true", help="Show the camera view")
    parser.add_argument("--every-frame", action="store_true", help="Print every frame")
    parser.add_argument("--log", help="Write every frame to this CSV file")
    args = parser.parse_args()
    if not 0 < args.radius <= 0.2:
        parser.error("--radius must be in (0, 0.2]; the controller rejects larger")
    if args.preview and sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        parser.error("--preview requires a graphical desktop session")

    import cv2
    import pyrealsense2 as rs
    from check_camera import camera_session, print_stream_profiles

    try:
        with camera_session(rs, fps=args.fps) as (pipeline, profile):
            print_stream_profiles(rs, profile)
            run(rs, cv2, pipeline, profile, args)
    except KeyboardInterrupt:
        print("\nStopped. If publishing, the controller stops and returns home on its own.")
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if args.preview:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
