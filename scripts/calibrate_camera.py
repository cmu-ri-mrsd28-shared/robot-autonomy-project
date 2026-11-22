"""Find where the D435 camera sits relative to the Franka base (camera-to-base calibration).

How it works, in plain words:
  1. Stick a ping-pong ball onto the box on the robot, where the camera can see it.
  2. Hand-guide the arm to a spot, let it settle, press Enter. The script records
     where the camera sees the ball and the robot's end-effector pose.
  3. Repeat at 10-15 spots spread over the catching area. Also turn the wrist
     differently between spots; this lets the script work out exactly where the
     ball sits on the box, so you do not have to measure that precisely.
  4. Type "solve". The script prints the error at each spot and saves the result.

Afterwards a camera point p_cam (meters, camera frame) converts to the robot base
frame with:  p_base = R @ p_cam + t, using R and t from the saved JSON file.
Velocities convert with R only.

The ball must be still. This script does not move the robot.

Examples:
  # Read the robot pose over ROS 1 (franka_ros):
  python scripts/calibrate_camera.py --robot ros --preview
  # Read the robot pose with franky (libfranka Python bindings):
  python scripts/calibrate_camera.py --robot franky --robot-ip 172.16.0.2 --preview
  # Type or paste the robot pose by hand:
  python scripts/calibrate_camera.py --robot manual --preview
  # Re-solve saved samples, leaving out bad spots 3 and 7:
  python scripts/calibrate_camera.py --refit calibration/camera_samples.json --drop 3 7
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

from track_ball import Detection, estimate_position


WINDOW = "Camera calibration: still ball"
DEFAULT_OUT = Path("calibration/camera_to_base.json")
DEFAULT_SAMPLES = Path("calibration/camera_samples.json")
MIN_SAMPLES = 5
# Turning the wrist at least this much between spots lets the solver find the
# ball's position on the box. Below it, a measured --ball-offset is required.
MIN_ROTATION_SPREAD_DEG = 25.0


# --------------------------------------------------------------------------
# Math: fitting the camera pose
# --------------------------------------------------------------------------

def nearest_rotation(m):
    """Closest proper rotation matrix to m."""
    u, _, vt = np.linalg.svd(m)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt


def kabsch(source, target):
    """Rotation R and translation t minimizing |R @ source + t - target|."""
    sc, tc = source.mean(axis=0), target.mean(axis=0)
    r = nearest_rotation((target - tc).T @ (source - sc))
    return r, tc - r @ sc


def rotation_spread_deg(rotations):
    """Largest angle between any two end-effector orientations."""
    worst = 0.0
    for i in range(len(rotations)):
        for j in range(i + 1, len(rotations)):
            c = (np.trace(rotations[i].T @ rotations[j]) - 1) / 2
            worst = max(worst, math.degrees(math.acos(np.clip(c, -1, 1))))
    return worst


def solve(camera_points, ee_poses, ball_offset=None, iterations=200):
    """Fit camera-to-base (R, t) and, if not given, the ball offset in the EE frame.

    camera_points: (N, 3) ball centers in the camera frame.
    ee_poses: (N, 4, 4) end-effector poses in the base frame.
    Model: R @ p_cam + t = R_ee @ offset + p_ee.
    """
    cam = np.asarray(camera_points, float)
    rot = np.asarray(ee_poses, float)[:, :3, :3]
    pos = np.asarray(ee_poses, float)[:, :3, 3]
    fixed = ball_offset is not None

    if fixed:
        offset = np.asarray(ball_offset, float)
    else:
        # Linear first guess: treat the 9 entries of R as free, solve for
        # R, t and offset together, then snap R to a true rotation.
        rows, rhs = [], []
        for p, r_ee, p_ee in zip(cam, rot, pos):
            a = np.zeros((3, 15))
            for k in range(3):
                a[k, 3 * k:3 * k + 3] = p
            a[:, 9:12] = np.eye(3)
            a[:, 12:15] = -r_ee
            rows.append(a)
            rhs.append(p_ee)
        x = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]
        offset = x[12:15]

    # Alternate: best camera pose for this offset, then best offset for that pose.
    for _ in range(iterations):
        target = np.einsum("nij,j->ni", rot, offset) + pos
        r, t = kabsch(cam, target)
        if fixed:
            break
        world = cam @ r.T + t
        new_offset = np.linalg.lstsq(
            np.vstack(rot), np.concatenate(world - pos), rcond=None
        )[0]
        if np.linalg.norm(new_offset - offset) < 1e-9:
            offset = new_offset
            break
        offset = new_offset

    target = np.einsum("nij,j->ni", rot, offset) + pos
    r, t = kabsch(cam, target)
    errors = np.linalg.norm(cam @ r.T + t - target, axis=1)
    return r, t, offset, errors


# --------------------------------------------------------------------------
# Camera: finding a still white ball
# --------------------------------------------------------------------------

def find_still_ball(cv2, image, brightness):
    """Return round white blobs, most circular first. No motion is required."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, brightness), (179, 75, 255))
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for contour in contours:
        (u, v), radius = cv2.minEnclosingCircle(contour)
        perimeter = cv2.arcLength(contour, True)
        if not 3 <= radius <= 60 or perimeter == 0:
            continue
        circularity = 4 * math.pi * cv2.contourArea(contour) / perimeter**2
        if circularity >= 0.6:
            found.append((circularity, Detection(round(u), round(v), radius)))
    found.sort(key=lambda item: -item[0])
    return [d for _, d in found]


def locate(rs, cv2, align, frames, brightness):
    """Ball center in the camera frame for one frame pair, plus the detection."""
    aligned = align.process(frames)
    color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
    if not (color and depth):
        return None, None, None
    image = np.asanyarray(color.get_data())
    for ball in find_still_ball(cv2, image, brightness):
        # estimate_position also checks the blob's size against its depth,
        # which rejects most white things that are not a 40 mm ball.
        position, _ = estimate_position(rs, np, depth, ball)
        if position is not None:
            return position, ball, image
    return None, None, image


def draw(cv2, image, ball, text):
    preview = image.copy()
    if ball is not None:
        cv2.circle(preview, (ball.u, ball.v), round(ball.radius), (70, 240, 80), 2)
    cv2.rectangle(preview, (0, 0), (preview.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(preview, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow(WINDOW, preview)
    cv2.waitKey(1)


def average_ball(rs, cv2, pipeline, align, args):
    """Average the still ball's camera position over about args.average_s seconds."""
    points, ball, image = [], None, None
    frames_seen = 0
    end = time.monotonic() + args.average_s
    while time.monotonic() < end:
        frames = pipeline.wait_for_frames(2000)
        frames_seen += 1
        p, b, img = locate(rs, cv2, align, frames, args.brightness)
        if img is not None:
            image = img
        if p is not None:
            points.append(p)
            ball = b
        if args.preview and image is not None:
            draw(cv2, image, ball, f"averaging... {len(points)}/{frames_seen} frames valid")
    if frames_seen == 0 or len(points) < 0.6 * frames_seen:
        return None, f"ball found in only {len(points)}/{frames_seen} frames"
    pts = np.array(points)
    median = np.median(pts, axis=0)
    keep = pts[np.linalg.norm(pts - median, axis=1) < 0.01]
    if len(keep) < 0.5 * len(pts):
        return None, "ball position jumped around; is it still and clearly visible?"
    spread_mm = 1000 * keep.std(axis=0).max()
    return keep.mean(axis=0), f"{len(keep)} frames, spread {spread_mm:.1f} mm"


def live_preview(rs, cv2, pipeline, align, args):
    """Keep the preview window updated while waiting for typed input."""
    frames = pipeline.poll_for_frames()
    if frames:
        p, ball, image = locate(rs, cv2, align, frames, args.brightness)
        if image is not None:
            text = ("ball found  XYZ=({:+.3f}, {:+.3f}, {:.3f}) m".format(*p)
                    if p is not None else "ball NOT found")
            draw(cv2, image, ball, text + " | press Enter in terminal")


# --------------------------------------------------------------------------
# Robot: reading the end-effector pose (read only, never commands motion)
# --------------------------------------------------------------------------

def column_major_to_matrix(values):
    """Franka's O_T_EE is 16 numbers in column-major order."""
    m = np.asarray(values, float).reshape(4, 4).T
    if not np.allclose(m[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("pose is not a valid 4x4 transform (check number order)")
    if not np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-3):
        raise ValueError("pose rotation is not orthonormal (check number order)")
    return m


class RosPose:
    """Read O_T_EE from franka_ros (ROS 1)."""

    def __init__(self, topic):
        import rospy
        from franka_msgs.msg import FrankaState
        self.rospy, self.msg_type, self.topic = rospy, FrankaState, topic
        rospy.init_node("catchbox_camera_calibration", anonymous=True,
                        disable_signals=True)

    def read(self):
        msg = self.rospy.wait_for_message(self.topic, self.msg_type, timeout=2.0)
        return column_major_to_matrix(msg.O_T_EE)


class FrankyPose:
    """Read the end-effector pose with franky (libfranka). Does not move the robot."""

    def __init__(self, ip):
        from franky import Robot
        self.robot = Robot(ip)

    def read(self):
        state = self.robot.state
        return column_major_to_matrix(np.asarray(state.O_T_EE.matrix).T.reshape(-1))


class ManualPose:
    """Paste the 16 O_T_EE numbers (column-major, as Franka prints them)."""

    def read(self):
        text = input("  Paste O_T_EE (16 numbers): ").replace(",", " ").replace("[", " ")
        values = [float(x) for x in text.replace("]", " ").split()]
        if len(values) != 16:
            raise ValueError(f"expected 16 numbers, got {len(values)}")
        return column_major_to_matrix(values)


# --------------------------------------------------------------------------
# Saving and reporting
# --------------------------------------------------------------------------

def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def report_and_save(samples, args):
    keep = [s for i, s in enumerate(samples, 1) if i not in set(args.drop or [])]
    if len(keep) < MIN_SAMPLES:
        print(f"Need at least {MIN_SAMPLES} spots to solve (have {len(keep)}).")
        return False
    cam = np.array([s["camera_xyz_m"] for s in keep])
    poses = np.array([s["ee_pose_base"] for s in keep])
    spread = rotation_spread_deg(poses[:, :3, :3])
    offset = args.ball_offset
    if offset is None and spread < MIN_ROTATION_SPREAD_DEG:
        print(f"The wrist only turned {spread:.0f} degrees across spots. Either turn it "
              f"more (at least {MIN_ROTATION_SPREAD_DEG:.0f}) between spots, or measure "
              "the ball position on the box and pass --ball-offset X Y Z.")
        return False

    r, t, offset, errors = solve(cam, poses, offset)
    print("\nError at each spot (distance between camera and robot after calibration):")
    numbered = [i for i in range(1, len(samples) + 1) if i not in set(args.drop or [])]
    for i, e in zip(numbered, errors):
        flag = "   <-- bad spot? rerun with --drop " + str(i) if e > 2.5 * np.median(errors) and e > 0.01 else ""
        print(f"  spot {i:2d}: {1000 * e:6.1f} mm{flag}")
    rms = float(np.sqrt(np.mean(errors**2)))
    print(f"  average (RMS): {1000 * rms:.1f} mm   worst: {1000 * errors.max():.1f} mm")
    verdict = ("good" if rms < 0.008 else "usable, but try more spread-out spots"
               if rms < 0.015 else "poor; check the ball is firmly attached and redo")
    print(f"  -> {verdict}")

    camera_in_base = t
    print(f"\nCamera position in robot base frame: "
          f"x={t[0]:+.3f}  y={t[1]:+.3f}  z={t[2]:+.3f} m "
          f"(distance {np.linalg.norm(t):.3f} m from base)")
    print(f"Ball offset on the end-effector frame: "
          f"({offset[0]:+.3f}, {offset[1]:+.3f}, {offset[2]:+.3f}) m"
          f"{' (given)' if args.ball_offset is not None else ' (solved)'}")

    result = {
        "description": "p_base = R @ p_camera + t (meters). Velocity: v_base = R @ v_camera.",
        "camera_frame": "RealSense D435 color optical frame (x right, y down, z forward)",
        "base_frame": "Franka base (panda_link0)",
        "R": r.tolist(),
        "t": camera_in_base.tolist(),
        "ball_offset_in_ee_m": offset.tolist(),
        "ball_offset_was_solved": args.ball_offset is None,
        "rms_error_m": rms,
        "max_error_m": float(errors.max()),
        "num_spots": len(keep),
        "dropped_spots": sorted(set(args.drop or [])),
        "wrist_rotation_spread_deg": spread,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_json(args.out, result)
    print(f"\nSaved calibration to {args.out}")
    return True


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def wait_for_line(rs, cv2, pipeline, align, args, prompt):
    """input() that keeps the preview window alive on Linux."""
    print(prompt, end="", flush=True)
    if not args.preview:
        return sys.stdin.readline().strip()
    import select
    while True:
        live_preview(rs, cv2, pipeline, align, args)
        ready, _, _ = select.select([sys.stdin], [], [], 0.03)
        if ready:
            return sys.stdin.readline().strip()


def collect(args):
    import cv2
    import pyrealsense2 as rs
    from check_camera import camera_session, print_stream_profiles

    robot = {"ros": lambda: RosPose(args.ros_topic),
             "franky": lambda: FrankyPose(args.robot_ip),
             "manual": ManualPose}[args.robot]()

    samples = []
    if args.samples.exists() and args.resume:
        samples = json.loads(args.samples.read_text())["samples"]
        print(f"Resumed {len(samples)} saved spots from {args.samples}")

    with camera_session(rs, fps=30) as (pipeline, profile):
        print_stream_profiles(rs, profile)
        align = rs.align(rs.stream.color)
        print(__doc__.split("Afterwards")[0])
        while True:
            cmd = wait_for_line(
                rs, cv2, pipeline, align, args,
                f"\n[{len(samples)} spots] Enter = record spot | 'undo' | 'solve' | 'quit': ",
            ).lower()
            if cmd in ("q", "quit", "exit"):
                break
            if cmd == "undo":
                if samples:
                    samples.pop()
                    print("  removed last spot")
                continue
            if cmd == "solve":
                report_and_save(samples, args)
                continue
            if cmd:
                print("  unknown command")
                continue

            point, note = average_ball(rs, cv2, pipeline, align, args)
            if point is None:
                print(f"  NOT recorded: {note}")
                continue
            try:
                pose = robot.read()
            except Exception as exc:  # report and let the user retry
                print(f"  NOT recorded: could not read robot pose ({exc})")
                continue
            samples.append({"camera_xyz_m": point.tolist(), "ee_pose_base": pose.tolist()})
            save_json(args.samples, {"samples": samples})
            print(f"  spot {len(samples)} recorded | camera XYZ=({point[0]:+.3f}, "
                  f"{point[1]:+.3f}, {point[2]:.3f}) m | robot EE=({pose[0, 3]:+.3f}, "
                  f"{pose[1, 3]:+.3f}, {pose[2, 3]:+.3f}) m | {note}")
    if args.preview:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", choices=("ros", "franky", "manual"), default="manual",
                        help="How to read the robot pose (default manual)")
    parser.add_argument("--robot-ip", default="172.16.0.2", help="Robot IP for --robot franky")
    parser.add_argument("--ros-topic", default="/franka_state_controller/franka_states",
                        help="franka_msgs/FrankaState topic for --robot ros")
    parser.add_argument("--ball-offset", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="Measured ball center in the end-effector frame (m). "
                             "If omitted, it is solved from wrist rotations.")
    parser.add_argument("--preview", action="store_true", help="Show the camera view")
    parser.add_argument("--brightness", type=int, default=170,
                        help="Minimum white-pixel brightness (0-255, default 170)")
    parser.add_argument("--average-s", type=float, default=1.0,
                        help="Seconds of frames averaged per spot (default 1)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Result JSON path")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES,
                        help="Where recorded spots are saved after each spot")
    parser.add_argument("--resume", action="store_true",
                        help="Continue adding to spots saved in --samples")
    parser.add_argument("--refit", type=Path, metavar="SAMPLES_JSON",
                        help="Solve from saved spots without the camera or robot")
    parser.add_argument("--drop", type=int, nargs="+", metavar="N",
                        help="Spot numbers to leave out when solving")
    args = parser.parse_args()

    if args.refit:
        samples = json.loads(args.refit.read_text())["samples"]
        return 0 if report_and_save(samples, args) else 1
    if args.preview and sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        parser.error("--preview requires a graphical desktop session")
    try:
        collect(args)
    except KeyboardInterrupt:
        print("\nStopped. Recorded spots are saved in", args.samples)
    except RuntimeError as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
