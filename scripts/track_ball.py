"""Track a moving white table-tennis ball in the D435 camera frame."""

import argparse
from collections import deque
from dataclasses import dataclass
import math
import os
import sys
import time

from check_camera import camera_session, print_stream_profiles, rgb_depth_skew_ms


BALL_RADIUS_M = 0.020
WINDOW = "D435 white-ball tracker"
MOTION_WINDOW_MS = 150
MAX_SAMPLE_GAP_MS = 75
MIN_VELOCITY_SAMPLES = 4
MIN_VELOCITY_SPAN_MS = 45
MAX_FIT_ERROR_M = 0.030
LOOKAHEAD_MS = 50


@dataclass(frozen=True)
class Detection:
    """Center and apparent radius of the 2D ball candidate, in pixels."""

    u: int
    v: int
    radius: float


class MotionHistory:
    """Fit camera-frame velocity to recent 3D detections from the current track."""

    def __init__(self):
        self.samples = deque()

    def update(self, np, timestamp_ms, position):
        """Return a velocity in m/s when the latest positions support a fit."""
        if not math.isfinite(timestamp_ms):
            self.samples.clear()
            return None

        if self.samples:
            elapsed_ms = timestamp_ms - self.samples[-1][0]
            if elapsed_ms < 0 or elapsed_ms > MAX_SAMPLE_GAP_MS:
                # Start a new fit after a clock reset or a tracking gap.
                self.samples.clear()
            elif elapsed_ms == 0:
                return None

        if position is None or not np.isfinite(position).all():
            return None
        self.samples.append((timestamp_ms, position))
        while timestamp_ms - self.samples[0][0] > MOTION_WINDOW_MS:
            self.samples.popleft()
        if (
            len(self.samples) < MIN_VELOCITY_SAMPLES
            or timestamp_ms - self.samples[0][0] < MIN_VELOCITY_SPAN_MS
        ):
            return None

        # Use capture times in seconds relative to the latest frame. Taking the
        # slope of a fit across several positions smooths noisy depth readings.
        elapsed_s = np.array([(t - timestamp_ms) / 1000 for t, _ in self.samples])
        positions = np.stack([point for _, point in self.samples])
        design = np.column_stack((elapsed_s, np.ones_like(elapsed_s)))
        coefficients = np.linalg.lstsq(design, positions, rcond=None)[0]
        errors = np.linalg.norm(positions - design @ coefficients, axis=1)
        if errors.max() > MAX_FIT_ERROR_M:
            # A bounce or a false detection can break the straight-line fit.
            self.samples.clear()
            self.samples.append((timestamp_ms, position))
            return None
        return coefficients[0]


def detect_ball(cv2, np, image, gray, previous_gray, brightness):
    """Find the strongest moving, bright, roughly circular component."""
    if previous_gray is None:
        return None

    # White has low saturation and high brightness, regardless of hue.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, brightness), (179, 75, 255))
    # Pixels that got brighter help distinguish the arriving ball from
    # stationary white objects and the dark patch it leaves behind.
    brighter = cv2.subtract(gray, previous_gray) >= 18
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_score = 0.0
    best = None
    for contour in contours:
        (u, v), radius = cv2.minEnclosingCircle(contour)
        if not 3 <= radius <= 42:
            continue
        perimeter = cv2.arcLength(contour, True)
        # Circularity is near 1 for a disk and drops for elongated shapes.
        circularity = 4 * math.pi * cv2.contourArea(contour) / perimeter**2
        if circularity < 0.55:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, cv2.FILLED, offset=(-x, -y))
        # Measure the fraction of the candidate that got brighter.
        motion = brighter[y:y + height, x:x + width][mask != 0].mean()
        score = circularity * motion
        if motion >= 0.12 and score > best_score:
            best_score = score
            best = Detection(round(u), round(v), radius)
    return best


def estimate_position(rs, np, depth, ball):
    """Return the ball center or a reason the depth estimate was rejected."""
    intrinsics = depth.profile.as_video_stream_profile().intrinsics
    # Stay near the middle of the detection to avoid background pixels at its edge.
    half = max(1, int(0.3 * ball.radius))
    values = [
        depth.get_distance(u, v)
        for v in range(max(0, ball.v - half), min(intrinsics.height, ball.v + half + 1))
        for u in range(max(0, ball.u - half), min(intrinsics.width, ball.u + half + 1))
    ]
    valid = np.asarray([z for z in values if math.isfinite(z) and z > 0])
    required = max(3, math.ceil(0.5 * len(values)))
    if len(valid) < required:
        return None, f"depth pixels {len(valid)}/{len(values)}"

    surface_depth = float(np.median(valid))
    if not 0.3 <= surface_depth <= 3.0:
        return None, f"depth {surface_depth:.2f} m out of range"
    # Reject patches with competing depth readings, even if most pixels are nonzero.
    consistent = np.count_nonzero(np.abs(valid - surface_depth) <= 0.03)
    if consistent < max(required, math.ceil(0.7 * len(valid))):
        return None, "mixed foreground and background depth"

    point = np.asarray(
        rs.rs2_deproject_pixel_to_point(intrinsics, [ball.u, ball.v], surface_depth)
    )
    # Near the center pixel, the visible surface sits about one radius
    # closer to the imager than the center, measured along the viewing ray.
    center = point + BALL_RADIUS_M * point / np.linalg.norm(point)
    # A 40 mm ball should have a plausible image size at this distance.
    expected_radius = intrinsics.fx * BALL_RADIUS_M / center[2]
    if not 0.6 * expected_radius <= ball.radius <= 1.65 * expected_radius:
        return None, "apparent size disagrees with depth"
    # Optical axes are right, down, forward. The Franka base frame needs calibration.
    return center, None


def describe_detection(rs, np, align, frames, color, depth, ball, max_skew_ms):
    """Return numeric camera-frame XYZ and a message for the preview."""
    if ball is None:
        return None, "ball=not detected"

    # Alignment fixes pixel coordinates only. The moving ball must also have
    # been at nearly the same position at both capture times.
    pixel = f"pixel=({ball.u}, {ball.v})"
    skew_ms = rgb_depth_skew_ms(rs, color, depth)
    if skew_ms is None:
        return None, f"{pixel} depth=invalid | RGB-depth timing unavailable | skew=unavailable"
    timing = f"skew={skew_ms:.1f} ms"
    if skew_ms > max_skew_ms:
        return None, f"{pixel} depth=invalid | RGB-depth skew exceeds limit | {timing}"

    aligned_frames = align.process(frames)
    aligned_depth = aligned_frames.get_depth_frame()
    if not aligned_depth:
        return None, f"{pixel} depth=invalid | aligned depth unavailable | {timing}"
    position, rejection = estimate_position(rs, np, aligned_depth, ball)
    if position is None:
        return None, f"{pixel} depth=invalid | {rejection} | {timing}"

    x, y, z = position
    return position, f"{pixel} XYZ=({x:+.3f}, {y:+.3f}, {z:.3f}) m | {timing}"


def show_detection(cv2, image, ball, valid, status):
    """Draw on a copy of the RGB frame and tell the capture loop whether to continue."""
    preview = image.copy()
    if ball is not None:
        # Green means a valid 3D estimate. Yellow means we only have a 2D candidate.
        tint = (70, 240, 80) if valid else (0, 210, 255)
        cv2.circle(preview, (ball.u, ball.v), round(ball.radius), tint, 2)
        cv2.circle(preview, (ball.u, ball.v), 2, tint, -1)
    cv2.rectangle(preview, (0, 0), (preview.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        preview, status, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.imshow(WINDOW, preview)
    key = cv2.waitKey(1) & 0xFF
    return key not in (ord("q"), 27) and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1


def track(rs, cv2, np, pipeline, args):
    """Process fresh frame pairs and display the detection and its measured depth."""
    align = rs.align(rs.stream.color)
    previous_gray = None
    motion_history = MotionHistory()
    started = last_pair = time.monotonic()
    first_pair = None
    pair_count = 0
    previous_color_number = previous_depth_number = None
    last_report = -math.inf
    timeout_ms = 6000
    while True:
        frames = pipeline.wait_for_frames(timeout_ms)
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        now = time.monotonic()
        # The SDK may reuse a frame when pairing the streams. Count and
        # process each RGB and depth frame only once.
        color_number = color.get_frame_number() if color else None
        depth_number = depth.get_frame_number() if depth else None
        finished = args.seconds is not None and now - started >= args.seconds
        if not (color and depth) or (
            color_number == previous_color_number or depth_number == previous_depth_number
        ):
            if now - last_pair >= timeout_ms / 1000:
                raise RuntimeError("No fresh paired RGB and depth frames arrived.")
            if pair_count and finished:
                break
            continue
        previous_color_number, previous_depth_number = color_number, depth_number
        if first_pair is None:
            first_pair = now
        last_pair = now
        pair_count += 1
        timeout_ms = 2000

        image = np.asanyarray(color.get_data())
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ball = detect_ball(cv2, np, image, gray, previous_gray, args.brightness)
        previous_gray = gray
        timestamp_ms = color.get_timestamp()
        position, status = describe_detection(
            rs, np, align, frames, color, depth, ball, args.max_skew_ms
        )
        velocity = motion_history.update(np, timestamp_ms, position)
        motion = "velocity=unavailable"
        if velocity is not None:
            vx, vy, vz = velocity
            # Short look-ahead only. The ballistic catch predictor needs a
            # calibrated gravity direction and box opening in the Franka frame.
            fx, fy, fz = position + velocity * (LOOKAHEAD_MS / 1000)
            motion = (
                f"velocity=({vx:+.2f}, {vy:+.2f}, {vz:+.2f}) m/s | "
                f"linear XYZ +{LOOKAHEAD_MS} ms=({fx:+.3f}, {fy:+.3f}, {fz:.3f}) m"
            )

        # The detector runs on every fresh pair. This only limits terminal output.
        if args.every_frame or now - last_report >= 0.1 or finished:
            message = f"t={timestamp_ms:.1f} ms | valid={position is not None} | {status}"
            if position is not None:
                message += f" | {motion}"
            print(message, flush=True)
            last_report = now

        if args.preview and not show_detection(
            cv2, image, ball, position is not None, status
        ):
            break
        if finished:
            break

    if pair_count > 1:
        rate = (pair_count - 1) / (last_pair - first_pair)
        print(f"Processed  {pair_count} fresh RGB-D pairs | average {rate:.1f} pairs/s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="Show the RGB detection overlay")
    parser.add_argument("--fps", type=int, choices=(30, 60), default=60,
                        help="RGB and depth stream frame rate (default 60)")
    parser.add_argument("--seconds", type=float, help="Stop after this many seconds")
    parser.add_argument("--every-frame", action="store_true",
                        help="Print each RGB-depth pair instead of about 10 lines per second")
    parser.add_argument("--brightness", type=int, default=170,
                        help="Minimum white-pixel brightness (0–255, default 170)")
    parser.add_argument("--max-skew-ms", type=float, default=12.0,
                        help="Maximum RGB-depth timestamp difference (default 12 ms)")
    args = parser.parse_args()
    if not 0 <= args.brightness <= 255:
        parser.error("--brightness must be between 0 and 255")
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be positive and finite")
    if not math.isfinite(args.max_skew_ms) or args.max_skew_ms <= 0:
        parser.error("--max-skew-ms must be positive and finite")
    if args.preview and sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        parser.error("--preview requires a graphical desktop session")

    try:
        import cv2
        import numpy as np
        import pyrealsense2 as rs
    except ImportError as exc:
        print(f"Camera dependency unavailable ({exc}). Install requirements-camera.txt.",
              file=sys.stderr)
        return 1

    try:
        with camera_session(rs, fps=args.fps) as (pipeline, profile):
            print_stream_profiles(rs, profile)
            track(rs, cv2, np, pipeline, args)
    except KeyboardInterrupt:
        print("\nBall tracker stopped.")
    except (RuntimeError, cv2.error) as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        if args.fps == 60:
            print("If 60 fps is unavailable on this connection, try --fps 30.", file=sys.stderr)
        return 1
    finally:
        if args.preview:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
