"""Lock on to a white or orange (--ball orange) table-tennis ball, then follow it as it moves, and report its distance.

How it works:
  1. LEARN    It learns the empty scene for a moment (keep the ball out of view at
              start), so white/orange things already in the room (robot arm, lights,
              walls) are ignored.
  2. SEARCH   It looks for whitish pixels that changed from the empty scene,
              grouped into blobs that are round and solid (a filled disk).
  3. LOCKING  When the same ball-like blob is seen in several frames in a row, it
              counts up ("LOCKING 5/8"). You can also CLICK the ball to lock at once.
  4. LOCKED   A green "LOCKED ON" banner confirms the lock. From then on it only
              looks where the ball is expected next, so other white things can't
              steal the lock. The ball may get bigger (closer) or smaller (farther):
              a 40 mm ball's apparent size is what tells its distance when depth
              is missing, and each frame may change size only gradually.
  5. IN BOX   The black catch box is found automatically every frame (a solid dark
              blob whose real size, from depth, is about the box's 15 x 10 cm, and
              which carries the dark-blue TAPE STRIPES stuck on the box), so it can move
              with the arm and other black things are ignored. --no-box-tape drops the
              tape requirement. Each locked ball gets a number (#1, #2, ...). If
              it disappears at the box - inside
              the outline and within 10 cm of the box's front wall or beyond it - it
              is counted as IN THE BOX and the tracker goes back to searching for the
              next ball. A ball that then appears from inside the box is tracked as
              the same ball again (old number). Press c when you empty the box.
              The front-wall distance is measured by depth when you mark the box, or
              set it with --box-distance.
  6. LOST     If the ball disappears anywhere else for LOST_SECONDS it says LOST
              and searches again, first where it vanished (quick 2-frame re-lock).

Distance comes from the depth camera when it can measure the ball (accurate, about
1 cm at 1 m), otherwise from the ball's apparent size ("size": rougher, several
percent). The D435 has no depth closer than ~30 cm. Set --max-distance to ignore
anything farther than that many meters (default: no limit).

Distances are from the camera's COLOR lens to the CENTER of the ball, in cm:
  X = sideways   (+ right, as seen from behind the camera looking out)
  Y = up/down    (+ DOWN, - up)
  Z = straight out from the camera face (not the diagonal)

Mouse / keys (click the preview window first):
  click   lock on to the ball you clicked
  v       start/stop RECORDING (same as the RECORD button): video, depth and tracking
          for scripts/recording_to_sim.py
  p       capture 10 frames (same as clicking the CAPTURE button) into --capture-dir
  k       draw the catch box by hand instead (drag a rectangle; turns off auto-finding)
  c       the box was emptied: clear the in-box count
  r       release the lock and search again
  s       save the current steady (1-second average) reading as the next point
  b       relearn the empty scene (take the ball out of view first)
  q, Esc  quit

Files:
  --output (ball_measurements.csv)  saved points, with empty tape_*_cm columns
  --log    (ball_track_log.csv)     every locked frame: time, X, Y, Z, source, speed

Example:
  python -B scripts/measure_ball.py --fps 30
  python -B scripts/measure_ball.py --fps 30 --ball orange
"""

import argparse
from collections import deque
import csv
from dataclasses import dataclass
import json
import math
import os
import sys
import time

from check_camera import camera_session, print_stream_profiles
from camera_controls import rgb_sensor_settings, color_frame_metadata
from track_ball import BALL_RADIUS_M, Detection


WINDOW = "D435 ball lock-on tracker"
SCALE = 1.5  # preview enlargement
LEARN_SECONDS = 1.5
LOCK_FRAMES = 5          # consistent frames needed to lock automatically
LOCKING_MISS_SECONDS = 0.5   # a brief miss (hand in the way) doesn't restart locking
FRESH_LOCK_SECONDS = 0.4  # right after locking, allow big size changes (ball emerging)
REACQUIRE_FRAMES = 2     # frames to re-lock a ball reappearing where it was lost / at the box
REACQUIRE_SECONDS = 5.0  # how long after a loss the quick re-lock applies
JUST_OUT_SECONDS = 0.5       # right after a ball comes out, don't count it going back in
RISING_PX_PER_S = 120        # moving up the picture faster than this = being lifted out
IN_BOX_GRACE_SECONDS = 0.1   # locked ball unseen this long inside the box area -> IN BOX
BOX_FRONT_MARGIN_M = 0.04    # "visibly in the box": center at most this much before the front wall
BOX_NEAR_MARGIN_M = 0.10     # "vanished at the box": not nearer than front wall - this - 5%
BOX_BACK_MARGIN_M = 0.30     # "visibly in the box": at most this much beyond the back wall
# Measured: a ball dropped into the box (box ~1.3 m by its width, no depth on black)
# read 1.6-2.5 m in its last frames: half-hidden by the rim it looks small (= far).
RECENT_Z_SECONDS = 0.5       # recent ball distances kept for the "in front of box" test
FALLING_PX_PER_S = 60        # moving down the picture faster than this = falling/dropped
FALL_REACH_SECONDS = 0.15    # a falling ball may vanish this far ahead of where last seen
# The printed catch box (assets/armbox-thin-meters.STL on main): cup about 15 x 15 cm,
# 10 cm tall, opening 13.5 x 13.5 cm, plus a 4.5 cm handle (19.5 cm overall).
BOX_WIDTH_M = 0.15                  # cup width, used when the box has no depth
BOX_WIDTH_RANGE_M = (0.09, 0.28)    # seen width allowed, any viewing angle
BOX_HEIGHT_RANGE_M = (0.05, 0.25)   # seen height allowed
BOX_DARK_VALUE = 60                 # "black": brightness below this (0-255)
BOX_KEEP_SECONDS = 1.0              # keep using the last box this long if it isn't found
# Blue tape stuck on the box. Only a black blob carrying this tape counts as the box.
# The box's dark navy tape measured RGB ~(12, 58, 112) = HSV hue 104-108, sat 200-244,
# value 75-130. The lab's blue-green frames are hue ~99, sat ~125: they fail sat_min.
# The bright blue track on the table and the blue part on another robot are much
# brighter (value ~190-210): they fail value_max.
TAPE = {"hue_min": 100, "hue_max": 115, "sat_min": 150, "value_min": 50, "value_max": 165}
TAPE_MIN_PIXELS = 12      # smallest stripe, pixels (a stripe at 2 m is ~50 x 6 px)
TAPE_MIN_STRIPES = 2      # stripes needed (the box has 3; allow one to be hidden or merged)
LOST_SECONDS = 0.7       # locked ball unseen this long -> LOST, search again
HOLD_SECONDS = 4.0       # keep the last reading on screen this long after losing the ball
AVERAGE_SECONDS = 1.0
SPEED_SECONDS = 0.3
MOVED_M = 0.03
STEADY_SPREAD_MM = 5.0
MIN_STEADY_SAMPLES = 10
MAX_BALL_SPEED_MPS = 4.0
MAX_CANDIDATES = 8
EDGE_MIN_RADIUS_PX = 30   # a blob touching the picture edge must be at least this big
MIN_RADIUS_PX = 3   # a 40 mm ball ~4 m away
MAX_RADIUS_PX = 450  # a 40 mm ball ~3 cm from the lens
MIN_CIRCULARITY = 0.8
MIN_FILL = 0.65
MIN_SOLIDITY = 0.6
MIN_DEPTH_M = 0.3
NEUTRAL_SATURATION = 30   # below this a pixel counts as gray/white whatever its hue
WARM_HUE_MAX = 45         # OpenCV hue 0-179: 0-45 red/orange/yellow (cream ball ~20)
WARM_HUE_MIN_WRAP = 165   # 165-179 is red again; 46-164 (green, cyan, blue) is never ball
# Ball color settings, filled in from the command line (--ball white|orange).
BALL = {"color": "white", "hue_min": 6, "hue_max": 25, "sat_min": 235, "value_min": 100,
        "loose_sat_min": 160, "loose_value_min": 150}
# The orange ball's colorfulness depends on the lighting and the camera's auto exposure:
# 240-255 under the warm lights close up, but only ~180-230 at 1-2 m against the white
# wall. So orange is tested twice: strict (sat >= 235: keeps the ball apart from skin
# under warm light, where skin reaches ~215) and loose (sat >= 160 and bright, value
# >= 150: skin measured ~170 sat but only ~100-130 bright). Round blobs from either count.
CSV_FIELDS = ["point", "time", "source", "cam_x_cm", "cam_y_cm", "cam_z_cm", "distance_cm",
              "spread_mm", "samples", "tape_x_cm", "tape_y_cm", "tape_z_cm"]
LOG_FIELDS = ["t_s", "state", "ball_id", "x_cm", "y_cm", "z_cm", "distance_cm", "source",
              "radius_px", "u_px", "v_px", "box_px", "box_front_cm",
              "speed_toward_camera_mps", "balls_in_box", "event"]

GREEN, YELLOW, RED, GRAY, WHITE = (70, 240, 80), (0, 210, 255), (60, 60, 255), (150, 150, 150), (255, 255, 255)
BLUE = (255, 170, 40)


@dataclass(frozen=True)
class Candidate:
    """A whitish, round, solid blob in the color image."""

    detection: Detection
    circularity: float


@dataclass(frozen=True)
class Measurement:
    """A ball candidate with a 3D center in meters and where the distance came from."""

    detection: Detection
    position: object  # numpy array (x, y, z) in the color optical frame
    source: str       # "depth" or "size"


def scene_changed(np, image, background, min_change, cv2=None, where=None):
    """Pixels that differ from the learned empty scene (True = changed).

    Corrects for the camera's automatic brightness/white-balance adjusting
    when the ball appears. With cv2 the full-image steps use OpenCV (about 4x
    faster); with `where` (a True/False image) only those pixels are compared and
    the rest are left False - much faster when only ball-colored pixels matter.
    """
    # Global per-channel gain, estimated from every 4th pixel each way (fast, plenty).
    flat_frame = image[::4, ::4].reshape(-1, 3).astype(np.float32)
    flat_bg = background[::4, ::4].reshape(-1, 3)
    gain = np.clip(np.median(flat_frame, axis=0)
                   / np.maximum(np.median(flat_bg, axis=0), 1.0), 0.5, 2.0)
    # Refine using only pixels that still look like the empty scene, so a
    # ball filling much of the picture does not skew the correction.
    same = np.abs(flat_frame - flat_bg * gain).max(axis=1) < min_change
    if same.mean() > 0.2:
        gain = np.clip(np.median(flat_frame[same], axis=0)
                       / np.maximum(np.median(flat_bg[same], axis=0), 1.0), 0.5, 2.0)
    if where is not None:
        changed = np.zeros(image.shape[:2], bool)
        rows, cols = np.nonzero(where)
        diff = np.abs(image[rows, cols].astype(np.float32) - background[rows, cols] * gain)
        changed[rows, cols] = diff.max(axis=1) >= min_change
        return changed
    if cv2 is None:
        frame = image.astype(np.float32)
        return np.abs(frame - background * gain).max(axis=2) >= min_change
    scaled = cv2.multiply(background, tuple(float(g) for g in gain) + (0.0,))
    blue, green, red = cv2.split(cv2.absdiff(image.astype(np.float32), scaled))
    return cv2.max(cv2.max(blue, green), red) >= min_change


def ball_mask(cv2, np, image, background, brightness, max_saturation, min_change, loose=False,
              changed=None):
    """Ball-colored pixels that differ from the learned empty scene, with gaps filled.

    Indoors a white ball looks light gray or warm cream (hue ~20, some
    saturation), about as bright as walls and light-blue tables. So:
      - color: neutral, or warm (red/orange/yellow) - never blue/green/cyan;
      - change: must differ from the empty scene, after correcting for the
        camera's automatic brightness/white-balance adjusting to the ball.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if BALL["color"] == "orange":
        # Orange: a table-tennis ball is almost fully saturated (colorfulness ~240-255
        # measured in the lab), while warm-lit white walls/pipes reach ~150-200 and
        # skin (hands, faces) up to ~215.
        # Hue 6-25 keeps orange and drops pure red (red boxes, the red stick on the arm).
        sat_min = BALL["loose_sat_min"] if loose else BALL["sat_min"]
        value_min = BALL["loose_value_min"] if loose else BALL["value_min"]
        mask = cv2.inRange(hsv, (BALL["hue_min"], sat_min, value_min),
                           (BALL["hue_max"], 255, 255))
    else:
        hue, sat = hsv[..., 0], hsv[..., 1]
        mask = cv2.inRange(hsv, (0, 0, brightness), (179, max_saturation, 255))
        cool = (sat >= NEUTRAL_SATURATION) & (hue > WARM_HUE_MAX) & (hue < WARM_HUE_MIN_WRAP)
        mask[cool] = 0
    if changed is None and background is not None:
        changed = scene_changed(np, image, background, min_change)
    if changed is not None:
        mask[~changed] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # Closing fills the shadowed rim and highlight so the blob is a full disk.
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def fit_circle(np, points):
    """Least-squares circle through points; return (u, v, radius, rms error) or None."""
    x, y = points[:, 0].astype(float), points[:, 1].astype(float)
    a = np.column_stack((x, y, np.ones_like(x)))
    b = x**2 + y**2
    (c0, c1, c2), *_ = np.linalg.lstsq(a, b, rcond=None)
    u, v = c0 / 2, c1 / 2
    r2 = c2 + u**2 + v**2
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    rms = float(np.sqrt(np.mean((np.hypot(x - u, y - v) - r) ** 2)))
    return u, v, r, rms


def robust_circle(np, points, iterations=150, seed=0):
    """RANSAC circle: the circle most edge points agree with; stray straight edges are ignored.

    Returns (u, v, radius, rms, inliers) or None, where inliers are the points on it.
    """
    if len(points) < 20:
        return None
    rng = np.random.default_rng(seed)
    pts = points.astype(float)
    # Score candidate circles on a sample of points (fast); refit on all inliers after.
    sample = pts if len(pts) <= 300 else pts[rng.choice(len(pts), 300, replace=False)]
    best, best_count = None, 0
    for _ in range(iterations):
        fit = fit_circle(np, sample[rng.choice(len(sample), 3, replace=False)])
        if fit is None or not MIN_RADIUS_PX <= fit[2] <= MAX_RADIUS_PX:
            continue
        u, v, r, _ = fit
        count = np.count_nonzero(np.abs(np.hypot(sample[:, 0] - u, sample[:, 1] - v) - r)
                                 <= max(2.0, 0.02 * r))
        if count > best_count:
            best, best_count = fit, count
    if best is None:
        return None
    u, v, r, _ = best
    inliers = pts[np.abs(np.hypot(pts[:, 0] - u, pts[:, 1] - v) - r) <= max(2.5, 0.03 * r)]
    if len(inliers) < 20:
        return None
    u, v, r, rms = fit_circle(np, inliers)
    return u, v, r, rms, inliers


def circle_from_edge(np, contour, width, height):
    """Fit the ball's circle to a blob's visible edge (ignoring the picture border).

    Works for a ball cut off by the frame edge, or one with bits of background
    stuck to it. Returns a Detection or None.
    """
    points = contour.reshape(-1, 2)
    inside = ((points[:, 0] > 2) & (points[:, 0] < width - 3)
              & (points[:, 1] > 2) & (points[:, 1] < height - 3))
    arc = points[inside]
    fit = robust_circle(np, arc)
    if fit is None:
        return None
    u, v, r, rms, inliers = fit
    # The center must be in the picture (so depth/size can be read there).
    if rms > max(2.0, 0.03 * r) or not (0 <= u < width and 0 <= v < height):
        return None
    if not MIN_RADIUS_PX <= r <= MAX_RADIUS_PX:
        return None
    # Points on the circle must cover a good part of it (at least ~150 degrees),
    # with enough of them for that length of arc.
    angles = np.arctan2(inliers[:, 1] - v, inliers[:, 0] - u)
    covered = np.count_nonzero(np.histogram(angles, bins=24, range=(-math.pi, math.pi))[0])
    if covered < 10 or len(inliers) < 0.6 * (covered / 24) * 2 * math.pi * r:
        return None
    # The part of the circle inside the picture must be mostly filled by the blob.
    import cv2
    x0, y0 = max(0, int(u - r)), max(0, int(v - r))
    x1, y1 = min(width, int(u + r) + 1), min(height, int(v + r) + 1)
    disk = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.circle(disk, (int(round(u)) - x0, int(round(v)) - y0), int(round(r)), 1, -1)
    blob = np.zeros_like(disk)
    cv2.drawContours(blob, [contour], -1, 1, cv2.FILLED, offset=(-x0, -y0))
    if disk.sum() == 0 or (disk & blob).sum() < 0.7 * disk.sum():
        return None
    return Detection(int(round(u)), int(round(v)), float(r))


def edge_cut_candidate(np, contour, width, height):
    """A ball so close it runs off the picture."""
    points = contour.reshape(-1, 2)
    inside = ((points[:, 0] > 2) & (points[:, 0] < width - 3)
              & (points[:, 1] > 2) & (points[:, 1] < height - 3))
    if inside.all():
        return None
    detection = circle_from_edge(np, contour, width, height)
    return None if detection is None else Candidate(detection, 0.9)


def find_candidates(cv2, np, mask):
    """Round, solid blobs in the ball mask (including a big ball cut off by the frame edge)."""
    height, width = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    found = []
    for contour in contours:
        if cv2.contourArea(contour) < math.pi * MIN_RADIUS_PX**2:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if x <= 2 or y <= 2 or x + w >= width - 2 or y + h >= height - 2:
            # Only a big, close ball is plausibly cut off by the edge; small
            # edge blobs are background objects peeking into the picture.
            if max(w, h) / 2 < EDGE_MIN_RADIUS_PX:
                continue
            partial = edge_cut_candidate(np, contour, width, height)
            if partial is not None:
                found.append(partial)
                continue
        (u, v), radius = cv2.minEnclosingCircle(contour)
        if not MIN_RADIUS_PX <= radius <= MAX_RADIUS_PX:
            continue
        # Judge shape on the outline (convex hull) so a gap where the ball's
        # color matches the background does not disqualify it.
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        perimeter = cv2.arcLength(hull, True)
        if perimeter == 0 or hull_area == 0:
            continue
        circularity = 4 * math.pi * hull_area / perimeter**2
        # Fill: how much of the enclosing circle the outline covers. A ball is a
        # disk; arm links, squares and strips leave much of the circle empty.
        fill = hull_area / (math.pi * radius**2)
        solidity = cv2.contourArea(contour) / hull_area
        if circularity < MIN_CIRCULARITY or fill < MIN_FILL or solidity < MIN_SOLIDITY:
            continue
        found.append(Candidate(Detection(round(u), round(v), radius), circularity))
    return found


def merge_candidates(first, second):
    """Join two candidate lists; where both found the same ball, keep the bigger circle.

    The strict orange test often catches only the ball's brightest middle (too small
    a circle = too far by size); the loose test then has the whole ball.
    """
    merged = list(first)
    for c in second:
        d = c.detection
        same = [i for i, o in enumerate(merged)
                if math.hypot(d.u - o.detection.u, d.v - o.detection.v)
                <= max(d.radius, o.detection.radius)]
        if not same:
            merged.append(c)
        elif all(d.radius > merged[i].detection.radius for i in same):
            for i in reversed(same):
                merged.pop(i)
            merged.append(c)
    return merged


def size_distance_m(fx, radius_px):
    """Distance to the ball center from its apparent radius (pinhole camera)."""
    return fx * BALL_RADIUS_M / radius_px


def locate(rs, np, depth, intrinsics, detection, max_distance_m):
    """Return (Measurement, None) or (None, reason) for one candidate."""
    half = max(1, int(0.3 * detection.radius))
    u0, v0 = detection.u, detection.v
    values = []
    if depth is not None:
        values = [
            depth.get_distance(u, v)
            for v in range(max(0, v0 - half), min(intrinsics.height, v0 + half + 1))
            for u in range(max(0, u0 - half), min(intrinsics.width, u0 + half + 1))
        ]
    valid = np.asarray([z for z in values if math.isfinite(z) and z > 0])
    size_z = size_distance_m(intrinsics.fx, detection.radius)
    # Use depth only when most of the patch has agreeing, in-range readings.
    surface = float(np.median(valid)) if valid.size else 0.0
    usable = (
        valid.size >= max(3, math.ceil(0.5 * len(values)))
        and surface >= MIN_DEPTH_M
        and np.count_nonzero(np.abs(valid - surface) <= 0.03) >= 0.7 * valid.size
    )
    if usable:
        point = np.asarray(rs.rs2_deproject_pixel_to_point(intrinsics, [u0, v0], surface))
        center = point + BALL_RADIUS_M * point / np.linalg.norm(point)
        if 0.6 * size_z <= center[2] <= 1.65 * size_z:
            if center[2] > max_distance_m:
                return None, f"too far ({center[2]:.2f} m by depth)"
            return Measurement(detection, center, "depth"), None
        # The blob is the wrong size for that depth. For a white ball that usually
        # means it isn't a ball. The orange color test is so specific that it's
        # more likely the depth patch caught the background behind a small ball,
        # so fall back to the size estimate below.
        if BALL["color"] != "orange":
            return None, f"wrong size for its depth ({center[2]:.2f} m)"

    # No usable depth: the ball may be closer than the D435's minimum range.
    if size_z > max_distance_m:
        return None, f"too far ({size_z:.2f} m by size), no depth"
    if (BALL["color"] == "white" and size_z >= MIN_DEPTH_M + 0.1 and valid.size == 0
            and depth is not None):
        # At this distance the camera should see depth; an all-empty white patch
        # is more likely a light, window or reflection than a ball (lights are
        # never orange, so this check is for the white ball only).
        return None, f"no depth at {size_z:.2f} m (light or reflection?)"
    ray = np.asarray(rs.rs2_deproject_pixel_to_point(intrinsics, [u0, v0], 1.0))
    return Measurement(detection, ray * (size_z / ray[2]), "size"), None


def _only(np, mask, contour):
    """A mask containing just one contour's filled area."""
    import cv2
    single = np.zeros_like(mask)
    cv2.drawContours(single, [contour], -1, 255, cv2.FILLED)
    return single


def ball_at(cv2, np, image, u, v, brightness, max_saturation, radius_hint=None,
            scene_mask=None):
    """Find the ball around pixel (u, v) using color and shape only (no empty-scene test).

    Used when you click the ball, and to recover a locked ball that blended into
    the background. Tries, in order: the ball-colored blob under the point (or a
    circle fitted to its curved edge if it runs off the picture), then a circle
    search in the gray image around the point.
    """
    height, width = image.shape[:2]
    u, v = int(round(min(max(u, 0), width - 1))), int(round(min(max(v, 0), height - 1)))
    # First the "changed from the empty scene" mask (cleanest), then color alone.
    masks = [] if scene_mask is None else [scene_mask]
    masks.append(ball_mask(cv2, np, image, None, brightness, max_saturation, 0))
    if BALL["color"] == "orange":
        masks.append(ball_mask(cv2, np, image, None, brightness, max_saturation, 0, loose=True))
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if cv2.pointPolygonTest(contour, (u, v), True) < -5:
                continue
            # A clean round blob: use it as is.
            for candidate in find_candidates(cv2, np, _only(np, mask, contour)):
                d = candidate.detection
                if math.hypot(d.u - u, d.v - v) <= d.radius:
                    return d
            detection = circle_from_edge(np, contour, width, height)
            if detection is not None and (
                    math.hypot(detection.u - u, detection.v - v) <= detection.radius):
                if radius_hint is None or 0.5 <= detection.radius / radius_hint <= 2.0:
                    return detection
    # Circle search (Hough) in the gray image near the point; the circle found
    # must be mostly ball-colored.
    gray = cv2.medianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 5)
    max_r = int(min(MAX_RADIUS_PX, 2.0 * radius_hint)) if radius_hint else 200
    min_r = int(max(MIN_RADIUS_PX, 0.5 * radius_hint)) if radius_hint else 8
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=20, param1=100,
                               param2=30, minRadius=min_r, maxRadius=max_r)
    if circles is None:
        return None
    color = masks[-1]
    for cu, cv_, r in sorted(circles[0], key=lambda c: math.hypot(c[0] - u, c[1] - v)):
        if math.hypot(cu - u, cv_ - v) > r:
            break
        disk = np.zeros_like(color)
        cv2.circle(disk, (round(float(cu)), round(float(cv_))), round(float(r)), 255, -1)
        if np.count_nonzero(color & disk) >= 0.6 * np.count_nonzero(disk):
            return Detection(round(float(cu)), round(float(cv_)), float(r))
    return None


class BallLock:
    """Search -> locking -> locked; balls that vanish at the box are remembered as in the box.

    Each locked ball gets a number (#1, #2, ...). When a locked ball vanishes near
    the catch box it is added to `in_box_balls` and the tracker goes straight back
    to searching for the next ball. A ball that appears from inside the box is
    taken to be the (most recently boxed, nearest) ball coming back out, and is
    tracked again under its old number.
    """

    def __init__(self):
        self.box = None           # (u0, v0, u1, v1) catch-box area in the image, or None
        self.box_front_m = None   # distance (Z) from the camera to the box's front wall
        self.box_depth_m = 0.15   # how deep the box is, front wall to back wall
        self.in_box_balls = []    # [{"id", "pixel", "position", "time", "box"}], oldest first
        self.junk = []            # (u, v, r) of blobs already visible when the ball went in
        self.taken_out = False    # the would-be ball was seen away from the box
        self.out_time = -math.inf # when a ball last came out of the box
        self.locked_at = -math.inf
        self.next_id = 1
        self.events = []          # messages for the caller to print/show
        self.last_event = None    # (time, text) of the latest box event, for the banner
        self.release()

    def release(self, keep_lost=False):
        self.state = "SEARCHING"
        self.hits = 0
        self.needed = LOCK_FRAMES
        self.ball_id = None
        self.from_box = False     # the would-be ball came out of the box
        if not keep_lost:
            self.lost_pixel = None    # where the ball was last seen before a loss
            self.lost_time = -math.inf
        self.last = None          # last Measurement
        self.last_time = -math.inf
        self.pixel_velocity = (0.0, 0.0)
        self.history = deque()    # (t, position) while locked, for speed
        self.recent = deque()     # (t, z, source) of recent sightings, for the box check
        self.lost_at = None

    # ----- the catch box -----
    def in_box(self, u, v, margin=0.0):
        if self.box is None:
            return False
        u0, v0, u1, v1 = self.box
        return u0 - margin <= u <= u1 + margin and v0 - margin <= v <= v1 + margin

    def _box_distance_ok(self, position, front_margin_m, also=()):
        """Is the ball at the box's distance? `also`: other distance estimates to try."""
        if self.box_front_m is None:
            return True
        return any(self.box_front_m - front_margin_m <= z
                   <= self.box_front_m + self.box_depth_m + BOX_BACK_MARGIN_M
                   for z in (float(position[2]), *also))

    def in_box_3d(self, detection, position, margin_px=0.0):
        """Visibly inside the box: inside its outline AND at the box's distance."""
        return (self.in_box(detection.u, detection.v, margin_px)
                and self._box_distance_ok(position, BOX_FRONT_MARGIN_M))

    def near_box(self, detection, position, now=None):
        """Close enough to the box that vanishing here means it went in.

        A dropped ball falls fast and may vanish just above the box (motion blur),
        so for a falling ball the spot it was heading for counts too.
        """
        margin = detection.radius + 10
        spots = [(detection.u, detection.v)]
        if now is not None and self.pixel_velocity[1] > FALLING_PX_PER_S:
            du, dv = self.pixel_velocity
            for k in (0.5, 1.0):
                t = k * FALL_REACH_SECONDS
                spots.append((detection.u + du * t, detection.v + dv * t))
        return any(self.in_box(u, v, margin) for u, v in spots) and not self.in_front_of_box(
            position)

    def in_front_of_box(self, position):
        """Was the ball clearly nearer the camera than the box (passing in front of it)?

        Only "in front" is ruled out. Vanishing at or behind the box both mean the
        box now hides it. The distances here are rough: a black box often has no
        depth (its distance then comes from its width), and a ball half-hidden by
        the rim looks small (= far). So use the farthest recent reading of the
        ball, and a margin that grows with distance.
        """
        if self.box_front_m is None:
            return False
        farthest = max([float(position[2])] + [z for _, z, _ in self.recent])
        return farthest < self.box_front_m - (BOX_NEAR_MARGIN_M + 0.05 * self.box_front_m)

    def watch_area(self):
        """While a ball is in the box: the box (where it is now, else where it was)."""
        if not self.in_box_balls:
            return None
        return self.box if self.box is not None else self.in_box_balls[-1]["box"]

    def at_box_exit(self, detection):
        """Where a ball lifted or bouncing out of the box would appear: the box, one
        box-width either side, and everything above it up to the top of the picture."""
        area = self.watch_area()
        if area is None:
            return False
        u0, v0, u1, v1 = area
        side = max(60.0, u1 - u0, 3 * detection.radius)
        below = max(60.0, 3 * detection.radius)
        return u0 - side <= detection.u <= u1 + side and detection.v <= v1 + below

    def is_junk(self, detection):
        """At a spot where an orange blob already was when the ball went in."""
        return any(math.hypot(detection.u - u, detection.v - v) <= max(20.0, 2 * r)
                   for u, v, r in self.junk)

    def came_from_box(self, measurement):
        """A newly seen ball that appears at the box while a ball is in it."""
        return self.at_box_exit(measurement.detection)

    def _event(self, now, text):
        self.events.append(text)
        self.last_event = (now, text)

    # ----- following one ball -----
    def _remember_loss(self, now):
        d = self.last.detection
        self.lost_pixel, self.lost_time = (d.u, d.v, d.radius), now

    def near_where_lost(self, detection, now):
        """Is this blob where the ball recently vanished?"""
        if self.lost_pixel is None or now - self.lost_time > REACQUIRE_SECONDS:
            return False
        u, v, r = self.lost_pixel
        return math.hypot(detection.u - u, detection.v - v) <= max(80.0, 4 * r)

    def predicted_pixel(self, now):
        du, dv = self.pixel_velocity
        dt = min(now - self.last_time, 0.2)
        return self.last.detection.u + du * dt, self.last.detection.v + dv * dt

    def fits(self, measurement, now):
        """Is this measurement plausibly the same ball one frame (or a few) later?"""
        if self.last is None:
            return True
        dt = max(now - self.last_time, 1e-3)
        d, prev = measurement.detection, self.last.detection
        pu, pv = self.predicted_pixel(now)
        speed_px = math.hypot(*self.pixel_velocity)
        reach = max(40.0, 2.0 * prev.radius) + 0.5 * speed_px * min(dt, 0.2)
        if math.hypot(d.u - pu, d.v - pv) > reach:
            return False
        # Apparent size changes gradually as the ball rolls closer or farther.
        frames = max(1.0, dt * 30)
        low, high = 0.75 ** frames, 1.35 ** frames
        if now - self.locked_at < FRESH_LOCK_SECONDS:
            # Just locked, maybe on a partly hidden ball (box rim, fingers):
            # it can look much bigger a frame later as it comes into full view.
            low, high = min(low, 0.5), max(high, 2.5)
        if not low <= d.radius / prev.radius <= high:
            return False
        if now - self.locked_at < FRESH_LOCK_SECONDS:
            return True  # a partly hidden ball's size-based distance is unreliable
        jump = float(((measurement.position - self.last.position) ** 2).sum()) ** 0.5
        return jump <= max(0.15, MAX_BALL_SPEED_MPS * dt)

    def accept(self, measurement, now):
        if self.last is not None:
            dt = now - self.last_time
            if 0 < dt <= 0.2:
                d, prev = measurement.detection, self.last.detection
                new = ((d.u - prev.u) / dt, (d.v - prev.v) / dt)
                old = self.pixel_velocity
                self.pixel_velocity = (0.5 * old[0] + 0.5 * new[0], 0.5 * old[1] + 0.5 * new[1])
        self.last, self.last_time = measurement, now
        self.lost_at = None
        self.recent.append((now, float(measurement.position[2]), measurement.source))
        while now - self.recent[0][0] > RECENT_Z_SECONDS:
            self.recent.popleft()
        if self.state == "LOCKED":
            self.history.append((now, measurement.position))
            while now - self.history[0][0] > SPEED_SECONDS:
                self.history.popleft()

    def _become_locked(self, measurement, now):
        """Lock on and give the ball its number (an old one if it came out of the box)."""
        self.state = "LOCKED"
        self.locked_at = now
        self.lost_pixel = None
        self.history.clear()
        if (self.from_box or self.taken_out) and self.in_box_balls:
            d = measurement.detection
            ball = min(self.in_box_balls,
                       key=lambda b: math.hypot(b["pixel"][0] - d.u, b["pixel"][1] - d.v))
            self.in_box_balls.remove(ball)
            self.ball_id = ball["id"]
            self._event(now, "ball came back OUT of the box" if self.from_box else
                        "ball is out of the box (seen away from the box)")
            self.out_time = now
            if not self.in_box_balls:
                self.junk = []
        else:
            self.ball_id = self.next_id
            self.next_id += 1
            self._event(now, "ball locked")
        self.from_box = self.taken_out = False
        self.accept(measurement, now)

    def lock_now(self, measurement, now):
        """Lock immediately (you clicked the ball)."""
        from_box = self.came_from_box(measurement)
        self.release()
        self.from_box = from_box
        self._become_locked(measurement, now)

    def clear_box(self, now):
        self.in_box_balls.clear()
        self.junk = []
        self._event(now, "box emptied (count cleared)")

    def speed_toward_camera(self, np):
        """Rate the ball's Z is shrinking, m/s (positive = coming closer), or None."""
        if len(self.history) < 4 or self.history[-1][0] - self.history[0][0] < 0.1:
            return None
        t = np.array([h[0] for h in self.history])
        z = np.array([h[1][2] for h in self.history])
        return -float(np.polyfit(t - t[-1], z, 1)[0])

    def update(self, rs, np, depth, intrinsics, candidates, now, max_distance_m):
        """Advance the state machine; return (Measurement or None, shown blob, reason)."""
        if self.state == "LOCKED":
            pu, pv = self.predicted_pixel(now)
            ordered = sorted(candidates, key=lambda c: math.hypot(c.detection.u - pu,
                                                                  c.detection.v - pv))
        elif self.lost_pixel is not None and now - self.lost_time <= REACQUIRE_SECONDS:
            # Look first where the ball vanished.
            lu, lv, _ = self.lost_pixel
            ordered = sorted(candidates, key=lambda c: math.hypot(c.detection.u - lu,
                                                                  c.detection.v - lv))
        else:
            # Prefer bigger (closer) blobs.
            ordered = sorted(candidates, key=lambda c: -c.detection.radius)
        shown, reason = None, "no new round ball-colored blob"
        if self.in_box_balls and self.state != "LOCKED":
            # The ball is in the box: watch the box first; elsewhere, ignore the spots
            # where orange things already were when it went in.
            ordered = sorted((c for c in ordered
                              if self.at_box_exit(c.detection) or not self.is_junk(c.detection)),
                             key=lambda c: not self.at_box_exit(c.detection))
            reason = "ball in box - watching the box"
        for candidate in ordered[:MAX_CANDIDATES]:
            found, why = locate(rs, np, depth, intrinsics, candidate.detection, max_distance_m)
            if found is None:
                if shown is None:
                    shown, reason = candidate.detection, why
                continue
            if self.state == "LOCKED":
                if self.fits(found, now):
                    self.accept(found, now)
                    return found, found.detection, None
                if shown is None:
                    shown, reason = candidate.detection, "blob does not match the locked ball"
                continue
            # SEARCHING / LOCKING / LOST: count consistent sightings.
            if self.state == "LOCKING" and self.fits(found, now):
                self.hits += 1
            else:
                from_box = self.came_from_box(found)
                taken_out = bool(self.in_box_balls) and not from_box
                quick = self.near_where_lost(found.detection, now)
                self.release(keep_lost=True)
                self.state, self.hits = "LOCKING", 1
                self.from_box, self.taken_out = from_box, taken_out
                # Coming out of the box: lock at once (only the box is being watched).
                self.needed = 1 if from_box else (REACQUIRE_FRAMES if quick else LOCK_FRAMES)
            self.accept(found, now)
            if self.hits >= self.needed:
                self._become_locked(found, now)
            return found, found.detection, None

        # Nothing acceptable this frame.
        missing = now - self.last_time
        if self.state == "LOCKED":
            d = self.last.detection
            # Being lifted out (moving up, or just came out): not going in.
            rising = (self.pixel_velocity[1] < -RISING_PX_PER_S
                      or now - self.out_time < JUST_OUT_SECONDS)
            if (self.near_box(d, self.last.position, now) and not rising
                    and missing > IN_BOX_GRACE_SECONDS):
                # It vanished at the box: it went in. Remember it, look for the next ball.
                self.in_box_balls.append({"id": self.ball_id, "pixel": (d.u, d.v),
                                          "position": self.last.position, "time": now,
                                          "box": self.box})
                # Whatever orange is still visible now isn't the ball: remember the spots.
                self.junk = [(c.detection.u, c.detection.v, c.detection.radius)
                             for c in candidates]
                self._event(now, "ball went INTO the box")
                self.release()
            elif missing > LOST_SECONDS:
                self._remember_loss(now)
                self.state, self.lost_at = "LOST", now
        elif self.state == "LOCKING" and missing > LOCKING_MISS_SECONDS:
            self.state, self.hits = "SEARCHING", 0
            self.from_box = self.taken_out = False
        elif self.state == "LOST" and now - self.lost_at > 1.0:
            self.state = "SEARCHING"
        return None, shown, reason


class Averager:
    """Average the last second of positions; restart when the ball moves."""

    def __init__(self, seconds=AVERAGE_SECONDS):
        self.seconds = seconds
        self.samples = deque()

    def add(self, np, t, position, source):
        if self.samples:
            mean = np.mean([p for _, p, _ in self.samples], axis=0)
            if np.linalg.norm(position - mean) > MOVED_M:
                self.samples.clear()
        self.samples.append((t, position, source))
        while t - self.samples[0][0] > self.seconds:
            self.samples.popleft()

    def clear(self):
        self.samples.clear()

    def summary(self, np):
        """Return (mean m, spread mm, count, source) or None."""
        if not self.samples:
            return None
        points = np.stack([p for _, p, _ in self.samples])
        mean = points.mean(axis=0)
        spread_mm = 1000 * float(np.sqrt(((points - mean) ** 2).sum(axis=1).mean()))
        sources = {s for _, _, s in self.samples}
        source = sources.pop() if len(sources) == 1 else "mixed"
        return mean, spread_mm, len(points), source


def is_steady(summary):
    return (summary is not None and summary[2] >= MIN_STEADY_SAMPLES
            and summary[1] <= STEADY_SPREAD_MM)


def cm(position):
    x, y, z = 100 * position
    return f"X {x:+6.1f}  Y {y:+6.1f}  Z {z:6.1f} cm"


def distance_cm(position):
    return 100 * math.sqrt(float((position**2).sum()))


def next_point_number(path):
    if not os.path.exists(path):
        return 1
    with open(path, newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle)) + 1


def save_point(path, number, summary):
    mean, spread_mm, samples, source = summary
    x, y, z = 100 * mean
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "point": number, "time": time.strftime("%H:%M:%S"), "source": source,
            "cam_x_cm": f"{x:.1f}", "cam_y_cm": f"{y:.1f}", "cam_z_cm": f"{z:.1f}",
            "distance_cm": f"{distance_cm(mean):.1f}",
            "spread_mm": f"{spread_mm:.1f}", "samples": samples,
            "tape_x_cm": "", "tape_y_cm": "", "tape_z_cm": "",
        })


def tape_mask(cv2, np, hsv, tape):
    """Pixels that look like the blue tape on the box (vivid, fairly bright blue)."""
    mask = cv2.inRange(hsv, (tape["hue_min"], tape["sat_min"], tape["value_min"]),
                       (tape["hue_max"], 255, tape["value_max"]))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def box_distance(np, depth_m, region, fx, width_px):
    """(front_m, z_m) of the box from depth inside `region`, else from its apparent width."""
    values = depth_m[region]
    values = values[(values >= MIN_DEPTH_M) & (values <= 6.0)]
    if values.size >= max(20, 0.15 * np.count_nonzero(region)):
        middle = float(np.median(values))
        values = values[np.abs(values - middle) <= 0.25]   # drop stray depth specks
        return float(np.percentile(values, 10)), middle
    # Black surfaces often give no depth: the wider the box looks, the closer it is.
    z = fx * BOX_WIDTH_M / width_px
    return z, z


def count_stripes(np, hsv, box):
    """How many horizontal tape stripes cross `box` (x, y, w, h).

    Reads the average colorfulness of each row down the middle of the patch:
    each stripe is a high band, the black gaps between stripes are dips. This
    still works when the camera blurs the stripes into one blue blob.
    """
    x, y, w, h = box
    if h < 5:
        return 0
    rows = hsv[y:y + h, x + w // 5:x + w - w // 5, 1].astype(np.float32).mean(axis=1)
    rows = np.convolve(rows, np.ones(3) / 3, mode="same")
    top = rows.max()
    high, low = 0.88 * top, 0.80 * top
    bands, inside = 0, False
    for value in rows:
        if not inside and value >= high:
            bands, inside = bands + 1, True
        elif inside and value <= low:
            inside = False
    return bands


def find_taped_box(cv2, np, hsv, depth_m, fx, previous, tape):
    """Find the box from its striped blue tape (e.g. 3 dark-navy stripes on the front).

    Tape-coloured pixels lying close together are joined into one patch; that
    patch is the box face if it shows at least TAPE_MIN_STRIPES stripes
    (count_stripes) and its real size (from depth, or from its width when there
    is no depth) is about the box's. Single blue things - a part on
    another robot, LEDs, a line of the blue track on the table - don't count.
    """
    height, width = hsv.shape[:2]
    blue = tape_mask(cv2, np, hsv, tape)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(blue)
    stripes = [k for k in range(1, count) if stats[k][4] >= TAPE_MIN_PIXELS]
    if not stripes:
        return None
    # Join stripes separated by thin gaps: close vertically and horizontally a bit.
    keep = np.isin(labels, stripes).astype(np.uint8)
    joined = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((9, 5), np.uint8))
    n, parts, part_stats, _ = cv2.connectedComponentsWithStats(joined)
    best, best_score = None, math.inf
    for part in range(1, n):
        bx, by, bw, bh, area = part_stats[part]
        if count_stripes(np, hsv, (bx, by, bw, bh)) < TAPE_MIN_STRIPES:
            continue
        if area < 0.5 * bw * bh:
            continue  # scattered bits, not a solid striped face
        region = np.zeros(depth_m.shape, bool)
        region[by:by + bh, bx:bx + bw] = parts[by:by + bh, bx:bx + bw] == part
        front, z = box_distance(np, depth_m, region, fx, bw)
        width_m, height_m = bw * z / fx, bh * z / fx
        if not (BOX_WIDTH_RANGE_M[0] <= width_m <= BOX_WIDTH_RANGE_M[1]
                and 0.03 <= height_m <= BOX_HEIGHT_RANGE_M[1]):
            continue
        # The tape is on the front face; the box opening is just above it.
        rim = round(0.25 * bh)
        box = (int(bx), int(max(by - rim, 0)), int(bx + bw - 1), int(by + bh - 1))
        if previous is not None:
            score = math.hypot((box[0] + box[2] - previous[0] - previous[2]) / 2,
                               (box[1] + box[3] - previous[1] - previous[3]) / 2)
        else:
            score = -float(area)
        if score < best_score:
            best, best_score = (box, front, width_m, height_m), score
    return best


def find_catch_box(cv2, np, image, depth_m, fx, previous=None, tape=None):
    """Find the black catch box: a solid dark blob whose real size matches the box.

    With `tape` (HSV limits), the dark blob must also carry the blue tape, so
    other black things (a hat, a monitor, a sleeve) are ignored (find_taped_box).

    depth_m: aligned depth image in meters (0 = no reading). Returns
    ((u0, v0, u1, v1), front_m, width_m, height_m) or None.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if tape is not None:
        return find_taped_box(cv2, np, hsv, depth_m, fx, previous, tape)
    dark = (hsv[..., 2] < BOX_DARK_VALUE).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, math.inf
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 300:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if hull_area == 0 or area / hull_area < 0.65:
            continue  # the box is a solid shape (a hand over it is OK), not a frame or scatter
        x, y, w, h = cv2.boundingRect(contour)
        inside = np.zeros((h, w), np.uint8)
        cv2.drawContours(inside, [contour], -1, 1, cv2.FILLED, offset=(-x, -y))
        values = depth_m[y:y + h, x:x + w][(inside != 0) & (depth_m[y:y + h, x:x + w] > 0)]
        values = values[(values >= MIN_DEPTH_M) & (values <= 6.0)]
        if values.size >= max(20, 0.1 * area):
            z = float(np.median(values))
            front = float(np.percentile(values, 10))  # nearest part = front wall
            size_only = False
        else:
            # Black surfaces often give no depth. Then use the box's known width:
            # the wider it looks, the closer it is (same idea as the ball's size).
            if not 0.4 <= h / w <= 1.3:
                continue  # not box-shaped (the box is about 15 wide x 10 tall)
            z = front = fx * BOX_WIDTH_M / w
            size_only = True
        width_m, height_m = w * z / fx, h * z / fx
        if not (BOX_WIDTH_RANGE_M[0] <= width_m <= BOX_WIDTH_RANGE_M[1]
                and BOX_HEIGHT_RANGE_M[0] <= height_m <= BOX_HEIGHT_RANGE_M[1]):
            continue
        box = (x, y, x + w - 1, y + h - 1)
        if previous is not None:
            pu, pv = (previous[0] + previous[2]) / 2, (previous[1] + previous[3]) / 2
            score = math.hypot(x + w / 2 - pu, y + h / 2 - pv)
        else:
            score = abs(width_m - 0.15) + abs(height_m - 0.10)
        if size_only:
            score += 1000.0  # last resort: only if nothing is confirmed by depth
        if score < best_score:
            best, best_score = (box, front, width_m, height_m), score
    return best


def box_list(lock):
    """e.g. '2 (#1, #3)' or '0'."""
    ids = [f"#{b['id']}" for b in lock.in_box_balls]
    return f"{len(ids)} ({', '.join(ids)})" if ids else "0"


def load_box(path):
    """The marked catch box: ((u0, v0, u1, v1) in image pixels, front distance m or None)."""
    import json
    try:
        with open(path) as handle:
            data = json.load(handle)
        front = data.get("front_m")
        return tuple(int(c) for c in data["box_px"]), (None if front is None else float(front))
    except (OSError, ValueError, KeyError, TypeError):
        return None, None


def save_box(path, box, front_m):
    import json
    with open(path, "w") as handle:
        json.dump({"box_px": list(box), "front_m": front_m,
                   "note": "catch box area in the 640x480 color image and the distance (Z, m) "
                           "from the camera to its front wall; re-mark with k if the camera "
                           "or box moves"}, handle, indent=2)


def measure_box_front(np, depth, box):
    """Distance to the box's nearest face (its front wall) from depth, or None."""
    if depth is None:
        return None
    u0, v0, u1, v1 = box
    values = np.asanyarray(depth.get_data())[v0:v1 + 1, u0:u1 + 1].astype(float)
    values = values[values > 0] * depth.get_units()
    values = values[(values >= MIN_DEPTH_M) & (values <= 8.0)]
    if values.size < 30:
        return None
    # The front wall is the nearest big surface: a low percentile ignores stray specks.
    return float(np.percentile(values, 10))


CAPTURE_FRAMES = 10
BUTTON = (230, 52)   # capture button size (w, h), drawn at the top-right of the panel


def button_rect(width, image_height):
    """Capture button (x0, y0, x1, y1) in window pixels: top-right of the panel."""
    top = round(image_height * SCALE)
    return width - BUTTON[0] - 12, top + 10, width - 12, top + 10 + BUTTON[1]


def record_rect(width, image_height):
    """Record button (x0, y0, x1, y1) in window pixels: left of the capture button."""
    x0, y0, _, y1 = button_rect(width, image_height)
    return x0 - 12 - BUTTON[0], y0, x0 - 12, y1


def draw(cv2, np, image, ball, banner, banner_color, headline, lines, color, box=None,
         box_label="BOX", button_text="CAPTURE 10", record_text=None):
    """Show the video (enlarged) with a state banner and a readable panel underneath."""
    preview = cv2.resize(image, None, fx=SCALE, fy=SCALE)
    if box is not None:
        u0, v0, u1, v1 = (round(c * SCALE) for c in box)
        cv2.rectangle(preview, (u0, v0), (u1, v1), BLUE, 2)
        cv2.putText(preview, box_label or "BOX", (u0 + 4, max(v0 - 8, 70)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLUE, 2, cv2.LINE_AA)
    if ball is not None:
        center = (round(ball.u * SCALE), round(ball.v * SCALE))
        cv2.circle(preview, center, int(round(ball.radius * SCALE)), color, 3)
        cv2.circle(preview, center, 4, color, -1)
    cv2.rectangle(preview, (0, 0), (preview.shape[1], 52), (0, 0, 0), -1)
    cv2.putText(preview, banner, (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, banner_color, 3,
                cv2.LINE_AA)
    panel = np.zeros((70 + 34 * len(lines), preview.shape[1], 3), dtype=np.uint8)
    cv2.putText(panel, headline, (14, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.25, color, 3,
                cv2.LINE_AA)
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (14, 92 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    WHITE, 2, cv2.LINE_AA)
    shown = np.vstack([preview, panel])
    if button_text is None:   # no capture button (e.g. the ROS node's preview)
        cv2.imshow(WINDOW, shown)
        return cv2.waitKey(1) & 0xFF
    x0, y0, x1, y1 = button_rect(shown.shape[1], image.shape[0])
    busy = button_text != "CAPTURE 10"
    cv2.rectangle(shown, (x0, y0), (x1, y1), (0, 140, 255) if busy else (90, 90, 90), -1)
    cv2.rectangle(shown, (x0, y0), (x1, y1), WHITE, 2)
    cv2.putText(shown, button_text, (x0 + 14, y1 - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.85, WHITE, 2,
                cv2.LINE_AA)
    if record_text is not None:
        x0, y0, x1, y1 = record_rect(shown.shape[1], image.shape[0])
        recording = record_text != "RECORD"
        cv2.rectangle(shown, (x0, y0), (x1, y1), (0, 0, 210) if recording else (90, 90, 90), -1)
        cv2.rectangle(shown, (x0, y0), (x1, y1), WHITE, 2)
        cv2.circle(shown, (x0 + 24, (y0 + y1) // 2), 9, WHITE if recording else (0, 0, 230), -1)
        cv2.putText(shown, record_text, (x0 + 44, y1 - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    WHITE, 2, cv2.LINE_AA)
    cv2.imshow(WINDOW, shown)
    return cv2.waitKey(1) & 0xFF


RECORD_FIELDS = ["frame", "t_s", "state", "ball_id", "ball_u", "ball_v", "ball_r_px",
                 "cam_x_m", "cam_y_m", "cam_z_m", "source", "box_u0", "box_v0", "box_u1",
                 "box_v1", "box_front_m", "balls_in_box", "visible_in_box", "event",
                 "rgb_exposure_us", "rgb_gain"]
RECORD_MAX_SECONDS = 60.0


class Recorder:
    """RECORD button: save a throw for replay and for recreating it in simulation.

    Each recording is a folder (under --record-dir) with:
      color.mp4   the raw color video (no drawings)
      depth.npz   the aligned depth frames (uint16, times depth_units = meters)
      track.csv   what the tracker saw on every frame: ball pixel, size and camera
                  XYZ, the box outline and distance, in-box state, events
      meta.json   camera serial, intrinsics, frame rate, depth units
    scripts/recording_to_sim.py turns a recording into a MuJoCo replay.
    """

    def __init__(self, cv2, np, root, intrinsics, serial, fps):
        self.cv2, self.np, self.root = cv2, np, root
        self.intrinsics, self.serial, self.fps = intrinsics, serial, fps
        self.folder = None

    @property
    def active(self):
        return self.folder is not None

    def start(self, now):
        self.folder = os.path.join(self.root, time.strftime("throw_%Y%m%d_%H%M%S"))
        os.makedirs(self.folder, exist_ok=True)
        size = (self.intrinsics.width, self.intrinsics.height)
        self.video = self.cv2.VideoWriter(os.path.join(self.folder, "color.mp4"),
                                          self.cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size)
        self.rows, self.depth, self.started, self.units = [], [], now, None
        print(f"RECORDING to {self.folder} (click STOP or press v to stop)", flush=True)

    def add(self, now, image, aligned, lock, found, events, color_metadata=None):
        np = self.np
        if aligned is not None:
            self.units = aligned.get_units()
            depth = np.asanyarray(aligned.get_data()).astype(np.uint16, copy=True)
        else:
            depth = np.zeros((self.intrinsics.height, self.intrinsics.width), np.uint16)
        self.video.write(image)
        self.depth.append(depth)
        row = {"frame": len(self.rows), "t_s": f"{now - self.started:.4f}", "state": lock.state,
               "ball_id": lock.ball_id if lock.ball_id is not None else "",
               "balls_in_box": len(lock.in_box_balls), "event": " | ".join(events)}
        if color_metadata:
            row.update(color_metadata)
        if found is not None and lock.state == "LOCKED":
            d = found.detection
            row.update(ball_u=d.u, ball_v=d.v, ball_r_px=f"{d.radius:.2f}",
                       cam_x_m=f"{found.position[0]:.4f}", cam_y_m=f"{found.position[1]:.4f}",
                       cam_z_m=f"{found.position[2]:.4f}", source=found.source,
                       visible_in_box=int(lock.in_box_3d(d, found.position)))
        if lock.box is not None:
            row.update(zip(("box_u0", "box_v0", "box_u1", "box_v1"), lock.box))
            if lock.box_front_m is not None:
                row["box_front_m"] = f"{lock.box_front_m:.3f}"
        self.rows.append(row)
        if now - self.started > RECORD_MAX_SECONDS:
            print(f"Recording reached {RECORD_MAX_SECONDS:.0f} s: stopping.", flush=True)
            self.stop()

    def stop(self):
        np = self.np
        self.video.release()
        np.savez_compressed(os.path.join(self.folder, "depth.npz"),
                            depth=np.stack(self.depth) if self.depth else np.zeros((0, 1, 1)))
        with open(os.path.join(self.folder, "track.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RECORD_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        i = self.intrinsics
        meta = {"camera_serial": self.serial, "fps": self.fps, "frames": len(self.rows),
                "depth_units_m": self.units, "ball": BALL["color"],
                "ball_radius_m": BALL_RADIUS_M,
                "intrinsics": {"width": i.width, "height": i.height, "fx": i.fx, "fy": i.fy,
                               "ppx": i.ppx, "ppy": i.ppy, "model": str(i.model),
                               "coeffs": list(i.coeffs)},
                "camera_frame": "RealSense RGB optical: x right, y down, z forward, meters"}
        with open(os.path.join(self.folder, "meta.json"), "w") as handle:
            json.dump(meta, handle, indent=2)
        seen = sum(1 for r in self.rows if r.get("cam_z_m"))
        print(f"SAVED recording {self.folder}: {len(self.rows)} frames, ball tracked in "
              f"{seen}", flush=True)
        self.folder = None


class VisionTracker:
    """The per-frame vision pipeline, without any window: learn the empty scene, find
    ball candidates and the catch box, and run the lock-on state machine.

    measure() (the on-screen tool) and ball_vision_node.py (ROS) both use this, so
    they behave the same. Settings come from the command-line `args` (see main()).
    """

    def __init__(self, rs, cv2, np, intrinsics, args):
        self.rs, self.cv2, self.np, self.intrinsics, self.args = rs, cv2, np, intrinsics, args
        self.lock = BallLock()
        self.lock.box_depth_m = args.box_depth
        self.box_auto = not args.manual_box
        self.tape = None if args.no_box_tape else TAPE
        self.learn_frames = max(1, round(LEARN_SECONDS * args.fps))
        self.relearn()
        self.box_info, self.box_seen = None, -math.inf
        self.mask = self.candidates = self.depth_m = self.found_box = None

    @property
    def learning(self):
        return self.background is None

    def relearn(self):
        """Forget the empty scene (and the lock); learn it again from the next frames."""
        self.background, self.total, self.learned = None, None, 0
        self.lock.release()

    def learn(self, image):
        """Add a frame of the empty scene. Returns True when learning just finished."""
        frame = image.astype(self.np.float32)
        self.total = frame if self.total is None else self.total + frame
        self.learned += 1
        if self.learned >= self.learn_frames:
            self.background = self.total / self.learned  # average picture of the empty scene
            return True
        return False

    def detect(self, image, aligned, now):
        """Find ball candidates and (automatically) the catch box in this frame."""
        cv2, np, args, lock = self.cv2, self.np, self.args, self.lock
        where = None
        if BALL["color"] == "orange":
            # Only orange pixels can be the ball: compare just those with the empty scene.
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            where = cv2.inRange(hsv, (BALL["hue_min"], min(BALL["sat_min"], BALL["loose_sat_min"]),
                                      min(BALL["value_min"], BALL["loose_value_min"])),
                                (BALL["hue_max"], 255, 255)) > 0
        changed = scene_changed(np, image, self.background, args.min_change, cv2, where)
        self.mask = ball_mask(cv2, np, image, self.background, args.brightness,
                              args.max_saturation, args.min_change, changed=changed)
        self.candidates = find_candidates(cv2, np, self.mask)
        if BALL["color"] == "orange":
            loose_mask = ball_mask(cv2, np, image, self.background, args.brightness,
                                   args.max_saturation, args.min_change, loose=True,
                                   changed=changed)
            self.candidates = merge_candidates(self.candidates,
                                               find_candidates(cv2, np, loose_mask))
        self.depth_m = self.found_box = None
        if self.box_auto and aligned is not None:
            self.depth_m = (np.asanyarray(aligned.get_data()).astype(np.float32)
                            * aligned.get_units())
            self.found_box = find_catch_box(cv2, np, image, self.depth_m, self.intrinsics.fx,
                                            lock.box, self.tape)
            if self.found_box is not None:
                lock.box, lock.box_front_m, w_m, h_m = self.found_box
                if args.box_distance is not None:
                    lock.box_front_m = args.box_distance
                self.box_info = (f"BOX {100 * w_m:.0f}x{100 * h_m:.0f} cm, "
                                 f"front {100 * lock.box_front_m:.0f} cm")
                self.box_seen = now
            elif now - self.box_seen > BOX_KEEP_SECONDS:
                lock.box, lock.box_front_m, self.box_info = None, None, None

    def track(self, image, aligned, now):
        """Advance the lock with this frame's candidates. Returns (found, shown, reason)."""
        args, lock = self.args, self.lock
        found, shown, reason = lock.update(self.rs, self.np, aligned, self.intrinsics,
                                           self.candidates, now, args.max_distance)
        if found is None and lock.state == "LOCKED" and lock.last is not None:
            # The locked ball may have blended into the background: look right
            # where it should be, by color and shape alone.
            pu, pv = lock.predicted_pixel(now)
            target = ball_at(self.cv2, self.np, image, pu, pv, args.brightness,
                             args.max_saturation, lock.last.detection.radius,
                             scene_mask=self.mask)
            if target is not None:
                recovered, _ = locate(self.rs, self.np, aligned, self.intrinsics, target,
                                      args.max_distance)
                if recovered is not None and lock.fits(recovered, now):
                    lock.accept(recovered, now)
                    found, shown, reason = recovered, target, None
        return found, shown, reason


def measure(rs, cv2, np, pipeline, profile, args):
    align = rs.align(rs.stream.color)
    intrinsics = (profile.get_stream(rs.stream.color)
                  .as_video_stream_profile().get_intrinsics())
    tracker = VisionTracker(rs, cv2, np, intrinsics, args)
    lock = tracker.lock
    averager = Averager()
    shown_position = None   # last locked reading, kept on screen for HOLD_SECONDS
    previous_numbers = (None, None)
    last_print = -math.inf
    previous_state = None
    clicks = []
    point = next_point_number(args.output)
    started = time.monotonic()
    pending_box = []   # a newly dragged (manual) box waiting for its depth to be measured
    if tracker.box_auto:
        print("Catch box: found automatically (black, about 15 x 10 cm"
              + (")" if args.no_box_tape else ", with the BLUE TAPE on it)")
              + ". Press k to draw it by hand instead.", flush=True)
    else:
        lock.box, lock.box_front_m = load_box(args.box_file)
        if args.box_distance is not None:
            lock.box_front_m = args.box_distance
        if lock.box is not None and lock.box_front_m is None:
            pending_box.append(lock.box)   # older saved box: measure its distance now
        if lock.box is None:
            print("No catch box marked yet: press k, then drag a rectangle around the box.",
                  flush=True)
    marking = {"on": False, "start": None, "now": None}
    capture = {"left": 0, "folder": None, "count": 0, "found": 0}
    window_width = round(intrinsics.width * SCALE)
    recorder = Recorder(cv2, np, args.record_dir, intrinsics,
                        profile.get_device().get_info(rs.camera_info.serial_number), args.fps)
    toggle_record = []
    rate = {"frames": 0, "skipped": 0, "since": time.monotonic()}
    last_draw = -math.inf

    def start_capture():
        if capture["left"] > 0:
            return
        folder = os.path.join(args.capture_dir, time.strftime("box_%Y%m%d_%H%M%S"))
        os.makedirs(folder, exist_ok=True)
        capture.update(left=CAPTURE_FRAMES, folder=folder, count=0, found=0)
        print(f"Capturing {CAPTURE_FRAMES} frames to {folder}", flush=True)

    cv2.namedWindow(WINDOW)

    def on_mouse(event, x, y, flags, userdata):
        bx0, by0, bx1, by1 = button_rect(window_width, intrinsics.height)
        if event == cv2.EVENT_LBUTTONDOWN and bx0 <= x <= bx1 and by0 <= y <= by1:
            start_capture()
            return
        rx0, ry0, rx1, ry1 = record_rect(window_width, intrinsics.height)
        if event == cv2.EVENT_LBUTTONDOWN and rx0 <= x <= rx1 and ry0 <= y <= ry1:
            toggle_record.append(True)
            return
        x, y = x / SCALE, y / SCALE
        if marking["on"]:
            if event == cv2.EVENT_LBUTTONDOWN:
                marking["start"] = marking["now"] = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and marking["start"] is not None:
                marking["now"] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and marking["start"] is not None:
                (xa, ya), (xb, yb) = marking["start"], (x, y)
                box = (min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb))
                marking.update(on=False, start=None, now=None)
                if box[2] - box[0] >= 10 and box[3] - box[1] >= 10:
                    pending_box.append(tuple(round(c) for c in box))
                else:
                    print("Box too small; press k and drag again.", flush=True)
        elif event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    cv2.setMouseCallback(WINDOW, on_mouse)
    log_handle = open(args.log, "w", newline="")
    log = csv.DictWriter(log_handle, fieldnames=LOG_FIELDS)
    log.writeheader()
    print("Learning the empty scene: keep the ball OUT of view...", flush=True)

    try:
        while True:
            frames = pipeline.wait_for_frames(5000)
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not (color and depth):
                continue
            numbers = (color.get_frame_number(), depth.get_frame_number())
            if numbers[0] == previous_numbers[0] or numbers[1] == previous_numbers[1]:
                continue
            if previous_numbers[0] is not None and numbers[0] > previous_numbers[0] + 1:
                rate["skipped"] += numbers[0] - previous_numbers[0] - 1
            previous_numbers = numbers
            now = time.monotonic()
            image = np.asanyarray(color.get_data())
            rate["frames"] += 1
            if now - rate["since"] >= 5.0:
                seen = rate["frames"] + rate["skipped"]
                print(f"RATE: processed {rate['frames'] / (now - rate['since']):.0f} fps of "
                      f"{seen / (now - rate['since']):.0f} from the camera "
                      f"({rate['skipped']} frames skipped)", flush=True)
                rate.update(frames=0, skipped=0, since=now)

            if tracker.learning:
                if tracker.learn(image):
                    print("Empty scene learned. Put the ball in view.", flush=True)
                clicks.clear()
                key = draw(cv2, np, image, None, "LEARNING EMPTY SCENE", WHITE,
                           "Keep the ball OUT of view",
                           [f"{tracker.learned}/{tracker.learn_frames} frames"], WHITE)
                if key in (ord("q"), 27):
                    break
                continue

            aligned = align.process(frames).get_depth_frame() or None
            tracker.detect(image, aligned, now)
            mask, candidates = tracker.mask, tracker.candidates
            if args.show_mask:
                cv2.imshow("ball mask (white = could be the ball)", mask)
                if not args.no_box_tape:
                    cv2.imshow("tape mask (white = counted as the box's blue tape)",
                               tape_mask(cv2, np, cv2.cvtColor(image, cv2.COLOR_BGR2HSV), TAPE))
            if tracker.depth_m is not None:
                depth_m, found_box = tracker.depth_m, tracker.found_box
                if capture["left"] > 0:
                    capture["count"] += 1
                    capture["left"] -= 1
                    capture["found"] += found_box is not None
                    stem = os.path.join(capture["folder"], f"frame{capture['count']:02d}")
                    cv2.imwrite(stem + "_color.png", image)
                    np.save(stem + "_depth.npy", depth_m)
                    marked = image.copy()
                    if found_box is not None:
                        u0, v0, u1, v1 = found_box[0]
                        cv2.rectangle(marked, (u0, v0), (u1, v1), BLUE, 2)
                        cv2.putText(marked, f"BOX front {100 * found_box[1]:.0f} cm",
                                    (u0, max(v0 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1)
                    else:
                        cv2.putText(marked, "NO BOX FOUND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9, (0, 0, 255), 2)
                    cv2.imwrite(stem + "_found.png", marked)
                    if capture["left"] == 0:
                        print(f"Captured {capture['count']} frames: box found in "
                              f"{capture['found']}/{capture['count']} ({capture['folder']})",
                              flush=True)
            if pending_box:
                tracker.box_auto = False   # you drew the box by hand: stop finding it automatically
                lock.box = pending_box.pop()
                pending_box.clear()
                front = measure_box_front(np, aligned, lock.box)
                if args.box_distance is not None:
                    lock.box_front_m = args.box_distance
                elif front is not None:
                    lock.box_front_m = front
                save_box(args.box_file, lock.box, lock.box_front_m)
                if lock.box_front_m is None:
                    print("Box saved, but its distance couldn't be measured (black surfaces can "
                          "give no depth). Measure camera-to-box-front with a tape and restart "
                          "with --box-distance <meters>.", flush=True)
                else:
                    print(f"Box saved: {lock.box}, front wall at {100 * lock.box_front_m:.0f} cm "
                          f"({'from --box-distance' if args.box_distance else 'measured by depth'}).",
                          flush=True)

            found = None
            if clicks:
                # Lock on to what you clicked: a found blob there, else search at the spot.
                cu, cv_ = clicks.pop()
                clicks.clear()
                near = [c.detection for c in candidates
                        if math.hypot(c.detection.u - cu, c.detection.v - cv_)
                        <= max(25.0, 1.2 * c.detection.radius)]
                target = near[0] if near else ball_at(cv2, np, image, cu, cv_,
                                                      args.brightness, args.max_saturation,
                                                      scene_mask=mask)
                if target is None:
                    print("Click: couldn't find a round ball there.", flush=True)
                else:
                    found, why = locate(rs, np, aligned, intrinsics, target, args.max_distance)
                    if found is not None:
                        lock.lock_now(found, now)
                        shown, reason = found.detection, None
                        print(f"Click: LOCKED on ball {2 * target.radius:.0f} px wide.",
                              flush=True)
                    else:
                        print(f"Click: found a ball shape but can't measure it ({why}).",
                              flush=True)
            if found is None:
                found, shown, reason = tracker.track(image, aligned, now)

            if lock.state != previous_state:
                print(f"--- {lock.state} ---", flush=True)
                if lock.state != "LOCKED":
                    averager.clear()
                previous_state = lock.state
            for text in lock.events:
                print(f"*** {text} | ball in box: {'YES' if lock.in_box_balls else 'NO'}",
                      flush=True)
                log.writerow({"t_s": f"{now - started:.3f}", "state": "EVENT",
                              "balls_in_box": len(lock.in_box_balls), "event": text,
                              "box_px": "" if lock.box is None else " ".join(map(str, lock.box)),
                              "box_front_cm": "" if lock.box_front_m is None
                              else f"{100 * lock.box_front_m:.0f}"})
            if toggle_record:
                toggle_record.clear()
                if recorder.active:
                    recorder.stop()
                else:
                    recorder.start(now)
            if recorder.active:
                recorder.add(now, image, aligned, lock, found, list(lock.events),
                             color_frame_metadata(rs, color))
            lock.events.clear()

            speed = None
            if lock.state == "LOCKED" and found is not None:
                averager.add(np, now, found.position, found.source)
                speed = lock.speed_toward_camera(np)
                shown_position = (found.position, found.source)
                x, y, z = 100 * found.position
                log.writerow({
                    "t_s": f"{now - started:.3f}", "state": lock.state,
                    "ball_id": lock.ball_id, "balls_in_box": len(lock.in_box_balls),
                    "x_cm": f"{x:.1f}", "y_cm": f"{y:.1f}", "z_cm": f"{z:.1f}",
                    "distance_cm": f"{distance_cm(found.position):.1f}",
                    "source": found.source, "radius_px": f"{found.detection.radius:.1f}",
                    "u_px": found.detection.u, "v_px": found.detection.v,
                    "box_px": "" if lock.box is None else " ".join(map(str, lock.box)),
                    "box_front_cm": "" if lock.box_front_m is None
                    else f"{100 * lock.box_front_m:.0f}",
                    "speed_toward_camera_mps": "" if speed is None else f"{speed:.3f}",
                })
            summary = averager.summary(np)
            steady = lock.state == "LOCKED" and is_steady(summary)

            # ----- what to show -----
            recent_event = None
            if lock.last_event is not None and now - lock.last_event[0] <= 2.0:
                recent_event = lock.last_event[1]
            if marking["on"]:
                banner, banner_color = "MARK THE BOX: drag a rectangle around it", BLUE
            elif lock.state == "LOCKED" and found is not None and lock.in_box_3d(
                    found.detection, found.position):
                banner, banner_color = "BALL IN BOX (still visible)", BLUE
            elif lock.state == "LOCKED":
                banner, banner_color = "LOCKED ON - ball NOT in box", GREEN
            elif lock.in_box_balls:
                banner, banner_color = "BALL IN BOX", BLUE
            elif lock.state == "LOCKING":
                banner, banner_color = (f"LOCKING {lock.hits}/{lock.needed} - hold the ball in view",
                                        YELLOW)
            elif lock.state == "LOST":
                banner, banner_color = "LOST - searching again", RED
            else:
                banner, banner_color = "SEARCHING - put the ball in view or click it", YELLOW

            held = (found is None and shown_position is not None
                    and now - lock.last_time <= HOLD_SECONDS)
            boxed_recently = lock.state != "LOCKED" and bool(lock.in_box_balls)
            lines = []
            if lock.state == "LOCKED" and found is not None:
                position, source = (summary[0], summary[3]) if steady else (found.position,
                                                                            found.source)
                x, y, z = 100 * position
                headline, tint = f"X {x:+.1f}   Y {y:+.1f}   Z {z:.1f} cm", GREEN
                lines.append(f"distance {distance_cm(position):.1f} cm  |  from {source}"
                             f"  |  ball {2 * found.detection.radius:.0f} px wide")
                if steady:
                    lines.append(f"STILL (wobble {summary[1]:.1f} mm, 1 s average): press s to save")
                elif speed is not None and abs(speed) >= 0.02:
                    way = "TOWARD camera" if speed > 0 else "AWAY from camera"
                    lines.append(f"moving {way} at {abs(speed):.2f} m/s")
                else:
                    lines.append("hold still to get a steady reading")
            elif boxed_recently:
                ball = lock.in_box_balls[-1]
                x, y, z = 100 * ball["position"]
                headline, tint = "BALL IN BOX", BLUE
                lines.append(f"went in {now - ball['time']:.1f} s ago, last seen "
                             f"X {x:+.1f}  Y {y:+.1f}  Z {z:.1f} cm")
                lines.append("lift it out and it is tracked again")
            elif held:
                position, source = shown_position
                x, y, z = 100 * position
                headline, tint = f"X {x:+.1f}   Y {y:+.1f}   Z {z:.1f} cm", GRAY
                lines.append(f"LAST READING ({source}) - ball not seen for "
                             f"{now - lock.last_time:.1f} s")
            elif found is not None and lock.state == "LOCKING":
                x, y, z = 100 * found.position
                headline, tint = f"X {x:+.1f}   Y {y:+.1f}   Z {z:.1f} cm", YELLOW
                lines.append("not locked yet")
            else:
                headline, tint = "No ball", YELLOW
                lines.append(reason or "")
            in_now = (lock.in_box_balls and lock.state != "LOCKED") or (
                lock.state == "LOCKED" and found is not None
                and lock.in_box_3d(found.detection, found.position))
            lines.append(f"Ball in box: {'YES' if in_now else 'NO'}   |   "
                         f"X right+ Y down+ Z out")
            if lock.box is None:
                lines.append(("catch box not found (black, ~15 x 10 cm)" if args.no_box_tape
                               else "catch box not found (black + blue tape)") if tracker.box_auto else
                             "no box marked: press k, then drag around the catch box")
            elif tracker.box_auto:
                lines.append(f"catch box found automatically, front wall at "
                             f"{100 * lock.box_front_m:.0f} cm")
            elif lock.box_front_m is None:
                lines.append("box distance unknown: in-box uses the outline only")
            else:
                lines.append(f"box front wall at {100 * lock.box_front_m:.0f} cm")
            lines.append("click=lock r=release s=save b=relearn k=draw box c=reset q=quit")

            if now - last_print >= 0.2:
                if lock.state == "LOCKED" and found is not None:
                    extra = f" | {speed:+.2f} m/s toward camera" if speed is not None else ""
                    print(f"LOCKED {cm(found.position)} | {distance_cm(found.position):.1f} cm "
                          f"({found.source}){' | STEADY' if steady else ''}{extra}", flush=True)
                elif lock.state != "LOCKED":
                    print(f"{lock.state}: {reason or 'candidate seen'}", flush=True)
                last_print = now

            box = lock.box
            if marking["on"] and marking["start"] is not None:
                (xa, ya), (xb, yb) = marking["start"], marking["now"]
                box = (min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb))
            if now - last_draw < 1 / 31 and args.fps > 30:
                continue   # at 60 fps every frame is tracked; the window is redrawn ~30/s
            last_draw = now
            key = draw(cv2, np, image, shown, banner, banner_color, headline, lines, tint,
                       box=box, box_label=tracker.box_info if tracker.box_auto else "BOX (drawn)",
                       button_text=(f"SAVING {capture['count']}/{CAPTURE_FRAMES}"
                                    if capture["left"] > 0 else "CAPTURE 10"),
                       record_text=(f"STOP {now - recorder.started:4.1f}s" if recorder.active
                                    else "RECORD"))
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("b"):
                tracker.relearn()
                averager.clear()
                shown_position = None
                print("Relearning the empty scene: keep the ball OUT of view...", flush=True)
            elif key == ord("c"):
                lock.clear_box(now)
            elif key == ord("p"):
                start_capture()
            elif key == ord("v"):
                toggle_record.append(True)
            elif key == ord("k"):
                marking.update(on=True, start=None, now=None)
                print("Mark the box: drag a rectangle around the catch box opening.",
                      flush=True)
            elif key == ord("r"):
                lock.release()
                averager.clear()
                print("Lock released.", flush=True)
            elif key == ord("s"):
                if steady:
                    save_point(args.output, point, summary)
                    print(f"SAVED point {point}: {cm(summary[0])} ({summary[3]}, "
                          f"wobble {summary[1]:.1f} mm) -> {args.output}", flush=True)
                    point += 1
                else:
                    print("Not saved: need LOCKED ON and the ball still for ~1 s.", flush=True)
    finally:
        if recorder.active:
            recorder.stop()
        log_handle.close()


def build_parser(description=__doc__):
    """Command-line options for the tracker (also used by ball_vision_node.py)."""
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fps", type=int, choices=(30, 60), default=30,
                        help="RGB and depth frame rate (default 30; 60 needs USB 3)")
    parser.add_argument("--rgb-exposure", type=float, default=None,
                        help="Manual RGB exposure in raw SDK units; default leaves auto setting unchanged")
    parser.add_argument("--rgb-gain", type=float, default=None,
                        help="Manual RGB gain in raw SDK units; default unchanged")
    parser.add_argument("--width", type=int, default=848,
                        help="Color and depth width: 848 (16:9, ~69 deg wide, default) or 640 (4:3, ~55 deg)")
    parser.add_argument("--height", type=int, default=480, help="Color and depth height (default 480)")
    parser.add_argument("--ball", choices=("white", "orange"), default="white",
                        help="Ball color (default white)")
    parser.add_argument("--hue-min", type=int, default=6,
                        help="Orange only: lowest hue counted as orange, 0-179 (default 6)")
    parser.add_argument("--hue-max", type=int, default=25,
                        help="Orange only: highest hue counted as orange, 0-179 (default 25)")
    parser.add_argument("--sat-min", type=int, default=235,
                        help="Orange only: lowest colorfulness counted as orange (default 235;"
                             " skin under warm light reaches ~215, the ball ~240-255)")
    parser.add_argument("--value-min", type=int, default=100,
                        help="Orange only: lowest brightness counted as orange (default 100)")
    parser.add_argument("--loose-sat-min", type=int, default=BALL["loose_sat_min"],
                        help="Orange only: second, looser colorfulness test for dimmer or "
                             f"farther balls (default {BALL['loose_sat_min']})")
    parser.add_argument("--loose-value-min", type=int, default=BALL["loose_value_min"],
                        help="Orange only: brightness the looser test also needs "
                             f"(default {BALL['loose_value_min']}; skin is ~100-130)")
    parser.add_argument("--brightness", type=int, default=120,
                        help="Minimum ball-pixel brightness, 0-255 (default 120)")
    parser.add_argument("--max-saturation", type=int, default=130,
                        help="Maximum ball-pixel colorfulness, 0-255 (default 130)")
    parser.add_argument("--min-change", type=float, default=30.0,
                        help="How different from the empty scene a pixel must be (default 30)")
    parser.add_argument("--show-mask", action="store_true",
                        help="Also show which pixels count as possible ball")
    parser.add_argument("--max-distance", type=float, default=math.inf,
                        help="Ignore anything farther than this, meters (default: no limit)")
    parser.add_argument("--output", default="ball_measurements.csv",
                        help="CSV file for saved points (default ball_measurements.csv)")
    parser.add_argument("--manual-box", action="store_true",
                        help="Don't find the black box automatically; draw it with k "
                             "(saved in --box-file)")
    parser.add_argument("--box-distance", type=float,
                        help="Camera-to-box-front-wall distance in meters (tape measure); "
                             "overrides the depth-measured value")
    parser.add_argument("--box-depth", type=float, default=0.15,
                        help="Box depth, front wall to back wall, meters (default 0.15)")
    parser.add_argument("--no-box-tape", action="store_true",
                        help="Accept any black box-sized blob as the box (no blue tape needed)")
    parser.add_argument("--tape-hue-min", type=int, default=TAPE["hue_min"],
                        help=f"Blue tape: lowest hue, 0-179 (default {TAPE['hue_min']})")
    parser.add_argument("--tape-hue-max", type=int, default=TAPE["hue_max"],
                        help=f"Blue tape: highest hue, 0-179 (default {TAPE['hue_max']})")
    parser.add_argument("--tape-sat-min", type=int, default=TAPE["sat_min"],
                        help=f"Blue tape: lowest colorfulness (default {TAPE['sat_min']})")
    parser.add_argument("--tape-value-min", type=int, default=TAPE["value_min"],
                        help=f"Blue tape: lowest brightness (default {TAPE['value_min']})")
    parser.add_argument("--capture-dir", default="box_captures",
                        help="Where the CAPTURE button saves frames (default box_captures)")
    parser.add_argument("--record-dir", default="recordings",
                        help="Where the RECORD button saves throws (default recordings)")
    parser.add_argument("--box-file", default="box_region.json",
                        help="Where the marked catch-box area is saved (default box_region.json)")
    parser.add_argument("--log", default="ball_track_log.csv",
                        help="CSV of every locked frame, overwritten each run "
                             "(default ball_track_log.csv)")
    return parser


def apply_settings(parser, args):
    """Copy color settings from the options into BALL/TAPE and check the options."""
    TAPE.update(hue_min=args.tape_hue_min, hue_max=args.tape_hue_max,
                sat_min=args.tape_sat_min, value_min=args.tape_value_min)
    BALL.update(loose_sat_min=args.loose_sat_min, loose_value_min=args.loose_value_min)
    BALL.update(color=args.ball, hue_min=args.hue_min, hue_max=args.hue_max, sat_min=args.sat_min,
                value_min=args.value_min)
    if not 0 <= args.brightness <= 255:
        parser.error("--brightness must be between 0 and 255")
    if not args.max_distance > MIN_DEPTH_M:
        parser.error(f"--max-distance must be more than {MIN_DEPTH_M} m")


def main():
    parser = build_parser()
    args = parser.parse_args()
    apply_settings(parser, args)
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        parser.error("this tool needs a graphical desktop for its preview window")

    try:
        import cv2
        import numpy as np
        import pyrealsense2 as rs
    except ImportError as exc:
        print(f"Camera dependency unavailable ({exc}). Install requirements-camera.txt.",
              file=sys.stderr)
        return 1

    try:
        with camera_session(rs, fps=args.fps, width=args.width, height=args.height) as (pipeline, profile):
            print_stream_profiles(rs, profile)
            with rgb_sensor_settings(rs, profile, args.rgb_exposure, args.rgb_gain,
                                     pipeline=pipeline):
                measure(rs, cv2, np, pipeline, profile, args)
    except KeyboardInterrupt:
        print("\nStopped.")
    except (RuntimeError, cv2.error) as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
