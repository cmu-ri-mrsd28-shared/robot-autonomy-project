"""Recreate a recorded throw (measure_ball.py RECORD button) in the MuJoCo simulation.

  python3 scripts/recording_to_sim.py recordings/throw_YYYYMMDD_HHMMSS \\
      --calibration calibration/camera_to_base.json

What it does:
  1. Ball, every frame: the live tracker's sightings, plus the frames it missed.
     A thrown ball at 30 fps is an orange STREAK (motion blur), which the live
     tracker rightly ignores; here it is recovered from the saved video by linking
     orange blobs that moved from the frame before, frame to frame.
  2. Flight arcs: the path is split into free-flight arcs (the throw, then any
     bounce). Each arc is fitted in 3D from its pixel path and gravity, which fixes
     the scale without good depth (streak depth mixes the ball with the wall behind
     it), plus the depth readings where they are reliable.
  3. Box: placed where the camera saw it. If the ball went into the box, its last
     sighting fixes where the opening is (the ball's depth is good, the black box's
     isn't); otherwise the box outline and its width-based distance are used. There
     is no arm data, so the sim arm is posed by IK to put its box opening there,
     level, like the simulation's own catches.
  4. Simulation: the first arc (the release) starts the ball in MuJoCo, which then
     simulates the flight and any contact with the box while the arm holds still.
  5. Output, in the recording folder:
       sim_replay.mp4   real video | simulation seen from the real camera | side view
       sim_replay.json  ball path, arcs, box opening, release, real vs sim outcome

Assumptions: the box/arm don't move during the recording; the ball's spin and air
drag are ignored (as in the simulation's training).
"""
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

GRAVITY = 9.81
BOX_DEPTH_M = 0.15          # camera-facing box depth used by measure_ball (front to back)
W, H = 640, 480  # replaced by the recording's size in main()
# Recovering the missed frames from the video.
STREAK_HSV_LOW = (0, 120, 160)   # orange, allowing blur (paler); bright, unlike skin
STREAK_HSV_HIGH = (25, 255, 255)
STREAK_CHANGE = 30               # differs from the recording's median frame by this much
STREAK_MIN_AREA = 12
MAX_MISSES = 6                   # stop following after this many frames without the ball
# Fitting flight arcs.
ARC_MIN_POINTS = 5
ARC_MAX_RMS_PX = 5.0             # an arc must fit a gravity path this well (pixels)
ARC_MAX_GAP = 4                  # frames
DEPTH_SIGMA_TRACKER_M = 0.05     # how much to trust depth, compared with 1 px of position
DEPTH_SIGMA_ROUND_M = 0.10       # a round (not blurred) ball found in the video: fairly
DEPTH_SIGMA_STREAK_M = 0.5       # streak depth: barely (it mixes with the background)
ROUND_ASPECT = 1.4               # length/width below this = round, not a streak
ARC_MIN_TRAVEL_PX = 40           # a flight must cross at least this much of the image
ARC_SPEED_RANGE = (1.0, 6.0)     # m/s: a hand toss (the arm's swing alone fits faster)


def load(folder):
    meta = json.loads((folder / "meta.json").read_text())
    with (folder / "track.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    return meta, rows


def read_video(cv2, path):
    video, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, frame = video.read()
        if not ok:
            return frames
        frames.append(frame)


def orange_blobs(cv2, frame, background, depth_m):
    """Orange things that moved: [(u, v, area, depth_m or None, round)]."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    changed = np.abs(frame.astype(np.float32) - background).max(axis=2) >= STREAK_CHANGE
    mask = ((cv2.inRange(hsv, STREAK_HSV_LOW, STREAK_HSV_HIGH) > 0) & changed).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    blobs = []
    for i in range(1, n):
        if stats[i][4] < STREAK_MIN_AREA:
            continue
        inside = labels == i
        z = depth_m[inside]
        z = z[(z > 0.3) & (z < 6.0)]
        (_, _), (a, b), _ = cv2.minAreaRect(np.column_stack(np.nonzero(inside)[::-1])
                                           .astype(np.float32))
        is_round = max(a, b) <= ROUND_ASPECT * max(min(a, b), 1.0)
        blobs.append((float(centers[i][0]), float(centers[i][1]), int(stats[i][4]),
                      float(np.median(z)) if z.size >= 10 else None, is_round))
    return blobs


def ball_path(cv2, rows, frames, depth_m):
    """{frame: (u, v, camera_z or None, source)} - tracker sightings + recovered frames.

    source: "tracker", "round" (found in the video, not blurred) or "streak".
    """
    path = {}
    for r in rows:
        if r.get("ball_u"):
            z = float(r["cam_z_m"]) if r.get("source") == "depth" else None
            path[int(r["frame"])] = (float(r["ball_u"]), float(r["ball_v"]), z, "tracker")
    if not path or not frames:
        return path
    background = np.median(np.stack(frames[::3]), axis=0).astype(np.float32)
    misses, known = 0, sorted(path)
    for k in range(known[0] + 1, len(frames)):
        if k in path:
            misses = 0
            continue
        seen = sorted(f for f in path if f < k)
        if misses > MAX_MISSES or not seen:
            continue
        a = path[seen[-1]]
        step = (0.0, 0.0)
        if len(seen) > 1 and seen[-1] - seen[-2] <= ARC_MAX_GAP:
            b = path[seen[-2]]
            gap = seen[-1] - seen[-2]
            step = ((a[0] - b[0]) / gap, (a[1] - b[1]) / gap)
        ahead = k - seen[-1]
        pu, pv = a[0] + step[0] * ahead, a[1] + step[1] * ahead
        reach = max(50.0, 1.5 * math.hypot(*step)) * ahead
        near = [b for b in orange_blobs(cv2, frames[k], background, depth_m[k])
                if math.hypot(b[0] - pu, b[1] - pv) <= reach]
        if near:
            u, v, _, z, is_round = min(near, key=lambda b: math.hypot(b[0] - pu, b[1] - pv))
            path[k] = (u, v, z, "round" if is_round else "streak")
            misses = 0
        else:
            misses += 1
    return path


class Camera:
    """Projection between the Franka base frame and the image (pinhole, calibrated)."""

    def __init__(self, calibration, intrinsics):
        self.R, self.t = calibration.rotation, calibration.translation
        k = intrinsics
        self.fx, self.fy, self.ppx, self.ppy = k["fx"], k["fy"], k["ppx"], k["ppy"]

    def project(self, base):
        cam = (np.atleast_2d(base) - self.t) @ self.R   # = R^T (p - t), row-wise
        return (self.fx * cam[:, 0] / cam[:, 2] + self.ppx,
                self.fy * cam[:, 1] / cam[:, 2] + self.ppy, cam[:, 2])

    def to_base(self, u, v, z):
        cam = np.array([(u - self.ppx) / self.fx * z, (v - self.ppy) / self.fy * z, z])
        return self.R @ cam + self.t


def ballistic(p0, v0, dt):
    dt = np.asarray(dt, float)[:, None]
    return p0 + v0 * dt + np.array([0, 0, -0.5 * GRAVITY]) * dt**2


def fit_arc(camera, points, times):
    """Fit a free-flight arc to [(frame, u, v, z, source)]. Returns dict or None."""
    from scipy.optimize import least_squares

    t0 = times[points[0][0]]
    dt = np.array([times[p[0]] - t0 for p in points])
    uv = np.array([(p[1], p[2]) for p in points])
    sigma = {"tracker": DEPTH_SIGMA_TRACKER_M, "round": DEPTH_SIGMA_ROUND_M,
             "streak": DEPTH_SIGMA_STREAK_M}
    zs = [(i, p[3], sigma[p[4]]) for i, p in enumerate(points) if p[3] is not None]
    # Start from the points with depth (or 1.5 m), then refine on pixels + gravity.
    z_guess = np.median([z for _, z, _ in zs]) if zs else 1.5
    guess = np.array([camera.to_base(p[1], p[2], p[3] if p[3] is not None else z_guess)
                      for p in points])
    a = np.column_stack([np.ones_like(dt), dt])
    adjusted = guess.copy()
    adjusted[:, 2] += 0.5 * GRAVITY * dt**2
    (p0, v0), *_ = np.linalg.lstsq(a, adjusted, rcond=None)

    def residuals(x):
        u, v, z = camera.project(ballistic(x[:3], x[3:], dt))
        r = [u - uv[:, 0], v - uv[:, 1]]
        if zs:
            idx = np.array([i for i, _, _ in zs])
            r.append((z[idx] - np.array([z for _, z, _ in zs])) / np.array([s for _, _, s in zs]))
        return np.concatenate(r)

    fit = least_squares(residuals, np.r_[p0, v0], loss="soft_l1", f_scale=3.0)
    u, v, _ = camera.project(ballistic(fit.x[:3], fit.x[3:], dt))
    rms = float(np.sqrt(np.mean((u - uv[:, 0]) ** 2 + (v - uv[:, 1]) ** 2)))
    travel = float(np.hypot(*(uv.max(axis=0) - uv.min(axis=0))))
    return {"start_frame": points[0][0], "end_frame": points[-1][0], "t0": t0,
            "t1": times[points[-1][0]], "p0": fit.x[:3], "v0": fit.x[3:], "rms_px": rms,
            "points": len(points), "travel_px": travel}


def find_arcs(camera, path, times):
    """Split the ball path into free-flight arcs (greedy, earliest first)."""
    frames = sorted(path)
    points = [(f, *path[f]) for f in frames]
    arcs, i = [], 0
    while i <= len(points) - ARC_MIN_POINTS:
        best = None
        for j in range(i + ARC_MIN_POINTS, len(points) + 1):
            if points[j - 1][0] - points[j - 2][0] > ARC_MAX_GAP:
                break
            arc = fit_arc(camera, points[i:j], times)
            if arc["rms_px"] > ARC_MAX_RMS_PX:
                break
            best = arc
        if best is None:
            i += 1
            continue
        speed = float(np.linalg.norm(best["v0"]))
        # A ball held still also "fits" (gravity read as ~0.7 m/s up), and the arm's
        # swing can fit briefly: a flight must move across the image at a toss's speed.
        if (best["travel_px"] >= ARC_MIN_TRAVEL_PX
                and ARC_SPEED_RANGE[0] <= speed <= ARC_SPEED_RANGE[1]):
            arcs.append(best)
            i = frames.index(best["end_frame"])   # the next arc may start where this ends
        else:
            i += 1
    return arcs


def closest_approach(arc, point, horizon=1.0):
    """How near (m) an arc's continued free flight passes `point` within `horizon` s."""
    dt = np.linspace(0, (arc["t1"] - arc["t0"]) + horizon, 200)
    return float(np.linalg.norm(ballistic(arc["p0"], arc["v0"], dt) - point, axis=1).min())


def release_arc(arcs, opening, reach=0.30):
    """The throw: the first arc heading to the box (the arm's swing, ball still in the
    hand, can also fit an arc but goes elsewhere). Falls back to the longest arc."""
    for i, arc in enumerate(arcs):
        if closest_approach(arc, opening) <= reach:
            return i
    return max(range(len(arcs)), key=lambda i: arcs[i]["points"]) if arcs else None


def entry_frame(rows):
    for r in rows:
        if "INTO the box" in r.get("event", ""):
            return int(r["frame"])
    return None


def box_from_outline(rows, camera):
    boxes = [(float(r["box_u0"]), float(r["box_v0"]), float(r["box_u1"]), float(r["box_v1"]),
              float(r["box_front_m"])) for r in rows if r.get("box_front_m")]
    if not boxes:
        return None
    u0, v0, u1, v1, front = np.median(np.array(boxes), axis=0)
    return camera.to_base((u0 + u1) / 2, v0, front + BOX_DEPTH_M / 2)


def box_from_ball(path, entry, camera):
    """The ball's last depth sighting before it went into the box (None if it didn't)."""
    if entry is None:
        return None
    before = [(f, p) for f, p in sorted(path.items()) if f < entry and p[2] is not None]
    if not before:
        return None
    _, (u, v, z, _) = before[-1]
    return camera.to_base(u, v, z)


def camera_view(mujoco, calibration, fy, height=None):
    """A MuJoCo free camera at the real camera's pose (the free camera has no roll)."""
    forward = calibration.rotation[:, 2]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.5
    cam.lookat[:] = calibration.translation + forward * cam.distance
    cam.azimuth = math.degrees(math.atan2(forward[1], forward[0]))
    cam.elevation = math.degrees(math.asin(np.clip(forward[2], -1, 1)))
    return cam, 2 * math.degrees(math.atan((height or H) / 2 / fy))


def rounded(values, digits=4):
    return [round(float(x), digits) for x in values]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", type=Path, help="A recordings/throw_... folder")
    parser.add_argument("--calibration", default=str(ROOT / "calibration/camera_to_base.json"),
                        help="Camera-to-Franka-base calibration JSON")
    parser.add_argument("--opening", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="Box opening in the Franka base frame (m), e.g. measured from "
                             "another recording of the same box position; overrides the estimate")
    parser.add_argument("--extra", type=float, default=1.0,
                        help="Seconds of simulation to show after the recording ends (default 1)")
    args = parser.parse_args()

    import cv2
    import mujoco
    from calibrate_camera import load_calibration
    from catchbox.core import Config
    from catchbox.sim import Simulation

    folder = args.recording
    meta, rows = load(folder)
    global W, H
    W, H = int(meta["intrinsics"]["width"]), int(meta["intrinsics"]["height"])
    calibration = load_calibration(args.calibration, meta["camera_serial"])
    camera = Camera(calibration, meta["intrinsics"])
    frames = read_video(cv2, folder / "color.mp4")
    depth_m = np.load(folder / "depth.npz")["depth"].astype(np.float32) * float(
        meta.get("depth_units_m") or 0.001)
    times = {int(r["frame"]): float(r["t_s"]) for r in rows}
    fps = float(meta["fps"])

    path = ball_path(cv2, rows, frames, depth_m)
    if not path:
        sys.exit("No ball in this recording.")
    arcs = find_arcs(camera, path, times)
    entry = entry_frame(rows)
    from_ball, from_outline = box_from_ball(path, entry, camera), box_from_outline(rows, camera)
    opening = from_ball if from_ball is not None else from_outline
    if args.opening is not None:
        opening = np.array(args.opening, float)
    if opening is None:
        sys.exit("Neither the box nor a ball entry was seen: can't place the box.")
    release_index = release_arc(arcs, opening)
    release = None if release_index is None else arcs[release_index]

    sim = Simulation(Config(printed_box=True, duration=60.0))
    m, d = sim.model, sim.data
    goal, ik_error = sim.inverse_kinematics(np.asarray(opening, float))
    d.qpos[:7] = goal
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    placed = d.site_xpos[sim.site].copy()

    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W)
    m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    renderer = mujoco.Renderer(m, H, W)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[1] = 1
    opt.sitegroup[:] = 0
    real_cam, fovy = camera_view(mujoco, calibration, meta["intrinsics"]["fy"])
    m.vis.global_.fovy = fovy
    side = mujoco.MjvCamera()
    side.lookat[:] = placed
    side.distance, side.azimuth, side.elevation = 1.6, 120, -20
    # Measured ball positions to mark in the sim: arcs where fitted (reliable), and
    # tracker depth sightings elsewhere.
    marks = {}
    for arc in arcs:
        fs = [f for f in sorted(path) if arc["start_frame"] <= f <= arc["end_frame"]]
        pts = ballistic(arc["p0"], arc["v0"], [times[f] - arc["t0"] for f in fs])
        marks.update(zip(fs, pts))
    for f, (u, v, z, source) in path.items():
        if f not in marks and z is not None and source == "tracker":
            marks[f] = camera.to_base(u, v, z)

    def mark(scene, pos, rgba, size=0.008):
        if scene.ngeom < scene.maxgeom:
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([size, 0, 0]), np.asarray(pos, float),
                                np.eye(3).ravel(), np.asarray(rgba, np.float32))
            scene.ngeom += 1

    def render(view, upto):
        renderer.update_scene(d, view, opt)
        for f, p in marks.items():
            if f <= upto:
                mark(renderer.scene, p, [1, 1, 1, .7])   # the real ball's path so far
        mark(renderer.scene, opening, [0, 1, .3, 1], 0.01)  # where the opening was put
        return renderer.render()[:, :, ::-1]

    n_frames = len(rows)
    total = n_frames + int(args.extra * fps)
    out_path = folder / "sim_replay.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (3 * W, H + 40))
    steps_per_frame = max(1, round(1 / fps / m.opt.timestep))
    released, sim_in_box, last_real = False, False, np.zeros((H, W, 3), np.uint8)
    for k in range(total):
        t = times.get(k, times[n_frames - 1] + (k - n_frames + 1) / fps)
        if release is not None and not released and t >= release["t0"]:
            dt = t - release["t0"]
            d.qpos[sim.ball_q:sim.ball_q + 3] = ballistic(release["p0"], release["v0"], [dt])[0]
            d.qpos[sim.ball_q + 3:sim.ball_q + 7] = [1, 0, 0, 0]
            d.qvel[sim.ball_v:sim.ball_v + 3] = release["v0"] + [0, 0, -GRAVITY * dt]
            d.qvel[sim.ball_v + 3:sim.ball_v + 6] = 0
            released = True
        if released:
            for _ in range(steps_per_frame):
                sim.torque(goal)
                mujoco.mj_step(m, d)
        else:
            # Before release the sim ball follows the recording (in the hand).
            seen = [f for f in sorted(marks) if f <= k]
            d.qpos[sim.ball_q:sim.ball_q + 3] = marks[seen[-1]] if seen else [0, 0, -5]
            d.qvel[sim.ball_v:sim.ball_v + 6] = 0
            d.qpos[:7] = goal
            mujoco.mj_forward(m, d)
        local = d.qpos[sim.ball_q:sim.ball_q + 3] - d.site_xpos[sim.site]
        sim_in_box = bool(abs(local[0]) < 0.0675 and abs(local[1]) < 0.0675
                          and -0.10 < local[2] < 0.01)

        if k < len(frames):
            frame = frames[k].copy()
            if k in path:
                u, v, _, source = path[k]
                cv2.circle(frame, (int(u), int(v)), 10,
                           (70, 240, 80) if source == "tracker" else (0, 220, 255), 2)
            r = rows[k] if k < n_frames else {}
            if r.get("box_u0"):
                cv2.rectangle(frame, (int(r["box_u0"]), int(r["box_v0"])),
                              (int(r["box_u1"]), int(r["box_v1"])), (255, 170, 40), 2)
            last_real = frame
        image = np.hstack([last_real, render(real_cam, k), render(side, k)])
        bar = np.zeros((40, 3 * W, 3), np.uint8)
        label = (f"t {t:5.2f}s  {'REAL' if k < n_frames else 'after recording'}  |  "
                 f"green = live tracker, yellow = recovered from video  |  sim ball "
                 f"{'released (physics)' if released else 'follows recording'}, "
                 f"{'IN BOX' if sim_in_box else 'not in box'}")
        cv2.putText(bar, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(np.vstack([bar, image]))
    writer.release()

    summary = {
        "recording": str(folder), "frames": n_frames,
        "ball_frames": {"live_tracker": sum(1 for p in path.values() if p[3] == "tracker"),
                        "recovered_round": sum(1 for p in path.values() if p[3] == "round"),
                        "recovered_streak": sum(1 for p in path.values() if p[3] == "streak")},
        "arcs": [{"frames": [a["start_frame"], a["end_frame"]],
                  "time_s": [round(a["t0"], 3), round(a["t1"], 3)],
                  "start_position_m": rounded(a["p0"]), "start_velocity_mps": rounded(a["v0"], 3),
                  "speed_mps": round(float(np.linalg.norm(a["v0"])), 2),
                  "fit_rms_px": round(a["rms_px"], 2), "points": a["points"],
                  "passes_opening_within_m": round(closest_approach(a, opening), 3)}
                 for a in arcs],
        "release_arc": release_index,
        "real_ball_went_into_box": entry is not None,
        "opening_from_ball_m": None if from_ball is None else rounded(from_ball),
        "opening_from_box_outline_m": None if from_outline is None else rounded(from_outline),
        "opening_used_m": rounded(opening), "opening_given": args.opening is not None, "sim_opening_after_ik_m": rounded(placed),
        "ik_error_m": round(float(ik_error), 5), "sim_joints_rad": rounded(goal),
        "sim_ball_in_box_at_end": sim_in_box,
        "video": str(out_path),
    }
    (folder / "sim_replay.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
