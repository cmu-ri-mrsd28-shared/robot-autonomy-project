"""Replay a ROS pipeline run (docker/replay_pipeline.py) in MuJoCo.

  python3 scripts/ros_run_to_sim.py ros_runs/throw_... --recording recordings/throw_...

The sim arm follows the joint angles the adapter commanded (recorded on
/catchbox_command_adapter/joint_target), and the ball is drawn where the vision
node published it (/catchbox/ball_state) while its status was visible. The
model's opening target is the green dot. Output, in the run folder:
  ros_replay.mp4   real video + what the vision node said | sim from the real
                   camera's position | side view
  ros_replay.json  what was sent: statuses, inferences, targets, joint motion
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

W, H = 640, 480  # replaced by the recording's size in main()
STATUS = {-1: "NO BALL", 0: "HIDDEN", 1: "VISIBLE", 2: "TRACKED"}
IN_BOX = {-1: "unknown", 0: "outside", 1: "INSIDE"}


def latest(series, t):
    """Last (time, value) at or before t in a time-sorted list, else None."""
    times = [s[0] for s in series]
    i = np.searchsorted(times, t, side="right") - 1
    return series[i] if i >= 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="ros_runs/<recording name> folder")
    parser.add_argument("--recording", type=Path, required=True,
                        help="The recordings/<name> folder that was replayed")
    parser.add_argument("--calibration", default=str(ROOT / "calibration/camera_to_base.json"))
    parser.add_argument("--extra", type=float, default=1.5,
                        help="Seconds to show after the last frame (default 1.5)")
    args = parser.parse_args()

    import cv2
    import mujoco
    from calibrate_camera import load_calibration
    from catchbox.core import Config
    from catchbox.sim import Simulation
    from recording_to_sim import camera_view, read_video

    rows = [json.loads(line) for line in (args.run / "ros_messages.jsonl").read_text().splitlines()]
    by_topic = {}
    for r in rows:
        by_topic.setdefault(r["topic"], []).append(r)
    for series in by_topic.values():
        series.sort(key=lambda r: r["ros_time"])
    status = [(r["ros_time"], r["data"]) for r in by_topic.get("/catchbox/ball_tracking_status", [])]
    in_box = [(r["ros_time"], r["data"]) for r in by_topic.get("/catchbox/ball_in_box", [])]
    if not status:
        sys.exit("No vision messages in this run.")
    joints = [(r["ros_time"], np.array(r["position"])) for r in
              by_topic.get("/catchbox_command_adapter/joint_target", [])]
    balls = [(r["ros_time"], np.array(r["position"])) for r in by_topic.get("/catchbox/ball_state", [])]
    targets = [(r["ros_time"], np.array(r["point"])) for r in
               by_topic.get("/catchbox_command_adapter/opening_target", [])]
    home = [(r["ros_time"], np.array(r["position"])) for r in by_topic.get("/catchbox/robot_joints", [])]
    # The vision node publishes one status per video frame, in order.
    frame_times = [t for t, _ in status]
    meta = json.loads((args.recording / "meta.json").read_text())
    global W, H
    W, H = int(meta["intrinsics"]["width"]), int(meta["intrinsics"]["height"])
    frames = read_video(cv2, args.recording / "color.mp4")
    calibration = load_calibration(args.calibration, meta["camera_serial"])
    fps = float(meta["fps"])

    sim = Simulation(Config(printed_box=True, duration=60.0))
    m, d = sim.model, sim.data
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W)
    m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    renderer = mujoco.Renderer(m, H, W)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[1] = 1
    opt.sitegroup[:] = 0
    real_cam, fovy = camera_view(mujoco, calibration, meta["intrinsics"]["fy"], H)
    m.vis.global_.fovy = fovy
    side = mujoco.MjvCamera()
    side.lookat[:] = [0.5, 0.0, 0.35]
    side.distance, side.azimuth, side.elevation = 1.8, 120, -20

    def mark(scene, pos, rgba, size):
        if scene.ngeom < scene.maxgeom:
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([size, 0, 0]), np.asarray(pos, float),
                                np.eye(3).ravel(), np.asarray(rgba, np.float32))
            scene.ngeom += 1

    def render(view, t, target):
        renderer.update_scene(d, view, opt)
        for bt, p in balls:
            if bt <= t:
                mark(renderer.scene, p, [1, 1, 1, .6], 0.007)   # published ball path so far
        if target is not None:
            mark(renderer.scene, target, [0, 1, .3, 1], 0.012)  # the model's opening target
        return renderer.render()[:, :, ::-1]

    out_path = args.run / "ros_replay.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (3 * W, H + 70))
    total = len(frame_times) + int(args.extra * fps)
    first_q = joints[0][1] if joints else (home[0][1] if home else sim.home)
    openings = []
    for k in range(total):
        t = frame_times[k] if k < len(frame_times) else frame_times[-1] + (k - len(frame_times) + 1) / fps
        q = latest(joints, t)
        d.qpos[:7] = q[1] if q is not None else first_q
        s, b = latest(status, t), latest(in_box, t)
        s, b = (s[1] if s else -1), (b[1] if b else -1)
        ball = latest(balls, t)
        visible = s > 0 and ball is not None and t - ball[0] < 0.1
        d.qpos[sim.ball_q:sim.ball_q + 3] = ball[1] if visible else [0, 0, -5]
        mujoco.mj_forward(m, d)
        openings.append(d.site_xpos[sim.site].copy())
        target = latest(targets, t)
        target = target[1] if target is not None and t - target[0] < 0.2 else None

        real = frames[min(k, len(frames) - 1)].copy() if frames else np.zeros((H, W, 3), np.uint8)
        cv2.putText(real, f"vision node: {STATUS[s]}, in box {IN_BOX[b]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (70, 240, 80) if s > 0 else (0, 200, 255), 2)
        image = np.hstack([real, render(real_cam, t, target), render(side, t, target)])
        bar = np.zeros((70, 3 * W, 3), np.uint8)
        line1 = (f"t {t - frame_times[0]:5.2f}s  |  ROS: vision node -> model_node -> adapter "
                 f"-> joints (MuJoCo follows the commanded joints)")
        line2 = ("model target: " + ("waiting (no fresh ball/status)" if target is None else
                 f"opening at x {target[0]:+.3f} y {target[1]:+.3f} z {target[2]:+.3f} m") +
                 f"  |  joints {np.round(d.qpos[:7], 2).tolist()}")
        cv2.putText(bar, line1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(bar, line2, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 1)
        writer.write(np.vstack([bar, image]))
    writer.release()

    model_log = [json.loads(line) for line in (args.run / "model.jsonl").read_text().splitlines()
                 if line.strip()] if (args.run / "model.jsonl").exists() else []
    inferences = [r for r in model_log if r.get("event") == "inference"]
    q_all = np.array([q for _, q in joints]) if joints else np.zeros((0, 7))
    summary = {
        "frames": len(frame_times),
        "vision_status_frames": {STATUS[v]: sum(1 for _, x in status if x == v) for v in STATUS},
        "in_box_frames": {IN_BOX[v]: sum(1 for _, x in in_box if x == v) for v in IN_BOX},
        "ball_state_messages": len(balls),
        "model_inferences": len(inferences),
        "model_inferences_by_status": {STATUS[v]: sum(1 for r in inferences
                                                      if r.get("tracking_status") == v)
                                       for v in STATUS},
        "model_waits": sorted({r.get("reason") for r in model_log if r.get("event") == "input_wait"}),
        "opening_targets": len(targets),
        "target_range_m": None if not targets else {
            "min": np.round(np.min([p for _, p in targets], axis=0), 3).tolist(),
            "max": np.round(np.max([p for _, p in targets], axis=0), 3).tolist()},
        "joint_commands": len(joints),
        "joint_range_rad": None if not len(q_all) else
            np.round(q_all.max(axis=0) - q_all.min(axis=0), 3).tolist(),
        "sim_opening_range_m": np.round(np.ptp(np.array(openings), axis=0), 3).tolist(),
        "video": str(out_path),
    }
    (args.run / "ros_replay.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
