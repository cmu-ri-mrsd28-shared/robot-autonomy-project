"""The ROS camera node's outputs match what the catching model expects.

No ROS or camera needed: the node's output logic is plain Python, and the model
side is checked with its own code (model_node.odom_values, build_steady_observation,
PolicyFrame, the steady-motion actor) and against the simulated camera the model
was trained with (CameraEstimator).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ball_vision_node as node  # noqa: E402
import measure_ball as mb  # noqa: E402
from calibrate_camera import CameraToBase  # noqa: E402
from steady_motion_sensing import CameraEstimator  # noqa: E402
from track_ball import Detection  # noqa: E402
from catchbox.hardware import model_node  # noqa: E402
from catchbox.hardware.calibration import BASE_LINK  # noqa: E402
from catchbox.hardware.observation import build_steady_observation  # noqa: E402
from catchbox.hardware.panda_kinematics import PandaKinematics  # noqa: E402
from catchbox.hardware.policy import Actor, STEADY_MOTION_ACTOR  # noqa: E402
from catchbox.hardware.policy_frame import PolicyFrame  # noqa: E402

# The lab calibration (calibration/camera_to_base.json, 21 poses).
CALIBRATION = CameraToBase(
    np.array([[-0.6222648249976477, -0.1911009131525057, -0.7591224727031376],
              [0.7786182108748099, -0.05091242556645543, -0.6254291379661248],
              [0.08087131299018867, -0.9802491346158989, 0.18047566268066204]]),
    np.array([2.082019801503156, 0.6250852440931923, 0.2623475226945383]))
SETTINGS = yaml.safe_load((ROOT / "catchbox/hardware/configuration.yaml").read_text())
FPS = 30.0


def odometry_stub():
    """An object shaped like nav_msgs/Odometry (only the fields used)."""
    vec = lambda: SimpleNamespace(x=0.0, y=0.0, z=0.0)  # noqa: E731
    return SimpleNamespace(
        header=SimpleNamespace(stamp=None, frame_id=""), child_frame_id="",
        pose=SimpleNamespace(pose=SimpleNamespace(position=vec(),
                                                  orientation=SimpleNamespace(x=0., y=0., z=0., w=0.))),
        twist=SimpleNamespace(twist=SimpleNamespace(linear=vec(), angular=vec())))


def throw(n=12):
    """A ball flying toward the box opening and falling: base-frame positions."""
    start = np.array(SETTINGS["calibration"]["origin_xyz_m"]) + [0.30, 0.05, 0.45]
    velocity = np.array([-0.9, -0.1, 0.4])
    times = np.arange(n) / FPS
    return times, [start + velocity * t + [0, 0, -0.5 * 9.81 * t * t] for t in times]


def locked_lock():
    lock = mb.BallLock()
    lock.state, lock.ball_id = "LOCKED", 1
    return lock


def measured(base_position):
    """What the tracker measures: the same point in the camera frame."""
    camera = CALIBRATION.rotation.T @ (np.asarray(base_position) - CALIBRATION.translation)
    return mb.Measurement(Detection(320, 240, 8.0), camera, "depth")


def box_state():
    kin = PandaKinematics(SETTINGS["calibration"]["opening_offset_m"])
    return kin.position(SETTINGS["calibration"]["home_joints_rad"]), np.zeros(3), np.zeros(3)


def test_status_and_in_box_follow_the_documented_codes():
    found = measured([0.5, 0.0, 0.5])
    lock = locked_lock()
    assert node.camera_status(lock, found, 0.0) == (node.VISIBLE, node.OUTSIDE)
    # Locked ball not seen this frame / lost: active but hidden, in-box unknown.
    assert node.camera_status(lock, None, 0.0) == (node.HIDDEN, node.UNKNOWN)
    lock.state = "LOST"
    assert node.camera_status(lock, None, 0.0) == (node.HIDDEN, node.UNKNOWN)
    # It vanished into the box: hidden, inside (camera evidence).
    lock.state, lock.in_box_balls = "SEARCHING", [{"id": 1}]
    assert node.camera_status(lock, None, 0.0) == (node.HIDDEN, node.INSIDE)
    # Searching, or only a would-be ball: no active ball.
    for state in ("SEARCHING", "LOCKING"):
        lock = mb.BallLock()
        lock.state = state
        assert node.camera_status(lock, None, 0.0) == (node.NO_BALL, node.UNKNOWN)
    # Seen inside the box outline at the box's distance: inside.
    lock = locked_lock()
    lock.box, lock.box_front_m = (300, 200, 340, 280), 0.45
    inside = mb.Measurement(Detection(320, 240, 8.0), np.array([0.0, 0.0, 0.50]), "depth")
    assert node.camera_status(lock, inside, 0.0) == (node.VISIBLE, node.INSIDE)


def test_codes_are_the_ones_model_node_accepts():
    assert {node.NO_BALL, node.HIDDEN, node.VISIBLE, node.TRACKED} == {-1, 0, 1, 2}
    assert {node.UNKNOWN, node.OUTSIDE, node.INSIDE} == {-1, 0, 1}
    assert node.BASE_LINK == BASE_LINK


def test_velocity_matches_the_training_camera_estimator():
    times, positions = throw()
    ours = node.BallisticVelocity(np)
    sim = CameraEstimator(noise_m=0.0)
    box_p, _, _ = box_state()
    for t, p in zip(times, positions):
        v = ours.update(t, p)
        expected = sim.update(t, p, box_p, np.eye(3), 0.135, 0.135, 0.02)
        if v is None:
            assert not expected["tracking_valid"]
        else:
            assert expected["tracking_valid"]
            np.testing.assert_allclose(v, expected["velocity"], atol=1e-9)
    # For a ball in free fall the fit has no lag: it's the true latest velocity.
    true_v = np.array([-0.9, -0.1, 0.4 - 9.81 * times[-1]])
    np.testing.assert_allclose(v, true_v, atol=1e-6)


def test_published_ball_gives_the_same_25_inputs_as_in_training():
    frame = PolicyFrame(SETTINGS)
    actor = Actor(STEADY_MOTION_ACTOR)
    box_p, box_v, box_w = box_state()
    output = node.BallOutput(np, CALIBRATION)
    sim = CameraEstimator(noise_m=0.0)
    lock = locked_lock()
    times, positions = throw()
    tracked = predicted = 0
    for t, p in zip(times, positions):
        out = output.update(lock, measured(p), t, t)
        # Publish into an Odometry-shaped message and read it back like model_node.
        msg = node.fill_odometry(odometry_stub(), "stamp", out["position"], out["velocity"])
        assert msg.header.frame_id == BASE_LINK and msg.header.stamp == "stamp"
        position, velocity, _ = model_node.odom_values(msg)
        np.testing.assert_allclose(position, p, atol=1e-9)   # calibration round trip
        obs = build_steady_observation(frame, box_p, box_v, box_w, position, velocity,
                                       out["status"], out["in_box"])
        # The training camera on the same ball (visible, outside the box).
        expected = sim.update(t, p, box_p, np.eye(3), 0.135, 0.135, 0.02)
        status = node.TRACKED if expected["tracking_valid"] else node.VISIBLE
        assert out["status"] == status and out["in_box"] == node.OUTSIDE
        reference = build_steady_observation(frame, box_p, box_v, box_w, expected["position"],
                                             expected["velocity"], status, 0)
        assert obs.shape == (25,) and np.isfinite(obs).all() and np.abs(obs).max() <= 10
        np.testing.assert_allclose(obs, reference, atol=1e-6)
        assert obs[22] == (1.0 if status == node.TRACKED else 0.0) and obs[24] == 0.0
        action = actor.predict(obs)
        assert action.shape == (3,) and np.isfinite(frame.target(action)).all()
        tracked += status == node.TRACKED
        predicted += obs[23] == 1.0
    assert tracked == len(times) - 2   # velocity valid from the 3rd sighting on
    assert predicted > 0                # while above the opening, a catch point is predicted


def test_hidden_and_in_box_inputs():
    frame = PolicyFrame(SETTINGS)
    box_p, box_v, box_w = box_state()
    output = node.BallOutput(np, CALIBRATION)
    lock = locked_lock()
    output.update(lock, measured([0.5, 0.0, 0.5]), 0.0, 0.0)
    # The ball went into the box: hidden + inside. model_node runs inference for
    # status 0, with the ball zeroed and the in-box input set.
    lock.state, lock.in_box_balls = "SEARCHING", [{"id": 1}]
    out = output.update(lock, None, 0.1, 0.1)
    assert (out["status"], out["in_box"]) == (node.HIDDEN, node.INSIDE)
    assert out["position"] is None and out["velocity"] is None
    obs = build_steady_observation(frame, box_p, box_v, box_w, None, None,
                                   out["status"], out["in_box"])
    np.testing.assert_allclose(obs[9:22], 0)
    assert obs[22:25].tolist() == [0.0, 0.0, 1.0]
    # After a hide the velocity fit restarts (as in training): VISIBLE first.
    lock.state, lock.in_box_balls = "LOCKED", []
    assert output.update(lock, measured([0.5, 0.0, 0.5]), 0.2, 0.2)["status"] == node.VISIBLE


@pytest.mark.parametrize("domain_ok, age, expected_age", [(True, 0.03, 0.03), (True, 2.0, 0.0),
                                                          (False, 0.03, 0.0)])
def test_capture_time_stamp(domain_ok, age, expected_age):
    rs = SimpleNamespace(timestamp_domain=SimpleNamespace(global_time="global"))
    wall = 1000.0
    frame = SimpleNamespace(get_frame_timestamp_domain=lambda: "global" if domain_ok else "hw",
                            get_timestamp=lambda: 1000.0 * (wall - age))
    assert node.capture_time_s(rs, frame, wall) == pytest.approx(wall - expected_age)


def test_ball_output_rejects_position_spike_and_restarts_velocity():
    output = node.BallOutput(np, CALIBRATION)
    lock = locked_lock()
    base=np.array([0.5,0.,0.5])
    for i in range(3):
        result=output.update(lock,measured(base+np.array([i*.01,0,0])),i/30.,i/30.)
    assert result['status']==node.TRACKED
    # A one-frame depth/size mismatch must not be published as a real location.
    out=output.update(lock,measured(base+[1.,0,0]),.1,.1)
    assert out['status']==node.HIDDEN and out['in_box']==node.UNKNOWN
    assert out['position'] is None and out['velocity'] is None
    assert out['rejection']=='position_jump'
    # The next plausible point is accepted; velocity must be rebuilt from scratch.
    out=output.update(lock,measured(base+[.03,0,0]),4/30.,4/30.)
    assert out['status']==node.VISIBLE and out['velocity'] is None
    out=output.update(lock,measured(base+[.04,0,0]),5/30.,5/30.)
    assert out['status']==node.VISIBLE
    out=output.update(lock,measured(base+[.05,0,0]),6/30.,6/30.)
    assert out['status']==node.TRACKED


def test_fast_plausible_ball_is_not_rejected():
    output=node.BallOutput(np,CALIBRATION);lock=locked_lock()
    statuses=[]
    for i in range(6):
        out=output.update(lock,measured([.5+3.*i/30.,0,.5]),i/30.,i/30.)
        statuses.append(out['status'])
    assert statuses[-1]==node.TRACKED
    assert all(x!=node.HIDDEN for x in statuses)


def test_recorded_depth_size_jump_is_not_published():
    import csv
    path=ROOT/'data/videos/throw_20261125_183141/track.csv'
    if not path.is_file():pytest.skip('optional recorded flight not present')
    rows=list(csv.DictReader(path.open()))
    output=node.BallOutput(np,CALIBRATION);lock=locked_lock();lock.ball_id=8
    results={}
    for row in rows[52:62]:
        if not row['cam_x_m']:continue
        t=float(row['t_s'])
        found=mb.Measurement(Detection(int(row['ball_u']),int(row['ball_v']),float(row['ball_r_px'])),
               np.array([float(row['cam_x_m']),float(row['cam_y_m']),float(row['cam_z_m'])]),row['source'])
        results[int(row['frame'])]=output.update(lock,found,t,t)
    assert results[58]['position'] is not None
    assert results[59]['rejection']=='position_jump' and results[59]['position'] is None
    assert results[60]['position'] is not None and results[60]['velocity'] is None
