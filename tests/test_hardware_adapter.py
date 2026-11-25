import sys, types, time
from types import SimpleNamespace
import numpy as np
import pytest
from catchbox.hardware.panda_kinematics import PandaKinematics
from catchbox.hardware.limiter import JointLimiter

OFFSET = [0.07978, -0.00136, 0.04041]
# Calibrations recorded on the robot with FrankaPy: (home joints, opening origin).
RECORDED = [
    ([0.014610958134497495, -0.29511025689359294, 0.007370139104326856, -2.645974195781507,
      0.0023260972205963394, 2.3559938603212127, 0.8005809835021694],
     [0.4869837070562427, 0.010546743739530892, 0.19795320227164812]),
    ([0.013983789247349436, -0.439660208865364, 0.00737974706135298, -2.497182068061193,
      0.0027697606689467877, 2.0605968483289083, 0.7999429896904363],
     [0.48659539638260424, 0.010536461494865357, 0.2970724773569217]),
]


@pytest.mark.parametrize('home,origin', RECORDED)
def test_kinematics_reproduce_recorded_calibrations(home, origin):
    np.testing.assert_allclose(PandaKinematics(OFFSET).position(home), origin, atol=1e-4)


def test_limiter_moves_in_a_straight_line_within_limits():
    start, goal = np.zeros(7), np.array([.3, 0, 0, -.2, 0, .2, 0])
    lim, dt, path = JointLimiter(start, .3, 1.0), .01, []
    for _ in range(400):
        path.append(lim.step(goal, dt))
    path = np.array(path)
    np.testing.assert_allclose(path[-1], goal, atol=1e-6)
    steps = np.diff(np.vstack([start, path]), axis=0) / dt
    assert np.max(np.abs(steps)) <= .3 + 1e-9
    assert np.max(np.abs(np.diff(steps, axis=0))) / dt <= 1.0 + 1e-6
    direction = goal / np.linalg.norm(goal)
    off_line = path - np.outer(path @ direction, direction)
    assert np.max(np.abs(off_line)) < 1e-9


# --- adapter planning against a fake robot and fake rospy -------------------
HOME, ORIGIN = RECORDED[1]
SETTINGS = {'calibration': dict(tip_link='panda_hand_tcp', opening_offset_m=OFFSET, origin_xyz_m=ORIGIN,
                                home_joints_rad=HOME, min_opening_z_m=.04),
            'adapter': dict(enable_motion=False, rate_hz=100, max_joint_velocity=1.0, max_joint_acceleration=4.0),
            'policy': dict(sim_start_opening_m=[0.5547317061010182, 0.0, 0.17904387158518817],
                           sim_workspace_center_m=[0.45723170610101843, 0.0, 0.09904387158518815],
                           sim_workspace_radius_m=0.2, action_scale_m=[.23, .23, .13], action_z_range_m=[-.13, .08])}


class FakeArm:
    def __init__(self, q=HOME): self.q = np.array(q)
    def get_joints(self): return self.q.copy()
    def is_skill_done(self, ignore_errors=True): return False


@pytest.fixture
def adapter(monkeypatch):
    fake = SimpleNamespace(Publisher=lambda *a, **k: SimpleNamespace(publish=lambda m: None),
                           Subscriber=lambda *a, **k: None, logwarn_throttle=lambda *a: None,
                           logerr=lambda *a: None)
    monkeypatch.setitem(sys.modules, 'rospy', fake)
    for mod, attr in [('geometry_msgs.msg', 'PointStamped'), ('sensor_msgs.msg', 'JointState')]:
        monkeypatch.setitem(sys.modules, mod, SimpleNamespace(**{attr: object}))
    monkeypatch.setitem(sys.modules, 'geometry_msgs', types.ModuleType('geometry_msgs'))
    monkeypatch.setitem(sys.modules, 'sensor_msgs', types.ModuleType('sensor_msgs'))
    from catchbox.hardware.adapter import Adapter
    return Adapter(SETTINGS, FakeArm())


def plan_to(adapter, goal):
    adapter.goal = adapter.space.clamp(goal)
    adapter.goal_time, adapter.input_fault = time.monotonic(), False
    adapter.plan_once()
    return adapter.kin.position(adapter.joint_goal)


def test_adapter_plans_a_small_left_move(adapter):
    goal = np.array(ORIGIN) + [0, .03, 0]
    assert np.linalg.norm(plan_to(adapter, goal) - goal) < .005
    np.testing.assert_allclose(adapter.joint_goal[[1, 2, 4, 6]], np.array(HOME)[[1, 2, 4, 6]])


def test_adapter_goes_to_nearest_reachable_point_instead_of_rejecting(adapter):
    goal = np.array(ORIGIN) + [0, 0, .1]          # straight up is off the joints 1/4/6 surface
    reached = plan_to(adapter, goal)
    assert np.linalg.norm(reached - np.array(ORIGIN)) > .02   # it moves toward the goal...
    assert adapter.space.contains(reached)                     # ...and stays in the task space


def test_targets_are_clamped_into_the_task_space(adapter):
    space = adapter.space
    np.testing.assert_allclose(np.linalg.norm(space.clamp(np.array(ORIGIN) + [0, .5, 0]) - ORIGIN), .2)
    assert space.clamp(np.array(ORIGIN) + [0, 0, -.19])[2] >= .04 - 1e-12
    assert space.clamp(np.array(ORIGIN) + [.1, 0, -.3])[2] >= .04 - 1e-12


def test_adapter_holds_when_target_goes_stale(adapter):
    plan_to(adapter, np.array(ORIGIN) + [0, .03, 0])
    adapter.goal_time -= 1.0
    adapter.plan_once()
    np.testing.assert_allclose(adapter.joint_goal, adapter.limiter.q)


def test_adapter_refuses_to_start_away_from_home(adapter):
    from catchbox.hardware.adapter import Adapter
    with pytest.raises(ValueError, match='not at home'):
        Adapter(SETTINGS, FakeArm(np.array(HOME) + .1))


def test_policy_frame_shifts_observations_and_targets():
    from catchbox.hardware.policy_frame import PolicyFrame
    frame = PolicyFrame(SETTINGS)
    np.testing.assert_allclose(frame.target([0, 0, 0]), ORIGIN)          # zero action = stay at home
    sim_center = np.array(SETTINGS['policy']['sim_workspace_center_m'])
    for a in ([1, 1, 1], [-1, 1, -1], [1, 0, 0]):              # clamped to the simulated workspace
        assert np.linalg.norm(frame.target(a) - frame.offset - sim_center) <= .2 + 1e-9
    obs = np.arange(25, dtype=np.float32) / 10
    robot_obs = obs.copy()
    for s in (slice(0, 3), slice(9, 12), slice(15, 18)):
        robot_obs[s] += frame.offset
    np.testing.assert_allclose(frame.observation(robot_obs), obs, atol=1e-6)
