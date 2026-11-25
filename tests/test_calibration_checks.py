import sys, types
from types import SimpleNamespace
import numpy as np
import pytest
from catchbox.hardware import calibration as cal

R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
CONFIG = dict(tip_link='tool', opening_offset_m=[.1, 0, 0],
              origin_xyz_m=[.5, 0, .2], min_opening_z_m=0.0, home_joints_rad=[0, -.3, 0, -2.6, 0, 2.3, .8])


class FakeArm:
    """Records commanded opening positions; home joints put the opening at the center."""
    def __init__(self):
        self.pose = SimpleNamespace(translation=np.array([.4, .1, .3]), rotation=np.eye(3),
                                    from_frame='franka_tool', to_frame='world')
        self.openings, self.durations = [], []
    def get_pose(self): return self.pose
    def goto_joints(self, joints, duration, ignore_errors):
        assert ignore_errors is False
        self.pose = SimpleNamespace(translation=np.array(CONFIG['origin_xyz_m']) - R @ CONFIG['opening_offset_m'],
                                    rotation=R, from_frame='franka_tool', to_frame='world')
    def goto_pose(self, pose, duration, use_impedance, ignore_errors):
        assert ignore_errors is False and np.allclose(pose.rotation, R)
        self.pose = pose
        self.openings.append(pose.translation + pose.rotation @ CONFIG['opening_offset_m'])
        self.durations.append(duration)


@pytest.fixture(autouse=True)
def fake_autolab(monkeypatch):
    class RigidTransform(SimpleNamespace):
        pass
    monkeypatch.setitem(sys.modules, 'autolab_core', types.SimpleNamespace(RigidTransform=RigidTransform))


def run(check, **kw):
    args = SimpleNamespace(max_distance=.2, directions=None, count=10, seed=0, **kw)
    arm = FakeArm()
    check(cal.BoxArm(CONFIG, arm), args)
    return np.array(arm.openings), arm


def test_check2_goes_3cm_each_way_and_back():
    openings, _ = run(cal.check2_small)
    distances = np.linalg.norm(openings - CONFIG['origin_xyz_m'], axis=1)
    np.testing.assert_allclose(distances, [.03, 0] * 6, atol=1e-12)
    np.testing.assert_allclose(openings[0], [.5, .03, .2])  # left is +Y


def test_check3_steps_outward_and_respects_limits():
    openings, arm = run(cal.check3_steps)
    distances = np.linalg.norm(openings - CONFIG['origin_xyz_m'], axis=1)
    np.testing.assert_allclose(distances[::2], [.05, .1, .15, .2] * 6, atol=1e-12)
    assert min(arm.durations) >= cal.MIN_DURATION
    args = SimpleNamespace(max_distance=.1, directions=['up'], count=1, seed=0)
    arm = FakeArm(); cal.check3_steps(cal.BoxArm(CONFIG, arm), args)
    np.testing.assert_allclose(np.array(arm.openings)[::2], [[.5, 0, .25], [.5, 0, .3]])


def test_check4_points_stay_inside_sphere():
    openings, _ = run(cal.check4_catch)
    assert len(openings) == 11
    assert np.all(np.linalg.norm(openings - CONFIG['origin_xyz_m'], axis=1) <= .2)


def test_floor_limit_skips_low_steps():
    config = dict(CONFIG, min_opening_z_m=.08)
    arm = FakeArm()
    cal.check3_steps(cal.BoxArm(config, arm), SimpleNamespace(max_distance=.2, directions=['down'], count=1, seed=0))
    np.testing.assert_allclose(np.array(arm.openings)[::2, 2], [.15, .1])  # 15 and 20 cm skipped
    with pytest.raises(ValueError, match='min_opening_z_m'):
        box = cal.BoxArm(config, FakeArm()); box.go_home(); box.move_to([.5, 0, .05], 'low')


def test_move_rejects_points_outside_sphere_and_requires_home():
    box = cal.BoxArm(CONFIG, FakeArm())
    with pytest.raises(RuntimeError):
        box.move_to(CONFIG['origin_xyz_m'], 'center')
    box.go_home()
    with pytest.raises(ValueError, match='outside'):
        box.move_to([.5, .21, .2], 'too far')
