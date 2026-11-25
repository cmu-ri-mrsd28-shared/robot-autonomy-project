import importlib
import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def live_tool(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("calibrate_camera_live")


@pytest.fixture
def calibration_samples():
    rng = np.random.default_rng(174)
    camera = rng.uniform([-.2, -.2, .6], [.2, .2, .95], size=(12, 3))
    rotation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    translation = np.array([-.2, .1, .7])
    offset = np.array([.03, .04, .12])

    def make(vary_axes):
        poses = []
        for index, point in enumerate(camera):
            angle = index * .2
            cos, sin = np.cos(angle), np.sin(angle)
            wrist = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
            if vary_axes and index % 2:
                wrist = np.roll(wrist, shift=(1, 1), axis=(0, 1))
            pose = np.eye(4)
            pose[:3, :3] = wrist
            pose[:3, 3] = rotation @ point + translation - wrist @ offset
            poses.append(pose)
        return camera, np.array(poses), rotation, translation, offset

    return make


def test_same_axis_rotation_requires_measured_offset(live_tool, calibration_samples):
    camera, poses, rotation, translation, offset = calibration_samples(vary_axes=False)
    with pytest.raises(ValueError, match="different axes"):
        live_tool.solve(camera, poses)
    solved_r, solved_t, _, _ = live_tool.solve(camera, poses, offset)
    np.testing.assert_allclose(solved_r, rotation, atol=1e-10)
    np.testing.assert_allclose(solved_t, translation, atol=1e-10)


def test_varied_wrist_axes_recover_camera_and_offset(live_tool, calibration_samples):
    camera, poses, rotation, translation, offset = calibration_samples(vary_axes=True)
    solved_r, solved_t, solved_offset, _ = live_tool.solve(camera, poses)
    np.testing.assert_allclose(solved_r, rotation, atol=1e-10)
    np.testing.assert_allclose(solved_t, translation, atol=1e-10)
    np.testing.assert_allclose(solved_offset, offset, atol=1e-10)
    assert live_tool.check_errors(camera, poses).max() < 1e-9


@pytest.mark.parametrize(("line", "expected"), [("\n", ""), ("", "quit")])
def test_terminal_end_of_input_exits(live_tool, monkeypatch, line, expected):
    monkeypatch.setattr("sys.stdin", StringIO(line))
    args = SimpleNamespace(preview=False)
    assert live_tool.wait_for_line(None, None, None, None, args, "prompt") == expected



@pytest.mark.parametrize(
    ("topic", "message_type"),
    [
        ("/franka_state_controller/franka_states", "franka_msgs/FrankaState"),
        ("/robot_state_publisher_node_1/robot_state", "franka_interface_msgs/RobotState"),
    ],
)
def test_ros_pose_reads_the_advertised_state_type(live_tool, monkeypatch, topic, message_type):
    pose = np.eye(4)
    pose[:3, 3] = [0.42, -0.03, 0.34]
    msg_type = type("ArmState", (), {})
    seen = []

    def message_class(name):
        seen.append(name)
        return msg_type

    def wait_for_message(name, cls, timeout):
        assert (name, cls, timeout) == (topic, msg_type, 2.0)
        return SimpleNamespace(O_T_EE=pose.flatten(order="F"))

    monkeypatch.setitem(sys.modules, "rospy", SimpleNamespace(
        init_node=lambda *args, **kwargs: None,
        resolve_name=lambda name: name,
        get_published_topics=lambda: [(topic, message_type)],
        wait_for_message=wait_for_message,
    ))
    monkeypatch.setitem(sys.modules, "roslib", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "roslib.message", SimpleNamespace(
        get_message_class=message_class,
    ))
    np.testing.assert_array_equal(live_tool.RosPose(topic).read(), pose)
    assert seen == [message_type]



@pytest.mark.parametrize(("ball_color", "expected_x"), [("white", 40), ("orange", 110)])
def test_calibration_detector_selects_ball_color(live_tool, ball_color, expected_x):
    cv2 = pytest.importorskip("cv2")
    image = np.full((100, 160, 3), 45, dtype=np.uint8)
    cv2.circle(image, (40, 50), 10, (255, 255, 255), -1)
    cv2.circle(image, (110, 50), 10, (0, 165, 255), -1)
    detections = live_tool.find_still_ball(cv2, image, 170, ball_color)
    assert [(ball.u, ball.v) for ball in detections] == [(expected_x, 50)]


@pytest.mark.parametrize(("first_ready", "expected_timeout"), [(False, 6000), (True, 2000)])
def test_frame_timeout_skips_spot(live_tool, monkeypatch, first_ready, expected_timeout):
    requested = []

    def wait(timeout_ms):
        requested.append(timeout_ms)
        return first_ready and len(requested) == 1, object()

    monkeypatch.setattr(live_tool, "locate", lambda *args: (np.array([0, 0, 1]), None, None))
    args = SimpleNamespace(average_s=1.0, preview=False, brightness=170, ball_color="orange")
    pipeline = SimpleNamespace(try_wait_for_frames=wait)
    point, note = live_tool.average_ball(None, None, pipeline, None, args)
    assert point is None
    assert "retry this spot" in note
    assert requested[-1] == expected_timeout
