import importlib
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
