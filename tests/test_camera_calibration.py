import csv
import json

import numpy as np
import pytest

from scripts.calibrate_camera import fit_transform, load_calibration, main


def test_calibration_cli_recovers_transform_and_checks_serial(tmp_path, monkeypatch):
    camera = np.array([
        [0, 0, 0.5], [0.2, 0, 0.5], [0, 0.3, 0.5], [0.1, 0.1, 0.8],
        [0.2, 0.2, 0.7],
    ])
    rotation = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    translation = np.array([0.15, 0.25, 0.85])
    base = camera @ rotation.T + translation
    pairs = tmp_path / "pairs.csv"
    output = tmp_path / "camera_to_base.json"
    with pairs.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow("camera_x camera_y camera_z base_x base_y base_z set".split())
        for index, (source, target) in enumerate(zip(camera, base)):
            writer.writerow([*source, *target, "fit" if index < 4 else "check"])
    monkeypatch.setattr("sys.argv", [
        "calibrate_camera.py", str(pairs), "--serial", "01234", "--output", str(output),
    ])
    assert main() == 0

    transform = load_calibration(output, "01234")
    np.testing.assert_allclose(transform.point(camera[-1]), base[-1], atol=1e-12)
    velocity = np.array([0.2, -0.8, 1.6])
    np.testing.assert_allclose(transform.vector(velocity), rotation @ velocity, atol=1e-12)
    data = json.loads(output.read_text())
    assert data["check_error_mm"]["count"] == 1
    assert data["check_error_mm"]["max"] < 1e-9

    with pytest.raises(ValueError, match="serial"):
        load_calibration(output, "99999")

    data["source_frame"] = "realsense_left_ir_optical"
    output.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="RGB optical"):
        load_calibration(output, "01234")


def test_fit_rejects_collinear_points():
    camera = np.array([[0, 0, 0.5], [0, 0, 0.8], [0, 0, 1.0]])
    with pytest.raises(ValueError, match="collinear"):
        fit_transform(camera, camera)
