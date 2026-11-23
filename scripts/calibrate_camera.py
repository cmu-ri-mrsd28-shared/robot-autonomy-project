"""Fit a rigid transform from RealSense RGB optical coordinates to Panda base XYZ."""

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


SOURCE_FRAME = "realsense_rgb_optical"
TARGET_FRAME = "franka_base"
COORDINATES = ("camera_x", "camera_y", "camera_z", "base_x", "base_y", "base_z")


@dataclass(frozen=True)
class CameraToBase:
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self):
        r, t = self.rotation, self.translation
        if r.shape != (3, 3) or t.shape != (3,) or not np.isfinite(r).all() or not np.isfinite(t).all():
            raise ValueError("Calibration requires a finite 3x3 rotation and 3D translation")
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(
            np.linalg.det(r), 1.0, atol=1e-6, rtol=0
        ):
            raise ValueError("Calibration rotation must be orthonormal with determinant +1")

    def point(self, xyz):
        return self.rotation @ xyz + self.translation

    def vector(self, velocity):
        return self.rotation @ velocity


def fit_transform(camera, base):
    """Find the least-squares rigid transform, with no scale or reflection."""
    if camera.shape != base.shape or camera.ndim != 2 or camera.shape[1] != 3 or len(camera) < 3:
        raise ValueError("Supply at least three matching XYZ pairs")
    if not np.isfinite(camera).all() or not np.isfinite(base).all():
        raise ValueError("Calibration points must be finite")

    camera_center, base_center = camera.mean(axis=0), base.mean(axis=0)
    camera_offsets, base_offsets = camera - camera_center, base - base_center
    for points in (camera_offsets, base_offsets):
        spread = np.linalg.svd(points, compute_uv=False)
        if spread[1] <= max(1e-6, 1e-3 * spread[0]):
            raise ValueError("Fit points are repeated or nearly collinear. Spread them out.")

    u, _, vt = np.linalg.svd(camera_offsets.T @ base_offsets)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return CameraToBase(rotation, base_center - rotation @ camera_center)


def load_calibration(path, serial):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or data.get("source_frame") != SOURCE_FRAME or (
        data.get("target_frame") != TARGET_FRAME or data.get("translation_units") != "m"
    ):
        raise ValueError("Expected an RGB optical to Franka base calibration in meters")
    if data.get("camera_serial") != serial:
        raise ValueError(f"Calibration does not match RealSense serial {serial}")
    return CameraToBase(
        np.asarray(data.get("rotation"), dtype=float),
        np.asarray(data.get("translation_m"), dtype=float),
    )


def read_pairs(path):
    pairs = {"fit": [], "check": []}
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if not set((*COORDINATES, "set")).issubset(reader.fieldnames or []):
            raise ValueError("CSV must contain camera_x,camera_y,camera_z,base_x,base_y,base_z,set")
        for row in reader:
            try:
                coords = [float(row[name]) for name in COORDINATES]
                if not np.isfinite(coords).all() or coords[2] <= 0:
                    raise ValueError
                pairs[row["set"].strip()].append(coords)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid coordinates or set on CSV line {reader.line_num}") from exc
    if not pairs["check"]:
        raise ValueError("Include at least one separate check pair")
    return {name: np.asarray(rows, dtype=float).reshape(-1, 6) for name, rows in pairs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", type=Path, help="CSV with fit and check point pairs in meters")
    parser.add_argument("--serial", required=True, help="RealSense serial from the camera check")
    parser.add_argument("--output", required=True, type=Path, help="Calibration JSON path")
    args = parser.parse_args()
    if args.pairs.resolve() == args.output.resolve():
        parser.error("Use a separate path for the calibration JSON")
    try:
        pairs = read_pairs(args.pairs)
        fit = pairs["fit"]
        transform = fit_transform(fit[:, :3], fit[:, 3:])
        data = {
            "source_frame": SOURCE_FRAME,
            "target_frame": TARGET_FRAME,
            "camera_serial": args.serial.strip(),
            "translation_units": "m",
            "rotation": transform.rotation.tolist(),
            "translation_m": transform.translation.tolist(),
        }
        if not data["camera_serial"]:
            raise ValueError("RealSense serial cannot be blank")
        for name, rows in pairs.items():
            residuals = rows[:, :3] @ transform.rotation.T + transform.translation - rows[:, 3:]
            errors_mm = 1000 * np.linalg.norm(residuals, axis=1)
            data[f"{name}_error_mm"] = {
                "count": len(rows),
                "rms": float(np.sqrt(np.mean(errors_mm**2))),
                "max": float(errors_mm.max()),
            }
            print(f"{name}: {len(rows)} points | RMS {np.sqrt(np.mean(errors_mm**2)):.1f} mm"
                  f" | max {errors_mm.max():.1f} mm")
        args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, np.linalg.LinAlgError) as exc:
        parser.exit(1, f"Calibration error: {exc}\n")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
