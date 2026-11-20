# Franka ball-catching simulation

A MuJoCo / robosuite simulation of a Franka catching a table-tennis ball in a 3D-printed box. The gripper holds the box handle; a ballistic predictor and inverse-kinematics controller are augmented by a CrossQ residual policy.

**Robot control runs in simulation.** The standalone RealSense D435 tracker reports ball position and velocity in camera coordinates. Connecting those estimates to a physical Franka requires camera-to-base calibration and a hardware control adapter. Use the simulated torque controller only in MuJoCo.

## Latest model and results

- `artifacts/original-model/box/final_model.zip`: original CrossQ model trained for 300,000 online transitions; initialized from scratch.
- `artifacts/original-model/report.html`: offline report; keep `media/` beside it.
- `artifacts/original-model/box/`: evaluation, held-out throws, validation, training logs, and configuration. Replay buffer is intentionally omitted.
- On 50 held-out throws, the learned policy retained 36/50; the scripted controller retained 32/50 on the same throws. Both got the ball into the box on the same 36 throws. The 1 mm contact tolerance is described in the report.

The source STL is `assets/armbox-original.STL`; `assets/armBox-inmeters.STL` is the transformed, meter-scale mesh used by the simulator. The outer box is 15 × 15 × 10 cm, with about a 12 × 12 cm opening and a handle. The mesh is used for rendering; five rectangular pieces approximate the collision cavity. The gripper holds the handle rigidly without slip, and total box-plus-handle mass is assumed to be 250 g. Ball position, velocity and acceleration are taken directly from the simulator.

## Install and test

Requires Python 3.11 or 3.12. From this directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[training]'
python -m pip install pytest pillow
python -m pytest -q
```

The offline video renderer also requires `ffmpeg` on `PATH`. Lock files record packages from the original environment and may require adaptation for a different platform.

## Camera tracking

Use `check_camera.py` to inspect the RealSense streams and `track_ball.py` to track a white table-tennis ball. The [camera implementation guide](docs/CAMERA_IMPLEMENTATION.md) describes the planned calibration and controller integration.

### RealSense D435 terminal check

Use Python 3.11 or 3.12 in a camera-only virtual environment. Connect the camera
to a USB 3 port and run these commands from the repository root:

```sh
python3 -m venv ~/.venvs/catchbox-camera
source ~/.venvs/catchbox-camera/bin/activate
python -m pip install -r requirements-camera.txt
python scripts/check_camera.py --seconds 10
python scripts/check_camera.py --preview
```

The script prints the device and active stream profiles, then reports paired
frame rate and median depth in a 5 × 5 patch at the center of the RGB image.
Point the camera at a flat target and check that the reading follows a known
change in distance. Zero depth pixels are skipped. RGB–depth timestamp skew is
reported when both frames use the SDK's global timestamp domain. Omit
`--seconds` to run until Ctrl+C.

With `--preview`, a window shows RGB on the left and colorized depth aligned
to RGB on the right. Black areas in the depth view have no valid depth. Press
`q` or Esc in the window to stop. The terminal readings continue during preview.

On Linux, if USB enumeration sees the camera but the SDK cannot access it,
check the [RealSense Linux device permissions and udev instructions](https://github.com/realsenseai/librealsense/blob/master/doc/distribution_linux.md).
The Python package needs a compatible wheel for your platform. See the
[RealSense Python SDK installation guide](https://github.com/realsenseai/librealsense/blob/master/wrappers/python/readme.md)
for supported environments.

### Track a white table-tennis ball

```sh
python -B scripts/track_ball.py --preview --every-frame --seconds 10
```

The tracker requests 640 × 480 RGB and depth at 60 fps over USB 3. It prints
the active profiles and measured rate of fresh RGB-D pairs at normal exit.
Use `--fps 30` if the camera or connection cannot sustain 60 fps.

Keep the camera fixed and move the ball against a darker background. The
tracker looks for a bright, moving, round region. A green circle indicates a
valid camera-frame XYZ estimate. A yellow circle indicates a 2D detection with
unusable depth or timing. The tracker accepts surface depths from 0.3 to 3 m.
The terminal prints the camera timestamp, pixel center, and XYZ in meters
when available, or a reason the depth estimate was rejected. Use `--every-frame`
to log each fresh pair and `--brightness` to adjust the detection threshold.
Omit `--seconds` for a continuous preview. Press `q` or Esc to exit.

Position and velocity use the D435 optical frame, with axes pointing right,
down, and forward. After at least four valid positions over at least 45 ms,
the tracker fits velocity in m/s from the most recent 150 ms of capture times.
It restarts the fit after a gap longer than 75 ms or when positions disagree
with the fit by more than 3 cm. `valid=True` means the current XYZ is usable.
`velocity=unavailable` means the tracker is still gathering consistent
measurements for the fit.

When velocity is available, the terminal also prints a 50 ms straight-line
look-ahead in the camera frame. The ballistic prediction of where the ball
crosses the box opening will use the camera-to-Franka calibration and the
opening height.

To follow the code, start with `track()` in [track_ball.py](scripts/track_ball.py).
It reads each fresh pair, calls `detect_ball()` for the 2D candidate, then calls
`describe_detection()` to check timing and recover XYZ from aligned depth.
`MotionHistory` fits velocity from the valid timestamped positions.
[check_camera.py](scripts/check_camera.py) owns the shared camera startup and
cleanup and can also be run by itself to inspect the raw streams.

## Interface

`catchbox/learning/environment.py` constructs the 60-value normalized observation and applies the model's seven bounded residual corrections. `catchbox/learning/trajectory.py` predicts the ballistic intercept. `catchbox/sim.py` constructs the robot, box and simulation; `catchbox/learning/train.py` contains training and evaluation routines.

The model is an `sb3-contrib` CrossQ checkpoint. For example, load it without starting training:

```python
from sb3_contrib import CrossQ
model = CrossQ.load('artifacts/original-model/box/final_model.zip', device='cpu')
```

The model expects the exact observation order, normalization and controller state constructed in `CatchCore.observation()`. Its output is not a joint command. The hardware adapter must be designed and validated separately.

`scripts/train_original_model.py` documents the training entry point. **Running it starts a new training run**, writes a new output folder and takes substantial time; it refuses to overwrite existing run artifacts. Do not run it merely to load or evaluate the checkpoint.


