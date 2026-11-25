# Camera integration guide

## Requirements

Provide the controller with the ball’s 3D center position and velocity in the Franka base frame, with timestamps and a tracking-valid flag. Track the incoming flight and the region around the box. Provide a live in-box estimate that can return to false if the ball reappears outside. The RealSense scripts already provide camera-frame tracking; camera-to-base calibration, in-box estimation and hardware integration remain to be built.

- Use a fixed, calibrated stereo/depth setup or synchronized cameras; choose the hardware with the team. A single ordinary RGB image does not directly provide metric 3D position.
- Provide position in meters and velocity in meters/second, each as `(x, y, z)`. Provide gravity/acceleration in meters/second² as described below.
- Deliver estimates to the controller at **50 Hz**, with capture and processing timing exposed so latency and stale frames are detectable. The camera may need a higher frame rate; determine it from measured tracking performance rather than assuming 50 Hz is sufficient.
- Calibrate the camera-to-robot-base transform and validate 3D error and latency over the intended flight volume, including near the box and under occlusion.
- Report lost, stale, ambiguous or low-confidence tracking explicitly. Never turn an invalid measurement into a believable zero position.
- Run the tracker in observation-only mode. The paired-point RGB calibration and tracker commands are in the [README](../README.md#report-ball-coordinates-in-the-panda-base-frame).

The current checkpoint receives more than camera data: it also needs robot, box, prediction and controller state. The camera subsystem must not fabricate those other inputs.

## Instructions for implementation

1. Read `scripts/track_ball.py` and `scripts/check_camera.py`. Preserve the existing tracker. Calibrate camera intrinsics and the camera-to-Franka-base rigid transform, using known points or robot poses; validate on held-out points across the flight volume.
2. Publish timestamped ball center position (meters), velocity (m/s), and validity in the robot base frame. Document capture, estimate and publication times and clock synchronization. Reject stale or ambiguous estimates. During flight, use gravity as a prediction prior; validate this approximation against measured trajectories.
3. Derive the box opening position, linear velocity and angular velocity from robot feedback and the calibrated tool transform. Implement a live in-box estimate; account explicitly for occlusion and uncertainty. Do not assume an unseen ball is safely inside.
4. Reproduce the current 25-value model observation from `catchbox/learning/pose_environment.py` exactly: opening position (0–2), opening linear velocity / 3 (3–5), angular velocity / 3 (6–8), ball position (9–11), ball velocity / 5 (12–14), predicted crossing position (15–17), predicted crossing velocity / 5 (18–20), crossing time clipped to [0, 2] (21), tracking-valid (22), prediction-valid (23), and live in-box flag (24). Clip the full vector to [-10, 10] and use float32. Invalid predictions are zeros. The simulator predicts the descending crossing of the nominal opening-height plane and disables prediction while in-box.
5. Test recorded sequences for transforms, units, timing, tracking loss, reacquisition, prediction and in-box state. Compare the hardware encoder numerically against the simulation observation using identical synthetic inputs. Report measured latency and errors before connecting robot motion.

The current checkpoint is `artifacts/limited-thin-model/final_model.zip`. Its three outputs are desired opening-position values, not robot commands. A hardware adapter, IK, continuous joint controller and independently validated safety layer are still required. The existing simulation is not a hardware safety controller: its actual opening left the nominal sphere in the edge test. The older 60-input checkpoint is a different model and interface.
