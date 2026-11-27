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
4. Use `catchbox.hardware.observation.build_steady_observation` to reproduce the steady-motion 25-value model observation: opening position (0–2), opening linear velocity / 3 (3–5), angular velocity / 3 (6–8), ball position (9–11), ball velocity / 5 (12–14), predicted crossing position (15–17), predicted crossing velocity / 5 (18–20), crossing time clipped to [0, 2] (21), tracking-valid (22), prediction-valid (23), and live in-box flag (24). Clip the full vector to [-10, 10] and use float32. For a hidden ball, position and velocity are zeros with tracking-valid 0; the final flag is -1 for unknown, 0 for outside or 1 for inside/probably inside. Invalid predictions are zeros. The simulator predicts the descending crossing of the nominal opening-height plane and disables prediction while in-box.
5. Test recorded sequences for transforms, units, timing, tracking loss, reacquisition, prediction and in-box state. Compare the hardware encoder numerically against the simulation observation using identical synthetic inputs. Report measured latency and errors before connecting robot motion.

The current checkpoint is `artifacts/limited-thin-model/final_model.zip`. Its three outputs are desired opening-position values, not robot commands. A hardware adapter, IK, continuous joint controller and independently validated safety layer are still required. The existing simulation is not a hardware safety controller: its actual opening left the nominal sphere in the edge test. The older 60-input checkpoint is a different model and interface.


### ROS camera output for steady-motion

Publish `/catchbox/ball_tracking_status` as `std_msgs/Int8`: -1 means no active ball, 0 means an active but hidden ball, 1 means visible with velocity unavailable, and 2 means visible with valid velocity. Publish `/catchbox/ball_in_box` continuously as -1 unknown, 0 outside, or 1 inside/probably inside. Publish capture-stamped `/catchbox/ball_state` Odometry in `panda_link0` when visible. The model node requires fresh status and in-box messages, plus fresh ball and box Odometry while visible. A hidden ball still enables model commands; use no-active-ball status when the episode is over so the adapter can return after three seconds. See [hardware guide](HARDWARE_ROS1.md#policy-and-robot-state) for timing limits. The current camera tracker does not publish these ROS topics.

The model node now gates inference on recent measured ball motion and an interception inside the calibrated 20 cm sphere. The gate uses at least three recent positions and 4 cm net movement in 250 ms; it rejects implausible position jumps and waits for five clean observations after a jump to clear the current velocity fit. These are tentative thresholds and must be evaluated against real camera data, including missed frames and moving hands. If a real throw is not tracked or its predicted intercept is outside the sphere, no model target is published.
