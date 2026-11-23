# Franka hardware prototype: ROS 1 Noetic

This is a **prototype, not yet validated for robot motion**. It has not been built with Noetic, connected to a Franka or run in Gazebo. Motion is disabled in the controller configuration. A 20 cm command sphere and sampled state checks do not guarantee that the real box or the robot remains in that sphere or avoids collisions. The smoother and the actual robot dynamics can deviate from the checked path. Validate the robot, tool, planning scene, joint limits, stop behavior and collision protection independently before any physical test.

## Components

- `scripts/calibration.py` records a fixed sphere origin from a stationary robot, then previews or sends origin/up/down/left/right opening targets. Left/right are base-frame +Y/-Y; up/down are +Z/-Z. Some targets can be unreachable with the restricted joints.
- `catchbox/hardware/ik.py` uses a KDL forward-kinematics model of the actual URDF and the calibrated tool offset. It solves for joints 1 and 4, counter-rotates joint 6 as in simulation, and holds joints 2, 3, 5 and 7 at the recorded home. It rejects targets outside a configured residual tolerance. The coupling approximates level box orientation; verify actual orientation and mounting on the robot.
- `scripts/franka_control.py` receives base-frame opening targets, runs IK outside the realtime loop, screens sampled joint configurations with MoveIt `/check_state_validity`, and sends accepted joint targets. Targets below `min_opening_z_m` are rejected, and so is any path whose sampled opening positions dip below it. This is not continuous path checking. When target messages stop, the last target becomes stale after 0.2 s and the controller requests a stop. After 3 seconds without a target, it requests a return to the recorded center, using the same IK and path checks. This timer also starts when the adapter launches. It stops requesting movement within 1 cm and will not return again unless the opening drifts more than 1.5 cm away. A new target interrupts the return. Invalid targets, stale robot state, or failed checks cause it to stop publishing; the realtime controller requests a rate-limited stop. Motion must be enabled for return-to-center to move the robot. Its 50 Hz loop is a maximum rate; IK and MoveIt latency may be slower.
- `src/franka_control.cpp` is a ROS 1 `ros_control` plugin, run by Franka's realtime control loop. It consumes joint targets through a realtime buffer and uses `franka::limitRate` with the robot's previous desired joint position, velocity and acceleration. The controller is configured with motion disabled. A command timeout requests a rate-limited stop; this is not an emergency stop. The return-to-center behavior belongs to the Python adapter: if that node dies or loses robot feedback, the realtime controller stops rather than returning.

There is **no camera-to-policy observation adapter or model inference node yet**. The high-level ROS input accepts desired box-opening positions from calibration or a future policy adapter. In simulation the trained policy uses 25 scaled inputs and outputs positions relative to its starting opening; integrating it requires reproducing that contract and aligning calibrated frames. The simulated workspace was centered at the nominal gripper position, whereas this prototype records the **opening** as the sphere origin; these are different points. Resolve this reference difference before policy deployment.

## Build and configure

Use a Linux host with ROS 1 Noetic, compatible `franka_ros` and `libfranka`, MoveIt, PyKDL, `kdl_parser_py`, `urdf_parser_py`, SciPy, NumPy and PyYAML. Place or symlink `catchbox/hardware/ros1/catchbox_control` into a catkin workspace's `src`, build with `catkin_make` or `catkin build`, and source the workspace. Make this repository importable by the same Python used by ROS, for example:

```sh
export PYTHONPATH="/path/to/robot-autonomy-project:$PYTHONPATH"
```

The Python kinematics need `kdl_parser_py`, `urdf_parser_py` and `PyKDL`. On the station they were installed with:

```sh
apt install -y ros-noetic-kdl-parser-py ros-noetic-urdfdom-py python3-pykdl
python3 -c "import PyKDL, kdl_parser_py.urdf, urdf_parser_py.urdf; print('ok')"
```

The package uses only `catchbox.hardware` and does not require installing the simulator for the ROS adapter. Make sure `/robot_description` contains the actual Panda model, the configured seven-joint chain is ordered from base to tip, and the MoveIt planning scene includes the real box and obstacles. Measure the opening-center offset in the chosen tip frame; do not assume an STL offset is a calibrated grasp.

Review `catchbox/hardware/configuration.yaml` and load it with `rosparam load catchbox/hardware/configuration.yaml`. The controller reads joint position and velocity limits from `/robot_description` (the Panda URDF) and refuses to start if they are missing. The YAML contains conservative speed, acceleration and jerk settings that must be reviewed for your robot; these are not measured calibration values. Configure the controller manager to use the position and Franka state interfaces. Follow your installed `franka_ros`/`libfranka` version's filter and rate-limiter configuration: the `franka::limitRate` API documents that FCI filters must be deactivated for its intended behavior. Confirm controller update timing is 1 kHz; this code accepts targets only when its update period is within 0.5–1.5 ms.

## Capture and FrankaPy checks

First open `catchbox/hardware/configuration.yaml` and set `tip_link` and `opening_offset_m` to the measured box-opening offset in the frame returned by FrankaPy `get_pose()`. That frame must match `tip_link` in the robot URDF. The radius is fixed at 0.2 m. Set `min_opening_z_m` above the table plus the box depth.

```sh
python3 -m catchbox.hardware.calibration --set-origin   # record center and home joints; no motion
python3 -m catchbox.hardware.calibration --show         # print center and 0.2 m edge targets; no motion
```

`--set-origin` uses `FrankaArm.get_pose()` and `get_joints()`. Keep the robot stationary while recording.

Then test motion directly through FrankaPy, before the ROS node. Run the checks in order; each one only after the previous one behaves. They command the physical robot, with no MoveIt collision screening, so keep a hand on the e-stop. Each check prompts before moving (`--yes` skips the prompt).

```sh
python3 -m catchbox.hardware.calibration --check1-home    # joint move to recorded home; opening at center
python3 -m catchbox.hardware.calibration --check2-small   # 3 cm out and back: left, right, up, down, forward, back
python3 -m catchbox.hardware.calibration --check3-steps   # 5, 10, 15, 20 cm out and back in each direction
python3 -m catchbox.hardware.calibration --check4-catch   # 10 random points inside the sphere, chained
python3 -m catchbox.hardware.calibration --move left --distance 0.05   # home, then one step
```

Left/right are base-frame +Y/-Y, up/down +Z/-Z, forward/back +X/-X. All moves command the box opening and keep the box orientation from home. Each check starts with a joint move to home. Targets outside the 0.2 m sphere are rejected; check3 skips steps below `min_opening_z_m` and check4 only picks points above it. After every move the script prints the reached opening position and its error.

Options: `--speed` (opening speed in m/s, default 0.075), `--max-distance` (check3 largest step, check4 sphere radius), `--left`, `--right`, `--up`, `--down`, `--forward`, `--back` to run check3 in only those directions (combine them, e.g. `--left --right`), `--count` and `--seed` (check4).

The same YAML file is used directly by the ROS calibration script and command adapter. The adapter verifies the recorded center against URDF forward kinematics; fix any mismatch between the reported FrankaPy tool frame, `tip_link`, and the opening offset before moving. Do not commit a local calibration file without review.

The ROS capture method also uses the same file: `rosrun catchbox_control calibration.py set-origin`. To preview an edge through ROS:

```sh
rosrun catchbox_control calibration.py move left --distance 0.2
```

`move origin`, `move up`, `move down`, `move left` and `move right` select tests. Adding `--execute` publishes a target to the ROS command adapter; it is not a direct FrankaPy motion command.

For integration testing, after loading the plugin into the controller manager, run the adapter:

```sh
rosrun catchbox_control franka_control.py \
  '~joint_target:=/catchbox_joint_controller/joint_target'
```

It receives `/catchbox_command_adapter/opening_target` as `geometry_msgs/PointStamped` in the configured base frame. `calibration.py move ... --execute` publishes a target for five seconds; the adapter and controller must both be running, configured and enabled for motion. Test with motion disabled and in a realistic simulation first. Do not treat this flag as a substitute for a robot safety system.

The latest model must not be connected to physical motion until its observation encoder, tool transform, target origin, joint coupling, timing, collision checks and stop behavior have been validated end to end.
