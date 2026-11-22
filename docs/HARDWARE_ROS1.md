# Franka hardware prototype: ROS 1 Noetic

This is a **prototype, not yet validated for robot motion**. It has not been built with Noetic, connected to a Franka or run in Gazebo. Motion is disabled in the controller configuration. A 20 cm command sphere and sampled state checks do not guarantee that the real box or the robot remains in that sphere or avoids collisions. The smoother and the actual robot dynamics can deviate from the checked path. Validate the robot, tool, planning scene, joint limits, stop behavior and collision protection independently before any physical test.

## Components

- `scripts/calibration.py` records a fixed sphere origin from a stationary robot, then previews or sends origin/up/down/left/right opening targets. Left/right are base-frame +Y/-Y; up/down are +Z/-Z. Some targets can be unreachable with the restricted joints.
- `catchbox/hardware/ik.py` uses a KDL forward-kinematics model of the actual URDF and the calibrated tool offset. It solves for joints 1 and 4, counter-rotates joint 6 as in simulation, and holds joints 2, 3, 5 and 7 at the recorded home. It rejects targets outside a configured residual tolerance. The coupling approximates level box orientation; verify actual orientation and mounting on the robot.
- `scripts/franka_control.py` receives base-frame opening targets, runs IK outside the realtime loop, screens sampled joint configurations with MoveIt `/check_state_validity`, and sends accepted joint targets. This is not continuous path checking. It rejects stale data and stops publishing on failure. Its 50 Hz loop is a maximum rate; IK and MoveIt latency may be slower.
- `src/franka_control.cpp` is a ROS 1 `ros_control` plugin, run by Franka's realtime control loop. It consumes joint targets through a realtime buffer and uses `franka::limitRate` with the robot's previous desired joint position, velocity and acceleration. The controller is configured with motion disabled. A command timeout requests a rate-limited stop; this is not an emergency stop.

There is **no camera-to-policy observation adapter or model inference node yet**. The high-level ROS input accepts desired box-opening positions from calibration or a future policy adapter. In simulation the trained policy uses 25 scaled inputs and outputs positions relative to its starting opening; integrating it requires reproducing that contract and aligning calibrated frames. The simulated workspace was centered at the nominal gripper position, whereas this prototype records the **opening** as the sphere origin; these are different points. Resolve this reference difference before policy deployment.

## Build and configure

Use a Linux host with ROS 1 Noetic, compatible `franka_ros` and `libfranka`, MoveIt, PyKDL, `kdl_parser_py`, `urdf_parser_py`, SciPy, NumPy and PyYAML. Place or symlink `catchbox/hardware/ros1/catchbox_control` into a catkin workspace's `src`, build with `catkin_make` or `catkin build`, and source the workspace. Make this repository importable by the same Python used by ROS, for example:

```sh
export PYTHONPATH="/path/to/robot-autonomy-project:$PYTHONPATH"
```

The package uses only `catchbox.hardware` and does not require installing the simulator for the ROS adapter. Make sure `/robot_description` contains the actual Panda model, the configured seven-joint chain is ordered from base to tip, and the MoveIt planning scene includes the real box and obstacles. Measure the opening-center offset in the chosen tip frame; do not assume an STL offset is a calibrated grasp.

Review `catchbox/hardware/configuration.yaml` and load it with `rosparam load catchbox/hardware/configuration.yaml`. The controller reads joint position and velocity limits from `/robot_description` (the Panda URDF) and refuses to start if they are missing. The YAML contains conservative speed, acceleration and jerk settings that must be reviewed for your robot; these are not measured calibration values. Configure the controller manager to use the position and Franka state interfaces. Follow your installed `franka_ros`/`libfranka` version's filter and rate-limiter configuration: the `franka::limitRate` API documents that FCI filters must be deactivated for its intended behavior. Confirm controller update timing is 1 kHz; this code accepts targets only when its update period is within 0.5–1.5 ms.

## Capture and preview

First open `catchbox/hardware/configuration.yaml` and set `tip_link` and `opening_offset_m` to the measured box-opening offset in the frame returned by FrankaPy `get_pose()`. That frame must match `tip_link` in the robot URDF. The radius is fixed at 0.2 m.

```sh
# Edit catchbox/hardware/configuration.yaml: tip_link and opening_offset_m
python -m catchbox.hardware.calibration --set-origin
python -m catchbox.hardware.calibration --show
```

The script runs as ordinary Python and uses `FrankaArm.get_pose()` and `get_joints()` to record the current opening center and home joints. FrankaPy still needs its ROS/robot infrastructure running. Position the robot safely and keep it stationary while recording; pose and joints are separate readings. The script does not move the robot. `--show` previews origin and four sphere-edge coordinates; `--target left --distance 0.03` previews a point 3 cm from the center. Targets are not guaranteed reachable by the restricted joints.

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
