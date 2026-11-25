# Franka hardware: FrankaPy + ROS 1

This is a **prototype, not yet validated for robot motion through the adapter**. The FrankaPy calibration checks have run on the robot; the adapter has only been tested offline against a fake robot. A 20 cm command sphere, a floor height and sampled state checks do not guarantee that the real box or arm stays in that sphere or avoids collisions. Keep a hand on the e-stop.

Everything runs in the lab's Docker container on the User PC, through FrankaPy. FrankaPy's control-PC server (`franka-interface`, started by `start_control_pc.sh`) is the only thing that talks to the robot; nothing here needs franka_ros, a catkin build or a real-time kernel.

## Components

- `catchbox/hardware/calibration.py` records the sphere center (the box opening at home) and runs direct FrankaPy motion checks.
- `catchbox/hardware/panda_kinematics.py` is Panda forward kinematics in NumPy, using Franka's DH parameters and FrankaPy's default tool frame (hand TCP). It reproduces both recorded calibrations to well under 0.1 mm.
- `catchbox/hardware/ik.py` solves for joints 1 and 4, counter-rotates joint 6 as in simulation, and holds joints 2, 3, 5 and 7 at the recorded home. Only a narrow band around home is reachable this way (from the current home: about ±5 cm left/right, almost nothing straight up or down).
- `catchbox/hardware/adapter.py` is the ROS node. The task space is the 0.2 m sphere around the calibrated origin, above `min_opening_z_m`. Requested targets are clamped into it. IK goes to the nearest point reachable with joints 1 and 4 (joint 6 keeps the box level), as in simulation; if that point is outside the task space the adapter holds. Accepted goals are streamed at `rate_hz` to a FrankaPy dynamic joint skill. `limiter.py` turns each goal into small steps within `max_joint_velocity` and `max_joint_acceleration`; every streamed command is checked against the task space (5 mm tolerance) and the node stops if one leaves it.
  - When target messages stop, the target becomes stale after 0.2 s and the adapter stops where it is. After 3 s without a target it returns to the center; it stops returning within 1 cm and restarts only if the opening drifts more than 1.5 cm. A new target interrupts the return.
  - It waits for fresh `/catchbox/robot_joints` feedback and refuses to start unless the robot is within 0.05 rad of the recorded home (the IK holds joints 2, 3, 5 and 7 there) and the home joints reproduce `origin_xyz_m`. It stops if the FrankaPy skill ends or faults; FrankaPy's own collision and limit reflexes still apply.
  - With `enable_motion: false` it does everything except start the skill, and publishes what it would send.
- `catchbox/hardware/send_target.py` publishes a test target to the adapter for a few seconds.
- `catchbox/hardware/policy_frame.py` converts between the trained policy's simulation frame and the robot: `observation()` shifts the ball, box and predicted-contact positions by `origin_xyz_m - sim_start_opening_m`, and `target()` reproduces the simulator's action-to-target mapping, then shifts it back. Simulated against the 30-throw test bank, it matches the simulator's targets exactly.

## Policy and robot state

These are ordinary Python scripts using ROS 1 topics; they do not need a catkin build. `python3 -m catchbox.hardware.robot_state` reads actual FrankaPy joint positions and velocities and publishes `/catchbox/robot_joints` (JointState) and `/catchbox/box_state` (Odometry). Both are stamped after polling, not at the hardware measurement time. The Odometry position, linear velocity and angular velocity are in `panda_link0`; they come from calibrated forward kinematics. Position and velocity are read in separate calls. The adapter requires fresh measured joints to track or return; stale feedback makes it hold.

`python3 -m catchbox.hardware.model_node` runs the exported actor on CPU with NumPy and publishes `/catchbox_command_adapter/opening_target`. It builds the exact 25 inputs including prediction at the fixed start height, and applies `PolicyFrame`. The NumPy actor was exported from the saved checkpoint and compared against it; inference requires no PyTorch or SB3. `scripts/export_actor.py` requires the training environment to reproduce the export; a stored checkpoint checksum rejects a mismatched actor. ROS 1 Python, NumPy, SciPy, PyYAML, FrankaPy and ROS message packages are still required on the robot side.

The model accepts `/catchbox/ball_state` (nav_msgs/Odometry) with measured ball-center position and linear velocity in `panda_link0`, and `/catchbox/ball_in_box` (std_msgs/Int8: 0 outside, 1 inside, -1 unknown). Send these continuously, including the flag. Odometry `header.stamp` must represent the estimated state time; both Odometry headers use `panda_link0`, and both linear/angular twist vectors are in that frame. The flag is receipt-timed because Int8 has no header; a production camera adapter needs to account for capture time and uncertainty. A missing, stale, invalid or unknown input stops model target output; the adapter holds after 0.2 s and requests return after 3 s of no valid target. Ball and box state timestamps must be within 50 ms and each no older than 200 ms. The ROS camera publisher and reliable in-box estimator are not implemented; the current camera tracker does not publish these messages. Never interpret an occluded ball as inside without evidence.

All three scripts write JSON-lines logs under `logs/`: `robot_state.jsonl` (measured joints, opening and poll/FK duration), `model.jsonl` (observation, action, target, sensor ages and inference duration) and `adapter.jsonl` (target receipt, mode, planning duration and sampled joint commands). ROS stamps allow correlating messages; wall and monotonic timestamps are also recorded. These logs do not measure camera capture-to-robot latency unless the camera supplies a synchronized estimate timestamp. Inspect dropped events if the asynchronous log queue fills.

## Mock pipeline

Start with `adapter.enable_motion: false`. **Mock ball inputs can move the real robot if another running adapter has motion enabled.** Use one publisher per topic. In separate terminals, from the project root with `PYTHONPATH=$PWD:$PYTHONPATH` and ROS/FrankaPy running:

```sh
python3 -m catchbox.hardware.robot_state
python3 -m catchbox.hardware.adapter
python3 -m catchbox.hardware.model_node
python3 -m catchbox.hardware.mock_inputs --seconds 10
```

Watch `/catchbox/box_state`, `/catchbox_command_adapter/opening_target` and `/catchbox_command_adapter/joint_target` with `rostopic echo`. The real robot state publisher reads but does not move the robot. The mock publisher supplies fake ball data only; use `--mock-box-state` instead of starting `robot_state` to test the model alone without hardware. The real adapter still initializes FrankaPy even in dry-run mode. For a direct mock target through IK and smoothing, use `python3 -m catchbox.hardware.send_target left --distance 0.03 --execute` with adapter motion disabled. A physical movement test is separate: review the calibration and floor, enable the adapter explicitly, and validate on the station with a controlled small move. The full stack has not yet been tested there.

## Setup

Start the robot and FrankaPy as in Lab 1 (unlock in Desk at 172.16.0.2 from the Control PC, `roscore`, then `start_control_pc.sh`). In every container terminal:

```sh
cd /home/ros_ws/src/devel_packages/robot-autonomy-project
export PYTHONPATH=$PWD:$PYTHONPATH
```

All settings live in `catchbox/hardware/configuration.yaml`: the `calibration` section (tool offset, center, home joints, floor), the `adapter` section (motion switch, stream rate, joint limits) and the `policy` section (the simulation frame; do not edit).

## Capture and FrankaPy checks

Set `opening_offset_m` to the measured box-opening offset in FrankaPy's tool frame (`get_pose()`), and `min_opening_z_m` above the table plus the box depth. The radius is fixed at 0.2 m.

```sh
python3 -m catchbox.hardware.calibration --set-origin   # record center and home joints; no motion
python3 -m catchbox.hardware.calibration --show         # print center and 0.2 m edge targets; no motion
```

Keep the robot stationary while recording. Then test motion directly through FrankaPy. Run the checks in order, each only after the previous one behaves. Each prompts before moving (`--yes` skips the prompt).

```sh
python3 -m catchbox.hardware.calibration --check1-home    # joint move to recorded home; opening at center
python3 -m catchbox.hardware.calibration --check2-small   # 3 cm out and back: left, right, up, down, forward, back
python3 -m catchbox.hardware.calibration --check3-steps   # 5, 10, 15, 20 cm out and back in each direction
python3 -m catchbox.hardware.calibration --check4-catch   # 10 random points inside the sphere, chained
python3 -m catchbox.hardware.calibration --move left --distance 0.05   # home, then one step
```

Left/right are base-frame +Y/-Y, up/down +Z/-Z, forward/back +X/-X. All moves command the box opening and keep the box orientation from home, using FrankaPy's full-arm motion (not the adapter's restricted IK). Targets outside the sphere are rejected; check3 skips steps below `min_opening_z_m` and check4 only picks points above it. After every move the script prints the reached opening position and its error.

Options: `--speed` (opening speed in m/s, default 0.075), `--max-distance` (check3 largest step, check4 sphere radius), `--left`, `--right`, `--up`, `--down`, `--forward`, `--back` to run check3 in only those directions (combine them, e.g. `--left --right`), `--count` and `--seed` (check4).

## Adapter bring-up

1. Put the robot at home: `python3 -m catchbox.hardware.calibration --check1-home`.
2. With `adapter.enable_motion: false`, start the adapter and send a target:

   ```sh
   python3 -m catchbox.hardware.robot_state  # separate terminal
   python3 -m catchbox.hardware.adapter
   rostopic echo /catchbox_command_adapter/joint_target       # second terminal
   python3 -m catchbox.hardware.send_target left --distance 0.03 --execute   # third terminal
   ```

   Joints 1, 4 and 6 should change smoothly, the others stay at home, and the robot does not move.
3. Set `enable_motion: true`, restart the adapter, and repeat with left/right at 3 cm, then 5 cm. Stop the adapter with Ctrl-C; it ends the FrankaPy skill.

Up/down targets move the box to the nearest point joints 1, 4 and 6 can reach, so expect it to move partly forward or back rather than straight up or down.

## Hardware-free ROS test in Docker

From the project root, start Docker and run:

```sh
docker build -t catchbox-ros-test .
docker run --rm --network none catchbox-ros-test
```

The container starts its own ROS master on loopback, a fake robot, the normal adapter in `--mock-robot` mode, the model node, and a fake ball publisher. It checks a target flowing through IK and smoothing, the three-second return to center, and a fake ball passing through the policy and adapter. `--mock-robot` forces motion disabled even if a configuration file says otherwise; the image does not contain FrankaPy. `--network none` also prevents a connection to an external robot or ROS master. The test prints PASS lines and exits nonzero with node logs if a check fails.

The fake robot instantly follows the joint commands; this test checks ROS messages and control logic, not real robot dynamics, frame calibration, collision clearance, grasp security, or actual hardware timing. The image uses a Noetic/Focal base and requires network access during `docker build`; the test run itself needs no network.
