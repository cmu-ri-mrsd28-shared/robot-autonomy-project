# Franka hardware: FrankaPy + ROS 1

This is a **prototype, not yet validated for robot motion through the adapter**. The FrankaPy calibration checks have run on the robot; the adapter has only been tested offline against a fake robot. A 20 cm command sphere, a floor height and sampled state checks do not guarantee that the real box or arm stays in that sphere or avoids collisions. Keep a hand on the e-stop.

Everything runs in the lab's Docker container on the User PC, through FrankaPy. FrankaPy's control-PC server (`franka-interface`, started by `start_control_pc.sh`) is the only thing that talks to the robot; nothing here needs franka_ros, a catkin build or a real-time kernel.

## Components

- `catchbox/hardware/calibration.py` records the sphere center (the box opening at home) and runs direct FrankaPy motion checks.
- `catchbox/hardware/panda_kinematics.py` is Panda forward kinematics in NumPy, using Franka's DH parameters and FrankaPy's default tool frame (hand TCP). It reproduces both recorded calibrations to well under 0.1 mm.
- `catchbox/hardware/ik.py` solves for joints 1 and 4, counter-rotates joint 6 as in simulation, and holds joints 2, 3, 5 and 7 at the recorded home. Only a narrow band around home is reachable this way (from the current home: about ±5 cm left/right, almost nothing straight up or down).
- `catchbox/hardware/adapter.py` is the ROS node. The task space is the 0.2 m sphere around the calibrated origin, above `min_opening_z_m`. Requested targets are clamped into it. IK goes to the nearest point reachable with joints 1 and 4 (joint 6 keeps the box level), as in simulation; if that point is outside the task space the adapter holds. Accepted goals are streamed at `rate_hz` to a FrankaPy dynamic joint skill. `limiter.py` turns each goal into small steps within `max_joint_velocity` and `max_joint_acceleration`; every streamed command is checked against the task space (5 mm tolerance) and the node stops if one leaves it.
  - When target messages stop, the target becomes stale after 0.2 s and the adapter stops where it is. After 3 s without a target it returns to the center; it stops returning within 1 cm and restarts only if the opening drifts more than 1.5 cm. A new target interrupts the return.
  - It refuses to start unless the robot is within 0.05 rad of the recorded home (the IK holds joints 2, 3, 5 and 7 there) and the home joints reproduce `origin_xyz_m`. It stops if the FrankaPy skill ends or faults; FrankaPy's own collision and limit reflexes still apply.
  - With `enable_motion: false` it does everything except start the skill, and publishes what it would send.
- `catchbox/hardware/send_target.py` publishes a test target to the adapter for a few seconds.
- `catchbox/hardware/policy_frame.py` converts between the trained policy's simulation frame and the robot: `observation()` shifts the ball, box and predicted-contact positions by `origin_xyz_m - sim_start_opening_m`, and `target()` reproduces the simulator's action-to-target mapping, then shifts it back. Simulated against the 30-throw test bank, it matches the simulator's targets exactly.

There is **no camera-to-policy observation adapter or model inference node yet**. The adapter accepts desired box-opening positions from `send_target.py` or a future policy node. The policy node still has to build the 25-value observation from the tracked ball (in panda_link0, after camera-to-robot calibration) and the box state, then use `PolicyFrame` to convert observations and actions.

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
   python3 -m catchbox.hardware.adapter
   rostopic echo /catchbox_command_adapter/joint_target       # second terminal
   python3 -m catchbox.hardware.send_target left --distance 0.03 --execute   # third terminal
   ```

   Joints 1, 4 and 6 should change smoothly, the others stay at home, and the robot does not move.
3. Set `enable_motion: true`, restart the adapter, and repeat with left/right at 3 cm, then 5 cm. Stop the adapter with Ctrl-C; it ends the FrankaPy skill.

Up/down targets move the box to the nearest point joints 1, 4 and 6 can reach, so expect it to move partly forward or back rather than straight up or down.
