# Franka hardware: FrankaPy + ROS 1

This is a **prototype, not yet validated for robot motion through the adapter**. The FrankaPy calibration checks have run on the robot; the adapter has only been tested offline against a fake robot. A 20 cm command sphere, a floor height and sampled state checks do not guarantee that the real box or arm stays in that sphere or avoids collisions. Keep a hand on the e-stop.

Everything runs in the lab's Docker container on the User PC, through FrankaPy. FrankaPy's control-PC server (`franka-interface`, started by `start_control_pc.sh`) is the only thing that talks to the robot; nothing here needs franka_ros, a catkin build or a real-time kernel.

## Components

- `catchbox/hardware/calibration.py` records the sphere center (the box opening at home) and runs direct FrankaPy motion checks.
- `catchbox/hardware/panda_kinematics.py` is Panda forward kinematics in NumPy, using Franka's DH parameters and FrankaPy's default tool frame (hand TCP). It reproduces both recorded calibrations to well under 0.1 mm.
- `catchbox/hardware/ik.py` solves for joints 1 and 4, counter-rotates joint 6 as in simulation, and holds joints 2, 3, 5 and 7 at the recorded home. Only a narrow band around home is reachable this way (from the current home: about ±5 cm left/right, almost nothing straight up or down).
- `catchbox/hardware/adapter.py` is the ROS node. It receives box-opening targets, solves IK, screens 21 joint configurations along the straight joint-space path (joint limits, 0.2 m sphere, `min_opening_z_m`, FrankaPy virtual walls and, unless `--no-moveit`, MoveIt `/check_state_validity`), and streams accepted joint targets at `rate_hz` to a FrankaPy dynamic joint skill. `limiter.py` turns each accepted goal into small steps within `max_joint_velocity` and `max_joint_acceleration`, so a move from rest follows the checked line.
  - When target messages stop, the target becomes stale after 0.2 s and the adapter stops where it is. After 3 s without a target it returns to the center with the same checks; it stops returning within 1 cm and restarts only if the opening drifts more than 1.5 cm. A new target interrupts the return.
  - It refuses to start unless the robot is within 0.05 rad of the recorded home and the home joints reproduce `origin_xyz_m`. It stops and exits if the FrankaPy skill ends or faults, or if any measured joint lags the command by more than `tracking_limit_rad`.
  - With `enable_motion: false` it does everything except start the skill, and publishes what it would send.
- `catchbox/hardware/send_target.py` publishes a test target to the adapter for a few seconds.
- `catchbox/hardware/planning_scene.py` adds the table and the catch box to MoveIt's planning scene.

There is **no camera-to-policy observation adapter or model inference node yet**. The adapter accepts desired box-opening positions from `send_target.py` or a future policy node. In simulation the trained policy uses 25 scaled inputs and outputs positions relative to its starting opening; integrating it requires reproducing that contract and aligning calibrated frames. The simulated workspace was centered at the nominal gripper position, whereas this prototype records the **opening** as the sphere origin; resolve this before policy deployment.

## Setup

Start the robot and FrankaPy as in Lab 1 (unlock in Desk at 172.16.0.2 from the Control PC, `roscore`, then `start_control_pc.sh`). In every container terminal:

```sh
cd /home/ros_ws/src/devel_packages/robot-autonomy-project
export PYTHONPATH=$PWD:$PYTHONPATH
```

All settings live in `catchbox/hardware/configuration.yaml`: the `calibration` section (tool offset, center, home joints, floor), the `adapter` section (motion switch, stream rate, joint limits, tracking limit) and the `planning_scene` section.

## Capture and FrankaPy checks

Set `opening_offset_m` to the measured box-opening offset in FrankaPy's tool frame (`get_pose()`), and `min_opening_z_m` above the table plus the box depth. The radius is fixed at 0.2 m.

```sh
python3 -m catchbox.hardware.calibration --set-origin   # record center and home joints; no motion
python3 -m catchbox.hardware.calibration --show         # print center and 0.2 m edge targets; no motion
```

Keep the robot stationary while recording. Then test motion directly through FrankaPy. Run the checks in order, each only after the previous one behaves. They have no MoveIt collision screening. Each prompts before moving (`--yes` skips the prompt).

```sh
python3 -m catchbox.hardware.calibration --check1-home    # joint move to recorded home; opening at center
python3 -m catchbox.hardware.calibration --check2-small   # 3 cm out and back: left, right, up, down, forward, back
python3 -m catchbox.hardware.calibration --check3-steps   # 5, 10, 15, 20 cm out and back in each direction
python3 -m catchbox.hardware.calibration --check4-catch   # 10 random points inside the sphere, chained
python3 -m catchbox.hardware.calibration --move left --distance 0.05   # home, then one step
```

Left/right are base-frame +Y/-Y, up/down +Z/-Z, forward/back +X/-X. All moves command the box opening and keep the box orientation from home, using FrankaPy's full-arm motion (not the adapter's restricted IK). Targets outside the sphere are rejected; check3 skips steps below `min_opening_z_m` and check4 only picks points above it. After every move the script prints the reached opening position and its error.

Options: `--speed` (opening speed in m/s, default 0.075), `--max-distance` (check3 largest step, check4 sphere radius), `--left`, `--right`, `--up`, `--down`, `--forward`, `--back` to run check3 in only those directions (combine them, e.g. `--left --right`), `--count` and `--seed` (check4).

## Planning scene (optional during bring-up)

MoveIt screening needs `move_group` running with the Panda model, which needs `panda_moveit_config`:

```sh
roslaunch panda_moveit_config move_group.launch load_robot_description:=true
python3 -m catchbox.hardware.planning_scene            # add table + box, then check the home pose
python3 -m catchbox.hardware.planning_scene --remove   # take them out again
```

Both objects are simple boxes, configured under `planning_scene:`. The table is a 2 x 2 m slab with its top at z = 0, since the robot base sits on it; it is attached to `panda_link0` so the base itself does not count as a collision. The catch box is the bounding box of `assets/armbox-thin-meters.STL` plus 1 cm padding, attached to `tip_link` and placed from `opening_offset_m`, assuming the box is level and square to the base at home. The script exits with an error and lists the touching links if the home pose collides. Rerun it whenever `move_group` restarts.

Without MoveIt, run the adapter with `--no-moveit`; the sphere, floor, joint-limit and FrankaPy virtual-wall checks still apply, but nothing checks the arm or box against the table.

## Adapter bring-up

1. Put the robot at home: `python3 -m catchbox.hardware.calibration --check1-home`.
2. With `adapter.enable_motion: false`, start the adapter and send a target:

   ```sh
   python3 -m catchbox.hardware.adapter --no-moveit          # or without --no-moveit once MoveIt is up
   rostopic echo /catchbox_command_adapter/joint_target       # second terminal
   python3 -m catchbox.hardware.send_target left --distance 0.03 --execute   # third terminal
   ```

   Joints 1, 4 and 6 should change smoothly, the others stay at home, and the robot does not move.
3. Set `enable_motion: true`, restart the adapter, and repeat with left/right at 3 cm, then 5 cm. Stop the adapter with Ctrl-C; it ends the FrankaPy skill.

Up/down targets are expected to log "Target unreachable under joint restrictions": the restricted IK only moves joints 1, 4 and 6.
