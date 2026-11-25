# Steady-motion experiment

This is simulation-only. No training begins by importing these scripts.

- The fresh CrossQ actor has 25 inputs, three outputs, and two 128-unit hidden layers. Its twin critics use two 256-unit hidden layers each. The earlier original checkpoint is read, not modified.
- Episode category is chosen once: 70% incoming throws, 30% overhead drops in expectation; candidates are rejected within the chosen category until a valid one is found. Drops do not guarantee bounces. The outcome depends on contact physics.
- The simulated camera assumes a fixed position `[1.2, 0, 1.4]` meters in the simulation base frame. This is an uncalibrated placeholder, not a measured hardware pose. Visibility is a center-ray test with idealized optics, 2 mm position noise, and no latency or frame-rate limitation. Camera field of view and partial ball visibility are not modeled.
- Occluded position and velocity are zero, and tracking and prediction flags are false. The last flag uses -1 for unknown, 0 for visible/outside, 1 for probably in box. This differs from the original binary flag convention and requires corresponding hardware adapter work before deployment. Probable in-box is a fallible inference, never a simulated contact sensor. Robot state is assumed available.
- Reward per step is `-.02*r_dist + .08*r_in_box + .8*r_catch + .1*r_steady`. Distance is squared relative to 20 cm, capped at 1 and disabled inside. Catch is a one-time 1-second retention event. The stillness reward is inside times `max(0, 1-speed/.05)`. The 5 cm/s threshold is provisional. Reward uses simulator truth, while ball observations use the camera estimator. The capped distance gives no gradient beyond 20 cm.
- Episodes continue after retention to expose bounces. A 3.5-second time limit remains, so late catches may have little follow-up time.
- The independent movement-required bank rejects cases where a stationary box enters/catches or where the actual descending crossing does not clear the fixed box. It is conditioned on feasibility and is not a general throw distribution. The overhead-drop bank contains a variety of drops; it is not guaranteed to contain bounces. Report recovery only for an observed escape followed by one second inside, and report how many escapes occurred.

Generate the test banks with `scripts/prepare_steady_motion_evaluation.py`. The training runner `scripts/train_steady_motion_50k.py` requires the resulting banks and refuses to overwrite an existing run. It evaluates the original and new policy using the same simulated camera, but the original was not trained with the new unknown-flag convention; that is a distribution shift and must be reported.

Neither the simulated ray caster nor camera estimator is integrated into the hardware ROS camera node. The current hardware policy node withholds targets when ball tracking is invalid, whereas this simulation allows the policy to act during occlusion; this mismatch must be resolved before deployment. Hardware camera occlusion, latency, camera calibration, grasp, and tracking remain unvalidated.

## Saved model and control-PC inference

`artifacts/steady-motion-model/final_model.zip` contains the trained CrossQ checkpoint. `artifacts/steady-motion-model/actor.npz` contains its deterministic CPU actor; it was checked against the checkpoint on 1,000 inputs. Both are needed because the actor loader checks the checkpoint checksum. The original model remains in `artifacts/limited-thin-model/`.

For standalone CPU inference after building the correct 25-value observation:

```python
from catchbox.hardware.policy import Actor

actor = Actor('artifacts/steady-motion-model/actor.npz')
action = actor.predict(observation)  # 25 finite values; returns 3 bounded values
```

The existing `model_node.py` still loads the original model and does not implement the steady-motion model's occlusion/unknown observations. **Do not replace its actor file or run the steady-motion model through that node yet.** A camera-to-policy adapter must implement the observation convention above, including unknown-state handling, before hardware control. This inference example does not control the robot.

## Simulation results

On the independently generated 50 movement-required throws, the original retained 40 for one second and 8 for two seconds; steady-motion retained 39 and 39. On 50 overhead drops, the original retained 42 and 15; steady-motion retained 46 and 46. Both were evaluated using the new simulated camera; the original was trained under different sensing assumptions. No escape above the box after entry was recorded, so these results do not establish bounce recovery. See `data/steady-motion-evaluation/manifest.json` for sampling limitations.
