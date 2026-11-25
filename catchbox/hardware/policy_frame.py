"""Convert between the trained policy's simulation frame and the robot's frame.

The policy saw absolute positions in the simulator's base frame, with the box
opening starting at `sim_start_opening_m`. On the robot the opening starts at
the calibrated `origin_xyz_m`. Both frames share axes (panda_link0), so the
conversion is one translation.
"""
import numpy as np

# Observation slices holding absolute positions: box opening, ball, predicted contact point.
POSITION_SLICES = (slice(0, 3), slice(9, 12), slice(15, 18))
OBSERVATION_SIZE = 25


class PolicyFrame:
    def __init__(self, settings):
        cal, pol = settings['calibration'], settings['policy']
        self.sim_start = np.asarray(pol['sim_start_opening_m'], dtype=float)
        self.sim_center = np.asarray(pol['sim_workspace_center_m'], dtype=float)
        self.sim_radius = float(pol['sim_workspace_radius_m'])
        self.scale = np.asarray(pol['action_scale_m'], dtype=float)
        self.z_low, self.z_high = map(float, pol['action_z_range_m'])
        self.offset = np.asarray(cal['origin_xyz_m'], dtype=float) - self.sim_start

    def observation(self, robot_observation):
        """Robot-frame 25-value observation -> what the policy saw in simulation."""
        obs = np.array(robot_observation, dtype=np.float32)
        if obs.shape != (OBSERVATION_SIZE,):
            raise ValueError(f'Expected {OBSERVATION_SIZE} observation values')
        for s in POSITION_SLICES:
            obs[s] -= self.offset
        return np.clip(obs, -10, 10)

    def target(self, action):
        """Policy action (3 values in [-1, 1]) -> box-opening target in the robot frame.

        Reproduces LimitedPoseCatchCore.step: scale around the start, clip each
        axis, clamp to the simulated workspace sphere, then shift to the robot.
        """
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        if a.shape != (3,):
            raise ValueError('Expected three action values')
        t = self.sim_start + self.scale * a
        t[:2] = np.clip(t[:2], self.sim_start[:2] - self.scale[:2], self.sim_start[:2] + self.scale[:2])
        t[2] = np.clip(t[2], self.sim_start[2] + self.z_low, self.sim_start[2] + self.z_high)
        delta = t - self.sim_center
        distance = np.linalg.norm(delta)
        if distance > self.sim_radius:
            t = self.sim_center + delta * self.sim_radius / distance
        return t + self.offset
