"""Turn a joint goal into small, smooth steps for streaming to FrankaPy."""
import numpy as np


class JointLimiter:
    """Velocity- and acceleration-limited follower.

    Joint limits are scaled so all joints arrive together; starting from rest
    the commanded path is the straight joint-space line the adapter checked.
    """

    def __init__(self, q, max_velocity, max_acceleration):
        self.q = np.asarray(q, dtype=float).copy()
        self.v = np.zeros(7)
        self.vmax, self.amax = float(max_velocity), float(max_acceleration)
        if self.q.shape != (7,) or not np.isfinite(self.q).all():
            raise ValueError('Expected seven finite joint angles')
        if not (0 < self.vmax <= 2.0 and 0 < self.amax <= 10.0):
            raise ValueError('Joint velocity/acceleration limits out of range')

    def step(self, goal, dt):
        goal = np.asarray(goal, dtype=float)
        error = goal - self.q
        span = np.max(np.abs(error))
        if span < 1e-6 and np.max(np.abs(self.v)) < 1e-4:
            self.q, self.v = goal.copy(), np.zeros(7)
            return self.q.copy()
        share = np.abs(error) / span if span > 0 else np.ones(7)
        vmax, amax = self.vmax * share, self.amax * share
        # Fastest speed from which each joint can still stop at the goal, using
        # the exact discrete-time braking distance for steps of length dt.
        brake = np.maximum(amax, 1e-12) * dt
        stop_speed = brake * (np.sqrt(.25 + 2 * np.abs(error) / (brake * dt)) - .5)
        desired = np.sign(error) * np.minimum(vmax, stop_speed)
        # Speed up with the scaled limits (keeps the line); brake with the full limit.
        braking = (self.v * desired < 0) | (np.abs(desired) < np.abs(self.v))
        bound = np.where(braking, self.amax, amax) * dt
        self.v = self.v + np.clip(desired - self.v, -bound, bound)
        step = self.v * dt
        # Never step past the goal on a joint that is closing in on it.
        closing = np.sign(step) == np.sign(error)
        step = np.where(closing & (np.abs(step) > np.abs(error)), error, step)
        self.q = self.q + step
        return self.q.copy()
