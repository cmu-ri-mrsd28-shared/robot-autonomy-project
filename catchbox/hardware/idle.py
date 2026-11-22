"""Select tracking, return-to-center or hold behavior outside the realtime loop."""
import numpy as np

FRESHNESS_SECONDS = .2
RETURN_DELAY_SECONDS = 3.0
CENTER_STOP_M = .01
CENTER_RESTART_M = .015


def select_target(center, position, goal, goal_age, state_age, returning=False, input_fault=False):
    """Return (mode, target, returning). Invalid/stale robot state always holds."""
    if position is None or state_age < 0 or state_age > FRESHNESS_SECONDS or input_fault:
        return 'hold', None, False
    if goal is not None and 0 <= goal_age <= FRESHNESS_SECONDS:
        return 'tracking', np.asarray(goal, dtype=float).copy(), False
    if not np.isfinite(goal_age) or goal_age < RETURN_DELAY_SECONDS:
        return 'hold', None, False
    distance = float(np.linalg.norm(np.asarray(position) - np.asarray(center)))
    if not np.isfinite(distance):
        return 'hold', None, False
    # Hysteresis prevents repeat movements if measurement noise straddles 1 cm.
    if distance > (CENTER_STOP_M if returning else CENTER_RESTART_M):
        return 'return', np.asarray(center, dtype=float).copy(), True
    return 'hold', None, False
