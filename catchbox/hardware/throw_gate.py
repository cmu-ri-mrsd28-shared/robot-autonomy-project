"""Conservative motion evidence for a camera-observed throw.

This is a heuristic, not proof of release: a hand can move a ball too. The
thresholds are initial simulation-test values and require camera validation.
"""
from collections import deque
import numpy as np

WINDOW_S = .25
MAX_GAP_S = .20
MIN_SAMPLES = 3
MIN_DISPLACEMENT_M = .04
MAX_SPEED_M_S = 6.0
POSITION_JITTER_ALLOWANCE_M = .02
# BallOutput fits velocity over five positions. Wait until a rejected point
# has left that fit before trusting the reported velocity again.
CLEAN_SAMPLES_AFTER_JUMP = 5

class ThrowGate:
    def __init__(self):
        self.reset()

    def reset(self):
        self.history = deque()
        self.active = False
        self.last_stamp = None
        self.last_accepted = False
        self.clean_samples_needed = 0

    def observe(self, position, stamp):
        """Return False for a tracking jump; repeated samples are idempotent."""
        stamp = float(stamp);position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(stamp):
            return False
        if self.last_stamp is not None and stamp == self.last_stamp:
            return self.last_accepted
        if self.last_stamp is not None and stamp < self.last_stamp:
            self.history.clear();self.last_stamp=stamp;self.last_accepted=False
            return False
        self.last_stamp=stamp
        if self.history:
            prev_t,prev_pos=self.history[-1];dt=stamp-prev_t
            if dt<=0:
                self.last_accepted=False;return False
            if dt>MAX_GAP_S:
                self.history.clear()
            elif np.linalg.norm(position-prev_pos)>MAX_SPEED_M_S*dt+POSITION_JITTER_ALLOWANCE_M:
                # The camera's velocity fit may still include the rejected point.
                self.history.clear();self.clean_samples_needed=CLEAN_SAMPLES_AFTER_JUMP
                self.last_accepted=False;return False
        self.history.append((stamp,position.copy()))
        if self.clean_samples_needed:
            self.clean_samples_needed-=1
        while self.history and stamp-self.history[0][0]>WINDOW_S:
            self.history.popleft()
        self.last_accepted=True
        return True

    def has_motion(self):
        return (self.clean_samples_needed==0 and len(self.history)>=MIN_SAMPLES and
                np.linalg.norm(self.history[-1][1]-self.history[0][1])>=MIN_DISPLACEMENT_M)
