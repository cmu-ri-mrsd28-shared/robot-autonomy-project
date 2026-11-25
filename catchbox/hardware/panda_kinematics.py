"""Panda forward kinematics for the box opening, in panda_link0 (NumPy only).

Uses Franka's published DH parameters and FrankaPy's default tool frame (the
hand TCP: 0.1034 m past the flange, rotated -45 degrees about z). It reproduces
the recorded calibration origins from their home joints to well under 0.1 mm.
"""
import numpy as np

# (a, d, alpha) for joints 1-7, Craig convention, from Franka's documentation.
DH = [(0, .333, 0), (0, 0, -np.pi / 2), (0, .316, np.pi / 2), (.0825, 0, np.pi / 2),
      (-.0825, .384, -np.pi / 2), (0, 0, np.pi / 2), (.088, 0, np.pi / 2)]
FLANGE_D = .107
TCP_D = .1034
TCP_YAW = -np.pi / 4
LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


def _dh(a, d, alpha, theta):
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa],
                     [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1]])


def tool_pose(q):
    """4x4 pose of FrankaPy's tool frame (hand TCP) for seven joint angles."""
    q = np.asarray(q, dtype=float)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError('Expected seven finite joint angles')
    m = np.eye(4)
    for (a, d, alpha), theta in zip(DH, q):
        m = m @ _dh(a, d, alpha, theta)
    m = m @ _dh(0, FLANGE_D, 0, 0)
    tcp = np.eye(4)
    c, s = np.cos(TCP_YAW), np.sin(TCP_YAW)
    tcp[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    tcp[2, 3] = TCP_D
    return m @ tcp


class PandaKinematics:
    """Same interface the IK and adapter use: position(q), lower, upper."""

    def __init__(self, opening_offset):
        self.offset = np.asarray(opening_offset, dtype=float)
        if self.offset.shape != (3,) or not np.isfinite(self.offset).all():
            raise ValueError('opening_offset must be xyz in the tool frame')
        self.lower, self.upper = LOWER.copy(), UPPER.copy()

    def position(self, q):
        m = tool_pose(q)
        return m[:3, 3] + m[:3, :3] @ self.offset
