"""Restricted IK: joints 1 and 4 translate, joint 6 counter-rotates joint 4."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class IKResult:
    joints: np.ndarray
    error_m: float
    reachable: bool


def solve_restricted(target, home, current, lower, upper, fk_position, tolerance_m=.005):
    """Closest opening position using a calibrated base-frame FK callable.

    Joint numbering is one-based in the robot documentation, zero-based here.
    Fixed joints use home positions. Joint 6 offsets joint 4 to match simulation.
    The caller must independently check the entire path for collisions.
    """
    target, home, current, lower, upper = [np.asarray(x, dtype=float) for x in
                                           (target, home, current, lower, upper)]
    if target.shape != (3,) or any(x.shape != (7,) for x in (home,current,lower,upper)):
        raise ValueError('Expected target xyz and four seven-joint vectors')
    if not all(np.isfinite(x).all() for x in (target,home,current,lower,upper)):
        raise ValueError('Non-finite IK input')
    if np.any(lower >= upper) or np.any(home < lower) or np.any(home > upper):
        raise ValueError('Invalid joint limits or home pose')
    const=home[3]+home[5]
    lo=np.array([lower[0],max(lower[3],const-upper[5])])
    hi=np.array([upper[0],min(upper[3],const-lower[5])])
    if np.any(lo>=hi):raise ValueError('No joint 4/6 range at this home')
    def joints(v):
        q=home.copy();q[0],q[3]=v;q[5]=const-q[3];return q
    def residual(v):
        p=np.asarray(fk_position(joints(v)),float)
        if p.shape!=(3,) or not np.isfinite(p).all():raise ValueError('Invalid FK output')
        return p-target
    x0=np.maximum(lo+1e-9,np.minimum(hi-1e-9,current[[0,3]]))
    result=least_squares(residual,x0,bounds=(lo,hi),max_nfev=80,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    q=joints(result.x);error=float(np.linalg.norm(residual(result.x)))
    return IKResult(q,error,bool(result.success and error<=tolerance_m))
