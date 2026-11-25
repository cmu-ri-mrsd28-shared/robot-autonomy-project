"""Recreate the 25 inputs used by the trained policy."""
import numpy as np

GRAVITY = -9.81

def vec(x, name):
    a=np.asarray(x,dtype=float)
    if a.shape!=(3,) or not np.isfinite(a).all():raise ValueError(name+' must contain three finite values')
    return a

def interception(position, velocity, height, in_box=False):
    p,v=vec(position,'ball position'),vec(velocity,'ball velocity')
    if in_box:return None
    roots=np.roots([.5*GRAVITY,v[2],p[2]-height])
    candidates=[float(t.real) for t in roots if abs(t.imag)<1e-8 and t.real>0 and v[2]+GRAVITY*t.real<0]
    if not candidates:return None
    t=min(candidates);acc=np.array([0.,0.,GRAVITY])
    return p+v*t+.5*acc*t*t,v+acc*t,t

def build_observation(frame, box_position, box_velocity, box_angular_velocity, ball_position, ball_velocity, in_box):
    if not isinstance(in_box,(bool,np.bool_)):raise ValueError('in_box must be explicitly known')
    bp,bv,bw=vec(box_position,'box position'),vec(box_velocity,'box velocity'),vec(box_angular_velocity,'box angular velocity')
    p,v=vec(ball_position,'ball position'),vec(ball_velocity,'ball velocity')
    pred=interception(p,v,float(frame.sim_start[2]+frame.offset[2]),in_box)
    ip,iv,t=pred if pred is not None else (np.zeros(3),np.zeros(3),0.)
    raw=np.r_[bp,bv/3,bw/3,p,v/5,ip,iv/5,np.clip(t,0,2),1.,float(pred is not None),float(in_box)]
    # Translate absolute positions before final clipping, just as in training.
    return frame.observation(raw)
