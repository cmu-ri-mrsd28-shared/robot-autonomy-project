"""25-value state and six-value direct box-pose actions for catching."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from catchbox.core import BallState
from .environment import CatchCore
from .trajectory import clamp_workspace

class PoseCatchCore(CatchCore):
    # One policy decision per 1 ms physics step; hardware inference is asynchronous.
    dt=.001
    action_dim=6
    def __init__(self,seed=0,training=False,total_steps=50000,record=False):
        self.tracking_valid=True
        super().__init__('box',seed,training,total_steps,record,printed_box=True)
        # The real ball is 2 g; update both mass and inertia before resetting.
        old=float(self.sim.model.body_mass[self.sim.ball_id]); factor=.002/old
        self.sim.model.body_mass[self.sim.ball_id]=.002
        self.sim.model.body_inertia[self.sim.ball_id]*=factor
        self.cfg.mass=.002
        mujoco.mj_setConst(self.sim.model,self.sim.data)
        self.reset(seed=seed)

    def box_ball_state(self):
        s=self.sim;d=s.data;local,_,linear=safe_local(self)
        angular=s.jacr@d.qvel
        c=self.cfg
        in_box=bool(abs(local[0])<=c.box_width/2 and abs(local[1])<=c.box_depth/2 and -c.box_height<=local[2]<=0)
        return d.site_xpos[s.site].copy(),linear.copy(),angular.copy(),in_box

    def prediction_now(self,in_box):
        raw=self.sim.read()
        # Ballistic prediction is valid only while outside the box and tracking is valid.
        state=BallState(raw.time,raw.position,raw.velocity,valid=bool(self.tracking_valid),acceleration=np.array([0.,0.,-9.81]))
        return self.predictor.predict(state,self.height,in_contact=in_box)

    def observation(self):
        box_pos,box_v,box_w,in_box=self.box_ball_state();ball=self.sim.read();pred=self.prediction_now(in_box)
        p=pred.position if pred.valid else np.zeros(3);v=pred.velocity if pred.valid else np.zeros(3);t=pred.time_to_contact if pred.valid else 0.
        values=np.r_[box_pos,box_v/3,box_w/3,ball.position,ball.velocity/5,p,v/5,np.clip(t,0,2),float(self.tracking_valid),float(pred.valid),float(in_box)]
        assert values.shape==(25,)
        return np.asarray(np.clip(values,-10,10),dtype=np.float32)

    def step(self,action):
        if self.done:raise RuntimeError('Reset after episode termination')
        a=np.asarray(action,dtype=float)
        if a.shape!=(6,) or not np.isfinite(a).all():raise ValueError('Expected finite six-value pose action')
        a=np.clip(a,-1,1);s=self.sim;d=s.data;m=s.model;c=self.cfg
        _,_,_,in_box=self.box_ball_state();pred=self.prediction_now(in_box)
        if pred.valid:self.pred=pred;self.flight=float(d.time)+pred.time_to_contact;self.target=pred.position.copy()
        # Absolute world-frame opening position encoded relative to fixed home, not a correction to a predictor.
        target=self.start_pos+np.array([.23,.23,.13])*a[:3]
        target[:2]=np.clip(target[:2],s.nominal_opening[:2]-.23,s.nominal_opening[:2]+.23)
        target[2]=np.clip(target[2],s.nominal_opening[2]-.13,s.nominal_opening[2]+.08)
        target=clamp_workspace(target,self.workspace_center,self.workspace_radius)
        # XYZ fixed-axis roll, pitch, yaw in radians; bounded to +/- 0.35 rad.
        rotation=Rotation.from_euler('xyz',.35*a[3:]).as_matrix()
        desired,_=s.inverse_kinematics(target,rotation)
        dt=self.dt; v_des=np.clip((desired-self.last_qcmd)/dt,-6.,6.)
        vcmd=np.clip(v_des,self.last_vcmd-12.*dt,self.last_vcmd+12.*dt)
        qcmd=self.last_qcmd+vcmd*dt
        qcmd=np.clip(qcmd,m.jnt_range[:7,0]+.01,m.jnt_range[:7,1]-.01)
        vcmd=(qcmd-self.last_qcmd)/dt;acmd=np.clip((vcmd-self.last_vcmd)/dt,-12.,12.)
        old_retained=self.success
        s.torque(qcmd,vcmd,acmd);mujoco.mj_step(m,d);mujoco.mj_forward(m,d);self._metrics()
        self.last_qcmd=qcmd;self.last_vcmd=vcmd;self.steps_seen+=1
        if not np.isfinite(d.qpos).all() or any(w.number for w in d.warning):raise RuntimeError('MuJoCo numerical failure')
        local,relative,_=self.local_state();error=float(np.linalg.norm(d.site_xpos[s.site]-self.target))
        track=float(np.exp(-error**2/.015**2)) if not self.contact else 0.
        stable=float(np.exp(-np.dot(local[:2],local[:2])/.025**2)*np.exp(-np.dot(relative,relative)/2)) if self.contact else 0.
        # Scale per-step shaping to preserve per-second reward when changing 20 ms to 1 ms.
        terms=dict(interception=.15*track*(dt/.02),retention=.4*float(self.inside)*(dt/.02),stability=.08*stable*(dt/.02),success=20.*float(self.success and not old_retained))
        reward=sum(terms.values());self.return_sum+=reward;self.reward_terms=terms
        ball=s.read();lost=bool(ball.position[2]<.04 or np.linalg.norm(ball.position-d.site_xpos[s.site])>2.5)
        terminated=bool(self.success or lost);truncated=bool(d.time>=c.duration-1e-8);self.done=terminated or truncated
        if self.record and self.steps_seen%20==0:self.trace.append(dict(t=float(d.time),qpos=d.qpos.tolist(),ball=ball.position.tolist(),opening=d.site_xpos[s.site].tolist()))
        return self.observation(),float(reward),terminated,truncated,dict(result=self.result(),reward_terms=terms)

def safe_local(core):
    return core.local_state()

def pose_gym_env(seed=0,training=False,total_steps=50000):
    import gymnasium as gym
    class Adapter(gym.Env):
        def __init__(self):
            self.core=PoseCatchCore(seed,training,total_steps)
            self.action_space=gym.spaces.Box(-1.,1.,shape=(6,),dtype=np.float32)
            self.observation_space=gym.spaces.Box(-10.,10.,shape=(25,),dtype=np.float32)
        def reset(self,seed=None,options=None):
            super().reset(seed=seed)
            return self.core.reset(seed=seed),{}
        def step(self,action):return self.core.step(action)
    return Adapter()
