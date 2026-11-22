"""Three-joint level-box catching environment; no training is started here."""
import numpy as np
import mujoco
from .pose_environment import PoseCatchCore
from .trajectory import clamp_workspace


def proximity_reward(distance_m):
    """Continuous reward that increases as the ball approaches the box center."""
    return .2 * float(np.exp(-max(0., float(distance_m)) / .15))


class LimitedPoseCatchCore(PoseCatchCore):
    """Position policy with joint 1, 4 and 6 targets; 1 kHz physics."""
    dt=.02
    action_dim=3

    def restricted_ik(self,target):
        """Optimize joint 1 and 4; counter-rotate 6 to keep box pitch level.

        Joints 2, 3, 5 and 7 remain at the episode's starting positions. The
        reachable positions are a surface; return the nearest point, not a
        claim that every x/y/z target is reachable.
        """
        s=self.sim;m=s.model;ik=s.ikdata;start=self.start_q
        ik.qpos[:]=s.data.qpos;ik.qpos[:7]=start
        bounds=m.jnt_range[:7];margin=.01
        q0min,q0max=bounds[0,0]+margin,bounds[0,1]-margin
        # q6 = start6 - (q4-start4); satisfy both joints' limits.
        const=start[3]+start[5]
        q4min=max(bounds[3,0]+margin,const-(bounds[5,1]-margin))
        q4max=min(bounds[3,1]-margin,const-(bounds[5,0]+margin))
        if q4min>q4max:raise RuntimeError('No joint 4/6 range for level-box constraint')
        ik.qpos[0]=np.clip(self.last_qcmd[0],q0min,q0max)
        ik.qpos[3]=np.clip(self.last_qcmd[3],q4min,q4max)
        ik.qpos[5]=const-ik.qpos[3]
        jacp=np.zeros((3,m.nv));jacr=np.zeros((3,m.nv))
        for _ in range(40):
            mujoco.mj_forward(m,ik)
            error=np.asarray(target)-ik.site_xpos[s.site]
            if np.linalg.norm(error)<1e-4:break
            mujoco.mj_jacSite(m,ik,jacp,jacr,s.site)
            j=np.column_stack((jacp[:,0],jacp[:,3]-jacp[:,5]))
            dq=np.linalg.solve(j.T@j+1e-4*np.eye(2),j.T@error)
            dq=np.clip(dq,-.08,.08)
            if np.linalg.norm(dq)<1e-7:break
            ik.qpos[0]=np.clip(ik.qpos[0]+dq[0],q0min,q0max)
            ik.qpos[3]=np.clip(ik.qpos[3]+dq[1],q4min,q4max)
            ik.qpos[5]=const-ik.qpos[3]
        mujoco.mj_forward(m,ik)
        return ik.qpos[:7].copy(),float(np.linalg.norm(np.asarray(target)-ik.site_xpos[s.site]))

    def step(self,action):
        if self.done:raise RuntimeError('Reset after episode termination')
        a=np.asarray(action,dtype=float)
        if a.shape!=(3,) or not np.isfinite(a).all():raise ValueError('Expected finite three-value position action')
        a=np.clip(a,-1,1);s=self.sim;d=s.data;m=s.model;c=self.cfg
        _,_,_,in_box=self.box_ball_state();pred=self.prediction_now(in_box)
        if pred.valid:self.pred=pred;self.flight=float(d.time)+pred.time_to_contact;self.target=pred.position.copy()
        target=self.start_pos+np.array([.23,.23,.13])*a
        target[:2]=np.clip(target[:2],s.nominal_opening[:2]-.23,s.nominal_opening[:2]+.23)
        target[2]=np.clip(target[2],s.nominal_opening[2]-.13,s.nominal_opening[2]+.08)
        target=clamp_workspace(target,self.workspace_center,self.workspace_radius)
        desired,ik_error=self.restricted_ik(target)
        dt=self.dt;vcmd=np.zeros(7);qcmd=self.start_q.copy()
        for idx in (0,3):
            requested=np.clip((desired[idx]-self.last_qcmd[idx])/dt,-2.175,2.175)
            vcmd[idx]=np.clip(requested,self.last_vcmd[idx]-12*dt,self.last_vcmd[idx]+12*dt)
            qcmd[idx]=self.last_qcmd[idx]+vcmd[idx]*dt
        limits=m.jnt_range[:7];margin=.01
        qcmd[0]=np.clip(qcmd[0],limits[0,0]+margin,limits[0,1]-margin)
        const=self.start_q[3]+self.start_q[5]
        q4min=max(limits[3,0]+margin,const-(limits[5,1]-margin))
        q4max=min(limits[3,1]-margin,const-(limits[5,0]+margin))
        qcmd[3]=np.clip(qcmd[3],q4min,q4max);qcmd[5]=const-qcmd[3]
        vcmd=(qcmd-self.last_qcmd)/dt
        acmd=np.clip((vcmd-self.last_vcmd)/dt,-12,12)
        old_retained=self.success
        # The policy updates every 20 ms; physics and the joint servo remain 1 kHz.
        for k in range(round(dt/c.timestep)):
            fraction=(k+1)*c.timestep/dt
            qt=self.last_qcmd+(qcmd-self.last_qcmd)*fraction
            s.torque(qt,vcmd,acmd);mujoco.mj_step(m,d);mujoco.mj_forward(m,d);self._metrics()
        self.last_qcmd=qcmd;self.last_vcmd=vcmd;self.steps_seen+=1
        if not np.isfinite(d.qpos).all() or any(w.number for w in d.warning):raise RuntimeError('MuJoCo numerical failure')
        local,relative,_=self.local_state()
        # Current ball-to-center distance yields a useful gradient farther away.
        distance=float(np.linalg.norm(local));near=proximity_reward(distance) if self.tracking_valid else 0.
        _,_,_,in_box=self.box_ball_state();current_pred=self.prediction_now(in_box)
        # Do not gate this on the historical, latched contact flag: after a bounce
        # an outside ball can again receive interception reward.
        intercept_error=float(np.linalg.norm(d.site_xpos[s.site]-current_pred.position)) if current_pred.valid else None
        interception=.15*np.exp(-intercept_error**2/.015**2) if intercept_error is not None else 0.
        stable=float(np.exp(-np.dot(local[:2],local[:2])/.025**2)*np.exp(-np.dot(relative,relative)/2)) if in_box else 0.
        terms=dict(proximity=near,interception=float(interception),retention=.4*float(self.inside),stability=.08*stable,success=20.*float(self.success and not old_retained))
        reward=sum(terms.values());self.return_sum+=reward;self.reward_terms=terms
        ball=s.read();lost=bool(ball.position[2]<.04 or np.linalg.norm(ball.position-d.site_xpos[s.site])>2.5)
        terminated=bool(self.success or lost);truncated=bool(d.time>=c.duration-1e-8);self.done=terminated or truncated
        if self.record:self.trace.append(dict(t=float(d.time),qpos=d.qpos.tolist(),ball=ball.position.tolist(),opening=d.site_xpos[s.site].tolist()))
        return self.observation(),float(reward),terminated,truncated,dict(result=self.result(),reward_terms=terms,ik_error_m=ik_error)


def limited_pose_gym_env(seed=0,training=False,total_steps=50000):
    import gymnasium as gym
    class Adapter(gym.Env):
        def __init__(self):
            self.core=LimitedPoseCatchCore(seed,training,total_steps)
            self.action_space=gym.spaces.Box(-1.,1.,shape=(3,),dtype=np.float32)
            self.observation_space=gym.spaces.Box(-10.,10.,shape=(25,),dtype=np.float32)
        def reset(self,seed=None,options=None):
            super().reset(seed=seed)
            return self.core.reset(seed=seed),{}
        def step(self,action):return self.core.step(action)
    return Adapter()
