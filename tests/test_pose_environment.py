import numpy as np
import mujoco
from catchbox.learning.pose_environment import PoseCatchCore
from catchbox.learning.trajectory import freeze_bank,in_workspace
import catchbox.learning.environment as environment_module

def test_pose_interface_and_simulated_ball():
    e=PoseCatchCore(seed=2)
    assert e.observation().shape==(25,)
    assert e.action_dim==6 and e.dt==.001
    assert e.sim.model.body_mass[e.sim.ball_id]==.002
    assert e.workspace_radius==.2
    t=e.sim.data.time
    obs,reward,done,truncated,info=e.step(np.zeros(6))
    assert obs.shape==(25,) and np.isfinite(obs).all()
    assert e.sim.data.time-t==.001

def test_in_box_flag_is_live_and_prediction_can_resume():
    e=PoseCatchCore(seed=1);s=e.sim;d=s.data
    site=d.site_xpos[s.site].copy();r=d.site_xmat[s.site].reshape(3,3)
    d.qpos[s.ball_q:s.ball_q+3]=site+r@np.array([0.,0.,-.04]);d.qvel[s.ball_v:s.ball_v+3]=0
    mujoco.mj_forward(s.model,d);inside=e.observation()
    assert inside[22]==1 and inside[23]==0 and inside[24]==1
    d.qpos[s.ball_q:s.ball_q+3]=site+r@np.array([0.,0.,.12]);d.qvel[s.ball_v:s.ball_v+3]=[0,0,-1]
    mujoco.mj_forward(s.model,d);outside=e.observation()
    assert outside[22]==1 and outside[23]==1 and outside[24]==0
    e.tracking_valid=False;unknown=e.observation()
    assert unknown[22]==0 and unknown[23]==0

def test_printed_box_heldout_bank_respects_workspace():
    e=PoseCatchCore(seed=2)
    bank=freeze_bank(e.sim,20261210,8)
    assert bank['workspace_radius_m']==.2
    assert all(in_workspace(x['meta']['target'],bank['workspace_center'],.2) for x in bank['throws'])

def test_printed_box_can_retry_more_than_100_rejected_candidates(monkeypatch):
    e=PoseCatchCore(seed=2)
    real=environment_module.varied_throw
    attempts=[0]
    def candidates(*args,**kwargs):
        throw,meta=real(*args,**kwargs)
        attempts[0]+=1
        if attempts[0]<=100:
            meta['target']=(e.workspace_center+np.array([2.,0.,0.])).tolist()
        return throw,meta
    monkeypatch.setattr(environment_module,'varied_throw',candidates)
    e.reset(seed=3)
    assert e.rejected>=100
