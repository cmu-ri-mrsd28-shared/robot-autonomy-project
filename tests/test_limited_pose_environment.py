import numpy as np
import mujoco
from catchbox.learning.limited_pose_environment import LimitedPoseCatchCore,limited_pose_gym_env,proximity_reward


def test_three_joint_action_and_20ms_physics():
    e=LimitedPoseCatchCore(seed=2);t=e.sim.data.time
    obs,reward,done,trunc,info=e.step(np.array([.5,.3,.4]))
    assert obs.shape==(25,) and abs(e.sim.data.time-t-.02)<1e-9
    assert e.action_dim==3 and e.cfg.timestep==.001
    np.testing.assert_allclose(e.last_qcmd[[1,2,4,6]],e.start_q[[1,2,4,6]],atol=1e-12)
    np.testing.assert_allclose(e.last_qcmd[3]+e.last_qcmd[5],e.start_q[3]+e.start_q[5],atol=1e-12)
    assert abs(float(e.sim.data.site_xmat[e.sim.site].reshape(3,3)[2,2])-1)<1e-3
    adapter=limited_pose_gym_env(seed=2)
    assert adapter.action_space.shape==(3,) and adapter.observation_space.shape==(25,)


def test_closer_ball_gets_more_reward():
    assert proximity_reward(0)>proximity_reward(.05)>proximity_reward(.2)>0


def test_interception_reward_resumes_after_latched_contact():
    e=LimitedPoseCatchCore(seed=2);s=e.sim;d=s.data
    # A ball visible outside after prior contact must receive interception shaping.
    e.contact=True;e.first_contact=0.
    p=d.site_xpos[s.site].copy();d.qpos[s.ball_q:s.ball_q+3]=p+np.array([0,0,.12])
    d.qvel[s.ball_v:s.ball_v+3]=[0,0,-1.]
    mujoco.mj_forward(s.model,d)
    _,_,_,_,info=e.step(np.zeros(3))
    assert info['reward_terms']['interception']>0
    assert info['reward_terms']['proximity']>0
