import numpy as np
import pytest
from catchbox.hardware.policy import Actor
from catchbox.hardware.policy_frame import PolicyFrame
from catchbox.hardware.observation import build_observation,build_steady_observation,interception
from catchbox.hardware.panda_kinematics import PandaKinematics,tool_pose

SETTINGS={'calibration':{'origin_xyz_m':[.48,.01,.3]},'policy':{'sim_start_opening_m':[.5547317061010182,0,.17904387158518817], 'sim_workspace_center_m':[.45723170610101843,0,.09904387158518815], 'sim_workspace_radius_m':.2,'action_scale_m':[.23,.23,.13],'action_z_range_m':[-.13,.08]}}

def test_ballistic_crossing_and_missing_prediction_zeros():
    frame=PolicyFrame(SETTINGS);origin=np.array(SETTINGS['calibration']['origin_xyz_m']);ball=origin+[.1,0,.4]
    pred=interception(ball,[0,0,-1],origin[2]);assert pred is not None
    assert pred[0][2]==pytest.approx(origin[2]);assert pred[2]>0 and pred[1][2]<0
    obs=build_observation(frame,origin,[0,0,0],[0,0,0],ball,[0,0,-1],True)
    np.testing.assert_allclose(obs[15:22],0);assert obs[22:25].tolist()==[1,0,1]
    with pytest.raises(ValueError):build_observation(frame,origin,[0,0,0],[0,0,0],ball,[0,0,-1],None)

def test_numpy_actor_matches_saved_policy():
    from sb3_contrib import CrossQ
    model=CrossQ.load('artifacts/limited-thin-model/final_model.zip',device='cpu')
    actor=Actor();rng=np.random.default_rng(7)
    for obs in rng.uniform(-3,3,(100,25)).astype(np.float32):
        np.testing.assert_allclose(actor.predict(obs),model.predict(obs,deterministic=True)[0],rtol=1e-4,atol=1e-4)

def test_robot_opening_twist_matches_finite_displacement():
    kin=PandaKinematics([.07978,-.00136,.04041]);q=np.array([.01,-.44,.007,-2.49,.003,2.06,.8]);dq=np.array([.1,.02,-.03,.04,.01,-.02,.03])
    p,r,v,w=kin.state(q,dq);dt=1e-5
    np.testing.assert_allclose(v,(kin.position(q+dq*dt)-kin.position(q-dq*dt))/(2*dt),atol=1e-8)
    rp=tool_pose(q+dq*dt)[:3,:3];rm=tool_pose(q-dq*dt)[:3,:3];skew=((rp-rm)/(2*dt))@r.T
    np.testing.assert_allclose(w,[skew[2,1],skew[0,2],skew[1,0]],atol=1e-8)


def test_observation_matches_simulator_at_same_state():
    from catchbox.learning.pose_environment import PoseCatchCore
    e=PoseCatchCore(seed=5)
    origin=e.sim.nominal_opening.copy()
    settings={**SETTINGS,'calibration':{'origin_xyz_m':origin.tolist()},'policy':{**SETTINGS['policy'],'sim_start_opening_m':origin.tolist()}}
    frame=PolicyFrame(settings)
    box,linear,angular,in_box=e.box_ball_state();ball=e.sim.read()
    obs=build_observation(frame,box,linear,angular,ball.position,ball.velocity,in_box)
    np.testing.assert_allclose(obs,e.observation(),atol=1e-5)


def test_model_node_gates_throw_and_predicted_intercept(monkeypatch):
    import sys,types
    from types import SimpleNamespace as N
    class Stamp:
        def __init__(self,t):self.t=t
        def to_sec(self):return self.t
        def is_zero(self):return self.t==0
        def __sub__(self,other):return Stamp(self.t-other.t)
    class Time:
        current=100.
        @staticmethod
        def now():return Stamp(Time.current)
    class PointStamped:
        def __init__(self):self.header=N(frame_id='',stamp=None);self.point=N(x=0.,y=0.,z=0.)
    published=[];events=[]
    fake=N(Time=Time,Publisher=lambda *a,**k:N(publish=lambda x:published.append(x)),Subscriber=lambda *a,**k:None)
    monkeypatch.setitem(sys.modules,'nav_msgs.msg',N(Odometry=object))
    monkeypatch.setitem(sys.modules,'geometry_msgs.msg',N(PointStamped=PointStamped))
    monkeypatch.setitem(sys.modules,'std_msgs.msg',N(Int8=object))
    from catchbox.hardware.model_node import ModelNode
    log=N(write=lambda event,**kw:events.append((event,kw)))
    node=ModelNode(SETTINGS,Actor(),fake,log,throw_gate=True)
    def message(p,velocity=(0,0,0),t=None):
        point=N(x=p[0],y=p[1],z=p[2]);v=N(x=velocity[0],y=velocity[1],z=velocity[2]);z=N(x=0.,y=0.,z=0.)
        return N(header=N(stamp=Stamp(Time.current if t is None else t),frame_id='panda_link0'),pose=N(pose=N(position=point)),twist=N(twist=N(linear=v,angular=z)))
    c=np.array(SETTINGS['calibration']['origin_xyz_m'])
    node.on_flag(N(data=0));node.on_status(N(data=2))
    # A held ball, even with the free-fall estimator's downward velocity, cannot start.
    for _ in range(4):
        Time.current+=.03;node.on_box(message(c));node.on_ball(message(c+[.1,0,.4],(0,0,-.6)));assert not node.step()
    assert not published
    # Three moving observations and an in-sphere interception activate the policy.
    for x in (.115,.14,.165):
        Time.current+=.03;node.on_box(message(c));node.on_ball(message(c+[x,0,.4],(0,0,-1)));node.on_flag(N(data=0));node.on_status(N(data=2));node.step()
    assert published
    # An out-of-sphere prediction revokes the throw.
    Time.current+=.03;node.on_box(message(c));node.on_ball(message(c+[.17,0,.4],(2.,0,-1)));assert not node.step()
    assert any(e=='input_wait' and d['reason']=='predicted_intercept_outside_sphere' for e,d in events)
    # Resume an in-zone trajectory, then reject a tracking jump.
    Time.current+=.03;node.on_box(message(c));node.on_ball(message(c+[.18,0,.4],(0,0,-1)));assert node.step()
    Time.current+=.03;node.on_box(message(c));node.on_ball(message(c+[1.,0,.4],(0,0,-1)));assert not node.step()
    Time.current+=.03;node.on_box(message(c));node.on_ball(message(c+[.17,0,.4],(2.,0,-1)));assert not node.step()
    # The jump revoked the throw; a hidden ball cannot reactivate it.
    node.on_status(N(data=0));node.on_flag(N(data=1));assert not node.step()
    # Wait for five clean observations after the jump to flush the velocity fit.
    node.on_flag(N(data=0));node.on_status(N(data=2))
    for x in (.12,.10,.08,.06,.04):
        Time.current+=.03;node.on_box(message(c));node.on_ball(message(c+[x,0,.4],(0,0,-1)));node.on_flag(N(data=0));node.on_status(N(data=2));node.step()
    assert node.throw_gate.active
    # A known catch may continue; a hidden ball of unknown whereabouts may not.
    node.on_status(N(data=0));node.on_flag(N(data=1));assert node.step()
    node.on_flag(N(data=-1));assert not node.step()
    node.on_status(N(data=-1));assert not node.step() and not node.throw_gate.active
    assert any(e=='input_wait' and d['reason']=='predicted_intercept_outside_sphere' for e,d in events)


def test_model_node_throw_gate_is_off_by_default(monkeypatch):
    import sys
    from types import SimpleNamespace as N
    class Stamp:
        def __init__(self,t):self.t=t
        def to_sec(self):return self.t
        def is_zero(self):return self.t==0
        def __sub__(self,other):return Stamp(self.t-other.t)
    class Time:
        current=100.
        @staticmethod
        def now():return Stamp(Time.current)
    class PointStamped:
        def __init__(self):self.header=N(frame_id='',stamp=None);self.point=N(x=0.,y=0.,z=0.)
    published=[];events=[]
    fake=N(Time=Time,Publisher=lambda *a,**k:N(publish=lambda x:published.append(x)),Subscriber=lambda *a,**k:None)
    monkeypatch.setitem(sys.modules,'nav_msgs.msg',N(Odometry=object))
    monkeypatch.setitem(sys.modules,'geometry_msgs.msg',N(PointStamped=PointStamped))
    monkeypatch.setitem(sys.modules,'std_msgs.msg',N(Int8=object))
    from catchbox.hardware.model_node import ModelNode
    node=ModelNode(SETTINGS,Actor(),fake,N(write=lambda event,**kw:events.append((event,kw))))
    assert not node.gate_enabled
    def message(p,velocity=(0,0,0)):
        point=N(x=p[0],y=p[1],z=p[2]);v=N(x=velocity[0],y=velocity[1],z=velocity[2]);z=N(x=0.,y=0.,z=0.)
        return N(header=N(stamp=Stamp(Time.current),frame_id='panda_link0'),pose=N(pose=N(position=point)),twist=N(twist=N(linear=v,angular=z)))
    c=np.array(SETTINGS['calibration']['origin_xyz_m'])
    # A held ball with an out-of-sphere intercept: the gate would withhold it, the default does not.
    Time.current+=.03;node.on_box(message(c));node.on_ball(message(c+[.5,0,.4],(2.,0,-.6)))
    node.on_flag(N(data=0));node.on_status(N(data=2))
    assert node.step() and len(published)==1
    # A hidden ball of unknown whereabouts still gets a target (the pre-gate behaviour).
    node.on_status(N(data=0));node.on_flag(N(data=-1));assert node.step()
    assert not any(e=='throw_accepted' for e,_ in events)


def test_steady_observation_hidden_and_visible_states():
    frame=PolicyFrame(SETTINGS); box=np.array(SETTINGS['calibration']['origin_xyz_m']);ball=box+[.1,0,.4]
    hidden=build_steady_observation(frame,box,[0]*3,[0]*3,None,None,0,-1)
    np.testing.assert_array_equal(hidden[9:22],np.zeros(13))
    assert hidden[22:25].tolist()==[0,0,-1]
    visible=build_steady_observation(frame,box,[0]*3,[0]*3,ball,[5,5,5],1,0)
    np.testing.assert_allclose(visible[9:12],ball-frame.offset,rtol=1e-6,atol=1e-7)
    np.testing.assert_array_equal(visible[12:22],np.zeros(10))
    assert visible[22:25].tolist()==[0,0,0]
    tracked=build_steady_observation(frame,box,[0]*3,[0]*3,ball,[0,0,-1],2,0)
    assert tracked[22:25].tolist()==[1,1,0]


def test_steady_actor_matches_checkpoint():
    from sb3_contrib import CrossQ
    from catchbox.hardware.policy import STEADY_MOTION_ACTOR
    model=CrossQ.load('artifacts/steady-motion-model/final_model.zip',device='cpu')
    actor=Actor(STEADY_MOTION_ACTOR);rng=np.random.default_rng(42)
    for obs in rng.uniform(-3,3,(100,25)).astype(np.float32):
        np.testing.assert_allclose(actor.predict(obs),model.predict(obs,deterministic=True)[0],rtol=1e-4,atol=1e-4)
