import sys,types
from types import SimpleNamespace as N
import numpy as np
from catchbox.hardware.mock_robot import MockRobot
from catchbox.hardware.calibration import JOINT_NAMES
from test_hardware_adapter import SETTINGS, HOME

class JointState:
    def __init__(self):self.header=N(stamp=None);self.name=[];self.position=[];self.velocity=[]

class Odometry:
    def __init__(self):
        vec=lambda:N(x=0.,y=0.,z=0.)
        self.header=N(stamp=None,frame_id='');self.child_frame_id=''
        self.pose=N(pose=N(position=vec(),orientation=N(x=0.,y=0.,z=0.,w=1.)))
        self.twist=N(twist=N(linear=vec(),angular=vec()))

def test_mock_robot_publishes_feedback_from_commands(monkeypatch):
    monkeypatch.setitem(sys.modules,'sensor_msgs.msg',N(JointState=JointState))
    monkeypatch.setitem(sys.modules,'nav_msgs.msg',N(Odometry=Odometry))
    outputs={}
    rospy=N(Time=N(now=lambda:123),Publisher=lambda name,*a,**k:N(publish=lambda msg:outputs.update({name:msg})),Subscriber=lambda *a,**k:None)
    robot=MockRobot(SETTINGS,rospy);robot.step()
    np.testing.assert_allclose(outputs['/catchbox/robot_joints'].position,HOME)
    msg=JointState();msg.name=JOINT_NAMES;msg.position=(np.array(HOME)+[.03,0,0,0,0,0,0]).tolist()
    robot.on_target(msg);robot.step()
    np.testing.assert_allclose(outputs['/catchbox/robot_joints'].position,msg.position)
    assert outputs['/catchbox/box_state'].header.frame_id=='panda_link0'


def test_mock_adapter_forces_motion_off(monkeypatch,tmp_path):
    import yaml
    from catchbox.hardware import adapter
    config=tmp_path/'config.yaml';data={**SETTINGS,'adapter':{**SETTINGS['adapter'],'enable_motion':True}}
    config.write_text(yaml.safe_dump(data))
    called={}
    class FakeAdapter:
        def __init__(self, settings, arm):called['enabled']=settings['adapter']['enable_motion'];called['joints']=arm.get_joints()
        def run(self):called['ran']=True
    class FakeLog:
        def __init__(self,*args):pass
        def close(self):pass
    monkeypatch.setitem(sys.modules,'rospy',N(myargv=lambda:['node','--config',str(config),'--mock-robot'],init_node=lambda *a,**k:None))
    monkeypatch.setattr(adapter,'Adapter',FakeAdapter)
    import catchbox.hardware.event_log as event_log
    monkeypatch.setattr(event_log,'EventLog',FakeLog)
    # A FrankaPy import is neither available nor needed in mock mode.
    monkeypatch.setitem(sys.modules,'frankapy',None)
    adapter.main()
    assert called['ran'] and not called['enabled']
    np.testing.assert_allclose(called['joints'],HOME)
