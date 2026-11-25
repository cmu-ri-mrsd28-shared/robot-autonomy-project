import json
import threading
from types import SimpleNamespace
import numpy as np
from catchbox.hardware import adapter
from catchbox.hardware.event_log import EventLog


def test_logger_reports_drops_on_close(tmp_path):
    path=tmp_path/'events.jsonl';log=EventLog(path)
    log.write('sample',value=3);log.dropped=2;log.close()
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]['event']=='sample'
    assert rows[-1]['event']=='log_summary' and rows[-1]['dropped']==2


def test_adapter_logs_every_command_cycle(monkeypatch):
    entries=[];counter={'n':0}
    class Thread:
        def __init__(self,**kwargs):pass
        def start(self):pass
    class Stamp:
        def to_sec(self):return 1.
    class JointState:
        def __init__(self):self.header=SimpleNamespace(stamp=None);self.name=None;self.position=None
    def sleep():counter['n']+=1
    rospy=SimpleNamespace(loginfo=lambda *a:None,Rate=lambda hz:SimpleNamespace(sleep=sleep),
         is_shutdown=lambda:counter['n']>=5,Time=SimpleNamespace(now=lambda:Stamp()))
    monkeypatch.setattr(adapter.threading,'Thread',Thread)
    a=adapter.Adapter.__new__(adapter.Adapter);a.motion=False;a.arm=None;a.rate_hz=100
    a.limiter=SimpleNamespace(q=np.zeros(7),v=np.zeros(7),step=lambda goal,dt:np.zeros(7))
    a.lock=threading.Lock();a.joint_goal=np.zeros(7);a.rospy=rospy;a.JointState=JointState
    a.kin=SimpleNamespace(position=lambda q:np.zeros(3));a.space=SimpleNamespace(contains=lambda p,tol:True)
    a.debug=SimpleNamespace(publish=lambda msg:None);a.fault=None
    a.log=SimpleNamespace(write=lambda event,**data:entries.append((event,data)))
    a.run()
    commands=[data for event,data in entries if event=='joint_command']
    assert len(commands)==5
    assert all('loop_ms' in x and 'limiter_step_ms' in x for x in commands)
