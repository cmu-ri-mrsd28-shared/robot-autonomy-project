"""A status check cannot precede creation of the dynamic skill."""
from types import SimpleNamespace
from catchbox.hardware import adapter


def test_start_skill_before_planner(monkeypatch):
    events=[]
    class Stream:
        def __init__(self, arm, q): events.append('start skill')
        def close(self): events.append('close skill')
    class Thread:
        def __init__(self, target, daemon): pass
        def start(self): events.append('start planner')
    monkeypatch.setattr(adapter,'Stream',Stream)
    monkeypatch.setattr(adapter.threading,'Thread',Thread)
    instance=adapter.Adapter.__new__(adapter.Adapter)
    instance.motion=True;instance.arm=object();instance.limiter=SimpleNamespace(q=[0]*7)
    instance.rospy=SimpleNamespace(loginfo=lambda *a:None,Rate=lambda hz:None,is_shutdown=lambda:True)
    instance.rate_hz=100;instance.fault=None
    instance.run()
    assert events == ['start skill','start planner','close skill']
