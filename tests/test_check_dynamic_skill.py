import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

spec=importlib.util.spec_from_file_location('check_dynamic_skill',Path(__file__).parents[1]/'scripts'/'check_dynamic_skill.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_hold_sends_and_closes(monkeypatch):
    ticks=iter([0.,0.,0.1,0.2])
    monkeypatch.setattr(module.time,'monotonic',lambda:next(ticks))
    events=[]
    class Stream:
        def __init__(self,arm,q):events.append('start')
        def send(self,q):events.append('send')
        def close(self):events.append('close')
    arm=SimpleNamespace(get_joints=lambda:[0.]*7,is_skill_done=lambda **kwargs:False)
    ros=SimpleNamespace(Rate=lambda hz:SimpleNamespace(sleep=lambda:None),is_shutdown=lambda:False)
    module.check_hold(arm,ros,Stream,seconds=.15)
    assert events==['start','send','send','close']


def test_skill_ending_is_reported_and_closed():
    events=[]
    class Stream:
        def __init__(self,arm,q):pass
        def send(self,q):events.append('send')
        def close(self):events.append('close')
    arm=SimpleNamespace(get_joints=lambda:[0.]*7,is_skill_done=lambda **kwargs:True)
    ros=SimpleNamespace(Rate=lambda hz:SimpleNamespace(sleep=lambda:None),is_shutdown=lambda:False)
    with pytest.raises(RuntimeError,match='ended early'):
        module.check_hold(arm,ros,Stream)
    assert events==['send','close']
