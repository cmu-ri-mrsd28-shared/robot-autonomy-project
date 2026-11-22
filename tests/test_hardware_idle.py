import numpy as np
from catchbox.hardware.idle import select_target

CENTER=np.zeros(3)


def test_no_input_returns_when_away_and_holds_near_center():
    mode,target,returning=select_target(CENTER,[.05,0,0],None,5,.01)
    assert mode=='return' and returning
    np.testing.assert_allclose(target,CENTER)
    mode,target,returning=select_target(CENTER,[.009,0,0],None,5,.01,returning)
    assert mode=='hold' and target is None and not returning
    assert select_target(CENTER,[.012,0,0],None,5,.01,returning)[0]=='hold'
    assert select_target(CENTER,[.016,0,0],None,5,.01,returning)[0]=='return'


def test_new_target_interrupts_return_and_stale_robot_holds():
    goal=np.array([.1,0,0])
    mode,result,returning=select_target(CENTER,[.05,0,0],goal,.01,.01,True)
    assert mode=='tracking' and not returning
    np.testing.assert_allclose(result,goal)
    assert select_target(CENTER,[.05,0,0],goal,.3,.01)[0]=='hold'
    assert select_target(CENTER,[.05,0,0],goal,2.99,.01)[0]=='hold'
    assert select_target(CENTER,[.05,0,0],goal,3.,.01)[0]=='return'
    assert select_target(CENTER,[.05,0,0],None,0.1,.01)[0]=='hold'
    assert select_target(CENTER,[.05,0,0],goal,.01,.3)[0]=='hold'
    assert select_target(CENTER,[.05,0,0],None,2,.01,input_fault=True)[0]=='hold'
