import numpy as np
import pytest
import mujoco
from catchbox.hardware.ik import solve_restricted
from catchbox.learning.limited_pose_environment import LimitedPoseCatchCore


def test_restricted_ik_reaches_a_known_reachable_opening():
    core=LimitedPoseCatchCore(seed=4);sim=core.sim;home=core.start_q.copy()
    d=sim.ikdata;d.qpos[:]=sim.data.qpos
    def fk(q):
        d.qpos[:7]=q;mujoco.mj_forward(sim.model,d)
        return d.site_xpos[sim.site].copy()
    expected=home.copy();expected[0]+=.08;expected[3]+=.07;expected[5]-=.07
    target=fk(expected);bounds=sim.model.jnt_range[:7]
    result=solve_restricted(target,home,home,bounds[:,0]+.01,bounds[:,1]-.01,fk,tolerance_m=.001)
    assert result.reachable and result.error_m<.001
    np.testing.assert_allclose(result.joints[[1,2,4,6]],home[[1,2,4,6]])
    assert result.joints[3]+result.joints[5]==pytest.approx(home[3]+home[5])


def test_restricted_ik_rejects_nonfinite_and_reports_unreachable():
    q=np.zeros(7);lo=np.full(7,-1.);hi=np.full(7,1.)
    fk=lambda x:np.array([x[0],x[3],0.])
    r=solve_restricted([0,0,2],q,q,lo,hi,fk)
    assert not r.reachable and r.error_m>=2
    with pytest.raises(ValueError):solve_restricted([float('nan'),0,0],q,q,lo,hi,fk)
