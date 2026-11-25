import importlib.util
from pathlib import Path
import numpy as np
import pytest
from catchbox.hardware.observation import interception

spec=importlib.util.spec_from_file_location('mock_directions',Path(__file__).parents[1]/'catchbox'/'hardware'/'mock_directions.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

@pytest.mark.parametrize('name,expected',[('left',[0,.05,0]),('right',[0,-.05,0]),('forward',[.05,0,0]),('back',[-.05,0,0])])
def test_predicted_landing_remains_fixed_and_in_sphere(name,expected):
    center=np.array([.48,.01,.3]);initial,velocity,landing=module.make_throw(center,name,.05)
    np.testing.assert_allclose(landing,center+expected,atol=1e-8)
    for t in np.linspace(0,module.FLIGHT_SECONDS-.02,20):
        p,v=module.state_at(initial,velocity,t)
        crossing=interception(p,v,center[2])
        assert crossing is not None
        np.testing.assert_allclose(crossing[0],landing,atol=1e-7)
        assert np.linalg.norm(crossing[0]-center)<=.2

@pytest.mark.parametrize('distance',[-.1,0,.201,float('nan')])
def test_invalid_distance_rejected(distance):
    with pytest.raises(ValueError):module.make_throw([0,0,.3],'left',distance)
