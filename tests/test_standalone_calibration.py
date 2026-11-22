import numpy as np
import pytest
from catchbox.hardware.calibration import capture_origin, target


def config():
    return dict(tip_link='tool',opening_offset_m=[.1,0,0])


def test_capture_applies_tool_rotation_and_previews_edges():
    class Pose:
        translation=[1,2,3]
        rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    c=capture_origin(config(),Pose(),np.zeros(7))
    np.testing.assert_allclose(c['origin_xyz_m'],[1,2.1,3])
    np.testing.assert_allclose(target(c,'up'),[1,2.1,3.2])
    np.testing.assert_allclose(target(c,'left',.03),[1,2.13,3])
    with pytest.raises(ValueError):target(c,'left',.21)


def test_capture_requires_calibrated_tool_offset():
    c=config();c['opening_offset_m']=None
    with pytest.raises(ValueError,match='opening_offset_m'):
        capture_origin(c,None,np.zeros(7))


def test_shared_configuration_preserves_controller_when_recording(tmp_path, monkeypatch):
    import sys, types, yaml
    from catchbox.hardware import calibration
    p=tmp_path/'configuration.yaml'
    data={'calibration':config(),'catchbox_joint_controller':{'enable_motion':False,'max_velocity':.1}}
    p.write_text("""# Top comment
calibration:
  tip_link: tool
  # Offset comment
  opening_offset_m: [0.1, 0, 0]
  # Center comment
  origin_xyz_m: []
  home_joints_rad: []
catchbox_joint_controller:
  enable_motion: false
  max_velocity: 0.1
""")
    class Pose:
        translation=[1,2,3]
        rotation=np.eye(3)
    class Arm:
        def get_pose(self):return Pose()
        def get_joints(self):return np.zeros(7)
    monkeypatch.setitem(sys.modules,'frankapy',types.SimpleNamespace(FrankaArm=Arm))
    monkeypatch.setattr(sys,'argv',['calibration','--config',str(p),'--set-origin'])
    calibration.main()
    actual=yaml.safe_load(p.read_text())
    np.testing.assert_allclose(actual['calibration']['origin_xyz_m'],[1.1,2,3])
    assert actual['catchbox_joint_controller']==data['catchbox_joint_controller']
    assert '# Top comment' in p.read_text() and '# Offset comment' in p.read_text() and '# Center comment' in p.read_text()
