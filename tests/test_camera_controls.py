from types import SimpleNamespace as N
import pytest
from camera_controls import rgb_sensor_settings,color_frame_metadata

class Sensor:
    def __init__(self):self.values={'ae':1,'exposure':156.,'gain':64.};self.writes=[]
    def is_color_sensor(self):return True
    def supports(self,opt):return opt in self.values
    def get_option(self,opt):return self.values[opt]
    def get_option_range(self,opt):return N(min=1.,max=1000.,step=1.)
    def set_option(self,opt,value):self.values[opt]=value;self.writes.append((opt,value))

RS=N(option=N(exposure='exposure',gain='gain',enable_auto_exposure='ae'),
     camera_info=N(name='name'),frame_metadata_value=N(actual_exposure='exp',gain_level='gain'))

def test_default_does_not_touch_sensor():
    with rgb_sensor_settings(RS,object()):pass

def test_manual_settings_are_restored_even_on_failure():
    sensor=Sensor();profile=N(get_device=lambda:N(query_sensors=lambda:[sensor]))
    with pytest.raises(RuntimeError,match='test'):
        with rgb_sensor_settings(RS,profile,exposure=30.,gain=70.):
            assert sensor.values=={'ae':0,'exposure':30.,'gain':70.}
            raise RuntimeError('test')
    assert sensor.values=={'ae':1,'exposure':156.,'gain':64.}

def test_invalid_exposure_does_not_change_device():
    sensor=Sensor();profile=N(get_device=lambda:N(query_sensors=lambda:[sensor]))
    with pytest.raises(ValueError):
        with rgb_sensor_settings(RS,profile,exposure=2000.):pass
    assert sensor.writes==[]

def test_color_metadata_reports_actual_exposure():
    frame=N(supports_frame_metadata=lambda f:True,get_frame_metadata=lambda f:{'exp':3000,'gain':80}[f])
    assert color_frame_metadata(RS,frame)=={'rgb_exposure_us':3000,'rgb_gain':80}
    assert color_frame_metadata(N(),object())=={}


def test_recording_keeps_frame_metadata():
    import numpy as np
    import measure_ball as mb
    recorder=mb.Recorder(N(),np,'unused',N(width=2,height=2),'serial',30)
    recorder.video=N(write=lambda image:None);recorder.rows=[];recorder.depth=[];recorder.started=0.;recorder.units=None
    lock=N(state='SEARCHING',ball_id=None,in_box_balls=[],box=None)
    recorder.add(.1,np.zeros((2,2,3),dtype=np.uint8),None,lock,None,[],{'rgb_exposure_us':3000,'rgb_gain':80})
    assert recorder.rows[0]['rgb_exposure_us']==3000
    assert recorder.rows[0]['rgb_gain']==80
    assert {'rgb_exposure_us','rgb_gain'} <= set(mb.RECORD_FIELDS)
