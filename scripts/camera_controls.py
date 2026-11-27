"""Optional RGB sensor controls and frame metadata for RealSense recordings."""
from contextlib import contextmanager
import math
import sys


def _color_sensor(rs, profile):
    for sensor in profile.get_device().query_sensors():
        is_color=getattr(sensor,'is_color_sensor',None)
        if callable(is_color) and is_color():
            return sensor
        try:
            name=sensor.get_info(rs.camera_info.name).lower()
        except (AttributeError,RuntimeError):
            continue
        if 'rgb' in name or 'color' in name:
            return sensor
    raise RuntimeError('Could not find the RealSense RGB sensor')


@contextmanager
def rgb_sensor_settings(rs, profile, exposure=None, gain=None):
    """Apply optional raw SDK values during streaming, then restore old settings.

    RGB exposure control units vary by device/backend; use per-frame actual
    exposure metadata in microseconds to determine what the camera delivered.
    """
    if exposure is None and gain is None:
        yield
        return
    sensor=_color_sensor(rs,profile)
    requested=[(rs.option.exposure,exposure,'RGB exposure'),(rs.option.gain,gain,'RGB gain')]
    if not sensor.supports(rs.option.enable_auto_exposure):
        raise RuntimeError('RGB sensor cannot control auto exposure; refusing manual settings')
    for option,value,label in requested:
        if value is None:continue
        if not math.isfinite(value) or not sensor.supports(option):
            raise ValueError(label+' is invalid or unsupported')
        limits=sensor.get_option_range(option)
        if not limits.min <= value <= limits.max:
            raise ValueError('%s %.3f outside device range %.3f..%.3f' % (label,value,limits.min,limits.max))
    saved={rs.option.enable_auto_exposure:sensor.get_option(rs.option.enable_auto_exposure)}
    for option,value,_ in requested:
        if value is not None:saved[option]=sensor.get_option(option)
    try:
        sensor.set_option(rs.option.enable_auto_exposure,0)
        for option,value,label in requested:
            if value is not None:
                sensor.set_option(option,value)
                print('%s: requested %s, device reports %s (SDK units)'%(label,value,sensor.get_option(option)),flush=True)
        print('RGB auto exposure disabled for this session; prior settings will be restored.',flush=True)
        yield
    finally:
        for option in (rs.option.exposure,rs.option.gain,rs.option.enable_auto_exposure):
            if option in saved:
                try:sensor.set_option(option,saved[option])
                except RuntimeError as exc:print('Could not restore RGB setting: %s'%exc,file=sys.stderr,flush=True)


def color_frame_metadata(rs, frame):
    """Actual sensor exposure (microseconds) and gain, when the SDK provides them."""
    result={}
    for key,attribute in [('rgb_exposure_us','actual_exposure'),('rgb_gain','gain_level')]:
        try:
            field=getattr(rs.frame_metadata_value,attribute)
            if frame.supports_frame_metadata(field):result[key]=int(frame.get_frame_metadata(field))
        except (AttributeError,RuntimeError,ValueError):
            pass
    return result
