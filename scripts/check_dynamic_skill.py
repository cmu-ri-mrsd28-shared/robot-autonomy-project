"""Test the FrankaPy dynamic joint stream while holding current joint positions.

Run only while the normal catchbox adapter and any other skill are stopped:
    python3 scripts/check_dynamic_skill.py
This sends real robot commands for three seconds. It never operates the gripper.
"""
import sys
import time
import numpy as np


def check_hold(arm, rospy, stream_class, seconds=3.0):
    """Start the same stream as the adapter, report early exits, then terminate."""
    q = np.asarray(arm.get_joints(), dtype=float)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError('FrankaPy returned invalid joint positions')
    print('Starting dynamic joint skill at current joint positions.', flush=True)
    stream = stream_class(arm, q)
    try:
        start = time.monotonic()
        rate = rospy.Rate(100)
        n = 0
        while time.monotonic() - start < seconds and not rospy.is_shutdown():
            stream.send(q)
            n += 1
            if arm.is_skill_done(ignore_errors=False):
                raise RuntimeError('Dynamic skill ended early after %.2f seconds (%d targets sent). See FrankaPy/control-PC errors.' % (time.monotonic()-start, n))
            rate.sleep()
        if rospy.is_shutdown():
            raise RuntimeError('ROS shut down before the check finished')
        print('PASS: dynamic skill remained active for %.1f seconds; sent %d hold targets.' % (seconds, n), flush=True)
    finally:
        stream.close()


def main():
    import rospy
    from frankapy import FrankaArm
    from catchbox.hardware.adapter import Stream
    print('This test sends hold commands to the REAL arm for 3 seconds.', flush=True)
    print('Stop the adapter and other arm-control scripts first; keep the e-stop accessible.', flush=True)
    rospy.init_node('catchbox_dynamic_skill_check')
    arm = FrankaArm(init_node=False, with_gripper=False)
    check_hold(arm, rospy, Stream)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('FAIL: %s: %s' % (type(exc).__name__, exc), file=sys.stderr, flush=True)
        raise
