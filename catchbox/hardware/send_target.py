"""Publish a box-opening test target to the adapter for a few seconds.

    python3 -m catchbox.hardware.send_target left --distance 0.03   # preview only
    python3 -m catchbox.hardware.send_target left --distance 0.03 --execute

After publishing stops, the adapter holds, then returns to center after 3 s.
"""
import argparse
import time
from pathlib import Path

from catchbox.hardware.calibration import AXES, BASE_LINK, DEFAULT_CONFIG, target, validate_config

TOPIC = '/catchbox_command_adapter/opening_target'


def main():
    import rospy
    import yaml
    from geometry_msgs.msg import PointStamped
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('direction', choices=list(AXES))
    parser.add_argument('--distance', type=float, default=.03, help='Offset in meters (default 0.03)')
    parser.add_argument('--duration', type=float, default=5, help='Seconds to publish (default 5)')
    parser.add_argument('--execute', action='store_true', help='Publish; without it, only print the target')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(rospy.myargv()[1:])
    cfg = yaml.safe_load(args.config.read_text())['calibration']
    validate_config(cfg)
    if not 0 < args.duration <= 60:
        parser.error('--duration must be between 0 and 60 s')
    goal = target(cfg, args.direction, args.distance)
    print('Base-frame target:', goal.tolist())
    if not args.execute:
        print('Preview only; add --execute to send it to the adapter.')
        return
    rospy.init_node('catchbox_send_target', anonymous=True)
    pub = rospy.Publisher(TOPIC, PointStamped, queue_size=1)
    rate, deadline = rospy.Rate(20), time.monotonic() + args.duration
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        msg = PointStamped()
        msg.header.frame_id, msg.header.stamp = BASE_LINK, rospy.Time.now()
        msg.point.x, msg.point.y, msg.point.z = map(float, goal)
        pub.publish(msg)
        rate.sleep()
    print('Stopped publishing; the adapter will hold, then return to center.')


if __name__ == '__main__':
    main()
