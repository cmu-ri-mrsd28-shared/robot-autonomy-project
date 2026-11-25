"""Publish a labeled sequence of simulated throws; no real ball is used.

Each throw predicts a landing point inside the sphere at the fixed opening
height. An active adapter may move the REAL robot. Stop other ball publishers.
"""
import argparse
import math
import time
import numpy as np

from catchbox.hardware.calibration import BASE_LINK, DEFAULT_CONFIG, WORKSPACE_RADIUS_M, vector
from catchbox.hardware.observation import interception

# Base-frame axes: left/right are +/- Y; forward/back are +/- X.
DIRECTIONS = {'left':(0.,1.,0.), 'right':(0.,-1.,0.),
              'forward':(1.,0.,0.), 'back':(-1.,0.,0.)}
GRAVITY = np.array([0.,0.,-9.81])
FLIGHT_SECONDS = .65
CATCH_SIGNAL_SECONDS = 1.0
IDLE_SECONDS = 4.0


def make_throw(center, direction, distance):
    """Construct and check a trajectory and its fixed-height predicted landing."""
    center = vector(center, 3, 'origin_xyz_m')
    if direction not in DIRECTIONS:
        raise ValueError('Unknown direction')
    if not math.isfinite(distance) or not 0.0 < distance <= WORKSPACE_RADIUS_M:
        raise ValueError('distance must be within the 0.2 m sphere')
    landing = center + distance*np.asarray(DIRECTIONS[direction])
    initial = landing + np.array([.25, 0., .35])
    velocity = (landing-initial-.5*GRAVITY*FLIGHT_SECONDS**2)/FLIGHT_SECONDS
    prediction = interception(initial,velocity,center[2])
    if prediction is None or not np.allclose(prediction[0],landing,atol=1e-7):
        raise ValueError('Could not generate the requested predicted landing')
    if np.linalg.norm(prediction[0]-center)>WORKSPACE_RADIUS_M+1e-9:
        raise ValueError('Predicted landing outside sphere')
    return initial,velocity,prediction[0]


def state_at(initial, velocity, t):
    return initial+velocity*t+.5*GRAVITY*t*t, velocity+GRAVITY*t


def main():
    import rospy, yaml
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Int8
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(DEFAULT_CONFIG))
    parser.add_argument('--distance',type=float,default=.05,help='Landing offset in meters (default 0.05)')
    parser.add_argument('--direction',choices=list(DIRECTIONS),help='Run just one throw')
    parser.add_argument('--rate',type=float,default=50.)
    args=parser.parse_args(rospy.myargv()[1:])
    if not 0 < args.rate <= 100:raise ValueError('rate must be between 0 and 100 Hz')
    with open(args.config) as f:settings=yaml.safe_load(f)
    center=vector(settings['calibration']['origin_xyz_m'],3,'origin_xyz_m')
    cases=[(name,make_throw(center,name,args.distance)) for name in ([args.direction] if args.direction else DIRECTIONS)]
    rospy.init_node('catchbox_mock_directions')
    ball_pub=rospy.Publisher('/catchbox/ball_state',Odometry,queue_size=1)
    flag_pub=rospy.Publisher('/catchbox/ball_in_box',Int8,queue_size=1)
    status_pub=rospy.Publisher('/catchbox/ball_tracking_status',Int8,queue_size=1)
    rate=rospy.Rate(args.rate)
    print('Fake data can MOVE the REAL robot if the adapter has motion enabled.',flush=True)
    print('Predictions cross the fixed opening-height plane; up/down landings are not supported.',flush=True)
    for name,(initial,velocity,landing) in cases:
        if rospy.is_shutdown():break
        print('THROW %s: predicted landing %s m; %.1f cm from center' %
              (name, np.round(landing,4).tolist(), 100*np.linalg.norm(landing-center)),flush=True)
        start=time.monotonic(); announced_catch=False;announced_idle=False
        while not rospy.is_shutdown():
            elapsed=time.monotonic()-start
            if elapsed >= FLIGHT_SECONDS+CATCH_SIGNAL_SECONDS+IDLE_SECONDS:break
            if elapsed < FLIGHT_SECONDS-.02:
                p,v=state_at(initial,velocity,elapsed)
                msg=Odometry();msg.header.stamp=rospy.Time.now();msg.header.frame_id=BASE_LINK;msg.child_frame_id=BASE_LINK
                msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z=map(float,p)
                msg.twist.twist.linear.x,msg.twist.twist.linear.y,msg.twist.twist.linear.z=map(float,v)
                ball_pub.publish(msg);flag_pub.publish(Int8(data=0));status_pub.publish(Int8(data=2))
            elif elapsed < FLIGHT_SECONDS+CATCH_SIGNAL_SECONDS:
                if not announced_catch:
                    print('  Simulated catch: ball hidden, probably inside (not a real detection).',flush=True);announced_catch=True
                flag_pub.publish(Int8(data=1));status_pub.publish(Int8(data=0))
            else:
                if not announced_idle:
                    print('  No active ball; waiting for return to center.',flush=True);announced_idle=True
                flag_pub.publish(Int8(data=-1));status_pub.publish(Int8(data=-1))
            rate.sleep()
    print('Mock throw sequence finished.',flush=True)

if __name__=='__main__':main()
