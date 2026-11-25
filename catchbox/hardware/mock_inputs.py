"""Publish synthetic ball data for pipeline testing. Never starts robot motion.

WARNING: a running adapter with enable_motion=true may move in response.
"""
import argparse, math

def main():
    import rospy
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Int8
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=None);parser.add_argument('--rate',type=float,default=50.)
    parser.add_argument('--seconds',type=float,default=10.)
    parser.add_argument('--mock-box-state',action='store_true',help='Also publish a stationary synthetic box and joints; do not run alongside robot_state')
    args=parser.parse_args(rospy.myargv()[1:])
    if not 0<args.rate<=100 or not 0<args.seconds<=300:raise ValueError('Invalid rate or duration')
    from catchbox.hardware.calibration import BASE_LINK, DEFAULT_CONFIG
    import yaml
    with open(args.config or DEFAULT_CONFIG) as f:settings=yaml.safe_load(f)
    center=settings['calibration']['origin_xyz_m']
    if len(center)!=3 or not all(math.isfinite(float(x)) for x in center):raise ValueError('Calibrate origin first')
    rospy.init_node('catchbox_mock_ball')
    pub=rospy.Publisher('/catchbox/ball_state',Odometry,queue_size=1);flag=rospy.Publisher('/catchbox/ball_in_box',Int8,queue_size=1)
    if args.mock_box_state:
        from sensor_msgs.msg import JointState
        from catchbox.hardware.calibration import JOINT_NAMES
        from catchbox.hardware.panda_kinematics import PandaKinematics
        q=settings['calibration']['home_joints_rad'];kin=PandaKinematics(settings['calibration']['opening_offset_m'])
        position=kin.position(q)
        boxpub=rospy.Publisher('/catchbox/box_state',Odometry,queue_size=1)
        jointpub=rospy.Publisher('/catchbox/robot_joints',JointState,queue_size=1)
    rate=rospy.Rate(args.rate);start=rospy.Time.now().to_sec()
    while not rospy.is_shutdown() and rospy.Time.now().to_sec()-start<args.seconds:
        # A repeating, descending ball trajectory: synthetic measurements in robot base frame.
        t=(rospy.Time.now().to_sec()-start)%1.0
        m=Odometry();m.header.stamp=rospy.Time.now();m.header.frame_id=BASE_LINK;m.child_frame_id=BASE_LINK
        m.pose.pose.position.x=center[0]+.20-.15*t;m.pose.pose.position.y=center[1]+.03
        m.pose.pose.position.z=center[2]+.45+.3*t-4.905*t*t
        m.twist.twist.linear.x=-.15;m.twist.twist.linear.z=.3-9.81*t
        pub.publish(m);flag.publish(Int8(data=0))
        if args.mock_box_state:
            box=Odometry();box.header.stamp=m.header.stamp;box.header.frame_id=BASE_LINK;box.child_frame_id=BASE_LINK
            box.pose.pose.position.x,box.pose.pose.position.y,box.pose.pose.position.z=map(float,position)
            boxpub.publish(box)
            joints=JointState();joints.header.stamp=m.header.stamp;joints.name=JOINT_NAMES;joints.position=q;joints.velocity=[0.]*7;jointpub.publish(joints)
        rate.sleep()
if __name__=='__main__':main()
