"""Publish measured robot joints and box-opening state. No motion commands.

Run: python3 -m catchbox.hardware.robot_state
"""
import argparse, time
import numpy as np
from catchbox.hardware.calibration import BASE_LINK, DEFAULT_CONFIG, JOINT_NAMES
from catchbox.hardware.event_log import EventLog
from catchbox.hardware.panda_kinematics import PandaKinematics
from catchbox.hardware.rotation import quaternion_from_matrix

def main():
    import rospy, yaml
    from sensor_msgs.msg import JointState
    from nav_msgs.msg import Odometry
    from frankapy import FrankaArm
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(DEFAULT_CONFIG));parser.add_argument('--rate',type=float,default=50.)
    parser.add_argument('--log',default='logs/robot_state.jsonl');args=parser.parse_args(rospy.myargv()[1:])
    if not 0<args.rate<=100:raise ValueError('rate must be between 0 and 100 Hz')
    with open(args.config) as f:settings=yaml.safe_load(f)
    kin=PandaKinematics(settings['calibration']['opening_offset_m'])
    rospy.init_node('catchbox_robot_state');arm=FrankaArm(init_node=False,with_gripper=False)
    joints=rospy.Publisher('/catchbox/robot_joints',JointState,queue_size=1)
    opening=rospy.Publisher('/catchbox/box_state',Odometry,queue_size=1)
    log=EventLog(args.log);rate=rospy.Rate(args.rate)
    try:
        while not rospy.is_shutdown():
            started=time.monotonic()
            try:
                q=np.asarray(arm.get_joints(),dtype=float);dq=np.asarray(arm.get_joint_velocities(),dtype=float)
                p,r,v,w=kin.state(q,dq);quat=quaternion_from_matrix(r);stamp=rospy.Time.now()
                js=JointState();js.header.stamp=stamp;js.name=JOINT_NAMES;js.position=q.tolist();js.velocity=dq.tolist()
                msg=Odometry();msg.header.stamp=stamp;msg.header.frame_id=BASE_LINK;msg.child_frame_id=BASE_LINK
                msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z=map(float,p)
                msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w=map(float,quat)
                msg.twist.twist.linear.x,msg.twist.twist.linear.y,msg.twist.twist.linear.z=map(float,v)
                msg.twist.twist.angular.x,msg.twist.twist.angular.y,msg.twist.twist.angular.z=map(float,w)
                joints.publish(js);opening.publish(msg)
                log.write('robot_state',stamp_s=stamp.to_sec(),poll_and_fk_ms=1000*(time.monotonic()-started),joints=q.tolist(),opening=p.tolist(),linear_velocity=v.tolist(),angular_velocity=w.tolist())
            except Exception as exc:
                rospy.logerr_throttle(2,'Robot state unavailable: %s',exc);log.write('state_error',error=str(exc))
            rate.sleep()
    finally:log.close()
if __name__=='__main__':main()
