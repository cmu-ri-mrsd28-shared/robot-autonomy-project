#!/usr/bin/env python3
"""Convert base-frame opening targets to checked, restricted joint targets.

This non-real-time node does not load the policy or estimate camera state.
"""
import argparse, os, sys, threading, time
import numpy as np
import rospy, yaml
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest
from catchbox.hardware.calibration import BASE_LINK, JOINT_NAMES, MOVE_GROUP, WORKSPACE_RADIUS_M, DEFAULT_CONFIG, validate_config
from catchbox.hardware.ik import solve_restricted
from catchbox.hardware.ros_kinematics import KDLKinematics

class Controller:
    def __init__(self, calibration):
        validate_config(calibration)
        self.cfg=calibration;self.names=JOINT_NAMES;self.lock=threading.Lock()
        self.origin=np.asarray(calibration['origin_xyz_m'],float);self.home=np.asarray(calibration['home_joints_rad'],float)
        self.radius=WORKSPACE_RADIUS_M;self.base=BASE_LINK;self.group=MOVE_GROUP
        self.kin=KDLKinematics(self.base,calibration['tip_link'],self.names,calibration['opening_offset_m'])
        if len(self.names)!=7 or self.origin.shape!=(3,) or self.home.shape!=(7,) or not np.isfinite(np.r_[self.origin,self.home,self.radius]).all() or not 0<self.radius<=.2:
            raise ValueError('Invalid calibration or radius; maximum is 0.2 m')
        if np.linalg.norm(self.kin.position(self.home)-self.origin)>.002:
            raise ValueError('Calibration origin does not match FK of recorded home')
        self.q=None;self.q_time=0;self.goal=None;self.goal_time=0
        self.pub=rospy.Publisher('~joint_target',JointState,queue_size=1)
        self.state_check=rospy.ServiceProxy('/check_state_validity',GetStateValidity)
        rospy.wait_for_service('/check_state_validity',timeout=10)
        self.sub1=rospy.Subscriber('/joint_states',JointState,self.on_state,queue_size=1)
        self.sub2=rospy.Subscriber('~opening_target',PointStamped,self.on_target,queue_size=1)
    def on_state(self,msg):
        try:
            age=(rospy.Time.now()-msg.header.stamp).to_sec()
            if msg.header.stamp.is_zero() or age<0 or age>.2:return
            if len(msg.name)!=len(msg.position) or len(set(msg.name))!=len(msg.name):return
            mapping=dict(zip(msg.name,msg.position));q=np.array([mapping[n] for n in self.names],float)
            if not np.isfinite(q).all():return
            with self.lock:self.q=q;self.q_time=time.monotonic()
        except KeyError:return
    def on_target(self,msg):
        p=np.array([msg.point.x,msg.point.y,msg.point.z]);age=(rospy.Time.now()-msg.header.stamp).to_sec()
        if msg.header.frame_id!=self.base or not np.isfinite(p).all() or msg.header.stamp.is_zero() or age<0 or age>.25 or np.linalg.norm(p-self.origin)>self.radius+1e-9:
            with self.lock:self.goal=None
            rospy.logwarn_throttle(2,'Invalid, stale or out-of-sphere target; rejecting')
            return
        with self.lock:self.goal=p;self.goal_time=time.monotonic()
    def path_valid(self,start,end):
        # Discrete screening, not continuous collision checking or a safety guarantee.
        for f in np.linspace(0,1,21):
            q=start+(end-start)*f
            if np.linalg.norm(self.kin.position(q)-self.origin)>self.radius+.001:return False
            req=GetStateValidityRequest();req.group_name=self.group;req.robot_state=RobotState();req.robot_state.is_diff=True
            req.robot_state.joint_state.name=self.names;req.robot_state.joint_state.position=q.tolist()
            if not self.state_check(req).valid:return False
        return True
    def spin(self):
        rate=rospy.Rate(50)
        while not rospy.is_shutdown():
            with self.lock:
                q=None if self.q is None else self.q.copy();p=None if self.goal is None else self.goal.copy();qt=self.q_time;pt=self.goal_time
            if q is None or p is None or time.monotonic()-qt>.2 or time.monotonic()-pt>.2:
                rate.sleep();continue
            try:
                ik=solve_restricted(p,self.home,q,self.kin.lower+.01,self.kin.upper-.01,self.kin.position)
                if not ik.reachable:
                    rospy.logwarn_throttle(2,'Target unreachable under joint restrictions (error %.3f m)',ik.error_m)
                elif self.path_valid(q,ik.joints):
                    with self.lock:
                        still_current=(self.goal is not None and np.array_equal(self.goal,p) and time.monotonic()-self.goal_time<=.2 and time.monotonic()-self.q_time<=.2)
                    if still_current:
                        msg=JointState();msg.header.stamp=rospy.Time.now();msg.name=self.names;msg.position=ik.joints.tolist();self.pub.publish(msg)
                else:rospy.logwarn_throttle(2,'Joint path failed collision or sphere screening')
            except (ValueError,RuntimeError,rospy.ServiceException) as e:rospy.logerr_throttle(2,'IK/validation failed: %s',e)
            rate.sleep()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--calibration',default=str(DEFAULT_CONFIG));args=parser.parse_args(rospy.myargv()[1:])
    with open(os.path.expanduser(args.calibration)) as f:cfg=yaml.safe_load(f)["calibration"]
    rospy.init_node('catchbox_command_adapter');Controller(cfg).spin()
if __name__=='__main__':main()
