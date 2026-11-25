"""Simulated robot feedback for ROS pipeline tests; never connects to hardware.

Follows the adapter's joint_target messages instantly. This is not a dynamics
simulation and cannot validate physical motion, collision or safety behavior.
"""
import argparse, threading, time
import numpy as np
from catchbox.hardware.calibration import BASE_LINK, DEFAULT_CONFIG, JOINT_NAMES
from catchbox.hardware.panda_kinematics import PandaKinematics
from catchbox.hardware.rotation import quaternion_from_matrix

class MockRobot:
    def __init__(self, settings, rospy):
        from sensor_msgs.msg import JointState
        from nav_msgs.msg import Odometry
        self.rospy,self.JointState,self.Odometry=rospy,JointState,Odometry
        self.kin=PandaKinematics(settings['calibration']['opening_offset_m'])
        self.q=np.asarray(settings['calibration']['home_joints_rad'],float)
        if self.q.shape!=(7,) or not np.isfinite(self.q).all():raise ValueError('Invalid home joints')
        self.lock=threading.Lock();self.previous=self.q.copy();self.last=time.monotonic()
        self.joints=rospy.Publisher('/catchbox/robot_joints',JointState,queue_size=1)
        self.box=rospy.Publisher('/catchbox/box_state',Odometry,queue_size=1)
        self.sub=rospy.Subscriber('/catchbox_command_adapter/joint_target',JointState,self.on_target,queue_size=1)
    def on_target(self,msg):
        if list(msg.name)!=JOINT_NAMES or len(msg.position)!=7:return
        q=np.asarray(msg.position,float)
        if not np.isfinite(q).all():return
        with self.lock:self.q=q.copy()
    def step(self):
        with self.lock:q=self.q.copy()
        now=time.monotonic();dt=now-self.last
        dq=(q-self.previous)/dt if dt>1e-6 else np.zeros(7)
        self.last=now;self.previous=q.copy()
        p,r,v,w=self.kin.state(q,dq);quat=quaternion_from_matrix(r);stamp=self.rospy.Time.now()
        js=self.JointState();js.header.stamp=stamp;js.name=JOINT_NAMES;js.position=q.tolist();js.velocity=dq.tolist()
        o=self.Odometry();o.header.stamp=stamp;o.header.frame_id=BASE_LINK;o.child_frame_id=BASE_LINK
        o.pose.pose.position.x,o.pose.pose.position.y,o.pose.pose.position.z=map(float,p)
        o.pose.pose.orientation.x,o.pose.pose.orientation.y,o.pose.pose.orientation.z,o.pose.pose.orientation.w=map(float,quat)
        o.twist.twist.linear.x,o.twist.twist.linear.y,o.twist.twist.linear.z=map(float,v)
        o.twist.twist.angular.x,o.twist.twist.angular.y,o.twist.twist.angular.z=map(float,w)
        self.joints.publish(js);self.box.publish(o)

def main():
    import rospy,yaml
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default=str(DEFAULT_CONFIG));p.add_argument('--rate',type=float,default=50.)
    a=p.parse_args(rospy.myargv()[1:])
    if not 0<a.rate<=100:raise ValueError('rate must be between 0 and 100 Hz')
    with open(a.config) as f:settings=yaml.safe_load(f)
    rospy.init_node('catchbox_mock_robot');robot=MockRobot(settings,rospy);rate=rospy.Rate(a.rate)
    while not rospy.is_shutdown():robot.step();rate.sleep()
if __name__=='__main__':main()
