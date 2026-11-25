"""ROS 1 CPU policy node. It only publishes targets; it never commands joints.

Inputs: /catchbox/ball_state and /catchbox/box_state (nav_msgs/Odometry,
base-frame position and twist); /catchbox/ball_in_box (std_msgs/Int8,
-1 unknown, 0 outside, 1 inside). All must be refreshed continuously.
Output: /catchbox_command_adapter/opening_target (geometry_msgs/PointStamped).
"""
import argparse, threading, time
import numpy as np
from catchbox.hardware.calibration import BASE_LINK, DEFAULT_CONFIG
from catchbox.hardware.event_log import EventLog
from catchbox.hardware.observation import build_observation
from catchbox.hardware.policy import Actor
from catchbox.hardware.policy_frame import PolicyFrame

MAX_AGE=.2
MAX_SKEW=.05

def odom_values(msg):
    p=msg.pose.pose.position;v=msg.twist.twist.linear;w=msg.twist.twist.angular
    return np.array([p.x,p.y,p.z],float),np.array([v.x,v.y,v.z],float),np.array([w.x,w.y,w.z],float)

class ModelNode:
    def __init__(self, settings, actor, rospy, log):
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import PointStamped
        from std_msgs.msg import Int8
        self.rospy,self.PointStamped,self.log=rospy,PointStamped,log
        self.frame=PolicyFrame(settings);self.actor=actor;self.lock=threading.Lock()
        self.ball=None;self.box=None;self.flag=None;self.flag_time=0.
        self.pub=rospy.Publisher('/catchbox_command_adapter/opening_target',PointStamped,queue_size=1)
        self.subscribers=[rospy.Subscriber('/catchbox/ball_state',Odometry,self.on_ball,queue_size=1),
                          rospy.Subscriber('/catchbox/box_state',Odometry,self.on_box,queue_size=1),
                          rospy.Subscriber('/catchbox/ball_in_box',Int8,self.on_flag,queue_size=1)]
    def _sample(self,msg):
        age=(self.rospy.Time.now()-msg.header.stamp).to_sec()
        p,v,w=odom_values(msg)
        if msg.header.frame_id!=BASE_LINK or msg.header.stamp.is_zero() or not 0<=age<=MAX_AGE or not np.isfinite(np.r_[p,v,w]).all():return None
        return (p,v,w,msg.header.stamp.to_sec(),time.monotonic())
    def on_ball(self,msg):
        sample=self._sample(msg)
        with self.lock:self.ball=sample
    def on_box(self,msg):
        sample=self._sample(msg)
        with self.lock:self.box=sample
    def on_flag(self,msg):
        with self.lock:self.flag=msg.data if msg.data in (0,1) else None;self.flag_time=time.monotonic()
    def step(self):
        with self.lock:ball,box,flag,flag_time=self.ball,self.box,self.flag,self.flag_time
        mono=time.monotonic();now=self.rospy.Time.now().to_sec()
        if (ball is None or box is None or flag is None or
            not 0<=mono-ball[4]<=MAX_AGE or not 0<=mono-box[4]<=MAX_AGE or not 0<=mono-flag_time<=MAX_AGE or
            not 0<=now-ball[3]<=MAX_AGE or not 0<=now-box[3]<=MAX_AGE or abs(ball[3]-box[3])>MAX_SKEW):
            return False
        started=time.monotonic()
        obs=build_observation(self.frame,box[0],box[1],box[2],ball[0],ball[1],bool(flag))
        action=self.actor.predict(obs);target=self.frame.target(action)
        if not np.isfinite(target).all():raise ValueError('Model returned non-finite target')
        # Recheck age after inference, and discard if any input was replaced.
        with self.lock:
            unchanged=self.ball is ball and self.box is box and self.flag==flag and self.flag_time==flag_time
        mono=time.monotonic();now=self.rospy.Time.now().to_sec()
        if not unchanged or not 0<=now-ball[3]<=MAX_AGE or not 0<=now-box[3]<=MAX_AGE or mono-flag_time>MAX_AGE:return False
        out=self.PointStamped();out.header.frame_id=BASE_LINK;out.header.stamp=self.rospy.Time.now()
        out.point.x,out.point.y,out.point.z=map(float,target);self.pub.publish(out)
        self.log.write('inference',ball_stamp_s=ball[3],box_stamp_s=box[3],output_stamp_s=out.header.stamp.to_sec(),
                       ball_age_ms=1000*(out.header.stamp.to_sec()-ball[3]),box_age_ms=1000*(out.header.stamp.to_sec()-box[3]),
                       inference_and_encode_ms=1000*(time.monotonic()-started),observation=obs.tolist(),action=action.tolist(),target=target.tolist())
        return True

def main():
    import rospy,yaml
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default=str(DEFAULT_CONFIG))
    parser.add_argument('--rate',type=float,default=50.);parser.add_argument('--log',default='logs/model.jsonl')
    args=parser.parse_args(rospy.myargv()[1:])
    if not 0<args.rate<=100:raise ValueError('rate must be between 0 and 100 Hz')
    with open(args.config) as f:settings=yaml.safe_load(f)
    rospy.init_node('catchbox_model');log=EventLog(args.log);node=ModelNode(settings,Actor(),rospy,log);rate=rospy.Rate(args.rate)
    try:
        while not rospy.is_shutdown():
            try:
                if not node.step():rospy.logwarn_throttle(5,'Waiting for fresh synchronized ball, box and in-box state')
            except Exception as exc:
                rospy.logerr_throttle(2,'Policy output withheld: %s',exc);log.write('model_error',error=str(exc))
            rate.sleep()
    finally:log.close()
if __name__=='__main__':main()
