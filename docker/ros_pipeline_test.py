"""Exercise the ROS pipeline against an isolated simulated robot, no hardware."""
import json, math, os, subprocess, sys, tempfile, threading, time, traceback
from pathlib import Path
from xmlrpc.client import ServerProxy

CONFIG='docker/test_configuration.yaml'
CENTER=[.48659539638260424,.010536461494865357,.2970724773569217]

class Observe:
    def __init__(self):
        self.lock=threading.Lock();self.box=None;self.joints=None;self.targets=[];self.joint_count=0
    def on_box(self,msg):
        p=msg.pose.pose.position
        with self.lock:self.box=(p.x,p.y,p.z)
    def on_joints(self,msg):
        with self.lock:self.joints=list(msg.position);self.joint_count+=1
    def on_target(self,msg):
        p=msg.point
        with self.lock:self.targets.append((p.x,p.y,p.z))
    def sample(self):
        with self.lock:return self.box,None if self.joints is None else self.joints[:],len(self.targets),self.joint_count

def wait_for(predicate,seconds,label,processes):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        for name,proc in processes:
            if proc.poll() is not None:raise RuntimeError('%s exited early with code %s'%(name,proc.returncode))
        if predicate():return
        time.sleep(.05)
    raise RuntimeError('Timed out waiting for '+label)

def main():
    processes=[]
    with tempfile.TemporaryDirectory(prefix='catchbox-ros-test-') as tmp:
        out=Path(tmp)/'nodes.log';f=out.open('w+')
        def start(name,cmd):
            p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
            processes.append((name,p));return p
        try:
            start('roscore',['roscore'])
            def master_ready():
                try:return ServerProxy(os.environ['ROS_MASTER_URI']).getPid('/pipeline_test')[0]==1
                except (OSError,ConnectionError):return False
            wait_for(master_ready,30,'ROS master',processes)
            import rospy,yaml
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import JointState
            from geometry_msgs.msg import PointStamped
            rospy.init_node('catchbox_pipeline_test',disable_signals=True)
            observed=Observe()
            subs=[rospy.Subscriber('/catchbox/box_state',Odometry,observed.on_box,queue_size=1),
                  rospy.Subscriber('/catchbox_command_adapter/joint_target',JointState,observed.on_joints,queue_size=1),
                  rospy.Subscriber('/catchbox_command_adapter/opening_target',PointStamped,observed.on_target,queue_size=100)]
            pub=rospy.Publisher('/catchbox_command_adapter/opening_target',PointStamped,queue_size=1)
            start('mock robot',[sys.executable,'-m','catchbox.hardware.mock_robot','--config',CONFIG])
            start('adapter',[sys.executable,'-m','catchbox.hardware.adapter','--mock-robot','--config',CONFIG,'--log',tmp+'/adapter.jsonl'])
            wait_for(lambda:observed.sample()[0] is not None and observed.sample()[1] is not None and pub.get_num_connections()>0,30,'mock robot and adapter',processes)
            print('PASS: ROS master, fake robot, and motion-disabled adapter started',flush=True)
            target=[CENTER[0],CENTER[1]+.03,CENTER[2]]
            deadline=time.monotonic()+4
            while time.monotonic()<deadline:
                m=PointStamped();m.header.stamp=rospy.Time.now();m.header.frame_id='panda_link0'
                m.point.x,m.point.y,m.point.z=target;pub.publish(m);time.sleep(.05)
            box,_,_,_=observed.sample()
            if box is None or box[1]-CENTER[1]<.012:raise RuntimeError('Fake robot did not follow the smoothed left target: %s'%(box,))
            print('PASS: target passed through IK and smoothing to the fake robot',flush=True)
            time.sleep(1.)
            box=observed.sample()[0]
            if math.dist(box,CENTER)<.011:raise RuntimeError('Returned to center before the 3 second delay')
            wait_for(lambda: observed.sample()[0] is not None and math.dist(observed.sample()[0],CENTER)<.012,12,'return to center',processes)
            print('PASS: held before returning toward center after 3 seconds',flush=True)
            start('model',[sys.executable,'-m','catchbox.hardware.model_node','--config',CONFIG,'--log',tmp+'/model.jsonl','--throw-gate'])
            start('mock ball',[sys.executable,'-m','catchbox.hardware.mock_inputs','--config',CONFIG,'--seconds','25','--exercise-occlusion'])
            initial=observed.sample(); initial_targets=initial[2];initial_joints=initial[3];initial_q=initial[1]
            wait_for(lambda:observed.sample()[2]>initial_targets+10 and observed.sample()[3]>initial_joints+10,25,'model and adapter output',processes)
            def model_moved_fake_robot():
                q=observed.sample()[1]
                return q is not None and initial_q is not None and max(abs(a-b) for a,b in zip(q,initial_q))>.001
            wait_for(model_moved_fake_robot,20,'model output moving fake joints through adapter',processes)
            model_log=Path(tmp)/'model.jsonl'
            wait_for(lambda:model_log.exists() and '"inference"' in model_log.read_text(),10,'model inference log',processes)
            sample=next(json.loads(line) for line in model_log.read_text().splitlines() if json.loads(line).get('event')=='inference')
            if len(sample['observation'])!=25 or len(sample['action'])!=3 or not all(math.isfinite(v) for v in sample['action']):raise RuntimeError('Invalid policy input or output')
            wait_for(lambda:any(json.loads(line).get('event')=='input_wait' and json.loads(line).get('reason') in ('hidden_ball_location_unknown','waiting_for_active_throw') for line in model_log.read_text().splitlines()),10,'hidden unknown input hold',processes)
            if any(json.loads(line).get('event')=='inference' and json.loads(line).get('tracking_status')==0 and json.loads(line).get('in_box')==-1 for line in model_log.read_text().splitlines()):
                raise RuntimeError('Unknown hidden ball must withhold model targets')
            print('PASS: hidden ball with unknown location withholds model targets',flush=True)
            f.flush()
            if 'steady-motion-model/actor.npz' not in out.read_text():raise RuntimeError('Steady-motion actor was not loaded')
            print('PASS: steady-motion actor loaded; fake ball -> 25 model inputs -> policy targets -> adapter joint stream',flush=True)
            print('PASS: timing and action logs recorded; no robot connection was opened',flush=True)
            print('All ROS pipeline checks passed.',flush=True)
        except BaseException:
            traceback.print_exc();f.flush()
            print('\n--- Node output ---',flush=True);print(out.read_text()[-16000:],flush=True)
            raise
        finally:
            for _,p in reversed(processes):
                if p.poll() is None:p.terminate()
            for _,p in reversed(processes):
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            f.close()
if __name__=='__main__':main()
