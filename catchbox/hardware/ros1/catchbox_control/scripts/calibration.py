#!/usr/bin/env python3
"""Capture the sphere origin, or request an origin/up/down/left/right test."""
import argparse, os, time
import numpy as np
import rospy, yaml
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
from catchbox.hardware.ros_kinematics import KDLKinematics
from catchbox.hardware.calibration import BASE_LINK, JOINT_NAMES, WORKSPACE_RADIUS_M, DEFAULT_CONFIG, validate_config, target, save_origin

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration',default=str(DEFAULT_CONFIG),help='Hardware configuration YAML file')
    sub=p.add_subparsers(dest='command',required=True)
    capture=sub.add_parser('set-origin',help='Record the current box opening and seven joint positions')
    move=sub.add_parser('move',help='Request a calibration target through the checked command adapter')
    move.add_argument('target',choices=['origin','up','down','left','right'])
    move.add_argument('--distance',type=float,default=.03,help='Offset in meters (default 0.03)')
    move.add_argument('--duration',type=float,default=5);move.add_argument('--execute',action='store_true')
    args=p.parse_args(rospy.myargv()[1:]);rospy.init_node('catchbox_calibration');path=os.path.expanduser(args.calibration)
    with open(path) as f:settings=yaml.safe_load(f)
    cfg=settings['calibration']
    validate_config(cfg)
    if args.command=='set-origin':
        names=JOINT_NAMES
        msg=rospy.wait_for_message('/joint_states',JointState,timeout=5);mapping=dict(zip(msg.name,msg.position))
        age=(rospy.Time.now()-msg.header.stamp).to_sec()
        if msg.header.stamp.is_zero() or age<0 or age>.2 or len(msg.velocity)!=len(msg.name):raise ValueError('A fresh joint state with velocities is required')
        velocities=dict(zip(msg.name,msg.velocity))
        if any(not np.isfinite(velocities[n]) or abs(velocities[n])>.01 for n in names):raise ValueError('Stop the robot before recording origin')
        q=np.array([mapping[n] for n in names],float)
        if not np.isfinite(q).all():raise ValueError('Invalid robot joint state')
        kin=KDLKinematics(BASE_LINK,cfg['tip_link'],names,cfg['opening_offset_m']);origin=kin.position(q)
        cfg['home_joints_rad']=q.tolist();cfg['origin_xyz_m']=origin.tolist()
        save_origin(path,cfg)
        print('Saved origin in base frame:',origin,'to',path);return
    if not 0<args.duration<=60:raise ValueError('Invalid duration')
    goal=target(cfg,args.target,args.distance);print('Base-frame target:',goal)
    if not args.execute:print('Preview only; add --execute to send this target.');return
    pub=rospy.Publisher('/catchbox_command_adapter/opening_target',PointStamped,queue_size=1);rate=rospy.Rate(20);deadline=time.monotonic()+args.duration
    while not rospy.is_shutdown() and time.monotonic()<deadline:
        msg=PointStamped();msg.header.frame_id=BASE_LINK;msg.header.stamp=rospy.Time.now()
        msg.point.x,msg.point.y,msg.point.z=map(float,goal);pub.publish(msg);rate.sleep()
    print('Target publishing ended; controller will stop receiving updates.')
if __name__=='__main__':main()
