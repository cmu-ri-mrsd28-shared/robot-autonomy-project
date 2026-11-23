#!/usr/bin/env python3
"""Add the table and the catch box to MoveIt's planning scene, then check home.

Run once after move_group is up:  rosrun catchbox_control planning_scene.py
The table is attached to the base link and the box to the tool link, so MoveIt
ignores their intended contacts but checks them against the rest of the arm.
"""
import argparse, os, sys
import numpy as np
import rospy, yaml, moveit_commander
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest
from catchbox.hardware.calibration import BASE_LINK, JOINT_NAMES, MOVE_GROUP, DEFAULT_CONFIG, validate_config, vector

BOX_TOUCH_LINKS = ['panda_hand', 'panda_leftfinger', 'panda_rightfinger', 'panda_hand_tcp']


def box_in_tip(offset, box_min, box_max, padding):
    """Center, size and orientation of the padded box in the tool frame.

    Box bounds are given in a frame at the opening that is level at home (x
    forward, z up). At home the tool frame points down: 180 degrees about x.
    """
    flip = np.diag([1.0, -1.0, -1.0])
    center = offset + flip @ ((box_min + box_max) / 2)
    return center, box_max - box_min + 2 * padding, (1.0, 0.0, 0.0, 0.0)  # quaternion x, y, z, w


def pose(frame, xyz, quat=(0.0, 0.0, 0.0, 1.0)):
    p = PoseStamped(); p.header.frame_id = frame
    p.pose.position.x, p.pose.position.y, p.pose.position.z = map(float, xyz)
    o = p.pose.orientation; o.x, o.y, o.z, o.w = quat
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', default=str(DEFAULT_CONFIG), help='Hardware configuration YAML file')
    parser.add_argument('--remove', action='store_true', help='Remove the table and box and exit')
    args = parser.parse_args(rospy.myargv()[1:])
    with open(os.path.expanduser(args.calibration)) as f:
        settings = yaml.safe_load(f)
    cfg, sc = settings['calibration'], settings['planning_scene']
    offset, _ = validate_config(cfg)
    tip = cfg['tip_link']

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('catchbox_planning_scene')
    scene = moveit_commander.PlanningSceneInterface(synchronous=True)
    rospy.sleep(1.0)
    for link, name in ((BASE_LINK, 'table'), (tip, 'catch_box')):
        scene.remove_attached_object(link, name)
        scene.remove_world_object(name)
    if args.remove:
        print('Removed table and catch_box'); return

    table = vector(sc['table_size_m'], 3, 'table_size_m')
    top = float(sc['table_top_z_m'])
    scene.attach_box(BASE_LINK, 'table', pose(BASE_LINK, [0, 0, top - table[2] / 2]), size=tuple(table),
                     touch_links=[BASE_LINK])
    center, size, quat = box_in_tip(offset, vector(sc['box_min_m'], 3, 'box_min_m'),
                                    vector(sc['box_max_m'], 3, 'box_max_m'), float(sc['padding_m']))
    scene.attach_box(tip, 'catch_box', pose(tip, center, quat), size=tuple(size), touch_links=BOX_TOUCH_LINKS)
    attached = scene.get_attached_objects(['table', 'catch_box'])
    if set(attached) != {'table', 'catch_box'}:
        raise RuntimeError(f'Planning scene did not update; attached: {sorted(attached)}')
    print(f'Table: {table.tolist()} m, top at z={top} m in {BASE_LINK}')
    print(f'Catch box: {np.round(size, 3).tolist()} m, center {np.round(center, 4).tolist()} m in {tip}')

    rospy.wait_for_service('/check_state_validity', timeout=10)
    req = GetStateValidityRequest(); req.group_name = MOVE_GROUP; req.robot_state = RobotState()
    req.robot_state.is_diff = True; req.robot_state.joint_state.name = JOINT_NAMES
    req.robot_state.joint_state.position = list(map(float, cfg['home_joints_rad']))
    result = rospy.ServiceProxy('/check_state_validity', GetStateValidity)(req)
    if result.valid:
        print('Home pose is collision-free with the table and box.')
    else:
        pairs = sorted({f'{c.contact_body_1} <-> {c.contact_body_2}' for c in result.contacts})
        print('Home pose is IN COLLISION:', ', '.join(pairs) or 'no contact details')
        sys.exit(1)


if __name__ == '__main__':
    main()
