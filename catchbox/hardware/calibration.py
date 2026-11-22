"""Record the hardware sphere center with FrankaPy and preview test targets.

This script does not command motion. ROS bring-up and motion tests use the
separate ROS controller. FrankaPy itself still needs its robot infrastructure.
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np

BASE_LINK = "panda_link0"
JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
MOVE_GROUP = "panda_arm"
WORKSPACE_RADIUS_M = 0.2
DEFAULT_CONFIG = Path(__file__).with_name("configuration.yaml")


def vector(value, size, name):
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must have {size} finite numbers') from exc
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f'{name} must have {size} finite numbers')
    return a


def validate_config(config):
    if not isinstance(config.get('tip_link'), str) or not config['tip_link']:
        raise ValueError('Set tip_link to the frame represented by FrankaPy get_pose()')
    offset = vector(config.get('opening_offset_m'), 3, 'opening_offset_m')
    return offset, WORKSPACE_RADIUS_M


def capture_origin(config, pose, joints):
    offset, _ = validate_config(config)
    position = vector(pose.translation, 3, 'robot pose translation')
    rotation = np.asarray(pose.rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-3):
        raise ValueError('Robot pose rotation must be a valid 3x3 rotation')
    home = vector(joints, 7, 'robot joints')
    result = dict(config)
    result['origin_xyz_m'] = (position + rotation @ offset).tolist()
    result['home_joints_rad'] = home.tolist()
    return result


def target(config, direction, distance=None):
    _, radius = validate_config(config)
    center = vector(config.get('origin_xyz_m'), 3, 'origin_xyz_m: record the center first')
    axes = {'origin': [0,0,0], 'up': [0,0,1], 'down': [0,0,-1], 'left': [0,1,0], 'right': [0,-1,0]}
    if direction not in axes:
        raise ValueError('Unknown target')
    distance = radius if distance is None else float(distance)
    if not np.isfinite(distance) or not 0 <= distance <= radius:
        raise ValueError('distance must be between zero and 0.2 m')
    return center + distance * np.asarray(axes[direction])



def save_origin(path, captured):
    """Update the recorded values without removing configuration comments."""
    import yaml
    text = Path(path).read_text()
    match = re.search(r'(?ms)^calibration:[^\n]*\n(?:(?:[ \t]+[^\n]*|[ \t]*)\n)*', text)
    if not match:
        raise ValueError('Could not find calibration section')
    section = match.group()
    for key in ('origin_xyz_m', 'home_joints_rad'):
        value = json.dumps(captured[key], allow_nan=False)
        section, count = re.subn(r'(?m)^(  ' + key + r':)[^\n]*$',
                                 lambda m: m.group(1) + ' ' + value, section)
        if count != 1:
            raise ValueError(f'Expected exactly one {key} field')
    updated = text[:match.start()] + section + text[match.end():]
    parsed = yaml.safe_load(updated)
    if parsed['calibration']['origin_xyz_m'] != captured['origin_xyz_m'] or parsed['calibration']['home_joints_rad'] != captured['home_joints_rad']:
        raise ValueError('Updated calibration did not parse as expected')
    Path(path).write_text(updated)

def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--set-origin', action='store_true', help='Read and record the stationary robot pose; no motion')
    modes.add_argument('--show', action='store_true', help='Show recorded center and all sphere-edge points')
    modes.add_argument('--target', choices=['origin','up','down','left','right'], help='Preview one target; no motion')
    parser.add_argument('--distance', type=float, help='Target offset in meters; defaults to radius')
    args = parser.parse_args()
    with args.config.open() as f:
        settings = yaml.safe_load(f)
    config = settings['calibration']
    validate_config(config)
    if args.set_origin:
        from frankapy import FrankaArm
        arm = FrankaArm()
        captured = capture_origin(config, arm.get_pose(), arm.get_joints())
        save_origin(args.config, captured)
        print('Saved opening center:', captured['origin_xyz_m'], 'm in', BASE_LINK)
        print('Saved home joints to', args.config)
    elif args.show:
        for name in ['origin','up','down','left','right']:
            print(f'{name}: {target(config, name).tolist()} m')
    else:
        print(f'{args.target}: {target(config, args.target, args.distance).tolist()} m')
        print('Preview only; no robot motion commanded.')


if __name__ == '__main__':
    main()
