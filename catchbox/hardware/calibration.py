"""Record the sphere center and run FrankaPy hardware checks of increasing reach.

Run from the project folder in the lab's FrankaPy environment:

    python -m catchbox.hardware.calibration --set-origin       # record center; no motion
    python -m catchbox.hardware.calibration --show             # print test targets; no motion
    python -m catchbox.hardware.calibration --check1-home      # go to recorded home joints
    python -m catchbox.hardware.calibration --check2-small     # 3 cm out and back, six directions
    python -m catchbox.hardware.calibration --check3-steps     # 5/10/15/20 cm out and back
    python -m catchbox.hardware.calibration --check4-catch     # chain random points, no return between
    python -m catchbox.hardware.calibration --move left --distance 0.05

All moves keep the box orientation recorded at home and command the box
opening, not the tool frame. Every target is checked against the 0.2 m sphere
before anything is sent. These checks bypass the ROS adapter and its MoveIt
collision screening: they command the physical robot directly.
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
DEFAULT_SPEED = 0.05          # m/s of the box opening
MIN_DURATION = 2.0            # s, for any single move
DEFAULT_MIN_Z = 0.10          # m, lowest opening height if min_opening_z_m is not set
# Base-frame directions. left/right are +Y/-Y, forward/back are +X/-X.
AXES = {'origin': [0, 0, 0], 'up': [0, 0, 1], 'down': [0, 0, -1], 'left': [0, 1, 0],
        'right': [0, -1, 0], 'forward': [1, 0, 0], 'back': [-1, 0, 0]}
DIRECTIONS = ['left', 'right', 'up', 'down', 'forward', 'back']


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
    if direction not in AXES:
        raise ValueError('Unknown target')
    distance = radius if distance is None else float(distance)
    if not np.isfinite(distance) or not 0 <= distance <= radius:
        raise ValueError('distance must be between zero and 0.2 m')
    return center + distance * np.asarray(AXES[direction])


def random_targets(config, count, radius, seed=0):
    """Points uniformly inside a sphere of the given radius around the center."""
    _, limit = validate_config(config)
    center = vector(config.get('origin_xyz_m'), 3, 'origin_xyz_m: record the center first')
    if not 0 < radius <= limit:
        raise ValueError('radius must be between zero and 0.2 m')
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(count, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points *= radius * rng.uniform(size=(count, 1)) ** (1 / 3)
    return center + points


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


class BoxArm:
    """Moves the box opening with FrankaPy, keeping the home orientation."""

    def __init__(self, config, arm, speed=DEFAULT_SPEED):
        self.offset, self.radius = validate_config(config)
        self.config = config
        self.center = vector(config.get('origin_xyz_m'), 3, 'origin_xyz_m: record the center first')
        self.home = vector(config.get('home_joints_rad'), 7, 'home_joints_rad: record the center first')
        self.speed = float(speed)
        if not np.isfinite(self.speed) or not 0 < self.speed <= 0.2:
            raise ValueError('speed must be between zero and 0.2 m/s')
        self.min_z = float(config.get('min_opening_z_m', DEFAULT_MIN_Z))
        self.arm = arm
        self.rotation = None
        self.frames = None

    def allowed(self, goal):
        return np.linalg.norm(goal - self.center) <= self.radius + 1e-9 and goal[2] >= self.min_z

    def opening(self):
        pose = self.arm.get_pose()
        return np.asarray(pose.translation, float) + np.asarray(pose.rotation, float) @ self.offset

    def go_home(self):
        """Joint move to the recorded home; this also fixes the box orientation."""
        start = self.opening()
        duration = max(5.0, 2 * np.linalg.norm(start - self.center) / self.speed)
        print(f'Home: joint move to recorded home joints, {duration:.1f} s', flush=True)
        self.arm.goto_joints(self.home.tolist(), duration=duration, ignore_errors=False)
        pose = self.arm.get_pose()
        self.rotation = np.asarray(pose.rotation, float).copy()
        self.frames = (pose.from_frame, pose.to_frame)
        self.report(self.center, 'home')

    def move_to(self, goal, label):
        if self.rotation is None:
            raise RuntimeError('Call go_home() first so the box orientation is known')
        goal = vector(goal, 3, 'target')
        if np.linalg.norm(goal - self.center) > self.radius + 1e-9:
            raise ValueError(f'{label}: target {goal.tolist()} is outside the 0.2 m sphere')
        if goal[2] < self.min_z:
            raise ValueError(f'{label}: target z {goal[2]:.3f} m is below min_opening_z_m {self.min_z:.3f} m')
        distance = float(np.linalg.norm(goal - self.opening()))
        duration = max(MIN_DURATION, 1.875 * distance / self.speed)
        from autolab_core import RigidTransform
        pose = RigidTransform(rotation=self.rotation, translation=goal - self.rotation @ self.offset,
                              from_frame=self.frames[0], to_frame=self.frames[1])
        print(f'{label}: opening -> {np.round(goal, 4).tolist()} ({distance * 100:.1f} cm, {duration:.1f} s)', flush=True)
        self.arm.goto_pose(pose, duration=duration, use_impedance=False, ignore_errors=False)
        self.report(goal, label)

    def report(self, goal, label):
        actual = self.opening()
        print(f'{label}: reached {np.round(actual, 4).tolist()}, error {np.linalg.norm(actual - goal) * 1000:.1f} mm, '
              f'{np.linalg.norm(actual - self.center) * 100:.1f} cm from center', flush=True)


def check1_home(box, args):
    """Joint move to the recorded home; the opening should sit at the center."""
    box.go_home()


def check2_small(box, args):
    """3 cm out and back in each of six directions."""
    box.go_home()
    for name in DIRECTIONS:
        box.move_to(target(box.config, name, 0.03), f'{name} 3 cm')
        box.move_to(box.center, 'center')


def check3_steps(box, args):
    """Each direction at 5, 10, 15 and 20 cm, returning to center after each."""
    distances = [d for d in (0.05, 0.10, 0.15, 0.20) if d <= args.max_distance + 1e-9]
    box.go_home()
    for name in args.directions or DIRECTIONS:
        for d in distances:
            goal = target(box.config, name, d)
            if not box.allowed(goal):
                print(f'{name} {d * 100:.0f} cm: skipped, below min_opening_z_m {box.min_z} m', flush=True)
                continue
            box.move_to(goal, f'{name} {d * 100:.0f} cm')
            box.move_to(box.center, 'center')


def check4_catch(box, args):
    """Chain random points inside the sphere without returning, like successive catches."""
    candidates = random_targets(box.config, 50 * args.count, args.max_distance, args.seed)
    points = [p for p in candidates if box.allowed(p)][:args.count]
    if len(points) < args.count:
        raise ValueError('Too few random points above min_opening_z_m; lower --max-distance')
    box.go_home()
    for i, p in enumerate(points, 1):
        box.move_to(p, f'point {i}/{args.count}')
    box.move_to(box.center, 'center')


CHECKS = {'check1_home': check1_home, 'check2_small': check2_small,
          'check3_steps': check3_steps, 'check4_catch': check4_catch}


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--set-origin', action='store_true', help='Record the stationary opening as the sphere center; no motion')
    modes.add_argument('--show', action='store_true', help='Print the center and edge targets; no motion')
    modes.add_argument('--move', choices=list(AXES), help='Go home, then move the opening one step in this direction')
    modes.add_argument('--check1-home', dest='check', action='store_const', const='check1_home', help='Joint move to the recorded home')
    modes.add_argument('--check2-small', dest='check', action='store_const', const='check2_small', help='3 cm out and back in six directions')
    modes.add_argument('--check3-steps', dest='check', action='store_const', const='check3_steps', help='5/10/15/20 cm out and back per direction')
    modes.add_argument('--check4-catch', dest='check', action='store_const', const='check4_catch', help='Chain random points inside the sphere')
    parser.add_argument('--distance', type=float, default=0.03, help='With --move: offset in meters (default 0.03)')
    parser.add_argument('--max-distance', type=float, default=0.20, help='check3: largest step; check4: sphere radius used (default 0.20)')
    parser.add_argument('--directions', nargs='+', choices=DIRECTIONS, help='check3: only these directions')
    parser.add_argument('--count', type=int, default=10, help='check4: number of points (default 10)')
    parser.add_argument('--seed', type=int, default=0, help='check4: random seed (default 0)')
    parser.add_argument('--speed', type=float, default=DEFAULT_SPEED, help=f'Opening speed in m/s (default {DEFAULT_SPEED})')
    parser.add_argument('--yes', action='store_true', help='Skip the Enter prompt before motion')
    args = parser.parse_args()
    if not 0 < args.max_distance <= WORKSPACE_RADIUS_M:
        parser.error('--max-distance must be between zero and 0.2 m')
    if args.count < 1:
        parser.error('--count must be at least 1')
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
        return
    if args.show:
        print(f'origin: {target(config, "origin").tolist()} m')
        for name in DIRECTIONS:
            print(f'{name}: {target(config, name).tolist()} m')
        return
    goal = target(config, args.move, args.distance) if args.move else None
    name = f'--move {args.move} {args.distance} m' if args.move else args.check
    print(f'{name}: speed {args.speed} m/s, center {config["origin_xyz_m"]}')
    print('This moves the physical robot directly through FrankaPy. Keep a hand on the e-stop.')
    if not args.yes and input('Press Enter to start, or q to quit: ').strip().lower() == 'q':
        return
    from frankapy import FrankaArm
    box = BoxArm(config, FrankaArm(), args.speed)
    if args.move:
        box.go_home()
        box.move_to(goal, f'{args.move} {args.distance * 100:.0f} cm')
    else:
        CHECKS[args.check](box, args)


if __name__ == '__main__':
    main()
