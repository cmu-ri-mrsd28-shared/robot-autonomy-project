"""ROS check of the camera node's messages against the real model_node, no camera.

Runs inside the catchbox-ros-test image with the project folder mounted:

    docker run --rm --network none -v "$PWD":/app:ro catchbox-ros-test \\
        python3 docker/vision_node_ros_test.py

It replays a scripted session through ball_vision_node's own output and publishing
code (BallOutput, CameraTopics) at the camera's 30 Hz: no ball, then throws, then
the ball in the box. The real model_node and mock_robot (for the box state) run
alongside, and the model's log must show it accepted each phase and built the
expected inputs from it.
"""
import json, math, os, subprocess, sys, tempfile, time, traceback
from pathlib import Path
from xmlrpc.client import ServerProxy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import numpy as np  # noqa: E402
import ball_vision_node as node  # noqa: E402
import measure_ball as mb  # noqa: E402
from calibrate_camera import CameraToBase  # noqa: E402
from track_ball import Detection  # noqa: E402

CONFIG = 'docker/test_configuration.yaml'
CENTER = np.array([.48659539638260424, .010536461494865357, .2970724773569217])
FPS = 30.0
# The lab calibration (calibration/camera_to_base.json).
CALIBRATION = CameraToBase(
    np.array([[-0.6222648249976477, -0.1911009131525057, -0.7591224727031376],
              [0.7786182108748099, -0.05091242556645543, -0.6254291379661248],
              [0.08087131299018867, -0.9802491346158989, 0.18047566268066204]]),
    np.array([2.082019801503156, 0.6250852440931923, 0.2623475226945383]))


def measured(base_position):
    """The tracker's measurement of a base-frame point (camera frame)."""
    camera = CALIBRATION.rotation.T @ (np.asarray(base_position) - CALIBRATION.translation)
    return mb.Measurement(Detection(320, 240, 8.0), camera, 'depth')


def throw_position(t):
    """A ball thrown toward the box opening, t seconds after release."""
    return CENTER + [.30 - .9 * t, .05 - .1 * t, .45 + .4 * t - 4.905 * t * t]


def records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    processes = []
    with tempfile.TemporaryDirectory(prefix='catchbox-vision-test-') as tmp:
        out = Path(tmp) / 'nodes.log'
        f = out.open('w+')

        def start(name, cmd):
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=os.environ.copy())
            processes.append((name, p))

        try:
            start('roscore', ['roscore'])
            deadline = time.monotonic() + 30
            while True:
                try:
                    if ServerProxy(os.environ['ROS_MASTER_URI']).getPid('/t')[0] == 1:
                        break
                except (OSError, ConnectionError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError('ROS master did not start')
                time.sleep(.05)
            import rospy
            rospy.init_node('catchbox_vision_test', disable_signals=True)
            topics = node.CameraTopics(rospy)
            output = node.BallOutput(np, CALIBRATION)
            model_log = Path(tmp) / 'model.jsonl'
            start('mock robot', [sys.executable, '-m', 'catchbox.hardware.mock_robot', '--config', CONFIG])
            start('model', [sys.executable, '-m', 'catchbox.hardware.model_node', '--config', CONFIG,
                            '--log', str(model_log)])
            time.sleep(3.0)   # let model_node subscribe and receive the box state
            for name, p in processes:
                if p.poll() is not None:
                    raise RuntimeError('%s exited early' % name)

            lock = mb.BallLock()
            phases = []

            def run(label, seconds, frame):
                start_t = time.time()
                phases.append((label, start_t))
                k = 0
                while time.time() - start_t < seconds:
                    now = time.time()
                    found = frame(k, now - start_t)
                    topics.publish(output.update(lock, found, now, now), now)
                    k += 1
                    time.sleep(max(0.0, start_t + k / FPS - time.time()))
                phases.append((label + ' end', time.time()))

            # 1. Searching: no active ball -> the model must withhold targets.
            lock.state = 'SEARCHING'
            run('no ball', 1.5, lambda k, t: None)

            # 2. Throws: the ball is locked and visible every frame.
            def throwing(k, t):
                t = t % .4
                if t < 1 / FPS:   # each throw is a new ball (as when re-locked)
                    lock.ball_id = (lock.ball_id or 0) + 1
                lock.state = 'LOCKED'
                return measured(throw_position(t))
            run('throws', 3.0, throwing)

            # 3. It went into the box: hidden, inside.
            lock.state, lock.in_box_balls = 'SEARCHING', [{'id': lock.ball_id}]
            run('in box', 1.5, lambda k, t: None)
            time.sleep(.5)

            logs = records(model_log)
            if not logs:
                raise RuntimeError('model_node wrote no log')
            key = 'wall_time_s'   # catchbox.hardware.event_log stamps every record

            def phase(label):
                begin, end = dict(phases)[label], dict(phases)[label + ' end']
                return [r for r in logs if begin + .3 <= r[key] <= end]

            waits = [r for r in phase('no ball') if r.get('event') == 'input_wait']
            infer = [r for r in phase('no ball') if r.get('event') == 'inference']
            if infer or not any(r.get('reason') == 'no_active_ball_or_invalid_status'
                                for r in logs if r.get('event') == 'input_wait'):
                raise RuntimeError('No-ball status did not withhold targets: %s' % (waits[:3],))
            print('PASS: status -1 (no active ball) withholds model targets', flush=True)

            thrown = [r for r in phase('throws') if r.get('event') == 'inference']
            bad = [r for r in phase('throws') if r.get('event') == 'input_wait']
            tracked = [r for r in thrown if r['tracking_status'] == 2]
            if len(thrown) < 50 or len(tracked) < 0.6 * len(thrown):
                raise RuntimeError('Too few accepted ball frames: %d inferences, %d tracked, waits %s'
                                   % (len(thrown), len(tracked), [r.get('reason') for r in bad][:5]))
            for r in tracked:
                obs = r['observation']
                if len(obs) != 25 or not all(math.isfinite(v) for v in obs + r['action']):
                    raise RuntimeError('Invalid model input or output')
                if obs[22] != 1.0 or obs[24] != 0.0 or not any(obs[9:15]):
                    raise RuntimeError('Tracked ball not in the model inputs: %s' % obs)
                if not 0 <= r['ball_age_ms'] <= 200:
                    raise RuntimeError('Ball too old at inference: %.0f ms' % r['ball_age_ms'])
            ages = sorted(r['ball_age_ms'] for r in tracked)
            print('PASS: thrown ball accepted: %d inferences, %d with velocity (status 2), '
                  'flags in inputs, median ball age %.0f ms%s' % (
                      len(thrown), len(tracked), ages[len(ages) // 2],
                      '' if not bad else ', waits: %s' % sorted({r.get('reason') for r in bad})),
                  flush=True)
            if not any(r['observation'][23] == 1.0 for r in tracked):
                raise RuntimeError('No interception prediction for the thrown ball')
            print('PASS: thrown ball gets an interception prediction', flush=True)

            boxed = [r for r in phase('in box') if r.get('event') == 'inference']
            if not boxed or any(r['tracking_status'] != 0 or r['in_box'] != 1 or
                                r['observation'][9:22] != [0.] * 13 or
                                r['observation'][22:25] != [0., 0., 1.] for r in boxed):
                raise RuntimeError('In-box phase not seen as hidden + inside: %s' % boxed[:1])
            print('PASS: ball in box -> status 0, in-box 1, ball inputs zeroed, inference continues',
                  flush=True)
            print('All camera node ROS checks passed.', flush=True)
        except BaseException:
            traceback.print_exc()
            f.flush()
            print('\n--- Node output ---', flush=True)
            print(out.read_text()[-8000:], flush=True)
            raise
        finally:
            for _, p in reversed(processes):
                if p.poll() is None:
                    p.terminate()
            for _, p in reversed(processes):
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            f.close()


if __name__ == '__main__':
    main()
