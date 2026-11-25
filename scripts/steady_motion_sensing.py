"""Camera visibility and mixed throws for a future steady-motion experiment.

Camera pose and viewing angle are simulation assumptions, not calibrated hardware.
The estimator is deliberately separate from MuJoCo's ground truth: it only sees
measurements when the ball is visible. This module does not start training.
"""
from collections import deque
import numpy as np
import mujoco
from catchbox.core import Throw
from catchbox.learning.trajectory import varied_throw


class CameraEstimator:
    """25-input convention: hidden XYZ/velocity are zero, flag -1/0/1.

    The last flag means unknown/outside/probably inside. It is an estimate, not
    proof: losing sight near the rim on a descending path can be misleading.
    Tracking and prediction must both be invalid while hidden.
    """
    def __init__(self, seed=0, noise_m=.002, entry_window_s=.25):
        self.rng = np.random.default_rng(seed)
        self.noise_m = noise_m
        self.entry_window_s = entry_window_s
        self.reset()

    def reset(self):
        self.history = deque(maxlen=5)
        self.last_visible_time = None
        self.entry_candidate = False
        self.belief = -1

    def update(self, time, measured_position, box_position, box_rotation, width, depth, radius, height=.095):
        """Pass None for measured_position if occluded or outside camera view."""
        t = float(time)
        if measured_position is None:
            if self.last_visible_time is not None and t-self.last_visible_time > self.entry_window_s:
                if self.entry_candidate:
                    self.belief = 1
                self.entry_candidate = False
            self.history.clear()
            return dict(position=np.zeros(3), velocity=np.zeros(3), tracking_valid=False,
                        in_box=self.belief if self.belief == 1 else -1)
        p = np.asarray(measured_position, dtype=float) + self.rng.normal(0, self.noise_m, 3)
        if self.last_visible_time is not None and t <= self.last_visible_time:
            raise ValueError('Camera timestamps must increase')
        self.history.append((t, p.copy()))
        self.last_visible_time = t
        velocity = np.zeros(3)
        valid = len(self.history) >= 3
        if valid:
            times = np.array([x[0]-t for x in self.history]); xyz = np.array([x[1] for x in self.history])
            adjusted = xyz.copy(); adjusted[:, 2] += .5*9.81*times**2
            centered = times-times.mean()
            velocity = (centered[:, None]*(adjusted-adjusted.mean(axis=0))).sum(axis=0)/(centered@centered)
        local = np.asarray(box_rotation).reshape(3, 3).T @ (p-np.asarray(box_position))
        local_velocity = np.asarray(box_rotation).reshape(3, 3).T @ velocity
        # Use only measured ball history. A plausible entry is a descending ball
        # near the opening; it is not a claim of an actual catch.
        self.entry_candidate = bool(valid and local_velocity[2] < 0 and
            abs(local[0]) < width/2 and abs(local[1]) < depth/2 and
            -radius <= local[2] <= 2*radius)
        visible_inside = bool(abs(local[0]) <= width/2 and abs(local[1]) <= depth/2 and -height <= local[2] < 0)
        self.belief = int(visible_inside)
        return dict(position=p, velocity=velocity, tracking_valid=bool(valid), in_box=self.belief)


def ball_visible(model, data, camera_position, ball_position, ball_body_id,
                 forward=None, half_fov_rad=None):
    """Point-centre ray test, including robot and box collision geometry.

    A centre-ray approximation; no image segmentation, partial visibility,
    camera latency or real optical properties are simulated.
    """
    camera = np.asarray(camera_position, dtype=float)
    delta = np.asarray(ball_position, dtype=float)-camera
    distance = float(np.linalg.norm(delta))
    if distance <= 0:
        return False
    ray = delta/distance
    if forward is not None and half_fov_rad is not None:
        direction = np.asarray(forward, dtype=float)
        direction /= np.linalg.norm(direction)
        if float(ray @ direction) < np.cos(half_fov_rad):
            return False
    geom = np.array([-1], dtype=np.int32)
    hit = mujoco.mj_ray(model, data, camera, ray, None, 1, int(ball_body_id), geom)
    return bool(hit < 0 or hit >= distance)


def mixed_throw(rng, origin, level=1.0, drop_fraction=.3):
    """Sample incoming lobs and overhead drops; returned metadata labels type.

    Drop distribution includes center and rim locations. Actual contacts and
    bounces are determined by physics, not forced by this generator.
    The caller must apply its usual workspace/feasibility checks.
    """
    if not 0 <= drop_fraction <= 1:
        raise ValueError('drop_fraction must be between 0 and 1')
    origin = np.asarray(origin, dtype=float)
    if rng.random() >= drop_fraction:
        throw, meta = varied_throw(rng, origin, level)
        meta['category'] = 'incoming'
        return throw, meta
    offset = rng.uniform(-.075, .075, size=2)
    height = float(rng.uniform(.08, .35))
    velocity = np.array([*rng.uniform(-.15, .15, size=2), -rng.uniform(.05, 1.0)])
    pos = origin + np.array([offset[0], offset[1], height])
    return Throw(pos.tolist(), velocity.tolist(), 'overhead_drop'), dict(
        category='overhead_drop', offset_xy_m=offset.tolist(), height_m=height,
        initial_speed_m_s=float(np.linalg.norm(velocity)))


from catchbox.core import BallState
from catchbox.learning.limited_pose_environment import LimitedPoseCatchCore


class SteadyMotionObservationCore(LimitedPoseCatchCore):
    """Camera-based 25-input simulation with the steady-motion reward.

    Camera pose is a placeholder until the physical setup is calibrated.
    """
    def __init__(self, seed=0, training=False, total_steps=50000, record=False,
                 camera_position=(1.2, 0., 1.4), drop_fraction=.3):
        self.camera = CameraEstimator(seed=seed+9171)
        self.camera_position = np.asarray(camera_position, dtype=float)
        self.drop_fraction = drop_fraction
        self.camera_cache = None
        super().__init__(seed, training, total_steps, record)

    def reset(self, throw=None, meta=None, seed=None):
        self.camera.reset(); self.camera_cache = None
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.camera.rng = np.random.default_rng(int(seed)+9171)
        if throw is not None:
            return super().reset(throw, meta, seed=None)
        # Choose the category once so rejection does not bias the accepted mix.
        category_drop = self.rng.random() < self.drop_fraction
        # Let the existing reset validate descending prediction and sphere.
        # Sampling is independent of policy performance.
        for _ in range(5000):
            candidate, metadata = mixed_throw(self.rng, self.sim.nominal_opening,
                                             drop_fraction=float(category_drop))
            try:
                return super().reset(candidate, metadata, seed=None)
            except ValueError:
                self.camera.reset(); self.camera_cache = None
        raise RuntimeError('Could not sample a valid mixed throw')

    def camera_frame(self):
        t = float(self.sim.data.time)
        if self.camera_cache is not None and self.camera_cache['time'] == t:
            return self.camera_cache
        s = self.sim; ball = s.read().position
        visible = ball_visible(s.model, s.data, self.camera_position, ball, s.ball_id)
        box = s.data.site_xpos[s.site]; rotation = s.data.site_xmat[s.site]
        obs = self.camera.update(t, ball if visible else None, box, rotation,
                                 self.cfg.box_width, self.cfg.box_depth, self.cfg.radius, self.cfg.box_height)
        obs.update(time=t, visible=visible)
        self.camera_cache = obs
        return obs

    def prediction_now(self, in_box):
        camera = self.camera_frame()
        state = BallState(camera['time'], camera['position'], camera['velocity'],
                          valid=camera['tracking_valid'], acceleration=np.array([0., 0., -9.81]))
        return self.predictor.predict(state, self.height, in_contact=camera['in_box'] == 1)

    def observation(self):
        camera = self.camera_frame(); self.tracking_valid = camera['tracking_valid']
        box, box_v, box_w, _ = self.box_ball_state()
        pred = self.prediction_now(False)
        p, v, t = (pred.position, pred.velocity, pred.time_to_contact) if pred.valid else (np.zeros(3), np.zeros(3), 0.)
        values = np.r_[box, box_v/3, box_w/3, camera['position'], camera['velocity']/5,
                       p, v/5, np.clip(t, 0, 2), float(camera['tracking_valid']),
                       float(pred.valid), float(camera['in_box'])]
        return np.asarray(np.clip(values, -10, 10), dtype=np.float32)

    def step(self, action):
        """Apply the agreed bounded reward, continuing after first catch."""
        old_caught = self.success
        obs, old_reward, _, _, info = super().step(action)
        box, linear, _, _ = self.box_ball_state()
        inside = bool(self.inside)
        distance = float(np.linalg.norm(self.sim.read().position-box))
        r_dist = (1.0-float(inside))*min(distance/.20, 1.0)**2
        r_catch = float(self.success and not old_caught)
        # 5 cm/s is a provisional physical threshold, not a calibrated limit.
        r_steady = float(inside)*max(0., 1.-float(np.linalg.norm(linear))/.05)
        terms = dict(distance=-.02*r_dist, in_box=.08*float(inside),
                     catch=.8*r_catch, steady=.1*r_steady)
        reward = sum(terms.values())
        self.return_sum += reward-old_reward
        self.reward_terms = terms; info['reward_terms'] = terms
        # Retention does not end the episode; bounces can be recovered.
        ball = self.sim.read()
        lost = bool(ball.position[2]<.04 or np.linalg.norm(ball.position-box)>2.5)
        terminated = lost
        truncated = bool(self.sim.data.time >= self.cfg.duration-1e-8)
        self.done = terminated or truncated
        info['result'] = self.result()
        return obs, float(reward), terminated, truncated, info


def steady_motion_gym_env(seed=0, training=False, total_steps=50000, **kwargs):
    """Gym adapter; constructing it does not start training."""
    import gymnasium as gym
    class Adapter(gym.Env):
        def __init__(self):
            self.core = SteadyMotionObservationCore(seed, training, total_steps, **kwargs)
            self.observation_space = gym.spaces.Box(-10., 10., shape=(25,), dtype=np.float32)
            self.action_space = gym.spaces.Box(-1., 1., shape=(3,), dtype=np.float32)
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            return self.core.reset(seed=seed), {}
        def step(self, action):
            return self.core.step(action)
    return Adapter()
