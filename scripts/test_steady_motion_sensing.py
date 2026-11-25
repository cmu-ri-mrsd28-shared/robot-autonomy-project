import numpy as np
import mujoco
from steady_motion_sensing import CameraEstimator, ball_visible, mixed_throw


def test_camera_disappearance_does_not_reveal_hidden_ball():
    c = CameraEstimator(seed=1, noise_m=0, entry_window_s=.04)
    for i, z in enumerate([.08, .06, .04, .02]):
        obs = c.update(i*.02, [0, 0, z], [0, 0, 0], np.eye(3), .135, .135, .02)
    assert obs['tracking_valid'] and obs['in_box'] == 0
    hidden = c.update(.08, None, [0, 0, 0], np.eye(3), .135, .135, .02)
    assert not hidden['tracking_valid'] and hidden['in_box'] == -1
    hidden = c.update(.14, None, [0, 0, 0], np.eye(3), .135, .135, .02)
    assert hidden['in_box'] == 1
    assert np.all(hidden['position'] == 0) and np.all(hidden['velocity'] == 0)
    visible = c.update(.16, [.2, 0, .1], [0, 0, 0], np.eye(3), .135, .135, .02)
    assert visible['in_box'] == 0 and not visible['tracking_valid']


def test_unknown_loss_and_ray_occlusion():
    c = CameraEstimator(noise_m=0)
    assert c.update(0, None, [0]*3, np.eye(3), .135, .135, .02)['in_box'] == -1
    xml = '<mujoco><worldbody><geom type="box" pos="0 0 0" size=".1 .1 .1"/><body name="ball" pos="0 0 -1"><geom type="sphere" size=".02"/></body></worldbody></mujoco>'
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m); mujoco.mj_forward(m,d)
    ball = m.body('ball').id
    assert not ball_visible(m, d, [0,0,1], [0,0,-1], ball)
    assert ball_visible(m, d, [1,0,-1], [0,0,-1], ball)


def test_mixed_throws():
    rng = np.random.default_rng(4); origin = np.array([.4, 0, .3])
    samples = [mixed_throw(rng, origin) for _ in range(100)]
    categories = {meta['category'] for _, meta in samples}
    assert categories == {'incoming', 'overhead_drop'}
    for throw, meta in samples:
        if meta['category'] == 'overhead_drop':
            assert throw.position[2] > origin[2] and throw.velocity[2] < 0


def test_environment_reward_and_hidden_observation(monkeypatch):
    import steady_motion_sensing as module
    monkeypatch.setattr(module, 'ball_visible', lambda *args, **kwargs: False)
    core = module.SteadyMotionObservationCore(seed=72, drop_fraction=1)
    obs = core.reset(seed=72)
    assert obs.shape == (25,)
    assert np.all(obs[9:15] == 0) and np.all(obs[15:22] == 0)
    assert tuple(obs[22:25]) == (0, 0, -1)
    obs, reward, term, trunc, info = core.step(np.zeros(3))
    terms = info['reward_terms']
    assert abs(reward-sum(terms.values())) < 1e-12
    assert -.020001 <= terms['distance'] <= 0
    assert terms['in_box'] in (0, .08) and 0 <= terms['steady'] <= .1
    assert np.all(obs[9:15] == 0)
    assert tuple(obs[22:25]) == (0, 0, -1)


def test_post_catch_continues(monkeypatch):
    import steady_motion_sensing as module
    core = module.SteadyMotionObservationCore(seed=73, drop_fraction=1)
    core.reset(seed=73)
    core.metrics.retained = True
    _, _, terminated, truncated, _ = core.step(np.zeros(3))
    assert not terminated and not truncated and not core.done
