"""Build independent stationary-miss and drop evaluation banks; no policy training."""
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import mujoco
import numpy as np
from catchbox.learning.limited_pose_environment import LimitedPoseCatchCore
from catchbox.learning.trajectory import varied_throw, in_workspace
from steady_motion_sensing import mixed_throw

OUT = Path('data/steady-motion-evaluation')
SEED = 20270521
COUNT = 50
if OUT.exists():
    raise RuntimeError('Evaluation directory exists; refusing to overwrite')
OUT.mkdir(parents=True)
rng = np.random.default_rng(SEED)
core = LimitedPoseCatchCore(seed=SEED)
reasons = Counter(); incoming = []; drops = []
for attempt in range(100000):
    if len(incoming) >= COUNT:
        break
    throw, meta = varied_throw(rng, core.sim.nominal_opening)
    if meta['speed_m_s'] > 5:
        reasons['speed'] += 1; continue
    try:
        core.reset(throw, meta)
    except ValueError:
        reasons['release_prediction'] += 1; continue
    s = core.sim; start = core.start_q.copy(); initial = core.start_pos.copy()
    prev = s.read().position.copy(); crossing = None; before_contact = False
    for _ in range(round(core.cfg.duration/core.cfg.timestep)):
        s.torque(start); mujoco.mj_step(s.model,s.data); mujoco.mj_forward(s.model,s.data); core._metrics()
        p = s.read().position.copy()
        if crossing is None and prev[2] >= core.height > p[2]:
            f = (prev[2]-core.height)/(prev[2]-p[2]); crossing = prev+f*(p-prev)
            crossing_time = float(s.data.time-core.cfg.timestep+f*core.cfg.timestep)
            before_contact = core.contact
        prev = p
        if core.success or p[2] < .04 or np.linalg.norm(p-s.data.site_xpos[s.site]) > 2.5:
            break
    if crossing is None or before_contact:
        reasons['crossing_or_early_contact'] += 1; continue
    if not in_workspace(crossing,core.workspace_center,.2):
        reasons['crossing_outside_sphere'] += 1; continue
    if max(abs(crossing[0]-initial[0]),abs(crossing[1]-initial[1])) < .075+core.cfg.radius+.005:
        reasons['not_clear_miss'] += 1; continue
    if core.entered or core.success:
        reasons['stationary_entered'] += 1; continue
    core.reset(throw,meta); q,err = core.restricted_ik(crossing)
    motion_time = max(np.max(np.abs(q-core.start_q))/2.175, np.sqrt(2*np.max(np.abs(q-core.start_q))/12))
    if err > .04 or motion_time+.05 > crossing_time:
        reasons['ik_or_time'] += 1; continue
    incoming.append(dict(throw=asdict(throw),meta=meta,crossing=crossing.tolist(),stationary_entered=False,ik_error_m=err))
    if len(incoming)%10 == 0:
        print('movement-required %d/%d after %d proposals' % (len(incoming),COUNT,attempt+1),flush=True)
if len(incoming) != COUNT:
    raise RuntimeError('Could not produce required independent movement cases')
for attempt in range(10000):
    if len(drops) >= COUNT: break
    throw,meta = mixed_throw(rng,core.sim.nominal_opening,drop_fraction=1.)
    try: core.reset(throw,meta)
    except ValueError: continue
    drops.append(dict(throw=asdict(throw),meta=meta))
if len(drops) != COUNT: raise RuntimeError('Could not build drops')
for name,bank in [('movement-required',incoming),('overhead-drops',drops)]:
    (OUT/(name+'.json')).write_text(json.dumps(dict(seed=SEED,throws=bank),indent=2))
(OUT/'manifest.json').write_text(json.dumps(dict(seed=SEED,counts={'movement_required':len(incoming),'overhead_drops':len(drops)},rejections=dict(reasons),note='Overhead drops are not guaranteed to bounce. Movement-required bank is conditioned on stationary miss and feasibility; not a general throw distribution.'),indent=2))
print('Evaluation banks complete',flush=True)
