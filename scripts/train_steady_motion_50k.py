"""One fresh 50k CrossQ run. Invoke explicitly; never resumes or overwrites."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import CrossQ
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from catchbox.core import Throw
from steady_motion_sensing import SteadyMotionObservationCore, steady_motion_gym_env

OUT=Path('results/steady-motion-50k')
BANKS=Path('data/steady-motion-evaluation')
ORIGINAL=Path('artifacts/limited-thin-model/final_model.zip')

def main():
    for p in [ORIGINAL, BANKS/'movement-required.json', BANKS/'overhead-drops.json']:
        if not p.is_file(): raise FileNotFoundError(p)
    OUT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    env=steady_motion_gym_env(seed=20270522,training=True,total_steps=50000)
    model=CrossQ('MlpPolicy',env,learning_rate=.001,buffer_size=250000,learning_starts=1000,
        batch_size=256,gamma=.99,train_freq=1,gradient_steps=1,policy_delay=3,ent_coef='auto_0.1',
        policy_kwargs=dict(net_arch=dict(pi=[128,128],qf=[256,256]),n_critics=2),seed=20270522,device='cpu')
    model.set_logger(configure(str(OUT),['json']))
    spec=dict(steps=50000,seed=20270522,initialization='fresh model and replay',camera_position=[1.2,0,1.4],
        camera='center-ray visibility; 2mm position noise; no latency; unknown=-1, outside=0, probable_inside=1',
        camera_calibrated=False,proposed_drop_fraction=.3,steady_speed_threshold_m_s=.05,
        reward='-0.02*r_dist + 0.08*r_in_box + 0.8*r_catch + 0.1*r_steady',
        original_sha256=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
        actor_parameters=sum(p.numel() for p in model.actor.parameters()),
        critic_parameters=sum(p.numel() for p in model.critic.parameters()))
    (OUT/'spec.json').write_text(json.dumps(spec,indent=2))
    class Progress(BaseCallback):
        def __init__(self): super().__init__(); self.start=time.monotonic()
        def _on_step(self):
            if self.num_timesteps%1000 == 0:
                row={k:float(v) for k,v in self.model.logger.name_to_value.items() if isinstance(v,(int,float,np.number))}
                row.update(step=self.num_timesteps,elapsed_s=time.monotonic()-self.start)
                with (OUT/'training.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                print('step %d/50000'%self.num_timesteps,flush=True)
            if self.num_timesteps%10000 == 0:
                self.model.save(OUT/('checkpoint-%d'%self.num_timesteps))
                self.model.save_replay_buffer(OUT/'checkpoint-replay.pkl')
            return True
    model.learn(total_timesteps=50000,callback=Progress())
    model.save(OUT/'final_model'); model.save_replay_buffer(OUT/'replay.pkl')
    for label, policy in [('original',CrossQ.load(str(ORIGINAL),device='cpu')),('steady-motion',model)]:
        for bank_name in ('movement-required','overhead-drops'):
            bank=json.loads((BANKS/(bank_name+'.json')).read_text())['throws']; rows=[]
            for i,item in enumerate(bank):
                core=SteadyMotionObservationCore(seed=20270523+i)
                obs=core.reset(Throw(**item['throw']),item['meta'],seed=20270523+i)
                path=0.; last=None; first_catch=False; escaped_seen=False; recovered=False
                while True:
                    obs,_,terminated,truncated,info=core.step(policy.predict(obs,deterministic=True)[0])
                    escaped_seen = escaped_seen or core.metrics.escaped
                    if escaped_seen and core.inside and core.metrics.since is not None:
                        recovered = recovered or core.sim.data.time-core.metrics.since >= 1.-1e-9
                    pos=core.sim.data.site_xpos[core.sim.site].copy()
                    if first_catch and last is not None: path+=float(np.linalg.norm(pos-last))
                    if core.success: first_catch=True; last=pos
                    if terminated or truncated: break
                rows.append(dict(result=info['result'],retained_2s=core.metrics.max_retention>=2.-1e-9,escaped_after_entry=escaped_seen,recovered_after_escape=recovered,post_catch_path_m=path if first_catch else None))
            (OUT/(label+'-'+bank_name+'.json')).write_text(json.dumps(rows,indent=2))
            print(label,bank_name,'1s',sum(r['result']['success'] for r in rows),'2s',sum(r['retained_2s'] for r in rows),flush=True)

if __name__=='__main__': main()
