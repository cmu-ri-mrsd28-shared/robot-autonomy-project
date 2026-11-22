"""One fresh CrossQ run for the thin box and restricted three-joint controller."""
import json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sb3_contrib import CrossQ
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from catchbox.learning.limited_pose_environment import LimitedPoseCatchCore, limited_pose_gym_env
from catchbox.learning.trajectory import freeze_bank
from catchbox.learning.train import evaluate
from catchbox.learning.report import video, curve

root=Path('results/limited-thin-50k'); out=root/'box'
if root.exists(): raise RuntimeError('Run directory exists; refusing to overwrite or restart')
out.mkdir(parents=True)
torch.set_num_threads(4); torch.set_num_interop_threads(1)
env=limited_pose_gym_env(seed=112023,training=True,total_steps=50000)
val=LimitedPoseCatchCore(seed=20261209); vb=freeze_bank(val.sim,20261209,8,out/'validation-throws.json')
test=LimitedPoseCatchCore(seed=20261210); tb=freeze_bank(test.sim,20261210,50,out/'heldout-throws.json')
model=CrossQ('MlpPolicy',env,learning_rate=.001,buffer_size=250000,learning_starts=1000,batch_size=256,gamma=.99,train_freq=1,gradient_steps=1,policy_delay=3,ent_coef='auto_0.1',policy_kwargs=dict(net_arch=dict(pi=[128,128],qf=[256,256]),n_critics=2),seed=112023,device='cpu',verbose=0)
model.set_logger(configure(str(out),['json']))
params=dict(actor=sum(p.numel() for p in model.actor.parameters()),critics=sum(p.numel() for p in model.critic.parameters()))
spec=dict(algorithm='CrossQ',initialization='fresh; no prior model or replay',steps=50000,physics_step_s=.001,decision_interval_s=.02,ball_mass_kg=.002,ball_diameter_m=.04,observations=25,actions=3,controlled_joints=[1,4,6],workspace_radius_m=.2,train_seed=112023,validation_seed=20261209,heldout_seed=20261210,heldout_episodes=50,parameters=params,physics=vars(env.core.cfg),gamma=.99,box='armbox-thin',grasp='ideal rigid, 0.25 kg assumed',reward='distance, intercept, retention, stability, success')
(out/'spec.json').write_text(json.dumps(spec,indent=2));print('Started fresh thin box 50k:',json.dumps(params),flush=True)
class Progress(BaseCallback):
 def __init__(self): super().__init__(); self.history=[]; self.start=time.monotonic()
 def _on_step(self):
  if self.num_timesteps%1000==0:
   row={k:float(v) for k,v in self.model.logger.name_to_value.items() if isinstance(v,(int,float,np.number))};row.update(step=self.num_timesteps,wall_s=time.monotonic()-self.start,episodes=self.model._episode_num)
   with (out/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   self.model.logger.dump(self.num_timesteps)
   print(f"step {self.num_timesteps}/50000; episodes {self.model._episode_num}; elapsed {time.monotonic()-self.start:.0f}s",flush=True)
  if self.num_timesteps%10000==0:
   self.model.save(out/f'checkpoint-{self.num_timesteps}');self.model.save_replay_buffer(out/'checkpoint-replay.pkl')
   r=evaluate(self.model,val,vb);item={k:v for k,v in r.items() if k!='results'};item['step']=self.num_timesteps;self.history.append(item);(out/'validation.json').write_text(json.dumps(self.history,indent=2))
   print(f"validation {r['successes']}/{r['n']}",flush=True)
  return True
callback=Progress();start=time.monotonic();model.learn(total_timesteps=50000,callback=callback,log_interval=100)
model.save(out/'final_model');model.save_replay_buffer(out/'replay.pkl')
print('Evaluating final checkpoint on 50 held-out throws',flush=True)
r=evaluate(model,test,tb,out/'examples');(out/'evaluation.json').write_text(json.dumps(r,indent=2))
summary=dict(steps=50000,successes=r['successes'],episodes=r['n'],rate=r['rate'],entered=sum(x['entered'] for x in r['results']),wilson95=r['wilson95'],train_seconds=time.monotonic()-start,throws_sha256=r['throws_sha256'],parameters=params)
(root/'summary.json').write_text(json.dumps(summary,indent=2));print('RESULT',json.dumps(summary),flush=True)
parts=[]
for tag in ['passing','failing']:
 src=out/'examples'/f'{tag}.json';dest=root/'media'/f'{tag}.mp4'
 if src.exists():
  record=json.loads(src.read_text());record['controller']='trained 3-joint policy';src.write_text(json.dumps(record));video('box',src,dest,printed_box=True,printed_box_variant='thin')
  parts.append(f'<article><h3>{tag.title()}</h3><video controls preload="metadata" src="media/{tag}.mp4"></video></article>')
 else:parts.append(f'<article><h3>{tag.title()}</h3><p>None observed in the held-out test.</p></article>')
fig=curve([(x['step'],100*x['rate']) for x in callback.history],'Validation retention','Percent')
(root/'report.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>Thin box 50k</title><style>body{{font:16px system-ui;max-width:950px;margin:30px auto}}video{{width:100%}}article{{max-width:650px}}</style><h1>Thin box: 50,000-step policy</h1><p>Simulation only. Fresh CrossQ, 25 inputs, 3 position outputs, joints 1, 4 and 6, 20 ms decisions.</p><h2>Final checkpoint: held-out throws</h2><p>Entered {summary['entered']}/50; retained one second {summary['successes']}/50 ({100*summary['rate']:.1f}%).</p>{''.join(parts)}<h2>Validation</h2>{fig}<p>Ideal ball sensing and rigid no-slip grasp; STL collision is approximated by box walls. Workspace radius 20 cm.</p>''')
print('Report complete',flush=True)
