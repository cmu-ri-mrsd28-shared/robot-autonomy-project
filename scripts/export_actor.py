"""Export the trained deterministic actor for NumPy-only inference.

Requires the training environment's sb3-contrib and PyTorch. Does not train.
"""
import argparse, hashlib
from pathlib import Path
import numpy as np
from sb3_contrib import CrossQ

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=Path('artifacts/limited-thin-model/final_model.zip'))
    p.add_argument('--output',type=Path,default=Path('artifacts/limited-thin-model/actor.npz'))
    a=p.parse_args();model=CrossQ.load(str(a.checkpoint),device='cpu');model.policy.set_training_mode(False)
    s=model.actor.state_dict();data={'checkpoint_sha256':np.array(hashlib.sha256(a.checkpoint.read_bytes()).hexdigest())}
    for i,j in enumerate([0,3,6]):
        layer=model.actor.latent_pi[j]
        if layer.eps!=.001:raise ValueError('Unexpected batch renormalization epsilon')
        for short,long in [('mean','ra_mean'),('var','ra_var'),('scale','scale'),('bias','bias')]:
            data[f'n{i}_{short}']=s[f'latent_pi.{j}.{long}'].detach().cpu().numpy()
    for i,prefix in [(1,'latent_pi.1'),(2,'latent_pi.4'),(3,'mu')]:
        for short,long in [('w','weight'),('b','bias')]:data[f'{short}{i}']=s[f'{prefix}.{long}'].detach().cpu().numpy()
    np.savez(a.output,**data)
    from catchbox.hardware.policy import Actor
    actor=Actor(a.output,checkpoint=a.checkpoint);rng=np.random.default_rng(42)
    errors=[np.max(np.abs(actor.predict(x)-model.predict(x,deterministic=True)[0])) for x in rng.uniform(-3,3,(1000,25)).astype(np.float32)]
    worst=float(max(errors))
    if worst>1e-4:raise RuntimeError('Export mismatch: max error %g'%worst)
    print('Exported %s; max actor error %.8f'%(a.output,worst))
if __name__=='__main__':main()
