"""CPU-only inference for the exported, deterministic CrossQ actor."""
from pathlib import Path
import hashlib
import numpy as np

DEFAULT_ACTOR = Path(__file__).resolve().parents[2] / 'artifacts' / 'limited-thin-model' / 'actor.npz'
STEADY_MOTION_ACTOR = Path(__file__).resolve().parents[2] / 'artifacts' / 'steady-motion-model' / 'actor.npz'

class Actor:
    def __init__(self, path=DEFAULT_ACTOR, checkpoint=None):
        with np.load(path, allow_pickle=False) as f:
            self.weights = {k: f[k].copy() for k in f.files}
        checkpoint=Path(checkpoint) if checkpoint is not None else Path(path).with_name('final_model.zip')
        if 'checkpoint_sha256' not in self.weights or not checkpoint.is_file():
            raise ValueError('Actor checkpoint or checksum missing')
        digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if str(self.weights['checkpoint_sha256'])!=digest:raise ValueError('Actor does not match saved checkpoint; re-export')
        expected={'n0_mean':(25,), 'n0_var':(25,), 'n0_scale':(25,), 'n0_bias':(25,),
                  'w1':(128,25),'b1':(128,), 'n1_mean':(128,), 'n1_var':(128,), 'n1_scale':(128,), 'n1_bias':(128,),
                  'w2':(128,128),'b2':(128,), 'n2_mean':(128,), 'n2_var':(128,), 'n2_scale':(128,), 'n2_bias':(128,),
                  'w3':(3,128),'b3':(3,)}
        if any(k not in self.weights or self.weights[k].shape!=shape or not np.isfinite(self.weights[k]).all() for k,shape in expected.items()):
            raise ValueError('Invalid actor weights')
        if any(np.any(self.weights['n%d_var'%i]<0) for i in range(3)):raise ValueError('Invalid actor variance')
    def _norm(self, x, i):
        w=self.weights;prefix='n%d_'%i
        return (x-w[prefix+'mean']) / np.sqrt(w[prefix+'var']+np.float32(.001)) * w[prefix+'scale'] + w[prefix+'bias']
    def predict(self, observation):
        x=np.asarray(observation,dtype=np.float32)
        if x.shape!=(25,) or not np.isfinite(x).all():raise ValueError('Expected 25 finite observations')
        w=self.weights
        x=self._norm(x,0);x=np.maximum(w['w1']@x+w['b1'],0)
        x=self._norm(x,1);x=np.maximum(w['w2']@x+w['b2'],0)
        x=self._norm(x,2);return np.tanh(w['w3']@x+w['b3'])
