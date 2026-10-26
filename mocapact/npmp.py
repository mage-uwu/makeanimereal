"""MoCapAct's multi-clip policy in numpy (weights from export.py).

    p = Npmp('mocapact/npmp.npz'); z = p.initial(); action, z = p.act(obs, z)

obs: dm_control's observation dict for the CMU humanoid (MultiClipMocapTracking, ref_steps 1..5).
"""
import json

import numpy as np


def _elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def _ln(x, g, b):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * g + b


class Npmp:
    def __init__(self, path):
        f = np.load(path)
        self.w = {k: f[k] for k in f.files if k != 'meta'}
        self.meta = json.loads(f['meta'].tobytes())
        self.embed = self.meta['embed_size']

    def _mlp(self, prefix, idx, x):
        w = self.w
        for i in idx[:-1]:
            x = _elu(_ln(x @ w[f'{prefix}_{i}_weight'].T + w[f'{prefix}_{i}_bias'], w[f'{prefix}_{i + 1}_weight'], w[f'{prefix}_{i + 1}_bias']))
        return x @ w[f'{prefix}_{idx[-1]}_weight'].T + w[f'{prefix}_{idx[-1]}_bias']

    def features(self, obs, net):
        x = np.concatenate([np.asarray(obs[k], np.float64).ravel() for k in self.meta['observables'][net]])
        return (x - self.w[f'{net}_mean']) / self.w[f'{net}_std']

    def initial(self):
        return np.zeros(self.embed)

    def act(self, obs, z):
        """Deterministic: the embedding's mean, the action's mean (clipped to the actuators' range)."""
        z = self._mlp('reference_encoder_encoder', [0, 3, 6], np.concatenate([self.features(obs, 'ref_encoder'), z]))[:self.embed]
        a = self._mlp('action_decoder', [0, 3, 6, 9], np.concatenate([self.features(obs, 'decoder'), z]))
        return np.clip(a, -1, 1), z
