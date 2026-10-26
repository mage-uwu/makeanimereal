"""MoCapAct's multi-clip policy (an NPMP) as plain arrays: mocapact/npmp.npz.

    python mocapact/export.py <multiclip_policy/full_dataset/model/model.ckpt> [out.npz]

Layers: the reference encoder (Linear, LayerNorm, ELU) x 2 then Linear to 120 = embedding mean and
log std; the action decoder (Linear, LayerNorm, ELU) x 3 then Linear to the 56 actions. Each net's
input is its observables, concatenated in the checkpoint's order and normalized with its running
mean and std (sqrt(var + 1e-8)); the encoder also takes the previous embedding (not normalized).
"""
import json
import sys

import numpy as np

from ckpt import load

src, out = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else 'mocapact/npmp.npz')
ck = load(src)
hp, sd = ck['hyper_parameters'], ck['state_dict']
keys = hp['observables'].state['_fields']
rms = hp['features_extractor_kwargs']['obs_rms']
arrays = {k.replace('.', '_'): v.numpy().astype(np.float32) for k, v in sd.items()}
for net in ('ref_encoder', 'decoder'):
    arrays[f'{net}_mean'] = np.concatenate([rms[k].state['mean'] for k in keys[net]]).astype(np.float32)
    arrays[f'{net}_std'] = np.concatenate([np.sqrt(rms[k].state['var'] + 1e-8) for k in keys[net]]).astype(np.float32)
sizes = {k: int(hp['observation_space'].state['spaces'][k].state['_shape'][0]) for k in set(keys['ref_encoder']) | set(keys['decoder'])}
meta = dict(observables={k: list(v) for k, v in keys.items()}, sizes=sizes, ref_steps=[int(s) for s in hp['ref_steps']],
            embed_size=int(hp['embed_size']), activation='elu', layer_norm=bool(hp['layer_norm']), std_dev=float(hp['std_dev']))
np.savez(out, meta=np.frombuffer(json.dumps(meta).encode(), np.uint8), **arrays)
print(out, sum(a.size for k, a in arrays.items()), 'numbers;', json.dumps(meta)[:400])
