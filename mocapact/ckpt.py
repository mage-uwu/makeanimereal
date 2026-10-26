"""Read a MoCapAct checkpoint (PyTorch Lightning .ckpt) without running its pickled code.

The pickle names classes from MoCapAct, stable-baselines3, gym and ml_collections; none of them is
imported. Tensors and numpy arrays are rebuilt, every other class becomes an inert stub that keeps
its constructor arguments and state, which is all we need: the weights, the hyperparameters and the
observation normalizer.

    ck = load('model.ckpt'); ck['state_dict'], ck['hyper_parameters']
"""
import codecs
import collections
import io
import pickle
import zipfile

import numpy as np
import torch


class Stub:
    def __init__(self, *args, **kw):
        self.args, self.kw, self.state = args, kw, None

    def __setstate__(self, state):
        self.state = state

    def __getitem__(self, k):
        return self.state[k]


SAFE = {
    ('collections', 'OrderedDict'): collections.OrderedDict,
    ('builtins', 'set'): set, ('builtins', 'slice'): slice, ('_codecs', 'encode'): codecs.encode,
    ('torch._utils', '_rebuild_tensor_v2'): torch._utils._rebuild_tensor_v2,
    ('torch._utils', '_rebuild_parameter'): torch._utils._rebuild_parameter,
    ('torch', 'Size'): torch.Size,
    ('numpy', 'ndarray'): np.ndarray, ('numpy', 'dtype'): np.dtype,
    ('numpy.core.multiarray', '_reconstruct'): np.core.multiarray._reconstruct,
    ('numpy.core.multiarray', 'scalar'): np.core.multiarray.scalar,
}


def load(path):
    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split('/')[0]

    def persistent_load(pid):
        _, storage_type, key, _, _ = pid
        raw = torch.frombuffer(bytearray(z.read(f'{root}/data/{key}')), dtype=torch.uint8).untyped_storage()
        return torch.storage.TypedStorage(wrap_storage=raw, dtype=storage_type.dtype, _internal=True)

    class Unpickler(pickle.Unpickler):
        def find_class(self, mod, name):
            if (mod, name) in SAFE:
                return SAFE[(mod, name)]
            if mod == 'torch' and name.endswith('Storage'):
                return getattr(torch, name)
            return type(name, (Stub,), {'_name': f'{mod}.{name}'})

    u = Unpickler(io.BytesIO(z.read(f'{root}/data.pkl')))
    u.persistent_load = persistent_load
    return u.load()
