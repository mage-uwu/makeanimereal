"""Check parcgen.c against parcgen/model.py: same weights, terrain, frames and noise.

    python3 parcgen/test.py parc_mdm.npz dec_release.zip

Runs one generation window on PARC terrain at several settings (DDIM stride 10 and 20, CFG off
and at PARC's 0.65) and reports the largest root-position and joint-rotation difference, and the
native C time per window.
"""
import ctypes
import os
import subprocess
import sys
import time
import zipfile

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here); sys.path.insert(0, os.path.join(here, '..', 'parkour'))
import parc  # noqa: E402
from model import Controller, ParcGen  # noqa: E402

npz, src = sys.argv[1], sys.argv[2]
gen = ParcGen(npz)
so = os.path.join(here, 'libparcgen.so')
subprocess.run(['cc', '-O2', '-shared', '-fPIC', os.path.join(here, 'parcgen.c'), '-o', so], check=True)
lib = ctypes.CDLL(so)
for f in ('pg_weights', 'pg_prev', 'pg_noise', 'pg_out'):
    getattr(lib, f).restype = ctypes.POINTER(ctypes.c_float)
lib.pg_terrain.restype = ctypes.POINTER(ctypes.c_float)
lib.pg_terrain.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float]
lib.pg_generate.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_float]
assert lib.pg_create() == 0
blob = np.fromfile(npz.replace('.npz', '.bin'), '<f4')
assert blob.size == lib.pg_num_weights()
ctypes.memmove(lib.pg_weights(), blob.ctypes.data, blob.nbytes)
lib.pg_bind()

put = lambda ptr, a: ctypes.memmove(ptr, np.ascontiguousarray(a, np.float32).ctypes.data, np.asarray(a, np.float32).nbytes)
z = zipfile.ZipFile(src)
name = sorted(n for n in z.namelist() if n.endswith('teaser_3439_1_opt_dm.pkl'))[0]
d = parc.load(z.read(name)); m, t = d['motion'], d['terrain']
hf = np.asarray(t['hf'], np.float32)
put(lib.pg_terrain(hf.shape[0], hf.shape[1], float(t['min_point'][0]), float(t['min_point'][1]), float(t['dx'])), hf)
f0 = 40
prev = {'root_pos': m['root_pos'][f0 - 1:f0 + 1].astype(float), 'root_rot': m['root_rot'][f0 - 1:f0 + 1].astype(float),
        'joint_rot': m['joint_rot'][f0 - 1:f0 + 1].astype(float), 'contacts': m['body_contacts'][f0 - 1:f0 + 1].astype(float)}
put(lib.pg_prev(), np.concatenate([prev['root_pos'], prev['root_rot'], prev['joint_rot'].reshape(2, -1), prev['contacts']], -1))
noise = np.random.default_rng(3).standard_normal((15, 91))
put(lib.pg_noise(), noise)
ctl = Controller(gen, prev, hf, t['min_point'], float(t['dx']))
target = np.array([0.6, 0.8])


def angle(a, b):
    b = np.where((a * b).sum(-1, keepdims=True) < 0, -b, b)
    return np.degrees(2 * np.arctan2(np.linalg.norm(a - b, axis=-1), np.linalg.norm(a + b, axis=-1)))


worst = 0
for stride, cfg in ((10, None), (10, 0.65), (20, None)):
    ctl.stride, ctl.cfg_scale = stride, cfg
    ref = ctl.generate(prev, target, noise=noise)
    t0 = time.perf_counter()
    lib.pg_generate(float(target[0]), float(target[1]), stride, -1.0 if cfg is None else cfg)
    ms = (time.perf_counter() - t0) * 1000
    out = np.ctypeslib.as_array(lib.pg_out(), (15 * 78,)).reshape(15, 78).astype(float)
    dp = np.abs(out[:, :3] - ref['root_pos']).max()
    da = max(angle(out[:, 3:7], ref['root_rot']).max(), angle(out[:, 7:63].reshape(15, 14, 4), ref['joint_rot']).max())
    dc = np.abs(out[:, 63:] - ref['contacts']).max()
    worst = max(worst, da)
    print(f'stride {stride}, cfg {cfg}: C vs numpy max root {dp:.2e} m, rotation {da:.2e} deg, contacts {dc:.2e} (root moves {np.abs(out[-1, :2] - out[0, :2]).max():.2f} m); C {ms:.0f} ms/window')
assert worst < 0.5, 'C and numpy disagree'
print('C matches numpy')
