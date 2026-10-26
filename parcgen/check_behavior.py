"""Check parcgen's sampling pipeline on PARC's own data.

    python3 parcgen/check_behavior.py parc_mdm.npz dec_release.zip [n]

For n dataset motions: start the generator from two real frames on the real terrain, steered the
way the real motion went over the next half second, and compare its window with what happened:
the root's position after 0.4 s (against standing still, and against carrying on at the same
velocity), and how far feet go below the terrain. Canonicalization or feature bugs show up as
wrong directions and feet through the ground.
"""
import os
import sys
import zipfile

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here); sys.path.insert(0, os.path.join(here, '..', 'parkour'))
import parc  # noqa: E402
from model import Controller, ParcGen, fk  # noqa: E402

gen = ParcGen(sys.argv[1])
z = zipfile.ZipFile(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
names = sorted(x for x in z.namelist() if x.endswith('_opt_dm.pkl') and 'rerun_april272025_iter_5' in x)
rng = np.random.default_rng(0)
err_gen, err_still, err_vel, pen, speed_ratio = [], [], [], [], []
for name in rng.choice(names, n, replace=False):
    d = parc.load(z.read(name)); m, t = d['motion'], d['terrain']
    F = len(m['root_pos'])
    if F < 30:
        continue
    f0 = int(rng.integers(5, F - 20))
    prev = {'root_pos': m['root_pos'][f0 - 1:f0 + 1].astype(float), 'root_rot': m['root_rot'][f0 - 1:f0 + 1].astype(float),
            'joint_rot': m['joint_rot'][f0 - 1:f0 + 1].astype(float), 'contacts': m['body_contacts'][f0 - 1:f0 + 1].astype(float)}
    truth = m['root_pos'][f0 + 12].astype(float)                       # 12 frames (0.4 s) after the last prev frame
    move = m['root_pos'][f0 + 13, :2] - m['root_pos'][f0, :2]
    target = move / np.linalg.norm(move) if np.linalg.norm(move) > 0.1 else np.zeros(2)
    ctl = Controller(gen, prev, t['hf'], t['min_point'], float(t['dx']))
    out = ctl.generate(prev, target)
    got = out['root_pos'][13]                                          # window frame 13 = 12 after prev frame 1
    err_gen.append(np.linalg.norm(got - truth))
    err_still.append(np.linalg.norm(prev['root_pos'][1] - truth))
    err_vel.append(np.linalg.norm(prev['root_pos'][1] + 12 * (prev['root_pos'][1] - prev['root_pos'][0]) - truth))
    real_speed = np.linalg.norm(truth[:2] - prev['root_pos'][1, :2]); gen_speed = np.linalg.norm(got[:2] - prev['root_pos'][1, :2])
    if real_speed > 0.3:
        speed_ratio.append(gen_speed / real_speed)
    body = fk(out['root_pos'], out['root_rot'], out['joint_rot'])
    feet = body[2:, [11, 14]]
    # Ground under a foot: the lowest of the cells around it (a foot beside a wall is not inside it).
    i = (feet[..., :2] - ctl.min_point) / ctl.cell
    lo = np.clip(np.floor(i).astype(int), 0, np.array(ctl.hf.shape) - 1); hi = np.clip(lo + 1, 0, np.array(ctl.hf.shape) - 1)
    ground = np.minimum.reduce([ctl.hf[a[..., 0], b[..., 1]] for a in (lo, hi) for b in (lo, hi)])
    pen.append(max(0.0, -(feet[..., 2] - ground).min()))
print(f'{len(err_gen)} windows from real frames on real terrain, root position 0.4 s later vs the data:')
print(f'  generator      median error {np.median(err_gen):.3f} m  (mean {np.mean(err_gen):.3f})')
print(f'  stand still    median error {np.median(err_still):.3f} m')
print(f'  same velocity  median error {np.median(err_vel):.3f} m')
print(f'  generated / real distance covered: median {np.median(speed_ratio):.2f}')
print(f'  deepest foot below terrain (ankle): median {np.median(pen) * 100:.1f} cm, worst {np.max(pen) * 100:.1f} cm')
