#!/usr/bin/env python3
"""Pick posed training views for the shader: frames of the mocap clips where the arms are raised or
held out (the hands, palms and armpits the stand-pose films never showed), at varied cameras.

    python3 coverage/motion/pick_poses.py --out out/posed/poses.json [--n 120] [--seed 0]
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--motions', default='coverage/live/motions.json')
ap.add_argument('--out', required=True)
ap.add_argument('--n', type=int, default=120)
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
m = json.load(open(a.motions))
B = m['bones']
cands, weight = [], []
for name, c in m['clips'].items():
    ax = np.array(c['axes']).reshape(c['frames'], len(B), 6)
    # How far each arm is up and out from hanging down: 0 hanging, 1 horizontal, 2 straight up.
    lift = 1 + np.maximum(ax[:, B.index('leftUpperArm'), 1], ax[:, B.index('rightUpperArm'), 1])
    fore = 1 + np.maximum(ax[:, B.index('leftLowerArm'), 1], ax[:, B.index('rightLowerArm'), 1])
    score = np.maximum(lift, fore)
    for f in range(0, c['frames'], 3):
        cands.append((name, round(f / m['fps'], 3)))
        weight.append(0.15 + max(0.0, score[f] - 0.4) ** 2)
# Each clip gets the same share whatever its length.
names = np.array([c for c, _ in cands])
weight = np.array(weight)
for name in m['clips']:
    weight[names == name] /= weight[names == name].sum()
weight /= weight.sum()
pick = rng.choice(len(cands), a.n, replace=False, p=weight)
poses = []
for k, i in enumerate(sorted(pick, key=lambda i: cands[i])):
    clip, t = cands[i]
    # Mostly from the front half, where palms and armpits face the camera; some from behind.
    yaw = float(rng.uniform(-100, 100) if rng.random() < 0.8 else rng.uniform(100, 260))
    poses.append({'clip': clip, 't': t, 'yaw': round(yaw % 360, 2), 'pitch': round(float(rng.uniform(-15, 35)), 2)})
json.dump(poses, open(a.out, 'w'), indent=0)
print({c: sum(p['clip'] == c for p in poses) for c in m['clips']})
