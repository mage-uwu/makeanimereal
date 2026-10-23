#!/usr/bin/env python3
"""Pick lit training views for the shader: poses and cameras as pick_poses.py, each under a light
from one of a few families (side key, sunset, backlight, noon, moonlight, neon, front), with the
lighting spelled out relative to the picture for the redraw (anchors.py reads light.desc).

    python3 coverage/motion/pick_lights.py --out out/lit/poses.json [--n 200] [--seed 1]
"""
import argparse
import json
import math

import numpy as np

FAMILIES = {
    # name: (yaw relative to the camera, degrees; pitch; key colour; key strength; ambient; ambient colour; words)
    'side': ((60, 120), (10, 40), [(1, 0.97, 0.92)], (2.8, 3.6), (0.35, 0.55), (1, 1, 1), 'a strong white key light'),
    'sunset': ((40, 140), (0, 15), [(1, 0.72, 0.45), (1, 0.62, 0.4)], (2.8, 3.4), (0.4, 0.6), (0.6, 0.62, 1), 'a warm orange sunset light'),
    'back': ((150, 210), (15, 45), [(1, 1, 1), (1, 0.95, 0.85)], (3.2, 3.8), (0.35, 0.5), (0.85, 0.88, 1), 'a bright backlight'),
    'noon': ((-40, 40), (65, 85), [(1, 0.98, 0.9)], (2.6, 3.2), (0.5, 0.7), (0.9, 0.95, 1), 'a bright overhead sun'),
    'moon': ((30, 150), (35, 65), [(0.7, 0.82, 1.15)], (2.0, 2.6), (0.3, 0.4), (0.5, 0.55, 0.95), 'cool blue moonlight'),
    'neon': ((50, 130), (-10, 20), [(1, 0.45, 0.85), (0.45, 0.8, 1)], (2.8, 3.4), (0.35, 0.5), (0.7, 0.6, 1), 'a coloured neon light'),
    'front': ((-25, 25), (10, 35), [(1, 1, 1)], (2.4, 3.0), (0.6, 0.8), (1, 1, 1), 'a soft front light'),
}
NEON_NAMES = {(1, 0.45, 0.85): 'magenta-pink neon light', (0.45, 0.8, 1): 'cyan neon light'}


def describe(fam, rel, pitch, key):
    words = FAMILIES[fam][6]
    if fam == 'neon':
        words = 'a strong ' + NEON_NAMES[tuple(key)]
    s, c = math.sin(math.radians(rel)), math.cos(math.radians(rel))
    side = 'right' if s > 0 else 'left'
    if pitch > 60:
        where = 'from directly above her'
    elif c < -0.6:
        where = f'from behind her (slightly to the {side} of the picture)'
    elif abs(s) > 0.5:
        where = f'from the {"upper " if pitch > 25 else ""}{side} side of the picture'
    else:
        where = 'from the front, near the camera'
    other = 'left' if side == 'right' else 'right'
    if c < -0.6:
        shade = 'her front mostly in one flat shadow tone, with bright rim highlights along her hair, shoulders and arms'
    elif pitch > 60:
        shade = 'shadows under her bangs, chin, the bow and the skirt, cast straight down'
    elif abs(s) > 0.5:
        shade = f'the {other} side of her face, body and hair in one flat shadow tone'
    else:
        shade = 'only small shadows at the edges of her form'
    return f'{words} {where}, {shade}'


ap = argparse.ArgumentParser()
ap.add_argument('--motions', default='coverage/live/motions.json')
ap.add_argument('--out', required=True)
ap.add_argument('--n', type=int, default=200)
ap.add_argument('--seed', type=int, default=1)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
m = json.load(open(a.motions))
clips = list(m['clips'])
out = []
fams = list(FAMILIES)
for i in range(a.n):
    fam = fams[i % len(fams)]
    yr, pr, keys, kr, ar, amb_col, _ = FAMILIES[fam]
    cam_yaw = float(rng.uniform(-110, 110) if rng.random() < 0.8 else rng.uniform(110, 250))
    cam_pitch = float(rng.uniform(-10, 30))
    rel = float(rng.uniform(*yr)) * (1 if rng.random() < 0.5 else -1)
    lp = float(rng.uniform(*pr))
    ly = math.radians(cam_yaw + rel)
    d = [math.sin(ly) * math.cos(math.radians(lp)), math.sin(math.radians(lp)), math.cos(ly) * math.cos(math.radians(lp))]
    key = list(keys[rng.integers(len(keys))])
    if rng.random() < 0.4:
        clip, t = 'stand', 0
    else:
        clip = clips[rng.integers(len(clips))]
        t = round(float(rng.uniform(0, (m['clips'][clip]['frames'] - 1) / m['fps'])), 2)
    out.append({'clip': clip, 't': t, 'yaw': round(cam_yaw % 360, 2), 'pitch': round(cam_pitch, 2), 'light': {
        'dir': [round(v * 3, 3) for v in d], 'color': key, 'intensity': round(float(rng.uniform(*kr)), 2),
        'amb': round(float(rng.uniform(*ar)), 2), 'ambColor': list(amb_col), 'family': fam,
        'desc': describe(fam, rel, lp, key)}})
json.dump(out, open(a.out, 'w'), indent=0)
print(len(out), 'views;', out[0]['light']['desc'], '|', out[2]['light']['desc'])
