#!/usr/bin/env python3
"""Line-art targets for the Tree-SSM shader's ink branch (shader_nts.c --lines), from the drawn film.

    python3 make_lines.py ../../../out/shader-data/w1 ../../../out/lines/w1 [--every 2]

Each frame keeps its G-buffer (linked) and gets an ink target: the film drawing's thin dark strokes,
found by a black top-hat (grey closing of the luminance over white, 3 x 3, minus the luminance): it
picks out lines a pixel or two wide and leaves the broad cel shadows alone. Ink ramps from 0 at a
top-hat of --lo to 1 at --hi, kept to the character and a pixel around her (her outer contour).

Only strokes on the geometry's edges count (within --reach pixels of one): her silhouette, a depth step
(one part in front of another), a crease (the normal turning by more than --crease degrees) or a seam
between parts. Lines drawn into her textures (hair strands, embroidery, eyes) are the colour shader's
job and stay out: inked again they would darken the textures, and the network can't place them from
the G-buffer, so it hedges with grey haze. The target PNG is (ink, ink, ink, 255): shader_nts.c reads
its red byte.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('src')
ap.add_argument('out')
ap.add_argument('--every', type=int, default=1)
ap.add_argument('--lo', type=float, default=0.05)
ap.add_argument('--hi', type=float, default=0.18)
ap.add_argument('--reach', type=int, default=1)
ap.add_argument('--depth-step', type=int, default=5, help='8-bit depth units')
ap.add_argument('--crease', type=float, default=60)
a = ap.parse_args()

load = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).astype(np.float32)).permute(2, 0, 1)[None]


def spread(x):   # max - min over each pixel's 3 x 3
    return F.max_pool2d(x, 3, 1, 1) + F.max_pool2d(-x, 3, 1, 1)


def edges_of(src, name):
    uvb, dep = load(src / f'{name}_uvb.png'), load(src / f'{name}_depth.png')[:, :1]
    n = load(src / f'{name}_normal.png') / 255 * 2 - 1
    n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-6)
    m = (uvb[:, 2:3] > 127).float()
    e = spread(m) > 0                                            # silhouette
    e |= (spread(uvb[:, 1:2] * m + (1 - m) * 99) > 0) & (m > 0)  # seam between parts
    e |= (spread(dep + (1 - m) * 999) > a.depth_step) & (m > 0)  # depth step
    # crease: the smallest cosine to a neighbour, from the max over shifts of -dot
    c = torch.stack([(n * torch.roll(n, (dy, dx), (2, 3))).sum(1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]).amin(0)[:, None]
    e |= (c < np.cos(np.radians(a.crease))) & (-F.max_pool2d(-m, 3, 1, 1) > 0)   # inside her
    return F.max_pool2d(e.float(), 2 * a.reach + 1, 1, a.reach)[0, 0] > 0


def ink_of(path):
    t = torch.from_numpy(np.asarray(Image.open(path).convert('RGBA')).astype(np.float32) / 255)
    al = t[..., 3]
    ow = t[..., :3] * al[..., None] + 1 - al[..., None]
    L = (ow @ torch.tensor([0.299, 0.587, 0.114]))[None, None]
    closing = -F.max_pool2d(-F.max_pool2d(L, 3, 1, 1), 3, 1, 1)
    near = F.max_pool2d(al[None, None], 3, 1, 1)[0, 0] > 0.5
    return (((closing - L)[0, 0] - a.lo) / (a.hi - a.lo)).clamp(0, 1) * near


src, out = Path(a.src), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
meta = json.loads((src / 'meta.json').read_text())
frames = meta['frames'][::a.every]
for f in frames:
    name = f'{f["frame"]:04d}'
    for k in ['color', 'normal', 'depth', 'uva', 'uvb']:
        dst = out / f'{name}_{k}.png'
        if not dst.exists():
            os.symlink((src / f'{name}_{k}.png').resolve(), dst)
    k = ((ink_of(src / f'{name}_target.png') * edges_of(src, name)).numpy() * 255 + 0.5).astype(np.uint8)
    Image.fromarray(np.stack([k, k, k, np.full_like(k, 255)], -1), 'RGBA').save(out / f'{name}_target.png')
(out / 'meta.json').write_text(json.dumps({**meta, 'frames': frames, 'ink': {'lo': a.lo, 'hi': a.hi, 'reach': a.reach, 'depthStep': a.depth_step, 'crease': a.crease}}))
print(f'{len(frames)} frames -> {out}')
