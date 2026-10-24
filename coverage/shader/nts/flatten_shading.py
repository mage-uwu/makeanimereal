#!/usr/bin/env python3
"""Flatten the film's lighting in a dataset's targets (build_dataset.py folders), before training.

The film's frames weren't drawn under one consistent light: each chunk and anchor shades her a little
differently, and the shader, fitting all of them, averages those shadows into soft splotches. The raw
render (each frame's colour tile) is lit the same way in every frame, so a target's shading can be
measured against it: the ratio of the drawing's luminance to the render's, smoothed within each part
(robustly, so ink and fine detail don't count as shading), is the frame's light. Dividing most of it
out (--amount, of the deviation from the part's median) leaves the drawing's palette, lines and
detail with the render's even light.

    python3 flatten_shading.py data/top data/bottom --out-suffix _flat [--amount .8] [--radius 12]

Writes <dir><suffix>/ with the same files, the targets flattened (the G-buffer copied as is).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def lum(rgb):
    return rgb[..., 0] * .299 + rgb[..., 1] * .587 + rgb[..., 2] * .114


def flatten(color, target, uvb, amount, radius):
    """color (h, w, 3) render, target (h, w, 4) drawing, uvb (h, w, 3): -> flattened target (h, w, 4)."""
    c = torch.from_numpy(color).float() / 255
    t = torch.from_numpy(target).float() / 255
    part = torch.from_numpy(uvb[..., 1].astype(np.int64))
    a = t[..., 3]
    lc, lt = lum(c).clamp_min(.03), lum(t[..., :3])
    ratio = (lt / lc).clamp(0, 4)
    # what counts as shading: covered, not ink (much darker than its neighbourhood), not a silhouette edge
    local = F.avg_pool2d(lt[None, None], 7, 1, 3, count_include_pad=False)[0, 0]
    ink = (local - lt) > .06
    w = ((a > .95) & ~ink).float()
    out = t.clone()
    k = 2 * radius + 1
    for p in part[a > .5].unique():
        m = (part == p).float() * w
        if m.sum() < 50: continue
        num = F.avg_pool2d((ratio * m)[None, None], k, 1, radius, count_include_pad=False)[0, 0]
        den = F.avg_pool2d(m[None, None], k, 1, radius, count_include_pad=False)[0, 0]
        light = num / den.clamp_min(1e-4)
        med = ratio[m > 0].median()
        corr = (med / light.clamp_min(.05)) ** amount
        sel = (part == p) & (a > .5) & (den > .05)
        out[..., :3][sel] = (t[..., :3][sel] * corr[sel][:, None]).clamp(0, 1)
    return (out * 255 + .5).byte().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--out-suffix', default='_flat')
    ap.add_argument('--amount', type=float, default=.8)
    ap.add_argument('--radius', type=int, default=12)
    ap.add_argument('--only', help='frames to do, e.g. 120,600 (a preview)')
    a = ap.parse_args()
    for d in map(Path, a.dirs):
        out = Path(str(d) + a.out_suffix); out.mkdir(parents=True, exist_ok=True)
        meta = json.loads((d / 'meta.json').read_text())
        frames = [f['frame'] for f in meta['frames']]
        if a.only: frames = [f for f in frames if f in {int(x) for x in a.only.split(',')}]
        for i, f in enumerate(frames):
            n = f'{f:04d}'
            img = lambda k, m: np.asarray(Image.open(d / f'{n}_{k}.png').convert(m))
            Image.fromarray(flatten(img('color', 'RGB'), img('target', 'RGBA'), img('uvb', 'RGB'), a.amount, a.radius), 'RGBA').save(out / f'{n}_target.png')
            for k in ('color', 'normal', 'depth', 'uva', 'uvb'):
                if not (out / f'{n}_{k}.png').exists(): shutil.copy(d / f'{n}_{k}.png', out / f'{n}_{k}.png')
            if i % 200 == 0: print(d.name, i, '/', len(frames), flush=True)
        (out / 'meta.json').write_text(json.dumps(meta))


if __name__ == '__main__':
    main()
