#!/usr/bin/env python3
"""make_style_data.py -- style_torch.py's data: the G-buffers of the keyframes' views, half size,
and each keyframe as the target: an RGBA image (a film frame, build_dataset.py's target) with its
own alpha, or a drawing on green, its alpha keyed from its own background (the drawing's silhouette,
not the render's).

    python3 make_style_data.py keys.json --out style.npz [--half]
keys.json: [{"dir": "w1_hd", "frame": 240, "pitch": 10.0, "key": "keys/w1_hd_0240.png", "held": false}, ...]
(dir: a build_dataset.py folder: <frame>_{color,normal,depth,uva,uvb}.png)
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def box2(a):   # (H, W, C) -> half size, the mean of each 2 x 2
    a = a.astype(np.float32)
    return (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]) / 4


def keyed(img):
    """A drawing on green -> straight RGBA: alpha from how green each pixel is, the green spill
    taken out of the edge."""
    c = img.astype(np.float32)
    k = c[..., 1] - np.maximum(c[..., 0], c[..., 2])
    alpha = np.clip((96 - k) / 64, 0, 1)
    c[..., 1] = np.where(alpha < 1, np.minimum(c[..., 1], np.maximum(c[..., 0], c[..., 2])), c[..., 1])
    return np.concatenate([c, alpha[..., None] * 255], -1)


def build(keys, full=False):
    """keys (as keys.json) -> the style_torch.py data: px (N, H, W, 17) uint8, pitch, held, parts."""
    parts = len(json.loads((Path(keys[0]['dir']) / 'meta.json').read_text())['parts'])
    pxs, pitch, held = [], [], []
    for k in keys:
        stem = f"{k['dir']}/{k['frame']:04d}"
        img = lambda s: np.asarray(Image.open(f'{stem}_{s}.png').convert('RGB'))
        col, nrm, dep, uva, uvb = (img(s) for s in ('color', 'normal', 'depth', 'uva', 'uvb'))
        ki = Image.open(k['key'])
        if ki.mode == 'RGBA':   # a film frame (build_dataset.py's target): its own alpha
            tgt = np.asarray(ki.resize((col.shape[1], col.shape[0]), Image.LANCZOS)).astype(np.float32)
        else:                   # a drawing on green
            tgt = keyed(np.asarray(ki.convert('RGB').resize((col.shape[1], col.shape[0]), Image.LANCZOS)))
        if not full:   # smooth buffers by area; the encoded ones (uv, part, coverage) by their top-left texel
            col, nrm, dep = box2(col), box2(nrm), box2(dep)
            uva, uvb = uva[::2, ::2], uvb[::2, ::2]
            pa = box2(np.concatenate([tgt[..., :3] * tgt[..., 3:4] / 255, tgt[..., 3:4]], -1))   # premultiplied
            tgt = np.concatenate([pa[..., :3] * 255 / np.maximum(pa[..., 3:4], 1e-3), pa[..., 3:4]], -1)
        px = np.concatenate([col, nrm, dep[..., :1], uva, uvb, tgt], -1)
        pxs.append(np.clip(np.round(px), 0, 255).astype(np.uint8)); pitch.append(k['pitch']); held.append(k['held'])
    return dict(px=np.stack(pxs), pitch=np.array(pitch, np.float32), held=np.array(held), parts=parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('keys'); ap.add_argument('--out', required=True)
    ap.add_argument('--full', action='store_true', help='keep the full size')
    a = ap.parse_args()
    d = build(json.loads(Path(a.keys).read_text()), a.full)
    np.savez(a.out, **d)
    print(f"{len(d['px'])} frames {d['px'].shape[2]}x{d['px'].shape[1]}, {d['held'].sum()} held out -> {a.out}")

if __name__ == '__main__':
    main()
