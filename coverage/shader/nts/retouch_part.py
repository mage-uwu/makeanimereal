#!/usr/bin/env python3
"""retouch_part.py -- repaint one region of a part where the film got it wrong, in a trained model.

The video model draws what it expects, not always what is there: the bunny's ears from behind are
white in the model (separate back faces, their own UV islands), but the film drew them pink, as seen
from the front, and the shader learned it. This fine-tunes only that part's neural texture (fine and
coarse levels), the network frozen, so that:
  - pixels on the region (the part's pixels with u in --u ranges) match the raw render's colour
    (--keep-shade: its shading kept, the film's flat lighting otherwise), alpha as the film's
  - every other pixel in the crop keeps the model's own output (no drift on the rest of her)

    python3 retouch_part.py --model f6.ntst --data out/c2/data/top out/c2/data/bottom \
        --part Accessory_RabbitEar_01_CLOTH --u 0:0.275 --u 0.725:1 --save f6e.ntst

Runs on the CPU in minutes: crops around the part in the frames that show the region.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from shader_torch import Net, forward, FINE, COARSE, FCH, DIL_DEFAULT

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True); ap.add_argument('--save', required=True)
ap.add_argument('--data', nargs='+', required=True, help='build_dataset.py folders')
ap.add_argument('--part', required=True)
ap.add_argument('--u', action='append', required=True, help='lo:hi, a u range of the region (repeat)')
ap.add_argument('--tree', default='4,4'); ap.add_argument('--blocks', type=int, default=8)
ap.add_argument('--min-px', type=int, default=40, help="frames showing at least this much of the region")
ap.add_argument('--frames', type=int, default=160); ap.add_argument('--pad', type=int, default=44)
ap.add_argument('--steps', type=int, default=600); ap.add_argument('--batch', type=int, default=4)
ap.add_argument('--lr', type=float, default=0.02)
a = ap.parse_args()
torch.manual_seed(0); rng = np.random.default_rng(0)

metas = [json.loads((Path(d) / 'meta.json').read_text()) for d in a.data]
names = metas[-1]['parts']               # the last folder's list (a superset: the bottom wave adds Shorts)
pid = names.index(a.part)
uranges = [tuple(map(float, r.split(':'))) for r in a.u]

# ---- the frames showing the region, a crop around the part in each ----
def load(stem):
    img = lambda s, c: np.asarray(Image.open(f'{stem}_{s}.png').convert(c))
    return np.concatenate([img('color', 'RGB'), img('normal', 'RGB'), img('depth', 'RGB')[..., :1],
                           img('uva', 'RGB'), img('uvb', 'RGB'), img('target', 'RGBA')], -1)
def region_of(px):
    part = (px[..., 11] == pid) & (px[..., 12] > 127)
    u = (px[..., 7].astype(np.float32) + px[..., 8] / 255) / 255
    reg = np.zeros_like(part)
    for lo, hi in uranges: reg |= part & (u >= lo) & (u < hi)
    return part, reg
cands = []
for d, meta in zip(a.data, metas):
    for f in meta['frames']:
        stem = f'{d}/{f["frame"]:04d}'
        uvb = np.asarray(Image.open(f'{stem}_uvb.png').convert('RGB')); uva = np.asarray(Image.open(f'{stem}_uva.png').convert('RGB'))
        px = np.zeros(uvb.shape[:2] + (17,), np.uint8); px[..., 7:10] = uva; px[..., 10:13] = uvb
        part, reg = region_of(px)
        if reg.sum() >= a.min_px: cands.append((stem, f['pitch'], int(reg.sum())))
print(f'{len(cands)} frames show the region', flush=True)
pick = [cands[i] for i in sorted(rng.choice(len(cands), min(a.frames, len(cands)), replace=False))]

crops = []
for stem, pitch, _ in pick:
    px = load(stem)
    part, reg = region_of(px)
    ys, xs = np.nonzero(part)
    H, W = part.shape
    y0, y1 = max(0, ys.min() - a.pad), min(H, ys.max() + 1 + a.pad)
    x0, x1 = max(0, xs.min() - a.pad), min(W, xs.max() + 1 + a.pad)
    c = px[y0:y1, x0:x1].copy()
    # the region's target: the render's colour, the film's alpha
    r = reg[y0:y1, x0:x1]
    c[..., 13:16][r] = c[..., 0:3][r]
    crops.append((torch.from_numpy(c)[None], torch.tensor([pitch], dtype=torch.float32), torch.from_numpy(r)[None]))

# ---- the model; the part's texels the only thing that moves ----
T, V = map(int, a.tree.split(','))
net = Net(a.blocks, len(names), DIL_DEFAULT[:a.blocks], T, V)
net.load(a.model)
P = net.P
fa, fn = pid * FINE * FINE * FCH, FINE * FINE * FCH
ca, cn = P * FINE * FINE * FCH + pid * COARSE * COARSE * FCH, COARSE * COARSE * FCH
base = net.tex.clone()
fine = torch.nn.Parameter(base[fa:fa + fn].clone()); coarse = torch.nn.Parameter(base[ca:ca + cn].clone())
def splice():
    net.tex = torch.cat([base[:fa], fine, base[fa + fn:ca], coarse, base[ca + cn:]])
with torch.no_grad():   # the model's own output: what everything off the region keeps
    keep = [forward(net, px, pitch, trees=True)[0] for px, pitch, _ in crops]

opt = torch.optim.Adam([fine, coarse], lr=a.lr)
def loss_of(k):
    px, pitch, r = crops[k]
    p, _ = forward(net, px, pitch, trees=True)
    q = px[..., 13:17].float() / 255
    r = r[..., None].float()
    on = (((p[..., :3] - q[..., :3]).abs() * q[..., 3:4]).sum(-1, keepdim=True) / 3 + (p[..., 3:4] - q[..., 3:4]).abs()) * r
    off = (p - keep[k]).abs().mean(-1, keepdim=True) * (1 - r)
    return on.sum() / r.sum().clamp(min=1), off.mean()
t0 = time.time()
for it in range(a.steps):
    splice()
    opt.zero_grad()
    tot_on = tot_off = 0
    for k in rng.choice(len(crops), a.batch, replace=False):
        on, off = loss_of(k)
        (on + 4 * off).backward()
        tot_on += on.item() / a.batch; tot_off += off.item() / a.batch
    opt.step()
    if it % 50 == 0 or it == a.steps - 1:
        print(f'it {it}  region L1 {tot_on:.4f}  elsewhere drift {tot_off:.5f}  {time.time() - t0:.0f}s', flush=True)
with torch.no_grad():
    splice(); net.tex = net.tex.detach().clone()
net.save(a.save)
print('saved', a.save)
