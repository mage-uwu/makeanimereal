#!/usr/bin/env python3
"""Train the learned render: a shader that turns her mesh's G-buffer into the key drawings' art.

    python3 scripts/learned-train.py [--renders out/g2-renders] [--web out/web2] [--cache out/cache2]
                                     [--data out/learned] [--steps 3000] [--holdout 12]

Inputs per pixel (web/learned.js draws the same G-buffer live): base colour, coverage, view-space
normal, depth, and 8 learned features sampled from a 3D grid around her at the surface point. The
grid is the "neural texture": it learns what the drawings paint at each spot on her body, whatever
the angle. A small U-Net turns that into straight RGBA in the render frame, where it can learn
screen-space things a texture can't: outlines, rim shading, hair drawn past the mesh.

Targets are the key drawings moved onto their renders by each drawing's fit (from build-web.py),
so every training pair is pixel-aligned. A few keys are held out to measure how well the render
does at angles it never saw. Writes net.json + net.png (float16 weights, TensorFlow.js layout) and
grid.json + grid.png (uint8) into <web>/learned, and a comparison sheet of the held-out keys.
"""
import argparse
import gzip
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

W, H = 400, 600           # GBUFFER in web/learned.js
GRID_Y = 128              # grid cells along her height; the other axes keep the cells cubic
FEATURES = 8
CHANNELS = [16, 32, 64]   # per U-Net level; the bottleneck has MID
MID = 128


class Net(nn.Module):
    """U-Net mirrored by loadNet() in web/learned.js: two 3x3 convs per level down, average-pool,
    nearest upsample, skip concat, two convs per level up, 3x3 conv to RGBA, sigmoid."""

    def __init__(self, cin=8 + FEATURES):
        super().__init__()
        def block(a, b):
            return nn.ModuleList([nn.Conv2d(a, b, 3, padding=1), nn.Conv2d(b, b, 3, padding=1)])
        self.down = nn.ModuleList()
        c = cin
        for ch in CHANNELS:
            self.down.append(block(c, ch))
            c = ch
        self.mid = block(c, MID)
        self.up = nn.ModuleList()
        c = MID
        for ch in reversed(CHANNELS):
            self.up.insert(0, block(c + ch, ch))
            c = ch
        self.out = nn.Conv2d(c, 4, 3, padding=1)

    @staticmethod
    def run(block, x):
        for conv in block:
            x = F.leaky_relu(conv(x), 0.1)
        return x

    def forward(self, x):
        skips = []
        for b in self.down:
            x = self.run(b, x)
            skips.append(x)
            x = F.avg_pool2d(x, 2)
        x = self.run(self.mid, x)
        for b, s in zip(reversed(self.up), reversed(skips)):
            x = F.interpolate(x, size=s.shape[-2:], mode='nearest')
            x = self.run(b, torch.cat([x, s], 1))
        return torch.sigmoid(self.out(x))


def save_bytes(path, data):
    """Raw bytes as a lossless greyscale PNG, 1024 per row (fetchBytes() in web/learned.js)."""
    a = np.frombuffer(data, np.uint8)
    a = np.pad(a, (0, -len(a) % 1024)).reshape(-1, 1024)
    Image.fromarray(a, 'L').save(path, optimize=True)


def load_gbuffer(path):
    g = np.frombuffer(gzip.open(path).read(), np.uint16).reshape(H, W, 12).astype(np.float32) / 65535
    g[..., 4:8] = g[..., 4:8] * 2 - 1  # normal, depth
    return g


def aligned_target(stem, key, cache, size):
    """The key drawing moved onto its render: target(x) = placed drawing at x + fit(x)."""
    cut = np.asarray(Image.open(cache / f'{stem}.png').convert('RGBA').resize(size, Image.LANCZOS)).astype(np.float32) / 255
    w, h = size
    m = np.float32([[key['s'], 0, key['ox'] * w], [0, key['s'], key['oy'] * h]])
    placed = cv2.warpAffine(cut, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    fit = np.load(cache / f'{stem}.fit.npy')
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pre = placed.copy()
    pre[..., :3] *= pre[..., 3:]  # resample premultiplied
    t = cv2.remap(pre, gx + fit[..., 0], gy + fit[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return cv2.resize(t, (W, H), interpolation=cv2.INTER_AREA)  # premultiplied RGBA


def features(grid, g):
    """Sample the grid at each pixel's surface point (B,H,W,3 in 0..1), zero off the mesh, exactly
    as the viewer's linear-filtered 3D texture does (texel centres, clamp to edge)."""
    pos = g[..., 8:11] * 2 - 1
    f = F.grid_sample(grid, pos.reshape(1, -1, 1, 1, 3), mode='bilinear', padding_mode='border', align_corners=False)
    f = f.reshape(FEATURES, *pos.shape[:3]).permute(1, 0, 2, 3)
    return f * g[..., 3].unsqueeze(1)


def net_input(grid, g):
    return torch.cat([g[..., :8].permute(0, 3, 1, 2), features(grid, g)], 1)


def loss_fn(pred, tgt):
    pa = pred[:, 3:]
    pp = torch.cat([pred[:, :3] * pa, pa], 1)  # premultiplied
    l1 = (pp - tgt).abs().mean()
    dx = lambda x: x[..., :, 1:] - x[..., :, :-1]
    dy = lambda x: x[..., 1:, :] - x[..., :-1, :]
    edges = (dx(pp) - dx(tgt)).abs().mean() + (dy(pp) - dy(tgt)).abs().mean()
    return l1 + 0.5 * edges


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--renders', default='out/g2-renders')
    p.add_argument('--web', default='out/web2')
    p.add_argument('--cache', default='out/cache2')
    p.add_argument('--data', default='out/learned')
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch', type=int, default=12)
    p.add_argument('--crop', type=int, default=192)
    p.add_argument('--holdout', type=int, default=12, help='keys kept out of training, to measure unseen angles')
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    renders, web, cache, data = Path(args.renders), Path(args.web), Path(args.cache), Path(args.data)
    manifest = json.loads((renders / 'manifest.json').read_text())
    size = (600, round(600 * manifest['height'] / manifest['width']))
    keys = json.loads((web / 'keys.json').read_text())['keys']
    box = json.loads((data / 'box.json').read_text())

    # Held out: spread around the dense band, never on the 15-degree grid the old keys sat on.
    dense = sorted((k for k in keys if k['yaw'] % 15 and -15 <= k['pitch'] <= 45), key=lambda k: (k['yaw'], k['pitch']))
    held = {Path(k['file']).stem for k in dense[::max(1, len(dense) // args.holdout)][:args.holdout]}
    stems, G, T = [], [], []
    for k in keys:
        stem = Path(k['file']).stem
        stems.append(stem)
        G.append(load_gbuffer(data / 'gbuffer' / f'{stem}.u16.gz').astype(np.float16))
        T.append(aligned_target(stem, k, cache, size).astype(np.float16))
    G, T = np.stack(G), np.stack(T)
    train = [i for i, s in enumerate(stems) if s not in held]
    test = [i for i, s in enumerate(stems) if s in held]
    print(f'{len(train)} training keys, {len(test)} held out: {", ".join(sorted(held))}')

    nx = round(GRID_Y * box['size'][0] / box['size'][1])
    nz = round(GRID_Y * box['size'][2] / box['size'][1])
    grid = nn.Parameter(torch.zeros(1, FEATURES, nz, GRID_Y, nx))
    net = Net()
    opt = torch.optim.Adam([{'params': net.parameters(), 'lr': 2e-3}, {'params': [grid], 'lr': 1e-2}])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[2e-3, 1e-2], total_steps=args.steps, pct_start=0.05)

    # Crops centred where there is art (the drawing or the mesh), so steps aren't spent on background.
    fg = [np.argwhere((T[i][..., 3] > 0.05) | (G[i][..., 3] > 0.5)) for i in range(len(stems))]
    c = args.crop

    def batch(idx):
        gs, ts = [], []
        for i in idx:
            y, x = fg[i][rng.integers(len(fg[i]))]
            y0 = int(np.clip(y - c // 2, 0, H - c)); x0 = int(np.clip(x - c // 2, 0, W - c))
            gs.append(G[i, y0:y0 + c, x0:x0 + c]); ts.append(T[i, y0:y0 + c, x0:x0 + c])
        return torch.from_numpy(np.stack(gs).astype(np.float32)), torch.from_numpy(np.stack(ts).astype(np.float32)).permute(0, 3, 1, 2)

    def evaluate(idx):
        net.eval()
        with torch.no_grad():
            out = []
            for i in idx:
                g = torch.from_numpy(G[i:i + 1].astype(np.float32))
                out.append(net(net_input(grid, g))[0].permute(1, 2, 0).numpy())
        net.train()
        def err(pred, i):
            t = T[i].astype(np.float32)
            pp = np.concatenate([pred[..., :3] * pred[..., 3:], pred[..., 3:]], -1)
            region = (t[..., 3] > 0.05) | (pred[..., 3] > 0.05)
            return float(np.abs(pp - t)[region].mean())
        return out, float(np.mean([err(o, i) for o, i in zip(out, idx)]))

    t0 = time.time()
    for step in range(1, args.steps + 1):
        g, t = batch(rng.choice(train, args.batch))
        pred = net(net_input(grid, g))
        tv = sum(((grid.narrow(d, 1, grid.shape[d] - 1) - grid.narrow(d, 0, grid.shape[d] - 1)) ** 2).mean() for d in (2, 3, 4))
        loss = loss_fn(pred, t) + 1e-3 * tv
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0 or step == args.steps:
            msg = f'step {step}/{args.steps}  loss {loss.item():.4f}  {time.time() - t0:.0f}s'
            if step % 500 == 0 or step == args.steps:
                msg += f'  held-out error {evaluate(test)[1]:.4f}  training-key error {evaluate(train[::25])[1]:.4f}'
            print(msg, flush=True)

    # Export, in the layouts web/learned.js reads.
    dest = web / 'learned'
    dest.mkdir(exist_ok=True)
    tensors, chunks, offset = {}, [], 0
    for name, mod in [*(((f'down{i}.{j}', b[j]) for i, b in enumerate(net.down) for j in range(2))),
                      *(((f'mid.{j}', net.mid[j]) for j in range(2))),
                      *(((f'up{i}.{j}', b[j]) for i, b in enumerate(net.up) for j in range(2))), ('out', net.out)]:
        for part, arr in (('w', mod.weight.detach().permute(2, 3, 1, 0)), ('b', mod.bias.detach())):
            a = arr.contiguous().numpy().astype(np.float16)
            tensors[f'{name}.{part}'] = {'shape': list(a.shape), 'offset': offset}
            chunks.append(a.ravel()); offset += a.size
    net_bytes = np.concatenate(chunks).tobytes()
    save_bytes(dest / 'net.png', net_bytes)
    (dest / 'net.json').write_text(json.dumps({'levels': len(CHANNELS), 'bytes': len(net_bytes), 'tensors': tensors}))
    gr = grid.detach()[0].numpy()  # C, z, y, x
    lo = gr.reshape(FEATURES, -1).min(1); hi = gr.reshape(FEATURES, -1).max(1)
    span = np.maximum(hi - lo, 1e-6)
    q = np.round((gr - lo[:, None, None, None]) / span[:, None, None, None] * 255).astype(np.uint8)
    save_bytes(dest / 'grid.png', q.tobytes())
    (dest / 'grid.json').write_text(json.dumps({'dims': [nx, GRID_Y, nz], 'bytes': q.size, 'lo': lo.tolist(), 'span': span.tolist(), 'box': box}))
    torch.save({'net': net.state_dict(), 'grid': grid.detach()}, data / 'model.pt')

    # Held-out keys: G-buffer base colour | learned render | the drawing (aligned), on grey.
    out, e = evaluate(test)
    rows = []
    for o, i in zip(out, test):
        t = T[i].astype(np.float32)
        on = lambda rgb, a: rgb * a + 0.5 * (1 - a)
        rows.append(np.concatenate([on(G[i][..., :3].astype(np.float32), G[i][..., 3:4].astype(np.float32)),
                                    on(o[..., :3], o[..., 3:]), t[..., :3] + 0.5 * (1 - t[..., 3:])], 1))
    sheet = np.concatenate([np.concatenate(rows[j:j + 2], 1) for j in range(0, len(rows) - 1, 2)], 0)
    Image.fromarray((np.clip(sheet, 0, 1) * 255).astype(np.uint8)).save(data / 'heldout.png')
    print(f'held-out error {e:.4f}; wrote {dest}/net.* and grid.*, sheet {data}/heldout.png')


if __name__ == '__main__':
    main()
