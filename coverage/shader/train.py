#!/usr/bin/env python3
"""Train the learned render shader: G-buffer of the base model in, the film's anime drawing out.

    python3 train.py --data data/w1 [--data data/bottom] --out runs/v1 [--iters 12000] [--arch hybrid|mlp]

Model (see coverage/SHADER_PLAN.md):
  * neural texture: a learned feature grid per part, in the part's own UV space, at two
    resolutions (fine detail plus a smooth base); it sits on the surface, so it follows the rig.
  * decoder: 'hybrid' is a small U-Net over the G-buffer plus the sampled features (it draws lines
    and silhouettes); 'mlp' is the per-pixel baseline (1x1 convolutions only).
  * per-frame appearance code (FiLM at the decoder's bottleneck), trained with a pull to zero; at
    run time the code is zero, so the shader keeps the style the frames agree on.
Every 10th frame is held out. Writes previews (held-out frames: render | ours | target) and
checkpoints into --out, and a log of the held-out L1.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

dev = 'cuda'


def load(folder):
    meta = json.loads((Path(folder) / 'meta.json').read_text())
    xs, ys, pitch = [], [], []
    for f in meta['frames']:
        name = Path(folder) / f'{f["frame"]:04d}'
        img = lambda k: np.asarray(Image.open(f'{name}_{k}.png'))
        color, normal, depth, uva, uvb = img('color'), img('normal'), img('depth'), img('uva').astype(np.float32), img('uvb').astype(np.float32)
        u = (uva[..., 0] + uva[..., 1] / 255) / 255
        v = (uva[..., 2] + uvb[..., 0] / 255) / 255
        part = uvb[..., 1]
        mask = (uvb[..., 2] > 127).astype(np.float32)
        x = np.concatenate([color / 255.0, normal / 255.0 * 2 - 1, depth[..., :1] / 255.0, mask[..., None],
                            u[..., None], v[..., None], part[..., None]], -1).astype(np.float32)
        xs.append(x.astype(np.float16))  # half storage: 1,440 frames fit in GPU memory
        ys.append((img('target').astype(np.float32) / 255.0).astype(np.float16))
        pitch.append(f['pitch'])
    return meta, np.stack(xs), np.stack(ys), np.array(pitch, np.float32)


class NeuralTexture(nn.Module):
    """Per-part feature grids, sampled at each pixel's (u, v) in its part's slice."""

    def __init__(self, parts, ch=8, fine=512, coarse=64):
        super().__init__()
        self.parts = parts
        self.fine = nn.Parameter(torch.zeros(1, ch, parts, fine, fine))
        self.coarse = nn.Parameter(torch.zeros(1, ch, parts, coarse, coarse))
        nn.init.normal_(self.fine, std=0.01)
        nn.init.normal_(self.coarse, std=0.1)

    def forward(self, u, v, part):  # each [B, H, W]
        z = (2 * part + 1) / self.parts - 1  # slice centres: no blending between parts
        grid = torch.stack([u * 2 - 1, v * 2 - 1, z], -1)[:, None]  # [B, 1, H, W, 3]
        f = lambda t: F.grid_sample(t.expand(u.shape[0], -1, -1, -1, -1), grid, align_corners=False)[:, :, 0]
        return f(self.fine) + f(self.coarse)


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GELU(), nn.Conv2d(o, o, 3, padding=1), nn.GELU())


class Decoder(nn.Module):
    def __init__(self, cin, code, arch):
        super().__init__()
        self.arch = arch
        if arch == 'mlp':
            self.net = nn.Sequential(nn.Conv2d(cin, 64, 1), nn.GELU(), nn.Conv2d(64, 64, 1), nn.GELU(), nn.Conv2d(64, 4, 1))
            self.film = nn.Linear(code, 128)
            return
        c = [32, 48, 64, 96]
        self.down = nn.ModuleList([block(cin, c[0]), block(c[0], c[1]), block(c[1], c[2]), block(c[2], c[3])])
        self.up = nn.ModuleList([block(c[3] + c[2], c[2]), block(c[2] + c[1], c[1]), block(c[1] + c[0], c[0])])
        self.film = nn.Linear(code, 2 * c[3])
        self.head = nn.Conv2d(c[0] + cin, 4, 1)

    def forward(self, x, code):
        if self.arch == 'mlp':
            h = self.net[1](self.net[0](x))
            g, b = self.film(code).chunk(2, -1)
            h = h * (1 + g[..., None, None]) + b[..., None, None]
            return self.net[4](self.net[3](self.net[2](h)))
        skips, h = [], x
        for k, d in enumerate(self.down):
            h = d(h if k == 0 else F.avg_pool2d(h, 2))
            skips.append(h)
        g, b = self.film(code).chunk(2, -1)
        h = h * (1 + g[..., None, None]) + b[..., None, None]
        for u, s in zip(self.up, reversed(skips[:-1])):
            h = u(torch.cat([F.interpolate(h, size=s.shape[-2:], mode='bilinear', align_corners=False), s], 1))
        return self.head(torch.cat([h, x], 1))


class Shader(nn.Module):
    def __init__(self, parts, frames, arch, code=16):
        super().__init__()
        self.tex = NeuralTexture(parts)
        self.codes = nn.Embedding(frames, code)
        nn.init.zeros_(self.codes.weight)
        self.dec = Decoder(3 + 3 + 1 + 1 + 8 + 2, code, arch)

    def forward(self, x, pitch, idx=None):
        # x: [B, H, W, 11] = color 3, normal 3, depth 1, mask 1, u, v, part
        color, normal, depth, mask = x[..., :3], x[..., 3:6], x[..., 6:7], x[..., 7:8]
        u, v, part = x[..., 8], x[..., 9], x[..., 10]
        feats = self.tex(u, v, part) * mask.permute(0, 3, 1, 2)
        p = torch.deg2rad(pitch)[:, None, None, None].expand(-1, 1, *x.shape[1:3])
        inp = torch.cat([color.permute(0, 3, 1, 2), normal.permute(0, 3, 1, 2), depth.permute(0, 3, 1, 2),
                         mask.permute(0, 3, 1, 2), feats, torch.sin(p), torch.cos(p)], 1)
        code = self.codes(idx) if idx is not None else torch.zeros(x.shape[0], self.codes.embedding_dim, device=x.device)
        out = self.dec(inp, code)
        return torch.sigmoid(out)  # [B, 4, H, W], straight RGBA


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision
        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for q in vgg.parameters():
            q.requires_grad = False
        self.vgg = vgg
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, a, b):
        fa, fb = self.vgg((a - self.mean) / self.std), self.vgg((b - self.mean) / self.std)
        return F.l1_loss(fa, fb)


def over_white(rgba):
    return rgba[:, :3] * rgba[:, 3:4] + (1 - rgba[:, 3:4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', action='append', required=True)
    ap.add_argument('--out', default='runs/v1')
    ap.add_argument('--arch', default='hybrid', choices=['hybrid', 'mlp'])
    ap.add_argument('--iters', type=int, default=12000)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--crop', type=int, default=320)
    ap.add_argument('--vgg', type=float, default=0.1)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    metas, X, Y, P = [], [], [], []
    for d in a.data:
        m, x, y, p = load(d)
        metas.append(m), X.append(x), Y.append(y), P.append(p)
    parts = metas[0]['parts']
    X, Y, P = np.concatenate(X), np.concatenate(Y), np.concatenate(P)
    n = len(X)
    held = np.arange(n) % 10 == 5
    train_ids, test_ids = np.nonzero(~held)[0], np.nonzero(held)[0]
    print(f'{n} frames ({len(train_ids)} train, {len(test_ids)} held out), {len(parts)} parts, {X.shape[1:3]}', flush=True)
    Xg, Yg, Pg = torch.from_numpy(X).to(dev), torch.from_numpy(Y).to(dev), torch.from_numpy(P).to(dev)

    model = Shader(len(parts), n, a.arch).to(dev)
    vgg = VGG().to(dev) if a.vgg else None
    opt = torch.optim.Adam([
        {'params': model.tex.parameters(), 'lr': 1e-2},
        {'params': model.codes.parameters(), 'lr': 1e-3},
        {'params': model.dec.parameters(), 'lr': 1e-3},
    ])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1, t / 300) * 0.5 * (1 + math.cos(math.pi * t / a.iters)))
    print(f'decoder {sum(q.numel() for q in model.dec.parameters()) / 1e6:.2f} M params', flush=True)
    H, W = X.shape[1:3]
    log = open(out / 'log.txt', 'a')
    t0 = time.time()
    for it in range(1, a.iters + 1):
        idx = torch.from_numpy(np.random.choice(train_ids, a.batch)).to(dev)
        cy, cx = np.random.randint(0, H - a.crop + 1), np.random.randint(0, W - min(a.crop, W) + 1)
        cw = min(a.crop, W)
        x, y = Xg[idx, cy:cy + a.crop, cx:cx + cw].float(), Yg[idx, cy:cy + a.crop, cx:cx + cw].permute(0, 3, 1, 2).float()
        pred = model(x, Pg[idx], idx)
        loss = F.l1_loss(pred[:, 3], y[:, 3]) + F.l1_loss(pred[:, :3] * y[:, 3:4], y[:, :3] * y[:, 3:4])
        if vgg:
            loss = loss + a.vgg * vgg(over_white(pred), over_white(y))
        loss = loss + 1e-3 * model.codes(idx).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % 500 == 0 or it == a.iters:
            model.eval()
            with torch.no_grad():
                errs = []
                for k in range(0, len(test_ids), 8):
                    ids = torch.from_numpy(test_ids[k:k + 8]).to(dev)
                    pr = model(Xg[ids].float(), Pg[ids])  # code zero: the run-time shader
                    errs.append(F.l1_loss(over_white(pr), over_white(Yg[ids].permute(0, 3, 1, 2).float()), reduction='none').mean((1, 2, 3)))
                err = torch.cat(errs).mean().item()
                line = f'it {it}  loss {loss.item():.4f}  held-out L1 {err:.4f}  {time.time() - t0:.0f}s'
                print(line, flush=True)
                log.write(line + '\n')
                log.flush()
                if it % 2000 == 0 or it == a.iters:
                    pick = test_ids[np.linspace(0, len(test_ids) - 1, 6).astype(int)]
                    ids = torch.from_numpy(pick).to(dev)
                    pr = over_white(model(Xg[ids].float(), Pg[ids]))
                    tg = over_white(Yg[ids].permute(0, 3, 1, 2).float())
                    rd = Xg[ids][..., :3].permute(0, 3, 1, 2).float()
                    rows = torch.cat([torch.cat([r, q, t], 2) for r, q, t in zip(rd, pr, tg)], 2)
                    Image.fromarray((rows.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(out / f'preview_{it:05d}.png')
            model.train()
    torch.save({'model': model.state_dict(), 'parts': parts, 'arch': a.arch}, out / 'shader.pt')
    print('saved', out / 'shader.pt', flush=True)


if __name__ == '__main__':
    main()
