#!/usr/bin/env python3
"""Train the sharp shader: the fast decoder at half resolution plus a light full-resolution head.

    python3 train_sr.py --data data/w1_hd --data data/bottom_hd --init runs/slim_v1/shader.pt --out runs/sharp_v1
        [--iters 6000] [--sr 16]

The datasets hold full-resolution (720 x 1280) G-buffers and film frames (build_dataset.py --size
720x1280). The half-resolution input is made from the full one exactly as the browser makes it: 2 x 2
averages for colour, normal and depth, the top-left texel of each 2 x 2 for UV, part and coverage.
The decoder (from --init) runs at half resolution as before; its last features are upsampled and,
with the full-resolution input (where the neural texture is sampled at full detail), go through a
3 x 3 convolution and a 1 x 1 one to the drawing at full resolution. Everything trains together.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import train

dev = 'cuda'


def inputs(model, x, pitch):
    """x [B, H, W, 11] -> the network's 18-channel input [B, 18, H, W] (features sampled here)."""
    color, normal, depth, mask = x[..., :3], x[..., 3:6], x[..., 6:7], x[..., 7:8]
    feats = model.tex(x[..., 8], x[..., 9], x[..., 10]) * mask.permute(0, 3, 1, 2)
    p = torch.deg2rad(pitch)[:, None, None, None].expand(-1, 1, *x.shape[1:3])
    return torch.cat([color.permute(0, 3, 1, 2), normal.permute(0, 3, 1, 2), depth.permute(0, 3, 1, 2),
                      mask.permute(0, 3, 1, 2), feats, torch.sin(p), torch.cos(p)], 1)


def half(x):
    """Full-resolution G-buffer [B, H, W, 11] -> half resolution, as the browser does it."""
    cont = F.avg_pool2d(x[..., :7].permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
    return torch.cat([cont, x[:, 0::2, 0::2, 7:]], -1)


def features(dec, inp, code):
    """The decoder up to its last up-block (what its 1 x 1 head would read)."""
    skips, h = [], inp
    for k, d in enumerate(dec.down):
        h = d(h if k == 0 else F.avg_pool2d(h, 2))
        skips.append(h)
    g, b = dec.film(code).chunk(2, -1)
    h = h * (1 + g[..., None, None]) + b[..., None, None]
    for u, s in zip(dec.up, reversed(skips[:-1])):
        h = u(torch.cat([F.interpolate(h, size=s.shape[-2:], mode='bilinear', align_corners=False), s], 1))
    return h


class Critic(nn.Module):
    """PatchGAN critic: is this patch of drawing (over white, with alpha) a real film frame, given the
    G-buffer's colour and normals there? Conditioning on the G-buffer makes it judge style, not layout."""

    def __init__(self, ch=64):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.net = nn.Sequential(
            sn(nn.Conv2d(4 + 6, ch, 4, 2, 1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(ch, ch * 2, 4, 2, 1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(ch * 2, ch * 4, 4, 2, 1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(ch * 4, 1, 3, 1, 1)))

    def forward(self, rgba, x):
        cond = torch.cat([x[..., :3], x[..., 3:6]], -1).permute(0, 3, 1, 2)
        return self.net(torch.cat([train.over_white(rgba), rgba[:, 3:4], cond], 1))


class Sharp(nn.Module):
    def __init__(self, base, width, sr=16):
        super().__init__()
        self.base = base
        self.conv = nn.Conv2d(width + 18, sr, 3, padding=1)
        self.head = nn.Conv2d(sr + 18, 4, 1)

    def forward(self, x, pitch, idx=None):
        b = self.base
        lo = inputs(b, half(x), pitch)
        hi = inputs(b, x, pitch)
        code = b.codes(idx) if idx is not None else torch.zeros(x.shape[0], b.codes.embedding_dim, device=x.device)
        h = features(b.dec, lo, code)
        h = F.interpolate(h, size=hi.shape[-2:], mode='bilinear', align_corners=False)
        h = F.gelu(self.conv(torch.cat([h, hi], 1)))
        return torch.sigmoid(self.head(torch.cat([h, hi], 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', action='append', required=True)
    ap.add_argument('--init', required=True)
    ap.add_argument('--out', default='runs/sharp_v1')
    ap.add_argument('--iters', type=int, default=6000)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--crop', default='512x384', help='full-resolution training crop, HxW, multiples of 16')
    ap.add_argument('--sr', type=int, default=16)
    ap.add_argument('--vgg', type=float, default=0.1)
    ap.add_argument('--gan', type=float, default=0.0, help='adversarial weight (crisper lines); 0 trains without the critic')
    ap.add_argument('--resume', help='a train_sr.py checkpoint to continue from (with its head)')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    X, Y, P = [], [], []
    for d in a.data:
        _, x, y, p = train.load(d)
        X.append(x), Y.append(y), P.append(p)
    X, Y, P = np.concatenate(X), np.concatenate(Y), np.concatenate(P)
    n = len(X)
    held = np.arange(n) % 10 == 5
    tr, te = np.nonzero(~held)[0], np.nonzero(held)[0]
    Xg, Yg, Pg = torch.from_numpy(X).to(dev), torch.from_numpy(Y).to(dev), torch.from_numpy(P).to(dev)
    del X, Y
    ck = torch.load(a.init, map_location='cpu')
    widths = tuple(ck.get('widths', (32, 48, 64, 96)))
    base = train.Shader(len(ck['parts']), n, 'hybrid', widths=widths)
    sd = {k: v for k, v in ck['model'].items() if not k.startswith('codes.')}
    print('init:', base.load_state_dict(sd, strict=False), flush=True)
    model = Sharp(base, widths[0], a.sr)
    if a.resume:
        rk = torch.load(a.resume, map_location='cpu')
        base.load_state_dict({k: v for k, v in rk['model'].items() if not k.startswith('codes.')}, strict=False)
        model.conv.load_state_dict({k[5:]: v for k, v in rk['sr'].items() if k.startswith('conv.')})
        model.head.load_state_dict({k[5:]: v for k, v in rk['sr'].items() if k.startswith('head.')})
        print('resumed from', a.resume, flush=True)
    model = model.to(dev)
    critic = Critic().to(dev) if a.gan else None
    copt = torch.optim.Adam(critic.parameters(), lr=2e-4, betas=(0.5, 0.99)) if critic else None
    vgg = train.VGG().to(dev)
    opt = torch.optim.Adam([
        {'params': base.tex.parameters(), 'lr': 3e-3},
        {'params': base.codes.parameters(), 'lr': 1e-3},
        {'params': base.dec.parameters(), 'lr': 3e-4},
        {'params': list(model.conv.parameters()) + list(model.head.parameters()), 'lr': 1e-3},
    ])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1, t / 200) * 0.5 * (1 + math.cos(math.pi * t / a.iters)))
    H, W = Xg.shape[1:3]
    ch, cw = map(int, a.crop.split('x'))
    print(f'{n} frames at {W}x{H} ({len(tr)} train, {len(te)} held out); sharp head {a.sr} channels', flush=True)
    log = open(out / 'log.txt', 'a')
    t0 = time.time()
    for it in range(1, a.iters + 1):
        idx = torch.from_numpy(np.random.choice(tr, a.batch)).to(dev)
        cy = np.random.randint(0, (H - ch) // 16 + 1) * 16
        cx = np.random.randint(0, (W - cw) // 16 + 1) * 16
        x = Xg[idx, cy:cy + ch, cx:cx + cw].float()
        y = Yg[idx, cy:cy + ch, cx:cx + cw].permute(0, 3, 1, 2).float()
        pred = model(x, Pg[idx], idx)
        loss = F.l1_loss(pred[:, 3], y[:, 3]) + F.l1_loss(pred[:, :3] * y[:, 3:4], y[:, :3] * y[:, 3:4])
        loss = loss + a.vgg * vgg(train.over_white(pred), train.over_white(y)) + 1e-3 * base.codes(idx).pow(2).mean()
        if critic is not None:
            # Hinge loss: the critic learns film patches from shader patches; the shader learns to fool it.
            cl = F.relu(1 - critic(y, x)).mean() + F.relu(1 + critic(pred.detach(), x)).mean()
            copt.zero_grad(set_to_none=True)
            cl.backward()
            copt.step()
            loss = loss - a.gan * critic(pred, x).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % 500 == 0 or it == a.iters:
            model.eval()
            with torch.no_grad():
                errs = []
                for k in range(0, len(te), 4):
                    ids = torch.from_numpy(te[k:k + 4]).to(dev)
                    pr = model(Xg[ids].float(), Pg[ids])
                    errs.append(F.l1_loss(train.over_white(pr), train.over_white(Yg[ids].permute(0, 3, 1, 2).float()), reduction='none').mean((1, 2, 3)))
                err = torch.cat(errs).mean().item()
                line = f'it {it}  loss {loss.item():.4f}  held-out L1 {err:.4f}  {time.time() - t0:.0f}s'
                print(line, flush=True)
                log.write(line + '\n')
                log.flush()
                if it % 2000 == 0 or it == a.iters:
                    pick = te[np.linspace(0, len(te) - 1, 4).astype(int)]
                    ids = torch.from_numpy(pick).to(dev)
                    pr = train.over_white(model(Xg[ids].float(), Pg[ids]))
                    tg = train.over_white(Yg[ids].permute(0, 3, 1, 2).float())
                    rows = torch.cat([torch.cat([q, t], 2) for q, t in zip(pr, tg)], 2)
                    Image.fromarray((rows.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(out / f'preview_{it:05d}.png')
            model.train()
    torch.save({'model': base.state_dict(), 'sr': {'conv.weight': model.conv.weight, 'conv.bias': model.conv.bias,
                'head.weight': model.head.weight, 'head.bias': model.head.bias},
                'parts': ck['parts'], 'arch': 'hybrid', 'widths': widths, 'sr_width': a.sr}, out / 'shader.pt')
    print('saved', out / 'shader.pt', flush=True)


if __name__ == '__main__':
    main()
