#!/usr/bin/env python3
"""Teach the shader to light her: the drawing follows a light the viewer sets, in the film's flat style.

    python3 train_light.py --base runs/sharp_crisp/shader.pt --lit data/lit_hd \\
        --data data/w1_hd --data data/bottom_hd --posed data/posed_hd --out runs/sharp_light

The network gets 7 more input channels at both resolutions: the cosine between the light and the
surface (from the G-buffer's view-space normals), the key light's colour times its strength, and the
ambient's (both relative to the films' light, so the films' own light reads 1, 1, 1). The weights
reading them start at zero, so it starts as the base. --lit holds drawings of her under varied lights
(coverage/motion/pick_lights.py, render-orbit.mjs --poses, anchors.py with the lighting spelled out);
those train against their drawings. Everything else (--data, --posed) is held to the frozen base's own
drawing under the films' light, so with that light the result still draws as the base does. Low-rank
adapters (train_lora.py) on every convolution, the new input weights in full, and the shift loss
against flicker. The saved checkpoint carries 'light': 7; export.py writes it into shader.json.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import train
from train_lora import LoRAConv, wrap, unwrap
from train_sr import Sharp, inputs, half, features

dev = 'cuda'
LIGHT_CH = 7
FILM_LIGHT = {'dir': [1, 3, 2], 'color': [1, 1, 1], 'intensity': 2.2, 'amb': 1.2, 'ambColor': [1, 1, 1]}


def light_params(frame):
    """A frame's light -> [view-space direction (3), key colour x strength (3), ambient (3)], relative
    to the films' light. The camera orbits her at the frame's yaw and pitch, looking at her centre."""
    L = frame.get('light') or FILM_LIGHT
    y, p = math.radians(frame['yaw']), math.radians(frame['pitch'])
    back = np.array([math.sin(y) * math.cos(p), math.sin(p), math.cos(y) * math.cos(p)])
    right = np.cross([0, 1, 0], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    d = np.array(L['dir'], float)
    d /= np.linalg.norm(d)
    view = [d @ right, d @ up, d @ back]
    key = np.array(L.get('color', [1, 1, 1])) * L.get('intensity', 2.2) / 2.2
    amb = np.array(L.get('ambColor', [1, 1, 1])) * L.get('amb', 1.2) / 1.2
    return np.concatenate([view, key, amb]).astype(np.float32)


def light_channels(x, lp):
    """x [B, H, W, 11] G-buffer, lp [B, 9] -> [B, 7, H, W]: cos(light, normal), key, ambient, on her."""
    n = x[..., 3:6] * 2 - 1
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-3)
    mask = x[..., 7:8]
    ndl = (n * lp[:, None, None, :3]).sum(-1, keepdim=True)
    rest = lp[:, None, None, 3:].expand(-1, *x.shape[1:3], -1)
    return (torch.cat([ndl, rest], -1) * mask).permute(0, 3, 1, 2)


class Expanded(nn.Module):
    """A convolution with LIGHT_CH more input channels at the end, read by zero-initialised weights."""

    def __init__(self, conv):
        super().__init__()
        self.old = conv
        self.n = conv.in_channels
        self.w_new = nn.Parameter(torch.zeros(conv.out_channels, LIGHT_CH, *conv.kernel_size))
        self.pad = conv.padding

    def forward(self, x):
        return self.old(x[:, :self.n]) + F.conv2d(x[:, self.n:], self.w_new, padding=self.pad)

    def merged(self):
        c = self.old.merged() if isinstance(self.old, LoRAConv) else self.old
        out = nn.Conv2d(self.n + LIGHT_CH, c.out_channels, c.kernel_size, padding=c.padding)
        out.weight.data.copy_(torch.cat([c.weight.detach(), self.w_new.detach()], 1))
        out.bias.data.copy_(c.bias.detach())
        return out


def lit_forward(model, x, pitch, idx, lp):
    b = model.base
    xh = half(x)
    lo = torch.cat([inputs(b, xh, pitch), light_channels(xh, lp)], 1)
    hi = torch.cat([inputs(b, x, pitch), light_channels(x, lp)], 1)
    code = b.codes(idx) if idx is not None else torch.zeros(x.shape[0], b.codes.embedding_dim, device=x.device)
    h = features(b.dec, lo, code)
    h = F.interpolate(h, size=hi.shape[-2:], mode='bilinear', align_corners=False)
    h = F.gelu(model.conv(torch.cat([h, hi], 1)))
    return torch.sigmoid(model.head(torch.cat([h, hi], 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--lit', action='append', required=True)
    ap.add_argument('--data', action='append', required=True, help='the base\'s own datasets, in its order')
    ap.add_argument('--posed', action='append', default=[])
    ap.add_argument('--out', default='runs/sharp_light')
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--iters', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lit-share', type=float, default=0.5)
    ap.add_argument('--crop', default='512x384')
    ap.add_argument('--vgg', type=float, default=0.1)
    ap.add_argument('--equi', type=float, default=2.0)
    ap.add_argument('--shift', type=int, default=15)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def stack(dirs):
        X, Y, P, L = [], [], [], []
        for d in dirs:
            _, x, y, p = train.load(d)
            meta = json.loads((Path(d) / 'meta.json').read_text())
            X.append(x), Y.append(y), P.append(p), L.append(np.stack([light_params(f) for f in meta['frames']]))
        return [torch.from_numpy(np.concatenate(v)).to(dev) for v in (X, Y, P, L)]

    Xo, Yo, Po, Lo = stack(a.data)
    n_old = len(Xo)
    if a.posed:
        Xq, Yq, Pq, Lq = stack(a.posed)
    Xl, Yl, Pl, Ll = stack(a.lit)
    held = np.arange(len(Xl)) % 10 == 5
    ltr, lte = np.nonzero(~held)[0], np.nonzero(held)[0]
    otr = np.arange(n_old)
    ote = np.nonzero(np.arange(n_old) % 10 == 5)[0][::4]

    ck = torch.load(a.base, map_location='cpu')
    widths = tuple(ck['widths'])
    base = train.Shader(len(ck['parts']), n_old, 'hybrid', widths=widths)
    base.load_state_dict(ck['model'])
    model = Sharp(base, widths[0], ck.get('sr_width', 16))
    model.conv.load_state_dict({k[5:]: v for k, v in ck['sr'].items() if k.startswith('conv.')})
    model.head.load_state_dict({k[5:]: v for k, v in ck['sr'].items() if k.startswith('head.')})
    for p in model.parameters():
        p.requires_grad_(False)
    teacher = copy.deepcopy(model).to(dev).eval()
    # The light channels: the first convolution, and the full-resolution head's two.
    base.dec.down[0][0] = Expanded(base.dec.down[0][0])
    model.conv, model.head = Expanded(model.conv), Expanded(model.head)
    adapters = wrap(base.dec, a.rank)
    for e in (model.conv, model.head):
        e.old = LoRAConv(e.old, a.rank)
        adapters.append(e.old)
    model = model.to(dev)
    new_w = [base.dec.down[0][0].w_new, model.conv.w_new, model.head.w_new]
    lora = [p for ad in adapters for p in (ad.down.weight, ad.up.weight)]
    print(f'{len(adapters)} adapters rank {a.rank}, light inputs {sum(p.numel() for p in new_w)} weights; '
          f'{len(Xl)} lit frames ({len(lte)} held out), {n_old} film + {len(Xq) if a.posed else 0} posed frames', flush=True)
    vgg = train.VGG().to(dev)
    opt = torch.optim.Adam([{'params': lora, 'lr': 1e-3}, {'params': new_w, 'lr': 2e-3}])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1, t / 100) * 0.5 * (1 + math.cos(math.pi * t / max(1, a.iters))))
    H, W = Xo.shape[1:3]
    ch, cw = map(int, a.crop.split('x'))
    n_l = max(1, round(a.batch * a.lit_share))
    M = 32

    def target_loss(pred, y):
        return (F.l1_loss(pred[:, 3], y[:, 3]) + F.l1_loss(pred[:, :3] * y[:, 3:4], y[:, :3] * y[:, 3:4])
                + a.vgg * vgg(train.over_white(pred), train.over_white(y)))

    def equi(X, P, L, ids, codes, y0, x0, dy, dx):
        A = lit_forward(model, X[ids, y0:y0 + ch, x0:x0 + cw].float(), P[ids], ids if codes else None, L[ids])
        B = lit_forward(model, X[ids, y0 + dy:y0 + dy + ch, x0 + dx:x0 + dx + cw].float(), P[ids], ids if codes else None, L[ids])
        A, B = A[:, :, dy + M:ch - M, dx + M:cw - M], B[:, :, M:ch - dy - M, M:cw - dx - M]
        on = (torch.maximum(A[:, 3:4], B[:, 3:4]) > 0.02).float()
        return ((train.over_white(A) - train.over_white(B)).abs() * on).sum() / (on.sum() * 3 + 1)

    def evaluate():
        with torch.no_grad():
            lit, keep = [], []
            for k in range(0, len(lte), 4):
                i = torch.from_numpy(lte[k:k + 4]).to(dev)
                pr = lit_forward(model, Xl[i].float(), Pl[i], None, Ll[i])
                lit.append(F.l1_loss(train.over_white(pr), train.over_white(Yl[i].permute(0, 3, 1, 2).float())).item())
                base_pr = teacher(Xl[i].float(), Pl[i])
                keep.append(F.l1_loss(train.over_white(base_pr), train.over_white(Yl[i].permute(0, 3, 1, 2).float())).item())
            film = []
            for k in range(0, len(ote), 4):
                i = torch.from_numpy(ote[k:k + 4]).to(dev)
                film.append(F.l1_loss(train.over_white(lit_forward(model, Xo[i].float(), Po[i], i, Lo[i])),
                                      train.over_white(teacher(Xo[i].float(), Po[i], i))).item())
        return np.mean(lit), np.mean(keep), np.mean(film)

    log = open(out / 'log.txt', 'a')
    t0 = time.time()
    for it in range(0, a.iters + 1):
        if it % 250 == 0:
            model.eval()
            lit, base_lit, film = evaluate()
            line = f'it {it}  lit held-out L1 {lit:.4f} (base, unlit: {base_lit:.4f})  film vs base {film:.4f}  {time.time() - t0:.0f}s'
            print(line, flush=True)
            log.write(line + '\n')
            log.flush()
            model.train()
            if it == a.iters:
                break
        cy = np.random.randint(0, H - ch - a.shift + 1)
        cx = np.random.randint(0, W - cw - a.shift + 1)
        il = torch.from_numpy(np.random.choice(ltr, n_l)).to(dev)
        xl = Xl[il, cy:cy + ch, cx:cx + cw].float()
        loss = target_loss(lit_forward(model, xl, Pl[il], None, Ll[il]), Yl[il, cy:cy + ch, cx:cx + cw].permute(0, 3, 1, 2).float()) * n_l
        # The rest: film (and posed) frames, held to the base's own drawing under the films' light.
        for j in range(a.batch - n_l):
            use_posed = a.posed and j % 2 == 1
            X, P, L = (Xq, Pq, Lq) if use_posed else (Xo, Po, Lo)
            i = torch.from_numpy(np.random.choice(len(X) if use_posed else otr, 1)).to(dev)
            codes = None if use_posed else i
            xo = X[i, cy:cy + ch, cx:cx + cw].float()
            with torch.no_grad():
                t = teacher(xo, P[i], codes)
            pr = lit_forward(model, xo, P[i], codes, L[i])
            loss = loss + (F.l1_loss(train.over_white(pr), train.over_white(t)) + F.l1_loss(pr[:, 3], t[:, 3])) * 4
        loss = loss / a.batch
        if a.equi:
            dy, dx = np.random.randint(0, a.shift + 1, 2)
            loss = loss + a.equi * equi(Xl, Pl, Ll, il[:1], False, cy, cx, max(dy, 1), dx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        ids = lte[np.linspace(0, len(lte) - 1, 5).astype(int)]
        i = torch.from_numpy(ids).to(dev)
        pr = train.over_white(lit_forward(model, Xl[i].float(), Pl[i], None, Ll[i]))
        bp = train.over_white(teacher(Xl[i].float(), Pl[i]))
        tg = train.over_white(Yl[i].permute(0, 3, 1, 2).float())
        rows = torch.cat([torch.cat([q, r, t], 1) for q, r, t in zip(bp, pr, tg)], 2)  # base / lit / target
        Image.fromarray((rows.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(out / 'preview.png')
    unwrap(base.dec)
    base.dec.down[0][0] = base.dec.down[0][0].merged()
    model.conv, model.head = model.conv.merged(), model.head.merged()
    torch.save({'model': base.state_dict(), 'sr': {'conv.weight': model.conv.weight, 'conv.bias': model.conv.bias,
                'head.weight': model.head.weight, 'head.bias': model.head.bias},
                'parts': ck['parts'], 'arch': 'hybrid', 'widths': widths, 'sr_width': ck.get('sr_width', 16), 'light': LIGHT_CH},
               out / 'shader.pt')
    print('saved', out / 'shader.pt', flush=True)


if __name__ == '__main__':
    main()
