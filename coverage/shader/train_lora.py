#!/usr/bin/env python3
"""Teach the sharp shader new poses with low-rank adapters, without retraining it.

    python3 train_lora.py --base runs/sharp_v1/shader.pt --data data/w1_hd --data data/bottom_hd \\
        --posed data/posed_hd --out runs/sharp_lora [--rank 8] [--iters 3000]

The films only ever showed her standing with her arms down, so palms, raised arms and armpits are
new to the shader (it paints them in the dress's navy). --posed holds drawn frames of mocap poses
(coverage/motion/pick_poses.py, render-orbit.mjs --poses, anchors.py --every 1, build_dataset.py).
Each convolution of the decoder and the full-resolution head gets a rank-r adapter (a k x k
convolution down to r channels, then a 1 x 1 back up, starting at zero); those and the neural
texture train, the original weights stay frozen. Batches mix posed frames with the original film
(--data, in the base's training order so its appearance codes still apply), so what it already drew
well stays put. The adapters are merged into the weights at the end: the saved checkpoint has the
base's exact layout and export.py takes it unchanged.

--equi adds a shift-consistency loss against flicker: the browser runs the decoder at half
resolution inside a box that follows the bones, so as she moves her features slide across the 2 x 2
pooling grids, and a shader that draws a feature differently at different grid phases shimmers.
Each step also draws a crop shifted by 1 to --shift pixels and asks that drawing-then-shifting
equal shifting-then-drawing (away from the crops' edges). The log reports that shift error on
held-out frames, the number this loss pulls down.
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
from train_sr import Sharp

dev = 'cuda'


class LoRAConv(nn.Module):
    def __init__(self, conv, rank, alpha=1.0):
        super().__init__()
        self.conv = conv
        self.down = nn.Conv2d(conv.in_channels, rank, conv.kernel_size, padding=conv.padding, bias=False)
        self.up = nn.Conv2d(rank, conv.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scale = alpha

    def forward(self, x):
        return self.conv(x) + self.scale * self.up(self.down(x))

    def merged(self):
        """The plain convolution with the adapter folded in."""
        w = self.conv.weight + self.scale * torch.einsum('or,rikl->oikl', self.up.weight[:, :, 0, 0], self.down.weight)
        c = nn.Conv2d(self.conv.in_channels, self.conv.out_channels, self.conv.kernel_size, padding=self.conv.padding)
        c.weight.data.copy_(w.detach())
        c.bias.data.copy_(self.conv.bias.detach())
        return c


def wrap(module, rank):
    """Replace every Conv2d under module with an adapted one; returns the adapters."""
    out = []
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            lc = LoRAConv(child, rank)
            setattr(module, name, lc)
            out.append(lc)
        else:
            out += wrap(child, rank)
    return out


def unwrap(module):
    for name, child in module.named_children():
        if isinstance(child, LoRAConv):
            setattr(module, name, child.merged())
        else:
            unwrap(child)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--data', action='append', required=True, help='the base\'s own datasets, in its order')
    ap.add_argument('--posed', action='append', required=True)
    ap.add_argument('--out', default='runs/sharp_lora')
    ap.add_argument('--rank', type=int, default=8)
    ap.add_argument('--iters', type=int, default=3000)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--posed-share', type=float, default=0.5, help='fraction of each batch from --posed')
    ap.add_argument('--crop', default='512x384')
    ap.add_argument('--vgg', type=float, default=0.1)
    ap.add_argument('--equi', type=float, default=0.0, help='weight of the shift-consistency loss')
    ap.add_argument('--shift', type=int, default=15, help='largest shift for --equi, pixels')
    ap.add_argument('--tex-lr', type=float, default=2e-3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def stack(dirs):
        X, Y, P = [], [], []
        for d in dirs:
            _, x, y, p = train.load(d)
            X.append(x), Y.append(y), P.append(p)
        return [torch.from_numpy(np.concatenate(v)).to(dev) for v in (X, Y, P)]

    Xo, Yo, Po = stack(a.data)
    Xp, Yp, Pp = stack(a.posed)
    n_old, n_pose = len(Xo), len(Xp)
    held = np.arange(n_pose) % 10 == 5
    ptr, pte = np.nonzero(~held)[0], np.nonzero(held)[0]
    otr = np.nonzero(np.arange(n_old) % 10 != 5)[0]
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
    adapters = wrap(base.dec, a.rank)
    model.conv, model.head = LoRAConv(model.conv, a.rank), LoRAConv(model.head, a.rank)
    adapters += [model.conv, model.head]
    for p in base.tex.parameters():
        p.requires_grad_(True)
    model = model.to(dev)
    lora = [p for ad in adapters for p in (ad.down.weight, ad.up.weight)]
    print(f'{len(adapters)} adapters, rank {a.rank}: {sum(p.numel() for p in lora)} parameters; '
          f'{n_pose} posed frames ({len(pte)} held out), {n_old} film frames', flush=True)
    vgg = train.VGG().to(dev)
    opt = torch.optim.Adam([{'params': lora, 'lr': 1e-3}, {'params': base.tex.parameters(), 'lr': a.tex_lr}])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1, t / 100) * 0.5 * (1 + math.cos(math.pi * t / a.iters)))
    H, W = Xo.shape[1:3]
    ch, cw = map(int, a.crop.split('x'))
    n_p = max(1, round(a.batch * a.posed_share))

    def loss_of(pred, y):
        return (F.l1_loss(pred[:, 3], y[:, 3]) + F.l1_loss(pred[:, :3] * y[:, 3:4], y[:, :3] * y[:, 3:4])
                + a.vgg * vgg(train.over_white(pred), train.over_white(y)))

    def evaluate(X, Y, P, ids, codes):
        errs = []
        with torch.no_grad():
            for k in range(0, len(ids), 4):
                i = torch.from_numpy(ids[k:k + 4]).to(dev)
                pr = model(X[i].float(), P[i], i if codes else None)
                errs.append(F.l1_loss(train.over_white(pr), train.over_white(Y[i].permute(0, 3, 1, 2).float()), reduction='none').mean((1, 2, 3)))
        return torch.cat(errs).mean().item()

    M = 32  # margin kept clear of the crops' edges, where zero padding differs between the two

    def shifted_pair(X, P, ids, codes, y0, x0, dy, dx):
        """Drawings of a crop and of the same crop moved by (dy, dx), cut to where they overlap."""
        A = model(X[ids, y0:y0 + ch, x0:x0 + cw].float(), P[ids], ids if codes else None)
        B = model(X[ids, y0 + dy:y0 + dy + ch, x0 + dx:x0 + dx + cw].float(), P[ids], ids if codes else None)
        A = A[:, :, dy + M:ch - M, dx + M:cw - M]
        B = B[:, :, M:ch - dy - M, M:cw - dx - M]
        return A, B

    def equi_loss(A, B):
        on = (torch.maximum(A[:, 3:4], B[:, 3:4]) > 0.02).float()
        d = (train.over_white(A) - train.over_white(B)).abs() * on
        return d.sum() / (on.sum() * 3 + 1)

    def shift_error(X, P, ids, codes):
        """Held-out shift error: mean over shifts 1, 3, 5, 8 (diagonal) of the drawing's change."""
        errs = []
        with torch.no_grad():
            for k in range(0, len(ids), 2):
                i = torch.from_numpy(ids[k:k + 2]).to(dev)
                for s in (1, 3, 5, 8):
                    A, B = shifted_pair_full(X, P, i, codes, s)
                    errs.append(equi_loss(A, B).item())
        return float(np.mean(errs))

    def shifted_pair_full(X, P, i, codes, s):
        hc, wc = H - 16, W - 16
        A = model(X[i, :hc, :wc].float(), P[i], i if codes else None)[:, :, s + M:hc - M, s + M:wc - M]
        B = model(X[i, s:s + hc, s:s + wc].float(), P[i], i if codes else None)[:, :, M:hc - s - M, M:wc - s - M]
        return A, B

    log = open(out / 'log.txt', 'a')
    t0 = time.time()
    for it in range(0, a.iters + 1):
        if it % 250 == 0:
            model.eval()
            line = (f'it {it}  posed held-out L1 {evaluate(Xp, Yp, Pp, pte, False):.4f}  '
                    f'film held-out L1 {evaluate(Xo, Yo, Po, ote, True):.4f}  '
                    f'shift error posed {shift_error(Xp, Pp, pte[::2], False):.4f} '
                    f'film {shift_error(Xo, Po, ote[::3], True):.4f}  {time.time() - t0:.0f}s')
            print(line, flush=True)
            log.write(line + '\n')
            log.flush()
            model.train()
            if it == a.iters:
                break
        # Any offset, not just the 16-pixel grid: the browser's crop box lands anywhere.
        sh = a.shift if a.equi else 0
        cy = np.random.randint(0, H - ch - sh + 1)
        cx = np.random.randint(0, W - cw - sh + 1)
        ip = torch.from_numpy(np.random.choice(ptr, n_p)).to(dev)
        io = torch.from_numpy(np.random.choice(otr, a.batch - n_p)).to(dev)
        lp = loss_of(model(Xp[ip, cy:cy + ch, cx:cx + cw].float(), Pp[ip]), Yp[ip, cy:cy + ch, cx:cx + cw].permute(0, 3, 1, 2).float())
        lo = loss_of(model(Xo[io, cy:cy + ch, cx:cx + cw].float(), Po[io], io), Yo[io, cy:cy + ch, cx:cx + cw].permute(0, 3, 1, 2).float())
        loss = (lp * n_p + lo * (a.batch - n_p)) / a.batch
        if a.equi:
            dy, dx = np.random.randint(0, a.shift + 1, 2)
            if dy == dx == 0:
                dy = 1
            e = equi_loss(*shifted_pair(Xp, Pp, ip[:1], False, cy, cx, dy, dx)) + equi_loss(*shifted_pair(Xo, Po, io[:1], True, cy, cx, dy, dx))
            loss = loss + a.equi * e / 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

    # Previews: posed held-out frames, drawing beside target.
    model.eval()
    with torch.no_grad():
        ids = torch.from_numpy(pte[np.linspace(0, len(pte) - 1, 4).astype(int)]).to(dev)
        pr = train.over_white(model(Xp[ids].float(), Pp[ids]))
        tg = train.over_white(Yp[ids].permute(0, 3, 1, 2).float())
        rows = torch.cat([torch.cat([q, t], 2) for q, t in zip(pr, tg)], 2)
        Image.fromarray((rows.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(out / 'preview.png')
    unwrap(model)
    torch.save({'model': base.state_dict(), 'sr': {'conv.weight': model.conv.weight, 'conv.bias': model.conv.bias,
                'head.weight': model.head.weight, 'head.bias': model.head.bias},
                'parts': ck['parts'], 'arch': 'hybrid', 'widths': widths, 'sr_width': ck.get('sr_width', 16)}, out / 'shader.pt')
    print('saved (adapters merged)', out / 'shader.pt', flush=True)


if __name__ == '__main__':
    main()
