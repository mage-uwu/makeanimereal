#!/usr/bin/env python3
"""upsample_torch.py -- a learned, style-aware version of the viewer's edge-aware upscale (nts-gl.js
upscale()): the network draws at its training resolution, this lifts its drawing to the screen's.

The viewer's upscale: a screen pixel blends the 4 x 4 network pixels around it, weighted by bilinear's
weights and by how well each one's geometry (depth, normal, her mask) matches the screen-resolution
geometry there (guide.js), its colour weighted by alpha^2. Here each of the 16 gets a learned logit on
top, from a tiny MLP shared by all 16 (10 -> 12 -> 1) that sees:
    its colour minus the blend's (3), its alpha, its offset (2), the depth step (dz), 1 - n.gn, its
    mask, log of the hand-made weight
so it can learn which pixels to trust: a speckle (an outlier no neighbour agrees with) against a line
(dark, but on the drawing's lines), which side of an edge a screen pixel is on. The output colour is
still a blend of the drawing's own colours (it can't invent any), its coverage still the geometry's.
The MLP's last layer starts at zero: training starts from exactly the hand-made upscale.

Trained per character at 2x (the data's full size: 720 x 1280) against the film, with the style
model's drawing at 1x (box-averaged inputs, as it trained) fixed:
    python3 upsample_torch.py --keys keys-film.json --model v2/style.ntst --frozen v2/frozen_state.pt \
        --steps 3000 --out ups
Writes ups/upsample.json (the MLP for nts-gl.js) and held-out sheets: bilinear | hand-made | learned | film.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import shader_torch
from shader_torch import Net, forward
from ssm_tree import SSMTree
from make_style_data import build
from style_torch import VGG, perceptual, edges, over_white

H_IN = 10


def box2(t):   # (B, C, H, W) -> half size, the mean of each 2 x 2
    return F.avg_pool2d(t, 2)


def lowres(px):
    """px (B, H, W, 17) uint8 at full size -> the network's input at half size, as make_style_data.py
    makes it (area for colour, normal, depth and the target; the top-left texel for uv, part, mask)."""
    q = px.permute(0, 3, 1, 2).float()
    smooth = box2(q[:, 0:7]); enc = q[:, 7:13, ::2, ::2]
    t = q[:, 13:17]; pa = box2(torch.cat([t[:, :3] * t[:, 3:4] / 255, t[:, 3:4]], 1))
    tgt = torch.cat([pa[:, :3] * 255 / pa[:, 3:4].clamp_min(1e-3), pa[:, 3:4]], 1)
    return torch.cat([smooth, enc, tgt], 1).round().clamp(0, 255).byte().permute(0, 2, 3, 1).contiguous()


class Upsampler(torch.nn.Module):
    def __init__(self, hid=12, detail=0, parts=0, freqs=6):
        super().__init__()
        self.l1 = torch.nn.Linear(H_IN, hid); self.l2 = torch.nn.Linear(hid, 1)
        torch.nn.init.zeros_(self.l2.weight); torch.nn.init.zeros_(self.l2.bias)
        # detail: a colour residual at the output's resolution from its geometry (normal, a learned code
        # per part, sines of the uv at 2^0..2^(freqs-1) cycles): what the drawing is too coarse to hold
        self.detail, self.freqs = detail, freqs
        if detail:
            self.pcode = torch.nn.Parameter(torch.zeros(parts, 4))
            self.d1 = torch.nn.Linear(3 + 3 + 4 + 4 * freqs, detail); self.d2 = torch.nn.Linear(detail, 3)
            torch.nn.init.zeros_(self.d2.weight); torch.nn.init.zeros_(self.d2.bias)

    def forward(self, R, T1, T2, T4, gn, gd, cov, uv=None, part=None, learned=True, scale=None, off=0.0):
        """R (B, 4, h, w) the drawing (premultiplied); T1 normals (rgb, 0..1), T2 depth, T4 her mask, all
        (B, ., h, w) at the network's resolution; gn (B, 3, H, W) the guide's unit normal, gd its depth,
        cov its coverage, at the output's. scale: output pixels a drawing pixel (H / h); off: where the
        output's first pixel lies past the drawing's (in drawing pixels; a crop with a margin).
        Returns (B, 4, H, W) premultiplied, as the viewer's display."""
        B, _, h, w = R.shape; H, W = gd.shape[-2:]
        dev = R.device
        # each output pixel's position in the drawing (uv * lowRes - 0.5), its 4 x 4 around floor(P)
        sy, sx = (scale, scale) if scale else (H / h, W / w)
        Py = (torch.arange(H, device=dev) + .5) / sy - .5 + off; Px = (torch.arange(W, device=dev) + .5) / sx - .5 + off
        by, bx = torch.floor(Py).long(), torch.floor(Px).long()
        logits, cols, alphas, wh = [], [], [], []
        n_low = T1 * 2 - 1
        n_low = n_low / n_low.norm(dim=1, keepdim=True).clamp_min(1e-6)
        taps = []
        for j in range(-1, 3):
            for i in range(-1, 3):
                qy, qx = (by + j).clamp(0, h - 1), (bx + i).clamp(0, w - 1)
                g = lambda t: t[:, :, qy][:, :, :, qx]                        # (B, C, H, W)
                c = g(R); n = g(n_low); dz = (g(T2) - gd) * 80
                dy = (Py - (by + j).float())[:, None]; dx = (Px - (bx + i).float())[None, :]
                ty, tx = (1 - dy.abs()).clamp_min(0), (1 - dx.abs()).clamp_min(0)
                wgt = (ty * tx + .02 * torch.exp(-.5 * (dx * dx + dy * dy)))[None, None] \
                    * torch.exp(-.5 * dz * dz) * torch.exp(-8 * (1 - (n * gn).sum(1, keepdim=True)))
                msk = (g(T4) > .5).float()
                wgt = wgt * (msk + (1 - msk) * .05)
                taps.append((c, wgt, dz, 1 - (n * gn).sum(1, keepdim=True), msk, dx.expand(H, W), dy.expand(H, W)))
        a = torch.stack([t[0][:, 3] for t in taps], 1)                         # (B, 16, H, W)
        valid = (a >= .004).float()
        wk = torch.stack([t[1][:, 0] for t in taps], 1) * valid
        u = torch.stack([t[0][:, :3] for t in taps], 1) / a.clamp_min(.004)[:, :, None]   # (B, 16, 3, H, W)
        sa, saa = (wk * a).sum(1), (wk * a * a).sum(1)
        base = wk * a * a                                                      # the hand-made colour weights
        m = (base[:, :, None] * u).sum(1) / saa.clamp_min(1e-4)[:, None]       # (B, 3, H, W): its colour
        if learned:
            feat = torch.stack([*(u - m[:, None]).unbind(2), a,
                                torch.stack([t[5] for t in taps])[None].expand(B, -1, -1, -1),
                                torch.stack([t[6] for t in taps])[None].expand(B, -1, -1, -1),
                                torch.stack([t[2][:, 0] for t in taps], 1), torch.stack([t[3][:, 0] for t in taps], 1),
                                torch.stack([t[4][:, 0] for t in taps], 1), torch.log(wk + 1e-4)], -1)   # (B, 16, H, W, 10)
            lg = torch.log(base + 1e-8) + self.l2(F.relu(self.l1(feat)))[..., 0]
            lg = lg.masked_fill(valid == 0, -1e4)
            sw = torch.softmax(lg, 1)
            rgb = (sw[:, :, None] * u).sum(1)
        else:
            rgb = m
        if learned and self.detail and uv is not None:
            f = [(2 ** k) * 2 * math.pi * uv for k in range(self.freqs)]
            enc = torch.cat([torch.sin(x) for x in f] + [torch.cos(x) for x in f], 1)                 # (B, 4 freqs, H, W)
            code = self.pcode[part[:, 0].clamp(0, self.pcode.shape[0] - 1)].permute(0, 3, 1, 2)
            x = torch.cat([rgb, gn, code, enc], 1).permute(0, 2, 3, 1)
            rgb = (rgb + self.d2(F.relu(self.d1(x))).permute(0, 3, 1, 2)).clamp(0, 1)
        # her coverage: the guide's, faded where the drawing hardly has her (smoothstep(0.03, 0.3, saa / sa))
        t = torch.clamp((saa / sa.clamp_min(1e-6) - .03) / .27, 0, 1)[:, None]
        alpha = cov * t * t * (3 - 2 * t) * (saa >= 1e-4).float()[:, None]
        return torch.cat([rgb * alpha, alpha], 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--keys', required=True); ap.add_argument('--model', required=True); ap.add_argument('--frozen', required=True)
    ap.add_argument('--parts', type=int, default=0); ap.add_argument('--every', type=int, default=2, help='use every n-th frame')
    ap.add_argument('--steps', type=int, default=3000); ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--crop', type=int, default=384, help='output crop (full-size pixels)')
    ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--vgg-w', type=float, default=1.0); ap.add_argument('--lines', type=float, default=5.0)
    ap.add_argument('--out', default='ups')
    ap.add_argument('--detail', type=int, default=0, help='the detail head\'s width (0: none)')
    ap.add_argument('--draw-batch', type=int, default=8, help='frames a pass when drawing at 1x')
    ap.add_argument('--vgg', default='imagenet', help="'random': code checks only")
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    keys = json.loads(Path(a.keys).read_text())[::a.every]
    t0 = time.time()
    d = build(keys, full=True)
    px = torch.from_numpy(d['px']); held = torch.from_numpy(d['held']); pitch = torch.from_numpy(d['pitch']).float()
    P = a.parts or int(d['parts'])
    print(f'{len(px)} frames {px.shape[2]}x{px.shape[1]} ({int(held.sum())} held out), {P} parts, {time.time() - t0:.0f}s', flush=True)
    # the style model's drawing at 1x, once, as the viewer draws it (frozen node memories)
    net = Net(4, P, [1, 3, 9, 27], 4, 4, dev); net.init(0); net.load(a.model)
    frozen = torch.load(a.frozen, map_location=dev)
    shader_torch.SSM.clear()
    for b, sd in zip([1, 3], torch.load(a.model + '.ssm.pt', map_location=dev)):
        m = SSMTree().to(dev); m.load_state_dict(sd); m.frozen = frozen[b]; shader_torch.SSM[b] = m
    Rs, lows = [], []
    with torch.no_grad():
        for k in range(0, len(px), a.draw_batch):
            lo = lowres(px[k:k + a.draw_batch].to(dev))
            p = forward(net, lo, pitch[k:k + a.draw_batch].to(dev), True)[0]
            Rs.append(torch.cat([p[..., :3] * p[..., 3:4], p[..., 3:4]], -1).permute(0, 3, 1, 2).half().cpu())
            lows.append(lo.cpu())
    R_all, low_all = torch.cat(Rs), torch.cat(lows)
    print(f'drawings at 1x: {time.time() - t0:.0f}s', flush=True)

    def inputs(ids, y0=None, x0=None, c=None):
        """Full-size crop [y0, y0 + c) (even) and its half-size source with 2 pixels' margin."""
        hp = px[ids].to(dev).permute(0, 3, 1, 2).float(); lp = low_all[ids].to(dev).permute(0, 3, 1, 2).float(); R = R_all[ids].to(dev).float()
        if c is not None:
            ly0, lx0 = y0 // 2 - 2, x0 // 2 - 2
            hp = hp[:, :, y0:y0 + c, x0:x0 + c]
            pad = lambda t: F.pad(t, (2, 2, 2, 2), mode='replicate')
            lp, R = pad(lp)[:, :, ly0 + 2:ly0 + 2 + c // 2 + 4, lx0 + 2:lx0 + 2 + c // 2 + 4], pad(R)[:, :, ly0 + 2:ly0 + 2 + c // 2 + 4, lx0 + 2:lx0 + 2 + c // 2 + 4]
        nrm = hp[:, 3:6] / 255 * 2 - 1; nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp_min(1e-6)
        cov = (hp[:, 12:13] > 127).float()
        uv = torch.cat([(hp[:, 7:8] + hp[:, 8:9] / 255) / 255, (hp[:, 9:10] + hp[:, 10:11] / 255) / 255], 1)
        args = (R, lp[:, 3:6] / 255, lp[:, 6:7] / 255, (lp[:, 12:13] > 127).float(), nrm, hp[:, 6:7] / 255, cov, uv, hp[:, 11:12].long())
        tgt = hp[:, 13:17] / 255
        return args, torch.cat([tgt[:, :3] * tgt[:, 3:4], tgt[:, 3:4]], 1)

    ups = Upsampler(detail=a.detail, parts=P).to(dev)
    opt = torch.optim.Adam(ups.parameters(), lr=a.lr)
    vgg = VGG(a.vgg == 'random').to(dev)
    tr, hi = np.nonzero(~held.numpy())[0], np.nonzero(held.numpy())[0]
    rng = np.random.default_rng(0)
    H, W = px.shape[1:3]
    cover = [torch.nonzero(px[i, ..., 12] > 127) for i in range(len(px))]
    ow = lambda t: over_white(torch.cat([t[:, :3] / t[:, 3:4].clamp_min(1e-3), t[:, 3:4]], 1).permute(0, 2, 3, 1).clamp(0, 1))   # premultiplied -> over white

    def crop_of(i, c, r=None):
        r = r or rng
        yx = cover[i][r.integers(len(cover[i]))]
        y0 = int(np.clip(yx[0].item() - c // 2, 0, H - c)) // 2 * 2; x0 = int(np.clip(yx[1].item() - c // 2, 0, W - c)) // 2 * 2
        return y0, x0

    def evaluate(step):
        tot = {k: {'l1': 0.0, 'perc': 0.0, 'lines': 0.0} for k in ('hand', 'learned')}
        er = np.random.default_rng(1)   # the same crops at every eval
        with torch.no_grad():
            for i in hi[::max(1, len(hi) // 24)]:
                y0, x0 = crop_of(i, a.crop, er)
                args, tgt = inputs([i], y0, x0, a.crop)
                for k, learned in (('hand', False), ('learned', True)):
                    o = ups(*args, learned=learned, scale=2, off=2.0)
                    tot[k]['l1'] += (o - tgt).abs().mean().item()
                    tot[k]['perc'] += perceptual(vgg, ow(o), ow(tgt)).mean().item()
                    tot[k]['lines'] += (edges(ow(o)) - edges(ow(tgt))).abs().mean().item()
        n = len(hi[::max(1, len(hi) // 24)])
        return {k: {m: round(v / n, 4) for m, v in d_.items()} for k, d_ in tot.items()}

    print('it 0 held:', evaluate(0), flush=True)
    for it in range(1, a.steps + 1):
        lr = a.lr * min(1, it / 100) * .5 * (1 + math.cos(math.pi * it / a.steps))
        for g in opt.param_groups: g['lr'] = lr
        argss, tgts = [], []
        for i in rng.choice(tr, a.batch):
            y0, x0 = crop_of(i, a.crop)
            ar, t = inputs([i], y0, x0, a.crop); argss.append(ar); tgts.append(t)
        args = [torch.cat([x[k] for x in argss]) for k in range(9)]; tgt = torch.cat(tgts)
        o = ups(*args, scale=2, off=2.0)
        loss = (o - tgt).abs().mean() + a.vgg_w * perceptual(vgg, ow(o), ow(tgt)).mean() + a.lines * (edges(ow(o)) - edges(ow(tgt))).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0 or it == a.steps:
            print(f'it {it} loss {loss.item():.4f} held:', evaluate(it), f'{time.time() - t0:.0f}s', flush=True)
    sd = {k: v.detach().cpu().tolist() for k, v in ups.state_dict().items()}
    (out / 'upsample.json').write_text(json.dumps({'in': H_IN, 'hid': ups.l1.out_features, **sd}))
    # sheets: held-out crops, bilinear | hand-made | learned | film (over white)
    rows = []
    with torch.no_grad():
        for i in hi[np.linspace(0, len(hi) - 1, 6).astype(int)]:
            y0, x0 = crop_of(i, a.crop)
            args, tgt = inputs([i], y0, x0, a.crop)
            R = args[0]
            bil = F.interpolate(R, scale_factor=2, mode='bilinear', align_corners=False)[:, :, 4:4 + a.crop, 4:4 + a.crop]
            ims = [bil, ups(*args, learned=False, scale=2, off=2.0), ups(*args, scale=2, off=2.0), tgt]
            rows.append(torch.cat([ow(x)[0] for x in ims], 2))
    Image.fromarray(torch.cat(rows, 1).permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()).save(out / 'held.png')
    print(f'saved {out}/upsample.json, held.png', flush=True)


if __name__ == '__main__':
    main()
