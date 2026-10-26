#!/usr/bin/env python3
"""denoise_torch.py -- measure and reduce the noise a drawn film leaves in the nanotss shader, without a
teacher. The same network, data and optimisers as shader_torch.py (imported); the run-time model and
its files are unchanged, everything added here exists only while training.

Where the noise comes from: the film disagrees with itself (each VACE chunk drifts a little in colour,
each drawing sits a few pixels off the geometry, details come and go). Plain L1 fits the average of
those disagreements; the neural texture's per-texel updates let each texel absorb the error of the few
frames that saw it (splotches), misplaced ink averages into haze (softness), and the splotches slide
over her surface as the camera moves (jitter).

Measures (--metrics model.ntst [more.ntst ...]), each on the held-out frames:
  L1        over white, as shader_torch.py's evaluate()
  flicker   the same surface point (part and UV) drawn in two consecutive frames: mean colour change,
            away from part boundaries and her silhouette; also for the film itself and the raw render
  texture   the fine texture's high-frequency energy: mean |texel - its 3 x 3 mean| over the texels
            the frames read, relative to the coarse level's mean |value|
  lines     the drawing's mean gradient over hers in the target (1 = as crisp as the film)

Training (as shader_torch.py: --net-from, --tree, --seconds ...), with:
  --appearance w   per training frame, a 3 x 3 colour matrix and offset on the drawing before the loss
                   (identity at the start, pulled back to it by w): a chunk's colour drift goes there,
                   not into the texture. Zero at run time (it isn't part of the model).
  --align w        per training frame, a coarse grid (--align-grid, cells ~70 px) of 2D offsets that
                   moves the target onto the geometry before the loss, pulled to zero by w; the
                   drawing's few pixels of drift go there, not into soft lines.
  --coarse-first f the first fraction f of the run trains only the coarse texture level (64^2); the fine
                   level starts at zero and only then learns what the coarse one can't carry
  --tex-smooth e   every --smooth-every steps, a Perona-Malik step of rate e on the fine level over the
                   texels the frames read (edge-preserving: differences above --smooth-edge stay)
  --tex-decay d    the fine level's texels decay toward zero by d a step where they were read: detail
                   survives only where many frames keep asking for it
  --lines w        the target's ink (pixels clearly darker than their 7 x 7 neighbourhood, over white)
                   weighs w extra in the colour loss, and around it the drawing's gradients are matched
                   to the target's: lines as crisp as the film's, where the alignment put them

    python3 denoise_torch.py --data data.ntsd --metrics c2.ntst
    python3 denoise_torch.py --data data.ntsd --tree 4,4 --net-from tree1.ntst --seconds 1800 --save dn.ntst \\
        --appearance 1 --align 1 --coarse-first .3 --tex-smooth .2 --tex-decay 1e-4
"""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

import shader_torch as st
from shader_torch import Data, Net, Optim, crops, forward, DIL_DEFAULT, FINE, COARSE, FCH


def over_white(p):
    return p[..., :3] * p[..., 3:4] + 1 - p[..., 3:4]


# ---------------- measures ----------------
def surface_keys(px):
    """(h, w) int64 key per pixel: part and UV at 1/1024; -1 off her, at part boundaries and her silhouette."""
    q = px.long()
    u = q[..., 7] * 256 + q[..., 8]; v = q[..., 9] * 256 + q[..., 10]   # 16-bit UV
    part, cov = q[..., 11], q[..., 12] > 127
    key = (part << 20) | ((u >> 6) << 10) | (v >> 6)
    pf = part.float()[None, None]
    same = (F.max_pool2d(pf, 3, 1, 1) == F.max_pool2d(-pf, 3, 1, 1).neg())[0, 0]
    inner = F.max_pool2d((~cov).float()[None, None], 5, 1, 2)[0, 0] == 0
    return torch.where(cov & same & inner, key, torch.full_like(key, -1))


def flicker(a_img, b_img, ka, kb):
    """Mean |colour change| between two frames at the surface points both see (keys ka, kb)."""
    ka, kb = ka.view(-1), kb.view(-1)
    A, B = a_img.reshape(-1, 3), b_img.reshape(-1, 3)
    mb = kb >= 0
    ub, inv = torch.unique(kb[mb], return_inverse=True)
    s = torch.zeros(len(ub), 3, device=A.device).index_add_(0, inv, B[mb])
    n = torch.zeros(len(ub), device=A.device).index_add_(0, inv, torch.ones_like(inv, dtype=torch.float))
    cb = s / n[:, None]
    ma = ka >= 0
    pos = torch.searchsorted(ub, ka[ma]).clamp(max=len(ub) - 1)
    hit = ub[pos] == ka[ma]
    if hit.sum() == 0: return float('nan'), 0
    return (A[ma][hit] - cb[pos[hit]]).abs().mean().item(), int(hit.sum())


def grad_mag(img, m):
    gx = (img[:, 1:] - img[:, :-1]).abs().mean(-1)[:-1]
    gy = (img[1:] - img[:-1]).abs().mean(-1)[:, :-1]
    return ((gx + gy) * m[:-1, :-1]).sum().item()


@torch.no_grad()
def texture_stats(net, data, ids, trees):
    """Fine-level high-frequency energy over the texels the frames read, relative to the coarse level."""
    P = net.P
    seen = torch.zeros(P * FINE * FINE, dtype=torch.bool, device=net.dev)
    for i in ids:
        _, hits = st.inputs(net, data.px[i:i + 1].to(net.dev), data.pitch_t[i:i + 1].to(net.dev))
        seen[hits[hits < P * FINE * FINE]] = True
    fine = net.tex[:P * FINE * FINE * FCH].view(P, FINE, FINE, FCH).permute(0, 3, 1, 2)
    hf = (fine - F.avg_pool2d(fine, 3, 1, 1, count_include_pad=False)).abs().mean(1).view(-1)
    coarse = net.tex[P * FINE * FINE * FCH:].abs().mean().item()
    return (hf[seen].mean().item() / max(coarse, 1e-8)), seen


@torch.no_grad()
def metrics(net, data, trees, pairs, held):
    l1 = st.evaluate(net, data, held, trees)
    fl, fl_t, fl_r, gp, gt = [], [], [], 0.0, 0.0
    for i in pairs:
        px = data.px[i:i + 2].to(net.dev)
        p, _ = forward(net, px, data.pitch_t[i:i + 2].to(net.dev), trees)
        k0, k1 = surface_keys(px[0]), surface_keys(px[1])
        fl.append(flicker(over_white(p[0]), over_white(p[1]), k0, k1)[0])
        tg = px[..., 13:17].float() / 255
        fl_t.append(flicker(over_white(tg[0]), over_white(tg[1]), k0, k1)[0])
        rc = px[..., 0:3].float() / 255
        fl_r.append(flicker(rc[0], rc[1], k0, k1)[0])
        m = (px[0, ..., 12] > 127).float()
        gp += grad_mag(over_white(p[0]), m); gt += grad_mag(over_white(tg[0]), m)
    tex, _ = texture_stats(net, data, held, trees)
    return {'L1': l1, 'flicker': float(np.nanmean(fl)), 'flicker film': float(np.nanmean(fl_t)),
            'flicker render': float(np.nanmean(fl_r)), 'texture': tex, 'lines': gp / max(gt, 1e-8)}


# ---------------- training extras ----------------
class Nuisance:
    """Per training frame: colour matrix and offset (appearance), coarse offset grid (alignment)."""
    def __init__(self, N, H, W, grid, dev):
        self.A = torch.eye(3, device=dev).repeat(N, 1, 1).requires_grad_()
        self.b = torch.zeros(N, 3, device=dev, requires_grad=True)
        gh, gw = grid
        self.G = torch.zeros(N, 2, gh, gw, device=dev, requires_grad=True)   # pixels
        self.H, self.W = H, W
        self.opt = torch.optim.Adam([self.A, self.b, self.G], lr=1e-3)

    def colour(self, p, fis):
        rgb = torch.einsum('bij,bhwj->bhwi', self.A[fis], p[..., :3]) + self.b[fis][:, None, None, :]
        return torch.cat([rgb, p[..., 3:]], -1)

    def target(self, data, fis, cs, ch, cw):
        """The crops' targets, moved by each frame's offset field (bilinear, edges clamped)."""
        dev = self.G.device
        full = data.px[fis][..., 13:17].float().permute(0, 3, 1, 2) / 255          # (B, 4, H, W)
        disp = F.interpolate(self.G[fis], size=(self.H, self.W), mode='bilinear', align_corners=True)
        ys = torch.arange(self.H, device=dev, dtype=torch.float)[:, None].expand(self.H, self.W)
        xs = torch.arange(self.W, device=dev, dtype=torch.float)[None, :].expand(self.H, self.W)
        out = []
        for k, (fi, cy, cx) in enumerate(cs):
            gx = xs[cy:cy + ch, cx:cx + cw] + disp[k, 0, cy:cy + ch, cx:cx + cw]
            gy = ys[cy:cy + ch, cx:cx + cw] + disp[k, 1, cy:cy + ch, cx:cx + cw]
            grid = torch.stack([gx / (self.W - 1) * 2 - 1, gy / (self.H - 1) * 2 - 1], -1)[None]
            out.append(F.grid_sample(full[k:k + 1], grid, mode='bilinear', padding_mode='border', align_corners=True)[0])
        return torch.stack(out).permute(0, 2, 3, 1)                                 # (B, ch, cw, 4)

    def reg(self, fis, wa, wg):
        r = 0
        if wa: r = r + wa * (((self.A[fis] - torch.eye(3, device=self.A.device)) ** 2).sum() + (self.b[fis] ** 2).sum())
        if wg:
            G = self.G[fis]
            smooth = ((G[..., 1:, :] - G[..., :-1, :]) ** 2).sum() + ((G[..., :, 1:] - G[..., :, :-1]) ** 2).sum()
            r = r + wg * 1e-3 * ((G ** 2).sum() + smooth)
        return r


def loss_to(p, tgt, lines=0.0):
    """loss_of against a given target (B, h, w, 4): alpha L1 plus colour L1 where covered; with lines,
    the target's ink weighs extra and the drawing's gradients near it follow the target's."""
    ta = tgt[..., 3:4]
    l = (p[..., 3:4] - ta).abs().flatten(1).mean(1) + ((p[..., :3] - tgt[..., :3]) * ta).abs().flatten(1).mean(1)
    if lines > 0:
        ink = ink_of(tgt)
        l = l + lines * ((p[..., :3] - tgt[..., :3]).abs().mean(-1) * ink).flatten(1).mean(1)
        near = F.max_pool2d(ink[:, None], 5, 1, 2)[:, 0]
        po, to = over_white(p), over_white(tgt)
        gx = lambda x: x[:, :, 1:] - x[:, :, :-1]
        gy = lambda x: x[:, 1:] - x[:, :-1]
        e = ((gx(po) - gx(to)).abs().mean(-1) * near[:, :, 1:]).flatten(1).mean(1) + ((gy(po) - gy(to)).abs().mean(-1) * near[:, 1:]).flatten(1).mean(1)
        l = l + lines * e
    return l


def ink_of(tgt):
    """(B, h, w) in [0, 1]: how clearly each pixel is ink, darker than its neighbourhood (over white)."""
    lum = over_white(tgt).mean(-1)[:, None]
    local = F.avg_pool2d(lum, 7, 1, 3, count_include_pad=False)
    return ((local - lum - 0.04) / 0.08).clamp(0, 1)[:, 0] * tgt[..., 3]


@torch.no_grad()
def smooth_fine(net, seen, rate, edge):
    """One Perona-Malik step on the fine level, over the texels the frames read (and only between those)."""
    P = net.P
    t = net.tex[:P * FINE * FINE * FCH].view(P, FINE, FINE, FCH)
    s = seen.view(P, FINE, FINE, 1).float()
    acc = torch.zeros_like(t)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        n = torch.roll(t, (dy, dx), (1, 2)); ns = torch.roll(s, (dy, dx), (1, 2))
        d = n - t
        acc += torch.exp(-(d / edge) ** 2) * d * ns
    t += rate * 0.25 * acc * s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True)
    ap.add_argument('--metrics', nargs='+', help='models to measure (nothing trains)')
    ap.add_argument('--pairs', type=int, default=36, help='consecutive frame pairs for the flicker measure')
    ap.add_argument('--load'); ap.add_argument('--net-from'); ap.add_argument('--save', default='denoised.ntst')
    ap.add_argument('--tree', default='4,4'); ap.add_argument('--blocks', type=int, default=8)
    ap.add_argument('--dil', help="the blocks' dilations, e.g. 1,3,9,27 (default: the standard 1,2,4,8,16,...)")
    ap.add_argument('--ssm', help="blocks whose FFN is nanoTSS's SSM tree (ssm_tree.py), e.g. 1,3; saved beside --save as .ssm.pt")
    ap.add_argument('--seconds', type=float, default=1800); ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--crop', type=int, default=96)
    ap.add_argument('--lr', type=float, default=0.003); ap.add_argument('--lr-tex', type=float, default=0.01)
    ap.add_argument('--eval-every', type=int, default=4000)
    ap.add_argument('--appearance', type=float, default=0); ap.add_argument('--align', type=float, default=0)
    ap.add_argument('--align-grid', default='9x5', help='rows x columns of the offset grid')
    ap.add_argument('--coarse-first', type=float, default=0)
    ap.add_argument('--tex-smooth', type=float, default=0); ap.add_argument('--smooth-every', type=int, default=10)
    ap.add_argument('--smooth-edge', type=float, default=0.05)
    ap.add_argument('--tex-decay', type=float, default=0)
    ap.add_argument('--lines', type=float, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    dev = torch.device(a.device)
    T, V = (int(v) for v in a.tree.split(','))
    dil = [int(v) for v in a.dil.split(',')] if a.dil else DIL_DEFAULT
    assert len(dil) >= a.blocks, '--dil: one per block'
    data = Data(a.data, dev if dev.type == 'cuda' else torch.device('cpu'))
    held = [i for i in range(data.N) if i % 10 == 5]
    # consecutive pairs in one wave (the pack is the top wave then the bottom; a jump between them is big)
    pairs = [i for i in held if i + 1 < data.N and abs(data.pitch[i + 1] - data.pitch[i]) < 1][::max(1, len(held) // a.pairs)][:a.pairs]

    def ssm_blocks(path=None):   # --ssm: those blocks' SSM trees, registered with forward() (shader_torch.SSM)
        import shader_torch
        from ssm_tree import SSMTree
        shader_torch.SSM.clear()
        if not a.ssm: return []
        for b in map(int, a.ssm.split(',')): shader_torch.SSM[b] = SSMTree(trees=T, levels=V).to(dev)
        if path: [m.load_state_dict(sd) for m, sd in zip(shader_torch.SSM.values(), torch.load(path + '.ssm.pt', map_location=dev))]
        return [p for m in shader_torch.SSM.values() for p in m.parameters()]

    if a.metrics:
        for path in a.metrics:
            ssm_blocks(path)
            net = Net(a.blocks, data.P, dil, T, V, dev); net.init(0); net.load(path)
            m = metrics(net, data, True, pairs, held[::4])
            print(f'{path}: ' + '  '.join(f'{k} {v:.4f}' for k, v in m.items()), flush=True)
        return

    net = Net(a.blocks, data.P, dil, T, V, dev); net.init(0)
    if a.load: net.load(a.load)
    if a.net_from: net.load(a.net_from, tex=False)
    P, nfine = net.P, net.P * FINE * FINE
    if a.coarse_first > 0 and not a.load:
        net.tex[:nfine * FCH] = 0
    train_ids = [i for i in range(data.N) if i % 10 != 5]
    ch, cw = min(a.crop, data.H), min(a.crop, data.W)
    nui = Nuisance(data.N, data.H, data.W, tuple(int(x) for x in a.align_grid.split('x')), dev) if (a.appearance or a.align) else None
    seen = None
    if a.tex_smooth > 0:
        _, seen = texture_stats(net, data, train_ids[::3], True)
    print(f'{data.N} frames {data.W}x{data.H}, {P} parts; appearance {a.appearance} align {a.align} '
          f'coarse-first {a.coarse_first} smooth {a.tex_smooth} decay {a.tex_decay} lines {a.lines}', flush=True)

    leaves = net.leaves(); opt = Optim(net)
    ssm_params = ssm_blocks(a.load if a.load and a.ssm else None)
    ssm_opt = torch.optim.AdamW(ssm_params, lr=2e-3, weight_decay=0.01) if ssm_params else None
    if ssm_params: print(f'SSM tree blocks {a.ssm}: {sum(p.numel() for p in ssm_params)} parameters', flush=True)
    t0 = time.time(); ema = None; it = 0
    while True:
        it += 1
        el = time.time() - t0
        prog = min(1, el / a.seconds)
        sched = min(1, it / 300) * (.1 + .9 * .5 * (1 + math.cos(math.pi * prog)))
        for t in leaves: t.requires_grad_(True); t.grad = None
        cs = crops(it, a.batch, train_ids, data.H, data.W, ch, cw)
        fis = torch.tensor([fi for fi, _, _ in cs], device=dev)
        px = torch.stack([data.px[fi, cy:cy + ch, cx:cx + cw] for fi, cy, cx in cs]).to(dev)
        pitch = torch.tensor([data.pitch[fi] for fi, _, _ in cs], device=dev)
        if ssm_opt: ssm_opt.zero_grad()
        p, hits = forward(net, px, pitch, True, credit=True)
        if nui:
            nui.opt.zero_grad()
            tgt = nui.target(data, fis, cs, ch, cw) if a.align else px[..., 13:17].float() / 255
            pc = nui.colour(p, fis) if a.appearance else p
            l = loss_to(pc, tgt, a.lines)
            (l.sum() + nui.reg(fis, a.appearance, a.align)).backward()
        else:
            l = loss_to(p, px[..., 13:17].float() / 255, a.lines)
            l.sum().backward()
        for t in leaves: t.requires_grad_(False)
        if prog < a.coarse_first:
            hits = hits[hits >= nfine]   # the fine level waits
        opt.step(it, a.lr * sched, a.lr_tex * sched, a.batch, hits)
        if ssm_opt:
            for g in ssm_opt.param_groups: g['lr'] = 2e-3 * sched
            torch.nn.utils.clip_grad_norm_(ssm_params, 1.0); ssm_opt.step()
        if nui:
            for g in nui.opt.param_groups: g['lr'] = 1e-3 * sched
            nui.opt.step()
        with torch.no_grad():
            if a.tex_decay > 0 and prog >= a.coarse_first:
                fh = torch.unique(hits[hits < nfine])
                tv = net.tex.view(-1, FCH); tv[fh] *= (1 - a.tex_decay)
            if a.tex_smooth > 0 and prog >= a.coarse_first and it % a.smooth_every == 0:
                smooth_fine(net, seen, a.tex_smooth * sched, a.smooth_edge)
        loss = l.sum().item() / a.batch
        ema = loss if ema is None else .98 * ema + .02 * loss
        last = time.time() - t0 >= a.seconds
        if it % a.eval_every == 0 or last:
            m = metrics(net, data, True, pairs[:12], held[::8])
            print(f'it {it}  loss {ema:.4f}  ' + '  '.join(f'{k} {v:.4f}' for k, v in m.items() if 'film' not in k and 'render' not in k)
                  + f'  {time.time() - t0:.0f}s', flush=True)
            net.save(a.save)
            if ssm_params:
                import shader_torch
                torch.save([m.state_dict() for m in shader_torch.SSM.values()], a.save + '.ssm.pt')
        if last: break
    if nui:
        torch.save({'A': nui.A.detach().cpu(), 'b': nui.b.detach().cpu(), 'G': nui.G.detach().cpu()}, a.save + '.nuisance.pt')
    print(f'saved {a.save}')


if __name__ == '__main__':
    main()
