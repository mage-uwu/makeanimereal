#!/usr/bin/env python3
"""style_torch.py -- the SSM tree shader trained to draw: G-buffers in, free-hand anime keyframes as
targets, no neural texture, a perceptual + adversarial objective instead of L1.

The keyframes (an image editor's free redraws of the render, not pinned to its depth) don't line up
with the geometry pixel for pixel: anime is drawn, not projected. L1 against them pays off the
average of the possible drawings (blur); here the shader is judged by what its picture looks like:
    VGG16 features (relu1_2 .. relu4_3) of the drawing against the keyframe's, composited over white
    a patch discriminator (hinge, spectral norm) that sees the picture and the normals: is this a
    plausible drawing of this geometry
    a weak colour and coverage anchor at 1/8 resolution, where a drawing and the render do agree
No neural texture: its 8 slots carry what any renderer has, fixed, not learned: a code for the
mesh part (6 channels) and the UV coordinates (2).

    python3 style_torch.py --data style.npz --ssm 1,3 --seconds 5400 --out style_run
style.npz (make_style_data.py): px (N, H, W, 17) uint8 as pack_data.py, pitch (N,), held (N,) bool.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import shader_torch
from shader_torch import Net, Optim, forward, DIL_DEFAULT, FINE, FCH
from ssm_tree import SSMTree


def fixed_texture(net, seed=3):
    """The 'texture' as G-buffer facts: channels 0-5 a code per part, 6-7 the texel's u, v."""
    P = net.P
    g = torch.Generator().manual_seed(seed)
    code = torch.randn(P, 6, generator=g)
    code = code / code.norm(dim=1, keepdim=True) * 1.5
    t = net.tex.view(-1, FCH)
    with torch.no_grad():
        t.zero_()
        fine = t[:P * FINE * FINE].view(P, FINE, FINE, FCH)
        c = (torch.arange(FINE, dtype=torch.float32) + .5) / FINE
        fine[..., :6] = code.to(t.device)[:, None, None, :]
        fine[..., 6] = c.to(t.device)[None, None, :]
        fine[..., 7] = c.to(t.device)[None, :, None]


class VGG(torch.nn.Module):
    def __init__(self, random=False):   # random: an untrained VGG16 of the same shape (code checks, no torchvision)
        super().__init__()
        if random:
            L, c = [], 3
            for v in [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]:
                if v == 'M': L.append(torch.nn.MaxPool2d(2))
                else: L += [torch.nn.Conv2d(c, v, 3, padding=1), torch.nn.ReLU()]; c = v
            f = torch.nn.Sequential(*L).eval()
        else:
            import torchvision
            f = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features.eval()
        for p in f.parameters(): p.requires_grad_(False)
        self.f, self.cuts = f, [3, 8, 15, 22]   # relu1_2, relu2_2, relu3_3, relu4_3
        self.register_buffer('m', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('s', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, x):   # x (B, 3, H, W) in [0, 1] -> features
        x = (x - self.m) / self.s; out = []
        for i, layer in enumerate(self.f):
            x = layer(x)
            if i in self.cuts: out.append(x)
            if i >= self.cuts[-1]: break
        return out


def perceptual(vgg, a, b):
    return sum((fa - fb).abs().mean((1, 2, 3)) for fa, fb in zip(vgg(a), vgg(b)))   # per sample


class PatchD(torch.nn.Module):
    """70x70-style PatchGAN on (picture, alpha, normals)."""
    def __init__(self, cin=7, ch=64):
        super().__init__()
        sn = torch.nn.utils.spectral_norm
        L = [sn(torch.nn.Conv2d(cin, ch, 4, 2, 1)), torch.nn.LeakyReLU(.2)]
        for a, b, s in [(ch, ch * 2, 2), (ch * 2, ch * 4, 2), (ch * 4, ch * 8, 1)]:
            L += [sn(torch.nn.Conv2d(a, b, 4, s, 1)), torch.nn.LeakyReLU(.2)]
        L += [sn(torch.nn.Conv2d(ch * 8, 1, 4, 1, 1))]
        self.net = torch.nn.Sequential(*L)

    def forward(self, x): return self.net(x)


def over_white(rgba):   # (B, H, W, 4) straight -> (B, 3, H, W)
    a = rgba[..., 3:4]
    return (rgba[..., :3] * a + 1 - a).permute(0, 3, 1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--out', default='style_run')
    ap.add_argument('--blocks', type=int, default=4); ap.add_argument('--dil', default='1,3,9,27')
    ap.add_argument('--ssm', default='1,3'); ap.add_argument('--tree', default='4,4')
    ap.add_argument('--seconds', type=float, default=5400); ap.add_argument('--batch', type=int, default=6)
    ap.add_argument('--crop', type=int, default=256)
    ap.add_argument('--lr', type=float, default=0.003); ap.add_argument('--adv', type=float, default=0.05)
    ap.add_argument('--adv-from', type=int, default=1500, help='steps of perceptual only, before the discriminator joins')
    ap.add_argument('--anchor', type=float, default=1.0)
    ap.add_argument('--eval-every', type=int, default=500)
    ap.add_argument('--vgg', default='imagenet', help="'random': an untrained one (code checks only)")
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    z = np.load(a.data)
    px_all = torch.from_numpy(z['px']).to(dev); pitch_all = torch.from_numpy(z['pitch']).float().to(dev)
    held = np.nonzero(z['held'])[0]; train = np.nonzero(~z['held'])[0]
    N, H, W, _ = px_all.shape
    P = int(z['parts'])
    print(f'{N} frames {W}x{H} ({len(train)} train, {len(held)} held out), {P} parts', flush=True)

    T, V = map(int, a.tree.split(','))
    dil = [int(v) for v in a.dil.split(',')]
    net = Net(a.blocks, P, dil, T, V, dev); net.init(0); fixed_texture(net)
    shader_torch.SSM.clear()
    for b in map(int, a.ssm.split(',')) if a.ssm else []: shader_torch.SSM[b] = SSMTree(trees=T, levels=V).to(dev)
    ssm_params = [p for m in shader_torch.SSM.values() for p in m.parameters()]
    opt = Optim(net)
    ssm_opt = torch.optim.AdamW(ssm_params, lr=2e-3, weight_decay=0.01) if ssm_params else None
    vgg = VGG(a.vgg == 'random').to(dev); D = PatchD().to(dev)
    d_opt = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(.5, .999))
    leaves = [net.dense, net.W1, net.W2] + ([net.TI, net.TO] if net.T else [])
    nohits = torch.zeros(0, dtype=torch.long, device=dev)
    rng = np.random.default_rng(0)
    # crops around her: centres drawn from her coverage
    cover = [torch.nonzero(px_all[i, ..., 12] > 127).cpu().numpy() for i in range(N)]

    def batch_of(ids, c):
        crops, pit = [], []
        for i in ids:
            yx = cover[i][rng.integers(len(cover[i]))]
            y0 = int(np.clip(yx[0] - c // 2 + rng.integers(-c // 4, c // 4 + 1), 0, H - c)); x0 = int(np.clip(yx[1] - c // 2 + rng.integers(-c // 4, c // 4 + 1), 0, W - c))
            crops.append(px_all[i, y0:y0 + c, x0:x0 + c]); pit.append(pitch_all[i])
        return torch.stack(crops), torch.stack(pit)

    def d_in(rgba, px):   # the discriminator's view: picture over white, alpha, normals
        nrm = px[..., 3:6].float().permute(0, 3, 1, 2) / 255 * 2 - 1
        return torch.cat([over_white(rgba) * 2 - 1, rgba[..., 3:4].permute(0, 3, 1, 2) * 2 - 1, nrm], 1)

    @torch.no_grad()
    def evaluate(step):
        tot = {'perceptual': 0.0, 'anchor': 0.0}
        tiles = []
        for k, i in enumerate(held):
            px = px_all[i:i + 1]
            p, _ = forward(net, px, pitch_all[i:i + 1], True)
            tgt = px[..., 13:17].float() / 255
            tot['perceptual'] += perceptual(vgg, over_white(p), over_white(tgt)).item() / len(held)
            tot['anchor'] += (F.avg_pool2d(over_white(p), 8) - F.avg_pool2d(over_white(tgt), 8)).abs().mean().item() / len(held)
            if k % max(1, len(held) // 6) == 0 and len(tiles) < 6:
                ren = px[0, ..., :3].float() / 255
                row = torch.cat([ren, over_white(tgt)[0].permute(1, 2, 0), over_white(p)[0].permute(1, 2, 0)], 1)
                tiles.append(row)
        if tiles:
            img = torch.cat(tiles, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
            Image.fromarray(img).save(out / f'held_{step:06d}.png')
        return tot

    print(f'SSM blocks {a.ssm}: {sum(p.numel() for p in ssm_params)} parameters; adversarial x{a.adv} from step {a.adv_from}', flush=True)
    t0 = time.time(); it = 0; ema = {}
    while True:
        it += 1
        prog = min(1, (time.time() - t0) / a.seconds)
        sched = min(1, it / 300) * (.1 + .9 * .5 * (1 + math.cos(math.pi * prog)))
        ids = rng.choice(train, a.batch)
        px, pit = batch_of(ids, min(a.crop, H, W))
        tgt = px[..., 13:17].float() / 255
        # generator (the shader)
        for t in leaves: t.requires_grad_(True); t.grad = None
        if ssm_opt: ssm_opt.zero_grad()
        p, _ = forward(net, px, pit, True, credit=True)
        lp = perceptual(vgg, over_white(p), over_white(tgt))
        la = (F.avg_pool2d(over_white(p), 8) - F.avg_pool2d(over_white(tgt), 8)).abs().mean((1, 2, 3)) \
            + (F.avg_pool2d(p[..., 3:4].permute(0, 3, 1, 2), 8) - F.avg_pool2d(tgt[..., 3:4].permute(0, 3, 1, 2), 8)).abs().mean((1, 2, 3))
        use_adv = it > a.adv_from
        lg = -D(d_in(p, px)).mean((1, 2, 3)) if use_adv else torch.zeros_like(lp)
        (lp + a.anchor * la + (a.adv * min(1, (it - a.adv_from) / 500) if use_adv else 0) * lg).sum().backward()
        for t in leaves: t.requires_grad_(False)
        opt.step(it, a.lr * sched, 0.0, a.batch, nohits)
        if ssm_opt:
            for g in ssm_opt.param_groups: g['lr'] = 2e-3 * sched
            torch.nn.utils.clip_grad_norm_(ssm_params, 1.0); ssm_opt.step()
        # discriminator: keyframes real, the shader's drawings fake
        if it > a.adv_from - 500:
            d_opt.zero_grad()
            ld = F.relu(1 - D(d_in(tgt, px))).mean() + F.relu(1 + D(d_in(p.detach(), px))).mean()
            ld.backward(); d_opt.step()
        else:
            ld = torch.zeros(())
        for k, v in [('perc', lp.mean().item()), ('anchor', la.mean().item()), ('g', lg.mean().item()), ('d', ld.item())]:
            ema[k] = v if k not in ema else .98 * ema[k] + .02 * v
        last = time.time() - t0 >= a.seconds
        if it % a.eval_every == 0 or last:
            m = evaluate(it)
            print(f'it {it}  ' + '  '.join(f'{k} {v:.4f}' for k, v in ema.items()) + '  | held: ' + '  '.join(f'{k} {v:.4f}' for k, v in m.items())
                  + f'  {time.time() - t0:.0f}s', flush=True)
            net.save(str(out / 'style.ntst'))
            torch.save([m_.state_dict() for m_ in shader_torch.SSM.values()], out / 'style.ntst.ssm.pt')
        if last: break


if __name__ == '__main__':
    main()
