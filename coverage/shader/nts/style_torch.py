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

    python3 style_torch.py --data style.npz --ssm 1,3 --iters 6000 --out style_run
style.npz (make_style_data.py): px (N, H, W, 17) uint8 as pack_data.py, pitch (N,), held (N,) bool.
bravo.py runs the whole recipe (data, both phases, export); this is one phase.

Speed (GPU): the shader in float32 with TF32 matmuls (exact for its ternary x int8 sums), VGG and the
discriminator in bfloat16 autocast, channels-last; no host sync in a step (losses are read at the
evals). --bench N times N steps and prints the profile instead of training.
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


AMP = torch.bfloat16   # VGG's and the discriminator's autocast type (None: float32 throughout)


def amp():
    return torch.autocast('cuda', dtype=AMP) if AMP is not None and torch.cuda.is_available() else torch.autocast('cpu', enabled=False)


def cl(x): return x.contiguous(memory_format=torch.channels_last)


def perceptual(vgg, a, b):
    """Per sample: the L1 between VGG features of a (the drawing; gradient) and b (the target)."""
    with amp():
        with torch.no_grad(): fb = vgg(cl(b))
        fa = vgg(cl(a))
    return sum((x.float() - y.float()).abs().mean((1, 2, 3)) for x, y in zip(fa, fb))


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


def edges(img):
    """(B, 3, H, W) -> the luma's gradient magnitude, max-pooled 3 x 3: a line a pixel off still meets
    its match (the tolerant line loss)."""
    l = (img * img.new_tensor([.299, .587, .114]).view(1, 3, 1, 1)).sum(1, keepdim=True)
    g = F.pad((l[..., :, 1:] - l[..., :, :-1]).abs(), (0, 1, 0, 0)) + F.pad((l[..., 1:, :] - l[..., :-1, :]).abs(), (0, 0, 0, 1))
    return F.max_pool2d(g, 3, 1, 1)


def over_white(rgba):   # (B, H, W, 4) straight -> (B, 3, H, W)
    a = rgba[..., 3:4]
    return (rgba[..., :3] * a + 1 - a).permute(0, 3, 1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--out', default='style_run')
    ap.add_argument('--blocks', type=int, default=4); ap.add_argument('--dil', default='1,3,9,27')
    ap.add_argument('--ssm', default='1,3'); ap.add_argument('--tree', default='4,4')
    ap.add_argument('--iters', type=int, default=0, help='steps (sets the schedule); 0: --seconds does')
    ap.add_argument('--seconds', type=float, default=5400); ap.add_argument('--batch', type=int, default=6)
    ap.add_argument('--crop', type=int, default=256)
    ap.add_argument('--lr', type=float, default=0.003); ap.add_argument('--adv', type=float, default=0.05)
    ap.add_argument('--adv-from', type=int, default=1500, help='steps of perceptual only, before the discriminator joins')
    ap.add_argument('--anchor', type=float, default=1.0)
    ap.add_argument('--eval-every', type=int, default=500)
    ap.add_argument('--vgg', default='imagenet', help="'random': an untrained one (code checks only)")
    ap.add_argument('--load', help='a style.ntst to start from (its .ssm.pt beside it)')
    ap.add_argument('--coarse-tex', action='store_true', help='learn the coarse texture level (64 x 64 x 8 a part; the fine level stays the fixed part code and UV)')
    ap.add_argument('--lr-tex', type=float, default=0.01)
    ap.add_argument('--lines', type=float, default=0.0, help="the tolerant line loss's weight (edges())")
    ap.add_argument('--amp', default='bf16', choices=['bf16', 'off'], help="VGG and discriminator precision")
    ap.add_argument('--no-tf32', action='store_true', help='the shader in strict float32 (shader_torch.py parity)')
    ap.add_argument('--bench', type=int, default=0, help='time this many steps (after warm-up), print the profile, exit')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--compile', action='store_true', help="torch.compile the shader's training forward (fuses its elementwise work)")
    a = ap.parse_args()
    global AMP
    AMP = torch.bfloat16 if a.amp == 'bf16' else None
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = not a.no_tf32
    torch.backends.cudnn.benchmark = True
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
    net = Net(a.blocks, P, dil, T, V, dev); net.init(a.seed); fixed_texture(net)
    net.tex_grad_levels = (1,) if a.coarse_tex else ()   # the fine level is fixed: no gradient buffer for it
    shader_torch.SSM.clear()
    for b in map(int, a.ssm.split(',')) if a.ssm else []: shader_torch.SSM[b] = SSMTree(trees=T, levels=V).to(dev)
    if a.load:
        net.load(a.load)
        for m, sd in zip(shader_torch.SSM.values(), torch.load(a.load + '.ssm.pt', map_location=dev)): m.load_state_dict(sd)
        print(f'from {a.load}', flush=True)
    ssm_params = [p for m in shader_torch.SSM.values() for p in m.parameters()]
    nfine = P * FINE * FINE   # texels of the fine level (fixed); the coarse level after them
    opt = Optim(net); opt.tex_rows = slice(nfine, None)   # only the coarse level can move
    ssm_opt = torch.optim.AdamW(ssm_params, lr=2e-3, weight_decay=0.01) if ssm_params else None
    vgg = VGG(a.vgg == 'random').to(dev).to(memory_format=torch.channels_last); D = PatchD().to(dev).to(memory_format=torch.channels_last)
    d_opt = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(.5, .999))
    leaves = [net.dense, net.W1, net.W2] + ([net.TI, net.TO] if net.T else []) + ([net.tex] if a.coarse_tex else [])
    nohits = torch.zeros(0, dtype=torch.long, device=dev)
    rng = np.random.default_rng(a.seed)
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
        return cl(torch.cat([over_white(rgba) * 2 - 1, rgba[..., 3:4].permute(0, 3, 1, 2) * 2 - 1, nrm], 1))

    def Dp(x):   # the discriminator's patch scores, float32
        with amp(): return D(x).float()

    @torch.no_grad()
    def evaluate(step):
        tot = {k: torch.zeros((), device=dev) for k in ('perceptual', 'anchor', 'lines')}
        tiles, every = [], max(1, len(held) // 6)
        for k0 in range(0, len(held), 4):   # whole frames, 4 at a time
            ids = held[k0:k0 + 4]
            px = px_all[ids]
            p, _ = forward(net, px, pitch_all[ids], True)
            tgt = px[..., 13:17].float() / 255
            po, to = over_white(p), over_white(tgt)
            tot['perceptual'] += perceptual(vgg, po, to).sum() / len(held)
            tot['anchor'] += (F.avg_pool2d(po, 8) - F.avg_pool2d(to, 8)).abs().mean((1, 2, 3)).sum() / len(held)
            tot['lines'] += (edges(po) - edges(to)).abs().mean((1, 2, 3)).sum() / len(held)
            for j in range(len(ids)):
                if (k0 + j) % every == 0 and len(tiles) < 6:
                    tiles.append(torch.cat([px[j, ..., :3].float() / 255, to[j].permute(1, 2, 0), po[j].permute(1, 2, 0)], 1))
        if tiles:
            img = torch.cat(tiles, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
            Image.fromarray(img).save(out / f'held_{step:06d}.png')
        return {k: v.item() for k, v in tot.items()}

    print(f'SSM blocks {a.ssm}: {sum(p.numel() for p in ssm_params)} parameters; adversarial x{a.adv} from step {a.adv_from}', flush=True)
    ema = {}
    fwd = torch.compile(forward) if a.compile else forward

    def step(it, prog):
        """One training step: the shader against VGG, the anchor, the lines and the discriminator, then
        the discriminator. No host sync: the losses stay on the GPU (ema)."""
        sched = min(1, it / 300) * (.1 + .9 * .5 * (1 + math.cos(math.pi * prog)))
        ids = rng.choice(train, a.batch)
        px, pit = batch_of(ids, min(a.crop, H, W))
        tgt = px[..., 13:17].float() / 255
        # generator (the shader)
        for t in leaves: t.requires_grad_(True); t.grad = None
        if ssm_opt: ssm_opt.zero_grad()
        with torch.profiler.record_function('shader forward'):
            p, hits = fwd(net, px, pit, True, credit=True)
        po, to = over_white(p), over_white(tgt)
        with torch.profiler.record_function('vgg'):
            lp = perceptual(vgg, po, to)
        ll = (edges(po) - edges(to)).abs().mean((1, 2, 3)) if a.lines else torch.zeros_like(lp)
        la = (F.avg_pool2d(po, 8) - F.avg_pool2d(to, 8)).abs().mean((1, 2, 3)) \
            + (F.avg_pool2d(p[..., 3:4].permute(0, 3, 1, 2), 8) - F.avg_pool2d(tgt[..., 3:4].permute(0, 3, 1, 2), 8)).abs().mean((1, 2, 3))
        use_adv = it > a.adv_from
        with torch.profiler.record_function('discriminator (shader)'):
            lg = -Dp(d_in(p, px)).mean((1, 2, 3)) if use_adv else torch.zeros_like(lp)
        with torch.profiler.record_function('backward'):
            (lp + a.anchor * la + a.lines * ll + (a.adv * min(1, (it - a.adv_from) / 500) if use_adv else 0) * lg).sum().backward()
        for t in leaves: t.requires_grad_(False)
        with torch.profiler.record_function('optimisers'):
            # every texel read is passed; the fine level's have no gradient (fixed) and stay put
            opt.step(it, a.lr * sched, a.lr_tex * sched if a.coarse_tex else 0.0, a.batch, hits if a.coarse_tex else nohits)
            if ssm_opt:
                for g in ssm_opt.param_groups: g['lr'] = 2e-3 * sched
                torch.nn.utils.clip_grad_norm_(ssm_params, 1.0); ssm_opt.step()
        # discriminator: keyframes real, the shader's drawings fake
        with torch.profiler.record_function('discriminator step'):
            if it > a.adv_from - 500:
                d_opt.zero_grad()
                ld = F.relu(1 - Dp(d_in(tgt, px))).mean() + F.relu(1 + Dp(d_in(p.detach(), px))).mean()
                ld.backward(); d_opt.step()
            else:
                ld = torch.zeros((), device=dev)
        with torch.no_grad():
            for k, v in [('perc', lp.mean()), ('anchor', la.mean()), ('lines', ll.mean()), ('g', lg.mean()), ('d', ld)]:
                ema[k] = v.detach().clone() if k not in ema else .98 * ema[k] + .02 * v.detach()

    if a.bench:   # warm-up (with the discriminator on), then timed steps and the profile
        sync = torch.cuda.synchronize if dev.type == 'cuda' else (lambda: None)
        for k in range(3): step(a.adv_from + 1 + k, .5)
        sync(); t1 = time.time()
        for k in range(a.bench): step(a.adv_from + 4 + k, .5)
        sync(); dt = (time.time() - t1) / a.bench
        print(f'{dt * 1000:.1f} ms a step ({a.batch} crops of {min(a.crop, H, W)}^2, discriminator on)', flush=True)
        acts = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if dev.type == 'cuda' else [])
        with torch.profiler.profile(activities=acts) as prof:
            for k in range(5): step(a.adv_from + 4 + a.bench + k, .5)
            sync()
        key = 'cuda_time_total' if dev.type == 'cuda' else 'cpu_time_total'
        print(prof.key_averages().table(sort_by=key, row_limit=30), flush=True)
        return

    t0 = time.time(); it = 0
    if a.load:   # the starting point's held-out numbers, the baseline
        print('it 0  | held: ' + '  '.join(f'{k} {v:.4f}' for k, v in evaluate(0).items()), flush=True)
    while True:
        it += 1
        step(it, min(1, it / a.iters) if a.iters else min(1, (time.time() - t0) / a.seconds))
        last = it >= a.iters if a.iters else time.time() - t0 >= a.seconds
        if it % a.eval_every == 0 or last:
            m = evaluate(it)
            print(f'it {it}  ' + '  '.join(f'{k} {v.item():.4f}' for k, v in ema.items()) + '  | held: ' + '  '.join(f'{k} {v:.4f}' for k, v in m.items())
                  + f'  {time.time() - t0:.0f}s', flush=True)
            net.save(str(out / 'style.ntst'))
            torch.save([m_.state_dict() for m_ in shader_torch.SSM.values()], out / 'style.ntst.ssm.pt')
        if last: break

if __name__ == '__main__':
    main()
