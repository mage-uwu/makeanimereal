#!/usr/bin/env python3
"""shader_torch.py -- shader_nts.c's model and training in PyTorch, for GPUs: the same network, the
same optimisers, the same checkpoints (NTSW, NTST) and data (NTSD), so a model moves freely between
the two trainers and export_nts.py.

Faithful to shader_nts.c, to float rounding:
  inputs   decode() and tex_one(): bilinear, align_corners=False, zero padding, per part
  embed    dense 18 -> 64
  blocks   the engine's ternary stencil mixer (codes: latents thresholded at +-0.5, straight
           through; gate tanh), then either the dense FFN (tss_ternary rows, quant8 inputs and
           hidden, GELU, straight through) or --tree's fast feedforward trees (fff_fwd's path sum
           and fff_bwd's routing credit, every alternative walked)
  head     dense 64 -> 4, sigmoid; loss_grad's L1 on coverage plus L1 on colour where covered
  steps    the batch summed, the global gradient norm clipped at 1/B, AdamW (adam()) on the dense
           vector, Muon (muon_apply: 32-row blocks, or a tree each) on the ternary matrices, lazy
           Adam on the texels the step sampled; train_step's crops (urand) and its schedule
--distill: the trees learn a loaded dense model's FFNs, nothing else moving (shader_nts.c --distill).

    python3 shader_torch.py --data data.bin --load run2.ntsw --tree 4,4 --seconds 3000 --save m.ntst
    python3 shader_torch.py --data held.bin --load m.ntst --tree 4,4 --compare-ref ref.bin   # forward vs C
    python3 shader_torch.py --data data.bin --load m.ntsw --compare-grads grads.bin            # vs --dump-grads
Data: pack_data.py's NTSD, e.g. `python3 pack_data.py dirs... --out /dev/stdout | shader_torch.py --data -`.
"""
import argparse
import math
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

DW, KT, CIN, FCH, FINE, COARSE, TCG, MAXB = 64, 9, 18, 8, 512, 64, 4, 16
TC_SIZE = (KT + TCG + 1) * DW
DIL_DEFAULT = [1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16]
torch.backends.cuda.matmul.allow_tf32 = False   # float32 sums, as the C trainer's
torch.backends.cudnn.allow_tf32 = False


# ---------------- data ----------------
class Data:
    """NTSD: 'NTSD', N, H, W, P, pitch[N], then N x H x W x 17 bytes (pack_data.py)."""
    def __init__(self, path, device):
        f = sys.stdin.buffer if path == '-' else open(path, 'rb')
        assert f.read(4) == b'NTSD', 'not NTSD'
        self.N, self.H, self.W, self.P = struct.unpack('<4i', f.read(16))
        self.pitch = np.frombuffer(f.read(4 * self.N), np.float32).copy()
        n = self.N * self.H * self.W * 17
        buf = bytearray(n); mv = memoryview(buf); got = 0
        while got < n:
            k = f.readinto(mv[got:])
            if not k: raise SystemExit('short data')
            got += k
        px = torch.frombuffer(buf, dtype=torch.uint8).view(self.N, self.H, self.W, 17)
        try: self.px = px.to(device)          # on the GPU when it fits
        except RuntimeError: self.px = px.pin_memory() if device.type == 'cuda' else px
        self.pitch_t = torch.from_numpy(self.pitch).to(device)


# ---------------- train_step's random crops (urand, float32 as in C) ----------------
M64 = (1 << 64) - 1
def urand(s):
    s ^= s >> 12; s ^= (s << 25) & M64; s ^= s >> 27
    return s, np.float32(np.float32(((s * 2685821657736338717) & M64) >> 11) * np.float32(1.0 / 9007199254740992.0))
def crops(it, B, train_ids, H, W, ch, cw):
    out = []
    for b in range(B):
        s = (it * 0x9E3779B97F4A7C15 + b * 7919 + 1) & M64
        s, u = urand(s); fi = train_ids[int(np.float32(u * np.float32(len(train_ids))))]
        s, u = urand(s); cy = int(np.float32(u * np.float32(H - ch + 1)))
        s, u = urand(s); cx = int(np.float32(u * np.float32(W - cw + 1)))
        out.append((fi, cy, cx))
    return out


# ---------------- the model ----------------
class Net:
    def __init__(self, NB, P, dil, trees=0, levels=4, device='cpu'):
        self.NB, self.P, self.DIL, self.T, self.V = NB, P, list(dil[:NB]), trees, levels
        self.TN = (1 << levels) - 1 if trees else 0
        self.dev = torch.device(device)
        o = 0; self.o = {}
        def take(name, n):
            nonlocal o; self.o[name] = (o, n); o += n
        take('We', DW * CIN); take('be', DW)
        for b in range(NB):
            take(f'tc{b}', TC_SIZE); take(f'b1{b}', DW); take(f'b2{b}', DW)
            if trees: take(f'tb{b}', trees * self.TN)
        take('Wh', 4 * DW); take('bh', 4)
        self.n_dense = o
        self.n_tex = P * (FINE * FINE + COARSE * COARSE) * FCH
        z = lambda *s: torch.zeros(*s, device=self.dev)
        self.dense = z(self.n_dense)
        self.W1, self.W2 = z(NB, DW, DW), z(NB, DW, DW)                       # dense FFNs (the teachers in --distill)
        rows = trees * self.TN
        self.TI, self.TO = (z(NB, rows, DW), z(NB, rows, DW)) if trees else (None, None)
        self.tex = z(self.n_tex)

    def view(self, name, *shape):
        a, n = self.o[name]
        v = self.dense[a:a + n]
        return v.view(*shape) if shape else v

    def leaves(self):
        out = [self.dense, self.W1, self.W2, self.tex]
        if self.T: out += [self.TI, self.TO]
        return out

    # -- checkpoints (shader_nts.c save_net / load_net) --
    def load(self, path, tex=True):
        """tex=False: the network only (another character's model, any number of parts; the texture stays)."""
        raw = open(path, 'rb').read()
        mg = raw[:4]; NB, dw, P, kt = struct.unpack_from('<4i', raw, 4)
        assert (NB, dw, P if tex else self.P, kt) == (self.NB, DW, self.P, KT), f'{path}: other shape'
        dil = list(struct.unpack_from(f'<{MAXB}i', raw, 20))[:NB]
        assert dil == self.DIL, f'{path}: other dilations'
        off = 20 + 4 * MAXB
        f32 = lambda n: torch.from_numpy(np.frombuffer(raw, np.float32, n, off).copy()).to(self.dev)
        if mg == b'NTST':
            T, V = struct.unpack_from('<2i', raw, off); off += 8
            assert (T, V) == (self.T, self.V), f'{path}: trees {T},{V} (use --tree {T},{V})'
            self.dense.copy_(f32(self.n_dense)); off += 4 * self.n_dense
            rows = self.T * self.TN
            for b in range(NB):
                self.TI[b] = f32(rows * DW).view(rows, DW); off += 4 * rows * DW
                self.TO[b] = f32(rows * DW).view(rows, DW); off += 4 * rows * DW
        elif mg == b'NTSW':
            dense_layout = Net(NB, P, self.DIL, 0, device='cpu')      # a dense file's offsets
            old = f32(dense_layout.n_dense); off += 4 * dense_layout.n_dense
            for name, (a, n) in dense_layout.o.items(): self.dense[self.o[name][0]:self.o[name][0] + n] = old[a:a + n]
            for b in range(NB):
                self.W1[b] = f32(DW * DW).view(DW, DW); off += 4 * DW * DW
                self.W2[b] = f32(DW * DW).view(DW, DW); off += 4 * DW * DW
        else:
            raise SystemExit(f'{path}: not NTSW or NTST')
        n_tex = P * (FINE * FINE + COARSE * COARSE) * FCH
        if tex: self.tex.copy_(f32(self.n_tex))
        off += 4 * n_tex
        assert off == len(raw)

    def save(self, path):
        cpu = lambda t: t.detach().float().cpu().numpy().astype('<f4').tobytes()
        with open(path, 'wb') as f:
            f.write(b'NTST' if self.T else b'NTSW')
            f.write(struct.pack('<4i', self.NB, DW, self.P, KT))
            f.write(struct.pack(f'<{MAXB}i', *(self.DIL + DIL_DEFAULT[self.NB:])))
            if self.T: f.write(struct.pack('<2i', self.T, self.V))
            f.write(cpu(self.dense))
            for b in range(self.NB):
                if self.T: f.write(cpu(self.TI[b])); f.write(cpu(self.TO[b]))
                else: f.write(cpu(self.W1[b])); f.write(cpu(self.W2[b]))
            f.write(cpu(self.tex))

    def init(self, seed=0):
        """A fresh model, distributed as net_init() (its own random stream, not the C one's)."""
        g = torch.Generator(device='cpu').manual_seed(seed)
        r = lambda *s: torch.randn(*s, generator=g).to(self.dev)
        self.view('We').copy_(r(DW * CIN) / math.sqrt(CIN))
        for b in range(self.NB):
            tc = self.view(f'tc{b}')
            tc[:(KT + TCG) * DW] = (torch.rand((KT + TCG) * DW, generator=g) * 2 - 1).to(self.dev)
            self.W1[b] = r(DW, DW) / math.sqrt(DW); self.W2[b] = r(DW, DW) * 0.02
            if self.T: self.TI[b] = r(self.T * self.TN, DW) / math.sqrt(DW); self.TO[b] = r(self.T * self.TN, DW) * 0.02
        self.view('Wh').copy_(r(4 * DW) * 0.02)
        P = self.P
        self.tex[:P * FINE * FINE * FCH] = r(P * FINE * FINE * FCH) * 0.01
        self.tex[P * FINE * FINE * FCH:] = r(P * COARSE * COARSE * FCH) * 0.1

    def tex_to_8bit(self):
        """export_nts.py's 8 bits per channel per level (shader_nts.c --tex8)."""
        P = self.P
        with torch.no_grad():
            for lev, (a, S) in enumerate([(0, FINE), (P * FINE * FINE * FCH, COARSE)]):
                t = self.tex[a:a + P * S * S * FCH].view(-1, FCH)
                lo, hi = t.min(0).values, t.max(0).values
                st = (hi - lo) / 255; st[st <= 0] = 1
                t.copy_(lo + torch.round((t - lo) / st) * st)


# ---------------- quantisers (straight through) ----------------
def ternary(w):
    """tss_ternary, one group a row: mean |w| scale, codes round(clip(w / s)); straight through."""
    s = w.detach().abs().mean(-1, keepdim=True).clamp_min(1e-8)
    q = torch.round((w.detach() / s).clamp(-1, 1)) * s
    return w + (q - w).detach()
def quant8(x, codes=False):
    """quant8 on the last axis: absmax, 127 steps, ties to even, clamped; straight through.
    codes: also the integer codes (as floats) and the scale 1 / (127 / max)."""
    m = x.detach().abs().amax(-1, keepdim=True).clamp_min(1e-5)
    sc = 127 / m
    c = torch.round(x.detach() * sc).clamp(-128, 127)
    xq = x + (c / sc - x).detach()
    return (xq, c, 1 / sc) if codes else xq
def tc_codes(p):
    """tc_pack: latents above 0.5 are +1, below -0.5 -1; straight through to the latents."""
    q = (p.detach() > .5).float() - (p.detach() < -.5).float()
    return p + (q - p.detach()).detach()


def gelu(x): return 0.5 * x * (1 + torch.erf(x * 0.70710678118654752440))
def dgelu(x): return 0.5 * (1 + torch.erf(x * 0.70710678118654752440)) + x * torch.exp(-0.5 * x * x) * 0.39894228040143267794


class GeluAS(torch.autograd.Function):
    """shader_nts.c's gelu_n (the dense FFN's): erf by Abramowitz-Stegun 7.1.26 sharing exp(-x^2/2) with
    the derivative, which is the exact form's with that erf (not the polynomial's own derivative)."""
    @staticmethod
    def forward(ctx, v):
        z = (v * .70710678).abs(); e = torch.exp(-.5 * v * v); t = 1 / (z * .3275911 + 1)
        p = (((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - .284496736) * t + .254829592
        er = 1 - p * t * e; er = torch.where(v < 0, -er, er)
        c = .5 * (1 + er)
        ctx.save_for_backward(v * e * .39894228 + c)
        return v * c
    @staticmethod
    def backward(ctx, g):
        return g * ctx.saved_tensors[0]


class Trees(torch.autograd.Function):
    """fff_fwd / fff_bwd for every token at once. x: (N, 64) quant8 inputs, xc / xs their codes and
    scales; QI, QO: (T*TN, 64) ternary; bias: (T*TN,). Returns the path sum (N, 64). credit: the
    routing credit (training)."""
    @staticmethod
    def forward(ctx, x, QI, bias, QO, xc, xs, T, V, TN, temp, credit):
        N, dev = x.shape[0], x.device
        # every node's s (only visited ones count), exactly as shader_nts.c and infer.c: the integer sum
        # of codes (exact in float32), x w_in's row scale, x the input's scale, + the bias
        qi = QI.detach()
        S = ((xc @ torch.sign(qi).t()) * qi.abs().amax(1)) * xs + bias
        nodes = torch.empty(N, T, V, dtype=torch.long, device=dev)
        a = torch.zeros(N, T, dtype=torch.long, device=dev)
        base = torch.arange(T, device=dev) * TN
        for j in range(V):
            nodes[:, :, j] = base + a
            s = S.gather(1, nodes[:, :, j])
            a = 2 * a + 1 + (s > 0).long()
        s_on = S.gather(1, nodes.view(N, -1)).view(N, T, V)
        alt = []                                               # below each decision but the last: the other child's walk
        if credit:
            for j in range(V - 1):
                a2 = 2 * (nodes[:, :, j] - base) + 2 - (s_on[:, :, j] > 0).long()
                walk = []
                for jj in range(j + 1, V):
                    r = base + a2; walk.append(r)
                    a2 = 2 * a2 + 1 + (S.gather(1, r) > 0).long()
                alt.append(torch.stack(walk, -1))              # (N, T, V-1-j)
        C = torch.zeros_like(S).scatter_(1, nodes.view(N, -1), gelu(s_on).view(N, -1))
        ctx.save_for_backward(x, QI, QO, S, nodes, s_on, C, *alt)
        ctx.cfg = (T, V, TN, temp, credit)
        return C @ QO

    @staticmethod
    def backward(ctx, gy):
        x, QI, QO, S, nodes, s_on, C, *alt = ctx.saved_tensors
        T, V, TN, temp, credit = ctx.cfg
        N = x.shape[0]
        GO = gy @ QO.t()                                       # each node's w_out . gy
        go = GO.gather(1, nodes.view(N, -1)).view(N, T, V)
        on, ds = gelu(s_on) * go, dgelu(s_on) * go
        if credit:   # each decision: what its subtree gave against what the other one would have
            ds = ds.clone()
            for j in range(V - 1):
                o = on[:, :, j + 1:].sum(-1)
                rows = alt[j].view(N, -1)
                av = (gelu(S.gather(1, rows)) * GO.gather(1, rows)).view(N, T, -1).sum(-1)
                sj = s_on[:, :, j]; sg = torch.sigmoid(sj.abs() / temp)
                ds[:, :, j] += torch.where(sj > 0, 1.0, -1.0) * (o - av) * sg * (1 - sg) / temp
        dS = torch.zeros_like(S).scatter_(1, nodes.view(N, -1), ds.view(N, -1))
        return dS @ QI, dS.t() @ x, dS.sum(0), C.t() @ gy, None, None, None, None, None, None, None


def inputs(net, px, pitch):
    """decode() and the neural texture: px (B, h, w, 17) uint8, pitch (B,) degrees -> (B, h, w, 18),
    the texels each pixel read (for the lazy texture Adam)."""
    q = px.float()
    B, h, w, _ = px.shape
    col = q[..., 0:3] / 255
    nrm = q[..., 3:6] / 255 * 2 - 1
    dep = q[..., 6:7] / 255
    mask = (px[..., 12] > 127)
    u = (q[..., 7] + q[..., 8] / 255) / 255
    v = (q[..., 9] + q[..., 10] / 255) / 255
    part = torch.where(px[..., 11].long() < net.P, px[..., 11].long(), 0)
    feat = torch.zeros(B, h, w, FCH, device=px.device)
    tex = net.tex.view(-1, FCH)
    hits = []
    for lev, (base, S) in enumerate([(0, FINE), (net.P * FINE * FINE, COARSE)]):
        fx_, fy_ = u * S - 0.5, v * S - 0.5
        x0, y0 = torch.floor(fx_), torch.floor(fy_)
        fx, fy = fx_ - x0, fy_ - y0
        x0, y0 = x0.long(), y0.long()
        for j in (0, 1):
            for i in (0, 1):
                x, y = x0 + i, y0 + j
                ok = mask & (x >= 0) & (y >= 0) & (x < S) & (y < S)
                wgt = (fx if i else 1 - fx) * (fy if j else 1 - fy)
                idx = base + part * S * S + y.clamp(0, S - 1) * S + x.clamp(0, S - 1)
                idx = torch.where(ok, idx, 0)
                feat = feat + (wgt * ok)[..., None] * tex.index_select(0, idx.view(-1)).view(B, h, w, FCH)
                hits.append(idx[ok])
    p = pitch.float() * 0.017453292519943295
    pit = torch.stack([torch.sin(p), torch.cos(p)], -1)[:, None, None, :].expand(B, h, w, 2)
    return torch.cat([col, nrm, dep, mask.float()[..., None], feat, pit], -1), torch.cat(hits)


SSM = {}   # block -> ssm_tree.SSMTree: that block's FFN is nanoTSS's SSM tree (denoise_torch.py --ssm)


def forward(net, px, pitch, trees=False, credit=False, temp=1.0, keep=None):
    """shader_nts.c's forward(): px (B, h, w, 17) -> straight RGBA (B, h, w, 4), and the texels read.
    keep: a list to receive each block's (t, quant8(t), FFN output) (the dense teacher, --distill)."""
    x_in, hits = inputs(net, px, pitch)
    B, h, w, _ = x_in.shape
    x = x_in @ net.view('We', DW, CIN).t() + net.view('be')
    x = x.permute(0, 3, 1, 2)                                  # (B, 64, h, w)
    for b in range(net.NB):
        tc = net.view(f'tc{b}')
        q = tc_codes(tc[:(KT + TCG) * DW]).view(KT + TCG, DW)
        gate = torch.tanh(tc[(KT + TCG) * DW:])
        dil = net.DIL[b]
        # depthwise 3 x 3 at the dilation, zero padding: tap k = (dy + 1) * 3 + (dx + 1)
        dw = F.conv2d(x, q[:KT].t().reshape(DW, 1, 3, 3), padding=dil, dilation=dil, groups=DW) * (1 / 3)
        # the 4-channel group mix: channel i takes sum_j q[KT + j][i] dw[4 (i / 4) + j]
        mid = F.conv2d(dw, q[KT:].t().reshape(DW, TCG, 1, 1), groups=DW // TCG) * 0.5
        t = x + gate.view(1, DW, 1, 1) * mid
        tt = t.permute(0, 2, 3, 1).reshape(-1, DW)
        xq, xc, xs = quant8(tt, codes=True)
        b2 = net.view(f'b2{b}')
        if b in SSM:
            sample = torch.arange(B, device=px.device).repeat_interleave(h * w)
            o = SSM[b](tt, xq, sample, B) + b2
        elif trees:
            y = Trees.apply(xq, ternary(net.TI[b]), net.view(f'tb{b}'), ternary(net.TO[b]), xc, xs, net.T, net.V, net.TN, temp, credit)
            o = tt + b2 + y
        else:
            hb = xq @ ternary(net.W1[b]).t() + net.view(f'b1{b}')
            g = quant8(GeluAS.apply(hb))
            f = g @ ternary(net.W2[b]).t() + b2
            if keep is not None: keep.append((xq.detach(), xc, xs, (f - b2).detach()))
            o = tt + f
        x = o.view(B, h, w, DW).permute(0, 3, 1, 2)
    z = x.permute(0, 2, 3, 1) @ net.view('Wh', 4, DW).t() + net.view('bh')
    return torch.sigmoid(z), hits


def loss_of(p, px):
    """loss_grad: per crop, mean |alpha - target alpha| + mean over colour of |(p - target) target alpha| / 3."""
    q = px[..., 13:17].float() / 255
    ta = q[..., 3:4]
    la = (p[..., 3:4] - ta).abs().flatten(1).mean(1)
    lc = ((p[..., :3] - q[..., :3]) * ta).abs().flatten(1).mean(1)
    return la + lc   # mean over colour values is already / 3 of their sum over the crop's L tokens


# ---------------- optimisers (adam(), muon_apply()) ----------------
class Optim:
    def __init__(self, net, distill=False):
        self.net, self.distill = net, distill
        self.dm, self.dv = torch.zeros_like(net.dense), torch.zeros_like(net.dense)
        self.tm, self.tv = torch.zeros_like(net.tex), torch.zeros_like(net.tex)
        mats = [net.TI, net.TO] if net.T else [net.W1, net.W2]
        self.mom = [torch.zeros_like(m) for m in mats]
        self.mask = None
        if distill:   # only the trees' node biases (b2 is shared with the teacher)
            self.mask = torch.zeros_like(net.dense, dtype=torch.bool)
            for b in range(net.NB): a, n = net.o[f'tb{b}']; self.mask[a:a + n] = True

    @staticmethod
    def adam(p, g, m, v, lr, wd, b1, b2, eps, c1, c2):
        m.mul_(b1).add_(g, alpha=1 - b1)
        v.mul_(b2).addcmul_(g, g, value=1 - b2)
        p.sub_(lr * (wd * p + (m / c1) / (torch.sqrt(v / c2) + eps)))

    def muon(self, p, g, m, lr, wd, rows):
        """muon_apply on blocks of `rows` rows, all at once: p, g, m (NB, R, 64), R a multiple of rows."""
        NB, R, d = p.shape
        P, G, Mo = (t.view(-1, rows, d) for t in (p, g, m))
        Mo.mul_(.95).add_(G, alpha=.05)
        X = .95 * Mo + .05 * G
        X = X / (X.flatten(1).norm(dim=1).view(-1, 1, 1) + 1e-7)
        for _ in range(5):
            A = X @ X.transpose(1, 2)
            Bm = -4.775 * A + 2.0315 * (A @ A)
            X = 3.4445 * X + Bm @ X
        lr_m = lr * 3 * (.2 * math.sqrt(max(rows, d)))
        P.sub_(lr * wd * P + lr_m * X)

    @torch.no_grad()
    def step(self, it, lr, lr_tex, B, hits, clip=1.0, wd=.01, b1=.9, b2=.98, eps=1e-8):
        net = self.net
        mats = [net.TI, net.TO] if net.T else [net.W1, net.W2]
        gd = net.dense.grad if net.dense.grad is not None else torch.zeros_like(net.dense)
        if self.mask is not None: gd = gd * self.mask
        gmats = [m.grad if m.grad is not None else torch.zeros_like(m) for m in mats]
        gt = net.tex.grad if net.tex.grad is not None else torch.zeros_like(net.tex)
        hit = torch.zeros(net.n_tex // FCH, dtype=torch.bool, device=net.tex.device); hit[hits] = True
        nrm = (gd.double() ** 2).sum() + sum((g.double() ** 2).sum() for g in gmats) + (gt.view(-1, FCH)[hit].double() ** 2).sum()
        scale = 1 / B; norm = math.sqrt(nrm.item()) * scale
        if clip > 0 and norm > clip: scale *= clip / (norm + 1e-6)
        c1, c2 = 1 - b1 ** it, 1 - b2 ** it
        if self.mask is None:
            self.adam(net.dense, gd * scale, self.dm, self.dv, lr, wd, b1, b2, eps, c1, c2)
        else:   # only the masked part moves (and its moments)
            p, m, v = net.dense[self.mask], self.dm[self.mask], self.dv[self.mask]
            self.adam(p, gd[self.mask] * scale, m, v, lr, wd, b1, b2, eps, c1, c2)
            net.dense[self.mask], self.dm[self.mask], self.dv[self.mask] = p, m, v
        rows = net.TN if net.T else 32
        for p, g, mo in zip(mats, gmats, self.mom): self.muon(p, g * scale, mo, lr, wd, rows)
        # lazy texture Adam: only the texels the step read
        T, G, M, V = (t.view(-1, FCH) for t in (net.tex, gt, self.tm, self.tv))
        tp, tm, tv = T[hit], M[hit], V[hit]
        self.adam(tp, G[hit] * scale, tm, tv, lr_tex, 0.0, b1, b2, eps, c1, c2)
        T[hit], M[hit], V[hit] = tp, tm, tv
        for t in net.leaves(): t.grad = None


# ---------------- evaluation (shader_nts.c evaluate) ----------------
@torch.no_grad()
def evaluate(net, data, ids, trees):
    err = 0.0
    for i in ids:
        px = data.px[i:i + 1].to(net.dev)
        p, _ = forward(net, px, data.pitch_t[i:i + 1].to(net.dev), trees)
        q = px[..., 13:17].float() / 255; a, ta = p[..., 3:4], q[..., 3:4]
        err += ((p[..., :3] * a + 1 - a) - (q[..., :3] * ta + 1 - ta)).abs().mean().item()
    return err / len(ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True); ap.add_argument('--load'); ap.add_argument('--save', default='model.ntsw')
    ap.add_argument('--tree', help='T,V: the fast feedforward trees'); ap.add_argument('--distill', action='store_true')
    ap.add_argument('--tree-temp', type=float, default=1.0)
    ap.add_argument('--trees-from', help="an NTST whose trees (w_in, w_out, node biases) replace --load's, e.g. shader_nts --save-init's for --distill")
    ap.add_argument('--net-from', help="another character's model: its network (embed, stencils, FFNs or trees, head) to start "
                    'from, with this one\'s neural texture fresh (a new asset trains far faster than from nothing)')
    ap.add_argument('--blocks', type=int, default=8)
    ap.add_argument('--iters', type=int, default=12000); ap.add_argument('--seconds', type=float, default=0)
    ap.add_argument('--batch', type=int, default=8); ap.add_argument('--crop', type=int, default=96)
    ap.add_argument('--lr', type=float, default=0.003); ap.add_argument('--lr-tex', type=float, default=0.01)
    ap.add_argument('--eval-every', type=int, default=500); ap.add_argument('--tex8', action='store_true')
    ap.add_argument('--eval-all', action='store_true', help='every frame held out (a pack_data.py --held file)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--seed', type=int, default=0, help='a fresh model (no --load) from this seed')
    ap.add_argument('--compare-ref', help="a shader_nts --ref file: this forward pass against the C trainer's")
    ap.add_argument('--compare-grads', help="a shader_nts --dump-grads file: step 1's gradients against the C trainer's")
    a = ap.parse_args()
    dev = torch.device(a.device)
    T, V = (int(v) for v in a.tree.split(',')) if a.tree else (0, 4)
    data = Data(a.data, dev if dev.type == 'cuda' else torch.device('cpu'))
    net = Net(a.blocks, data.P, DIL_DEFAULT, T, V, dev)
    net.init(a.seed)   # as net_init() then load_net(): what the file doesn't hold (a dense model's trees) starts random
    if a.load: net.load(a.load)
    if a.net_from: net.load(a.net_from, tex=False)
    if a.trees_from:
        other = Net(net.NB, net.P, net.DIL, T, V, dev); other.load(a.trees_from)
        net.TI.copy_(other.TI); net.TO.copy_(other.TO)
        for b in range(net.NB): net.view(f'tb{b}').copy_(other.view(f'tb{b}'))
        del other
    if a.distill and not (T and a.load): raise SystemExit('--distill needs --tree and --load of a dense model')
    if a.tex8: net.tex_to_8bit()
    held = [i for i in range(data.N) if i % 10 == 5 or a.eval_all]
    train_ids = [i for i in range(data.N) if not (i % 10 == 5 or a.eval_all)]
    eval_ids = held[::1 if a.eval_all else 4]
    ch, cw = min(a.crop, data.H), min(a.crop, data.W)
    trees_on = bool(T) and not a.distill
    print(f'{data.N} frames {data.W}x{data.H} ({len(train_ids)} train, {len(held)} held out, {len(eval_ids)} evaluated), '
          f'{data.P} parts; {net.NB} blocks, dilations {" ".join(map(str, net.DIL))}; '
          f'{"trees %d x %d levels" % (T, V) if T else "dense FFNs"}{"; distilling" if a.distill else ""}; {dev}', flush=True)

    if a.compare_ref:
        raw = open(a.compare_ref, 'rb').read(); assert raw[:4] == b'NTSR'
        H, W = struct.unpack_from('<2i', raw, 4); pitch = struct.unpack_from('<f', raw, 12)[0]
        px = torch.from_numpy(np.frombuffer(raw, np.uint8, H * W * 17, 16).copy()).view(1, H, W, 17).to(dev)
        want = torch.from_numpy(np.frombuffer(raw, np.float32, H * W * 4, 16 + H * W * 17).copy()).view(1, H, W, 4).to(dev)
        with torch.no_grad(): p, _ = forward(net, px, torch.tensor([pitch], device=dev), bool(T))
        ow = lambda x: x[..., :3] * x[..., 3:] + 1 - x[..., 3:]
        e = (ow(p) - ow(want)).abs()
        print(f'forward vs the C trainer, over white: largest difference {e.max().item():.2e}, mean {e.mean().item():.2e}, '
              f'{(e > 2 / 255).sum().item()} values off by > 2/255 of {e.numel()}')
        return

    leaves = net.leaves()
    def batch_grads(it, B):
        for t in leaves: t.requires_grad_(True); t.grad = None
        cs = crops(it, B, train_ids, data.H, data.W, ch, cw)
        px = torch.stack([data.px[fi, cy:cy + ch, cx:cx + cw] for fi, cy, cx in cs]).to(dev)
        pitch = torch.tensor([data.pitch[fi] for fi, _, _ in cs], device=dev)
        if a.distill:   # the dense forward, then each block's trees against its FFN
            keep = []
            with torch.no_grad(): _, hits = forward(net, px, pitch, False, keep=keep)
            L = ch * cw; tot = 0
            for b, (xq, xc, xs, f) in enumerate(keep):
                y = Trees.apply(xq, ternary(net.TI[b]), net.view(f'tb{b}'), ternary(net.TO[b]), xc, xs, T, V, net.TN, a.tree_temp, True)
                tot = tot + ((y - f) ** 2).sum() / L
            tot.backward()
            return tot.item() / (DW * net.NB), hits[:0]   # the texture isn't trained here
        p, hits = forward(net, px, pitch, trees_on, credit=trees_on, temp=a.tree_temp)
        l = loss_of(p, px)
        l.sum().backward()
        return l.sum().item(), hits

    if a.compare_grads:
        raw = open(a.compare_grads, 'rb').read(); assert raw[:4] == b'NTSG'
        B, trees_c = struct.unpack_from('<2i', raw, 4); loss_c = struct.unpack_from('<d', raw, 12)[0]
        off = 20
        f32 = lambda n: torch.from_numpy(np.frombuffer(raw, np.float32, n, off).copy())
        loss_t, hits = batch_grads(1, B)
        def rel(name, g_t, g_c):
            g_t = g_t.detach().cpu().double(); g_c = g_c.double()
            print(f'  {name:<16} |grad| {g_c.norm():.4e}   relative error {((g_t - g_c).norm() / g_c.norm().clamp_min(1e-30)):.2e}')
        print(f'step 1, {B} crops: loss C {loss_c / B:.6f}, torch {loss_t / B:.6f}')
        gd_c = f32(net.n_dense); off += 4 * net.n_dense
        gd_t = net.dense.grad if net.dense.grad is not None else torch.zeros_like(net.dense)
        groups = {'embed': ['We', 'be'], 'stencils': [f'tc{b}' for b in range(net.NB)], 'FFN biases': [f'b{k}{b}' for k in (1, 2) for b in range(net.NB)],
                  'tree biases': [f'tb{b}' for b in range(net.NB)] if T else [], 'head': ['Wh', 'bh']}
        for gname, names in groups.items():
            if not names: continue
            idx = torch.cat([torch.arange(net.o[n][0], net.o[n][0] + net.o[n][1]) for n in names])
            rel(gname, gd_t.cpu()[idx], gd_c[idx])
        mats = [net.TI, net.TO] if T else [net.W1, net.W2]
        rows = T * net.TN if T else DW
        gm = [torch.zeros(net.NB, rows, DW), torch.zeros(net.NB, rows, DW)]
        for b in range(net.NB):
            for k in (0, 1): gm[k][b] = f32(rows * DW).view(rows, DW); off += 4 * rows * DW
        for k, nm in enumerate(['w_in' if T else 'W1', 'w_out' if T else 'W2']):
            g = mats[k].grad if mats[k].grad is not None else torch.zeros_like(mats[k])
            rel(('tree ' if T else 'FFN ') + nm, g, gm[k])
        nh = struct.unpack_from('<i', raw, off)[0]; off += 4
        rec = np.frombuffer(raw, np.dtype([('i', '<i4'), ('g', '<f4', FCH)]), nh, off)
        hit_c = torch.from_numpy(rec['i'].astype(np.int64)); g_c = torch.from_numpy(rec['g'].copy())
        hit_t = torch.unique(hits.cpu())
        print(f'  texels read: C {nh}, torch {hit_t.numel()} (same set: {bool(torch.equal(hit_t, torch.sort(hit_c).values))})')
        if nh: rel('texture', net.tex.grad.view(-1, FCH).cpu()[hit_c], g_c)
        return

    opt = Optim(net, a.distill)
    t0 = time.time(); ema = None
    iters = 1 << 40 if a.seconds > 0 else a.iters   # a time budget sets the length (and the schedule), not --iters
    for it in range(1, iters + 1):
        el = time.time() - t0
        prog = min(1, el / a.seconds) if a.seconds > 0 else it / a.iters
        sched = min(1, it / 300) * (.1 + .9 * .5 * (1 + math.cos(math.pi * prog)))
        loss, hits = batch_grads(it, a.batch)
        for t in leaves: t.requires_grad_(False)
        opt.step(it, a.lr * sched, a.lr_tex * sched, a.batch, hits)
        loss /= a.batch
        ema = loss if ema is None else .98 * ema + .02 * loss
        last = it == iters or (a.seconds > 0 and time.time() - t0 >= a.seconds)
        if it % a.eval_every == 0 or last:
            e = evaluate(net, data, eval_ids, bool(T))
            print(f'it {it}  loss {ema:.4f}  held-out L1 {e:.4f}  {time.time() - t0:.0f}s', flush=True)
            net.save(a.save)
        if last: break
    print(f'saved {a.save}')


if __name__ == '__main__':
    main()
