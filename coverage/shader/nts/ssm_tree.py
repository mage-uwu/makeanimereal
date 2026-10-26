"""ssm_tree.py -- nanoTSS's SSM tree layer (nanotss_blt.c layer_fwd) as an image block.

In nanotss_blt.c each token walks T trees of D+1 levels; every visited node keeps a state matrix
h (n x pv) and, per token:
    R = C . (dec h)                 read before write  (dec = exp(-exp(A_log) dt))
    h = dec h + dt B (x) V          write
    s = w_in . x + bias + R . Z     routing reads the memory; the child is sign(s)
    y += gelu(s) w_out              the tree's output, and U (sum R) projected back
B, C (n), V, Z (pv), dt (T) come from a ternary projection of the int8 codes; w_in, w_out, U are
ternary too. That is content-addressed memory: tokens routed to a node share its state.

For an image there is no token order and nothing to predict next: a frame is one step at which all
its pixels write at once, so a node's memory holds what all of that region said. Per image:
    1  each pixel walks with R from the node's state at the start (a learned h0 per node);
    2  each visited node:  h1 = dec h0 + mean over its pixels of dt B (x) V   (a mean: crop-size free)
    3  on the same path, R = C . h1:  s = w_in . x + bias + R . Z,  y += gelu(s) w_out,  sum R -> U
Trained with the image objective (denoise_torch.py), not next-token prediction. The block replaces a
tree block's FFN after its stencil (shader_torch.forward: SSM[b]).
"""
import math

import torch
import torch.nn.functional as F

from shader_torch import ternary, gelu


class SSMTree(torch.nn.Module):
    def __init__(self, d=64, trees=4, levels=4, n=8, pv=8):
        super().__init__()
        self.d, self.T, self.V, self.n, self.pv = d, trees, levels, n, pv
        self.TN = (1 << levels) - 1
        R = trees * self.TN
        g = torch.Generator().manual_seed(1)
        r = lambda *s: torch.randn(*s, generator=g)
        self.Pd = 2 * n + 2 * pv + trees
        self.proj = torch.nn.Parameter(r(self.Pd, d) / math.sqrt(d))
        self.w_in = torch.nn.Parameter(r(R, d) / math.sqrt(d))
        self.bias = torch.nn.Parameter(torch.zeros(R))
        self.w_out = torch.nn.Parameter(r(R, d) * 0.02)
        self.A_log = torch.nn.Parameter(torch.full((R,), math.log(0.5)))       # dec = exp(-0.5) ~ 0.61
        self.dt_bias = torch.nn.Parameter(torch.full((trees,), -1.0))
        self.U = torch.nn.Parameter(r(d, trees * pv) * 0.02)
        self.h0 = torch.nn.Parameter(torch.zeros(R, n, pv))

    def forward(self, tt, xq, sample, S):
        """tt: (N, d) the stencil's output (the residual), xq: (N, d) its int8 codes x scale (quant8),
        sample: (N,) which image each pixel is in, S: images. Returns (N, d), the block's output."""
        N, d, T, V, n, pv, TN = tt.shape[0], self.d, self.T, self.V, self.n, self.pv, self.TN
        p = xq @ ternary(self.proj).t()
        Bk, Cq, Vv, Zz, dtr = p.split([n, n, pv, pv, T], -1)
        gain = F.softplus(dtr + self.dt_bias)                               # (N, T)
        dec = torch.exp(-torch.exp(self.A_log))                             # (R,)
        S_in = xq @ ternary(self.w_in).t() + self.bias                      # (N, R): w_in . x + bias, every node
        # 1: the walk, reading the starting state
        rows = torch.empty(N, T, V, dtype=torch.long, device=tt.device)
        for tr in range(T):
            a = torch.zeros(N, dtype=torch.long, device=tt.device)
            for j in range(V):
                row = tr * TN + a
                rows[:, tr, j] = row
                h = self.h0[row] * dec[row, None, None]
                R = torch.einsum('pn,pnv->pv', Cq, h)
                s = S_in.gather(1, row[:, None])[:, 0] + (R * Zz).sum(-1)
                a = 2 * a + 1 + (s > 0).long()
        # 2: the write, per image and node: dec h0 + the mean of gain B (x) V over the pixels that visited it
        R_ = T * TN
        key = (sample[:, None, None] * R_ + rows).view(-1)                  # (N T V)
        w = (gain[:, :, None, None, None] * (Bk[:, :, None] * Vv[:, None, :])[:, None, None]).expand(N, T, V, n, pv).reshape(-1, n * pv)
        tot = torch.zeros(S * R_, n * pv, device=tt.device).index_add_(0, key, w)
        cnt = torch.zeros(S * R_, device=tt.device).index_add_(0, key, torch.ones_like(key, dtype=tt.dtype))
        h1 = (self.h0 * dec[:, None, None]).repeat(S, 1, 1).view(S * R_, n * pv) + tot / cnt.clamp(min=1)[:, None]
        # 3: the same path, reading the new state
        hs = h1[key].view(N, T, V, n, pv)
        R2 = torch.einsum('pn,ptjnv->ptjv', Cq, hs)                          # (N, T, V, pv)
        s2 = S_in.gather(1, rows.view(N, -1)).view(N, T, V) + (R2 * Zz[:, None, None]).sum(-1)
        Wo = ternary(self.w_out)
        y = torch.einsum('ptj,ptjd->pd', gelu(s2), Wo[rows])
        return tt + y + R2.sum(2).reshape(N, T * pv) @ ternary(self.U).t()
