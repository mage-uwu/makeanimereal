#!/usr/bin/env python3
"""SSMTree.forward (dense matmuls) against forward_ref (per-pixel gathers): outputs, the written
states, and every gradient, on random data. Routing can flip where a node's score is ~0 (sums in
another order), so the comparison is on the pixels whose paths agree, and the flips are counted.
    python3 test_ssm_fast.py"""
import torch
from ssm_tree import SSMTree
from shader_torch import quant8

torch.manual_seed(0)
S, Np, d = 3, 500, 64
N = S * Np
m = SSMTree()
with torch.no_grad():   # a trained-looking state: nonzero h0, varied decay
    m.h0.normal_(0, .5); m.A_log.normal_(-.7, .3); m.bias.normal_(0, .3)
tt0 = torch.randn(N, d)
sample = torch.arange(S).repeat_interleave(Np)
for frozen in (False, True):
    m.frozen = torch.randn(m.T * m.TN, m.n, m.pv) * .3 if frozen else None
    outs, grads, h1s = [], [], []
    for f in (m.forward_ref, m.forward):
        m.zero_grad()
        tt = tt0.clone().requires_grad_(True)
        xq = quant8(tt)
        y = f(tt, xq, sample, S)
        (y * torch.linspace(-1, 1, d)).sum().backward()
        outs.append(y.detach()); h1s.append(m.last_h1.clone())
        grads.append([tt.grad.clone()] + [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in m.parameters()])
    diff = (outs[0] - outs[1]).abs().amax(1)
    flips = (diff > 1e-3).sum().item()
    ok = diff <= 1e-3
    print(f'frozen={frozen}: output max diff (agreeing paths) {diff[ok].max():.2e}; pixels with a flipped route {flips}/{N}')
    print(f'  written states max diff {(h1s[0] - h1s[1]).abs().max():.2e}')
    names = ['input'] + [n for n, _ in m.named_parameters()]
    for nm, ga, gb in zip(names, *grads):
        rel = (ga - gb).norm() / ga.norm().clamp_min(1e-12)
        print(f'  grad {nm:<8} relative difference {rel:.2e}')
    assert flips <= N * 0.002 and diff[ok].max() < 1e-4
print('ok')
