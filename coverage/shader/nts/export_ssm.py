#!/usr/bin/env python3
"""export_ssm.py -- the SSM tree blocks of a style_torch.py model for the viewer (nts-gl.js), beside
export_nts.py's files: ssm.bin, and "ssm": [blocks] in meta.json.

The viewer reads each node's memory as fixed: step 1 (the walk) reads dec h0, step 3 the per-node mean
of the states the training frames wrote (frozen_state.pt, ssm_tree.py's last_h1 averaged over whole
frames). On held-out frames that drew within 0.63/255 on average of the per-frame write.

    python3 export_ssm.py style.ntst.ssm.pt frozen_state.pt out_dir --blocks 1,3

ssm.bin, little-endian: 'SSM1', int32 count, then per block: int32 block; proj codes int8 (36 x 64),
proj scales f32 (36); w_in codes int8 (R x 64), scales f32 (R), node biases f32 (R); w_out codes int8
(R x 64), scales f32 (R); dec h0 f32 (R x n x pv); the frozen state f32 (R x n x pv); U codes int8
(64 x T pv), scales f32 (64). R = T (2^V - 1) nodes; ternary as shader_torch.ternary (row mean |w|).
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch


def ternary(w):   # codes and row scales, as shader_torch.ternary
    s = w.abs().mean(-1, keepdim=True).clamp_min(1e-8)
    return torch.round((w / s).clamp(-1, 1)).to(torch.int8), s[:, 0].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ssm'); ap.add_argument('frozen'); ap.add_argument('out')
    ap.add_argument('--blocks', default='1,3')
    a = ap.parse_args()
    blocks = [int(b) for b in a.blocks.split(',')]
    sds = torch.load(a.ssm, map_location='cpu'); frozen = torch.load(a.frozen, map_location='cpu')
    out = bytearray(b'SSM1' + struct.pack('<i', len(blocks)))
    f32 = lambda t: t.detach().float().contiguous().numpy().astype('<f4').tobytes()
    for b, sd in zip(blocks, sds):
        out += struct.pack('<i', b)
        for name in ('proj', 'w_in', 'w_out'):
            c, s = ternary(sd[name])
            out += c.numpy().tobytes() + f32(s)
            if name == 'w_in': out += f32(sd['bias'])
        dec = torch.exp(-torch.exp(sd['A_log']))
        out += f32(sd['h0'] * dec[:, None, None]) + f32(frozen[b])
        c, s = ternary(sd['U'])
        out += c.numpy().tobytes() + f32(s)
        print(f'block {b}: proj {tuple(sd["proj"].shape)}, nodes {sd["w_in"].shape[0]}, state {tuple(sd["h0"].shape[1:])}, U {tuple(sd["U"].shape)}')
    Path(a.out, 'ssm.bin').write_bytes(bytes(out))
    meta = json.loads(Path(a.out, 'meta.json').read_text()); meta['ssm'] = blocks
    Path(a.out, 'meta.json').write_text(json.dumps(meta))
    print(f'{a.out}/ssm.bin: {len(out)} bytes')


if __name__ == '__main__':
    main()
