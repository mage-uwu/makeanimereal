#!/usr/bin/env python3
"""Export a shader_nts.c model (.ntsw) for the viewer's CPU mode (coverage/live/nts.js, infer.c).

    python3 export_nts.py model.ntsw ../../live/nts --parts ../../../out/shader-data/w1/meta.json

Writes net.bin (everything infer.c needs except the fine texture) and net.png (the same bytes packed
into an image, for the web), fine0..2.png (the fine texture,
8 bits per channel, parts tiled 5 x 4) and meta.json. The ternary codes, row scales, stencil codes
and gates are derived here exactly as shader_nts.c's requantize() does (tss_ternary, rows64_pack,
tc_pack), in float32, so the browser runs what was trained.

net.bin, little-endian: 'NTSV', int32 NB, parts, DIL[NB] (or 'NTS2', NB, parts, NBC, DIL[NB] for a model
with an ink branch: the colour head reads block NBC's output, the ink head the last block's); float32 We[64 x 18], be[64]; per block
int8 stencil codes[13 x 64], float32 gate[64], b1[64], b2[64], s1[64], s2[64], int8 c1[64 x 64],
c2[64 x 64] (row = output); float32 Wh[4 x 64], bh[4]; float32 lo[2][8], step[2][8] (fine, coarse);
uint8 coarse texture [parts][64][64][8]. NTS2 adds float32 Wl[64], bl after bh: black line art's opacity.
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image

DW, KT, CIN, FCH, FINE, COARSE, TCG, MAXB = 64, 9, 18, 8, 512, 64, 4, 16

ap = argparse.ArgumentParser()
ap.add_argument('model')
ap.add_argument('out')
ap.add_argument('--parts', required=True, help="a dataset's meta.json (its part names, in id order)")
a = ap.parse_args()

raw = Path(a.model).read_bytes()
assert raw[:4] in (b'NTSW', b'NTSL')
NB, dw, P, kt = struct.unpack_from('<4i', raw, 4)
assert (dw, kt) == (DW, KT)
DIL = struct.unpack_from(f'<{MAXB}i', raw, 20)[:NB]
off = 20 + 4 * MAXB
NBC = NB   # NTSL (shader_nts.c --lines): the colour head reads block NBC; blocks NBC.. feed the ink head
if raw[:4] == b'NTSL':
    NBC = struct.unpack_from('<i', raw, off)[0]; off += 4
tc_size = (KT + TCG + 1) * DW
n_dense = DW * CIN + DW + NB * (tc_size + 2 * DW) + 4 * DW + 4 + (DW + 4 if NBC < NB else 0)
f32 = lambda n: np.frombuffer(raw, np.float32, n, off)
dense = f32(n_dense); off += 4 * n_dense
W = []
for b in range(NB):
    w1 = f32(DW * DW).reshape(DW, DW); off += 4 * DW * DW
    w2 = f32(DW * DW).reshape(DW, DW); off += 4 * DW * DW
    W.append((w1, w2))
n_tex = P * (FINE * FINE + COARSE * COARSE) * FCH
tex = f32(n_tex); off += 4 * n_tex
assert off == len(raw), (off, len(raw))
fine = tex[:P * FINE * FINE * FCH].reshape(P, FINE, FINE, FCH)
coarse = tex[P * FINE * FINE * FCH:].reshape(P, COARSE, COARSE, FCH)


def ternary(w):
    """tss_ternary with one group per row, then rows64_pack: codes and the row scale (float32 throughout)."""
    codes = np.zeros((DW, DW), np.int8)
    scale = np.zeros(DW, np.float32)
    for r in range(DW):
        s = np.float32(0)
        for i in range(DW):
            s = np.float32(s + abs(w[r, i]))
        s = np.float32(s / np.float32(DW))
        s = max(s, np.float32(1e-8))
        v = np.clip(w[r] / s, -1, 1).astype(np.float32)
        codes[r] = np.rint(v).astype(np.int8)   # nearbyint: ties to even
        scale[r] = s if codes[r].any() else 0   # rows64_pack: max |Q| of the row
    return codes, scale


o = 0
We = dense[o:o + DW * CIN]; o += DW * CIN
be = dense[o:o + DW]; o += DW
out = bytearray(b'NTSV' + struct.pack(f'<2i{NB}i', NB, P, *DIL) if NBC == NB else b'NTS2' + struct.pack(f'<3i{NB}i', NB, P, NBC, *DIL))
out += We.tobytes() + be.tobytes()
for b in range(NB):
    tc = dense[o:o + tc_size]; o += tc_size
    b1 = dense[o:o + DW]; o += DW
    b2 = dense[o:o + DW]; o += DW
    q = np.where(tc[:(KT + TCG) * DW] > .5, 1, np.where(tc[:(KT + TCG) * DW] < -.5, -1, 0)).astype(np.int8)
    gate = np.tanh(tc[(KT + TCG) * DW:].astype(np.float64)).astype(np.float32)
    c1, s1 = ternary(W[b][0])
    c2, s2 = ternary(W[b][1])
    out += q.tobytes() + gate.tobytes() + b1.tobytes() + b2.tobytes() + s1.tobytes() + s2.tobytes() + c1.tobytes() + c2.tobytes()
Wh = dense[o:o + 4 * DW]; o += 4 * DW
bh = dense[o:o + 4]; o += 4
out += Wh.tobytes() + bh.tobytes()
if NBC < NB:   # the ink head: Wl[64], bl
    out += dense[o:o + DW].tobytes() + dense[o + DW:o + DW + 1].tobytes(); o += DW + 4
assert o == n_dense

# 8 bits per channel between each level's channel min and max (shader_nts.c --tex8 measures the same).
lo = np.zeros((2, FCH), np.float32)
st = np.zeros((2, FCH), np.float32)
q8 = []
for lev, t in enumerate((fine, coarse)):
    lo[lev] = t.min((0, 1, 2))
    st[lev] = (t.max((0, 1, 2)) - lo[lev]) / np.float32(255)
    st[lev][st[lev] <= 0] = 1
    q8.append(np.rint((t - lo[lev]) / st[lev]).clip(0, 255).astype(np.uint8))
out += lo.tobytes() + st.tobytes() + q8[1].tobytes()

dst = Path(a.out)
dst.mkdir(parents=True, exist_ok=True)
(dst / 'net.bin').write_bytes(bytes(out))
# The web copy: the bytes in an opaque RGB PNG, 3 a pixel after a 4-byte little-endian length (the
# viewer's model.vrm.png packing), since artifacts serve images but not raw binaries.
packed = struct.pack('<I', len(out)) + bytes(out)
side = int(np.ceil(np.sqrt(len(packed) / 3)))
px = np.zeros(side * side * 3, np.uint8)
px[:len(packed)] = np.frombuffer(packed, np.uint8)
Image.fromarray(px.reshape(side, side, 3)).save(dst / 'net.png', optimize=True)
# Fine texture: three RGB images, channels 3k..3k+2; part p at tile (p % 5, p // 5) of 512 x 512.
cols, rows = 5, (P + 4) // 5
for k in range(3):
    img = np.zeros((rows * FINE, cols * FINE, 3), np.uint8)
    for p in range(P):
        y, x = (p // cols) * FINE, (p % cols) * FINE
        ch = list(range(3 * k, min(3 * k + 3, FCH)))
        img[y:y + FINE, x:x + FINE, :len(ch)] = q8[0][p][..., ch]
    Image.fromarray(img).save(dst / f'fine{k}.png', optimize=True)
parts = json.loads(Path(a.parts).read_text())['parts']
assert len(parts) == P
(dst / 'meta.json').write_text(json.dumps({'parts': parts, 'blocks': NB, 'colourBlocks': NBC, 'dilations': list(DIL), 'width': DW, 'tile': FINE, 'cols': cols, 'rows': rows}))
print(f'{NB} blocks, {P} parts -> {dst}: net.bin {len(out) / 1e6:.2f} MB + fine textures')
