#!/usr/bin/env python3
"""Targets from crisp for the nanotss shader: crisp's own drawing of each G-buffer frame.

    python3 distill_data.py --teacher ../checkpoints/crisp.pt --gbuffer ../../../out/coverage-gb \\
        --out ../../../out/distill/w1 [--size 360x640] [--every 1] [--hd-preview 4]

crisp is the end of the U-Net line: perceptual loss, LoRA on posed mocap frames, the shift-consistency
pass and the baked unsharp mask. Fitting the drawn films directly with L1 lets their frame-to-frame
inconsistencies settle into the neural texture as splotches; crisp's drawing is one consistent look
(no appearance code, as the viewer runs it) and labels any frame, posed ones included. Each frame of a
render folder (render-orbit.mjs --gbuffer: frames, normal, depth, uva, uvb, cameras.json) is drawn by
crisp at the render's full size, then written as build_dataset.py writes a dataset at --size: colour,
normal and depth resized bilinear, UV and part nearest (parts remapped onto crisp's list), and the
target crisp's drawing averaged down by alpha. pack_data.py packs the folders for shader_nts.c.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train  # noqa: E402
import train_sr  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--teacher', required=True)
ap.add_argument('--gbuffer', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--size', default='360x640')
ap.add_argument('--every', type=int, default=1)
ap.add_argument('--hd-preview', type=int, default=0, help='also save this many of crisp\'s full-size drawings over white')
a = ap.parse_args()
torch.set_grad_enabled(False)

ck = torch.load(a.teacher, map_location='cpu', weights_only=False)
widths = tuple(ck['widths'])
base = train.Shader(len(ck['parts']), ck['model']['codes.weight'].shape[0], ck['arch'], widths=widths)
base.load_state_dict({k: v.float() for k, v in ck['model'].items()})
teacher = train_sr.Sharp(base, widths[0], ck['sr_width'])
teacher.conv.load_state_dict({k[len('conv.'):]: v.float() for k, v in ck['sr'].items() if k.startswith('conv.')})
teacher.head.load_state_dict({k[len('head.'):]: v.float() for k, v in ck['sr'].items() if k.startswith('head.')})
teacher.eval()
parts = ck['parts']

gb, out = Path(a.gbuffer), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
cams = json.loads((gb / 'cameras.json').read_text())
remap = np.arange(256, dtype=np.uint8)
for i, name in enumerate(cams['parts']):
    remap[i] = parts.index(name)
w, h = map(int, a.size.split('x'))
done, t0 = [], time.time()
for i, cam in enumerate(cams['cameras']):
    name = f'{i:04d}'
    if i % a.every or not (gb / 'uvb' / f'{name}.png').exists():
        continue
    img = {k: Image.open(gb / k / f'{name}.png').convert('RGB') for k in ['frames', 'normal', 'depth', 'uva', 'uvb']}
    uvb = np.asarray(img['uvb']).copy()
    uvb[..., 1] = remap[uvb[..., 1]]
    img['uvb'] = Image.fromarray(uvb)
    # crisp's input at full size, as train.load builds it
    color, normal, depth = (np.asarray(img[k]).astype(np.float32) for k in ['frames', 'normal', 'depth'])
    uva, uvbf = np.asarray(img['uva']).astype(np.float32), uvb.astype(np.float32)
    u = (uva[..., 0] + uva[..., 1] / 255) / 255
    v = (uva[..., 2] + uvbf[..., 0] / 255) / 255
    x = np.concatenate([color / 255, normal / 255 * 2 - 1, depth[..., :1] / 255, (uvbf[..., 2:3] > 127).astype(np.float32),
                        u[..., None], v[..., None], uvbf[..., 1:2]], -1)
    y = teacher(torch.from_numpy(x)[None], torch.tensor([cam['pitch']], dtype=torch.float32))[0]   # [4, H, W] straight
    # down to --size by alpha-weighted area averaging (the drawing as a smaller render of it would be)
    s = (y.shape[1] // h, y.shape[2] // w)
    al = F.avg_pool2d(y[3:4][None], s)[0]
    rgb = F.avg_pool2d((y[:3] * y[3:4])[None], s)[0] / al.clamp_min(1e-6)
    tgt = torch.cat([rgb, al]).clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray((tgt * 255 + .5).astype(np.uint8), 'RGBA').save(out / f'{name}_target.png')
    for kind, resample in [('frames', Image.BILINEAR), ('normal', Image.BILINEAR), ('depth', Image.BILINEAR),
                           ('uva', Image.NEAREST), ('uvb', Image.NEAREST)]:
        img[kind].resize((w, h), resample).save(out / f'{name}_{"color" if kind == "frames" else kind}.png')
    if len(done) < a.hd_preview:
        ow = (y[:3] * y[3:4] + 1 - y[3:4]).clamp(0, 1).permute(1, 2, 0).numpy()
        Image.fromarray((ow * 255 + .5).astype(np.uint8)).save(out / f'hd_{name}.png')
    done.append({'frame': i, 'pitch': cam['pitch'], 'yaw': cam['yaw']})
    if len(done) % 50 == 0:
        print(f'{len(done)} frames, {(time.time() - t0) / len(done):.2f} s each', flush=True)
(out / 'meta.json').write_text(json.dumps({'size': [w, h], 'parts': parts, 'frames': done, 'teacher': a.teacher}))
print(f'{len(done)} frames -> {out}')
