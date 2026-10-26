#!/usr/bin/env python3
"""style_finish.py -- a style_torch.py run made ready for the viewer: the node memories frozen, checked,
exported, and the sheets to judge it by.

    python3 style_finish.py RUN --data data.npz --keys keys.json --out VIEWER [--compare OTHER_RUN]

RUN holds style.ntst and style.ntst.ssm.pt. Writes:
  RUN/frozen_state.pt   each SSM node's state, the mean over the training frames' writes (whole frames,
                        as the viewer draws them); the viewer reads it instead of a per-frame write
  VIEWER/               export_nts.py's net.png, fine0..2.png, meta.json + export_ssm.py's ssm.bin:
                        drop it in as coverage/live/<character>/nts-style/
  RUN/held.png          held-out frames: [OTHER_RUN |] this run | the film, 8 rows
  RUN/faces.png         face crops on front views (training frames: what it learned to draw there)
  RUN/finish.json       held-out perceptual / anchor / lines, and |frozen - per-frame| on held-out frames
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import shader_torch
from shader_torch import Net, forward
from ssm_tree import SSMTree
from style_torch import VGG, perceptual, edges, over_white, amp

HERE = Path(__file__).resolve().parent


def load(run, P, a, dev):
    net = Net(a.blocks, P, [int(v) for v in a.dil.split(',')], *map(int, a.tree.split(',')), dev)
    net.init(0); net.load(str(Path(run) / 'style.ntst'))
    ssm = {}
    for b, sd in zip(map(int, a.ssm.split(',')), torch.load(Path(run) / 'style.ntst.ssm.pt', map_location=dev)):
        m = SSMTree(trees=net.T, levels=net.V).to(dev); m.load_state_dict(sd); ssm[b] = m
    return net, ssm


def draw(model, px, pitch):
    net, ssm = model
    shader_torch.SSM.clear(); shader_torch.SSM.update(ssm)
    return forward(net, px, pitch, True)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('--data', required=True); ap.add_argument('--keys', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--compare')
    ap.add_argument('--blocks', type=int, default=4); ap.add_argument('--dil', default='1,3,9,27')
    ap.add_argument('--tree', default='4,4'); ap.add_argument('--ssm', default='1,3')
    ap.add_argument('--vgg', default='imagenet', help="'random': code checks only")
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run = Path(a.run)
    z = np.load(a.data); keys = json.loads(Path(a.keys).read_text())
    px_all = torch.from_numpy(z['px']); pitch = torch.from_numpy(z['pitch']).float(); held = z['held']
    P = int(z['parts'])
    model = load(run, P, a, dev)
    other = load(a.compare, P, a, dev) if a.compare else None
    frames = lambda ids: (px_all[ids].to(dev), pitch[ids].to(dev))
    hi, ti = np.nonzero(held)[0], np.nonzero(~held)[0]
    report = {}
    with torch.no_grad():
        # the frozen state: the mean of every training frame's write (4 whole frames a pass)
        acc = {b: 0 for b in model[1]}
        for k in range(0, len(ti), 4):
            draw(model, *frames(ti[k:k + 4]))
            for b, m in model[1].items(): acc[b] = acc[b] + m.last_h1.sum(0)
        frozen = {b: v / len(ti) for b, v in acc.items()}
        torch.save({b: v.cpu() for b, v in frozen.items()}, run / 'frozen_state.pt')
        # held out: frozen against per-frame, and the scores the viewer's drawing gets
        vgg = VGG(a.vgg == 'random').to(dev)
        diff, tot = [], {'perceptual': 0.0, 'anchor': 0.0, 'lines': 0.0}
        for k in range(0, len(hi), 4):
            ids = hi[k:k + 4]; px, pt = frames(ids)
            for m in model[1].values(): m.frozen = None
            dyn = over_white(draw(model, px, pt))
            for b, m in model[1].items(): m.frozen = frozen[b]
            fro = over_white(draw(model, px, pt))
            diff.append(((dyn - fro).abs() * 255).mean((1, 2, 3)))
            to = over_white(px[..., 13:17].float() / 255)
            tot['perceptual'] += perceptual(vgg, fro, to).sum().item() / len(hi)
            tot['anchor'] += (F.avg_pool2d(fro, 8) - F.avg_pool2d(to, 8)).abs().mean((1, 2, 3)).sum().item() / len(hi)
            tot['lines'] += (edges(fro) - edges(to)).abs().mean((1, 2, 3)).sum().item() / len(hi)
        report['held_out'] = {k: round(v, 4) for k, v in tot.items()}
        report['frozen_vs_per_frame_255'] = round(torch.cat(diff).mean().item(), 3)
        print(f"held out ({len(hi)} frames, frozen state): {report['held_out']}; |frozen - per-frame| "
              f"{report['frozen_vs_per_frame_255']}/255", flush=True)
        if other:   # its own frozen state when it has one (as the viewer drew it), else per-frame
            fz = Path(a.compare) / 'frozen_state.pt'
            fz = torch.load(fz, map_location=dev) if fz.exists() else {}
            for b, m in other[1].items(): m.frozen = fz.get(b)
        # sheets: held-out rows, then face crops on front views
        def row(ids):
            px, pt = frames(ids)
            ims = ([over_white(draw(other, px, pt))] if other else []) + [over_white(draw(model, px, pt)), over_white(px[..., 13:17].float() / 255)]
            return ims
        sel = hi[np.linspace(0, len(hi) - 1, min(8, len(hi))).astype(int)]
        rows = [torch.cat([im[0] for im in row(np.array([i]))], 2) for i in sel]
        Image.fromarray(torch.cat(rows, 1).permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()).save(run / 'held.png')
        front = [i for i in ti if abs((keys[i]['yaw'] + 180) % 360 - 180) < 25 and abs(keys[i]['pitch']) < 15]
        front = front[::max(1, len(front) // 5)][:5]
        faces = []
        for i in front:
            t = px_all[i, ..., 16].float()
            ys, xs = torch.nonzero(t > 127, as_tuple=True)
            if len(ys) == 0: continue
            y0 = int(ys.min()); s = int(t.shape[0] * .16)
            xc = int(xs[ys < y0 + s].float().mean()); x0 = max(0, min(t.shape[1] - s, xc - s // 2))
            faces.append(torch.cat([F.interpolate(im[:, :, y0:y0 + s, x0:x0 + s], size=(320, 320), mode='bicubic')[0]
                                    for im in row(np.array([i]))], 2))
        if faces:
            Image.fromarray(torch.cat(faces, 1).permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()).save(run / 'faces.png')
    (run / 'finish.json').write_text(json.dumps(report, indent=1))
    # the viewer's files
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    parts = Path(keys[0]['dir']) / 'meta.json'
    subprocess.run([sys.executable, str(HERE / 'export_nts.py'), str(run / 'style.ntst'), str(out), '--parts', str(parts)], check=True)
    subprocess.run([sys.executable, str(HERE / 'export_ssm.py'), str(run / 'style.ntst.ssm.pt'), str(run / 'frozen_state.pt'), str(out), '--blocks', a.ssm], check=True)
    (out / 'net.bin').unlink(missing_ok=True)   # the viewer reads net.png (the same bytes)
    print(f'viewer files -> {out}', flush=True)


if __name__ == '__main__':
    main()
