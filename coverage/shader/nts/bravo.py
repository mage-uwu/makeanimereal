#!/usr/bin/env python3
"""bravo.py -- step bravo: a character's renders + its anime film in, the viewer's style shader out.

    python3 bravo.py --data DIR [DIR ...] --out OUT

DIR: build_dataset.py folders of one character (meta.json, <frame>_{color,normal,depth,uva,uvb,target}.png;
the targets are the stylised film's frames, step alpha's output). On one GPU it runs, resuming what
is done:
  1  data      OUT/keys.json, OUT/data.npz. Held out: every frame whose yaw is in --held-yaw (default
               140-180 degrees, ~11% of an orbit: angles never trained on). Frames over 900 px tall
               are halved (the shader trains at ~360 x 640).
  2  phase 1   OUT/phase1: style_torch.py from scratch, perceptual + anchor, adversarial x0.05 from
               step 1500, no learned texture (the v1 recipe)
  3  phase 2   OUT/phase2: from phase 1, + the learned coarse texture (64^2 a part), the tolerant line
               loss x5, adversarial x0.15 from step 300 (the v2 recipe)
  4  finish    style_finish.py: frozen node memories, held-out scores, OUT/phase2/{held,faces}.png
               against phase 1, and OUT/viewer: drop it in as coverage/live/<character>/nts-style/
Each stage's log is OUT/<stage>.log; a stage is skipped when its 'done' marker exists.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def keys_of(dirs, held_yaw):
    lo, hi = held_yaw
    keys = []
    for d in dirs:
        meta = json.loads((Path(d) / 'meta.json').read_text())
        for f in meta['frames']:
            k = f"{d}/{f['frame']:04d}_target.png"
            if not Path(k).exists(): continue
            keys.append({'dir': str(d), 'frame': f['frame'], 'pitch': f['pitch'], 'yaw': f['yaw'], 'key': k,
                         'held': bool(lo <= f['yaw'] % 360 < hi)})
    return keys


def stage(out, name, cmd):
    done = out / name / 'done' if name.startswith('phase') else out / f'{name}.done'
    if done.exists():
        print(f'== {name}: done already', flush=True); return
    print(f'== {name}: ' + ' '.join(map(str, cmd)), flush=True)
    t = time.time()
    with open(out / f'{name}.log', 'w') as log:
        p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=HERE)
        for line in p.stdout:
            log.write(line); log.flush(); print(f'  {line}', end='', flush=True)
        if p.wait(): raise SystemExit(f'{name} failed (see {out}/{name}.log)')
    done.parent.mkdir(parents=True, exist_ok=True); done.write_text(f'{time.time() - t:.0f}s\n')
    print(f'== {name}: {time.time() - t:.0f}s', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', nargs='+', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--held-yaw', default='140,180')
    ap.add_argument('--phase1', type=int, default=6000, help='steps'); ap.add_argument('--phase2', type=int, default=5600)
    ap.add_argument('--batch', type=int, default=6); ap.add_argument('--crop', type=int, default=256)
    ap.add_argument('--train-args', default='', help='more style_torch.py arguments for both phases')
    ap.add_argument('--no-compile', action='store_true', help="train without torch.compile (it is ~10%% faster with it)")
    ap.add_argument('--finish-args', default='', help='more style_finish.py arguments')
    ap.add_argument('--stages', default='data,phase1,phase2,finish', help='which stages to run (resume skips done ones)')
    a = ap.parse_args()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    dirs = [Path(d).resolve() for d in a.data]
    py = sys.executable
    # 1: data
    run = a.stages.split(',')
    if 'data' in run and not (out / 'data.done').exists():
        sys.path.insert(0, str(HERE))
        from make_style_data import build
        t = time.time()
        keys = keys_of(dirs, [float(v) for v in a.held_yaw.split(',')])
        h = json.loads((dirs[0] / 'meta.json').read_text())['size'][1]
        d = build(keys, full=h <= 900)
        (out / 'keys.json').write_text(json.dumps(keys))
        np.savez(out / 'data.npz', **d)
        (out / 'data.done').write_text(f'{time.time() - t:.0f}s\n')
        print(f"== data: {len(d['px'])} frames {d['px'].shape[2]}x{d['px'].shape[1]}, {int(d['held'].sum())} held out, "
              f"{d['parts']} parts ({time.time() - t:.0f}s)", flush=True)
    common = ['--data', out / 'data.npz', '--ssm', '1,3', '--batch', a.batch, '--crop', a.crop] + ([] if a.no_compile else ['--compile']) + a.train_args.split()
    # 2, 3: the two phases
    if 'phase1' in run: stage(out, 'phase1', [py, 'style_torch.py', *common, '--iters', a.phase1, '--out', out / 'phase1'])
    if 'phase2' in run: stage(out, 'phase2', [py, 'style_torch.py', *common, '--iters', a.phase2, '--out', out / 'phase2',
                          '--load', out / 'phase1' / 'style.ntst', '--coarse-tex', '--lines', 5, '--adv', 0.15, '--adv-from', 300])
    # 4: finish (phase 1 frozen too, so the comparison is as the viewer would draw it)
    if 'finish' in run: stage(out, 'finish1', [py, 'style_finish.py', out / 'phase1', '--data', out / 'data.npz', '--keys', out / 'keys.json', '--out', out / 'viewer-phase1', *a.finish_args.split()])
    if 'finish' in run: stage(out, 'finish', [py, 'style_finish.py', out / 'phase2', '--data', out / 'data.npz', '--keys', out / 'keys.json',
                          '--out', out / 'viewer', '--compare', out / 'phase1', *a.finish_args.split()])
    if 'finish' in run:
        print(f'== bravo done: {out}/viewer (the viewer files), {out}/phase2/held.png, faces.png, finish.json', flush=True)


if __name__ == '__main__':
    main()
