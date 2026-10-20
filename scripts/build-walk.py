#!/usr/bin/env python3
"""Pack an animation's drawings for the web viewer: one sheet per camera angle, all frames on it.

    python3 scripts/build-walk.py [--renders out/walk-alpha] [--keys out/walk] [--out out/web2]
                                  [--name walk] [--cache out/cache-walk]

Each drawing is cut out of its chroma background and placed in the render frame the same way
build-web.py places the still keys: scaled so her height matches the render's silhouette and
shifted so her head and legs line up. The scale is the median over the angle's frames, so the
sprite doesn't pulse when a frame's raised foot changes the silhouette's height; the offset follows
each frame, so the head bob comes from the render. Writes <out>/<name>/<yaw>_<pitch>.webp
(frames in a grid, row by row) and <out>/<name>.json.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

spec = importlib.util.spec_from_file_location('build_web', Path(__file__).with_name('build-web.py'))
bw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bw)

COLS = 4  # frames across a sheet


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--renders', default='out/walk-alpha', help='the animation rendered with --background none')
    p.add_argument('--keys', default='out/walk')
    p.add_argument('--out', default='out/web2')
    p.add_argument('--name', default='walk')
    p.add_argument('--cache', default='out/cache-walk')
    p.add_argument('--width', type=int, default=600)
    args = p.parse_args()
    renders, keys, out, cache = Path(args.renders), Path(args.keys), Path(args.out), Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    (out / args.name).mkdir(parents=True, exist_ok=True)
    manifest = json.loads((renders / 'manifest.json').read_text())
    size = (args.width, round(args.width * manifest['height'] / manifest['width']))

    angles = {}
    for v in manifest['views']:
        angles.setdefault((v['yaw'], v['pitch']), []).append(v)
    sheets = []
    for (yaw, pitch), views in sorted(angles.items(), key=lambda a: (a[0][1], a[0][0])):
        views.sort(key=lambda v: v['frame'])
        if not all((keys / v['file']).exists() for v in views):
            continue
        cuts, fits = [], []
        for v in views:
            src, cached = keys / v['file'], cache / v['file']
            if cached.exists() and cached.stat().st_mtime >= src.stat().st_mtime:
                cut = Image.open(cached)
            else:
                cut = bw.cut_out_chroma(Image.open(src).convert('RGB'), None)
                cut.save(cached)
            rt, rb, rx = bw.extent(Image.open(renders / v['file']).getchannel('A'))
            st, sb, sx = bw.extent(cut.getchannel('A'))
            cuts.append(cut)
            fits.append(((rb - rt) / (sb - st), rt, rx, st, sx))
        s = float(np.median([f[0] for f in fits]))
        frames = [{'s': round(s, 5), 'ox': round(rx - sx * s, 5), 'oy': round(rt - st * s, 5)} for _, rt, rx, st, sx in fits]
        rows = -(-len(views) // COLS)
        sheet = Image.new('RGBA', (size[0] * COLS, size[1] * rows), (0, 0, 0, 0))
        for i, cut in enumerate(cuts):
            sheet.paste(cut.resize(size, Image.LANCZOS), ((i % COLS) * size[0], (i // COLS) * size[1]))
        name = f'{args.name}/y{yaw:g}_p{pitch:g}.webp'
        sheet.save(out / name, 'WEBP', quality=90, method=6)
        sheets.append({'yaw': yaw, 'pitch': pitch, 'sheet': name, 'frames': frames})
        print(f'{name}  scale {s:.3f}  {len(views)} frames')

    times = sorted({v['time'] for v in manifest['views']})
    duration = (times[1] - times[0]) * manifest['frames'] if len(times) > 1 else 1
    (out / f'{args.name}.json').write_text(json.dumps({
        'anim': manifest.get('anim'), 'frames': manifest['frames'], 'duration': round(duration, 4),
        'frame': list(size), 'cols': COLS, 'angles': sheets,
    }))
    print(f'{len(sheets)} angles -> {out / args.name}')


if __name__ == '__main__':
    main()
