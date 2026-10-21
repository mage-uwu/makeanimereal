#!/usr/bin/env python3
"""Flag drawings where the outfit went missing: a bare, skin-toned torso instead of her clothes.

    python3 scripts/qc-scan.py [out/walk out/g2 ...] [--limit 0.45]

Scores each drawing by the share of skin-toned pixels across her torso (22-55% of the figure's
height, chroma background excluded). Dressed drawings score about 0.1-0.3 (the pale vest and
blouse sit near skin tone); the one undressed frame found so far scored 0.64. Prints the drawings
over --limit with their key IDs, ready for scripts/redraw.mjs, and the highest scorers for a look.
"""
import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image


def torso_skin(path):
    a = np.asarray(Image.open(path).convert('RGB').resize((208, 312))).astype(float)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    bg = (r > 140) & (b > 110) & (g < 90)
    skin = (r > 200) & (g > 165) & (b > 145) & (r - b > 18) & (r - b < 70) & (g < r)
    rows = np.nonzero((~bg).any(1))[0]
    top, h = rows[0], rows[-1] - rows[0]
    band = slice(top + int(0.22 * h), top + int(0.55 * h))
    fig = (~bg)[band]
    return (skin[band] & fig).sum() / max(1, fig.sum())


def key_id(path):
    m = re.match(r'(?:(\w+)_f(\d+)_)?y([\d.]+)_p(-?[\d.]+)$', path.stem)
    anim, frame, yaw, pitch = m.groups()
    yaw = f'{float(yaw):g}'
    return f'{anim.lower()}/f{frame}/y{yaw}/p{pitch}' if anim else f'still/y{yaw}/p{pitch}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dirs', nargs='*', default=['out/walk', 'out/g2'])
    p.add_argument('--limit', type=float, default=0.45)
    args = p.parse_args()
    scores = sorted(((torso_skin(f), f) for d in args.dirs for f in Path(d).glob('*.png') if f.parent.name != 'replaced'), reverse=True)
    flagged = [(s, f) for s, f in scores if s > args.limit]
    print(f'{len(scores)} drawings; median {np.median([s for s, _ in scores]):.3f}; {len(flagged)} over {args.limit}')
    for s, f in flagged:
        print(f'  FLAG {s:.3f}  {key_id(f)}')
    for s, f in scores[:5]:
        print(f'  top  {s:.3f}  {key_id(f)}')


if __name__ == '__main__':
    main()
