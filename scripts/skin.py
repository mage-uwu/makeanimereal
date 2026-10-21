#!/usr/bin/env python3
"""Stitch the painted mound pictures (scripts/mounds.py) into one 360-degree strip of painted hillside,
the skin that web/index.html projects onto the Bliss field's distant terrain.

    python3 scripts/skin.py [--mounds out/mounds] [--out out/web2/scene/bliss-skin.webp]

Each picture is cropped below its ridge (green only), and the pictures are laid side by side with
weighted cross-fades, wrapping round, so the strip has no seams.
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

spec = importlib.util.spec_from_file_location('build_web', Path(__file__).with_name('build-web.py'))
bw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bw)

TW, TH, F = 1024, 512, 192  # tile size, cross-fade width


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mounds', default='out/mounds')
    p.add_argument('--out', default='out/web2/scene/bliss-skin.webp')
    args = p.parse_args()
    tiles = []
    for f in sorted(Path(args.mounds).glob('mound*.png')):
        im = Image.open(f).convert('RGB')
        a = np.asarray(bw.cut_out_chroma(im, None))[..., 3] > 128
        W, H = im.size
        x0, x1 = int(W * 0.12), int(W * 0.88)
        y0 = int(np.percentile([np.argmax(a[:, x]) for x in range(x0, x1)], 92)) + 20
        tiles.append(np.asarray(im.crop((x0, y0, x1, H)).resize((TW, TH), Image.LANCZOS)).astype(np.float32))
    step = TW - F
    width = step * len(tiles)
    acc = np.zeros((TH, width, 3), np.float32)
    wsum = np.zeros((1, width, 1), np.float32)
    x = np.arange(TW, dtype=np.float32)
    w = np.clip(np.minimum((x + 0.5) / F, (TW - x - 0.5) / F), 0, 1)
    w = (w * w * (3 - 2 * w))[None, :, None] + 1e-4
    for i, t in enumerate(tiles):
        cols = (i * step + np.arange(TW)) % width
        np.add.at(acc, (slice(None), cols), t * w)
        np.add.at(wsum, (slice(None), cols), w)
    out = (acc / wsum).clip(0, 255).astype(np.uint8)
    Image.fromarray(out).save(args.out, 'WEBP', quality=88, method=6)
    print(f'{len(tiles)} pictures -> {args.out} ({width}x{TH})')


if __name__ == '__main__':
    main()
