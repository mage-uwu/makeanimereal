#!/usr/bin/env python3
"""Pack shader datasets (build_dataset.py folders) into one raw file for shader_nts.c.

    python3 pack_data.py out/shader-data/w1 out/shader-data/bottom --out data.bin [--every 1] [--limit N] [--held]

Per frame, per pixel, 17 bytes straight from the PNGs: colour rgb, normal rgb, depth, uva rgb
(u high, u low, v high), uvb rgb (v low, part, coverage), target rgba. shader_nts.c decodes them
exactly as train.py's load() does. Header: 'NTSD', frames, height, width, parts (int32), then each
frame's camera pitch (float32), then the frames. --out /dev/stdout streams the pack straight into
shader_nts --data /dev/stdin (progress goes to stderr), for sets larger than the disk.
"""
import argparse
import json
import sys
import struct
from pathlib import Path

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('dirs', nargs='+')
ap.add_argument('--out', required=True)
ap.add_argument('--every', type=int, default=1)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--held', action='store_true', help="only shader_nts.c's held-out frames (index % 10 == 5)")
a = ap.parse_args()
frames, parts, size = [], None, None
for d in a.dirs:
    meta = json.loads((Path(d) / 'meta.json').read_text())
    parts = parts or len(meta['parts'])
    size = size or meta['size']
    frames += [(Path(d) / f'{f["frame"]:04d}', f['pitch']) for f in meta['frames']][::a.every]
if a.held:
    frames = frames[5::10]
if a.limit:
    frames = frames[:a.limit]
W, H = size
with open(a.out, 'wb') as out:
    out.write(b'NTSD' + struct.pack('<4i', len(frames), H, W, parts))
    out.write(np.array([p for _, p in frames], np.float32).tobytes())
    for k, (name, _) in enumerate(frames):
        img = lambda s, c: np.asarray(Image.open(f'{name}_{s}.png').convert(c))
        px = np.concatenate([img('color', 'RGB'), img('normal', 'RGB'), img('depth', 'RGB')[..., :1],
                             img('uva', 'RGB'), img('uvb', 'RGB'), img('target', 'RGBA')], -1)
        assert px.shape == (H, W, 17), px.shape
        out.write(np.ascontiguousarray(px, np.uint8).tobytes())
        if k % 200 == 0:
            print(k, '/', len(frames), file=sys.stderr, flush=True)
print(f'{len(frames)} frames {W}x{H}, {parts} parts -> {a.out}', file=sys.stderr)
