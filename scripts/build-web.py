#!/usr/bin/env python3
"""Package the web viewer: key drawings cut out and fitted to the 3D model, plus its posed shape.

    python3 scripts/build-web.py [--renders out/relaxed-renders] [--alpha out/relaxed-alpha]
                                 [--keys out/relaxed] [--out out/web] [--width 600]

--renders  manifest with the keyframe cameras and mesh.json (render.mjs --export-mesh)
--alpha    the same views rendered with --background none, used for the model's silhouettes
--keys     the stylized drawings (from stylize.mjs)

Each drawing is fitted to its render in two steps: a scale and offset from the silhouettes, then a
smooth optical-flow warp that moves its features (eyes, bow, buttons, hems) onto the same features
of the render. The viewer bends the fitted drawing with the mesh, so a drawing that disagrees with
the mesh would paint the wrong art onto the wrong part of the body.

Needs Pillow (with WebP), numpy and opencv-python-headless.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent


def cut_out(im):
    """Alpha mask for a drawing on a flat light background.

    Flood-fills near-white, low-saturation pixels connected to the image border. The lineart
    closes the figure off, so white socks and sleeves inside it stay opaque.
    """
    a = np.asarray(im.convert('RGB')).astype(np.int16)
    lo, hi = a.min(axis=2), a.max(axis=2)
    # copy(): an image made by fromarray shares numpy's buffer, and floodfill's writes would be lost.
    bgish = Image.fromarray(np.where((lo > 232) & (hi - lo < 16), 255, 0).astype(np.uint8)).copy()
    w, h = bgish.size
    seeds = [(x, y) for x in range(0, w, 16) for y in (0, h - 1)] + [(x, y) for y in range(0, h, 16) for x in (0, w - 1)]
    for seed in seeds:
        if bgish.getpixel(seed) == 255:
            ImageDraw.floodfill(bgish, seed, 128, thresh=0)
    # Pockets the fill can't reach (background enclosed by a ribbon loop, between arm and body)
    # are found on a much stricter mask: within a few levels of the background colour itself.
    # That separates them from white cloth, whose cel shading sits a little darker.
    bg = np.median(a[np.asarray(bgish) == 128], axis=0)
    near = (np.abs(a - bg).max(axis=2) <= 3) & (np.asarray(bgish) != 128)
    strict = Image.fromarray(np.where(near, 255, 0).astype(np.uint8)).copy()
    filled = np.asarray(bgish).copy()
    for y in range(0, h, 6):
        for x in range(0, w, 6):
            if strict.getpixel((x, y)) != 255:
                continue
            ImageDraw.floodfill(strict, (x, y), 64, thresh=0)
            region = strict.point(lambda v: 255 if v == 64 else 0)
            box = region.getbbox()
            if (box[2] - box[0]) * (box[3] - box[1]) > 600:
                pocket = np.asarray(region.crop(box)) == 255
                if pocket.sum() > 600:
                    filled[box[1]:box[3], box[0]:box[2]][pocket] = 128
            ImageDraw.floodfill(strict, (x, y), 0, thresh=0)
    bgish = Image.fromarray(filled)
    alpha = Image.fromarray(np.where(np.asarray(bgish) == 128, 0, 255).astype(np.uint8))
    # Pull the edge in by a pixel so no rim of the old white background survives (it shows as a
    # halo on dark backdrops), then soften it so edges don't alias.
    return alpha.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.7))


def extent(mask):
    """Top, bottom and leg-centre x of a silhouette, as fractions of the image size."""
    m = np.asarray(mask) > 127
    h, w = m.shape
    rows = np.where(m.any(axis=1))[0]
    top, bot = rows[0] / h, (rows[-1] + 1) / h
    # Knees to shoes: below the ribbons and charms, so only the body sets the horizontal centre.
    y0 = int((bot - (bot - top) * 0.22) * h)
    xs = np.where(m[y0:int(bot * h)])[1]
    return top, bot, xs.mean() / w


def fit_to_render(drawing, render, s, ox, oy, smooth=9.0):
    """Warp an RGBA drawing (float 0..1) onto an RGBA render of the same size.

    Returns the drawing in render coordinates with its features moved onto the render's.
    """
    h, w = render.shape[:2]
    m = np.float32([[s, 0, ox * w], [0, s, oy * h]])
    placed = cv2.warpAffine(drawing, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))

    def gray(rgba):  # on mid grey, so silhouettes register as well as interior lines
        g = (rgba[..., :3] * rgba[..., 3:] + 0.5 * (1 - rgba[..., 3:])) @ np.float32([0.3, 0.59, 0.11])
        return (g * 255).astype(np.uint8)

    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(gray(render), gray(placed), None)
    # Trust the flow only on the body, smoothed so the warp bends the drawing like a rubber sheet
    # instead of tearing lineart. Off the body (ribbons) the warp eases into the body's average.
    body = (render[..., 3] > 0.5).astype(np.float32)

    def spread(sigma):
        return cv2.GaussianBlur(flow * body[..., None], (0, 0), sigma) / (cv2.GaussianBlur(body, (0, 0), sigma)[..., None] + 1e-4)

    near = cv2.GaussianBlur(body, (0, 0), 3)[..., None]
    field = near * spread(smooth) + (1 - near) * spread(30)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    fitted = cv2.remap(placed, gx + field[..., 0], gy + field[..., 1], cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    return fitted, float(np.median(np.hypot(field[..., 0], field[..., 1])[body > 0]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--renders', default='out/relaxed-renders')
    p.add_argument('--alpha', default='out/relaxed-alpha')
    p.add_argument('--keys', default='out/relaxed')
    p.add_argument('--out', default='out/web')
    p.add_argument('--width', type=int, default=600)
    args = p.parse_args()

    renders, alpha_dir, keys_dir, out = (Path(x) for x in (args.renders, args.alpha, args.keys, args.out))
    manifest = json.loads((renders / 'manifest.json').read_text())
    (out / 'keys').mkdir(parents=True, exist_ok=True)

    keys = []
    for v in manifest['views']:
        src = keys_dir / v['file']
        if not src.exists():
            print(f'skip {v["file"]}: no drawing')
            continue
        im = Image.open(src).convert('RGB')
        mask = cut_out(im)
        rt, rb, rx = extent(Image.open(alpha_dir / v['file']).getchannel('A'))
        st, sb, sx = extent(mask)
        # Map from render image coords to drawing coords: drawing = (render - o) / s.
        s = (rb - rt) / (sb - st)
        ox, oy = rx - sx * s, rt - st * s

        w = args.width
        size = (w, round(w * manifest['height'] / manifest['width']))
        rgba = im.copy()
        rgba.putalpha(mask)
        drawing = np.asarray(rgba.resize(size, Image.LANCZOS)).astype(np.float32) / 255
        render = np.asarray(Image.open(alpha_dir / v['file']).convert('RGBA').resize(size, Image.LANCZOS)).astype(np.float32) / 255
        fitted, moved = fit_to_render(drawing, render, s, ox, oy)
        name = Path(v['file']).with_suffix('.webp').name
        Image.fromarray((np.clip(fitted, 0, 1) * 255 + 0.5).astype(np.uint8)).save(out / 'keys' / name, 'WEBP', quality=86, method=6)
        # Fitted drawings already sit in the render's frame, so the viewer needs no per-key offset.
        keys.append({'file': f'keys/{name}', 'yaw': v['yaw'], 'pitch': v['pitch'], 's': 1, 'ox': 0, 'oy': 0})
        print(f'{name}  scale {s:.3f}  offset {ox:+.3f},{oy:+.3f}  warp {moved:.1f}px')

    cam = manifest['cameras'][0]
    (out / 'keys.json').write_text(json.dumps({
        'fov': manifest['fov'], 'aspect': manifest['width'] / manifest['height'],
        'yawStep': manifest['yawStep'], 'pitches': sorted(manifest['pitches']),
        'target': cam['target'], 'radius': cam['radius'], 'pose': manifest.get('pose', 'rest'),
        'keys': keys,
    }, indent=1))
    # The viewer never shows the model, so it gets only the posed shape (render.mjs --export-mesh).
    shutil.copy(renders / 'mesh.json', out / 'mesh.json')
    shutil.copy(ROOT / 'web' / 'index.html', out / 'index.html')
    for stale in ('model.glb', 'pose.js'):
        (out / stale).unlink(missing_ok=True)
    size = sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
    print(f'{len(keys)} keys -> {out} ({size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
