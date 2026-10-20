#!/usr/bin/env python3
"""Package the web viewer: key drawings cut out, how each fits the 3D model and its neighbours,
and the model's posed shape.

    python3 scripts/build-web.py [--renders out/relaxed-renders] [--alpha out/relaxed-alpha]
                                 [--keys out/relaxed] [--out out/web] [--width 600]

--renders  manifest with the keyframe cameras and mesh.json (render.mjs --export-mesh)
--alpha    the same views rendered with --background none, used for the model's silhouettes
--keys     the stylized drawings (from stylize.mjs)

Key drawings ship untouched: the viewer shows them exactly as drawn on their key angle. Beside each
one goes a warp atlas (warps/*.png), a few smooth displacement fields at quarter resolution:

  tile 0     fit to mesh: moves the drawing's features (eyes, bow, buttons, hems) onto the render's,
             so the hidden mesh can bend it ("projected" mode)
  tiles 1-6  stretch toward each neighbouring key drawing: dense correspondences between the two
             drawings, so one drawing can be stretched toward the next with no model ("flipstretch")

A drawing on a flat chroma background (the manifest's background colour) is keyed out; one on
white falls back to a flood fill from the border.

Needs Pillow (with WebP), numpy and opencv-python-headless.
"""
import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent

FIELD_RANGE = 128.0  # px at the output size; fields are stored as (d / FIELD_RANGE + 0.5) in 8 bits
FIELD_DOWNSAMPLE = 4  # fields are smooth, so quarter resolution holds them
ATLAS_COLS, ATLAS_ROWS = 4, 2  # tile 0 = fit to mesh, tiles 1..6 = neighbours, tile 7 unused
MAX_NEIGHBOURS = 6


def cut_out_white(im):
    """Alpha mask for a drawing on a flat white background (gen 1, before chroma backgrounds).

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
    alpha = Image.fromarray(np.where(filled == 128, 0, 255).astype(np.uint8))
    # Pull the edge in by a pixel so no rim of the old white background survives (it shows as a
    # halo on dark backdrops), then soften it so edges don't alias.
    alpha = alpha.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.7))
    out = im.convert('RGB').copy()
    out.putalpha(alpha)
    return out


def cut_out_chroma(im, key_rgb):
    """RGBA drawing from one on a flat chroma background: key out the colour, then remove its spill
    from the edges so no magenta fringe survives on other backdrops."""
    a = np.asarray(im.convert('RGB')).astype(np.float32) / 255
    border = np.concatenate([a[:8].reshape(-1, 3), a[-8:].reshape(-1, 3), a[:, :8].reshape(-1, 3), a[:, -8:].reshape(-1, 3)])
    key = np.median(border, axis=0) if key_rgb is None else np.float32(key_rgb) / 255
    dist = np.linalg.norm(a - key, axis=2)
    alpha = np.clip((dist - 0.16) / (0.32 - 0.16), 0, 1)
    alpha = cv2.GaussianBlur(cv2.erode(alpha, np.ones((3, 3), np.uint8)), (0, 0), 0.7)
    # Spill: the key's dominant channels pushed above the others. Pull them back toward the rest,
    # fully at the edge and not at all inside the figure.
    hi = key > key.mean()
    rest = a[..., ~hi].mean(axis=2, keepdims=True) if (~hi).any() else a.mean(axis=2, keepdims=True)
    fixed = a.copy()
    fixed[..., hi] = np.minimum(a[..., hi], rest + (a[..., hi] - rest) * alpha[..., None])
    rgba = np.dstack([fixed, alpha])
    return Image.fromarray((rgba * 255 + 0.5).astype(np.uint8), 'RGBA')


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


def gray(rgba):
    """Luma on mid grey, so silhouettes register as well as interior lines."""
    g = (rgba[..., :3] * rgba[..., 3:] + 0.5 * (1 - rgba[..., 3:])) @ np.float32([0.3, 0.59, 0.11])
    return (g * 255).astype(np.uint8)


def smooth_flow(flow, weight, sigma, far_sigma=30):
    """Smooth a flow field where `weight` says it is trustworthy, easing into the local average
    elsewhere, so the warp bends a drawing like a rubber sheet instead of tearing lineart."""
    def spread(s):
        return cv2.GaussianBlur(flow * weight[..., None], (0, 0), s) / (cv2.GaussianBlur(weight, (0, 0), s)[..., None] + 1e-4)
    near = cv2.GaussianBlur(weight, (0, 0), 3)[..., None]
    return near * spread(sigma) + (1 - near) * spread(far_sigma)


def place(drawing, s, ox, oy):
    """A drawing (float RGBA) moved into the render frame by its silhouette fit."""
    h, w = drawing.shape[:2]
    m = np.float32([[s, 0, ox * w], [0, s, oy * h]])
    return cv2.warpAffine(drawing, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


def flow_between(a, b, weight, sigma):
    """Field F on a's pixels: a's feature at x sits at x + F(x) in b. Both RGBA in the same frame."""
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(gray(a), gray(b), None)
    return smooth_flow(flow, weight, sigma)


def direction(yaw, pitch):
    y, p = math.radians(yaw), math.radians(pitch)
    return np.array([math.cos(p) * math.sin(y), math.sin(p), math.cos(p) * math.cos(y)])


def neighbours(views):
    """For each key, the keys around it on the view sphere: within 1.6x its nearest neighbour's
    angle, at most MAX_NEIGHBOURS, picked to spread around it rather than bunch on one side."""
    dirs = np.array([direction(v['yaw'], v['pitch']) for v in views])
    ang = np.degrees(np.arccos(np.clip(dirs @ dirs.T, -1, 1)))
    result = []
    for i in range(len(views)):
        order = [j for j in np.argsort(ang[i]) if j != i]
        nearest = ang[i, order[0]]
        cands = [j for j in order if ang[i, j] <= nearest * 1.6]
        picked = []
        for j in cands:
            # skip a candidate almost in the same direction as one already picked
            dj = dirs[j] - dirs[i]
            if all(np.dot(dj, dirs[k] - dirs[i]) / (np.linalg.norm(dj) * np.linalg.norm(dirs[k] - dirs[i]) + 1e-9) < 0.94 for k in picked):
                picked.append(j)
            if len(picked) == MAX_NEIGHBOURS:
                break
        result.append(picked)
    return result


def encode_tile(field, size):
    small = cv2.resize(field, (size[0] // FIELD_DOWNSAMPLE, size[1] // FIELD_DOWNSAMPLE), interpolation=cv2.INTER_AREA)
    enc = np.clip(small / FIELD_RANGE + 0.5, 0, 1)
    return np.dstack([enc[..., 0], enc[..., 1], np.full_like(enc[..., 0], 0.5)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--renders', default='out/relaxed-renders')
    p.add_argument('--alpha', default='out/relaxed-alpha')
    p.add_argument('--keys', default='out/relaxed')
    p.add_argument('--out', default='out/web')
    p.add_argument('--cache', default='out/cache')
    p.add_argument('--width', type=int, default=600)
    args = p.parse_args()

    renders, alpha_dir, keys_dir, out, cache = (Path(x) for x in (args.renders, args.alpha, args.keys, args.out, args.cache))
    manifest = json.loads((renders / 'manifest.json').read_text())
    for d in ('keys', 'warps'):
        shutil.rmtree(out / d, ignore_errors=True)
        (out / d).mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    bg = manifest.get('background', '#ffffff')
    chroma = bg not in ('#ffffff', '#fff', 'white', 'none')

    w = args.width
    size = (w, round(w * manifest['height'] / manifest['width']))
    views = [v for v in manifest['views'] if (keys_dir / v['file']).exists()]
    for v in manifest['views']:
        if v not in views:
            print(f'skip {v["file"]}: no drawing')

    # Pass 1: cut out, place in the render frame, fit to the render.
    placed, items = [], []
    for v in views:
        src = keys_dir / v['file']
        cached = cache / (src.stem + '.png')
        if cached.exists() and cached.stat().st_mtime >= src.stat().st_mtime:
            cut = Image.open(cached)
        else:
            im = Image.open(src).convert('RGB')
            # Key on each drawing's own background: the model's magenta varies a little from the request.
            cut = cut_out_chroma(im, None) if chroma else cut_out_white(im)
            cut.save(cached)
        rt, rb, rx = extent(Image.open(alpha_dir / v['file']).getchannel('A'))
        st, sb, sx = extent(cut.getchannel('A'))
        # Map from render image coords to drawing coords: drawing = (render - o) / s.
        s = (rb - rt) / (sb - st)
        ox, oy = rx - sx * s, rt - st * s
        drawing = np.asarray(cut.resize(size, Image.LANCZOS)).astype(np.float32) / 255
        render = np.asarray(Image.open(alpha_dir / v['file']).convert('RGBA').resize(size, Image.LANCZOS)).astype(np.float32) / 255
        pl = place(drawing, s, ox, oy)
        fit = flow_between(render, pl, (render[..., 3] > 0.5).astype(np.float32), 9.0)
        name = Path(v['file']).with_suffix('.webp').name
        # The key drawing itself, untouched apart from the background cut-out and resize.
        Image.fromarray((drawing * 255 + 0.5).astype(np.uint8)).save(out / 'keys' / name, 'WEBP', quality=90, method=6)
        placed.append(pl)
        items.append({'v': v, 'name': name, 's': s, 'ox': ox, 'oy': oy, 'fit': fit})
        print(f'{name}  scale {s:.3f}  offset {ox:+.3f},{oy:+.3f}')

    # Pass 2: correspondences between neighbouring drawings, packed with the fit into one atlas.
    near = neighbours([it['v'] for it in items])
    th, tw = size[1] // FIELD_DOWNSAMPLE, size[0] // FIELD_DOWNSAMPLE
    keys = []
    for i, it in enumerate(items):
        atlas = np.full((th * ATLAS_ROWS, tw * ATLAS_COLS, 3), 0.5, np.float32)
        tiles = [encode_tile(it['fit'], size)]
        for j in near[i]:
            weight = np.maximum(placed[i][..., 3], 0.0)
            tiles.append(encode_tile(flow_between(placed[i], placed[j], (weight > 0.5).astype(np.float32), 5.0), size))
        for t, tile in enumerate(tiles):
            r, c = divmod(t, ATLAS_COLS)
            atlas[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = tile
        wname = Path(it['v']['file']).with_suffix('.png').name
        Image.fromarray((atlas * 255 + 0.5).astype(np.uint8)).save(out / 'warps' / wname)
        keys.append({'file': f'keys/{it["name"]}', 'warps': f'warps/{wname}', 'yaw': it['v']['yaw'], 'pitch': it['v']['pitch'],
                     's': round(it['s'], 5), 'ox': round(it['ox'], 5), 'oy': round(it['oy'], 5),
                     'neighbours': [int(j) for j in near[i]]})
        print(f'{wname}  {len(near[i])} neighbours')

    cam = manifest['cameras'][0]
    (out / 'keys.json').write_text(json.dumps({
        'fov': manifest['fov'], 'aspect': manifest['width'] / manifest['height'],
        'target': cam['target'], 'radius': cam['radius'], 'pose': manifest.get('pose', 'rest'),
        'fieldRange': FIELD_RANGE, 'frame': list(size), 'atlas': [ATLAS_COLS, ATLAS_ROWS],
        'keys': keys,
    }, indent=1))
    # The viewer never shows the model, so it gets only the posed shape (render.mjs --export-mesh).
    shutil.copy(renders / 'mesh.json', out / 'mesh.json')
    shutil.copy(ROOT / 'web' / 'index.html', out / 'index.html')
    for stale in ('model.glb', 'pose.js', 'fields'):
        target = out / stale
        shutil.rmtree(target, ignore_errors=True) if target.is_dir() else target.unlink(missing_ok=True)
    total = sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
    print(f'{len(keys)} keys -> {out} ({total / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
