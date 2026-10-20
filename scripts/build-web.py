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
  tiles 1-6  stretch toward each neighbouring key drawing (lossless WebP): dense correspondences between the two
             drawings, so one drawing can be stretched toward the next with no model ("flipstretch")
  tile 7     Neuromation lattice tensor: for each vertex of a shared 75 x 112 grid laid over the
             drawing, its fixed position in the latent object (16-bit, split in high and low byte
             planes): on the ray from the drawing's learned vantage, at a smoothed depth, so each
             drawing is a gently curved rigid shell. Also whether the drawing has paint there,
             its nearest bone, and how far off her surface it lies.

Latent rotation: each drawing depicts the character from a slightly different vantage than the
camera it was asked for (the "30 degree" drawing may really be turned 27 degrees and seen from a
little higher). build-web.py learns that vantage per key (yaw and pitch offsets, roll, framing) from
the surface points visible in its render and where the drawing's fit puts them, and places every
lattice vertex on the ray from that vantage through its pixel, at the surface depth. Viewed from
its learned vantage, a lattice reproduces its drawing exactly; between keys the viewer rotates the
whole rigid latent object, with the vantage corrections blended, instead of morphing 2D warps.

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

from latent import KeyCamera, VantageCamera, read_depth, solve_vantage
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent

FIELD_RANGE = 128.0  # px at the output size; fields are stored as (d / FIELD_RANGE + 0.5) in 8 bits
FIELD_DOWNSAMPLE = 4  # fields are smooth, so quarter resolution holds them
ATLAS_COLS, ATLAS_ROWS = 4, 2  # tile 0 = fit to mesh, tiles 1..6 = neighbours, tile 7 = lattice tensor
LATTICE = (75, 112)  # Neuromation grid: vertices across and down, shared by every key
LATTICE_RANGE = 1.25  # metres either side of the target that the 16-bit latent coordinates cover
HANDOFF_GRID = (6, 9)  # control points across and down for the articulated hand-off field
HANDOFF_SIGMA = 70.0  # px at the output size
RELIEF_SIGMA = 4.0  # lattice cells (about 32 px at 600 wide) over which a drawing's depth is smoothed
RESIDUAL_RANGE = 0.08  # metres either way for a drawing's 8-bit offset from the latent surface
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
    """For each key, up to MAX_NEIGHBOURS keys around it on the view sphere, the nearest one in each
    direction: candidates (within 2.5x the nearest key's angle, so a key on a dense ring near the top
    still reaches the rings above and below) are taken nearest first, each only if it points at
    least 35 degrees away from those already taken."""
    dirs = np.array([direction(v['yaw'], v['pitch']) for v in views])
    ang = np.degrees(np.arccos(np.clip(dirs @ dirs.T, -1, 1)))
    result = []
    for i in range(len(views)):
        def heading(j):  # direction of key j from key i, on the plane touching the sphere at i
            d = dirs[j] - dirs[i]
            d = d - dirs[i] * np.dot(d, dirs[i])
            return d / (np.linalg.norm(d) + 1e-12)

        order = [j for j in np.argsort(ang[i]) if j != i]
        picked = []
        for j in order:
            if ang[i, j] > ang[i, order[0]] * 2.5 or len(picked) == MAX_NEIGHBOURS:
                break
            if all(np.dot(heading(j), heading(k)) < math.cos(math.radians(35)) for k in picked):
                picked.append(j)
        result.append(picked)
    return result


def fill_depth(dist):
    """Depth everywhere: off the surface, take the nearest surface pixel's depth, smoothed, so
    art drawn beyond the mesh (ribbons, hair) floats at the depth of the body beside it."""
    hole = np.isnan(dist)
    _, labels = cv2.distanceTransformWithLabels(hole.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(labels.max() + 1, np.float32)
    lut[labels[~hole]] = dist[~hole]
    filled = lut[labels]
    soft = cv2.GaussianBlur(filled, (0, 0), 8)
    return np.where(hole, soft, filled).astype(np.float32)


def lattice_tensor(cam, vantage, dist, fit, alpha, size, target, verts, vert_bone, raw):
    """Tile 7 for one key: where on the character each lattice vertex of the drawing sits.

    A vertex at frame point x shows the drawing's paint there. The drawing's fit says the render
    shows that feature at y with y + fit(y) = x; the render's depth map turns y into a 3D point.
    """
    W, H = size
    gx, gy = LATTICE
    X, Y = np.meshgrid(np.arange(gx) / (gx - 1), np.arange(gy) / (gy - 1))
    # The vertex is used if the drawing has paint within a cell of it.
    cell = int(math.ceil(W / (gx - 1)))
    near_paint = cv2.dilate((alpha > 0.05).astype(np.uint8), np.ones((2 * cell + 1, 2 * cell + 1), np.uint8))
    valid = near_paint[np.clip((Y * (H - 1)).round().astype(int), 0, H - 1), np.clip((X * (W - 1)).round().astype(int), 0, W - 1)] > 0
    # Invert the fit by fixed-point iteration: y = x - fit(y).
    yx, yy = (X * W).astype(np.float32), (Y * H).astype(np.float32)
    for _ in range(5):
        fx = cv2.remap(fit[..., 0], yx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        fy = cv2.remap(fit[..., 1], yx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        yx, yy = (X * W - fx).astype(np.float32), (Y * H - fy).astype(np.float32)
    dh, dw = dist.shape
    d = cv2.remap(dist, yx / W * dw, yy / H * dh, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    surface = cam.lift(yx / W, yy / H, d.astype(np.float64))
    # How far off her actual surface the vertex lies (ribbons, hair beyond the mesh), in output
    # pixels, saturating at 16: the viewer lets such art swing toward where neighbours drew it.
    off_px = cv2.distanceTransform(np.isnan(raw).astype(np.uint8), cv2.DIST_L2, 5) * (W / raw.shape[1])
    offness = np.clip(cv2.remap(off_px.astype(np.float32), yx / W * raw.shape[1], yy / H * raw.shape[0], cv2.INTER_LINEAR) / 16, 0, 1)
    # The vertex's latent position: on the ray from the drawing's learned vantage through its
    # pixel, at a smoothed depth. Exact per-pixel depth makes neighbouring points (a sleeve in
    # front of the body, hair drawn beyond the mesh) shear and tear apart as soon as the view
    # turns; a relief smoothed over RELIEF_SIGMA lattice cells turns the drawing into a gently
    # curved shell that rotates cleanly. The viewer can flatten it further, down to a flat card.
    depth_from_vantage = (surface - vantage.pos) @ vantage.forward
    weight = valid.astype(np.float32)
    smooth = cv2.GaussianBlur((depth_from_vantage * weight).astype(np.float32), (0, 0), RELIEF_SIGMA) / \
        (cv2.GaussianBlur(weight, (0, 0), RELIEF_SIGMA) + 1e-6)
    pts = vantage.lift(X, Y, smooth.astype(np.float64))
    offset = np.full(pts.shape, 0.5)  # unused since the shell is rigid; kept for the tile layout
    # Nearest bone, from the nearest mesh vertex.
    bone = np.zeros(X.shape, np.uint8)
    flat = surface[valid]
    for start in range(0, len(flat), 1024):
        chunk = flat[start:start + 1024]
        nearest = ((chunk[:, None, :] - verts[None]) ** 2).sum(-1).argmin(1)
        bone[np.nonzero(valid)[0][start:start + 1024], np.nonzero(valid)[1][start:start + 1024]] = vert_bone[nearest]
    q = np.clip((pts - np.asarray(target) + LATTICE_RANGE) / (2 * LATTICE_RANGE), 0, 1)
    q = (q * 65535 + 0.5).astype(np.uint32)
    tile = np.zeros((size[1] // FIELD_DOWNSAMPLE, size[0] // FIELD_DOWNSAMPLE, 3), np.float32)
    tile[:gy, :gx] = (q >> 8) / 255            # high bytes of x, y, z
    tile[gy:2 * gy, :gx] = (q & 255) / 255     # low bytes
    tile[:gy, gx:2 * gx, 0] = valid            # paint here
    tile[:gy, gx:2 * gx, 1] = bone / 255       # nearest bone
    tile[:gy, gx:2 * gx, 2] = offness          # 0 on her surface, 1 well off it
    tile[gy:2 * gy, gx:2 * gx] = offset         # the drawing's offset from the latent surface
    return tile, int(valid.sum())


def invert_tile(tile):
    """Exact inverse of an encoded correspondence tile: if A's feature at x sits at x + F(x) in B,
    then B's feature at y sits at y + G(y) in A, with G(y) = -F(y + G(y)). Using one field and its
    inverse for a pair (instead of two independent flows) means A stretched halfway toward B and
    B stretched halfway toward A put every pair of matching points on the same spot."""
    h, w = tile.shape[:2]
    f = (tile[..., :2].astype(np.float32) - 0.5) * FIELD_RANGE / FIELD_DOWNSAMPLE  # in tile cells
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    g = -f.copy()
    for _ in range(12):
        g = -np.dstack([cv2.remap(f[..., c], gx + g[..., 0], gy + g[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for c in (0, 1)])
    out = tile.copy()
    out[..., :2] = np.clip(g * FIELD_DOWNSAMPLE / FIELD_RANGE + 0.5, 0, 1)
    return out


def learn_vantage(view, cam0, fov, aspect, cam, raw_depth, fit, verts, size):
    """The vantage this drawing actually depicts, from where its fit puts the surface points
    visible in its render. Also reports the median distance left over, in output pixels."""
    W, H = size
    fx, fy, dist = cam.project(verts)
    dh, dw = raw_depth.shape
    px, py = np.clip((fx * dw).astype(int), 0, dw - 1), np.clip((fy * dh).astype(int), 0, dh - 1)
    seen = np.abs(raw_depth[py, px] - dist) < 0.015
    gx, gy = (fx[seen] * W).astype(np.float32), (fy[seen] * H).astype(np.float32)
    ffx = cv2.remap(fit[..., 0], gx, gy, cv2.INTER_LINEAR).ravel()
    ffy = cv2.remap(fit[..., 1], gx, gy, cv2.INTER_LINEAR).ravel()
    targets = np.stack([fx[seen] + ffx / W, fy[seen] + ffy / H], 1)
    vantage, r = solve_vantage(view, cam0, fov, aspect, verts[seen], targets)
    n = len(r) // 2
    vantage['residual_px'] = float(np.median(np.hypot(r[:n], r[n:])) * W)
    return vantage


VANTAGE_KEYS = ('d_yaw', 'd_pitch', 'roll', 'scale', 'tx', 'ty')


def smooth_vantages(views, learned, near, max_rounds=12):
    """Where two neighbouring keys' learned yaw or pitch corrections would swap their order (making
    the rotation run backwards for a moment), blend those keys' yaw and pitch corrections with
    their neighbours' until the order holds. Everything else keeps its learned value."""
    vals = [{k: float(l[k]) for k in VANTAGE_KEYS} for l in learned]

    def flipped():
        keys = set()
        for i, js in enumerate(near):
            for j in js:
                for ang, d in (('yaw', 'd_yaw'), ('pitch', 'd_pitch')):
                    a, b = views[i][ang], views[j][ang]
                    gap = (b - a + 180) % 360 - 180 if ang == 'yaw' else b - a
                    if abs(gap) > 1e-6 and (gap + vals[j][d] - vals[i][d]) * gap <= 0:
                        keys.update((i, j))
        return keys

    rounds = 0
    while (bad := flipped()) and rounds < max_rounds:
        for i in bad:
            for k in ('d_yaw', 'd_pitch'):
                vals[i][k] = 0.5 * vals[i][k] + 0.5 * float(np.mean([vals[j][k] for j in near[i]]))
        rounds += 1
    print(f'vantage field: {rounds} smoothing rounds, {len(flipped())} keys still out of order')
    return [{k: round(v[k], 5) for k in VANTAGE_KEYS} for v in vals]


def decode_lattice(tensor, target):
    """Latent positions and paint mask of a lattice tile (inverse of lattice_tensor's packing)."""
    gx, gy = LATTICE
    t = np.round(tensor * 255).astype(np.uint32)
    q = (t[:gy, :gx] << 8) + t[gy:2 * gy, :gx]
    pts = q / 65535 * 2 * LATTICE_RANGE - LATTICE_RANGE + np.asarray(target)
    return pts, t[:gy, gx:2 * gx, 0] > 127


def handoff_field(a, b, corr_tile, size, cam0, manifest, aspect):
    """What drawing a's rigid shell, seen from b's vantage, still gets wrong about where b drew
    things: x + F(x) (the a->b correspondence) minus b's vantage projection of a's latent point
    at x, as a coarse articulated field. The viewer moves a's drawing up to halfway along it by the
    hand-off, so the drawings on either side of the switch meet. Returns HANDOFF_GRID control
    points, row by row, as [dx, dy] pairs in output pixels."""
    gx, gy = LATTICE
    pts, valid = decode_lattice(a['tensor'], cam0['target'])
    vb = b['vantage']
    cam = VantageCamera(b['v']['yaw'], b['v']['pitch'], cam0['target'], cam0['radius'], manifest['fov'], aspect,
                        vb['d_yaw'], vb['d_pitch'], vb['roll'], vb['scale'], vb['tx'], vb['ty'])
    fx, fy, _ = cam.project(pts[valid])
    X, Y = np.meshgrid(np.arange(gx) / (gx - 1), np.arange(gy) / (gy - 1))
    th, tw = corr_tile.shape[:2]
    F = (corr_tile[..., :2].astype(np.float32) - 0.5) * FIELD_RANGE
    Fx = cv2.remap(F[..., 0], (X[valid] * tw).astype(np.float32), (Y[valid] * th).astype(np.float32), cv2.INTER_LINEAR).ravel() / size[0]
    Fy = cv2.remap(F[..., 1], (X[valid] * tw).astype(np.float32), (Y[valid] * th).astype(np.float32), cv2.INTER_LINEAR).ravel() / size[1]
    rx, ry = X[valid] + Fx - fx, Y[valid] + Fy - fy
    # Articulated, not local: splat the residual onto a coarse grid (control points ~100 px apart,
    # Gaussian over HANDOFF_SIGMA px), so head, torso and legs each get their own smooth move and
    # nothing smaller than a body part can deform. In output pixels, clipped.
    W, H = size
    px, py = X[valid] * W, Y[valid] * H
    rx_px, ry_px = rx * W, ry * H
    cols, rows = HANDOFF_GRID
    grid = []
    for j in range(rows):
        for i in range(cols):
            gx_, gy_ = i / (cols - 1) * W, j / (rows - 1) * H
            w = np.exp(-((px - gx_) ** 2 + (py - gy_) ** 2) / (2 * HANDOFF_SIGMA ** 2))
            total = w.sum()
            fade = total / (total + 20)  # fade to no move where the drawing has little paint nearby
            mx = float(np.clip((w * rx_px).sum() / max(total, 1e-6), -40, 40)) * fade
            my = float(np.clip((w * ry_px).sum() / max(total, 1e-6), -40, 40)) * fade
            grid += [round(mx * 2) / 2, round(my * 2) / 2]
    return grid


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
        (out / d).mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    bg = manifest.get('background', '#ffffff')
    chroma = bg not in ('#ffffff', '#fff', 'white', 'none')

    w = args.width
    size = (w, round(w * manifest['height'] / manifest['width']))
    # Drawings that can't be fixed (e.g. ones that show her face where the camera can't see it,
    # see scripts/face_check.py) are listed in <keys>/exclude.txt, one file per line with an
    # optional "# reason". Their neighbours cover the angle instead.
    exclude_file = keys_dir / 'exclude.txt'
    excluded = {line.split('#')[0].strip() for line in exclude_file.read_text().splitlines()} - {''} if exclude_file.exists() else set()
    views = [v for v in manifest['views'] if (keys_dir / v['file']).exists() and v['file'] not in excluded]
    for v in manifest['views']:
        if v not in views:
            print(f'skip {v["file"]}: {"excluded" if v["file"] in excluded else "no drawing"}')
    for f in (out / 'keys').glob('*.webp'):
        if f.with_suffix('.png').name in excluded:
            f.unlink()
    for f in (out / 'warps').glob('*.webp'):
        if f.with_suffix('.png').name in excluded:
            f.unlink()

    cam0 = manifest['cameras'][0]
    mesh = json.loads((renders / 'mesh.json').read_text())
    verts = np.asarray(mesh['position'], np.float64).reshape(-1, 3)
    vert_bone = np.asarray(mesh.get('bone', [0] * len(verts)), np.int64).clip(0, 255)
    assert 2 * LATTICE[1] <= size[1] // FIELD_DOWNSAMPLE and 2 * LATTICE[0] <= size[0] // FIELD_DOWNSAMPLE

    # Pass 1: cut out, place in the render frame, fit to the render, learn the vantage it depicts.
    aspect = manifest['width'] / manifest['height']
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
        fit_file = cache / (src.stem + '.fit.npy')
        if fit_file.exists() and fit_file.stat().st_mtime >= cached.stat().st_mtime:
            fit = np.load(fit_file)
        else:
            fit = flow_between(render, pl, (render[..., 3] > 0.5).astype(np.float32), 9.0)
            np.save(fit_file, fit)
        name = Path(v['file']).with_suffix('.webp').name
        # The key drawing itself, untouched apart from the background cut-out and resize.
        if not (out / 'keys' / name).exists() or (out / 'keys' / name).stat().st_mtime < cached.stat().st_mtime:
            Image.fromarray((drawing * 255 + 0.5).astype(np.uint8)).save(out / 'keys' / name, 'WEBP', quality=90, method=6)
        vantage_file = cache / (src.stem + '.vantage.json')
        if vantage_file.exists() and vantage_file.stat().st_mtime >= fit_file.stat().st_mtime:
            learned = json.loads(vantage_file.read_text())
        else:
            cam = KeyCamera(v['yaw'], v['pitch'], cam0['target'], cam0['radius'], manifest['fov'], aspect)
            raw = read_depth(renders / manifest['depth']['dir'] / v['file'], cam0['radius'], manifest['depth']['pad'])
            learned = learn_vantage(v, cam0, manifest['fov'], aspect, cam, raw, fit, verts, size)
            vantage_file.write_text(json.dumps(learned))
        placed.append(pl)
        items.append({'v': v, 'name': name, 's': s, 'ox': ox, 'oy': oy, 'fit': fit, 'learned': learned})
        print(f'{name}  scale {s:.3f}  offset {ox:+.3f},{oy:+.3f}  '
              f'vantage {learned["d_yaw"]:+.1f} {learned["d_pitch"]:+.1f} deg, residual {learned["residual_px"]:.1f}px')

    # The vantage corrections as a smooth field over the view sphere: each key's blended with its
    # neighbours' until no two neighbours swap order (which would make the rotation run backwards
    # for a moment). Whatever a key's own correction loses stays in its lattice, rigidly.
    near = neighbours([it['v'] for it in items])
    vantages = smooth_vantages([it['v'] for it in items], [it['learned'] for it in items], near)

    # Pass 1b: each lattice vertex's fixed place in the latent object, seen from its key's vantage.
    for it, vantage in zip(items, vantages):
        v, src = it['v'], keys_dir / it['v']['file']
        tensor_file = cache / (src.stem + '.latent.npy')
        stamp = cache / (src.stem + '.latent.json')
        if tensor_file.exists() and stamp.exists() and json.loads(stamp.read_text()) == {**vantage, 'format': 4}:
            tensor = np.load(tensor_file)
        else:
            cam = KeyCamera(v['yaw'], v['pitch'], cam0['target'], cam0['radius'], manifest['fov'], aspect)
            raw = read_depth(renders / manifest['depth']['dir'] / v['file'], cam0['radius'], manifest['depth']['pad'])
            vcam = VantageCamera(v['yaw'], v['pitch'], cam0['target'], cam0['radius'], manifest['fov'], aspect,
                                 vantage['d_yaw'], vantage['d_pitch'], vantage['roll'], vantage['scale'], vantage['tx'], vantage['ty'])
            tensor, _ = lattice_tensor(cam, vcam, fill_depth(raw), it['fit'], placed[items.index(it)][..., 3], size, cam0['target'], verts, vert_bone, raw)
            np.save(tensor_file, tensor)
            stamp.write_text(json.dumps({**vantage, 'format': 4}))
        it['tensor'], it['vantage'] = tensor, vantage

    # Pass 2: correspondences between neighbouring drawings, packed with the fit into one atlas.
    th, tw = size[1] // FIELD_DOWNSAMPLE, size[0] // FIELD_DOWNSAMPLE
    keys = []
    measured_tiles = {}

    def measured(i, j):
        """Optical-flow correspondence tile for drawing i -> drawing j (cached on disk)."""
        if (i, j) not in measured_tiles:
            stem_i, stem_j = (Path(items[n]['v']['file']).stem for n in (i, j))
            cached_pair = cache / f'{stem_i}__{stem_j}.npy'
            newest = max((cache / (Path(items[n]['v']['file']).stem + '.png')).stat().st_mtime for n in (i, j))
            if cached_pair.exists() and cached_pair.stat().st_mtime >= newest:
                tile = np.load(cached_pair).astype(np.float32)
            else:
                tile = encode_tile(flow_between(placed[i], placed[j], (placed[i][..., 3] > 0.5).astype(np.float32), 5.0), size)
                np.save(cached_pair, tile.astype(np.float16))
            measured_tiles[(i, j)] = tile
        return measured_tiles[(i, j)]

    for i, it in enumerate(items):
        atlas = np.full((th * ATLAS_ROWS, tw * ATLAS_COLS, 3), 0.5, np.float32)
        tiles = [encode_tile(it['fit'], size)]
        for j in near[i]:
            # The lower-numbered key of each pair owns the measured flow; the other gets its inverse.
            tiles.append(measured(i, j) if i < j or i not in near[j] else invert_tile(measured(j, i)))
        # One global affine per neighbour for the hand-off (see handoff_affine).
        handoffs = [handoff_field(it, items[j], tiles[1 + n], size, cam0, manifest, aspect) for n, j in enumerate(near[i])]
        while len(tiles) < ATLAS_COLS * ATLAS_ROWS - 1:
            tiles.append(np.full_like(tiles[0], 0.5))
        tiles.append(it['tensor'])
        for t, tile in enumerate(tiles):
            r, c = divmod(t, ATLAS_COLS)
            atlas[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = tile
        wname = Path(it['v']['file']).with_suffix('.webp').name
        # Lossless: the fields are displacements, and lossy noise would make the drawing shimmer.
        Image.fromarray((atlas * 255 + 0.5).astype(np.uint8)).save(out / 'warps' / wname, 'WEBP', lossless=True, quality=100, method=6)
        keys.append({'file': f'keys/{it["name"]}', 'warps': f'warps/{wname}', 'yaw': it['v']['yaw'], 'pitch': it['v']['pitch'],
                     's': round(it['s'], 5), 'ox': round(it['ox'], 5), 'oy': round(it['oy'], 5),
                     'neighbours': [int(j) for j in near[i]],
                     'vantage': it['vantage'],
                     'handoff': handoffs})
        print(f'{wname}  {len(near[i])} neighbours')

    cam = manifest['cameras'][0]
    (out / 'keys.json').write_text(json.dumps({
        'fov': manifest['fov'], 'aspect': manifest['width'] / manifest['height'],
        'target': cam['target'], 'radius': cam['radius'], 'pose': manifest.get('pose', 'rest'),
        'fieldRange': FIELD_RANGE, 'frame': list(size), 'atlas': [ATLAS_COLS, ATLAS_ROWS],
        'handoffGrid': list(HANDOFF_GRID),
        'lattice': {'size': list(LATTICE), 'range': LATTICE_RANGE, 'residual': RESIDUAL_RANGE, 'tile': ATLAS_COLS * ATLAS_ROWS - 1},
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
