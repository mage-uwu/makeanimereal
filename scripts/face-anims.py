#!/usr/bin/env python3
"""Blink, wink and talk for every forward-facing still key, by editing just her face.

    BFL_API_KEY=... python3 scripts/face-anims.py [--cone 60] [--max-pitch 45] [--model flux-2-klein-9b]
                    [--keys out/g2] [--cache out/cache2] [--renders out/g2-renders] [--web out/web2]
                    [--edits out/face] [--only y000_p15,...] [--seed 1] [--concurrency 6]

The rig has no facial animation, so these are 2D. For each key drawing within --cone degrees of
straight-on front (and no steeper than --max-pitch, where her face turns away under her bangs):

1. Her head is found by projecting the rig's head bones into the drawing (through the drawing's fit
   from build-web.py), and a square around it is cropped.
2. FLUX.2 (Black Forest Labs' API) redraws only the eyes or the mouth in that crop, at 512x512:
   cheap, and the face fills the frame.
3. Only what the edit changed is kept: a feathered patch, fading out before the crop's edges. The
   viewer draws it over the untouched key drawing, so nothing else on her ever moves.

Winks are built from the blink edits (one eye from the closed or half-closed edit, the other as
drawn): the model tends to close both eyes when asked to wink. The rest frame is the key itself.

Writes <web>/face/sheetNN.webp (one row of patches per key) and blink.json, wink.json, talk.json.
"""
import argparse
import base64
import hashlib
import json
import math
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from latent import KeyCamera  # noqa: E402

# Patch frames: 0 is the key itself (no patch).
EXPRESSIONS = [
    ('rest', None),
    ('eyes_half', 'both eyes HALF OPEN, mid-blink: the upper eyelids are lowered halfway, but the lower half of each green iris is still clearly visible under them; the eyes are NOT closed and NOT smiling curves'),
    ('eyes_closed', 'both eyes fully closed in a natural blink, eyelids shut, eyelashes drawn as gently curved lines'),
    ('wink_half', None),  # built from eyes_half
    ('wink', None),       # built from eyes_closed
    ('mouth_slight', 'her mouth slightly open as if starting to speak, a small dark gap between the lips'),
    ('mouth_ah', 'her mouth open saying "ah": an open rounded mouth showing a little of the tongue'),
    ('mouth_oh', 'her mouth in a small round "oh" shape'),
    ('mouth_ee', 'her mouth wide in an "ee" shape, a cheerful open smile showing the top teeth'),
]
F = {name: i for i, (name, _) in enumerate(EXPRESSIONS)}
FPS = 12
SEQUENCES = {
    'blink': [F['rest']] * 30 + [F['eyes_half'], F['eyes_closed'], F['eyes_closed'], F['eyes_half']],
    'wink': [F['rest']] * 14 + [F['wink_half']] + [F['wink']] * 6 + [F['wink_half']] + [F['rest']] * 6,
    'talk': [F[n] for n in ('rest', 'mouth_slight', 'mouth_ah', 'mouth_ah', 'mouth_slight', 'mouth_ee', 'mouth_ee', 'mouth_slight',
                            'mouth_oh', 'mouth_oh', 'mouth_slight', 'mouth_ah', 'mouth_slight', 'rest', 'mouth_ee', 'mouth_slight',
                            'mouth_oh', 'mouth_slight', 'rest', 'rest', 'rest', 'rest')],
}
KEEP = ('Change nothing else at all: the same face shape, hair, ribbons, colours, line style, background and '
        'framing, pixel for pixel.')
EDIT_SIZE = 512
SHEET_ROWS = 16


def bfl_edit(png_bytes, prompt, model, key, seed):
    body = {'prompt': prompt, 'input_image': base64.b64encode(png_bytes).decode(), 'width': EDIT_SIZE, 'height': EDIT_SIZE,
            'output_format': 'png', 'safety_tolerance': 2, 'seed': seed}
    headers = {'x-key': key, 'content-type': 'application/json', 'accept': 'application/json'}
    for attempt in range(4):
        try:
            job = json.load(urllib.request.urlopen(urllib.request.Request(
                f'https://api.bfl.ai/v1/{model}', data=json.dumps(body).encode(), headers=headers), timeout=60))
            while True:
                time.sleep(0.7)
                r = json.load(urllib.request.urlopen(urllib.request.Request(job['polling_url'], headers=headers), timeout=60))
                if r['status'] == 'Ready':
                    return urllib.request.urlopen(r['result']['sample'], timeout=60).read(), job.get('cost') or 0
                if r['status'] not in ('Pending', 'Processing', 'Queued'):
                    raise RuntimeError(r['status'])
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 402, 403, 422) or attempt == 3:
                raise
        except Exception:
            if attempt == 3:
                raise
        time.sleep(2 ** (attempt + 1))  # rate limits (429) and dropped connections get a few retries
    raise RuntimeError('no result after 4 attempts')


def head_square(k, view, manifest, bones, fit, size):
    """Centre and side (drawing pixels) of a square around her head: the rig's head bones projected
    through the key's camera, moved onto the drawing by its fit."""
    cam0 = manifest['cameras'][0]
    cam = KeyCamera(view['yaw'], view['pitch'], cam0['target'], cam0['radius'], manifest['fov'], manifest['width'] / manifest['height'])
    fh, fw = fit.shape[:2]
    pts = []
    for name in ('mixamorigHead', 'mixamorigHeadTop_End'):
        fx, fy, _ = cam.project(bones[name][None])
        fx, fy = float(fx[0]), float(fy[0])
        gx, gy = int(np.clip(fx * fw, 0, fw - 1)), int(np.clip(fy * fh, 0, fh - 1))
        fx, fy = fx + fit[gy, gx, 0] / fw, fy + fit[gy, gx, 1] / fh
        pts.append(np.array(((fx - k['ox']) / k['s'] * size[0], (fy - k['oy']) / k['s'] * size[1])))
    base, top = pts
    side = float(np.clip(1.9 * np.linalg.norm(top - base), 0.30 * size[0], 0.37 * size[0]))
    return (base + top) / 2, side


def changed(a, b):
    return np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(-1)


def patch_alpha(key_crop, edit_crop):
    """Where the edit changed her face, feathered, and fading out before the crop's edges."""
    m = (changed(key_crop, edit_crop) > 18).astype(np.float32)
    n = m.shape[0]
    m = cv2.dilate(m, np.ones((max(3, n // 40) | 1,) * 2, np.uint8))
    m = cv2.GaussianBlur(m, (0, 0), n / 120)
    edge = np.minimum.outer(*(np.clip(np.minimum(np.arange(n), n - 1 - np.arange(n)) / (n * 0.08), 0, 1),) * 2)
    return np.clip(m * edge, 0, 1)


def wink_from(key_crop, blink_crop, closed_crop):
    """One eye from the blink edit (the one on the image's right), the other as drawn."""
    d = changed(key_crop, closed_crop)
    n = d.shape[0]
    ys, xs = np.nonzero(d[: int(n * 0.75)] > 40)
    mid = int(np.median(xs)) if len(xs) else n // 2
    ramp = np.clip((np.arange(n) - mid) / (n / 40) + 0.5, 0, 1)[None, :, None]
    return (key_crop.astype(np.float32) * (1 - ramp) + blink_crop.astype(np.float32) * ramp).clip(0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--keys', default='out/g2')
    p.add_argument('--cache', default='out/cache2')
    p.add_argument('--renders', default='out/g2-renders')
    p.add_argument('--web', default='out/web2')
    p.add_argument('--edits', default='out/face')
    p.add_argument('--cone', type=float, default=60, help='degrees from straight-on front')
    p.add_argument('--max-pitch', type=float, default=45)
    p.add_argument('--only', help='comma-separated key stems to (re)do')
    p.add_argument('--model', default='flux-2-klein-9b')
    p.add_argument('--seed', type=int, default=1, help='change it to redraw an edit that came out wrong')
    p.add_argument('--concurrency', type=int, default=6)
    args = p.parse_args()
    keys_dir, cache, web, edits = Path(args.keys), Path(args.cache), Path(args.web), Path(args.edits)
    edits.mkdir(parents=True, exist_ok=True)
    (web / 'face').mkdir(exist_ok=True)
    api_key = os.environ.get('BFL_API_KEY')
    manifest = json.loads((Path(args.renders) / 'manifest.json').read_text())
    bones = {b['name']: np.array(b['pos']) for b in json.loads((Path(args.renders) / 'mesh.json').read_text())['skeleton']}
    views = {Path(v['file']).stem: v for v in manifest['views']}
    K = json.loads((web / 'keys.json').read_text())

    def direction(yaw, pitch):
        y, q = math.radians(yaw), math.radians(pitch)
        return np.array([math.cos(q) * math.sin(y), math.sin(q), math.cos(q) * math.cos(y)])
    cone = [k for k in K['keys'] if k['pitch'] <= args.max_pitch
            and math.degrees(math.acos(np.clip(direction(k['yaw'], k['pitch'])[2], -1, 1))) <= args.cone + 1e-6]
    cone.sort(key=lambda k: (k['pitch'], (k['yaw'] + 180) % 360))
    print(f'{len(cone)} keys within {args.cone:g} degrees of the front, up to {args.max_pitch:g} degrees high')

    # Crops and edits.
    jobs, crops = [], {}
    for k in cone:
        stem = Path(k['file']).stem
        src = Image.open(keys_dir / f'{stem}.png').convert('RGB')
        centre, side = head_square(k, views[stem], manifest, bones, np.load(cache / f'{stem}.fit.npy'), src.size)
        x0, y0, s = int(round(centre[0] - side / 2)), int(round(centre[1] - side / 2)), int(round(side))
        key_crop = src.crop((x0, y0, x0 + s, y0 + s))
        crops[stem] = (k, (x0, y0, s), src.size, key_crop)
        for name, change in EXPRESSIONS:
            f = edits / f'{stem}_{name}.crop.png'
            if change is None or (f.exists() and not (args.only and stem in args.only.split(','))):
                continue
            full = edits / f'{stem}_{name}.png'  # a whole-drawing edit from the first version
            if full.exists() and not args.only:
                Image.open(full).convert('RGB').resize(src.size, Image.LANCZOS).crop((x0, y0, x0 + s, y0 + s)).resize((EDIT_SIZE,) * 2, Image.LANCZOS).save(f)
                continue
            jobs.append((f, key_crop.resize((EDIT_SIZE,) * 2, Image.LANCZOS), change))
    if jobs and not api_key:
        sys.exit('set BFL_API_KEY')
    print(f'{len(jobs)} edits to draw')
    spent = [0.0]

    def run(job):
        f, img, change = job
        buf = __import__('io').BytesIO()
        img.save(buf, 'PNG')
        data, c = bfl_edit(buf.getvalue(), f'Edit only her face: {change}. {KEEP}', args.model, api_key, args.seed)
        f.write_bytes(data)
        spent[0] += c
        print(f'drew {f.name} ({c:g} credits)', flush=True)
    with ThreadPoolExecutor(args.concurrency) as pool:
        for r in pool.map(lambda j: _try(run, j), jobs):
            if r:
                print(r, flush=True)

    # Patches, packed one row per key.
    P = round(0.37 * 600)  # patch tile side in the web sheets (600-wide drawings)
    angles, rows = [], []
    for k, (x0, y0, s), (W, H), key_crop in crops.values():
        stem = Path(k['file']).stem
        kc = np.asarray(key_crop)
        frames = {}
        for name, change in EXPRESSIONS:
            f = edits / f'{stem}_{name}.crop.png'
            if change and f.exists():
                frames[name] = np.asarray(Image.open(f).convert('RGB').resize((s, s), Image.LANCZOS))
        if not all(n in frames for n in ('eyes_half', 'eyes_closed', 'mouth_slight', 'mouth_ah', 'mouth_oh', 'mouth_ee')):
            print(f'skip {stem}: edits missing')
            continue
        frames['wink'] = wink_from(kc, frames['eyes_closed'], frames['eyes_closed'])
        frames['wink_half'] = wink_from(kc, frames['eyes_half'], frames['eyes_closed'])
        cut_alpha = np.asarray(Image.open(cache / f'{stem}.png').convert('RGBA').resize((W, H), Image.LANCZOS))[y0:y0 + s, x0:x0 + s, 3] / 255
        row = Image.new('RGBA', (P * len(EXPRESSIONS), P), (0, 0, 0, 0))
        for i, (name, _) in enumerate(EXPRESSIONS):
            if name == 'rest':
                continue
            a = patch_alpha(kc, frames[name]) * cut_alpha[:s, :s]
            rgba = np.dstack([frames[name], (a * 255).astype(np.uint8)])
            row.paste(Image.fromarray(rgba, 'RGBA').resize((P, P), Image.LANCZOS), (i * P, 0))
        rows.append(row)
        angles.append({'yaw': k['yaw'], 'pitch': k['pitch'], 'key': k['file'], 'crop': [round(x0 / W, 5), round(y0 / H, 5), round(s / W, 5), round(s / H, 5)]})
    for f in (web / 'face').glob('*.webp'):
        f.unlink()
    for n in range(0, len(rows), SHEET_ROWS):
        name = f'face/sheet{n // SHEET_ROWS:02d}.webp'
        sheet = Image.new('RGBA', (rows[0].width, P * len(rows[n:n + SHEET_ROWS])), (0, 0, 0, 0))
        for r, row in enumerate(rows[n:n + SHEET_ROWS]):
            sheet.paste(row, (0, r * P))
            angles[n + r].update({'sheet': name, 'row': r})
        sheet.save(web / name, 'WEBP', quality=92, method=6)
        v = hashlib.md5((web / name).read_bytes()).hexdigest()[:8]
        for a in angles[n:n + SHEET_ROWS]:
            a['v'] = v
    for anim, seq in SEQUENCES.items():
        (web / f'{anim}.json').write_text(json.dumps({
            'anim': anim, 'frames': len(seq), 'duration': round(len(seq) / FPS, 4), 'fps': FPS, 'sequence': seq,
            'expressions': [n for n, _ in EXPRESSIONS], 'face': True, 'patch': P, 'angles': angles,
        }))
    print(f'{len(angles)} keys with face animations; this run cost {spent[0]:.1f} credits (${spent[0] / 100:.2f})')


def _try(fn, job):
    try:
        fn(job)
    except Exception as e:  # noqa: BLE001 - report and carry on with the rest
        return f'fail {job[0].name}: {e}'


if __name__ == '__main__':
    main()
