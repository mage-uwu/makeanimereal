#!/usr/bin/env python3
"""Blink, wink and talk animations for the front views, made by editing the still key drawings.

    BFL_API_KEY=... python3 scripts/face-anims.py [--keys out/g2] [--cache out/cache2] [--web out/web2]
                    [--angles 0:0,0:15] [--model flux-2-klein-9b] [--edits out/face]

The rig has no facial animation, so these are 2D: FLUX.2 (Black Forest Labs' API) redraws only her
eyes or mouth on a copy of the key drawing. The edit would also nudge lines everywhere else by a
pixel or two, which flickers in playback, so only the face is taken from it: wherever the edit
differs from the key inside the face, feathered, pasted onto the untouched key cut-out. The rest
frame is the key drawing itself, so on a key angle the animation starts and ends exactly as drawn.

Writes <web>/face/y<yaw>_p<pitch>.webp (all expressions in one sheet per angle) and blink.json,
wink.json and talk.json: the same layout as walk.json, plus a "sequence" of sheet frames played at
"fps", so held frames cost nothing.
"""
import argparse
import base64
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Sheet frames: 0 is the key drawing itself.
EXPRESSIONS = [
    ('rest', None),
    ('eyes_half', 'both eyes HALF OPEN, mid-blink: the upper eyelids are lowered halfway, but the lower half of each green iris is still clearly visible under them; the eyes are NOT closed and NOT smiling curves'),
    ('eyes_closed', 'both eyes fully closed in a natural blink, eyelids shut, eyelashes drawn as gently curved lines'),
    ('wink_half', 'a half wink: ONLY the eye on the RIGHT side of the image is half closed; the eye on the LEFT side of the image stays WIDE OPEN and unchanged, its green iris and highlight fully visible'),
    ('wink', 'a one-eyed WINK: ONLY the eye on the RIGHT side of the image is closed, drawn as a curved line; the eye on the LEFT side of the image stays WIDE OPEN and unchanged, its green iris and highlight fully visible; a small playful smile. Exactly one eye open and one eye closed'),
    ('mouth_slight', 'her mouth slightly open as if starting to speak, a small dark gap between the lips'),
    ('mouth_ah', 'her mouth open saying "ah": an open rounded mouth showing a little of the tongue'),
    ('mouth_oh', 'her mouth in a small round "oh" shape'),
    ('mouth_ee', 'her mouth wide in an "ee" shape, a cheerful open smile showing the top teeth'),
]
F = {name: i for i, (name, _) in enumerate(EXPRESSIONS)}
FPS = 12
SEQUENCES = {
    'blink': [F['rest']] * 30 + [F['eyes_half'], F['eyes_closed'], F['eyes_closed'], F['eyes_half']],
    'wink': [F['rest']] * 14 + [F['wink_half'], F['wink'], F['wink'], F['wink'], F['wink'], F['wink'], F['wink'], F['wink_half']] + [F['rest']] * 6,
    'talk': [F[n] for n in ('rest', 'mouth_slight', 'mouth_ah', 'mouth_ah', 'mouth_slight', 'mouth_ee', 'mouth_ee', 'mouth_slight',
                            'mouth_oh', 'mouth_oh', 'mouth_slight', 'mouth_ah', 'mouth_slight', 'rest', 'mouth_ee', 'mouth_slight',
                            'mouth_oh', 'mouth_slight', 'rest', 'rest', 'rest', 'rest')],
}
KEEP = ('Change nothing else at all: the same face shape, hair, ribbons, pose, outfit, colours, line style, '
        'background and framing, pixel for pixel.')


def bfl_edit(src, prompt, model, key, seed=1):
    w, h = Image.open(src).size
    body = {'prompt': prompt, 'input_image': base64.b64encode(Path(src).read_bytes()).decode(), 'width': w, 'height': h,
            'output_format': 'png', 'safety_tolerance': 2, 'seed': seed}
    headers = {'x-key': key, 'content-type': 'application/json', 'accept': 'application/json'}
    job = json.load(urllib.request.urlopen(urllib.request.Request(f'https://api.bfl.ai/v1/{model}', data=json.dumps(body).encode(), headers=headers)))
    while True:
        time.sleep(0.7)
        r = json.load(urllib.request.urlopen(urllib.request.Request(job['polling_url'], headers=headers)))
        if r['status'] == 'Ready':
            return urllib.request.urlopen(r['result']['sample']).read(), job.get('cost') or 0
        if r['status'] not in ('Pending', 'Processing', 'Queued'):
            raise RuntimeError(f'{src}: {r["status"]}')


def face_box(key_rgb, closed_rgb):
    """Her face, found from where the closed-eyes edit changed the eyes: widened to the cheeks and
    stretched down past the mouth."""
    d = np.abs(key_rgb.astype(np.float32) - closed_rgb.astype(np.float32)).mean(-1)
    h = d.shape[0]
    ys, xs = np.nonzero(cv2.morphologyEx((d[:int(h * 0.4)] > 40).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)))
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    eh, ew = y1 - y0, x1 - x0
    return int(max(0, y0 - eh * 0.8)), int(y1 + eh * 3.2), int(max(0, x0 - ew * 0.15)), int(x1 + ew * 0.15)


def synth_wink(edits, stem, key_rgb):
    """A wink built from the blink edits, for angles where the model closes both eyes when asked to
    wink: the eye on the image's right from the (half-)closed edit, the other eye as drawn."""
    for wink, blink in (('wink', 'eyes_closed'), ('wink_half', 'eyes_half')):
        src = np.asarray(Image.open(edits / f'{stem}_{blink}.png').convert('RGB').resize(key_rgb.shape[1::-1], Image.LANCZOS))
        y0, y1, x0, x1 = face_box(key_rgb, np.asarray(Image.open(edits / f'{stem}_eyes_closed.png').convert('RGB').resize(key_rgb.shape[1::-1], Image.LANCZOS)))
        mid = (x0 + x1) // 2
        ramp = np.clip((np.arange(key_rgb.shape[1]) - mid) / 12 + 0.5, 0, 1)[None, :, None]  # 12 px seam between the eyes
        out = key_rgb.astype(np.float32) * (1 - ramp) + src.astype(np.float32) * ramp
        Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(edits / f'{stem}_{wink}.png')


def composite(cut, key_rgb, edit_rgb, box):
    """The key cut-out with the edit's changes inside the face box, feathered in."""
    y0, y1, x0, x1 = box
    d = np.abs(key_rgb.astype(np.float32) - edit_rgb.astype(np.float32)).mean(-1)
    m = np.zeros_like(d)
    m[y0:y1, x0:x1] = (d[y0:y1, x0:x1] > 18).astype(np.float32)
    m = cv2.dilate(m, np.ones((15, 15), np.uint8))
    m = cv2.GaussianBlur(m, (0, 0), 5)[..., None]
    out = cut.astype(np.float32).copy()
    out[..., :3] = out[..., :3] * (1 - m) + edit_rgb.astype(np.float32) * m
    return out.clip(0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--keys', default='out/g2')
    p.add_argument('--cache', default='out/cache2')
    p.add_argument('--web', default='out/web2')
    p.add_argument('--edits', default='out/face')
    p.add_argument('--angles', default='0:0,0:15')
    p.add_argument('--model', default='flux-2-klein-9b')
    p.add_argument('--width', type=int, default=600)
    p.add_argument('--seed', type=int, default=1, help='change it to redraw an edit that came out wrong')
    p.add_argument('--synth-wink', action='store_true', help='build the winks from the blink edits instead of drawing them')
    args = p.parse_args()
    keys_dir, cache, web, edits = Path(args.keys), Path(args.cache), Path(args.web), Path(args.edits)
    edits.mkdir(parents=True, exist_ok=True)
    (web / 'face').mkdir(exist_ok=True)
    api_key = os.environ.get('BFL_API_KEY')
    keys = {(k['yaw'], k['pitch']): k for k in json.loads((web / 'keys.json').read_text())['keys']}
    angles, cost = [], 0
    for a in args.angles.split(','):
        yaw, pitch = (float(x) for x in a.split(':'))
        yaw, pitch = int(yaw) if yaw.is_integer() else yaw, int(pitch) if pitch.is_integer() else pitch
        k = keys[(yaw, pitch)]
        stem = Path(k['file']).stem
        src = keys_dir / f'{stem}.png'
        key_rgb = np.asarray(Image.open(src).convert('RGB'))
        frames = []
        if args.synth_wink:
            synth_wink(edits, stem, key_rgb)
        for name, change in EXPRESSIONS:
            if change is None:
                frames.append(key_rgb)
                continue
            f = edits / f'{stem}_{name}.png'
            if not f.exists():
                if not api_key:
                    sys.exit('set BFL_API_KEY')
                data, c = bfl_edit(src, f'Edit only her face: {change}. {KEEP}', args.model, api_key, seed=args.seed)
                f.write_bytes(data)
                cost += c
                print(f'drew {f.name} ({c} credits)', flush=True)
            frames.append(np.asarray(Image.open(f).convert('RGB').resize(key_rgb.shape[1::-1], Image.LANCZOS)))
        box = face_box(key_rgb, frames[F['eyes_closed']])
        cut = np.asarray(Image.open(cache / f'{stem}.png').convert('RGBA'))
        size = (args.width, round(args.width * cut.shape[0] / cut.shape[1]))
        cols, rows = 3, -(-len(frames) // 3)
        sheet = Image.new('RGBA', (size[0] * cols, size[1] * rows), (0, 0, 0, 0))
        for i, fr in enumerate(frames):
            img = Image.fromarray(cut if i == 0 else composite(cut, key_rgb, fr, box)).resize(size, Image.LANCZOS)
            sheet.paste(img, ((i % cols) * size[0], (i // cols) * size[1]))
        name = f'face/y{yaw:g}_p{pitch:g}.webp'
        sheet.save(web / name, 'WEBP', quality=92, method=6)
        place = {'s': k['s'], 'ox': k['ox'], 'oy': k['oy']}
        angles.append({'yaw': yaw, 'pitch': pitch, 'sheet': name, 'v': hashlib.md5((web / name).read_bytes()).hexdigest()[:8],
                       'frames': [place] * len(frames), 'key': stem})
        print(f'{name}: face box {box}')
    for anim, seq in SEQUENCES.items():
        (web / f'{anim}.json').write_text(json.dumps({
            'anim': anim, 'frames': len(seq), 'duration': round(len(seq) / FPS, 4), 'fps': FPS, 'sequence': seq,
            'expressions': [n for n, _ in EXPRESSIONS], 'frame': list(size), 'cols': cols, 'front': True, 'angles': angles,
        }))
    print(f'{len(angles)} angles; this run cost {cost:.1f} credits (${cost / 100:.2f})')


if __name__ == '__main__':
    main()
