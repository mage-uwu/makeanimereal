#!/usr/bin/env python3
"""In-between the walk cycle with FLUX 3 Video, keyed on our drawn walk frames.

    BFL_API_KEY=... python3 scripts/walk-video.py [--angles y22.5_p15,...] [--draft] [--concurrency 4]
                    [--keys out/walk] [--out out/walkvid]

For each camera angle, our drawn frames f00, f02, ... f10 and f00 again are given to FLUX 3 Video
(image-to-video, keyframes) spread evenly across 5 seconds: the cycle in slow motion, so the model
only fills small gaps and every drawing sits at a known time. The clip is cut back to 24 frames, one
looping cycle, with our drawings landing on every 4th frame. Writes <out>/<angle>.mp4 and
<out>/<angle>/fNN.png.
"""
import argparse
import base64
import io
import json
import os
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

PROMPT = ('2D anime character animation, one smooth looping walk cycle in slow motion: the girl walks in place with natural '
          'steps, arms swinging gently, camera fixed, she stays in exactly the same spot, size and framing in every frame. '
          'Keep her design, hand-drawn anime art style, line art and colours exactly as in the keyframes (pale green '
          'translucent ribbon twin-tails with clover charms, black and green hair, sage vest with gold buttons, black bow). '
          'Flat solid magenta background throughout, no scenery, no shadow, no camera movement.')
OUT_FRAMES = 24


def b64(path):
    im = Image.open(path).convert('RGB')
    im.thumbnail((720, 1080))
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=93)
    return base64.b64encode(buf.getvalue()).decode()


def run(angle, keys, out, draft, key):
    mp4 = out / f'{angle}.mp4'
    if not mp4.exists():
        drawn = [keys / f'Walking_f{f:02d}_{angle}.png' for f in (0, 2, 4, 6, 8, 10, 0)]
        kf = [[round(i * 5 / 6, 3), b64(p)] for i, p in enumerate(drawn)]
        kf[-1][0] = 4.99
        body = {'mode': 'i2v', 'prompt': PROMPT, 'keyframes': kf, 'duration': 5, 'resolution': 'hd', 'aspect_ratio': 'auto',
                'generate_audio': False, 'draft': draft}
        h = {'x-key': key, 'content-type': 'application/json', 'accept': 'application/json'}
        for attempt in range(4):
            try:
                job = json.load(urllib.request.urlopen(urllib.request.Request('https://api.bfl.ai/v1/flux-3-video', data=json.dumps(body).encode(), headers=h), timeout=120))
                while True:
                    time.sleep(5)
                    r = json.load(urllib.request.urlopen(urllib.request.Request(job['polling_url'], headers={'accept': 'application/json'}), timeout=60))
                    if r['status'] == 'Ready':
                        break
                    if r['status'] not in ('Pending', 'Reasoning', 'Generating', 'Processing', 'Queued'):
                        raise RuntimeError(r['status'])
                mp4.write_bytes(urllib.request.urlopen(r['result']['sample'], timeout=120).read())
                break
            except urllib.error.HTTPError as e:
                if e.code in (400, 401, 402, 403, 422) or attempt == 3:
                    raise RuntimeError(f'{e.code} {e.read()[:300]}')
            except RuntimeError:
                raise
            except Exception:
                if attempt == 3:
                    raise
            time.sleep(2 ** (attempt + 1))
    # One cycle = the whole 5 s clip; keep 24 evenly spaced frames (keyframes land on every 4th).
    d = out / angle
    d.mkdir(exist_ok=True)
    tmp = out / f'{angle}_all'
    tmp.mkdir(exist_ok=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(mp4), str(tmp / 'f%03d.png')], check=True)
    n = len(list(tmp.glob('f*.png')))
    for j in range(OUT_FRAMES):
        i = min(n, 1 + round(j * (n - 1) / OUT_FRAMES))
        (tmp / f'f{i:03d}.png').replace(d / f'f{j:02d}.png') if (tmp / f'f{i:03d}.png').exists() else None
    for f in tmp.glob('*.png'):
        f.unlink()
    tmp.rmdir()
    return angle


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--keys', default='out/walk')
    p.add_argument('--out', default='out/walkvid')
    p.add_argument('--angles')
    p.add_argument('--draft', action='store_true')
    p.add_argument('--concurrency', type=int, default=4)
    args = p.parse_args()
    keys, out = Path(args.keys), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.angles:
        angles = args.angles.split(',')
    else:
        angles = sorted({f.stem.split('_', 2)[2] for f in keys.glob('Walking_f00_*.png')})
        angles = [a for a in angles if all((keys / f'Walking_f{f:02d}_{a}.png').exists() for f in range(0, 12, 2))]
    key = os.environ['BFL_API_KEY']
    with ThreadPoolExecutor(args.concurrency) as ex:
        futs = {ex.submit(run, a, keys, out, args.draft, key): a for a in angles}
        for f in futs:
            try:
                print('done', f.result(), flush=True)
            except Exception as e:
                print('fail', futs[f], e, flush=True)


if __name__ == '__main__':
    main()
