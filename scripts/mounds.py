#!/usr/bin/env python3
"""Painted hill sprites for the Bliss field's distant mounds: one card per mound in the painting's style.

    OPENROUTER_API_KEY=... python3 scripts/mounds.py [--style assets/scene/bliss-sky.webp] [--out out/mounds] [--web out/web2]

Each hill shape is drawn by Hy Image with the Bliss painting as the style reference, on a chroma
background, then cut out. The cut-out keeps a soft ridge (its top edge feathered so it melts into the
sky) and fades out over its lower part, so the card blends into the meadow it stands in. Writes
<web>/scene/mound<N>.webp (RGBA) and mounds.json (aspect and where the ridge sits).
"""
import argparse
import base64
import importlib.util
import json
import os
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

spec = importlib.util.spec_from_file_location('build_web', Path(__file__).with_name('build-web.py'))
bw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bw)

SHAPES = [
    'one wide, gently rounded grassy hill, much wider than tall, with a smooth soft crest',
    'two overlapping rolling grassy hills, the nearer one lower, smooth soft crests',
    'one broad low grassy mound with a long gentle slope down to the right',
    'one rounded grassy hill a little taller than the others, with a few small bushes along its side',
]


def draw(style_b64, shape, key):
    prompt = ('Image 1 is a STYLE REFERENCE. Paint ' + shape + ', isolated, in exactly the hand-painted anime background style, '
              'brushwork, greens and sunlight of the big hill in image 1: long sunlit grass strokes, lighter yellow-green on the lit '
              'side, deeper green in shade, a few tiny white flowers. Wide panoramic framing, the hill fills the full width and '
              'sits at the bottom of the frame. Plain flat solid #c006a6 magenta background everywhere around it (no sky, no '
              'clouds, no other hills), no text.')
    body = {'model': 'tencent/hy-image-v3.5-preview', 'prompt': prompt,
            'input_references': [{'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{style_b64}'}}]}
    headers = {'Authorization': f'Bearer {key}', 'content-type': 'application/json'}
    for attempt in range(4):
        try:
            j = json.load(urllib.request.urlopen(urllib.request.Request(
                'https://openrouter.ai/api/v1/images', data=json.dumps(body).encode(), headers=headers), timeout=300))
            d = j['data'][0]
            return base64.b64decode(d['b64_json']) if d.get('b64_json') else urllib.request.urlopen(d['url'], timeout=60).read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))


def soften(rgba):
    """Feather the ridge into the sky and fade the lower part out into the meadow."""
    a = rgba[..., 3].astype(np.float32) / 255
    h, w = a.shape
    cols = np.nonzero((a > 0.5).any(0))[0]
    rows = np.nonzero((a > 0.5).any(1))[0]
    a = cv2.GaussianBlur(a, (0, 0), max(1.5, w / 400))          # soft ridge line
    top, bot = rows[0], rows[-1]
    ys = np.arange(h, dtype=np.float32)[:, None]
    fade = np.clip((bot - ys) / (0.55 * (bot - top)), 0, 1)       # gone at the bottom, full by 45% up
    a *= fade ** 1.4
    xs = np.arange(w, dtype=np.float32)[None, :]                  # and soft at both sides
    a *= np.clip(np.minimum(xs - cols[0], cols[-1] - xs) / (0.14 * (cols[-1] - cols[0])), 0, 1) ** 1.2
    rgba[..., 3] = (a * 255).astype(np.uint8)
    return rgba[top:bot + 1, cols[0]:cols[-1] + 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--style', default='assets/scene/bliss-sky.webp')
    p.add_argument('--out', default='out/mounds')
    p.add_argument('--web', default='out/web2')
    args = p.parse_args()
    out, web = Path(args.out), Path(args.web)
    out.mkdir(parents=True, exist_ok=True)
    (web / 'scene').mkdir(parents=True, exist_ok=True)
    key = os.environ['OPENROUTER_API_KEY']
    im = Image.open(args.style).convert('RGB')
    import io
    buf = io.BytesIO(); im.save(buf, 'PNG')
    style = base64.b64encode(buf.getvalue()).decode()
    info = []
    for i, shape in enumerate(SHAPES):
        f = out / f'mound{i}.png'
        if not f.exists():
            f.write_bytes(draw(style, shape, key))
            print(f'drew {f.name}', flush=True)
        cut = np.asarray(bw.cut_out_chroma(Image.open(f).convert('RGB'), None)).copy()
        rgba = soften(cut)
        name = f'scene/mound{i}.webp'
        Image.fromarray(rgba, 'RGBA').save(web / name, 'WEBP', quality=88, method=6)
        info.append({'sheet': name, 'v': bw.version(web / name), 'aspect': round(rgba.shape[1] / rgba.shape[0], 4)})
    (web / 'mounds.json').write_text(json.dumps(info))
    print(f'{len(info)} mounds -> {web}/mounds.json')


if __name__ == '__main__':
    main()
