#!/usr/bin/env python3
"""Draw the film's anchor keyframes: the render at each given frame, redrawn as a finished anime
frame in the style reference's look. The video model then only fills between anchors, so the look
can't drift along the film.

    OPENROUTER_API_KEY=... python3 coverage/anchors.py out/coverage --every 80 [--frames 320,400]
        [--ref coverage/assets/style-ref.png] [--out <render dir>/anchors] [--model tencent/hy-image-v3.5-preview]

Takes frames 0, every, 2*every, ... and the last frame (or just --frames). Writes <out>/NNNN.png at
the render's size; frames already drawn are skipped (delete one to redraw it).
"""
import argparse
import base64
import io
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

PROMPT = (
    'Image 1 is a 3D render of a character. Image 2 is the finished anime drawing of the same character: '
    'the art style to use. Redraw image 1 as a finished 2D anime cel illustration in exactly the art '
    'style, line weight, colours and shading of image 2. Keep image 1\'s camera angle, pose, framing, '
    'outfit and every proportion exactly; only the rendering style changes. Clean confident lineart, flat '
    'cel shading with one crisp shadow tone. Plain flat green background, the same green as image 1. '
    'No text, no extra props, no ground shadow, nothing added to her hair.'
)


def data_url(img):
    buf = io.BytesIO()
    img.save(buf, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def draw(render, ref, model):
    body = {'model': model, 'prompt': PROMPT, 'input_references': [
        {'type': 'image_url', 'image_url': {'url': data_url(render)}},
        {'type': 'image_url', 'image_url': {'url': data_url(ref)}},
    ]}
    req = urllib.request.Request('https://openrouter.ai/api/v1/images', json.dumps(body).encode(), {
        'Authorization': f"Bearer {os.environ['OPENROUTER_API_KEY']}", 'Content-Type': 'application/json'})
    d = json.load(urllib.request.urlopen(req, timeout=300))['data'][0]
    raw = base64.b64decode(d['b64_json']) if d.get('b64_json') else urllib.request.urlopen(d['url']).read()
    return Image.open(io.BytesIO(raw)).convert('RGB').resize(render.size, Image.LANCZOS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('renders', help='a render dir from render-orbit.mjs')
    p.add_argument('--every', type=int, default=80)
    p.add_argument('--frames', help='just these frames, e.g. 320,400')
    p.add_argument('--ref', default='coverage/assets/style-ref.png')
    p.add_argument('--out')
    p.add_argument('--model', default='tencent/hy-image-v3.5-preview')
    a = p.parse_args()
    renders = Path(a.renders)
    out = Path(a.out or renders / 'anchors')
    out.mkdir(parents=True, exist_ok=True)
    n = len(list((renders / 'frames').glob('*.png')))
    want = [int(f) for f in a.frames.split(',')] if a.frames else sorted(set(range(0, n, a.every)) | {n - 1})
    ref = Image.open(a.ref).convert('RGB')

    def one(i):
        path = out / f'{i:04d}.png'
        if path.exists():
            return f'{path.name} kept'
        for attempt in range(3):
            try:
                draw(Image.open(renders / 'frames' / f'{i:04d}.png').convert('RGB'), ref, a.model).save(path)
                return f'{path.name} drawn'
            except Exception as e:  # noqa: BLE001 - retry anything, report the last
                err = e
        return f'{path.name} FAILED: {err}'

    with ThreadPoolExecutor(4) as pool:
        for line in pool.map(one, want):
            print(line, flush=True)


if __name__ == '__main__':
    main()
