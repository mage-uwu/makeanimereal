#!/usr/bin/env python3
"""Draw the style reference for the coverage film: one frame of the orbit redrawn as a 2D anime
illustration. The video model takes it as its reference image, so this one drawing sets the look.

    OPENROUTER_API_KEY=... python3 coverage/style-ref.py out/coverage/frames/0240.png out/coverage/style-ref.png
        [--model tencent/hy-image-v3.5-preview] [--prompt "extra direction"]
"""
import argparse
import base64
import io
import json
import os
import urllib.request

from PIL import Image

PROMPT = (
    'Redraw this 3D character render as a hand-drawn 2D anime illustration: clean confident lineart, '
    'flat cel shading with one crisp shadow tone, soft anime highlights in the hair and eyes. Keep '
    'exactly the same character, outfit, colours, pose, camera angle, framing and full body. Plain '
    'flat green background, the same green as the render. No text, no extra props, no ground shadow.'
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('render')
    p.add_argument('out')
    p.add_argument('--model', default='tencent/hy-image-v3.5-preview')
    p.add_argument('--prompt', default='')
    a = p.parse_args()
    src = Image.open(a.render).convert('RGB')
    buf = io.BytesIO()
    src.save(buf, 'PNG')
    body = {
        'model': a.model,
        'prompt': f'{PROMPT} {a.prompt}'.strip(),
        'input_references': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()}}],
    }
    req = urllib.request.Request('https://openrouter.ai/api/v1/images', json.dumps(body).encode(), {
        'Authorization': f"Bearer {os.environ['OPENROUTER_API_KEY']}", 'Content-Type': 'application/json'})
    d = json.load(urllib.request.urlopen(req, timeout=300))['data'][0]
    raw = base64.b64decode(d['b64_json']) if d.get('b64_json') else urllib.request.urlopen(d['url']).read()
    img = Image.open(io.BytesIO(raw)).convert('RGB')
    # Back to the render's frame, so the reference lines up with the film.
    img.resize(src.size, Image.LANCZOS).save(a.out)
    print(a.out, img.size)


if __name__ == '__main__':
    main()
