#!/usr/bin/env python3
"""Draw the ground props' proxy renders as anime background art, and pack them for the viewer.

    BFL_API_KEY=... python3 scripts/props.py [--renders out/props-renders] [--out out/props]
                    [--web out/web2] [--style assets/scene/bliss-sky.webp] [--model flux-2-klein-4b]
                    [--only rock,bush] [--limit 1]

Each view of a prop is drawn by FLUX.2 (Black Forest Labs' API) from its render (shape, angle,
framing), with the Bliss painting as the style reference. The prop's first drawn view goes along as
an identity reference for all its other views, so they draw the same rock, bush or tuft. Drawings
are 512x512, the smallest size, so each costs the base rate. Writes <web>/props/<prop>.webp (a
sheet: one row per pitch, one column per yaw, cut out of the chroma background) and props.json.
"""
import argparse
import base64
import importlib.util
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
spec = importlib.util.spec_from_file_location('build_web', Path(__file__).with_name('build-web.py'))
bw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bw)

SIZE = 512
WHAT = {
    'rock': 'a weathered grey field boulder with soft moss and a few tiny grass blades at its base',
    'bush': 'a round leafy green shrub with clusters of small leaves and a few tiny white flowers',
    'grass': 'a tuft of tall wild meadow grass blades with one or two small white clover flowers',
}


def png(img, size=SIZE):
    img = img.convert('RGB')
    img.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, 'PNG')
    return base64.b64encode(buf.getvalue()).decode()


def draw(refs, prompt, model, key, seed):
    body = {'prompt': prompt, 'width': SIZE, 'height': SIZE, 'output_format': 'png', 'safety_tolerance': 2, 'seed': seed}
    for i, r in enumerate(refs):
        body['input_image' if i == 0 else f'input_image_{i + 1}'] = r
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
        time.sleep(2 ** (attempt + 1))


def draw_openrouter(refs, prompt, key):
    """The same drawing on Hy Image (OpenRouter), which follows a render's camera angle closely."""
    body = {'model': 'tencent/hy-image-v3.5-preview', 'prompt': prompt,
            'input_references': [{'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{r}'}} for r in refs]}
    headers = {'Authorization': f'Bearer {key}', 'content-type': 'application/json'}
    for attempt in range(4):
        try:
            j = json.load(urllib.request.urlopen(urllib.request.Request(
                'https://openrouter.ai/api/v1/images', data=json.dumps(body).encode(), headers=headers), timeout=300))
            d = j['data'][0]
            data = base64.b64decode(d['b64_json']) if d.get('b64_json') else urllib.request.urlopen(d['url'], timeout=60).read()
            return data, 100 * (j.get('usage') or {}).get('cost', 0.024)
        except urllib.error.HTTPError as e:
            if e.code in (400, 401, 403) or attempt == 3:
                raise
        except Exception:
            if attempt == 3:
                raise
        time.sleep(2 ** (attempt + 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--renders', default='out/props-renders')
    p.add_argument('--out', default='out/props')
    p.add_argument('--web', default='out/web2')
    p.add_argument('--style', default='assets/scene/bliss-sky.webp')
    p.add_argument('--model', default='flux-2-klein-4b')
    p.add_argument('--only')
    p.add_argument('--limit', type=int, help='draw at most this many views (to price a run)')
    p.add_argument('--seed', type=int, default=3)
    p.add_argument('--backend', default='bfl', choices=['bfl', 'openrouter'], help='openrouter: Hy Image, best at following steep angles')
    p.add_argument('--pitches', help='only draw these pitches, e.g. 50,75')
    args = p.parse_args()
    renders, out, web = Path(args.renders), Path(args.out), Path(args.web)
    manifest = json.loads((renders / 'manifest.json').read_text())
    key = os.environ.get('OPENROUTER_API_KEY' if args.backend == 'openrouter' else 'BFL_API_KEY')
    only_pitches = {float(x) for x in args.pitches.split(',')} if args.pitches else None
    style = png(Image.open(args.style), 768)
    spent, drawn = 0.0, 0
    for name, prop in manifest['props'].items():
        if args.only and name not in args.only.split(','):
            continue
        (out / name).mkdir(parents=True, exist_ok=True)
        for v in prop['views']:
            f = out / v['file']
            if f.exists() or (args.limit is not None and drawn >= args.limit) or (only_pitches and v['pitch'] not in only_pitches):
                continue
            # Identity from the first view at the same height, so a steep view never copies a side view.
            first = out / next(w['file'] for w in prop['views'] if w['pitch'] == v['pitch'])
            ident = [png(Image.open(first))] if f != first and first.exists() else []
            steep = (f' The camera looks DOWN on it from high above at {v["pitch"]:g} degrees, exactly as in image 1: we see mostly its top, '
                     f'strongly foreshortened, NOT a side view.' if v['pitch'] >= 40 else '')
            prompt = (f'Image 1 is a 3D RENDER of {WHAT[name]}: copy its exact shape, size, camera angle and framing. '
                      f'Image 2 is a STYLE REFERENCE: paint it in exactly that hand-painted anime background-art style and palette '
                      f'(soft painterly shading, bright summer light, crisp clean edges).'
                      + (' Image 3 is the SAME object already painted from another angle: keep it the same object, colours and details.' if ident else '')
                      + steep
                      + f' Redraw image 1 as {WHAT[name]}, a single isolated prop centred in the frame, on a perfectly flat solid #c006a6 magenta '
                      f'background, no ground, no cast shadow, no other objects, no text.')
            refs = [png(Image.open(renders / v['file'])), style, *ident]
            if args.backend == 'openrouter':
                data, c = draw_openrouter(refs, prompt, key)
            else:
                data, c = draw(refs, prompt, args.model, key, args.seed)
            f.write_bytes(data)
            spent += c
            drawn += 1
            print(f'drew {v["file"]} ({c} credits)', flush=True)
    print(f'drew {drawn}; {spent:.1f} credits (${spent / 100:.2f})')

    # Pack every prop whose views are all drawn.
    (web / 'props').mkdir(parents=True, exist_ok=True)
    packed = {}
    for name, prop in manifest['props'].items():
        if not all((out / v['file']).exists() for v in prop['views']):
            continue
        T = 256  # tile size in the web sheet; drawings of another shape are centre-cropped square
        sheet = Image.new('RGBA', (T * len(prop['yaws']), T * len(prop['pitches'])), (0, 0, 0, 0))
        for v in prop['views']:
            im = Image.open(out / v['file']).convert('RGB')
            side = min(im.size)
            im = im.crop(((im.width - side) // 2, (im.height - side) // 2, (im.width + side) // 2, (im.height + side) // 2))
            cut = bw.cut_out_chroma(im, None).resize((T, T), Image.LANCZOS)
            sheet.paste(cut, (prop['yaws'].index(v['yaw']) * T, prop['pitches'].index(v['pitch']) * T))
        sheet.save(web / 'props' / f'{name}.webp', 'WEBP', quality=90, method=6)
        packed[name] = {'sheet': f'props/{name}.webp', 'v': bw.version(web / 'props' / f'{name}.webp'), 'yaws': prop['yaws'], 'pitches': prop['pitches'],
                        'frame': prop['frame'], 'center': prop['center'][1] - prop['base']}
    (web / 'props.json').write_text(json.dumps(packed))
    print(f'packed {", ".join(packed) or "nothing"}')


if __name__ == '__main__':
    main()
