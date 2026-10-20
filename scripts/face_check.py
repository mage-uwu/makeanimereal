#!/usr/bin/env python3
"""Find poisoned key drawings: ones that show her face where the camera can't see it.

    OPENROUTER_API_KEY=... python3 scripts/face_check.py [--renders out/g2-renders] [--keys out/g2]
                                                         [--model google/gemini-3.5-flash] [--report out/face-check.json]

Image models like to keep a character's face visible, so on back views they sometimes draw her
looking back over her shoulder. That drawing is not a view of the same latent object as its
neighbours, and since key drawings are shown exactly as drawn, it breaks the rotation.

Ground truth comes from the 3D model: the share of her face (the front of the head) that each key
camera actually sees, by depth test against the render's depth map. A vision model reads each
drawing and says whether her eyes or face are visible. A drawing that shows a face the model says is
hidden (under 10 % visible) is flagged for a redraw. Costs about $0.002 per drawing checked.
"""
import argparse
import base64
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from latent import KeyCamera, read_depth  # noqa: E402

QUESTION = """This is a drawing of an anime girl. Look only at her head.
Answer in JSON: {"eyes_visible": 0, 1 or 2 (how many of her eyes can be seen, even partly),
"face_visible": "none" | "sliver" (only a thin edge of cheek or the tip of the nose) | "profile" | "three-quarter" | "front",
"looking_back": true if her head is turned back toward the viewer over her shoulder while her body faces away, else false}"""


def face_vertices(mesh):
    """Vertices on the front of her face: head-bone vertices in the front fifth of the head."""
    pos = np.asarray(mesh['position'], float).reshape(-1, 3)
    bone = np.asarray(mesh['bone'])
    names = [b['name'].replace('mixamorig', '').lstrip(':') for b in mesh['skeleton']]
    head = pos[bone == names.index('Head')]
    front = head[:, 2] > head[:, 2].max() - (head[:, 2].max() - head[:, 2].min()) * 0.2
    # eye height band: middle of the head vertically
    y0, y1 = np.percentile(head[:, 1], [25, 75])
    band = (head[:, 1] > y0) & (head[:, 1] < y1)
    return head[front & band]


def visible_share(pts, view, cam0, manifest, renders):
    cam = KeyCamera(view['yaw'], view['pitch'], cam0['target'], cam0['radius'], manifest['fov'], manifest['width'] / manifest['height'])
    depth = read_depth(renders / manifest['depth']['dir'] / view['file'], cam0['radius'], manifest['depth']['pad'])
    fx, fy, dist = cam.project(pts)
    h, w = depth.shape
    px, py = np.clip((fx * w).astype(int), 0, w - 1), np.clip((fy * h).astype(int), 0, h - 1)
    return float(np.mean(np.abs(depth[py, px] - dist) < 0.01))


def ask(model, path, key):
    img = base64.b64encode(Path(path).read_bytes()).decode()
    body = {'model': model, 'temperature': 0, 'response_format': {'type': 'json_object'},
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': QUESTION},
                                                      {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{img}'}}]}]}
    for attempt in range(4):
        try:
            req = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions', json.dumps(body).encode(),
                                         {'content-type': 'application/json', 'authorization': f'Bearer {key}'})
            out = json.loads(urllib.request.urlopen(req, timeout=120).read())
            text = out['choices'][0]['message']['content']
            return json.loads(text[text.find('{'):text.rfind('}') + 1]), float(out.get('usage', {}).get('cost') or 0)
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f'{path}: {err}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--renders', default='out/g2-renders')
    p.add_argument('--keys', default='out/g2')
    p.add_argument('--model', default='google/gemini-3.5-flash')
    p.add_argument('--report', default='out/face-check.json')
    p.add_argument('--only', help='comma-separated files to check')
    p.add_argument('--hidden-below', type=float, default=0.10, help='face share below which the face counts as hidden')
    args = p.parse_args()
    key = os.environ['OPENROUTER_API_KEY']
    renders = Path(args.renders)
    manifest = json.loads((renders / 'manifest.json').read_text())
    mesh = json.loads((renders / 'mesh.json').read_text())
    cam0 = manifest['cameras'][0]
    face = face_vertices(mesh)
    views = [v for v in manifest['views'] if (Path(args.keys) / v['file']).exists()]
    if args.only:
        views = [v for v in views if v['file'] in args.only.split(',')]
    shares = {v['file']: visible_share(face, v, cam0, manifest, renders) for v in views}
    # Only drawings whose face should be (nearly) hidden need the judge.
    todo = [v for v in views if shares[v['file']] < args.hidden_below]
    with ThreadPoolExecutor(6) as pool:
        answers = list(pool.map(lambda v: ask(args.model, Path(args.keys) / v['file'], key), todo))
    report, cost = [], 0.0
    for v, (a, c) in zip(todo, answers):
        cost += c
        poisoned = bool(a.get('looking_back')) or int(a.get('eyes_visible', 0)) > 0 or a.get('face_visible') in ('profile', 'three-quarter', 'front')
        report.append({'file': v['file'], 'yaw': v['yaw'], 'pitch': v['pitch'], 'face_share': round(shares[v['file']], 3), **a, 'poisoned': poisoned})
    Path(args.report).write_text(json.dumps(report, indent=1))
    bad = [r for r in report if r['poisoned']]
    print(f'{len(views)} drawings, {len(todo)} with her face hidden in 3D; {len(bad)} poisoned (cost ${cost:.3f})')
    for r in sorted(bad, key=lambda r: (r['pitch'], r['yaw'])):
        print(f"  {r['file']:16s} 3D face share {r['face_share']:.2f}  drawn: eyes {r.get('eyes_visible')}, face {r.get('face_visible')}, looking back {r.get('looking_back')}")
    print(f'report -> {args.report}')


if __name__ == '__main__':
    main()
