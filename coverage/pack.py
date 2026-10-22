#!/usr/bin/env python3
"""Pack the restyled coverage film for the viewer: every frame keyed off its green background and
laid into sprite sheets, with the camera each frame was taken from.

    python3 coverage/pack.py <film> [<film> ...] [--cameras cameras.json ...] [--overlap 1] [--out coverage/web]
                             [--width 540] [--grid 4x4]

Writes <out>/film/sNN.webp (grid of frames, row by row) and <out>/film.json:
{ fps, frame: [w, h], grid: [cols, rows], sheets: [{file, v}], frames: [[yaw, pitch], ...] }.
"""
import argparse
import hashlib
import json
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def frames_of(video):
    """Decode every frame of a video to RGB arrays (ffmpeg, no extra Python deps)."""
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height',
                            '-of', 'csv=p=0', video], capture_output=True, text=True, check=True).stdout
    w, h = map(int, probe.strip().split(','))
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', video, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def frames_of_chunks(folder, total, chunk, overlap):
    """Join the pod's chunk_NN.mp4 files (as far as they go) back into one run of film frames."""
    stride = chunk - overlap
    starts = sorted({min(i * stride, total - chunk) for i in range(-(-(total - overlap) // stride))})
    out = []
    for k, s in enumerate(starts):
        path = Path(folder) / f'chunk_{k:02d}.mp4'
        if not path.exists():
            break
        out.extend(frames_of(str(path))[len(out) - s:])
    return np.stack(out)


def key_green(rgb):
    """Alpha from greenness: the background is a (slightly mottled) green, and nothing she wears is."""
    f = rgb.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    spill = g - np.maximum(r, b)  # how much greener than anything else
    alpha = np.clip(1 - (spill - 18) / 30, 0, 1)
    # Despill the edge: pull green down to the larger of red and blue where it pokes above them.
    out = f.copy()
    out[..., 1] = np.where(spill > 0, np.maximum(r, b) + spill * alpha * 0.3, g)
    a = Image.fromarray((alpha * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.6))
    return np.dstack([out.clip(0, 255).astype(np.uint8), np.asarray(a)])


def make_sheet(job):
    frames, w, h, cols, rows, path = job
    sheet = Image.new('RGBA', (w * cols, h * rows), (0, 0, 0, 0))
    for j, f in enumerate(frames):
        cut = Image.fromarray(key_green(f), 'RGBA').resize((w, h), Image.LANCZOS)
        sheet.paste(cut, ((j % cols) * w, (j // cols) * h))
    sheet.save(path, 'WEBP', quality=82, method=6)
    print(path.name, flush=True)
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('film', nargs='+', help='one or more films (each a film.mp4 or a folder of chunks), packed in order')
    p.add_argument('--cameras', action='append', help='cameras.json per film, in the same order')
    p.add_argument('--overlap', type=int, default=1, help='chunk overlap of folder films (1 for anchored films)')
    p.add_argument('--out', default='coverage/web')
    p.add_argument('--width', type=int, default=540)
    p.add_argument('--grid', default='4x4')
    a = p.parse_args()
    cols, rows = map(int, a.grid.split('x'))
    cam_files = a.cameras or ['out/coverage/cameras.json']
    videos, cam_list = [], []
    for film, cf in zip(a.film, cam_files):
        c = json.loads(Path(cf).read_text())
        v = frames_of_chunks(film, len(c['cameras']), 81, a.overlap) if Path(film).is_dir() else frames_of(film)
        videos.append(v)
        cam_list += c['cameras'][:len(v)]
        fps = c['fps']
    video = np.concatenate(videos)
    n = len(video)
    h = round(a.width * video.shape[1] / video.shape[2])
    out = Path(a.out)
    (out / 'film').mkdir(parents=True, exist_ok=True)
    jobs = [(video[s:s + cols * rows], a.width, h, cols, rows, out / f'film/s{s // (cols * rows):02d}.webp')
            for s in range(0, n, cols * rows)]
    with Pool() as pool:
        names = pool.map(make_sheet, jobs)
    sheets = [{'file': f'film/{p_.name}', 'v': hashlib.sha1(p_.read_bytes()).hexdigest()[:8]} for p_ in names]
    frames = [[c['yaw'] % 360, c['pitch']] for c in cam_list]
    (out / 'film.json').write_text(json.dumps({
        'fps': fps, 'frame': [a.width, h], 'grid': [cols, rows], 'sheets': sheets,
        'frames': [[round(y, 3), round(p_, 3)] for y, p_ in frames],
    }))
    print(f'{n} frames in {len(sheets)} sheets -> {out}')


if __name__ == '__main__':
    main()
