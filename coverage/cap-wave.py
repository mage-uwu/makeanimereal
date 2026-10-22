#!/usr/bin/env python3
"""Prepare a cap wave: a second camera path that continues the coverage film into the views it
missed (the top of the head), opening with the film's own finished frames so its look carries.

    python3 coverage/cap-wave.py <film: film.mp4 or folder of chunk_NN.mp4> <cap render dir>
                                 --seed 943:959  [--wave out/coverage]

--seed a:b picks the film frames that open the wave, in that order (b < a runs them backwards).
Writes, into the cap dir:
seed.mp4 (those styled frames) and wave-depth.mp4 (their depth, then the cap path's own depth).
"""
import argparse
import importlib.util
import subprocess
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('pack', Path(__file__).with_name('pack.py'))
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)


def encode(frames_or_pngs, path, fps=16):
    """Write RGB arrays (or PNG paths) as a near-lossless H.264 video."""
    if isinstance(frames_or_pngs[0], (str, Path)):
        lst = Path(path).with_suffix('.txt')
        lst.write_text(''.join(f"file '{Path(p).resolve()}'\nduration {1 / fps}\n" for p in frames_or_pngs))
        cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', str(lst)]
    else:
        h, w = frames_or_pngs[0].shape[:2]
        cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-']
    cmd += ['-r', str(fps), '-c:v', 'libx264', '-crf', '12', '-pix_fmt', 'yuv420p', str(path)]
    data = None if isinstance(frames_or_pngs[0], (str, Path)) else np.ascontiguousarray(np.stack(frames_or_pngs)).tobytes()
    subprocess.run(cmd, input=data, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('film')
    p.add_argument('cap')
    p.add_argument('--seed', required=True)
    p.add_argument('--wave', default='out/coverage', help='the first wave\'s render (its depth frames)')
    a = p.parse_args()
    s0, s1 = map(int, a.seed.split(':'))
    seed = list(range(s0, s1 + 1)) if s1 >= s0 else list(range(s0, s1 - 1, -1))
    film = pack.frames_of_chunks(a.film, 960, 81, 17) if Path(a.film).is_dir() else pack.frames_of(a.film)
    if max(seed) >= len(film):
        raise SystemExit(f'the film has only {len(film)} frames so far; seed needs frame {max(seed)}')
    cap = Path(a.cap)
    encode([film[i] for i in seed], cap / 'seed.mp4')
    depth = [Path(a.wave) / 'depth' / f'{i:04d}.png' for i in seed] + sorted((cap / 'depth').glob('*.png'))
    encode(depth, cap / 'wave-depth.mp4')
    print(f'{cap}: {len(seed)} seed frames + {len(depth) - len(seed)} new -> seed.mp4, wave-depth.mp4')


if __name__ == '__main__':
    main()
