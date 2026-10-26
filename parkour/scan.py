"""Summarize every motion in the PARC release: what each one does, measured from the data.

    python3 parkour/scan.py <dec_release.zip> --out out/parc_index.csv

Per motion: duration, distance travelled, height climbed and dropped (the pelvis' rise and fall
over the course), the share of frames with a hand on something (vaults, climbs), and its folder
(which PARC iteration and terrain family it came from).
"""
import argparse
import csv
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parc  # noqa: E402

SRC = None


def stats(name):
    m = parc.load(SRC.read(name))['motion']
    p = np.asarray(m['root_pos'], float)
    c = np.asarray(m['body_contacts']) > .5 if m['body_contacts'] is not None else np.zeros((len(p), 15), bool)
    z_ = p[:, 2]
    smooth = np.convolve(np.pad(z_, 7, mode='edge'), np.ones(15) / 15, 'valid')
    return {
        'file': name, 'folder': name.split('/')[1], 'frames': len(p), 'seconds': round(len(p) / m['fps'], 2),
        'travel_m': round(float(np.linalg.norm(p[-1, :2] - p[0, :2])), 2),
        'climb_m': round(float(np.clip(np.diff(smooth), 0, None).sum()), 2),
        'drop_m': round(float(np.clip(-np.diff(smooth), 0, None).sum()), 2),
        'hands': round(float(c[:, [5, 8]].any(1).mean()), 3),
        'arms': round(float(c[:, [3, 4, 6, 7]].any(1).mean()), 3),
    }


def init(src):
    global SRC
    SRC = zipfile.ZipFile(src)   # one handle per worker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('zip')
    ap.add_argument('--out', default='out/parc_index.csv')
    a = ap.parse_args()
    names = [n for n in zipfile.ZipFile(a.zip).namelist() if n.endswith('.pkl')]
    with ProcessPoolExecutor(os.cpu_count(), initializer=init, initargs=(a.zip,)) as ex:
        rows = list(ex.map(stats, names, chunksize=64))
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
    secs = sum(r['seconds'] for r in rows)
    print(f'{len(rows)} motions, {secs / 3600:.1f} hours; hands on something in {np.mean([r["hands"] > .05 for r in rows]):.0%} of them')


if __name__ == '__main__':
    main()
