#!/usr/bin/env python3
"""Pair the styled film with the G-buffer of the camera each frame came from, for training the
learned shader.

    python3 coverage/shader/build_dataset.py --gbuffer out/coverage-gb --film out/coverage-film_a \\
        --out out/shader-data/w1 [--size 360x640] [--overlap 1]

--film is a folder of chunk_NN.mp4 from coverage/pod/vace_film.py (used as far as it goes) or a
film.mp4. Writes per frame NNNN_{color,normal,depth,uva,uvb,target}.png at --size (UV and part are
resized nearest, so they stay exact) and meta.json (frame ids, camera pitch, part names).
"""
import argparse
import importlib.util
import json
from pathlib import Path

from PIL import Image

spec = importlib.util.spec_from_file_location('pack', Path(__file__).parents[1] / 'pack.py')
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gbuffer', required=True)
    p.add_argument('--film', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--size', default='360x640')
    p.add_argument('--overlap', type=int, default=1)
    p.add_argument('--cameras', help='cameras.json (default: the G-buffer dir\'s; any render of the same path will do)')
    p.add_argument('--parts-from', help='a cameras.json that lists the parts, if --cameras doesn\'t')
    a = p.parse_args()
    w, h = map(int, a.size.split('x'))
    gb, out = Path(a.gbuffer), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cams = json.loads(Path(a.cameras or gb / 'cameras.json').read_text())
    parts = cams.get('parts') or json.loads(Path(a.parts_from).read_text())['parts']
    n = len(cams['cameras'])
    film = pack.frames_of_chunks(a.film, n, 81, a.overlap) if Path(a.film).is_dir() else pack.frames_of(a.film)
    done = []
    for i, f in enumerate(film):
        name = f'{i:04d}'
        if not (gb / 'uvb' / f'{name}.png').exists():
            break
        Image.fromarray(pack.key_green(f), 'RGBA').resize((w, h), Image.LANCZOS).save(out / f'{name}_target.png')
        for kind, resample in [('frames', Image.BILINEAR), ('normal', Image.BILINEAR), ('depth', Image.BILINEAR),
                               ('uva', Image.NEAREST), ('uvb', Image.NEAREST)]:
            dst = out / f'{name}_{"color" if kind == "frames" else kind}.png'
            Image.open(gb / kind / f'{name}.png').convert('RGB').resize((w, h), resample).save(dst)
        done.append({'frame': i, 'pitch': cams['cameras'][i]['pitch'], 'yaw': cams['cameras'][i]['yaw']})
    (out / 'meta.json').write_text(json.dumps({'size': [w, h], 'parts': parts, 'frames': done}))
    print(f'{len(done)} frames -> {out}')


if __name__ == '__main__':
    main()
