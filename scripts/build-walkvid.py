#!/usr/bin/env python3
"""Pack the video-in-betweened walk (scripts/walk-video.py) for the viewer, in place of the drawn walk.

    python3 scripts/build-walkvid.py [--frames out/walkvid] [--keys out/walk] [--cache out/cache-walk]
                                     [--web out/web2]

Per angle: each of the 24 frames is cut out of its chroma background and cropped to the drawings'
2:3 frame. The video model washes colours out a little, so one colour correction per angle (Lab
mean and spread of the character, matched to our drawn frame f00) is applied to every frame: the
palette comes back to the drawings' and stays steady across the cycle. Placement: the frame f00 of
the clip is fitted (height and leg centre) onto where the viewer already places drawn frame f00,
and that one transform is used for every frame, so the motion is the clip's own. Writes
<web>/walk/<angle>.webp (4 x 6 tiles) and walk.json (24 frames).
"""
import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

spec = importlib.util.spec_from_file_location('build_web', Path(__file__).with_name('build-web.py'))
bw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bw)

TW, TH, COLS, N = 480, 720, 4, 24


def to_frame(img):
    """Centre-crop to 2:3."""
    w, h = img.size
    cw = min(w, round(h * 2 / 3))
    ch = min(h, round(w * 3 / 2))
    return img.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2))


def lab_stats(rgba):
    a = rgba[..., 3] > 128
    lab = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2LAB).astype(np.float32)[a]
    return lab.mean(0), lab.std(0) + 1e-3


def correct(rgba, src, ref):
    lab = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = (lab - src[0]) / src[1] * ref[1] + ref[0]
    out = rgba.copy()
    out[..., :3] = cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frames', default='out/walkvid')
    p.add_argument('--keys', default='out/walk')
    p.add_argument('--cache', default='out/cache-walk')
    p.add_argument('--web', default='out/web2')
    args = p.parse_args()
    frames, cache, web = Path(args.frames), Path(args.cache), Path(args.web)
    old = json.loads((web / 'walk.json').read_text())
    old_angles = {(a['yaw'], a['pitch']): a for a in old['angles']}
    angles = []
    for d in sorted(x for x in frames.iterdir() if x.is_dir()):
        if len(list(d.glob('f*.png'))) < N:
            continue
        yaw, pitch = (float(v[1:]) for v in d.name.split('_'))
        yaw, pitch = (int(v) if v.is_integer() else v for v in (yaw, pitch))
        oa = old_angles.get((yaw, pitch))
        drawn = cache / f'Walking_f00_{d.name}.png'
        if not oa or not drawn.exists():
            print(f'skip {d.name}: no drawn walk to match')
            continue
        ref_cut = np.asarray(Image.open(drawn).convert('RGBA'))
        cuts = [np.asarray(bw.cut_out_chroma(to_frame(Image.open(d / f'f{j:02d}.png').convert('RGB')), None)) for j in range(N)]
        fix = (lab_stats(cuts[0]), lab_stats(ref_cut))
        cuts = [correct(c, *fix) for c in cuts]
        # Fit clip frame f00 onto drawn f00's placement: same top, bottom and leg centre.
        st, sb, sx = bw.extent(Image.fromarray(cuts[0]).getchannel('A'))
        dt, db, dx = bw.extent(Image.fromarray(ref_cut).getchannel('A'))
        f0 = oa['frames'][0]
        top, bot, cx = f0['oy'] + dt * f0['s'], f0['oy'] + db * f0['s'], f0['ox'] + dx * f0['s']
        s = (bot - top) / (sb - st)
        place = {'s': round(s, 5), 'ox': round(cx - sx * s, 5), 'oy': round(top - st * s, 5)}
        sheet = Image.new('RGBA', (TW * COLS, TH * (N // COLS)), (0, 0, 0, 0))
        for j, c in enumerate(cuts):
            sheet.paste(Image.fromarray(c).resize((TW, TH), Image.LANCZOS), ((j % COLS) * TW, (j // COLS) * TH))
        name = f'walk/y{yaw:g}_p{pitch:g}.webp'
        sheet.save(web / name, 'WEBP', quality=90, method=6)
        angles.append({'yaw': yaw, 'pitch': pitch, 'sheet': name, 'v': bw.version(web / name), 'frames': [place] * N})
        print(f'{name}: scale {s:.3f}')
    (web / 'walk.json').write_text(json.dumps({**old, 'frames': N, 'frame': [TW, TH], 'cols': COLS, 'angles': angles, 'source': 'flux-3-video'}))
    print(f'{len(angles)} angles -> {web}/walk.json')


if __name__ == '__main__':
    main()
