#!/usr/bin/env python3
"""Export a trained shader (train.py) for the browser runtime (coverage/live).

    python3 export.py runs/hybrid_v1/shader.pt --data data/w1 [--data ...] --out export/

Writes:
  feat0.png, feat1.png, feat2.png  the neural texture, baked: the coarse grid upsampled into the fine
                                   one, 8 channels as RGB atlases (5 x 4 tiles of 512, one per part),
                                   zero outside the texels the training views ever touched
  shader.json                      per-channel ranges of the texture, and the decoder as a list of
                                   convolutions (float16 weights, base64) in the order they run
  reference.png                    the decoder's own output for one held-out frame, to check the
                                   browser against
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import train

COLS, ROWS, T = 5, 4, 512


def usage(datas, parts):
    """Which texels of each part's grid the training views sampled (dilated a little)."""
    hit = torch.zeros(parts, T, T)
    for d in datas:
        meta, x, _, _ = train.load(d)
        m = x[..., 7] > 0.5
        u, v, p = x[..., 8][m], x[..., 9][m], x[..., 10][m].astype(int)
        ui = np.clip((u * T).astype(int), 0, T - 1)
        vi = np.clip((v * T).astype(int), 0, T - 1)
        hit[p, vi, ui] = 1
    return F.max_pool2d(hit[:, None], 9, 1, 4)[:, 0] > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--data', action='append', required=True)
    ap.add_argument('--out', default='export')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(a.ckpt, map_location='cpu')
    sd, parts = dict(ck['model']), ck['parts']
    sharp = 'sr' in ck  # train_sr.py: decoder at half resolution, plus a full-resolution head
    light = ck.get('light', 0)  # train_light.py: light channels after the 18 inputs, at both resolutions
    if sharp:
        sd.update({f'sr.{k}': v.detach() for k, v in ck['sr'].items()})
    P = len(parts)

    # Neural texture: fine + coarse (upsampled), [8, P, T, T].
    fine, coarse = sd['tex.fine'][0], sd['tex.coarse'][0]
    up = F.interpolate(coarse.permute(1, 0, 2, 3), size=(T, T), mode='bilinear', align_corners=False).permute(1, 0, 2, 3)
    tex = fine + up
    mask = usage(a.data, P)
    tex = tex * mask[None]
    lo = tex.amin((1, 2, 3)).numpy()
    hi = tex.amax((1, 2, 3)).numpy()
    q = ((tex.numpy() - lo[:, None, None, None]) / (hi - lo)[:, None, None, None] * 255).round().clip(0, 255).astype(np.uint8)
    atlas = np.zeros((8, ROWS * T, COLS * T), np.uint8)
    for p in range(P):
        r, c = divmod(p, COLS)
        atlas[:, r * T:(r + 1) * T, c * T:(c + 1) * T] = q[:, p]
    # Texels nobody sampled hold the value that decodes to 0.
    zero = ((0 - lo) / (hi - lo) * 255).round().clip(0, 255).astype(np.uint8)
    for p in range(P, COLS * ROWS):
        r, c = divmod(p, COLS)
        atlas[:, r * T:(r + 1) * T, c * T:(c + 1) * T] = zero[:, None, None]
    for k, chans in enumerate([(0, 1, 2), (3, 4, 5), (6, 7, 7)]):
        Image.fromarray(np.stack([atlas[c] for c in chans], -1), 'RGB').save(out / f'feat{k}.png', optimize=True)

    # Decoder convolutions, in run order. Inputs are named tensors; channel maps say which input
    # channel of the torch weight comes from which tensor (concatenation order).
    def conv(name, key, inputs, act, out_name, affine=None):
        w, b = sd[f'{key}.weight'].float(), sd[f'{key}.bias'].float()
        layer = {'name': name, 'inputs': inputs, 'out': out_name, 'cout': w.shape[0], 'cin': w.shape[1],
                 'k': w.shape[2], 'act': act,
                 'w': base64.b64encode(w.numpy().astype(np.float16).tobytes()).decode(),
                 'b': b.tolist()}
        if affine is not None:
            layer['affine'] = [x.tolist() for x in affine]
        return layer

    g, bb = sd['dec.film.bias'].float().chunk(2)  # appearance code 0: FiLM is a fixed per-channel affine
    layers = []
    w = list(ck.get('widths', (32, 48, 64, 96)))
    chans = {'x': 18 + light, 'd0': w[0], 'd1': w[1], 'd2': w[2], 'd3': w[3], 'u0': w[2], 'u1': w[1], 'u2': w[0]}
    prev = 'x'
    for k, name in enumerate(['d0', 'd1', 'd2', 'd3']):
        src = prev if k == 0 else f'{prev}_pool'
        layers.append(conv(f'{name}a', f'dec.down.{k}.0', [src], 'gelu', f'{name}_mid'))
        layers.append(conv(f'{name}b', f'dec.down.{k}.2', [f'{name}_mid'], 'gelu', name,
                           affine=(1 + g, bb) if name == 'd3' else None))
        prev = name
    h = 'd3'
    for k, (name, skip) in enumerate([('u0', 'd2'), ('u1', 'd1'), ('u2', 'd0')]):
        layers.append(conv(f'{name}a', f'dec.up.{k}.0', [h, skip], 'gelu', f'{name}_mid'))
        layers.append(conv(f'{name}b', f'dec.up.{k}.2', [f'{name}_mid'], 'gelu', name))
        h = name
    if sharp:
        chans['xf'] = 18 + light
        layers.append(conv('srh', 'sr.conv', ['u2', 'xf'], 'gelu', 'srh'))
        layers.append(conv('srhead', 'sr.head', ['srh', 'xf'], 'sigmoid', 'rgba'))
    else:
        layers.append(conv('head', 'dec.head', ['u2', 'x'], 'sigmoid', 'rgba'))
    # Every tensor's size, so the runtime needn't know the architecture.
    full = [720, 1280] if sharp else [360, 640]
    dims = {'x': [360, 640], 'd0_mid': [360, 640], 'd0': [360, 640], 'd0_pool': [180, 320], 'd1_mid': [180, 320], 'd1': [180, 320],
            'd1_pool': [90, 160], 'd2_mid': [90, 160], 'd2': [90, 160], 'd2_pool': [45, 80], 'd3_mid': [45, 80], 'd3': [45, 80],
            'u0_mid': [90, 160], 'u0': [90, 160], 'u1_mid': [180, 320], 'u1': [180, 320], 'u2_mid': [360, 640], 'u2': [360, 640],
            'xf': full, 'srh': full, 'rgba': full}
    meta = {'parts': parts, 'tile': T, 'cols': COLS, 'rows': ROWS, 'feat_lo': lo.tolist(), 'feat_hi': hi.tolist(),
            'channels': chans, 'light': light, 'size': [360, 640], 'full': full, 'sharp': sharp, 'dims': dims, 'layers': layers,
            'pools': {'d0_pool': 'd0', 'd1_pool': 'd1', 'd2_pool': 'd2'}}
    (out / 'shader.json').write_text(json.dumps(meta))

    # Reference output for one held-out frame, through the exported (baked, quantised) texture.
    model = train.Shader(P, sd['codes.weight'].shape[0], ck['arch'], widths=tuple(w))
    if light:
        c0 = model.dec.down[0][0]
        model.dec.down[0][0] = torch.nn.Conv2d(18 + light, c0.out_channels, c0.kernel_size, padding=c0.padding)
    model.load_state_dict(ck['model'])
    if sharp:
        import train_sr
        model = train_sr.Sharp(model, w[0], ck['sr_width'])
        if light:
            model.conv = torch.nn.Conv2d(w[0] + 18 + light, ck['sr_width'], 3, padding=1)
            model.head = torch.nn.Conv2d(ck['sr_width'] + 18 + light, 4, 1)
        model.conv.load_state_dict({k[len('sr.conv.'):]: v for k, v in sd.items() if k.startswith('sr.conv.')})
        model.head.load_state_dict({k[len('sr.head.'):]: v for k, v in sd.items() if k.startswith('sr.head.')})
    model.eval()
    meta_d, x, y, pitch = train.load(a.data[0])
    i = 5
    with torch.no_grad():
        xi, pi = torch.from_numpy(x[i:i + 1].astype(np.float32)), torch.from_numpy(pitch[i:i + 1])
        if light:  # under the films' own light
            import train_light
            lp = torch.from_numpy(train_light.light_params(meta_d['frames'][i])[None])
            pred = train_light.lit_forward(model, xi, pi, None, lp)[0]
        else:
            pred = model(xi, pi)[0]
    rgba = (pred.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(rgba, 'RGBA').save(out / 'reference.png')
    (out / 'reference.json').write_text(json.dumps({'frame': meta_d['frames'][i]}))
    print('exported to', out, 'texture range', lo.round(2), hi.round(2))


if __name__ == '__main__':
    main()
