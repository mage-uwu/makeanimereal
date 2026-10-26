"""Export PARC parkour motions as VRM Animation clips on our rig, with their terrain.

    python3 parkour/export.py <dec_release.zip or dir> --out web/parkour --pick 'Climb=*/teaser_2779_0_opt_dm.pkl' ...

Each --pick is Label=glob (the first match is exported as <label>.vrma) or a bare glob (every match,
named after its file).

Each motion becomes <name>.vrma (rest rig and finger curl taken from web/moves/wave.vrma, so it
plays like the other moves, but it keeps its travel: the hips move through the course) and
<name>.json, its heightfield in the same units. PARC's humanoid has longer legs than our rig, so
the whole scene, motion and terrain alike, is scaled by the ratio of leg lengths: feet land where
PARC's did. The hands are a little off (the rigs' arms differ more); that is the physics stage's job.
"""
import argparse
import fnmatch
import json
import os
import sys
import zipfile

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(0, os.path.join(here, '..', 'vrmanet'))
import parc  # noqa: E402
from model import accessor, read_glb, write_glb  # noqa: E402

PARC_LEG = .421546 + .40987                                  # hip joint to ankle


def rig_from(template):
    js, b = read_glb(template)
    human = js['extensions']['VRMC_vrm_animation']['humanoid']['humanBones']
    node = {k: v['node'] for k, v in human.items()}
    world = {}
    parent = {c: i for i, n in enumerate(js['nodes']) for c in n.get('children', [])}
    for i, n in enumerate(js['nodes']):
        p, t = i, np.zeros(3)
        while p is not None:
            t += js['nodes'][p].get('translation', [0, 0, 0]); p = parent.get(p)
        world[i] = t
    leg = np.linalg.norm(world[node['leftUpperLeg']] - world[node['leftFoot']])
    # Held tracks in the template (the fingers' curl), to carry over.
    held = {}
    anim = js['animations'][0]
    for ch in anim['channels']:
        v = accessor(js, b, anim['samplers'][ch['sampler']]['output'])
        bone = next(k for k, n in node.items() if n == ch['target']['node'])
        if ch['target']['path'] == 'rotation' and np.allclose(v, v[0]) and bone.startswith(('left', 'right')) and \
                any(f in bone for f in ('Thumb', 'Index', 'Middle', 'Ring', 'Little')):
            held[bone] = v[0].copy()
    hips_above_legs = world[node['hips']] - (world[node['leftUpperLeg']] + world[node['rightUpperLeg']]) / 2
    return js, node, leg, held, hips_above_legs


def to_clip(data, rig, name, credit):
    js0, node, leg, held, hips_above_legs = rig
    m = data['motion']
    k = leg / PARC_LEG
    rots, hips = parc.to_vrm(m)
    # PARC's root is at hip-joint height; our hips sit a little above the legs, along the pelvis.
    hips = hips * k + parc.qrot(rots['hips'], np.broadcast_to(hips_above_legs, hips.shape))
    F, fps = len(hips), int(m['fps'])
    times = np.arange(F, dtype=np.float32) / fps
    bin_, views, accs = bytearray(), [], []

    def add(arr, typ):
        nonlocal bin_
        data_ = np.asarray(arr, '<f4').ravel().tobytes()
        views.append({'buffer': 0, 'byteOffset': len(bin_), 'byteLength': len(data_)})
        bin_ += data_
        acc = {'bufferView': len(views) - 1, 'componentType': 5126, 'count': len(arr), 'type': typ}
        if typ == 'SCALAR':
            acc.update(min=[float(arr[0])], max=[float(arr[-1])])
        accs.append(acc)
        return len(accs) - 1

    t_all = add(times, 'SCALAR')
    t_two = add(np.array([0, times[-1]], np.float32), 'SCALAR')
    samplers, channels = [], []
    for bone, q in list(rots.items()) + [(b, np.stack([q, q])) for b, q in held.items()]:
        q = np.array(q, np.float32)
        for f in range(1, len(q)):                                      # keep keys on one hemisphere
            if (q[f] * q[f - 1]).sum() < 0:
                q[f] = -q[f]
        samplers.append({'input': t_all if len(q) == F else t_two, 'output': add(q, 'VEC4'), 'interpolation': 'LINEAR'})
        channels.append({'sampler': len(samplers) - 1, 'target': {'node': node[bone], 'path': 'rotation'}})
    samplers.append({'input': t_all, 'output': add(hips.astype(np.float32), 'VEC3'), 'interpolation': 'LINEAR'})
    channels.append({'sampler': len(samplers) - 1, 'target': {'node': node['hips'], 'path': 'translation'}})
    js = {k_: v for k_, v in js0.items() if k_ not in ('animations', 'accessors', 'bufferViews', 'buffers', 'extras')}
    js.update(animations=[{'name': name, 'channels': channels, 'samplers': samplers}], accessors=accs, bufferViews=views,
              buffers=[{'byteLength': len(bin_)}],
              asset={**js0['asset'], 'generator': 'makeanimereal parkour/export.py', 'copyright': credit})
    t = data['terrain']
    terrain = {
        # heights[i][j] is the ground at VRM x = x0 + j * cell, z = z0 + i * cell (PARC x -> VRM z, y -> x).
        'heights': np.round(np.asarray(t['hf'], float) * k, 4).tolist(),
        'x0': round(float(t['min_point'][1]) * k, 4), 'z0': round(float(t['min_point'][0]) * k, 4),
        'cell': round(float(t['dx']) * k, 4), 'scale': round(k, 5), 'credit': credit,
    }
    return write_glb(js, bytes(bin_)), terrain


CREDIT = 'Motion and terrain: PARC dataset (Xu, Shi, Yin, Peng 2025; huggingface.co/datasets/mxucg/PARC, MIT License)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source', help='dec_release.zip or a directory of .pkl files')
    ap.add_argument('--out', default='web/parkour')
    ap.add_argument('--pick', action='append', required=True, help='Label=glob or glob, over the files inside the source')
    ap.add_argument('--template', default='web/moves/wave.vrma')
    a = ap.parse_args()
    rig = rig_from(a.template)
    os.makedirs(a.out, exist_ok=True)
    if a.source.endswith('.zip'):
        z = zipfile.ZipFile(a.source)
        names, read = [n for n in z.namelist() if n.endswith('.pkl')], z.read
    else:
        names = [os.path.relpath(os.path.join(r, f), a.source) for r, _, fs in os.walk(a.source) for f in fs if f.endswith('.pkl')]
        read = lambda n: open(os.path.join(a.source, n), 'rb').read()
    index, labels = [], {}
    for pick in a.pick:
        label, pat = pick.split('=', 1) if '=' in pick else (None, pick)
        found = sorted(fnmatch.filter(names, '*' + pat))
        if not found:
            raise SystemExit(f'nothing matches {pat}')
        for n in found[:1] if label else found:
            name = label.lower().replace(' ', '-') if label else os.path.splitext(os.path.basename(n))[0].replace('_opt_dm', '')
            data = parc.load(read(n))
            if data['terrain'] is None:
                raise SystemExit(f'{n} has no terrain')
            glb, terrain = to_clip(data, rig, name, CREDIT)
            terrain['source'] = n
            if label:
                labels[name] = label
            open(os.path.join(a.out, name + '.vrma'), 'wb').write(glb)
            json.dump(terrain, open(os.path.join(a.out, name + '.json'), 'w'), separators=(',', ':'))
            index.append(name)
            print(f'{name}: {len(glb) // 1024} KB, terrain {len(terrain["heights"])}x{len(terrain["heights"][0])}')
    json.dump({'clips': index, 'labels': labels, 'credit': CREDIT}, open(os.path.join(a.out, 'index.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
