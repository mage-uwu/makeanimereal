"""Ingest .vrma clips as the experts of one vrmanet model file.

    python3 vrmanet/build.py web/moves/wave.vrma web/moves/walk.vrma ... --out web/moves/vrmanet.vrma

Every clip must be baked on the same rest rig (scripts/bake-vrma.mjs does that). Each clip's
animation is copied in one to one, its accessors' bytes unchanged, so the model's decoder rows are
exactly the clips' keyframes. The controller starts as an exact follower of its inputs: one state
per output, A = -1/tau, B = 1/tau, C = I, D = 0, so each gate and rate eases to its input with time
constant tau and the untrained model plays the clips as they are.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import EXT, VRMANet, read_glb, write_glb  # noqa: E402

PARTS = ['lower', 'torso', 'arms']


def part_of(bone):
    if bone == 'hips' or any(s in bone for s in ('UpperLeg', 'LowerLeg', 'Foot', 'Toes')):
        return 0
    if bone in ('spine', 'chest', 'upperChest', 'neck', 'head', 'jaw', 'leftEye', 'rightEye'):
        return 1
    return 2


def build(paths, tau=0.3):
    js, bin_ = None, bytearray()
    for path in paths:
        cj, cb = read_glb(path)
        if js is None:
            js = {k: v for k, v in cj.items() if k not in ('animations', 'accessors', 'bufferViews', 'buffers', 'extras')}
            js.update(animations=[], accessors=[], bufferViews=[], buffers=[{}])
            credits = []
        assert [(n['name'], n.get('translation')) for n in cj['nodes']] == [(n['name'], n.get('translation')) for n in js['nodes']], \
            f'{path}: a different rest rig'
        # Copy the clip's buffer views and accessors, then its animation pointing at the copies.
        views0, acc0 = len(js['bufferViews']), len(js['accessors'])
        for v in cj['bufferViews']:
            bin_ += b'\0' * (-len(bin_) % 4)
            start = v.get('byteOffset', 0)
            js['bufferViews'].append({**v, 'buffer': 0, 'byteOffset': len(bin_)})
            bin_ += cb[start:start + v['byteLength']]
        for a in cj['accessors']:
            js['accessors'].append({**a, 'bufferView': a['bufferView'] + views0})
        for anim in cj['animations']:
            name = anim.get('name') or os.path.splitext(os.path.basename(path))[0]
            js['animations'].append({**anim, 'name': name, 'samplers': [
                {**s, 'input': s['input'] + acc0, 'output': s['output'] + acc0} for s in anim['samplers']]})
        c = cj.get('asset', {}).get('copyright')
        if c and c not in credits:
            credits.append(c)

    human = js['extensions']['VRMC_vrm_animation']['humanoid']['humanBones']
    E, P = len(js['animations']), len(PARTS)
    O = N = U = E * P + E

    def add(name, arr):
        nonlocal bin_
        bin_ += b'\0' * (-len(bin_) % 4)
        data = np.asarray(arr, '<f4').ravel().tobytes()
        js['bufferViews'].append({'buffer': 0, 'byteOffset': len(bin_), 'byteLength': len(data)})
        bin_ += data
        js['accessors'].append({'name': name, 'bufferView': len(js['bufferViews']) - 1, 'componentType': 5126,
                                'count': len(data) // 4, 'type': 'SCALAR'})
        return len(js['accessors']) - 1

    A_log = np.full(N, np.log(1 / tau))
    js['extensions'][EXT] = {
        'experts': [{'name': a['name'], 'animation': i} for i, a in enumerate(js['animations'])],
        'parts': PARTS,
        'boneParts': {b: part_of(b) for b in human},
        'kernelWidth': 1.0,
        'ssm': {
            'states': N, 'inputs': U, 'outputs': O,
            'A_log': add('vrmanet.ssm.A_log [N]', A_log),
            'B': add('vrmanet.ssm.B [N x U]', np.diag(np.exp(A_log))),
            'C': add('vrmanet.ssm.C [O x N]', np.eye(O, N)),
            'D': add('vrmanet.ssm.D [O x U]', np.zeros((O, U))),
        },
        'io': 'input and output e*P+p: gate of expert e on body part p; E*P+e: playback rate of expert e, minus 1',
    }
    js['extensionsUsed'] = sorted(set(js.get('extensionsUsed', [])) | {EXT})
    js['asset'] = {**js['asset'], 'generator': 'makeanimereal vrmanet/build.py', 'copyright': '; '.join(credits)}
    js['buffers'] = [{'byteLength': len(bin_)}]
    return js, bytes(bin_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('clips', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tau', type=float, default=0.3, help='seconds the untrained controller takes to follow its inputs')
    a = ap.parse_args()
    js, bin_ = build(a.clips, a.tau)
    open(a.out, 'wb').write(write_glb(js, bin_))
    net = VRMANet(js, bin_)
    print(f'{a.out}: {net.E} experts, {len(net.W)} keyframes x {net.D} (decoder), '
          f'SSM {net.N} states, {net.U} inputs, {net.O} outputs; {os.path.getsize(a.out) // 1024} KB')


if __name__ == '__main__':
    main()
