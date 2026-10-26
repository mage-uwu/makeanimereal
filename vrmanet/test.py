"""Check vrmanet: the untrained model against the clips, and the C runtime against model.py.

    python3 vrmanet/test.py [web/moves/vrmanet.vrma]

1. Cold start: gating one expert must replay its clip. Every expert is run for one loop at 60 fps
   and compared with glTF playback of its original .vrma (LINEAR keys, slerp, as three.js plays it).
2. C = numpy: the native build of vrmanet.c is driven with the same control script as model.py
   (switching moves, mixing arms from another expert, a rate change, retriggers) and must agree.
"""
import ctypes
import os
import subprocess
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from model import VRMANet, accessor, read_glb  # noqa: E402


def slerp(a, b, t):
    d = (a * b).sum(-1, keepdims=True)
    b = np.where(d < 0, -b, b)
    d = np.abs(d)
    th = np.arccos(np.clip(d, -1, 1))
    s = np.sin(th)
    wa = np.where(s > 1e-6, np.sin((1 - t) * th) / np.where(s > 1e-6, s, 1), 1 - t)
    wb = np.where(s > 1e-6, np.sin(t * th) / np.where(s > 1e-6, s, 1), t)
    return wa * a + wb * b


def play_clip(path, t, bones):
    """glTF playback of a clip at time t (looping): quaternions [J, 4] in `bones` order, hips."""
    js, b = read_glb(path)
    human = js['extensions']['VRMC_vrm_animation']['humanoid']['humanBones']
    by_node = {v['node']: k for k, v in human.items()}
    anim = js['animations'][0]
    q = np.tile([0, 0, 0, 1.0], (len(bones), 1))
    hips = None
    for ch in anim['channels']:
        s = anim['samplers'][ch['sampler']]
        times, vals = accessor(js, b, s['input'])[:, 0], accessor(js, b, s['output']).astype(float)
        tt = t % times[-1]
        i = min(np.searchsorted(times, tt, side='right') - 1, len(times) - 2)
        f = (tt - times[i]) / (times[i + 1] - times[i])
        j = bones.index(by_node[ch['target']['node']])
        if ch['target']['path'] == 'rotation':
            q[j] = slerp(vals[i], vals[i + 1], f)
        else:
            hips = (1 - f) * vals[i] + f * vals[i + 1]
    return q, hips


def angle(a, b):
    """Degrees between rotations (stable near zero, unlike arccos of the dot product)."""
    b = np.where((a * b).sum(-1, keepdims=True) < 0, -b, b)
    return np.degrees(2 * np.arctan2(np.linalg.norm(a - b, axis=-1), np.linalg.norm(a + b, axis=-1)))


def load_c(net):
    so = os.path.join(here, 'libvrmanet.so')
    subprocess.run(['cc', '-O2', '-shared', '-fPIC', os.path.join(here, 'vrmanet.c'), '-o', so, '-lm'], check=True)
    lib = ctypes.CDLL(so)
    lib.vn_f32.restype = ctypes.POINTER(ctypes.c_float)
    lib.vn_i32.restype = ctypes.POINTER(ctypes.c_int)
    lib.vn_step.argtypes = [ctypes.c_float]
    assert lib.vn_create(net.E, net.J, net.P, net.N, net.U, net.O, len(net.W), net.hips) == 0
    F = dict(W=0, PERIOD=1, A_LOG=2, B=3, C=4, D=5, REST_HIPS=6, WIDTH=7, U=8, X=9, Y=10, GATES=11, RATE=12, POSE=13)

    def put(arr, ptr):
        a = np.ascontiguousarray(arr, np.float32 if ptr._type_ is ctypes.c_float else np.int32).ravel()
        ctypes.memmove(ptr, a.ctypes.data, a.nbytes)

    for k, v in dict(W=net.W, PERIOD=net.period, A_LOG=net.A_log, B=net.B, C=net.C, D=net.Dm,
                     REST_HIPS=net.rest_hips, WIDTH=[net.width]).items():
        put(v, lib.vn_f32(F[k]))
    put(net.key_off, lib.vn_i32(0)); put(net.key_cnt, lib.vn_i32(1)); put(net.part, lib.vn_i32(2))
    lib.F, lib.put = F, put
    return lib


def main(path):
    net = VRMANet.load(path)
    moves = os.path.dirname(path)
    print(f'{net.E} experts, decoder {net.W.shape}, SSM {net.N} states')

    # 1. Cold start = playback.
    worst = 0
    for e, name in enumerate(net.experts):
        clip = os.path.join(moves, f'{name}.vrma')
        net.u[:] = net.control({name: 1}); net.reset(); net.steady()
        errs, herr, t = [], 0, 0.0
        for _ in range(int(net.period[e] * 60)):
            q, h = net.step(1 / 60); t += 1 / 60
            gq, gh = play_clip(clip, t, net.bones)
            errs.append(angle(q, gq).max())
            herr = max(herr, np.abs(h - gh).max())
        worst = max(worst, max(errs))
        print(f'  {name:11s} vs its clip: max bone error {max(errs):.5f} deg, hips {herr * 1000:.3f} mm')
    print(f'cold start: worst bone error {worst:.5f} deg')
    assert worst < 0.01

    # 2. C = numpy, over a control script.
    lib = load_c(net)
    script = [(0, dict(body={'walk': 1})), (90, dict(body={'walk': 1}, arms={'wave': 1})),
              (200, dict(body={'charleston': 1}, rate={'charleston': 1.4})), (330, dict(body={'twist': .5, 'jacks': .5})),
              (420, 'trigger jacks'), (500, None)]
    net.reset(); lib.vn_reset()
    worst_q = worst_h = 0
    for f in range(500):
        for at, what in script:
            if at == f and isinstance(what, dict):
                net.u[:] = net.control(**what); lib.put(net.u, lib.vn_f32(lib.F['U']))
            elif at == f and isinstance(what, str):
                net.trigger(what.split()[1]); lib.vn_trigger(net.experts.index(what.split()[1]))
        q, h = net.step(1 / 60)
        lib.vn_step(1 / 60)
        pose = np.ctypeslib.as_array(lib.vn_f32(lib.F['POSE']), (net.D,))
        worst_q = max(worst_q, angle(q, pose[:4 * net.J].reshape(-1, 4)).max())
        worst_h = max(worst_h, np.abs(h - pose[4 * net.J:]).max())
    print(f'C vs numpy over {f + 1} frames: max bone difference {worst_q:.6f} deg, hips {worst_h * 1000:.4f} mm')
    assert worst_q < 0.001 and worst_h < 1e-5

    # 3. Still a VRMA: every expert is an ordinary animation on the humanoid's nodes.
    js, _ = read_glb(path)
    assert 'VRMC_vrm_animation' in js['extensionsUsed'] and 'extensionsRequired' not in js
    print('all checks passed')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'web/moves/vrmanet.vrma')
