#!/usr/bin/env python3
"""Turn mocap BVH clips into motion data the live viewer retargets onto the VRM.

    python3 retarget_bvh.py <bvh dir> --out coverage/live/motions.json [--clip name=file.bvh ...]
    python3 retarget_bvh.py <cmu dir> --out web/motions.json --credit "..." --clip wave=141.asf:141_16.amc ...

Besides BVH it reads the CMU motion capture database's Acclaim files (a subject's .asf skeleton and
a trial's .amc motion, given as skeleton.asf:motion.amc; 120 fps unless --clip-fps says otherwise).

The BVH rigs (Bandai Namco Research motion dataset, Maya-style joint frames) have no T-pose to copy
angles from, so the clips are stored as semantic frames instead: for each joint and frame, the
world-space direction of the bone and of a reference axis (the body's left for the trunk, legs and
head; the body's forward for shoulders, arms and hands). The viewer builds the same frames from the
VRM's rest skeleton and takes the rotation between them, so neither rig's axis conventions nor
proportions matter. The actor is turned to face +Z with their left on +X; the hips' path is kept,
in units of the actor's leg length.
"""
import argparse
import json
from pathlib import Path

import numpy as np

# BVH joint -> VRM humanoid bone, and the reference axis each one is tracked with.
MAP = {
    'Hips': ('hips', 'left'), 'Spine': ('spine', 'left'), 'Chest': ('chest', 'left'),
    'Neck': ('neck', 'left'), 'Head': ('head', 'left'),
    'Shoulder_L': ('leftShoulder', 'fwd'), 'UpperArm_L': ('leftUpperArm', 'fwd'),
    'LowerArm_L': ('leftLowerArm', 'fwd'), 'Hand_L': ('leftHand', 'fwd'),
    'Shoulder_R': ('rightShoulder', 'fwd'), 'UpperArm_R': ('rightUpperArm', 'fwd'),
    'LowerArm_R': ('rightLowerArm', 'fwd'), 'Hand_R': ('rightHand', 'fwd'),
    'UpperLeg_L': ('leftUpperLeg', 'left'), 'LowerLeg_L': ('leftLowerLeg', 'left'),
    'Foot_L': ('leftFoot', 'left'), 'Toes_L': ('leftToes', 'left'),
    'UpperLeg_R': ('rightUpperLeg', 'left'), 'LowerLeg_R': ('rightLowerLeg', 'left'),
    'Foot_R': ('rightFoot', 'left'), 'Toes_R': ('rightToes', 'left'),
}


def parse(path):
    """-> joints [{name, parent, offset, channels, child}], motion [frames, channels], frame time."""
    tok = Path(path).read_text().split()
    joints, stack, i = [], [], 0
    while tok[i] != 'MOTION':
        t = tok[i]
        if t in ('ROOT', 'JOINT'):
            joints.append({'name': tok[i + 1], 'parent': stack[-1] if stack else -1, 'channels': [], 'child': None})
            i += 2
        elif t == 'End':
            joints.append({'name': None, 'parent': stack[-1], 'channels': [], 'child': None})
            i += 2
        elif t == '{':
            stack.append(len(joints) - 1)
            i += 1
        elif t == '}':
            stack.pop()
            i += 1
        elif t == 'OFFSET':
            joints[stack[-1]]['offset'] = np.array(tok[i + 1:i + 4], float)
            i += 4
        elif t == 'CHANNELS':
            n = int(tok[i + 1])
            joints[stack[-1]]['channels'] = tok[i + 2:i + 2 + n]
            i += 2 + n
        else:
            i += 1
    nf = int(tok[i + 2])
    dt = float(tok[i + 5])
    data = np.array(tok[i + 6:], float).reshape(nf, -1)
    for k, j in enumerate(joints):
        if j['parent'] >= 0 and joints[j['parent']]['child'] is None:
            joints[j['parent']]['child'] = k
    return joints, data, dt


def rot(axis, deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    r = np.zeros(a.shape + (3, 3))
    i, j = {'X': (1, 2), 'Y': (2, 0), 'Z': (0, 1)}[axis]
    k = 3 - i - j
    r[..., k, k] = 1
    r[..., i, i] = c; r[..., j, j] = c
    r[..., i, j] = -s; r[..., j, i] = s
    return r


def fk(joints, data):
    """World rotations [J, F, 3, 3] and positions [J, F, 3]."""
    nf = len(data)
    R, P, col = [], [], 0
    for j in joints:
        local_r = np.broadcast_to(np.eye(3), (nf, 3, 3)).copy()
        local_t = np.broadcast_to(j['offset'], (nf, 3)).copy()
        for c in j['channels']:
            v = data[:, col]
            col += 1
            if c.endswith('position'):
                local_t[:, 'XYZ'.index(c[0])] = v
            else:
                local_r = local_r @ rot(c[0], v)
        if j['parent'] < 0:
            R.append(local_r); P.append(local_t)
        else:
            pr, pp = R[j['parent']], P[j['parent']]
            R.append(pr @ local_r); P.append(pp + np.einsum('fij,fj->fi', pr, local_t))
    return np.stack(R), np.stack(P)


def bone_axis(joints, k):
    """The joint's local axis along its bone (towards its child, or continuing its parent's bone)."""
    c = joints[k]['child']
    o = joints[c]['offset'] if c is not None else np.zeros(3)
    if np.linalg.norm(o) < 1e-6:
        o = joints[k]['offset']
    return o / np.linalg.norm(o)


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def load_bvh(path):
    """-> {joint: (world rotations [F, 3, 3], positions [F, 3], local bone axis [3])}, frame time."""
    joints, data, dt = parse(path)
    R, P = fk(joints, data)
    return {j['name']: (R[k], P[k], bone_axis(joints, k)) for k, j in enumerate(joints) if j['name']}, dt


# CMU (Acclaim) bone -> the BVH joint names above. The Acclaim rest pose is in world axes, so a
# bone's local axis is its rest direction. Hips: the root, aimed up the lower back.
ASF_MAP = {
    'Hips': 'root', 'Spine': 'lowerback', 'Chest': 'thorax', 'Neck': 'lowerneck', 'Head': 'head',
    'Shoulder_L': 'lclavicle', 'UpperArm_L': 'lhumerus', 'LowerArm_L': 'lradius', 'Hand_L': 'lhand',
    'Shoulder_R': 'rclavicle', 'UpperArm_R': 'rhumerus', 'LowerArm_R': 'rradius', 'Hand_R': 'rhand',
    'UpperLeg_L': 'lfemur', 'LowerLeg_L': 'ltibia', 'Foot_L': 'lfoot', 'Toes_L': 'ltoes',
    'UpperLeg_R': 'rfemur', 'LowerLeg_R': 'rtibia', 'Foot_R': 'rfoot', 'Toes_R': 'rtoes',
}


def euler(axes, deg):
    """Acclaim Euler angles [F, n] about the named axes, applied in order -> [F, 3, 3]."""
    r = np.broadcast_to(np.eye(3), deg.shape[:1] + (3, 3)).copy()
    for k, a in enumerate(axes):
        r = rot(a.upper(), deg[:, k]) @ r
    return r


def load_asf_amc(asf, amc, fps):
    """The same as load_bvh, from an Acclaim skeleton and motion."""
    bones, parent, order, block = {'root': {'dir': np.zeros(3), 'len': 0.0, 'C': np.eye(3), 'dof': []}}, {}, [], None
    lines = Path(asf).read_text().splitlines()
    section = None
    for ln in lines:
        t = ln.split()
        if not t or t[0].startswith('#'):
            continue
        if t[0].startswith(':'):
            section = t[0]
            if section == ':root':
                bones['root']['dof'] = []
            continue
        if section == ':root' and t[0] == 'order':
            bones['root']['dof'] = [x.lower() for x in t[1:]]
        elif section == ':bonedata':
            if t[0] == 'begin':
                block = {'dof': []}
            elif t[0] == 'end':
                bones[block['name']] = block
            elif t[0] == 'name':
                block['name'] = t[1]
            elif t[0] == 'direction':
                block['dir'] = np.array(t[1:4], float)
            elif t[0] == 'length':
                block['len'] = float(t[1])
            elif t[0] == 'axis':
                block['C'] = euler(t[4], np.array([t[1:4]], float))[0]
            elif t[0] == 'dof':
                block['dof'] = t[1:]
        elif section == ':hierarchy' and t[0] not in ('begin', 'end'):
            for c in t[1:]:
                parent[c] = t[0]
                order.append(c)
    # Motion: per frame, each bone's values in its dof order.
    frames, cur = [], None
    for ln in Path(amc).read_text().splitlines():
        t = ln.split()
        if not t or t[0].startswith(('#', ':')):
            continue
        if len(t) == 1 and t[0].isdigit():
            cur = {}
            frames.append(cur)
        elif cur is not None:
            cur[t[0]] = np.array(t[1:], float)
    F = len(frames)
    R, P = {}, {}
    for name in ['root'] + order:
        b = bones[name]
        vals = np.stack([f.get(name, np.zeros(len(b['dof']))) for f in frames]) if b['dof'] else np.zeros((F, 0))
        rdofs = [(d, k) for k, d in enumerate(b['dof']) if d.startswith('r')]
        M = euler([d[1] for d, _ in rdofs], vals[:, [k for _, k in rdofs]]) if rdofs else np.broadcast_to(np.eye(3), (F, 3, 3))
        L = b['C'] @ M @ b['C'].T
        if name == 'root':
            R[name] = L
            P[name] = vals[:, [b['dof'].index(f't{a}') for a in 'xyz']]
        else:
            pa = parent[name]
            R[name] = R[pa] @ L
            # A bone starts where its parent ends.
            P[name] = P[pa] + (np.einsum('fij,j->fi', R[pa], bones[pa]['dir'] * bones[pa]['len']) if pa != 'root' else 0)
    axis = {n: bones[n]['dir'] / np.linalg.norm(bones[n]['dir']) for n in bones if n != 'root'}
    axis['root'] = axis['lowerback']
    return {k: (R[v], P[v], axis[v]) for k, v in ASF_MAP.items()}, 1 / fps


# Acclaim skeletons stand in a T-pose in world axes, facing +Z with their left on +X: the reference
# axes come from that rest pose rather than from a first frame (where a head may be turned).
ASF_REF = {name: [1.0, 0, 0] if kind == 'left' else [0, 0, 1.0] for name, (_, kind) in MAP.items()}


def load(spec, d, fps):
    """-> skeleton, frame time, reference axes (None: choose them from the first frame)."""
    if ':' in spec:
        asf, amc = spec.split(':', 1)
        return (*load_asf_amc(d / asf, d / amc, fps), ASF_REF)
    return (*load_bvh(d / spec), None)


def heading(P):
    """Per frame, the angle (degrees, unwrapped) of the hips' left-to-right line about Y."""
    left = P['UpperLeg_L'] - P['UpperLeg_R']
    return np.rad2deg(np.unwrap(np.arctan2(left[:, 2], left[:, 0])))


def smooth(x, w):
    pad = np.pad(x, ((w // 2, w // 2),) + ((0, 0),) * (x.ndim - 1), mode='edge')
    return np.stack([np.convolve(pad[:, k], np.ones(w) / w, mode='valid') for k in range(x.shape[1])], 1) if x.ndim > 1 \
        else np.convolve(pad, np.ones(w) / w, mode='valid')


def convert(skel, dt, ref_axes=None, face_front=False):
    R = {k: v[0] for k, v in skel.items()}
    P = {k: v[1] for k, v in skel.items()}
    # Face the actor along +Z with their left on +X (a turn about Y, from the first frame's hips).
    left0 = P['UpperLeg_L'][0] - P['UpperLeg_R'][0]
    yaw = np.arctan2(left0[2], left0[0])
    turn = rot('Y', np.rad2deg(np.array(yaw)))
    R = {k: turn @ v for k, v in R.items()}
    P = {k: v @ turn.T for k, v in P.items()}
    if face_front:
        # Take out the turning too (a 1.4 s moving average of the heading), so a clip that walks a
        # loop round the room plays facing the front; quick turns within it stay.
        # Turned about the actor's own (smoothed) path, not the room's origin, which would swing them wide.
        w = int(1.4 / dt) | 1
        back = rot('Y', smooth(heading(P), w))
        c = np.zeros_like(P['Hips'])
        c[:, [0, 2]] = smooth(P['Hips'][:, [0, 2]], w)
        R = {k: back @ v for k, v in R.items()}
        P = {k: np.einsum('fij,fj->fi', back, v - c) + c for k, v in P.items()}
    up = np.array([0, 1.0, 0])
    left = unit(P['UpperLeg_L'][0] - P['UpperLeg_R'][0])
    fwd = unit(np.cross(left, up))
    axes = {}
    out = {}
    for name, (vrm, kind) in MAP.items():
        Rk, d_loc = R[name], skel[name][2]
        if ref_axes is None:
            # The reference axis: whichever local axis across the bone lies nearest the body's left
            # (or forward) in the first frame.
            target = left if kind == 'left' else fwd
            best = None
            for ax in (np.eye(3) * s for s in (1, -1)):
                for a in ax:
                    if abs(a @ d_loc) > 0.5:
                        continue
                    score = (Rk[0] @ a) @ target
                    if best is None or score > best[0]:
                        best = (score, a)
            axes[name] = best[1].tolist()
        a_loc = np.array((ref_axes or axes)[name])
        d = unit(np.einsum('fij,j->fi', Rk, d_loc))
        s = np.einsum('fij,j->fi', Rk, a_loc)
        s = unit(s - (s * d).sum(-1, keepdims=True) * d)
        out[vrm] = np.concatenate([d, s], -1)
    leg = np.linalg.norm(P['LowerLeg_L'][0] - P['UpperLeg_L'][0]) + np.linalg.norm(P['Foot_L'][0] - P['LowerLeg_L'][0])
    hips = (P['Hips'] - P['Hips'][0] * np.array([1, 0, 1])) / leg
    # The ground: the lowest the feet get in the clip, so the hips' height is measured from it.
    ground = min(P[n][:, 1].min() for n in ('Toes_L', 'Toes_R', 'Foot_L', 'Foot_R'))
    hips[:, 1] = (P['Hips'][:, 1] - ground) / leg
    # In place: take out the travel (a 1.4 s moving average of the hips' ground path), keep the sway.
    hips[:, [0, 2]] -= smooth(hips[:, [0, 2]], int(1.4 / dt) | 1)
    return {'fps': round(1 / dt), 'frames': len(hips), 'hips': hips, 'frames_by_bone': out}, axes or ref_axes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bvh_dir')
    ap.add_argument('--out', default='coverage/live/motions.json')
    ap.add_argument('--clip', action='append', required=True, help='label=file.bvh or label=skeleton.asf:motion.amc (relative to bvh_dir)')
    ap.add_argument('--clip-fps', type=float, default=120, help='frame rate of the .amc files (CMU: 120, a few 60)')
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--face-front', action='store_true', help='also take out slow turning, so every clip faces the front')
    ap.add_argument('--credit', default='Motion: Bandai Namco Research Motion Dataset (CC BY-NC 4.0)')
    a = ap.parse_args()
    d = Path(a.bvh_dir)
    # Reference axes chosen once, from a clip that starts standing upright, and kept for every clip.
    c0, ref = convert(*load(a.clip[0].split('=', 1)[1], d, a.clip_fps), a.face_front)
    stand = float(np.percentile(c0['hips'][:, 1], 95))   # the hips' height standing tall
    bones = [v for v, _ in MAP.values()]
    clips = {}
    for spec in a.clip:
        label, f = spec.split('=', 1)
        skel, dt, rest_ref = load(f, d, a.clip_fps)
        c, _ = convert(skel, dt, rest_ref or ref, a.face_front)
        step = max(1, round(c['fps'] / a.fps))
        frames = np.stack([c['frames_by_bone'][b] for b in bones], 1)[::step]   # [F, B, 6]
        clips[label] = {
            'source': f, 'frames': len(frames),
            'hips': np.round(c['hips'][::step], 4).ravel().tolist(),
            'axes': np.round(frames, 4).ravel().tolist(),
        }
        print(f'{label}: {f}, {len(frames)} frames', flush=True)
    json.dump({'fps': a.fps, 'stand': round(stand, 4), 'bones': bones, 'kinds': [k for _, k in MAP.values()], 'clips': clips,
               'credit': a.credit},
              open(a.out, 'w'), separators=(',', ':'))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
