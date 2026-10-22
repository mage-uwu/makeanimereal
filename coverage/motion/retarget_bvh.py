#!/usr/bin/env python3
"""Turn mocap BVH clips into motion data the live viewer retargets onto the VRM.

    python3 retarget_bvh.py <bvh dir> --out coverage/live/motions.json [--clip name=file.bvh ...]

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


def convert(path, ref_axes=None):
    joints, data, dt = parse(path)
    R, P = fk(joints, data)
    idx = {j['name']: k for k, j in enumerate(joints) if j['name']}
    # Face the actor along +Z with their left on +X (a turn about Y, from the first frame's hips).
    left0 = P[idx['UpperLeg_L'], 0] - P[idx['UpperLeg_R'], 0]
    yaw = np.arctan2(left0[2], left0[0])
    turn = rot('Y', np.rad2deg(np.array(yaw)))
    R = turn @ R
    P = P @ turn.T
    up = np.array([0, 1.0, 0])
    left = unit(P[idx['UpperLeg_L'], 0] - P[idx['UpperLeg_R'], 0])
    fwd = unit(np.cross(left, up))
    axes = {}
    out = {}
    for name, (vrm, kind) in MAP.items():
        k = idx[name]
        d_loc = bone_axis(joints, k)
        if ref_axes is None:
            # The reference axis: whichever local axis across the bone lies nearest the body's left
            # (or forward) in the first frame.
            target = left if kind == 'left' else fwd
            best = None
            for ax in (np.eye(3) * s for s in (1, -1)):
                for a in ax:
                    if abs(a @ d_loc) > 0.5:
                        continue
                    score = (R[k, 0] @ a) @ target
                    if best is None or score > best[0]:
                        best = (score, a)
            axes[name] = best[1].tolist()
        a_loc = np.array((ref_axes or axes)[name])
        d = unit(np.einsum('fij,j->fi', R[k], d_loc))
        s = np.einsum('fij,j->fi', R[k], a_loc)
        s = unit(s - (s * d).sum(-1, keepdims=True) * d)
        out[vrm] = np.concatenate([d, s], -1)
    leg = np.linalg.norm(P[idx['LowerLeg_L'], 0] - P[idx['UpperLeg_L'], 0]) + np.linalg.norm(P[idx['Foot_L'], 0] - P[idx['LowerLeg_L'], 0])
    hips = (P[idx['Hips']] - P[idx['Hips'], 0] * np.array([1, 0, 1])) / leg
    # The ground: the lowest the feet get in the clip, so the hips' height is measured from it.
    ground = min(P[idx['Toes_L'], :, 1].min(), P[idx['Toes_R'], :, 1].min(), P[idx['Foot_L'], :, 1].min(), P[idx['Foot_R'], :, 1].min())
    hips[:, 1] = (P[idx['Hips'], :, 1] - ground) / leg
    # In place: take out the travel (a 1.4 s moving average of the hips' ground path), keep the sway.
    w = int(1.4 / dt) | 1
    pad = np.pad(hips[:, [0, 2]], ((w // 2, w // 2), (0, 0)), mode='edge')
    path = np.stack([np.convolve(pad[:, k], np.ones(w) / w, mode='valid') for k in range(2)], 1)
    hips[:, [0, 2]] -= path
    return {'fps': round(1 / dt), 'frames': len(data), 'hips': hips, 'frames_by_bone': out}, axes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bvh_dir')
    ap.add_argument('--out', default='coverage/live/motions.json')
    ap.add_argument('--clip', action='append', required=True, help='label=file.bvh (relative to bvh_dir)')
    ap.add_argument('--fps', type=int, default=30)
    a = ap.parse_args()
    d = Path(a.bvh_dir)
    # Reference axes chosen once, from a clip that starts standing upright, and kept for every clip.
    c0, ref = convert(d / a.clip[0].split('=', 1)[1])
    stand = float(np.percentile(c0['hips'][:, 1], 95))   # the hips' height standing tall
    bones = [v for v, _ in MAP.values()]
    clips = {}
    for spec in a.clip:
        label, f = spec.split('=', 1)
        c, _ = convert(d / f, ref)
        step = max(1, round(c['fps'] / a.fps))
        frames = np.stack([c['frames_by_bone'][b] for b in bones], 1)[::step]   # [F, B, 6]
        clips[label] = {
            'source': f, 'frames': len(frames),
            'hips': np.round(c['hips'][::step], 4).ravel().tolist(),
            'axes': np.round(frames, 4).ravel().tolist(),
        }
        print(f'{label}: {f}, {len(frames)} frames', flush=True)
    json.dump({'fps': a.fps, 'stand': round(stand, 4), 'bones': bones, 'kinds': [k for _, k in MAP.values()], 'clips': clips,
               'credit': 'Motion: Bandai Namco Research Motion Dataset (CC BY-NC 4.0)'},
              open(a.out, 'w'), separators=(',', ':'))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
