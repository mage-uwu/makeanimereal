"""Reference motions for the tracker: PARC clips with everything the environment needs, precomputed.

    clips = load_clips(zip_path, patterns, limit)

Per clip (30 fps): the PARC frames; MuJoCo qpos/qvel (hinges continuous, root velocities by finite
differences, angular ones in the root's frame as MuJoCo's free joint has them); body positions and
rotations by PARC's forward kinematics; and the clip's terrain, both as the heightfield the policy
samples and as boxes for the simulator (cells merged into as few boxes as possible).
"""
import fnmatch
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'parkour'))
import body  # noqa: E402
import parc  # noqa: E402

FPS = 30


def quat_log_vel(q0, q1, dt):
    """Angular velocity (in q0's frame) taking q0 to q1 in dt; quaternions (x, y, z, w) [..., 4]."""
    inv = q0 * np.array([-1, -1, -1, 1.0])
    d = body._qmul(inv, q1)
    d = np.where(d[..., 3:] < 0, -d, d)
    s = np.linalg.norm(d[..., :3], axis=-1, keepdims=True)
    angle = 2 * np.arctan2(s, d[..., 3:])
    axis = d[..., :3] / np.maximum(s, 1e-9)
    return axis * angle / dt


def terrain_boxes(hf, min_point, dx):
    """Heightfield -> (floor height, boxes [x0, x1, y0, y1, height above the floor] with the floor
    at 0): equal-height cells merged greedily into rectangles."""
    hf = np.asarray(hf, float)
    floor = hf.min()
    h = np.round(hf - floor, 3)
    done = h <= 1e-3
    boxes = []
    nx, ny = h.shape
    for i in range(nx):
        for j in range(ny):
            if done[i, j]:
                continue
            v = h[i, j]
            j1 = j
            while j1 + 1 < ny and not done[i, j1 + 1] and h[i, j1 + 1] == v:
                j1 += 1
            i1 = i
            while i1 + 1 < nx and all(not done[i1 + 1, jj] and h[i1 + 1, jj] == v for jj in range(j, j1 + 1)):
                i1 += 1
            done[i:i1 + 1, j:j1 + 1] = True
            # Cell (i, j) covers its sample point +- dx / 2 (PARC rounds points to the nearest sample).
            boxes.append([min_point[0] + (i - 0.5) * dx, min_point[0] + (i1 + 0.5) * dx,
                          min_point[1] + (j - 0.5) * dx, min_point[1] + (j1 + 0.5) * dx, v])
    return floor, boxes


class Clip:
    def __init__(self, name, data):
        m, t = data['motion'], data['terrain']
        F = len(m['root_pos'])
        self.name, self.F = name, F
        frames = np.concatenate([m['root_pos'], m['root_rot'], np.asarray(m['joint_rot']).reshape(F, -1),
                                 m['body_contacts'] if m['body_contacts'] is not None else np.zeros((F, 15))], -1).astype(float)
        # The terrain: the floor moves to z = 0 (the motion moves with it).
        self.hf, self.min_point, self.dx = np.asarray(t['hf'], float), np.asarray(t['min_point'], float), float(t['dx'])
        floor, self.boxes = terrain_boxes(self.hf, self.min_point, self.dx)
        self.hf = self.hf - floor
        frames[:, 2] -= floor
        self.frames = frames
        self.qpos = body.frames_to_qpos(frames)
        qvel = np.zeros((F, 34))
        dt = 1 / FPS
        qvel[:-1, 0:3] = (frames[1:, 0:3] - frames[:-1, 0:3]) / dt
        qvel[:-1, 3:6] = quat_log_vel(frames[:-1, 3:7], frames[1:, 3:7], dt)
        qvel[:-1, 6:] = (self.qpos[1:, 7:] - self.qpos[:-1, 7:]) / dt
        qvel[-1] = qvel[-2]
        self.qvel = qvel
        self.body_pos, self.body_rot = parc.fk({'root_pos': frames[:, 0:3], 'root_rot': frames[:, 3:7],
                                                'joint_rot': frames[:, 7:63].reshape(F, 14, 4)})
        self.contacts = frames[:, 63:78]

    def ground(self, xy):
        i = np.clip(np.round((xy - self.min_point) / self.dx).astype(int), 0, np.array(self.hf.shape) - 1)
        return self.hf[i[..., 0], i[..., 1]]


def load_clips(zip_path, patterns, limit=None, min_frames=45, seed=0):
    z = zipfile.ZipFile(zip_path)
    names = sorted(n for n in z.namelist() if n.endswith('.pkl') and any(fnmatch.fnmatch(n, '*' + p) for p in patterns))
    rng = np.random.default_rng(seed)
    if limit and len(names) > limit:
        names = list(rng.choice(names, limit, replace=False))
    clips = []
    for n in names:
        d = parc.load(z.read(n))
        if d['terrain'] is None or len(d['motion']['root_pos']) < min_frames:
            continue
        clips.append(Clip(n, d))
    return clips
