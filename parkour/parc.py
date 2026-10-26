"""Read PARC's parkour dataset (huggingface.co/datasets/mxucg/PARC, MIT) and map it onto the VRM rig.

Each .pkl holds a motion of PARC's 15-body humanoid (MimicKit's humanoid.xml: z up, facing +x, left
+y, a T-pose at zero rotation) and the heightfield it traverses. Both rest poses are T-poses with
no rotation, so retargeting is a change of axes: PARC (x fwd, y left, z up) -> VRM (z fwd, x left,
y up), applied to every rotation and position; each PARC joint drives its VRM counterpart and the
VRM's extra bones (chest, upper chest, shoulders, head, toes) stay at rest.

The files are pickles of pickles holding numpy arrays. They are read with an unpickler that can
only rebuild numpy arrays, so a file cannot run code.
"""
import io
import pickle

import numpy as np

BODIES = ['pelvis', 'torso', 'head', 'right_upper_arm', 'right_lower_arm', 'right_hand', 'left_upper_arm',
          'left_lower_arm', 'left_hand', 'right_thigh', 'right_shin', 'right_foot', 'left_thigh', 'left_shin', 'left_foot']
PARENT = [-1, 0, 1, 1, 3, 4, 1, 6, 7, 0, 9, 10, 0, 12, 13]
OFFSET = np.array([[0, 0, 0], [0, 0, .236151], [0, 0, .223894], [-.02405, -.18311, .2435], [0, -.274788, 0], [0, -.258947, 0],
                   [-.02405, .18311, .2435], [0, .274788, 0], [0, .258947, 0], [0, -.084887, 0], [0, 0, -.421546],
                   [0, 0, -.40987], [0, .084887, 0], [0, 0, -.421546], [0, 0, -.40987]])
# PARC body -> the VRM bone its joint drives.
TO_VRM = {'pelvis': 'hips', 'torso': 'spine', 'head': 'neck', 'right_upper_arm': 'rightUpperArm',
          'right_lower_arm': 'rightLowerArm', 'right_hand': 'rightHand', 'left_upper_arm': 'leftUpperArm',
          'left_lower_arm': 'leftLowerArm', 'left_hand': 'leftHand', 'right_thigh': 'rightUpperLeg',
          'right_shin': 'rightLowerLeg', 'right_foot': 'rightFoot', 'left_thigh': 'leftUpperLeg',
          'left_shin': 'leftLowerLeg', 'left_foot': 'leftFoot'}
# PARC axes -> VRM axes: x (forward) -> z, y (left) -> x, z (up) -> y.
AXES = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)

_ALLOW = {('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'), ('numpy', 'ndarray'),
          ('numpy', 'dtype'), ('numpy.core.multiarray', 'scalar'), ('numpy._core.multiarray', 'scalar')}


class _Arrays(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _ALLOW:
            raise pickle.UnpicklingError(f'refusing {module}.{name}')
        return super().find_class(module.replace('numpy.core', 'numpy._core'), name)


# Joint degrees of freedom per body in the raw frame format (after root position and rotation):
# 3 for a ball joint (an exp map), 1 for a hinge (about AXIS), 0 for the fixed hands.
DOFS = [0, 3, 3, 3, 1, 0, 3, 1, 0, 3, 1, 3, 3, 1, 3]
HINGE_AXIS = {4: [0, 0, 1], 7: [0, 0, 1], 10: [0, 1, 0], 13: [0, 1, 0]}   # elbows, knees


def exp_map_to_quat(e):
    angle = np.linalg.norm(e, axis=-1, keepdims=True)
    axis = np.where(angle > 1e-9, e / np.where(angle > 1e-9, angle, 1), [1, 0, 0])
    return np.concatenate([axis * np.sin(angle / 2), np.cos(angle / 2)], -1)


def frames_to_motion(frames, fps):
    """PARC's raw format ([F, 34]: root position, root exp map, 28 joint dofs) -> load()'s motion dict."""
    f = np.asarray(frames, float)
    joint, col = [], 6
    for k in range(1, 15):
        n = DOFS[k]
        if n == 3:
            joint.append(exp_map_to_quat(f[:, col:col + 3]))
        elif n == 1:
            joint.append(exp_map_to_quat(f[:, col:col + 1] * np.array(HINGE_AXIS[k], float)))
        else:
            joint.append(np.tile([0, 0, 0, 1.0], (len(f), 1)))
        col += n
    return {'root_pos': f[:, :3], 'root_rot': exp_map_to_quat(f[:, 3:6]), 'joint_rot': np.stack(joint, 1),
            'body_contacts': None, 'fps': int(fps)}


def load(path_or_bytes):
    """-> {'motion': {root_pos, root_rot, joint_rot, body_contacts, fps}, 'terrain': {hf, min_point, dx} or None}.
    The 'initial' folder (PARC's seed mocap) is raw frames on flat ground: no terrain, no contacts."""
    b = path_or_bytes if isinstance(path_or_bytes, bytes) else open(path_or_bytes, 'rb').read()
    outer = _Arrays(io.BytesIO(b)).load()
    if 'frames' in outer:
        return {'motion': frames_to_motion(outer['frames'], outer['fps']), 'terrain': None}
    return {'motion': _Arrays(io.BytesIO(outer['motion_data'])).load(),
            'terrain': _Arrays(io.BytesIO(outer['terrain_data'])).load()}


def qmul(a, b):
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def qrot(q, v):
    """Rotate vectors v by quaternions q (x, y, z, w)."""
    u, w = q[..., :3], q[..., 3:]
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def fk(motion):
    """World body positions [F, 15, 3] and rotations [F, 15, 4] in PARC's frame."""
    F = len(motion['root_pos'])
    rot = np.zeros((F, 15, 4)); pos = np.zeros((F, 15, 3))
    rot[:, 0], pos[:, 0] = motion['root_rot'], motion['root_pos']
    for k in range(1, 15):
        p = PARENT[k]
        rot[:, k] = qmul(rot[:, p], motion['joint_rot'][:, k - 1])
        pos[:, k] = pos[:, p] + qrot(rot[:, p], np.broadcast_to(OFFSET[k], (F, 3)))
    return pos, rot


def to_vrm(motion):
    """-> {vrm bone: local rotations [F, 4]}, hips positions [F, 3], in VRM axes (y up, facing +z)."""
    # A rotation R in PARC axes is AXES R AXES^T in VRM axes; on a quaternion that permutes its vector part.
    perm = lambda q: np.concatenate([q[..., :3] @ AXES.T, q[..., 3:]], -1)
    rots = {'hips': perm(motion['root_rot'])}
    for k in range(1, 15):
        rots[TO_VRM[BODIES[k]]] = perm(motion['joint_rot'][:, k - 1])
    return rots, motion['root_pos'] @ AXES.T


def terrain_to_vrm(terrain):
    """Heightfield -> {heights [nx, ny] (rows along VRM x... see below), origin, cell} in VRM axes.
    hf[i, j] is the height at PARC (x, y) = min_point + (i, j) * dx, i.e. VRM (z, x) = that, y = height."""
    hf = np.asarray(terrain['hf'], float)
    return {'heights': hf, 'min_point': np.asarray(terrain['min_point'], float), 'dx': float(terrain['dx'])}
