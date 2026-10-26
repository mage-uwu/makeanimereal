"""PARC's humanoid as a MuJoCo model, and PARC frames <-> MuJoCo states.

PARC's humanoid.xml (MimicKit's DeepMimic humanoid, copied as humanoid_parc.xml) was written for
Isaac Gym, which reads it differently from MuJoCo. build_xml() makes the MuJoCo model behave like
PARC's simulator:
  - each joint's stiffness and damping are PD gains, so they become MuJoCo position actuators
    (kp = stiffness, kv = damping) with the motor's gear as the torque limit; the joints themselves
    get no spring or damper;
  - geoms collide with friction (condim 3; condim 1 would be frictionless in MuJoCo);
  - terrain is static boxes (walls stay vertical), passed in as [x0, x1, y0, y1, height];
  - ball joints (three hinges) have no per-axis limits: PARC's data goes well past the XML's Euler
    ranges at the hips (Isaac Gym limits a ball joint as a whole), so only elbows and knees keep theirs.

A PARC frame is the 78 floats parcgen.c uses: root position 3, root rotation 4 (x, y, z, w), 14
joint rotations 56, contacts 15. Ball joints are three hinges, elbows and knees one. The hinges'
order decides where Euler angles lock (the middle angle at +-90 deg): the XML's x, y, z puts
flexion in the middle for the hips and waist, which reach 90 deg when climbing, so those turn about
y, then x, then z (R = Ry Rx Rz) and lock only at 90 deg of sideways lean, which parkour never does.
"""
import os
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BODIES = ['pelvis', 'torso', 'head', 'right_upper_arm', 'right_lower_arm', 'right_hand', 'left_upper_arm',
          'left_lower_arm', 'left_hand', 'right_thigh', 'right_shin', 'right_foot', 'left_thigh', 'left_shin', 'left_foot']
# The hinges of each body, in qpos order (the XML's order).
JOINTS = {'torso': ['abdomen_y', 'abdomen_x', 'abdomen_z'], 'head': ['neck_x', 'neck_y', 'neck_z'],
          'right_upper_arm': ['right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z'], 'right_lower_arm': ['right_elbow'],
          'left_upper_arm': ['left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z'], 'left_lower_arm': ['left_elbow'],
          'right_thigh': ['right_hip_y', 'right_hip_x', 'right_hip_z'], 'right_shin': ['right_knee'],
          'right_foot': ['right_ankle_x', 'right_ankle_y', 'right_ankle_z'],
          'left_thigh': ['left_hip_y', 'left_hip_x', 'left_hip_z'], 'left_shin': ['left_knee'],
          'left_foot': ['left_ankle_x', 'left_ankle_y', 'left_ankle_z']}
HINGE_AXIS = {'right_elbow': [0, 0, 1], 'left_elbow': [0, 0, 1], 'right_knee': [0, 1, 0], 'left_knee': [0, 1, 0]}
TIMESTEP = 1 / 300          # 10 physics steps per 30 Hz control step


def build_xml(boxes=(), floor=True):
    """The MuJoCo model as XML text."""
    root = ET.parse(os.path.join(HERE, 'humanoid_parc.xml')).getroot()
    gains = {}
    for j in root.iter('joint'):
        if j.get('type', 'hinge') == 'hinge' and j.get('name'):
            gains[j.get('name')] = (float(j.get('stiffness')), float(j.get('damping')))
            j.set('stiffness', '0'); j.set('damping', '0')
            if j.get('name') not in HINGE_AXIS:
                j.set('limited', 'false')
    for d in root.iter('default'):
        for j in d.findall('joint'):
            j.set('stiffness', '0'); j.set('damping', '0')
        for g in d.findall('geom'):
            g.set('condim', '3')
    # Hinges in JOINTS order within each body (the waist and hips turn about y first).
    for b in root.iter('body'):
        js = [j for j in b.findall('joint') if j.get('name')]
        if b.get('name') in JOINTS and len(js) == 3:
            order = JOINTS[b.get('name')]
            first = list(b).index(js[0])
            for j in js:
                b.remove(j)
            for i, name in enumerate(order):
                b.insert(first + i, next(j for j in js if j.get('name') == name))
    # One PD actuator per hinge, in qpos order, so ctrl[i] drives qpos[7 + i].
    act = root.find('actuator')
    motors = {m.get('joint'): float(m.get('gear')) for m in act.findall('motor')}
    for m in list(act):
        act.remove(m)
    for name in [j.get('name') for j in root.iter('joint') if j.get('name') in motors]:
        gear = motors[name]
        kp, kd = gains[name]
        ET.SubElement(act, 'position', name=name, joint=name, kp=str(kp), kv=str(kd), forcelimited='true',
                      forcerange=f'{-gear} {gear}', ctrllimited='false')
    world = root.find('worldbody')
    for g in list(world.findall('geom')):
        world.remove(g)
    if floor:
        ET.SubElement(world, 'geom', name='floor', type='plane', size='100 100 .2', condim='3', friction='1 .05 .05', rgba='.8 .8 .8 1')
    for i, (x0, x1, y0, y1, h) in enumerate(boxes):
        ET.SubElement(world, 'geom', name=f'box{i}', type='box', condim='3', friction='1 .05 .05', rgba='.7 .7 .7 1',
                      pos=f'{(x0 + x1) / 2} {(y0 + y1) / 2} {h / 2}', size=f'{(x1 - x0) / 2} {(y1 - y0) / 2} {h / 2}')
    opt = ET.SubElement(root, 'option', timestep=str(TIMESTEP), integrator='implicitfast')
    root.remove(opt); root.insert(0, opt)
    return ET.tostring(root, encoding='unicode')


# --- PARC frames <-> qpos -------------------------------------------------------------------------

def _rot(q):
    """Quaternions (x, y, z, w) [..., 4] -> matrices [..., 3, 3]."""
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def _quat(axis, angle):
    a = np.asarray(angle)[..., None] / 2
    return np.concatenate([np.asarray(axis, float) * np.sin(a), np.cos(a)], -1)


def _qmul(a, b):
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def euler(R, axes):
    """Rotation matrices -> the two angle triples (first, middle, last hinge) with
    R = R_axes[0] R_axes[1] R_axes[2], for axes 'xyz' or 'yxz'."""
    if axes == 'xyz':
        m = np.arcsin(np.clip(R[:, 0, 2], -1, 1))
        f = np.arctan2(-R[:, 1, 2], R[:, 2, 2]); l = np.arctan2(-R[:, 0, 1], R[:, 0, 0])
    else:                                                    # 'yxz'
        m = np.arcsin(np.clip(-R[:, 1, 2], -1, 1))
        f = np.arctan2(R[:, 0, 2], R[:, 2, 2]); l = np.arctan2(R[:, 1, 0], R[:, 1, 1])
    one = np.stack([f, m, l], 1)
    return one, np.stack([_wrap(f + np.pi), _wrap(np.pi - m), _wrap(l + np.pi)], 1)


def continuous(one, alt, prev=None):
    """Per frame, the solution nearest the previous frame's (angles unwrapped to it), so hinges never
    jump; the first frame takes the one with the smaller middle angle."""
    out = np.empty_like(one)
    for f in range(len(one)):
        if prev is None:
            out[f] = one[f] if np.abs(one[f, 1]) <= np.abs(alt[f, 1]) else alt[f]
        else:
            cands = [prev + _wrap(c - prev) for c in (one[f], alt[f])]
            out[f] = min(cands, key=lambda c: np.abs(c - prev).sum())
        prev = out[f]
    return out


def frames_to_qpos(frames, prev=None):
    """PARC frames [n, 78] -> MuJoCo qpos [n, 35]: root position, root quaternion (w, x, y, z), 28
    hinges, continuous over the frames (and from prev, a qpos to continue from, if given)."""
    f = np.asarray(frames, float)
    out = [f[:, 0:3], f[:, [6, 3, 4, 5]]]
    col = 7
    for k, body in enumerate(BODIES[1:], start=0):
        if body not in JOINTS:
            continue
        q = f[:, 7 + 4 * k:11 + 4 * k]
        names = JOINTS[body]
        if len(names) == 3:
            one, alt = euler(_rot(q), names[0][-1] + names[1][-1] + names[2][-1])
            out.append(continuous(one, alt, None if prev is None else np.asarray(prev, float)[col:col + 3]))
            col += 3
        else:                                               # the angle about the hinge's axis
            axis = np.array(HINGE_AXIS[names[0]], float)
            s = (q[:, :3] * axis).sum(-1)
            out.append((2 * np.arctan2(s, q[:, 3]))[:, None])
            col += 1
    return np.concatenate(out, 1)


def qpos_to_frames(qpos, contacts=None):
    """MuJoCo qpos [n, 35] -> PARC frames [n, 78]."""
    p = np.asarray(qpos, float)
    n = len(p)
    out = np.zeros((n, 78))
    out[:, 0:3] = p[:, 0:3]
    out[:, 3:7] = p[:, [4, 5, 6, 3]]
    col = 7
    for k, body in enumerate(BODIES[1:]):
        if body not in JOINTS:
            q = np.tile([0, 0, 0, 1.0], (n, 1))
        elif len(JOINTS[body]) == 3:
            ax = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}
            a0, a1, a2 = (ax[n[-1]] for n in JOINTS[body])
            q = _qmul(_qmul(_quat(a0, p[:, col]), _quat(a1, p[:, col + 1])), _quat(a2, p[:, col + 2]))
            col += 3
        else:
            q = _quat(HINGE_AXIS[JOINTS[body][0]], p[:, col])
            col += 1
        out[:, 7 + 4 * k:11 + 4 * k] = q
    if contacts is not None:
        out[:, 63:78] = contacts
    return out
