"""vrmanet: a motion network whose weights are a VRM Animation (.vrma) file.

The file is the model, one to one:

  decoder W    the .vrma's own animation tracks. Each glTF animation is one expert (a mocap clip);
               each of its keyframes is one row of W: every bone's rotation (x, y, z, w), then the
               hips' position.
  attention    an expert reads its rows with a clock. The clock's phase finds the keyframes around
               it and blends them with a triangle kernel; at width 1 that is exactly glTF LINEAR
               sampling. The keys are the keyframe times already in the file: they cost no weights,
               and only the rows inside the kernel window are touched.
  controller   a diagonal state-space model (S4D-style, zero-order hold) from control inputs to a
               gate per (expert, body part) and a playback rate per expert. Its weights A_log, B, C,
               D are glTF accessors under the MAKEANIMEREAL_vrmanet extension.
  mixture      each body part mixes the experts' poses by its gates (sign-aligned, normalized
               quaternion sums), so the arms can wave while the legs walk.

Untrained (build.py's initial weights) the controller just follows its inputs with a 0.3 s lag, so
gating one expert plays its clip exactly. A VRMA player still plays the file: it sees ordinary
animations and ignores the extension. vrmanet.c is step() below, line for line, in C (and WASM).

    net = VRMANet.load('web/moves/vrmanet.vrma')
    net.u[:] = net.control({'walk': 1}, arms={'wave': 1}); net.reset(); net.steady()
    quats, hips = net.step(1 / 60)
"""
import json
import struct

import numpy as np

EXT = 'MAKEANIMEREAL_vrmanet'
COMPS = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}


def read_glb(path_or_bytes):
    """-> (glTF JSON, binary chunk)."""
    b = path_or_bytes if isinstance(path_or_bytes, bytes) else open(path_or_bytes, 'rb').read()
    magic, version, _ = struct.unpack_from('<III', b, 0)
    assert magic == 0x46546C67 and version == 2, 'not a glTF binary'
    n = struct.unpack_from('<I', b, 12)[0]
    js = json.loads(b[20:20 + n])
    m = struct.unpack_from('<I', b, 20 + n)[0]
    return js, b[28 + n:28 + n + m]


def write_glb(js, bin_):
    pad = lambda b, fill: b + fill * (-len(b) % 4)
    j, b = pad(json.dumps(js, separators=(',', ':')).encode(), b' '), pad(bin_, b'\0')
    return (struct.pack('<III', 0x46546C67, 2, 28 + len(j) + len(b)) + struct.pack('<II', len(j), 0x4E4F534A) + j
            + struct.pack('<II', len(b), 0x004E4942) + b)


def accessor(js, bin_, i):
    """A float accessor as an array [count, components] (tightly packed views only)."""
    a = js['accessors'][i]
    v = js['bufferViews'][a['bufferView']]
    assert a['componentType'] == 5126 and 'byteStride' not in v, 'vrmanet reads tightly packed float32 accessors'
    c = COMPS[a['type']]
    return np.frombuffer(bin_, '<f4', a['count'] * c, v.get('byteOffset', 0) + a.get('byteOffset', 0)).reshape(-1, c)


class VRMANet:
    def __init__(self, js, bin_):
        ext = js['extensions'][EXT]
        human = js['extensions']['VRMC_vrm_animation']['humanoid']['humanBones']
        by_node = {v['node']: k for k, v in human.items()}
        self.bones = [by_node[i] for i in sorted(by_node)]          # bone j <-> node order
        self.parts = ext['parts']
        self.part = np.array([ext['boneParts'][b] for b in self.bones])
        self.experts = [x['name'] for x in ext['experts']]
        self.E, self.J, self.P = len(self.experts), len(self.bones), len(self.parts)
        self.D = 4 * self.J + 3                                     # row: J quaternions, hips position
        self.hips = self.bones.index('hips')
        assert all(c.get('children') is None or human['hips']['node'] not in c['children'] for c in js['nodes']), \
            'hips must be a root node'
        self.rest_hips = np.array(js['nodes'][human['hips']['node']].get('translation', [0, 0, 0]), float)
        self.width = float(ext['kernelWidth'])

        # Decoder: each expert's keyframes, one row each (the loop's closing key repeats the first).
        rows, self.key_off, self.key_cnt, self.period = [], [], [], []
        for x in ext['experts']:
            anim = js['animations'][x['animation']]
            times = max((accessor(js, bin_, s['input'])[:, 0] for s in anim['samplers']), key=len)
            K, T = len(times) - 1, float(times[-1])
            assert np.allclose(times, np.linspace(0, T, K + 1), atol=1e-4), f"{x['name']}: keys must be evenly spaced"
            W = np.zeros((K, self.D))
            W[:, 3:4 * self.J:4] = 1                                 # bones without a track stay at rest
            W[:, 4 * self.J:] = self.rest_hips
            for ch in anim['channels']:
                s = anim['samplers'][ch['sampler']]
                v, t = accessor(js, bin_, s['output']), accessor(js, bin_, s['input'])[:, 0]
                if len(t) == K + 1:
                    v = v[:K]
                elif np.allclose(v, v[0]):                           # a held track: one key over the loop
                    v = np.broadcast_to(v[0], (K, v.shape[1]))
                else:
                    raise ValueError(f"{x['name']}: track not on the expert's keys")
                j = self.bones.index(by_node[ch['target']['node']])
                if ch['target']['path'] == 'rotation':
                    W[:, 4 * j:4 * j + 4] = v
                elif j == self.hips:
                    W[:, 4 * self.J:] = v
            self.key_off.append(sum(self.key_cnt))
            self.key_cnt.append(K)
            self.period.append(T)
            rows.append(W)
        self.W = np.concatenate(rows)                               # [keys, D]
        self.period = np.array(self.period)

        # Controller: x' = A x + B u (A diagonal, A = -exp(A_log)), y = C x + D u.
        s = ext['ssm']
        self.N, self.U, self.O = s['states'], s['inputs'], s['outputs']
        assert self.O == self.E * self.P + self.E
        self.A_log = accessor(js, bin_, s['A_log'])[:, 0].astype(float)
        self.B = accessor(js, bin_, s['B']).reshape(self.N, self.U).astype(float)
        self.C = accessor(js, bin_, s['C']).reshape(self.O, self.N).astype(float)
        self.Dm = accessor(js, bin_, s['D']).reshape(self.O, self.U).astype(float)
        self.u = np.zeros(self.U)
        self.reset()

    @classmethod
    def load(cls, path):
        return cls(*read_glb(path))

    # --- control ---------------------------------------------------------------------------------

    def control(self, body, arms=None, rate=None):
        """Inputs: body = {expert: weight} for every part, arms = the same for the arms only,
        rate = {expert: playback rate}. Input e*P + p is expert e's gate on part p; E*P + e its rate - 1."""
        u = np.zeros(self.U)
        for p in range(self.P):
            for name, w in ((arms if arms and self.parts[p] == 'arms' else body) or {}).items():
                u[self.experts.index(name) * self.P + p] = w
        for name, r in (rate or {}).items():
            u[self.E * self.P + self.experts.index(name)] = r - 1
        return u

    def reset(self):
        self.x = np.zeros(self.N)
        self.phase = np.zeros(self.E)

    def steady(self):
        """Jump the state to its steady state for the current input (no fade in)."""
        self.x = self.B @ self.u / np.exp(self.A_log)

    def trigger(self, e):
        """Start expert e's clip from the top."""
        self.phase[self.experts.index(e) if isinstance(e, str) else e] = 0

    # --- forward ---------------------------------------------------------------------------------

    def step(self, dt):
        """Advance dt seconds -> (quaternions [J, 4], hips position [3]) for the normalized rig."""
        A = -np.exp(self.A_log)
        Ad = np.exp(A * dt)
        self.x = Ad * self.x + (Ad - 1) / A * (self.B @ self.u)     # zero-order hold
        y = self.C @ self.x + self.Dm @ self.u
        g = np.maximum(y[:self.E * self.P], 0).reshape(self.E, self.P)
        tot = g.sum(0)
        self.gates = np.where(tot > 1e-6, g / np.where(tot > 1e-6, tot, 1), 0)
        self.rate = 1 + y[self.E * self.P:]
        self.phase = (self.phase + self.rate * dt / self.period) % 1.0
        return self.decode()

    def row(self, e):
        """Expert e's row at its phase: the triangle kernel over its keys (sign-aligned per bone)."""
        K, off, w = self.key_cnt[e], self.key_off[e], self.width
        x = self.phase[e] * K
        r, tot = np.zeros(self.D), 0.0
        for k in range(int(np.ceil(x - w)), int(np.floor(x + w)) + 1):
            a = 1 - abs(x - k) / w
            if a <= 0:
                continue
            v = self.W[off + k % K].copy()
            q, rq = v[:4 * self.J].reshape(-1, 4), r[:4 * self.J].reshape(-1, 4)
            q[(q * rq).sum(1) < 0] *= -1
            r += a * v
            tot += a
        return r / tot

    def decode(self):
        quats, hips, h_w = np.zeros((self.J, 4)), np.zeros(3), 0.0
        for e in range(self.E):
            if self.gates[e].max() <= 0:
                continue
            r = self.row(e)
            g = self.gates[e][self.part]
            q = r[:4 * self.J].reshape(-1, 4)
            g = np.where((quats * q).sum(1) < 0, -g, g)
            quats += g[:, None] * q
            gh = self.gates[e][self.part[self.hips]]
            hips += gh * r[4 * self.J:]
            h_w += gh
        n = np.linalg.norm(quats, axis=1, keepdims=True)
        quats = np.where(n > 1e-8, quats / np.where(n > 1e-8, n, 1), [0, 0, 0, 1])
        return quats, (hips if h_w > 1e-6 else self.rest_hips.copy())
