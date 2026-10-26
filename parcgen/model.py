"""parcgen: PARC's kinematic motion generator, ported to numpy (reference for parcgen.c).

PARC (Xu, Shi, Yin, Peng, SIGGRAPH 2025; github.com/mshoe/PARC, MIT) generates parkour one half
second at a time: given the last two frames, the terrain around the character and a direction to
go, a diffusion transformer denoises the next 15 frames. This file is that model and its sampling
loop, written from PARC's source (mdm.py, mdm_transformer.py, diffusion_util.py, gen_util.py,
kin_controller.py) with numpy only:

  frame features (91)  root position 3 | root exp map 3 | 14 body positions 42 | 28 joint dofs | 15 contacts,
                       standardized per frame index with the checkpoint's mean and std
  conditions           a 31 x 31 heightmap around the character (0.2 m cells, 2 m behind to 4 m
                       ahead, 3 m to each side) through a 4-layer CNN into 64 tokens; the target
                       direction through an MLP into 1 token; the diffusion step; a flag token
  denoiser             4 post-norm transformer encoder layers (d 256, 8 heads, GELU) over
                       [step, 64 terrain, target, flag, 15 frames] that predict the clean frames
  sampler              DDIM over 21 cosine-schedule steps at stride 10 (steps 20, 10, 0); after
                       each prediction the body positions are recomputed by forward kinematics
                       and contacts clamped (PARC's project_dofs); optional CFG against a pass
                       that cannot see the previous frames
  controller           every 5 frames: canonicalize the last two frames (origin under the root,
                       facing +x, heights relative to the root), generate, uncanonicalize, play
                       the next 4 new frames

Everything is in PARC's frame: z up, the character facing +x. Quaternions are (x, y, z, w).
"""
import json

import numpy as np

# PARC's humanoid (data/assets/humanoid.xml): bodies, parents, offsets from the parent.
BODIES = ['pelvis', 'torso', 'head', 'right_upper_arm', 'right_lower_arm', 'right_hand', 'left_upper_arm',
          'left_lower_arm', 'left_hand', 'right_thigh', 'right_shin', 'right_foot', 'left_thigh', 'left_shin', 'left_foot']
PARENT = [-1, 0, 1, 1, 3, 4, 1, 6, 7, 0, 9, 10, 0, 12, 13]
OFFSET = np.array([[0, 0, 0], [0, 0, .236151], [0, 0, .223894], [-.02405, -.18311, .2435], [0, -.274788, 0], [0, -.258947, 0],
                   [-.02405, .18311, .2435], [0, .274788, 0], [0, .258947, 0], [0, -.084887, 0], [0, 0, -.421546],
                   [0, 0, -.40987], [0, .084887, 0], [0, 0, -.421546], [0, 0, -.40987]])
# Joint dofs per body: 3 = ball joint (exp map), 1 = hinge about HINGE[body], 0 = fixed.
DOFS = [0, 3, 3, 3, 1, 0, 3, 1, 0, 3, 1, 3, 3, 1, 3]
HINGE = {4: np.array([0., 0, 1]), 7: np.array([0., 0, 1]), 10: np.array([0., 1, 0]), 13: np.array([0., 1, 0])}


# --- rotations (torch_util.py) -------------------------------------------------------------------

def qmul(a, b):
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def qrot(q, v):
    u, w = q[..., :3], q[..., 3:]
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def quat_to_axis_angle(q):
    q = np.where(q[..., 3:] < 0, -q, q)
    length = np.linalg.norm(q[..., :3], axis=-1)
    angle = 2 * np.arctan2(length, q[..., 3])
    axis = q[..., :3] / np.maximum(length, 1e-6)[..., None]
    ok = length > 1e-5
    return np.where(ok[..., None], axis, [0, 0, 1.]), np.where(ok, angle, 0.)


def axis_angle_to_quat(axis, angle):
    axis = axis / np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-9)
    q = np.concatenate([axis * np.sin(angle / 2)[..., None], np.cos(angle / 2)[..., None]], -1)
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-9)


def quat_to_exp_map(q):
    axis, angle = quat_to_axis_angle(q)
    return axis * angle[..., None]


def exp_map_to_quat(e):
    angle = np.linalg.norm(e, axis=-1)
    axis = e / np.where(angle > 0, angle, 1)[..., None]
    angle = np.arctan2(np.sin(angle), np.cos(angle))
    ok = np.abs(angle) > 1e-5
    return axis_angle_to_quat(np.where(ok[..., None], axis, [0, 0, 1.]), np.where(ok, angle, 0.))


def heading(q):
    d = qrot(q, np.broadcast_to([1., 0, 0], q.shape[:-1] + (3,)))
    return np.arctan2(d[..., 1], d[..., 0])


def heading_quat(h):
    return axis_angle_to_quat(np.broadcast_to([0, 0, 1.], np.shape(h) + (3,)), h)


# --- the character (kin_char_model.py) ------------------------------------------------------------

def dofs_to_quats(dof):
    """[..., 28] -> [..., 14, 4] local rotations of bodies 1..14."""
    out, c = [], 0
    for k in range(1, 15):
        n = DOFS[k]
        if n == 3:
            out.append(exp_map_to_quat(dof[..., c:c + 3]))
        elif n == 1:
            out.append(axis_angle_to_quat(np.broadcast_to(HINGE[k], dof.shape[:-1] + (3,)), dof[..., c]))
        else:
            out.append(np.broadcast_to([0, 0, 0, 1.], dof.shape[:-1] + (4,)))
        c += n
    return np.stack(out, -2)


def quats_to_dofs(q):
    """[..., 14, 4] -> [..., 28] (a hinge keeps the angle about its axis, signed by the axis)."""
    out = []
    for k in range(1, 15):
        if DOFS[k] == 3:
            out.append(quat_to_exp_map(q[..., k - 1, :]))
        elif DOFS[k] == 1:
            axis, angle = quat_to_axis_angle(q[..., k - 1, :])
            out.append(np.where((axis * HINGE[k]).sum(-1) < 0, -angle, angle)[..., None])
    return np.concatenate(out, -1)


def fk(root_pos, root_rot, joint_rot):
    """Body positions [..., 15, 3]."""
    rot, pos = [root_rot], [root_pos]
    for k in range(1, 15):
        p = PARENT[k]
        pos.append(pos[p] + qrot(rot[p], np.broadcast_to(OFFSET[k], root_pos.shape)))
        rot.append(qmul(rot[p], joint_rot[..., k - 1, :]))
    return np.stack(pos, -2)


# --- the network (mdm_transformer.py, cnn_31xy_4layer_c64_out64.py) -------------------------------

def linear(x, W, b):
    return x @ W.T + b


def silu(x):
    return x / (1 + np.exp(-x))


def gelu(x):
    from math import erf
    return 0.5 * x * (1 + np.vectorize(erf)(x / np.sqrt(2)))


def layer_norm(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def conv2d(x, W, b):
    """Valid convolution, stride 1: x [C, H, W], W [O, C, k, k] -> [O, H-k+1, W-k+1]."""
    C, H, Wd = x.shape
    O, _, k, _ = W.shape
    cols = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(1, 2))   # [C, H', W', k, k]
    return np.einsum('chwij,ocij->ohw', cols, W) + b[:, None, None]


class ParcGen:
    def __init__(self, path):
        z = np.load(path)
        self.w = {k: z[k].astype(np.float64) for k in z.files if k != 'config_json'}
        self.cfg = json.loads(bytes(z['config_json']).decode())
        self.T = self.cfg['diffusion_timesteps']
        self.L, self.H, self.S, self.P = self.cfg['num_layers'], self.cfg['num_heads'], self.cfg['seq_len'], self.cfg['num_prev_states']
        g = self.cfg['heightmap']['local_grid']
        self.dx, self.max_h = self.cfg['heightmap']['horizontal_scale'], self.cfg['heightmap']['max_h']
        self.grid = (g['num_x_neg'], g['num_x_pos'], g['num_y_neg'], g['num_y_pos'])
        # Cosine schedule (diffusion_util.cosine_beta_schedule, in float32 like PARC).
        x = np.linspace(0, self.T, self.T + 1, dtype=np.float32)
        acp = np.cos(((x / self.T) + 0.008) / 1.008 * np.float32(3.1415927410125732) * 0.5) ** 2
        acp = acp / acp[0]
        betas = np.clip(1 - acp[1:] / acp[:-1], 0.0001, 0.9999)
        self.acp = np.cumprod(1 - betas.astype(np.float64))

    def mlp(self, x, name, n):
        """PARC's MLP: Linear, (Dropout, SiLU, Linear) ...; layers at indices 0, 3, 6, ..."""
        for i in range(n):
            x = linear(x, self.w[f'{name}.model.{3 * i}.weight'], self.w[f'{name}.model.{3 * i}.bias'])
            if i < n - 1:
                x = silu(x)
        return x

    # Condition tokens: terrain (64), target (1), previous-state flag (1).
    def cond_tokens(self, hf_obs, target, use_prev):
        x = hf_obs[None]
        for i in (0, 2, 4, 6):
            x = conv2d(x, self.w[f'_obs_conv_net._net.{i}.weight'], self.w[f'_obs_conv_net._net.{i}.bias'])
            x = np.where(x > 0, x, 0.01 * x)                               # LeakyReLU
        C, Hh, Ww = x.shape                                                # 64 x 16 x 16 -> 2x2 patches
        t = x.reshape(C, Hh // 2, 2, Ww // 2, 2).transpose(1, 3, 0, 2, 4).reshape(-1, C * 4)
        obs = linear(t, self.w['_obs_embed.weight'], self.w['_obs_embed.bias'])
        tgt = self.mlp(np.asarray(target, float)[None], '_in_mlp_target', 2)
        flag = self.w['_embed_prev_state_noise_indicator.weight'][int(use_prev)][None]
        return np.concatenate([obs, tgt, flag])

    def denoise(self, x, t, cond, mask_prev=False):
        """One network pass: noisy frames x [15, 91] at step t -> predicted clean frames."""
        pe_t = self.w['_embed_t.sequence_pos_encoder.pe'][0, t]
        tok_t = linear(silu(linear(pe_t, self.w['_embed_t.time_embed.0.weight'], self.w['_embed_t.time_embed.0.bias'])),
                       self.w['_embed_t.time_embed.2.weight'], self.w['_embed_t.time_embed.2.bias'])
        frames = self.mlp(x, '_in_mlp_gen_seq', 2)
        h = np.concatenate([tok_t[None], cond, frames]) + self.w['_in_pos_encoding.pe'][0]
        n_cond = 1 + len(cond)
        keymask = np.zeros(len(h), bool)
        if mask_prev:
            keymask[n_cond:n_cond + self.P] = True
        for l in range(self.L):
            p = f'_transformer_encoder.layers.{l}.'
            qkv = linear(h, self.w[p + 'self_attn.in_proj_weight'], self.w[p + 'self_attn.in_proj_bias'])
            q, k, v = np.split(qkv, 3, -1)
            dh = q.shape[-1] // self.H
            q, k, v = (a.reshape(-1, self.H, dh).transpose(1, 0, 2) for a in (q, k, v))
            s = q @ k.transpose(0, 2, 1) / np.sqrt(dh)
            s[:, :, keymask] = -np.inf
            s = np.exp(s - s.max(-1, keepdims=True))
            a = (s / s.sum(-1, keepdims=True)) @ v
            a = linear(a.transpose(1, 0, 2).reshape(len(h), -1), self.w[p + 'self_attn.out_proj.weight'], self.w[p + 'self_attn.out_proj.bias'])
            h = layer_norm(h + a, self.w[p + 'norm1.weight'], self.w[p + 'norm1.bias'])
            f = linear(gelu(linear(h, self.w[p + 'linear1.weight'], self.w[p + 'linear1.bias'])), self.w[p + 'linear2.weight'], self.w[p + 'linear2.bias'])
            h = layer_norm(h + f, self.w[p + 'norm2.weight'], self.w[p + 'norm2.bias'])
        return self.mlp(h, '_out_mlp', 2)[n_cond:]

    # --- features ----------------------------------------------------------------------------------

    def assemble(self, root_pos, root_rot, joint_rot, contacts):
        """Frames (canonical) -> standardized features [n, 91]."""
        body = fk(root_pos, root_rot, joint_rot)
        f = np.concatenate([root_pos, quat_to_exp_map(root_rot), body[:, 1:].reshape(len(root_pos), -1),
                            quats_to_dofs(joint_rot), contacts], -1)
        n = len(f)
        return (f - self.w['features_mean'][:n]) / self.w['features_std'][:n]

    def extract(self, f):
        """Standardized features -> root_pos, root_rot, joint_rot, contacts."""
        f = f * self.w['features_std'][:len(f)] + self.w['features_mean'][:len(f)]
        return f[:, 0:3], exp_map_to_quat(f[:, 3:6]), dofs_to_quats(f[:, 48:76]), f[:, 76:91]

    def project(self, f):
        """PARC's project_dofs: body positions from forward kinematics, contacts clamped to [0, 1]."""
        rp, rr, jr, c = self.extract(f)
        return self.assemble(rp, rr, jr, np.clip(c, 0, 1))

    # --- sampling ----------------------------------------------------------------------------------

    def sample(self, prev, hf_obs, target, noise, stride=10, cfg_scale=None):
        """DDIM: prev [2, 91] standardized, hf_obs [31, 31] heights (relative), target [2],
        noise [15, 91] -> [15, 91] standardized features."""
        obs = np.clip(hf_obs, -self.max_h, self.max_h) / self.max_h
        cond = self.cond_tokens(obs, target, True)
        cond_free = self.cond_tokens(obs, target, False) if cfg_scale is not None else None
        x = noise.copy()
        for t in reversed(range(0, self.T, stride)):                      # PARC: stride 10 -> 20, 10, 0
            x[:self.P] = prev                                            # the previous frames, given clean
            x0 = self.denoise(x, t, cond)
            if cfg_scale is not None:
                x0 = x0 + cfg_scale * (self.denoise(x, t, cond_free, mask_prev=True) - x0)
            x0 = self.project(x0)
            if t == 0:
                return x0
            n = t - stride
            a_t, a_n = np.sqrt(self.acp[t]), np.sqrt(self.acp[n])
            s_t, s_n = np.sqrt(1 - self.acp[t]), np.sqrt(1 - self.acp[n])
            x = (a_n - a_t * s_n / s_t) * x0 + (s_n / s_t) * x


class Controller:
    """PARC's KinematicController: replan every 5 frames from the last two, on a heightfield."""

    def __init__(self, gen, frames, hf, min_point, cell, stride=10, cfg_scale=None, seed=0):
        self.gen, self.hf, self.min_point, self.cell = gen, np.asarray(hf, float), np.asarray(min_point, float), cell
        self.frames = frames            # dict: root_pos [n,3], root_rot [n,4], joint_rot [n,14,4], contacts [n,15]
        self.cur, self.stride, self.cfg_scale = 1, stride, cfg_scale
        self.rng = np.random.default_rng(seed)

    def heights(self, xy):
        i = np.clip(np.round((xy - self.min_point) / self.cell).astype(int), 0, np.array(self.hf.shape) - 1)
        return self.hf[i[..., 0], i[..., 1]]

    def generate(self, prev, target_dir_world, noise=None):
        """One window from two world frames -> 15 world frames."""
        g = self.gen
        cxy, cz = prev['root_pos'][1, :2], prev['root_pos'][1, 2]
        h = heading(prev['root_rot'][1])
        hq, hq_inv = heading_quat(h), heading_quat(-h)
        rp = prev['root_pos'].copy(); rp[:, :2] -= cxy
        rp = qrot(hq_inv, rp); rp[:, 2] -= cz
        rr = qmul(hq_inv, prev['root_rot'])
        nxn, nxp, nyn, nyp = g.grid
        gx, gy = np.meshgrid(np.linspace(-g.dx * nxn, g.dx * nxp, nxn + nxp + 1),
                             np.linspace(-g.dx * nyn, g.dx * nyp, nyn + nyp + 1), indexing='ij')
        c, s = np.cos(h), np.sin(h)
        world = np.stack([gx * c - gy * s, gx * s + gy * c], -1) + cxy
        obs = self.heights(world) - cz
        td = np.asarray(target_dir_world, float)
        target = np.array([td[0] * c + td[1] * s, -td[0] * s + td[1] * c])   # into the canonical frame
        prev_f = g.assemble(rp, rr, prev['joint_rot'], prev['contacts'])
        if noise is None:
            noise = self.rng.standard_normal((g.S, 91))
        out = g.sample(prev_f, obs, target, noise, self.stride, self.cfg_scale)
        orp, orr, ojr, oc = g.extract(out)
        orr = qmul(hq, orr)
        orp = qrot(hq, orp); orp[:, :2] += cxy; orp[:, 2] += cz
        return {'root_pos': orp, 'root_rot': orr, 'joint_rot': ojr, 'contacts': np.clip(oc, 0, 1)}

    def step(self, target_dir_world):
        """Advance one frame (30 fps) -> that frame (dict of single-frame arrays)."""
        self.cur += 1
        if self.cur > 5 or self.cur >= len(self.frames['root_pos']):
            start = min(4, len(self.frames['root_pos']) - 2)
            prev = {k: v[start:start + 2] for k, v in self.frames.items()}
            self.frames = self.generate(prev, target_dir_world)
            self.cur = 2
        return {k: v[self.cur] for k, v in self.frames.items()}
