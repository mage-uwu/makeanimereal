"""The tracking environment: PARC's humanoid in MuJoCo following PARC clips over their terrain.

    env = VecEnv(clips, n_envs=256, n_workers=4)
    obs = env.reset()                                  # [n, OBS]
    obs, reward, done, truncated = env.step(actions)   # actions [n, 28]: offsets to the reference pose

Each control step (30 Hz, 10 physics steps) the PD targets are the reference's next pose plus the
policy's offsets. Episodes start at a random frame of a random clip (reference state init) and end
when any body is more than 0.7 m from where the reference has it, or the clip runs out.

Observations (all in the character's heading frame, relative to its root):
  self       height over the ground 1, body positions 42, body orientations 90 (two axes each),
             root velocity 3, root spin 3, hinge velocities 28                               = 167
  target     the reference at t + 1 and t + 2: body positions 45 and orientations 90 each      = 270
  terrain    heights on a 9 x 9 grid, 0.4 m apart, from 1.2 m behind to 2 m ahead               =  81
Reward: 0.5 pose (joint rotations) + 0.05 hinge velocities + 0.15 hands and feet + 0.3 root.
"""
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import body  # noqa: E402

OBS = 167 + 270 + 81
ACT = 28
SUBSTEPS = 10
TERMINATE = 0.7
END_EFFECTORS = [5, 8, 11, 14]            # hands, feet
PARENT = [-1, 0, 1, 1, 3, 4, 1, 6, 7, 0, 9, 10, 0, 12, 13]
GRID = np.stack(np.meshgrid(np.linspace(-1.2, 2.0, 9), np.linspace(-1.6, 1.6, 9), indexing='ij'), -1).reshape(-1, 2)


# --- batched quaternion maths, (x, y, z, w) ---------------------------------------------------------

def qmul(a, b):
    return body._qmul(a, b)


def qconj(q):
    return q * np.array([-1, -1, -1, 1.0])


def qrot(q, v):
    u, w = q[..., :3], q[..., 3:]
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def heading(q):
    d = qrot(q, np.broadcast_to([1.0, 0, 0], q.shape[:-1] + (3,)))
    return np.arctan2(d[..., 1], d[..., 0])


def zrot(angle):
    a = np.asarray(angle)[..., None] / 2
    return np.concatenate([np.zeros(a.shape[:-1] + (2,)), np.sin(a), np.cos(a)], -1)


def tan_norm(q):
    sh = q.shape[:-1] + (3,)
    return np.concatenate([qrot(q, np.broadcast_to([1.0, 0, 0], sh)), qrot(q, np.broadcast_to([0, 0, 1.0], sh))], -1)


def angle(q):
    """Rotation angle of quaternions."""
    return 2 * np.arctan2(np.linalg.norm(q[..., :3], axis=-1), np.abs(q[..., 3]))


# --- observations and rewards, for a batch ---------------------------------------------------------

def observe(pos, rot, qvel, ref_pos, ref_rot, heights):
    """pos [n,15,3], rot [n,15,4] (world, x y z w), qvel [n,34], ref_pos/rot [n,2,15,...] for t+1, t+2,
    heights [n, 81] (ground under the grid points, world z) -> [n, OBS]."""
    root, rq = pos[:, 0], rot[:, 0]
    h_inv = zrot(-heading(rq))                                         # world -> heading frame
    hb = h_inv[:, None]
    rel = qrot(hb, pos - root[:, None])
    loc_rot = qmul(np.broadcast_to(hb, rot.shape), rot)
    vel = qrot(h_inv, qvel[:, 0:3])
    spin = qrot(h_inv, qrot(rq, qvel[:, 3:6]))                         # MuJoCo's root spin is in its own frame
    self_ = [(root[:, 2] - heights[:, 40])[:, None], rel[:, 1:].reshape(len(pos), -1),
             tan_norm(loc_rot).reshape(len(pos), -1), vel, spin, qvel[:, 6:] * 0.1]
    hb2 = h_inv[:, None, None]
    trel = qrot(hb2, ref_pos - root[:, None, None])
    trot = qmul(np.broadcast_to(hb2, ref_rot.shape), ref_rot)
    target = [trel.reshape(len(pos), -1), tan_norm(trot).reshape(len(pos), -1)]
    return np.concatenate(self_ + target + [heights - root[:, 2:3]], 1).astype(np.float32)


def reward(pos, rot, qvel, ref_pos, ref_rot, ref_qvel):
    """Tracking reward and per-env max body distance; refs at the frame just reached."""
    n = len(pos)
    par = np.array(PARENT[1:])
    local = qmul(qconj(rot[:, par]), rot[:, 1:])
    ref_local = qmul(qconj(ref_rot[:, par]), ref_rot[:, 1:])
    pose_err = (angle(qmul(qconj(ref_local), local)) ** 2).sum(1)
    vel_err = ((qvel[:, 6:] - ref_qvel[:, 6:]) ** 2).sum(1)
    ee = pos[:, END_EFFECTORS] - pos[:, :1]
    ref_ee = ref_pos[:, END_EFFECTORS] - ref_pos[:, :1]
    ee_err = ((ee - ref_ee) ** 2).sum((1, 2))
    root_err = ((pos[:, 0] - ref_pos[:, 0]) ** 2).sum(1) + 0.1 * angle(qmul(qconj(ref_rot[:, 0]), rot[:, 0])) ** 2
    r = 0.5 * np.exp(-2 * pose_err) + 0.05 * np.exp(-0.1 * vel_err) + 0.15 * np.exp(-40 * ee_err) + 0.3 * np.exp(-10 * root_err)
    dist = np.linalg.norm(pos - ref_pos, axis=-1).max(1)
    return r.astype(np.float32), dist


# --- a worker's share of the environments ------------------------------------------------------------

class Envs:
    def __init__(self, clips, n, seed):
        import mujoco
        self.mj = mujoco
        self.clips = clips
        self.models = [mujoco.MjModel.from_xml_string(body.build_xml(c.boxes)) for c in clips]
        m0 = self.models[0]
        self.body_ids = [mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, b) for b in body.BODIES]
        self.rng = np.random.default_rng(seed)
        self.n = n
        self.clip = np.zeros(n, int); self.t = np.zeros(n, int)
        self.data = [None] * n
        for i in range(n):
            self._reset(i)

    def _reset(self, i):
        c = int(self.rng.integers(len(self.clips)))
        clip = self.clips[c]
        self.clip[i], self.t[i] = c, int(self.rng.integers(0, clip.F - 3))
        d = self.mj.MjData(self.models[c])
        d.qpos[:] = clip.qpos[self.t[i]]
        d.qvel[:] = clip.qvel[self.t[i]]
        self.mj.mj_forward(self.models[c], d)
        self.data[i] = d

    def _gather(self):
        pos = np.stack([d.xpos[self.body_ids] for d in self.data])
        q = np.stack([d.xquat[self.body_ids] for d in self.data])
        rot = q[..., [1, 2, 3, 0]]
        qvel = np.stack([d.qvel for d in self.data])
        return pos, rot, qvel

    def _refs(self, offsets):
        """Reference body positions/rotations at t + offset (clamped to the clip) -> [n, len, 15, ...]."""
        P, R = [], []
        for i in range(self.n):
            clip = self.clips[self.clip[i]]
            f = np.minimum(self.t[i] + np.array(offsets), clip.F - 1)
            P.append(clip.body_pos[f]); R.append(clip.body_rot[f])
        return np.stack(P), np.stack(R)

    def _heights(self, pos, rot):
        h = heading(rot[:, 0])
        c, s = np.cos(h), np.sin(h)
        g = GRID[None]
        xy = np.stack([g[..., 0] * c[:, None] - g[..., 1] * s[:, None], g[..., 0] * s[:, None] + g[..., 1] * c[:, None]], -1) + pos[:, None, 0, :2]
        return np.stack([self.clips[self.clip[i]].ground(xy[i]) for i in range(self.n)])

    def observe(self):
        pos, rot, qvel = self._gather()
        rp, rr = self._refs([1, 2])
        return observe(pos, rot, qvel, rp, rr, self._heights(pos, rot))

    def step(self, actions):
        for i, d in enumerate(self.data):
            clip = self.clips[self.clip[i]]
            d.ctrl[:] = clip.qpos[self.t[i] + 1, 7:] + actions[i]
            self.mj.mj_step(self.models[self.clip[i]], d, nstep=SUBSTEPS)
        self.t += 1
        pos, rot, qvel = self._gather()
        rp, rr = self._refs([0])
        ref_qvel = np.stack([self.clips[self.clip[i]].qvel[self.t[i]] for i in range(self.n)])
        r, dist = reward(pos, rot, qvel, rp[:, 0], rr[:, 0], ref_qvel)
        bad = ~np.isfinite(pos).all((1, 2))
        done = (dist > TERMINATE) | bad
        trunc = np.array([self.t[i] >= self.clips[self.clip[i]].F - 3 for i in range(self.n)]) & ~done
        r[bad] = 0
        # The observation the value function bootstraps from at a truncation, before resetting.
        final = self.observe() if trunc.any() else None
        for i in np.nonzero(done | trunc)[0]:
            self._reset(i)
        return self.observe(), r, done, trunc, final


def _worker(conn, clips, n, seed):
    envs = Envs(clips, n, seed)
    while True:
        cmd, arg = conn.recv()
        if cmd == 'obs':
            conn.send(envs.observe())
        elif cmd == 'step':
            conn.send(envs.step(arg))
        elif cmd == 'close':
            return


class VecEnv:
    def __init__(self, clips, n_envs=256, n_workers=4, seed=0):
        self.per = n_envs // n_workers
        self.n = self.per * n_workers
        ctx = mp.get_context('fork')
        self.conns, self.procs = [], []
        for w in range(n_workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, clips, self.per, seed * 1000 + w), daemon=True)
            p.start()
            self.conns.append(a); self.procs.append(p)

    def reset(self):
        for c in self.conns:
            c.send(('obs', None))
        return np.concatenate([c.recv() for c in self.conns])

    def step(self, actions):
        for w, c in enumerate(self.conns):
            c.send(('step', actions[w * self.per:(w + 1) * self.per]))
        res = [c.recv() for c in self.conns]
        obs = np.concatenate([r[0] for r in res]); rew = np.concatenate([r[1] for r in res])
        done = np.concatenate([r[2] for r in res]); trunc = np.concatenate([r[3] for r in res])
        final = obs.copy()
        for w, r in enumerate(res):                     # truncated envs bootstrap from their last state
            if r[4] is not None:
                final[w * self.per:(w + 1) * self.per][r[3]] = r[4][r[3]]
        return obs, rew, done, trunc, final

    def close(self):
        for c in self.conns:
            c.send(('close', None))
