"""PPO in numpy: MLPs with hand-written backprop, AdamW, running observation normalization.

    agent = Agent(obs_dim, act_dim)
    mean = agent.act(obs)                       # deterministic action (the policy's mean)
    agent.update(batch)                         # one PPO round over a rollout
    agent.save(path); Agent.load(path)

Policy: a Gaussian with a learned mean (MLP, ReLU; 512-256 by default) and a fixed std; value: its own MLP.
"""
import numpy as np


class MLP:
    def __init__(self, sizes, rng, out_scale=1.0):
        self.W, self.b = [], []
        for i, (a, c) in enumerate(zip(sizes[:-1], sizes[1:])):
            s = np.sqrt(2 / a) * (out_scale if i == len(sizes) - 2 else 1)
            self.W.append((rng.standard_normal((a, c)) * s).astype(np.float32))
            self.b.append(np.zeros(c, np.float32))

    def params(self):
        return self.W + self.b

    def forward(self, x):
        self.cache = []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = x @ W + b
            self.cache.append((x, z))
            x = z if i == len(self.W) - 1 else np.maximum(z, 0)                                   # ReLU
        return x

    def backward(self, g):
        """Gradient of the loss wrt the last forward's output -> parameter gradients (W..., b...)."""
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            x, z = self.cache[i]
            if i < len(self.W) - 1:
                g = g * (z > 0)
            gW[i], gb[i] = x.T @ g, g.sum(0)
            g = g @ self.W[i].T
        return gW + gb


class AdamW:
    def __init__(self, params, lr, wd=0.01, betas=(0.9, 0.999), eps=1e-8):
        self.p, self.lr, self.wd, self.b1, self.b2, self.eps = params, lr, wd, betas[0], betas[1], eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads, max_norm=1.0):
        norm = np.sqrt(sum(float((g * g).sum()) for g in grads))
        scale = min(1.0, max_norm / (norm + 1e-6))
        self.t += 1
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            g = g * scale
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            mh, vh = m / (1 - self.b1 ** self.t), v / (1 - self.b2 ** self.t)
            if p.ndim > 1:
                p *= 1 - self.lr * self.wd               # decoupled weight decay, matrices only
            p -= (self.lr * mh / (np.sqrt(vh) + self.eps)).astype(p.dtype)
        return norm


class Normalizer:
    def __init__(self, dim):
        self.n, self.mean, self.var = 1e-4, np.zeros(dim), np.ones(dim)

    def update(self, x):
        n, mean, var = len(x), x.mean(0), x.var(0)
        d = mean - self.mean
        tot = self.n + n
        self.mean = self.mean + d * n / tot
        self.var = (self.var * self.n + var * n + d * d * self.n * n / tot) / tot
        self.n = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -5, 5).astype(np.float32)


class Agent:
    def __init__(self, obs_dim, act_dim, std=0.1, lr=2e-5, hidden=(512, 256), seed=0):
        rng = np.random.default_rng(seed)
        self.pi = MLP([obs_dim, *hidden, act_dim], rng, out_scale=0.01)
        self.vf = MLP([obs_dim, *hidden, 1], rng)
        self.std = np.full(act_dim, std, np.float32)
        self.norm = Normalizer(obs_dim)
        self.opt_pi = AdamW(self.pi.params(), lr)
        self.opt_vf = AdamW(self.vf.params(), lr * 2)

    def act(self, obs):
        return self.pi.forward(self.norm(obs))

    def sample(self, obs, rng):
        mean = self.act(obs)
        a = mean + rng.standard_normal(mean.shape).astype(np.float32) * self.std
        return a, self.logp(a, mean), self.vf.forward(self.norm(obs))[:, 0]

    def logp(self, a, mean):
        return (-0.5 * (((a - mean) / self.std) ** 2).sum(1)).astype(np.float32)

    def value(self, obs):
        return self.vf.forward(self.norm(obs))[:, 0]

    def update(self, obs, act, logp_old, adv, ret, epochs=3, minibatch=4096, clip=0.2, rng=None):
        rng = rng or np.random.default_rng()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        x = self.norm(obs)
        n = len(x)
        stats = []
        for _ in range(epochs):
            for idx in np.array_split(rng.permutation(n), max(1, n // minibatch)):
                xb, ab, lb, advb, rb = x[idx], act[idx], logp_old[idx], adv[idx], ret[idx]
                mean = self.pi.forward(xb)
                ratio = np.exp(self.logp(ab, mean) - lb)
                unclipped = ratio * advb
                use = (unclipped <= np.clip(ratio, 1 - clip, 1 + clip) * advb) | ((ratio > 1 - clip) & (ratio < 1 + clip))
                # d(-mean(min(...)))/d logp = -adv * ratio where the unclipped term is the active one.
                dlogp = np.where(use, -advb * ratio, 0) / len(idx)
                dmean = dlogp[:, None] * (ab - mean) / self.std ** 2
                # Keep the mean inside +-1 rad of the reference.
                dmean = dmean + 0.1 * np.where(np.abs(mean) > 1, np.sign(mean) * (np.abs(mean) - 1), 0) / len(idx)
                gn = self.opt_pi.step(self.pi.backward(dmean.astype(np.float32)))
                v = self.vf.forward(xb)[:, 0]
                self.opt_vf.step(self.vf.backward(((v - rb) / len(idx))[:, None].astype(np.float32)))
                stats.append((float(np.mean(np.abs(ratio - 1) > clip)), float(((v - rb) ** 2).mean()), gn))
        s = np.array(stats).mean(0)
        return {'clip_frac': s[0], 'value_loss': s[1], 'grad_norm': s[2]}

    def save(self, path):
        arrays = {f'pi_W{i}': w for i, w in enumerate(self.pi.W)} | {f'pi_b{i}': b for i, b in enumerate(self.pi.b)}
        arrays |= {f'vf_W{i}': w for i, w in enumerate(self.vf.W)} | {f'vf_b{i}': b for i, b in enumerate(self.vf.b)}
        np.savez(path, std=self.std, norm_n=self.norm.n, norm_mean=self.norm.mean, norm_var=self.norm.var, **arrays)

    @classmethod
    def load(cls, path, **kw):
        z = np.load(path)
        a = cls(z['pi_W0'].shape[0], z['pi_W2'].shape[1], hidden=(z['pi_W0'].shape[1], z['pi_W1'].shape[1]), **kw)
        for i in range(3):
            a.pi.W[i][:] = z[f'pi_W{i}']; a.pi.b[i][:] = z[f'pi_b{i}']
            a.vf.W[i][:] = z[f'vf_W{i}']; a.vf.b[i][:] = z[f'vf_b{i}']
        a.std[:] = z['std']; a.norm.n, a.norm.mean, a.norm.var = float(z['norm_n']), z['norm_mean'], z['norm_var']
        return a
