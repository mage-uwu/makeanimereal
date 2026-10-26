"""Train the motion tracker with PPO.

    python3 physics/train.py dec_release.zip --clips 'rerun_april272025_iter_5*_opt_dm.pkl' --limit 64 \
        --out out/tracker [--resume] [--hours 6]

Each iteration: 32 control steps from every environment, advantages by GAE (gamma 0.95, lambda
0.95), 3 epochs of PPO in minibatches of 4096. Logs a line per iteration to <out>/log.tsv and saves <out>/agent.npz every
10 iterations (resume picks it up).
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from env import ACT, OBS, VecEnv  # noqa: E402
from motions import load_clips  # noqa: E402
from ppo import Agent  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('zip')
ap.add_argument('--clips', action='append', required=True)
ap.add_argument('--limit', type=int, default=64)
ap.add_argument('--out', default='out/tracker')
ap.add_argument('--envs', type=int, default=256)
ap.add_argument('--workers', type=int, default=4)
ap.add_argument('--horizon', type=int, default=32)
ap.add_argument('--lr', type=float, default=2e-5)
ap.add_argument('--hours', type=float, default=1.0)
ap.add_argument('--resume', action='store_true')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
clips = load_clips(a.zip, a.clips, a.limit)
with open(os.path.join(a.out, 'clips.txt'), 'w') as f:
    f.write('\n'.join(c.name for c in clips))
print(f'{len(clips)} clips, {sum(c.F for c in clips) / 30 / 60:.1f} minutes of motion', flush=True)
env = VecEnv(clips, a.envs, a.workers)
path = os.path.join(a.out, 'agent.npz')
agent = Agent.load(path, lr=a.lr) if a.resume and os.path.exists(path) else Agent(OBS, ACT, lr=a.lr)
rng = np.random.default_rng(int(time.time()))
log = open(os.path.join(a.out, 'log.tsv'), 'a')
if os.path.getsize(log.name) == 0:
    log.write('iter\tsamples\thours\treward\tep_len\tclip_frac\tvalue_loss\tsamples_per_s\n')
obs = env.reset()
gamma, lam = 0.95, 0.95
samples, it, t_start = 0, 0, time.time()
ep_len, ep_lens = np.zeros(env.n), []
while time.time() - t_start < a.hours * 3600:
    t0 = time.time()
    H, N = a.horizon, env.n
    B_obs = np.zeros((H, N, OBS), np.float32); B_act = np.zeros((H, N, ACT), np.float32)
    B_logp = np.zeros((H, N), np.float32); B_val = np.zeros((H + 1, N), np.float32)
    B_rew = np.zeros((H, N), np.float32); B_done = np.zeros((H, N), bool); B_boot = np.zeros((H, N), np.float32)
    for h in range(H):
        agent.norm.update(obs)
        act, logp, val = agent.sample(obs, rng)
        nobs, rew, done, trunc, final = env.step(act)
        B_obs[h], B_act[h], B_logp[h], B_val[h], B_rew[h] = obs, act, logp, val, rew
        B_done[h] = done | trunc
        # A truncated episode is not a failure: bootstrap from the value of where it stopped.
        if trunc.any():
            B_boot[h, trunc] = agent.value(final[trunc])
        ep_len += 1
        for i in np.nonzero(done | trunc)[0]:
            ep_lens.append(ep_len[i]); ep_len[i] = 0
        obs = nobs
    B_val[H] = agent.value(obs)
    adv = np.zeros((H, N), np.float32)
    last = np.zeros(N, np.float32)
    for h in reversed(range(H)):
        nxt = np.where(B_done[h], B_boot[h], B_val[h + 1])
        delta = B_rew[h] + gamma * nxt - B_val[h]
        last = delta + gamma * lam * np.where(B_done[h], 0, last)
        adv[h] = last
    ret = adv + B_val[:H]
    stats = agent.update(B_obs.reshape(-1, OBS), B_act.reshape(-1, ACT), B_logp.ravel(), adv.ravel(), ret.ravel(), rng=rng)
    samples += H * N
    it += 1
    sps = H * N / (time.time() - t0)
    el = np.mean(ep_lens[-200:]) if ep_lens else 0
    line = f'{it}\t{samples}\t{(time.time() - t_start) / 3600:.3f}\t{B_rew.mean():.4f}\t{el:.1f}\t{stats["clip_frac"]:.3f}\t{stats["value_loss"]:.4f}\t{sps:.0f}'
    log.write(line + '\n'); log.flush()
    if it % 5 == 0:
        print(line, flush=True)
    if it % 10 == 0:
        agent.save(path)
agent.save(path)
env.close()
print('done', flush=True)
