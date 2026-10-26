"""Record a deterministic tracker rollout in Python, for web/tracker.js to reproduce (test_browser.mjs).

    python3 physics/test_browser.py agent.npz dec_release.zip clip_glob out_dir
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from env import Envs  # noqa: E402
from motions import load_clips  # noqa: E402
from ppo import Agent  # noqa: E402

agent_path, zip_path, pattern, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
clip = load_clips(zip_path, [pattern], 1)[0]
agent = Agent.load(agent_path)
e = Envs([clip], 1, 0)
t0 = 10
e.clip[0], e.t[0] = 0, t0
d = e.data[0] = e.mj.MjData(e.models[0])
# Start like the browser does: in frame t0, moving toward t0 + 1.
d.qpos[:] = clip.qpos[t0]; d.qvel[:] = clip.qvel[t0]
e.mj.mj_forward(e.models[0], d)
obs0, qpos, acts = None, [], []
for k in range(45):
    o = e.observe()
    if obs0 is None:
        obs0 = o[0]
    a = agent.act(o)
    acts.append(a[0].tolist())
    _, r, done, trunc, _ = e.step(a)
    qpos.append(d.qpos.tolist())
    if done[0] or trunc[0]:
        break
subprocess.run([sys.executable, os.path.join(HERE, 'export.py'), agent_path, out], check=True)
json.dump({'frames': clip.frames.tolist(), 't0': t0, 'boxes': clip.boxes, 'hf': clip.hf.tolist(), 'min_point': clip.min_point.tolist(),
           'dx': clip.dx, 'obs0': obs0.tolist(), 'acts': acts, 'qpos': qpos}, open(os.path.join(out, 'rollout.json'), 'w'))
print(f'{len(qpos)} steps recorded')
