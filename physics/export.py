"""Write what the browser needs to run the tracker.

    python3 physics/export.py out/tracker/agent.npz web/physics

  humanoid.xml   the MuJoCo model (body.build_xml) without terrain: the page adds a floor and boxes
                 before </worldbody>
  tracker.bin    float32: observation mean [OBS], std [OBS], then each policy layer's W [in][out], b [out]
  tracker.json   layer sizes, observation size, the action std (unused at run time), and env constants
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import body  # noqa: E402
from env import ACT, OBS, SUBSTEPS, TERMINATE  # noqa: E402
from ppo import Agent  # noqa: E402

agent_path, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
open(os.path.join(out, 'humanoid.xml'), 'w').write(body.build_xml(floor=False))
a = Agent.load(agent_path)
parts = [a.norm.mean.astype('<f4'), np.sqrt(a.norm.var + 1e-8).astype('<f4')]
for W, b in zip(a.pi.W, a.pi.b):
    parts += [W.astype('<f4').ravel(), b.astype('<f4')]
np.concatenate(parts).tofile(os.path.join(out, 'tracker.bin'))
json.dump({'obs': OBS, 'act': ACT, 'layers': [list(W.shape) for W in a.pi.W], 'substeps': SUBSTEPS, 'terminate': TERMINATE,
           'hinges': [j for b in body.BODIES[1:] for j in body.JOINTS.get(b, [])]},
          open(os.path.join(out, 'tracker.json'), 'w'), indent=1)
print(f'{out}: humanoid.xml, tracker.bin ({sum(p.size for p in parts)} floats), tracker.json')
