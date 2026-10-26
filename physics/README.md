# physics

A physics tracker for the avatar: PARC's humanoid simulated in MuJoCo, driven by a PPO policy that
follows reference motion, trained here on CPU and run in the browser on MuJoCo's official WASM
build (the same engine version, 3.17.0). In the stage's Free run, with **Physics** on, PARC's motion
generator (parcgen) plans and the simulated body follows: she is drawn from the simulation.

| file | what |
|---|---|
| `humanoid_parc.xml` | PARC's humanoid (MimicKit's DeepMimic humanoid), as released |
| `body.py` | the MuJoCo model built from it, and PARC frames <-> MuJoCo states |
| `motions.py` | reference clips from PARC's dataset, precomputed, with their terrain as boxes |
| `env.py` | the tracking environment, vectorized over worker processes |
| `ppo.py` | PPO in numpy: MLPs with hand-written backprop, AdamW, observation normalization |
| `train.py` | the training loop (log.tsv, agent.npz, resumable) |
| `export.py` | model XML and policy weights for the browser (`web/physics/`) |
| `test_browser.py`, `test_browser.mjs` | web/tracker.js on WASM against env.py in Python, same policy and clip |
| `simulate_run.mjs` | physics free run offline in Node: falls per minute, progress round the lap |
| `../web/tracker.js` | the tracker in the browser |
| `../web/mujoco/` | MuJoCo 3.17.0's WASM build |

## The body

MuJoCo reads PARC's Isaac Gym XML differently, so `body.build_xml` makes it behave like PARC's sim:
joint stiffness and damping become PD gains (position actuators, torque limited to each motor's
gear), contacts get friction (`condim 3`), terrain is static boxes, and ball joints lose their
per-axis Euler limits (PARC's data exceeds them at the hips; Isaac Gym limits a ball joint as a whole).
The waist and hips turn about y, then x, then z, so their Euler angles lock only at 90 deg of
sideways lean rather than at 90 deg of flexion, which climbing reaches.

Checked: MuJoCo's kinematics from converted states equal PARC's forward kinematics to 0.0006 mm
over whole clips; frames -> MuJoCo -> frames round-trips to the same; MuJoCo's WASM build matches
Python MuJoCo to 5e-10 after 2 s of random PD targets.

## Training

Each environment follows one PARC clip over its terrain, from a random frame of it (reference state
init). Every 30 Hz step the PD targets are the reference's next pose plus the policy's offsets; 10
MuJoCo steps (300 Hz) follow. The episode ends when any body is more than 0.7 m from where the
reference has it, or the clip ends (bootstrapped). Observations (518) are in the character's heading
frame: its body positions, orientations and velocities, the reference's next two frames, and a 9 x 9
heightmap; the reward weighs joint rotations 0.5, joint velocities 0.05, hands and feet 0.15, root
0.3 (DeepMimic's terms). PPO: 256 environments x 32 steps per iteration, GAE 0.95/0.95, 3 epochs,
minibatches of 4096, AdamW at 2e-5, policy and value MLPs 512-256 (ReLU), action std 0.1 rad.
