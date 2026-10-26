# mocapact

Microsoft's [MoCapAct](https://github.com/microsoft/MoCapAct) multi-clip policy, pulled and run in
the browser: a pretrained physics controller for dm_control's CMU humanoid that tracks motion
capture. Its weights are CDLA-Permissive-2.0 (commercial use allowed; ship the license text with
them, `web/npmp/LICENSE.md`), trained on CMU mocap (free for commercial use). In the stage, the
**Physics** row runs it on MuJoCo's WASM build and poses the box character or the VRoid avatar from
the simulated body.

| file | what |
|---|---|
| `ckpt.py` | reads MoCapAct's Lightning checkpoint without running its pickled code |
| `export.py` | the policy as numpy arrays (`.npz`) |
| `npmp.py` | the policy in numpy (matches torch's layers to 1e-15) |
| `track_dm.py` | runs it in dm_control's own tracking task, as MoCapAct evaluates it |
| `export_web.py` | `web/npmp/`: the model XML, observable index, float16 weights, clips, and a test fixture |
| `npmp.c`, `build-wasm.sh` | the network's layers in WebAssembly SIMD: `web/npmp.wasm` |
| `test_browser.mjs` | `web/npmp.js` against a dm_control episode, step by step |
| `simulate.mjs` | every exported clip through `web/npmp.js` in Node: steps tracked, time per step |
| `../web/npmp.js` | the policy and simulation in the browser |
| `../web/simrun.js` | the stage's driver: simulation at 33 Hz, retargeted onto the character |

## The policy

An NPMP ("neural probabilistic motor primitives", Merel et al. 2019): a recurrent encoder reads the
body's state and the next five reference frames (the 31 tracked bodies' positions in the root's
frame and rotations relative to the body's) and outputs a 60-d motor intention; a decoder turns the
intention and the body's state (joint angles and velocities, root velocimeter and gyro, end
effectors, the up axis, actuator activations, touch and torque sensors) into 56 position targets.
4 x 1024 LayerNorm/ELU layers, 5.0M parameters. Every 0.03 s: one evaluation, then six 5 ms MuJoCo
steps.

## Checked

- `npmp.py` against torch's own layers: 5e-15.
- In dm_control 1.0.50 / MuJoCo 3.17.0, from each clip's start: 24 of dm_control's locomotion
  clips tracked on average 0.77 of their length, half to the end (the full-dataset and
  locomotion-dataset policies alike); long clips drift off after 3-6 s, as MoCapAct reports for
  this policy.
- `web/npmp.js` on MuJoCo's WASM build against dm_control, over 40 steps of a walk: observations
  to 5e-3 and actions to 1e-3 (float16 weights; float32 gives 2e-4 and 5e-5), and a free rollout
  stays within 4e-4 of dm_control's. dm_control steps with `mj_step2` then `mj_step1` and reads its
  sensors after a forward pass; anything else changes the contact and torque sensors the policy
  reads, and it drifts.
- `simulate.mjs`: the eight exported clips track as in dm_control (six to the end; Run and Hop
  forward stray near the end), 3.8 ms per 30 ms control step in Node.

## Rebuild

```
python3.12 -m venv venv && venv/bin/pip install mujoco==3.17.0 'dm_control[hdf5]==1.0.50' 'protobuf<6' torch
curl -LO https://huggingface.co/microsoft/mocapact-models/resolve/main/multiclip_policy.tar.gz && tar xzf multiclip_policy.tar.gz
venv/bin/python mocapact/export_web.py multiclip_policy/full_dataset/model/model.ckpt web/npmp --fixture fixture.json
mocapact/build-wasm.sh
node mocapact/test_browser.mjs fixture.json && node mocapact/simulate.mjs
```

dm_control downloads its CMU data (`cmu_2020_dfe3e9e0.h5`, 477 MB) on first use.
