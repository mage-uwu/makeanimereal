# parcgen

PARC's kinematic motion generator ([PARC](https://github.com/mshoe/PARC), Xu, Shi, Yin, Peng,
SIGGRAPH 2025), ported from PyTorch to numpy and to C/WASM, driving the avatar live in the stage's
**Free run** mode: WASD (or an auto-run lap) over a parkour course, every half second of motion
generated from the terrain around her.

| file | what |
|---|---|
| `model.py` | the reference port in numpy: the network, PARC's features, DDIM sampler and replanning controller |
| `parcgen.c` | the same in C, no libc: SIMD matrix kernels, its own sin/cos/atan2/erf; `build-wasm.sh` -> `web/parcgen.wasm` (30 KB) |
| `export_weights.py` | checkpoint -> `parc_mdm.npz` (numpy) and `web/parcgen.bin` (float32 blob in the C runtime's order) |
| `test.py` | C against numpy on PARC terrain |
| `check_behavior.py` | the port against PARC's own data |
| `simulate.mjs` | the free-run lap offline in Node: where she goes, what she climbs, ms per window |
| `../web/freerun.js`, `../web/parcgen-worker.js` | the course, scheduling, retargeting onto the VRM; the WASM in a worker |

## The model

2.2M parameters. Every 5 frames (30 fps) it takes the last two frames, the terrain around the
character (31 x 31 heights, 0.2 m cells, 2 m behind to 4 m ahead) and a direction to go, and
denoises the next 15 frames (0.5 s): 4 post-norm transformer layers (d 256, 8 heads) over
[diffusion step, 64 terrain tokens from a 4-layer CNN, target, flag, 15 frames]; each frame is 91
features (root position and exp map, 14 body positions, 28 joint dofs, 15 contacts). After every
prediction the body positions are rebuilt by forward kinematics and contacts clamped.

Free run samples with DDIM at stride 20 (steps 20, 0) and CFG 0.65 against a pass that cannot see
the previous frames (PARC's default). Without CFG the motion drifts into a crouch within ~15 s; with
it she laps the course indefinitely (`simulate.mjs`: vaults, a rooftop gap, a 2.2 m climb, stairs).
A window costs ~80 ms in WASM (Node, this machine), inside the 133 ms of motion it covers; slower
machines hand over later rather than stall.

## How it was checked

PARC's own PyTorch code was not run here. Instead:

- **Network:** `model.py` against stock `torch.nn` layers (TransformerEncoder, Conv2d, Unfold) loaded
  with the same weights: equal to 3e-15 (float64).
- **Pipeline** (canonicalization, features, heightmap, DDIM): from real frames on real terrain the
  generator's root lands 0.10 m (median) from where the data went 0.4 s later (constant velocity:
  0.20 m; standing still: 0.62 m), covering 1.02x the real distance; feet stay on the surface.
- **C = numpy:** 3e-7 m root position, 2e-5 deg joint rotation (float32), at stride 10 and 20, with and
  without CFG.

```sh
# weights (needs PyTorch; loads with weights_only=True, i.e. tensors only)
curl -LO https://huggingface.co/datasets/mxucg/PARC/resolve/main/smaller_model.zip && unzip smaller_model.zip
python3 parcgen/export_weights.py model_40000.ckpt parc_mdm.npz && mv parc_mdm.bin web/parcgen.bin
parcgen/build-wasm.sh
python3 parcgen/test.py parc_mdm.npz dec_release.zip
node parcgen/simulate.mjs 60 0.65 20
```

The weights come from PARC's Hugging Face release (MIT; notice in `web/parkour/LICENSE-PARC.md`).
