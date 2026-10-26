# vrmanet

A motion network whose weights are a VRM Animation file. `web/moves/vrmanet.vrma` is the model:
any VRMA player plays it as 11 ordinary mocap animations, and vrmanet reads the same bytes as a
decoder, an attention memory and a state-space controller.

| file | what |
|---|---|
| `model.py` | the reference model (numpy): file loader and `step()` |
| `build.py` | ingest `.vrma` clips as experts, write the model file with the controller's initial weights |
| `vrmanet.c` | `step()` in C: no libc, no parser, one static arena; native or WASM |
| `build-wasm.sh` | `clang --target=wasm32` build to `web/vrmanet.wasm` (7 KB) |
| `test.py` | untrained model vs the clips, C vs numpy |
| `../web/vrmanet.js` | the loader in JS + WASM glue; `web/stage.html` drives the avatar with it |

## The file is the model

```
glTF animation e  (one per ingested clip)      ->  expert e
  its keyframe k: J bone quaternions + hips xyz  ->  decoder row W[e, k]   (D = 4J + 3 = 211)
  its keyframe times                             ->  attention keys (free: already in the file)
extensions.MAKEANIMEREAL_vrmanet
  experts, parts, boneParts, kernelWidth         ->  structure (JSON, editable)
  ssm.A_log, ssm.B, ssm.C, ssm.D                 ->  controller weights (float32 accessors)
```

Nothing is converted: `build.py` copies each clip's accessor bytes unchanged, so editing a track in
Blender (or any glTF tool) edits that expert's weights. Held tracks (the fingers' fixed curl) are
one key in the file and broadcast to every row on load.

## Forward pass, per frame (`step(dt)`)

```
controller   x  <- exp(A dt) x + (exp(A dt) - 1) / A * B u        A = -exp(A_log), diagonal (ZOH)
             y   = C x + D u
gates        g[e, p] = relu(y[e*P + p]) / sum_e relu(y[e*P + p])  per body part p
clocks       phase[e] <- frac(phase[e] + (1 + y[E*P + e]) dt / T[e])
attention    expert e's row = sum_k a_k W[e, k],  a_k ∝ max(0, 1 - |phase[e] K_e - k| / width)
mixture      bone j: q_j = normalize( sum_e g[e, part(j)] * row_e[j] )   (sign-aligned)
             hips  : sum_e g[e, part(hips)] * row_e[hips]
```

`u` is the control input: `u[e*P + p]` asks for expert `e` on body part `p` (`lower`, `torso`,
`arms`), and `u[E*P + e]` for expert `e`'s playback rate minus 1. The attention is sparse: with
`width = 1` it touches two rows per active expert, exactly glTF LINEAR sampling (as nlerp).

## Cold start

`build.py` sets one state per output with `A = -1/tau`, `B = 1/tau`, `C = I`, `D = 0` (tau = 0.3 s):
the controller follows its input with a 0.3 s lag. Asking for one expert therefore replays its clip;
asking for another crossfades; asking for wave on the arms and walk elsewhere mixes them per part.
`test.py` checks the untrained model against glTF playback of every clip (worst bone error 0.005°,
the nlerp/slerp difference between 30 fps keys) and the C runtime against numpy over a control
script with switches, an arms mix, a rate change and retriggers (worst 0.0002°).

What is trainable is everything in the extension (`A_log`, `B`, `C`, `D`, `kernelWidth`) and, if you
want, the decoder rows themselves; nothing is trained yet.

```sh
python3 vrmanet/build.py web/moves/{wave,bow,stretch,walk,shywalk,coolwalk,moonwalk,twist,charleston,dance,jacks}.vrma --out web/moves/vrmanet.vrma
vrmanet/build-wasm.sh
python3 vrmanet/test.py
```
