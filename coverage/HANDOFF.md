# Handoff: VRoid → VACE film → neural texture + FFF shader

For the next session picking this up. This is a working, tested pipeline that turns one VRM
character into a **real-time anime-shaded character** in the browser: about 2 hours and ~$4 of GPU
time per character, about 22 fps on a phone at the standard size. Step-by-step commands are in
`coverage/asset/README.md`. This file covers what the pipeline is, what it achieved, what we learned,
and what is still open.

## What it is

A **per-character neural texture + tiny screen-space network**. three.js renders the VRM's G-buffer
(colour, normal, depth, UV, part ID). Each pixel looks up a learned 8-channel texture for its mesh
part. Then 4–8 small blocks turn the result into the character as an anime film drew her. Training
targets come from a video model (Wan2.1 VACE) that restyles an orbit of renders, guided by a few
image-editor keyframes.

It is good at:
- a clean, stable, anime-coloured skin;
- real-time on WebGL2 or WebGPU, phones included;
- any pose or motion clip, since it runs on the live G-buffer.

It is **not** a style-transfer shader. It does not redraw shapes (eyes, hair silhouettes, line
cheats). It recolours the geometry it is given. See "Limits" and "Open experiments".

## Results

| Character | Network | Held-out L1 | Flicker | Texture noise | Notes |
|---|---|---|---|---|---|
| Magical Girl (`#` default) | 8 blocks, 360×640 | – | – | – | trained earlier with the C trainer; ships as `nts-tree/` |
| Magical Girl HD (`#girlhd`) | 8 blocks, 720×1280 | 0.0142 | 0.0131 | 0.072 | 45 min from a warm start; replaced by ↓ |
| Magical Girl HD (shipped) | **4 blocks, dil 1,3,9,27**, 720×1280 | 0.0142 | 0.0130 | 0.069 | 60 min from scratch; ~55–60% of the 8-block GPU time |
| Bunny (`#bunny`) | 8 blocks, 360×640 | 0.0139* | 0.0083 | 0.086 | denoised + flattened lighting, ear backs retouched |

\* Against the unflattened film; it rises on purpose when lighting is flattened. Flicker is the colour
change of the same surface point between consecutive frames: the film itself scores 0.0200 and the
raw render 0.0049.

Speed: the standard character runs at ~22 fps on an iPhone with the GL engine. HD is about 2× the
standard character's cost with the 4-block network.

## The pipeline (details and commands: `coverage/asset/README.md`)

| # | Step | Where | Time | Tool |
|---|---|---|---|---|
| 1 | Orbit renders, 1440 frames + G-buffer (top 960, bottom 480) | local | ~80 min | `coverage/render-orbit.mjs` |
| 2 | Pod with RTX PRO 6000 96 GB ($2.49/h), weights | pod | 2–15 min | `coverage/pod/boot-asset.sh`, `server.py`, `pod.sh` |
| 3 | Style reference + anchor keyframes (every 80 frames) | pod | ~6–10 s each | `coverage/pod/edit_images.py` (Qwen-Image-Edit-2509 + Lightning 8-step) |
| 4 | Film: 81-frame chunks between anchors, from rendered depth | pod | ~45 s a chunk | `coverage/pod/vace_film.py` (Wan2.1 VACE 14B + lightx2v) |
| 5 | Dataset per frame: G-buffer + film target | local | minutes | `coverage/shader/build_dataset.py`, `nts/pack_data.py` |
| 6 | Flatten the film's lighting (optional, recommended) | pod | ~20 frames/min on CPU, fast on CUDA | `nts/flatten_shading.py` |
| 7 | Train (denoising objective, warm start) | pod | 30–60 min | `nts/denoise_torch.py` (`shader_torch.py` model) |
| 8 | Retouch film mistakes (optional) | local CPU | ~20 min | `nts/retouch_part.py` |
| 9 | Export + add to the viewer | local | seconds | `nts/export_nts.py`, `CHARS` in `coverage/live/cpu.html` |
| 10 | One-file viewer build / publish | local | seconds | `coverage/live/build_single.mjs` |

Recommended training command (the shipped HD recipe):

    python3 denoise_torch.py --data hd.ntsd --blocks 4 --dil 1,3,9,27 --seconds 3600 --batch 16 --crop 128 \
      --appearance 1 --align 1 --coarse-first .3 --tex-smooth .2 --tex-decay 1e-4 --save model.ntst

For a standard 8-block model, warm-start from the first character's network with
`--net-from tree1-net.ntst` (in the kit). The warm start converges faster but ends at a similar
quality. Fresh 4-block models (custom `--dil`) can't warm-start from 8-block files.

## The shader, component by component

1. **Inputs** (not learned): 5 G-buffer tiles from three.js (colour, normal, depth, UV packed in two
   RGB tiles, part ID + coverage) and the camera pitch.
2. **Neural texture** (learned, per character; most of the parameters): per mesh part, a fine
   512² and a coarse 64² grid of 8 channels, sampled bilinearly by UV.
3. **Embed**: 18 inputs → 64 channels.
4. **Blocks** (×8, or ×4 for HD):
   - ternary depthwise 3×3 stencil at the block's dilation, a 4-channel group mix, a tanh gate,
     residual;
   - int8 quantisation;
   - **fast feedforward trees**: 4 trees × 4 levels, ternary `w_in`/`w_out` with row scales, GELU,
     route on the sign. 16 of 60 nodes are visited per pixel.
5. **Head**: 64 → RGBA, sigmoid.
6. **Display**: an edge-aware upscale guided by the geometry, then compositing over the backdrop.

Receptive field: 38 px (8 blocks) or 40 px (4 blocks, dilations 1,3,9,27). There is no recurrence. The viewer
was called "Tree-SSM Shader" until it was renamed "Ternary Tree Shader"; see `nts/ssm_tree.py` for a real SSM block.

## Runtime engines (`coverage/live/`)

| Engine | File | Notes |
|---|---|---|
| GPU (default) | `nts-gl.js` | WebGL2 fragment passes. Strips of rows on phones (memory). Shared MSAA for the G-buffer. Half precision. Stencil cache. Ternary weights packed 2 bits each. Hash switches: `#fp32`, `#nocache`, `#unpacked`, `#strip=N`, `#prof` |
| Fused | `fused-nts.js` | WebGPU: one compute kernel per block, the G-buffer copied GPU-to-GPU, no readbacks. f16 activations where available. Matches `gpu-nts.js` within 1/255. No edge-aware upscale yet |
| WebGPU | `gpu-nts.js` | Older compute path with CPU round trips |
| CPU | `nts.js` + `infer.c` (WASM SIMD) | Workers in bands of rows |

All GPU-engine optimisations were checked bit-identical against the previous engine in headless
Chromium (SwiftShader), except the stencil cache and half precision, whose differences were measured:
<1% of bytes change, mostly by 1/255. SwiftShader timings say little about phone GPUs: measure on
the device with `#prof`.

## Lessons (the expensive ones)

- **The film's lighting varies from frame to frame.** The shader averages the shadows into splotches.
  Flatten against the render before training (`flatten_shading.py --amount .6`).
- **L1 on noisy film frames gives grainy texture and flicker.** The denoising objective fixes most of
  it: per-frame colour and offsets (`--appearance --align`), coarse-first, texture smoothing and
  decay.
- **The video model draws what it expects, not what is there** (e.g. the bunny's ear backs came out
  pink). Fix locally with `retouch_part.py` rather than retraining.
- **Image editor:** with two input images it copies the second image's pose or outfit, so use single
  image edits. `"key": true` puts the render's background back.
- **VACE is conditioned on depth**, so the film follows the 3D silhouette. That makes it a good
  texture source, but not a source of non-literal drawing.
- **HD:** 8 → 4 blocks with dilations 1,3,9,27 kept the quality (same reach). It's the cheapest
  speed-up we found.
- **Pod mechanics:**
  - The proxy upload needs a Content-Length, so buffer each 90 MB part to a file before PUT.
  - `pkill -f name` in a job whose command line contains `name` kills the job itself.
  - The local disk is small: stream tarballs, delete scratch.
  - A pod that lost funding was deleted by RunPod, so mirror logs locally every minute.
  - Always terminate the pod and confirm with `get-pod` → 404.
- **Licences:** the Bunny (pixiv VRoid stock avatar) is personal, non-profit use only, so keep its
  model and trained files out of git and out of anything shared. Motion data is Bandai Namco Research,
  CC BY-NC 4.0.

## Limits

- **Per character.** The neural texture memorises one character's appearance. A new character
  needs its own film and training run.
- **Recolours, does not redraw.** Each pixel stays at its geometric position, with ≤40 px of
  context. It can't move an eye, simplify a silhouette or draw lines off the geometry.
- **No static-texture baseline was measured.** Baking the film into a UV texture (per-texel median)
  plus a toon render might get much of the look at zero runtime cost. Worth measuring first in any
  "is the network worth it" discussion.

## Open experiments

- **SSM tree blocks** (`nts/ssm_tree.py`, `denoise_torch.py --ssm 1,3`). This is nanoTSS's SSM tree
  layer adapted to images: a per-node state, read → write the mean of dt·B⊗V over the node's pixels
  → read again. At equal steps it learned faster than the plain 4-block model (L1 0.0156 vs 0.0163
  at step 3,000). At equal time (60 min, 3.5× slower steps) the final L1 was 0.0148 vs 0.0142, with
  lower flicker (0.0117 vs 0.0130) and texture noise (0.050 vs 0.069). Next steps: carry the state
  across frames, and a WebGPU kernel.
- **Style shader** (`nts/style_torch.py`, `make_style_data.py`). The SSM shader with no neural texture
  (a fixed part code + UV instead), trained with VGG perceptual + PatchGAN + a weak 1/8-resolution
  anchor, on film frames or on free-drawn keyframes. The question it tests: can a small real-time
  network learn drawn (non-literal) anime features on held-out angles?

  **First result (film frames, 840 frames at 360×640, 90 held out, 90 min, 5,921 steps).** Yes,
  partly. With no learned texture it draws on held-out angles:
  - strand lines through the hair, where the render has a smooth gradient;
  - ink outlines on the jaw, neck and fingers;
  - stronger lash lines;
  - warm cel skin and neutral silver hair.

  Still missing: the film's blush, its heavier ink weight, the crown whorl, and anything outside the
  3D silhouette (the film follows the silhouette, so nothing asks for it). Held-out VGG perceptual
  went from 0.687 at step 500 to 0.646.

  The model is in the kit (`style/style.ntst` + `.ssm.pt`, 4 blocks with dilations 1,3,9,27 and SSM
  blocks 1,3), with its final held-out grids.

  **In the viewer** (`#style`, the "Style" chip, GL engine): `export_ssm.py` writes the SSM
  blocks with each node's memory fixed (the mean of the states the training frames wrote; on held-out
  frames this draws within 0.63/255 of the per-frame write). That makes every block per-pixel work for
  `nts-gl.js`. It matches PyTorch on the viewer's own G-buffer within 0.10/255. The true per-frame write
  is a frame-wide reduction: do it in the fused WebGPU engine, ideally as a temporal write (read the
  previous frame's state), which also removes the sync point.

  **Naming, honestly.** With the state frozen, the viewer's style model is not an SSM. It is the
  ternary tree shader, and in blocks 1 and 3 each tree node holds a fixed learned matrix that the pixel
  reads bilinearly (`C(x)·H·Z(x)`). In training it is one SSM step per image, with no recurrence over
  frames or pixels. The tiny frozen-vs-dynamic gap (0.54/255) says the per-frame write carries almost
  nothing. No style run without these blocks (`--ssm ''`) has been done, so their value is unmeasured.
  To make it a real SSM, carry the state across consecutive frames (`h_t = dec·h_{t-1} + write_t`).

  **v2: more linework and texture, still no neural texture** (now the viewer's `nts-style/`; v1 is in
  `out/style/final/`):
  ```
  style_torch.py --data film.npz --ssm 1,3 --seconds 5400 --batch 6 --crop 256 --load v1/style.ntst \
    --coarse-tex --lines 5 --adv 0.15 --adv-from 300
  ```
  Warm start from v1, 90 min, 5,600 steps on an RTX PRO 6000.
  - `--coarse-tex`: learns only the 64² coarse texture level (8 channels per part); the fine level stays
    the fixed part code + UV. It adds ~0.4 MB to `net.png` (0.43 MB in all, against v1's 23 KB).
  - `--lines 5`: an edge loss on max-pooled luma gradients, tolerant of a few pixels of misplacement.

  Held-out results:

  | | lines | colour anchor | perceptual |
  |---|---|---|---|
  | v1 | 0.0225 | 0.0111 | 0.646 |
  | v2 | 0.0204 | 0.0096 | 0.659 |

  - Front views (seen in training) gain strand lines through the bangs and crown, the film's cheek
    blush, and heavier jaw ink.
  - Mid-run (step 2,000) it drifted to lavender hair and yellow skin; this was gone by the end.
  - Still missing: the film's large glossy eyes and the hair's shine bands. That is fine detail a
    fixed fine texture can't place.
  - The frozen state is within 0.54/255 of the per-frame write. GL matches PyTorch within 0.12/255.
  - `style_finish.py` now does what the run's scratch scripts did: the frozen state, the
    held-out check, the sheets, and the export.
  - **Bunny Style** (`#bunnystyle`) is the same recipe through `bravo.py`, 18 minutes end to end.
    Its trained files stay out of git (licence).
- **Free-drawn keyframes:** `edit_images.py` with a "draw it as an anime artist would, not a trace"
  prompt and no keying. 115 of a planned 300 were drawn for the magical girl. They clearly redraw eyes,
  face and hair beyond the 3D silhouette. They are kept in the session's `out/style/keys.tar`, which is not in the kit;
  regenerate them with `jobs-keys.json` (in the tar) and `edit_images.py`. Next test: the style shader
  on these drawings, to see whether it learns shapes beyond the silhouette.

## Resolution without retraining (`dilScale`, the viewer's 2x chip)

The network is resolution-agnostic up to its stencils' reach: 3×3 depthwise taps at fixed dilations,
then per-pixel trees. On a grid k times finer with every dilation times k, each stencil reaches as far
on her as in training, so the same weights draw the same picture at k times the resolution.

- The magical girl's style v2 on a 720×1280 frame with dilations 2, 6, 18, 54, box-averaged back
  down, matches its 360×640 drawing within 0.83/255.
- At full size it draws visibly sharper eyes, lashes, necklace and strands.

The viewer offers it for the style characters as an opt-in (`CHARS[...].fine = 2`, `nts-gl.js`
`dilScale`; the 2x chip or `#fine`). It costs 4× the network's pixels, and on a phone the gain is
marginal, so it is off by default. GL matches PyTorch at
2× within 0.184/255.

A short fine-tune on the 2× grid (the girl has 720×1280 data, with `--dil 2,6,18,54`) could sharpen
it further. The bunny would need 2× G-buffers first.

## A learned upscale instead (`nts/upsample_torch.py`): tried, not shipped

The idea was to keep the 1× network and learn the step to the screen instead. The magical girl's
style v2 drew at 1×; training was at 2× against her 720×1280 film, 3000 steps, about 20 minutes on an
RTX PRO 6000. Held-out crops:

| | L1 | perceptual | lines |
|---|---|---|---|
| hand-made edge-aware upscale (the viewer's) | 0.0365 | 1.794 | 0.0678 |
| learned re-weighting (MLP 10 → 12 → 1 a tap) | 0.0365 | 1.736 | 0.0647 |
| + detail head (normal, part code, uv sines → colour) | 0.0448 | 1.614 | 0.0617 |

- **Re-weighting** gives a learned logit to each of the 16 network pixels a screen pixel blends. It
  can only choose among colours the drawing already has: the gain is real but barely visible.
- **The detail head** scored better but looks worse. Its gain is mostly a colour shift towards the
  film's brighter hair, at the cost of strand contrast.

What the 1× drawing lacks is detail it never drew, and no cheap per-pixel step at the screen's
resolution recovers it. The shader code was taken back out; the trainer stays for a later attempt.
Two options remain. One is training the network itself at 2× (it runs on the 2× grid, `#fine`); the
other is a detail head with its own small texture, which is a neural texture again.

## Style training speed (`style_torch.py`, `bravo.py`)

One step is 6 crops of 256² with the discriminator on, on an RTX PRO 6000:

| Version | ms a step |
|---|---|
| Before | 934 |
| Changes 1–4 below | 94 |
| Plus `--compile` (5) | 79 |

Overall that is ~12× faster. On the same data the old and new trainers' losses tracked each other
step for step.

1. **SSM blocks as dense matmuls** (`ssm_tree.py`, `test_ssm_fast.py`). Each pixel used to gather
   its visited nodes' state matrices: (N, 16, 8, 8) copies, about 2 GB a step, and a scatter in the
   backward pass. A node's read Cᵀ·h·Z is bilinear, so all nodes are read at once as
   (C⊗Z) @ hᵀ. The per-image write is (visits × gain)ᵀ @ (B⊗V), and the U term sums the visited
   states per tree. Same outputs and gradients to float rounding.
2. **Texture** (`shader_torch.inputs`): one gather per level instead of four. A level the trainer
   holds fixed (the style model's fine level) has no gradient buffer.
3. **Stencils in NCHW memory.** On the channels-last view cuDNN fell back to a slow fp32 depthwise
   weight-gradient kernel (17.9 → 6.6 ms for the 4 stencils).
4. **No host syncs in a step.** The norm clip became a tensor, the lazy texture Adam a masked dense
   update over the rows that can move, and the losses are read only at evals. VGG and the
   discriminator run in bf16 autocast, channels-last. The shader stays fp32 with TF32 matmuls, which
   is exact for its ternary × int8 sums.
5. **`--compile`** (`torch.compile` of the shader forward) fuses its elementwise work. `bravo.py` uses
   it by default.

`style_torch.py --bench 30` prints a step's time and the GPU profile. Steps are budgeted (`--iters`),
so the schedule no longer depends on how fast the GPU is.

## New character checklist

1. VRM → `slim_vrm.py` → `<name>/model.vrm.png`. Pick a shorts colour from the outfit.
2. Render the two orbit waves with `--gbuffer` (`--shorts --shorts-color` for the bottom wave).
3. Start the pod (`boot-asset.sh`). Draw the style reference (a few seeds; keep the most faithful),
   then the anchors every 80 frames.
4. VACE film with a character prompt (`--prompt`). Check one chunk before the rest.
5. `build_dataset.py --canon parts.json` per wave → flatten → pack → `denoise_torch.py` (warm start
   or `--blocks 4 --dil 1,3,9,27`).
6. `--metrics` against the held-out frames. Look at it from behind and from below. Retouch any part
   the film got wrong.
7. `export_nts.py` → copy the `.wasm` files from `nts-tree/` → add to `CHARS` → `build_single.mjs` →
   publish. Terminate the pod.

## Kit contents

`tree-ssm-kit.zip` holds this file and the pipeline code:
- `coverage/render-orbit.mjs`, `orbit-page.html`
- `coverage/pod/`
- `coverage/shader/build_dataset.py`
- `coverage/shader/nts/`: trainers, exporter, WASM kernel source, nanoTSS
- `coverage/live/`: viewer sources, engines, motions
- `coverage/asset/README.md`

It also holds a warm-start network (`tree1-net.ntst`, the first character's network with its texture
zeroed) and the Magical Girl's model and networks as a working example. Node deps: `npm install`
(three 0.170.0, @pixiv/three-vrm 3.5.5, playwright-core). Python: torch, numpy, Pillow (+ diffusers
on the pod).
