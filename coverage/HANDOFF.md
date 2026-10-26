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

Receptive field: 38 px (8 blocks) or 40 px (4 blocks, dilations 1,3,9,27). There is no recurrence:
"SSM" in the viewer's name is historical. See `nts/ssm_tree.py` for a real one.

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
  anchor, on film frames or on free-drawn keyframes. A first run on the film frames was in progress
  at handoff. The question it tests: can a small real-time network learn drawn (non-literal) anime
  features on held-out angles?
- **Free-drawn keyframes:** `edit_images.py` with a "draw it as an anime artist would, not a trace"
  prompt and no keying. 115 of a planned 300 were drawn for the magical girl. They clearly redraw eyes,
  face and hair beyond the 3D silhouette.

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
