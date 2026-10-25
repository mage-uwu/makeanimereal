# A new character for the Tree-SSM shader

The per-asset pathway: one VRM in, a trained nanotss tree shader out, as a character on the
Tree-SSM page. Everything heavy runs on one RunPod GPU pod (`coverage/pod/`); the renders and the
dataset are made locally.

The second character (pixiv's VRoid stock avatar, "bunny", `#bunny` on the page) went through it in
about 2 hours. The pod (RTX PRO 6000, $2.49/h) was up 1.6 h, about $3.90:

| Step | Where | Time |
|---|---|---|
| Orbit renders, 1440 frames with G-buffer | local, 4 vCPUs | ~80 min (two in parallel) |
| Weights (editor, VACE, LoRAs) | pod | 2 min |
| Style reference, 20 anchors | pod | ~10 s an image |
| Film, 18 chunks at 432×768 | pod | ~45 s a chunk, 14 min (alongside the renders) |
| Dataset, 1440 frames at 360×640 | local | 3 min |
| Training, warm start from the first character | pod | 30 min: ~40k steps, held-out L1 0.0128 |

Leave the bunny's model and trained files out of git: its VRM licence (personal non-profit, no
redistribution) covers this private page, not a public repository.

Each step below is a command; `$C` is the character's work folder (e.g. `out/c2`), `$P` is
`coverage/pod/pod.sh` with `POD` and `POD_TOKEN` set.

## 1. Render the coverage orbit (local, ~80 min on 4 vCPUs)

The same two camera waves as the first character: 960 frames from pitch −20° to 70° (4 turns), and
480 from −20° to −88° (2 turns back), with the G-buffer (normal, depth, UV, part) for every frame.

    node coverage/render-orbit.mjs --model $C/model.vrm --out $C/top --gbuffer
    node coverage/render-orbit.mjs --model $C/model.vrm --out $C/bottom --gbuffer --shorts --shorts-color "#f4dde3" \
        --yaw0 0 --turns=-2 --pitch=-20:-88 --seconds 30 --skip-first

- `--shorts` draws safety shorts over the hips for the views from below the hem; pick a colour from
  the outfit. The viewer draws the same shorts (`CHARS[...].shorts` in `cpu.html`).
- `--resume` continues a stopped render.
- Run the two in parallel; each writes ~1.2 MB a frame.

## 2. Start the pod

One RTX PRO 6000 (96 GB, Secure Cloud): the image editor and the 14B video model fit without
offloading. Boot with `coverage/pod/boot-asset.sh` (env `BOOT_SH`, `SERVER_PY`, `POD_TOKEN`; command
`echo $BOOT_SH | base64 -d > /boot.sh && bash /boot.sh; sleep infinity`, port `8000/http`). It
fetches Qwen-Image-Edit-2509, the Lightning LoRA, Wan2.1 VACE 14B and the lightx2v LoRA; on a
fast data centre that took two minutes.

## 3. Style reference and anchors (pod, ~10 s an image)

`coverage/pod/edit_images.py` (Qwen-Image-Edit-2509, 8 Lightning steps) redraws a render as a
finished anime frame. Each job is one image: the render alone.
- The style reference is the front frame (top wave frame 240), with `style-ref.py`'s prompt plus
  "Crisp dark lineart outlines…". Draw a few seeds and keep the most faithful.
- The anchors are every 80th frame of each wave and its last frame, with the same prompt and a
  fixed seed.
- With `"key": true` the background is put back from the render (her silhouette keyed off the
  render's green, grown 3 px), so the editor's drop shadows and shifted green never reach the film.

Two inputs per job (render plus a style exemplar) does not work with this editor: it copies the
second image's pose or outfit.

## 4. The film (pod, ~50 s a chunk)

    python3 vace_film.py --depth top/depth.mp4 --ref style-ref.png --anchors anchors_top --overlap 1 \
        --chunk 81 --size 432x768 --prompt "a girl with ..." --out film_top

- Each 81-frame chunk is pinned between two anchors and filled from the rendered depth.
- 432×768 is the render's aspect at a third of the pixels. The shader trains at 360×640, so the
  film needs no more.
- `--prompt` describes the character, in place of the built-in description of the first one.
- `--only 0,1,2` makes some chunks: the film can start while the render is still being drawn, as
  long as each chunk's frames and both its anchors exist.

## 5. Dataset (local) and training (pod)

    python3 coverage/shader/build_dataset.py --gbuffer $C/top --film $C/film_top --out $C/data/top --canon $C/parts.json
    python3 coverage/shader/build_dataset.py --gbuffer $C/bottom --film $C/film_bottom --out $C/data/bottom --canon $C/parts.json
    python3 coverage/shader/nts/pack_data.py $C/data/top $C/data/bottom --out /dev/stdout | zstd -3 > $C/data.ntsd.zst

`parts.json` is the bottom wave's part list (the top wave's plus `Shorts`, last). On the pod:

    zstd -dc data.ntsd.zst | python3 shader_torch.py --data - --tree 4,4 --net-from tree1.ntst \
        --seconds 1500 --batch 16 --save c2.ntst

`--net-from` starts from the first character's trained network (embed, stencils, trees, head) with a
fresh neural texture for this one's parts. After 2 minutes it was at held-out L1 0.0128 against 0.0134
from a random network: a head start, not a different end point. On the RTX PRO 6000 the trainer runs
~18 steps/s at batch 16 (2.6 M training pixels/s; the C trainer did 0.5 M on 16 CPU cores).

## 6. Export and view

    python3 coverage/shader/nts/export_nts.py c2.ntst coverage/live/bunny/nts-tree --parts $C/data/top/meta.json

Add the character to `CHARS` in `coverage/live/cpu.html` (model, network folder, shorts), pack its
VRM with `coverage/live/slim_vrm.py`, rebuild `build_single.mjs` and publish the model and network
files beside the page.

## 7. Denoise (`coverage/shader/nts/denoise_torch.py`)

A film drawn by a video model disagrees with itself a little from frame to frame, and plain L1 lets
each texel of the neural texture absorb the error of the few frames that saw it: splotches, which
slide over her as the camera moves (jitter). Train with the denoising objective instead of
`shader_torch.py`:

    python3 denoise_torch.py --data data.ntsd --net-from tree1.ntst --seconds 1800 --batch 16 --save c2.ntst \
        --appearance 1 --align 1 --coarse-first .3 --tex-smooth .2 --tex-decay 1e-4
    python3 denoise_torch.py --data data.ntsd --metrics c2.ntst     # L1, flicker, texture noise, lines

Measured on the bunny (144 held-out frames, 36 consecutive-frame pairs; flicker is the colour change
of the same surface point between frames, 0.0200 for the film itself and 0.0049 for the raw render):

| | L1 | flicker | texture noise | lines |
|---|---|---|---|---|
| plain L1, 40k steps (first ship) | 0.0128 | 0.0180 | 1.15 | 0.70 |
| plain L1, 9k steps | 0.0122 | 0.0115 | 0.75 | 0.64 |
| + per-frame colour and offsets | 0.0123 | 0.0109 | 0.69 | 0.63 |
| + coarse-first, smoothing, decay | 0.0122 | 0.0090 | 0.090 | 0.59 |
| both (shipped) | 0.0125 | 0.0083 | 0.086 | 0.58 |

"Lines" (edge energy against the film's) counts noise as edges. At the film's ink only, the shipped
models reach 0.55 → 0.47 of its edge strength, and in flat areas their gradient went 0.0186 → 0.0093
(the film: 0.0041). `--lines` (ink-weighted loss, gradient matching) did not sharpen them: the film's
lines sit a little differently in every frame, so fitting them harder only averages them.

## 8. Flat lighting (`coverage/shader/nts/flatten_shading.py`)

The film's frames weren't drawn under one light, and the shader averages their shadows into soft
splotches. Before packing, flatten each target's lighting against the raw render (lit the same in
every frame), then train as in step 7:

    python3 flatten_shading.py data/top data/bottom --out-suffix _f6 --amount .6
    python3 pack_data.py data/top_f6 data/bottom_f6 --out f6.ntsd

On the bunny (shipped with `--amount .6`): the skirt and frosting read as even cel colour and the
back's blotchy shading calms, at the same flicker and texture noise; its L1 against the unflattened
film rises (0.0125 → 0.0139), as it should. Shadows wider than `--radius` (the skirt's on her thighs,
drawn in every frame) count as real shading and stay. Run it on the pod: ~20 frames a minute on CPU.

## 9. High resolution (the first character, `#girlhd` on the page)

The same pipeline at twice the size: the film's frames and the G-buffer at 720×1280 (840 frames: 480
of the top wave, 240 of the bottom, 120 mocap poses), flattened (`--amount .6 --radius 24`), and the
denoising objective with `--crop 128` (the stencil's 38 px reach covers half as much of her at this
size, so the crops grow with it). Warm start from the first character's network; 45 min on the
RTX PRO 6000, 12.7k steps, held-out L1 0.0142, flicker 0.0131, texture noise 0.072.

In `CHARS`, `scale: 2` doubles the network's frame (1440×1600, the crop margins with it): four times
the pixels of the standard model per frame.

The shipped HD model is smaller: 4 blocks instead of 8, dilations 1, 3, 9, 27 (`--blocks 4 --dil
1,3,9,27`, from scratch, 60 min) so it still sees 40 px around each pixel. It matched the 8-block one
(held-out L1 0.0142 both, flicker 0.0130 vs 0.0131, texture noise 0.069 vs 0.072) at about 55-60% of
its GPU time: in SwiftShader an HD frame took 12.8-14.0 s against 21.0-24.1 s, the standard character
6.7-7.0 s.

## 10. Retouch what the film drew wrong (`coverage/shader/nts/retouch_part.py`)

The video model draws what it expects: the bunny's ears from behind are white in the model (their own
back faces and UV islands) but the film drew them pink, as from the front. `retouch_part.py` moves
only that part's neural texture (network frozen) so the region's pixels match the raw render's colour
and every other pixel keeps the model's own output:

    python3 retouch_part.py --model f6.ntst --data data/top data/bottom \
        --part Accessory_RabbitEar_01_CLOTH --u 0:0.28 --u 0.72:1 --steps 400 --save f6e.ntst

On the CPU (4 vCPUs) 400 steps took 20 min; the rest of her moved by under 0.002 (L1).
