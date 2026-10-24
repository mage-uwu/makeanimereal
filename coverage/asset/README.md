# A new character for the Tree-SSM shader

The per-asset pathway: one VRM in, a trained nanotss tree shader out, as a character on the
Tree-SSM page. Everything heavy runs on one RunPod GPU pod (`coverage/pod/`); the renders and the
dataset are made locally. The second character (pixiv's VRoid stock avatar, "bunny") went through it
in about 2 hours, most of it the local renders.

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
fresh neural texture for this one's parts: the network already knows how to draw the style, so the
new character trains mostly its surface.

## 6. Export and view

    python3 coverage/shader/nts/export_nts.py c2.ntst coverage/live/bunny/nts-tree --parts $C/data/top/meta.json

Add the character to `CHARS` in `coverage/live/cpu.html` (model, network folder, shorts), pack its
VRM with `coverage/live/slim_vrm.py`, rebuild `build_single.mjs` and publish the model and network
files beside the page.
