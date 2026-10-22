# Coverage film

A viewer-only app for one static model: a single camera spiral around the model ("the coverage
film") is rendered, restyled to 2D anime in one video pass, and the viewer shows whichever film
frame was taken nearest the camera you steer.

1. **Render the orbit**:
   ```
   node coverage/render-orbit.mjs --out out/coverage
   ```
   - Renders 960 frames (60 s at 16 fps) on a flat green background.
   - The spiral makes 4 turns and climbs from pitch −20° to 70°.
   - Writes `orbit.mp4`, `depth.mp4` (near is white) and `cameras.json` (the exact camera of every frame).
2. **Style reference**:
   ```
   python3 coverage/style-ref.py out/coverage/frames/0240.png coverage/assets/style-ref.png
   ```
   One front frame redrawn as an anime illustration (OpenRouter, $0.024). It sets the look for the whole film.
3. **Restyle on a GPU pod.** Run `coverage/pod/vace_film.py` (Wan2.1 VACE 14B with the lightx2v 4-step LoRA) on a 48 GB card:
   - `coverage/pod/boot.sh` installs the stack and fetches the weights.
   - `coverage/pod/server.py` takes uploads and jobs through the RunPod proxy.
   - `coverage/pod/pod.sh` talks to it.

   The film is made in 81-frame chunks. Each chunk holds the previous chunk's last 17 finished frames fixed and generates the rest from the rendered depth, with the style reference on every chunk.
   - Measured on an RTX A6000 (fp8 weight storage, 720×1280): 507 s per chunk, 15 chunks.
4. **Pack for the web**:
   ```
   python3 coverage/pack.py film.mp4 --out coverage/web
   ```
   Keys out the green and packs the frames into 4×4 WebP sheets with `film.json`.
5. **View**: open `coverage/web/index.html` through any static server. Drag to orbit, or use auto orbit or play the film.

`coverage/assets/magical_girl_02.vrm` is the model (VRoid Studio, by mage).
