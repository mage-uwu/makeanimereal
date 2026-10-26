# make anime real

Turn a 3D character into something that *looks* hand-drawn from every angle:

1. **render**: a headless three.js turntable takes the GLB from a grid of camera angles (and, optionally, animation frames).
2. **stylize**: each render goes to an OpenRouter image model together with a style reference image. The model redraws that exact view as an anime illustration. The front view is drawn first and passed in as a character anchor so every angle stays on-model.
3. **viewer**: drag to orbit. The viewer swaps in the pre-drawn illustration closest to the camera angle, so the "3D model" renders as stylized 2D art.

## Usage

```sh
npm install

# 1. render 12 yaws x 3 pitches = 36 views of assets/emerald-pixie.glb into out/renders
npm run render
#   options: --yaw-step 30 --pitches=-15,0,20 --size 768x1152 --fov 22 --background "#ffffff"
#            --anim Idle --frames 8      (animations: Running, Walking)
#   in the cloud sandbox: CHROMIUM_PATH=/opt/pw-browsers/chromium-1194/chrome-linux/chrome npm run render

# 2. stylize every view (skips ones already done; --force to redo)
OPENROUTER_API_KEY=sk-or-... npm run stylize
#   options: --model tencent/hy-image-v3.5-preview --notes assets/emerald-pixie-notes.txt --style assets/emerald-pixie-ref.jpg --style-only
#            --prompt "extra style notes" --concurrency 4 --only y000_p0.png,y030_p0.png
#            --no-anchor --dry-run (print prompts, no API calls)

# 3. view
npm run viewer   # then open http://localhost:8000/viewer/  (?set=out/renders shows the raw renders)
```

Image-only models (Hy Image, Grok Imagine) are sent to OpenRouter's `/images` endpoint automatically; `--endpoint chat|images` forces one. Each run prints its OpenRouter cost.

`OPENROUTER_BASE_URL` overrides the API endpoint, and `OPENROUTER_MODEL` sets the default model.

## VRM stage (live 3D)

`web/stage.html` renders a VRM avatar live: drag to orbit, MToon shading, spring-bone hair and
skirt, an idle sway, blinking, and buttons for pose (stand, wave, T-pose) and expression. With
`npm run viewer` open http://localhost:8000/web/stage.html: it loads `?model=<url>`, else
`web/stock.vrm.png` (a VRM packed by `coverage/live/slim_vrm.py`), else
`out/stock/stock_avatar.vrm` if you put one there, else waits for a dropped or chosen `.vrm`.

### Moves (VRM Animation)

The stage's moves are standard [VRM Animation](https://vrm.dev/en/vrma/) files, `web/moves/*.vrma`
(glTF with `VRMC_vrm_animation` 1.0), played by `@pixiv/three-vrm-animation`. They also work in
UniVRM, VRoid Hub and anything else that reads VRMA. Each file is one seamless loop.

They are 11 clips from the [CMU Graphics Lab Motion Capture Database](http://mocap.cs.cmu.edu/)
(free for all uses, including in commercial products; the raw data may not be resold), made in
two steps:

```sh
# 1. retarget: CMU .asf/.amc (or BVH) -> bone directions per frame, assets/motions/cmu.json
mkdir -p cmu && cd cmu && curl -O http://mocap.cs.cmu.edu/subjects/91/91.asf -O http://mocap.cs.cmu.edu/subjects/91/91_29.amc && cd ..
python3 coverage/motion/retarget_bvh.py cmu --out assets/motions/cmu.json --face-front \
  --credit "Motion: CMU Graphics Lab Motion Capture Database (mocap.cs.cmu.edu), created with funding from NSF EIA-0196217" \
  --clip walk=91.asf:91_29.amc --clip wave=141.asf:141_16.amc ...   # each clip's source is listed in the JSON

# 2. bake: pose a generic VRoid-proportioned T-pose rig with coverage/live/motion.js -> web/moves/<clip>.vrma
node scripts/bake-vrma.mjs --motions assets/motions/cmu.json --out web/moves
```

`--face-front` takes out slow turning as well as travel, so clips that walk loops round the room
play facing the front. The credit line goes into each file's glTF `asset.copyright`. The coverage
clips in `coverage/live/motions.json` are from the Bandai Namco Research Motion Dataset, which is
CC BY-NC 4.0: non-commercial only.

### vrmanet (a motion network whose weights are a .vrma)

The stage plays the moves through [vrmanet](vrmanet/README.md): `web/moves/vrmanet.vrma` holds all 11
clips as experts (decoder rows = keyframes) plus a state-space controller's weights in a glTF
extension, and a 7 KB C runtime (`web/vrmanet.wasm`) runs it. The stage's Arms row routes the arms
to a different expert than the body, e.g. waving while walking.

### Free run (PARC's motion generator, live)

The stage's **Free run** button runs [parcgen](parcgen/README.md): PARC's diffusion motion
generator ported to C/WASM (`web/parcgen.wasm` + `web/parcgen.bin`), generating her motion half a
second at a time from the terrain around her and the direction you steer (WASD / arrows, or an
auto-run lap of vaults, rooftops, a climb and stairs). Kinematic: no physics yet.

## Web viewer (anime key drawings you can orbit freely)

Gen 2: 174 key drawings on eight camera-height rings from -30 deg (looking up) to +75 deg (nearly
overhead), every 15 deg around up to +45 deg, then 18 and 12 keys on the top rings.

```sh
RINGS="--pose relaxed --rings=-30:24,-15:24,0:24,15:24,30:24,45:24,60:18,75:12"
# renders on the reference sheet's magenta, the same views with no background, and the posed mesh
npm run render -- $RINGS --background "#c006a6" --out out/g2-renders --export-mesh out/g2-renders/mesh.json --depth
npm run render -- $RINGS --background none --out out/g2-alpha
npm run stylize -- --renders out/g2-renders --out out/g2          # ~$4.20 for 174 keys
pip install opencv-python-headless                                # plus Pillow and numpy
python3 scripts/build-web.py --renders out/g2-renders --alpha out/g2-alpha --keys out/g2 \
  --out out/web2 --cache out/cache2                               # ~21 MB, streamed nearest-first
```

Draw on a chroma background from the start: the renders and the drawings both use the reference
sheet's magenta, so `build-web.py` keys the background out (and removes magenta spill from edges)
instead of guessing which white belongs to her socks.

The viewer (`web/index.html`) never interpolates a key drawing: on a key angle it shows that
drawing exactly as drawn. Interpolation only lives between keys, in one of three modes, and never
mixes two drawings on a pixel:

- **Projected**: the nearest drawing is projected onto an invisible 3D mesh of her from the camera
  it was drawn for and seen from the current camera, so it bends with her shape. Its fit to the
  mesh (a stored flow field) eases in from none at the key to all of it at the hand-off. Surfaces
  it could not see come whole from the next drawing that did; where nothing was drawn, nothing shows.
- **Flipstretch**: no model. The nearest drawing is stretched toward up to two neighbouring
  drawings along dense correspondences matched between the drawings themselves (optical flow),
  as far as the camera has moved toward each. It hands over halfway to the next key.
- **Flipbook**: the nearest drawing held still, like a 2D game sprite.
- **Neuromation** (latent rotation on rigid shells): every key drawing is one view of a single
  latent object, but each was drawn from a slightly different vantage than requested. `build-web.py`
  learns that vantage per key (yaw and pitch offsets, roll, framing; Gauss-Newton with Huber weights
  over the surface points visible in its render, smoothed only where neighbours would swap order)
  and turns the drawing into a rigid, gently curved shell: each vertex of a 75 x 112 lattice sits on
  the ray from that vantage through its pixel, at a depth smoothed over ~32 px. Seen from its
  vantage the shell is exactly the drawing; because the depth is smooth, it cannot shear, bulge or
  tear when the view turns. Between keys your angle maps to a latent angle blending the nearby
  keys' vantages and the shell simply rotates. Approaching a hand-off, an articulated correction
  (a 6 x 9 control grid, ~100 px apart, so head, torso and legs move as blocks) moves the drawing
  up to halfway toward where the next drawing put things. The Depth relief slider flattens the
  shell toward a flat card; full relief is the default, since flattening makes neighbouring
  drawings disagree more at the hand-off.

`build-web.py` writes per key the untouched drawing (`keys/`) and a warp atlas (`warps/`, lossless
WebP): tile 0 is the fit to the mesh, tiles 1-6 the correspondences to each neighbouring key (the
nearest one in each direction around it on the view sphere), tile 7 the Neuromation lattice tensor
(16-bit 3D point per vertex in high/low byte planes, paint mask, nearest bone). `scripts/latent.py`
rebuilds the key cameras and decodes the renderer's depth maps (`render.mjs --depth`).

`mesh.json` also carries the rig's skeleton and each vertex's main bone; the viewer can overlay the
skeleton, and the lattice records each vertex's bone so drawings can later follow a posed skeleton.

### Pose check (MediaPipe)

```sh
pip install mediapipe        # Linux also needs libegl1
curl -L -o out/models/pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
python3 scripts/pose_check.py   # -> out/pose-check.json, prints drawings to consider redrawing
```

The rig's own skeleton is the pose source: projected through each key camera and the drawing's fit,
it lands on the drawn joints in every view. MediaPipe, trained on photos of people, is used only as
an independent check on views where it is reliable (face visible, no steeper than 45 degrees, 75 of
174 keys here); it flags drawings whose limbs disagree with the rig.

The viewer gets only the posed shape (`mesh.json`, exported by the renderer with the same three.js
code that made the renders), not the GLB: no textures, skeleton or animation. `web/pose.js` holds
the relaxed pose.

## Cost

Measured per sprite (one angle of one frame), at about 840x1260:

| model | $/sprite | notes |
|---|---|---|
| `google/gemini-3.1-flash-lite-image` | ~$0.032 | needs `assets/emerald-pixie-notes.txt` to keep reference-only details (ribbons, plain back) |
| `google/gemini-2.5-flash-image` | ~$0.036 | follows the 3D render closely and drops reference-only details |
| `google/gemini-3.1-flash-image` | ~$0.09 | best at adding reference-only details |
| `tencent/hy-image-v3.5-preview` (default) | $0.024 | best results so far, input images free, 1248x1872 output, ~28s per image; uses OpenRouter's `/images` endpoint (picked automatically) |

36 angles is about $0.86 with Hy Image. OpenRouter refuses image requests once the account balance is under $1.

Start with `--only y000_p0.png,y090_p0.png,y180_p0.png` to tune the prompt and style before running the full set.

## Notes

- `assets/emerald-pixie-ref.jpg` is official art of the same character, so by default the model copies its design as well as its style. That fills in details the 3D model simplifies or lacks, such as the long twin-tail ribbons. Pass `--style-only` to use a reference image for its art style alone.
- Materials named `Outline` (inverted-hull outline shells) are drawn first without depth writes, so only the silhouette rim shows.
