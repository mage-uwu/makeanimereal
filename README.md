# make anime real

Turn a 3D character into something that *looks* hand-drawn from every angle:

1. **render**: a headless three.js turntable takes the GLB from a grid of camera angles (and, optionally, animation frames).
2. **stylize**: each render goes to an OpenRouter image model together with a style reference image. The model redraws that exact view as an anime illustration. The front view is drawn first and passed in as a character anchor so every angle stays on-model.
3. **viewer**: drag to orbit. The viewer swaps in the pre-drawn illustration closest to the camera angle, so the "3D model" renders as stylized 2D art.

## Usage

```sh
npm install

# 1. render 12 yaws x 3 pitches = 36 views of assets/nya-bytetuber.glb into out/renders
npm run render
#   options: --yaw-step 30 --pitches -15,0,20 --size 768x1152 --fov 22 --background "#ffffff"
#            --anim Idle --frames 8      (animations: Walk, Run, Idle, Slash, Bow)
#   in the cloud sandbox: CHROMIUM_PATH=/opt/pw-browsers/chromium-1194/chrome-linux/chrome npm run render

# 2. stylize every view (skips ones already done; --force to redo)
OPENROUTER_API_KEY=sk-or-... npm run stylize
#   options: --model google/gemini-3.1-flash-image --style assets/style-ref.jpg
#            --prompt "extra style notes" --concurrency 4 --only y000_p0.png,y030_p0.png
#            --no-anchor --dry-run (print prompts, no API calls)

# 3. view
npm run viewer   # then open http://localhost:8000/viewer/  (?set=out/renders shows the raw renders)
```

`OPENROUTER_BASE_URL` overrides the API endpoint, and `OPENROUTER_MODEL` sets the default model.

## Cost

36 views is 36 image generations. Start with `--only y000_p0.png,y090_p0.png,y180_p0.png` to tune the prompt and style before running the full set.

## Notes

- The model's outline shells don't come through the GLTF export as a working inverted hull. The renderer draws them first without depth writes, so only the silhouette rim shows.
- `assets/style-ref.jpg` sets the art style only. The prompt tells the model not to copy that character's design or colors.
