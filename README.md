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
#   options: --yaw-step 30 --pitches -15,0,20 --size 768x1152 --fov 22 --background "#ffffff"
#            --anim Idle --frames 8      (animations: Running, Walking)
#   in the cloud sandbox: CHROMIUM_PATH=/opt/pw-browsers/chromium-1194/chrome-linux/chrome npm run render

# 2. stylize every view (skips ones already done; --force to redo)
OPENROUTER_API_KEY=sk-or-... npm run stylize
#   options: --model google/gemini-3.1-flash-lite-image --notes assets/emerald-pixie-notes.txt --style assets/emerald-pixie-ref.jpg --style-only
#            --prompt "extra style notes" --concurrency 4 --only y000_p0.png,y030_p0.png
#            --no-anchor --dry-run (print prompts, no API calls)

# 3. view
npm run viewer   # then open http://localhost:8000/viewer/  (?set=out/renders shows the raw renders)
```

Image-only models (Hy Image, Grok Imagine) are sent to OpenRouter's `/images` endpoint automatically; `--endpoint chat|images` forces one. Each run prints its OpenRouter cost.

`OPENROUTER_BASE_URL` overrides the API endpoint, and `OPENROUTER_MODEL` sets the default model.

## Cost

Measured per sprite (one angle of one frame), at about 840x1260:

| model | $/sprite | notes |
|---|---|---|
| `google/gemini-3.1-flash-lite-image` (default) | ~$0.032 | needs `assets/emerald-pixie-notes.txt` to keep reference-only details (ribbons, plain back) |
| `google/gemini-2.5-flash-image` | ~$0.036 | follows the 3D render closely and drops reference-only details |
| `google/gemini-3.1-flash-image` | ~$0.09 | best at adding reference-only details |
| `tencent/hy-image-v3.5-preview` | ~$0.024 + input images? | image-only model, uses OpenRouter's `/images` endpoint (picked automatically); not tested yet |

36 angles is about $1.15 with flash-lite. OpenRouter refuses image requests once the account balance is under $1.

Start with `--only y000_p0.png,y090_p0.png,y180_p0.png` to tune the prompt and style before running the full set.

## Notes

- `assets/emerald-pixie-ref.jpg` is official art of the same character, so by default the model copies its design as well as its style. That fills in details the 3D model simplifies or lacks, such as the long twin-tail ribbons. Pass `--style-only` to use a reference image for its art style alone.
- Materials named `Outline` (inverted-hull outline shells) are drawn first without depth writes, so only the silhouette rim shows.
