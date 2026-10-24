#!/usr/bin/env python3
"""Draw the style reference and the anchor keyframes on the GPU pod, with an open image-edit model
(Qwen-Image-Edit-2509, Apache-2.0) instead of OpenRouter: the same prompts as style-ref.py and
anchors.py, the same inputs (the render; for anchors, the render and the style reference).

    python3 edit_images.py jobs.json [--steps 8] [--seed 7]

jobs.json: [{"out": "style-ref.png", "images": ["ref/0240.png"], "prompt": "...", "seed": 7, "key": true}, ...], run in order
(so the style reference can be drawn first and used by the anchors after it). Each output is resized
back to its first image's size, so it lines up with the render, and (with "key": true, the first image
a render on the green) its background is put back from the render: everything off her silhouette
(keyed from the render, grown 3 px for the drawing's outline) is the render's exact green, so an
edit's drop shadows, ground shadow or shifted green never reach the film. Jobs whose output exists are skipped.
With the Lightning LoRA (lightx2v/Qwen-Image-Lightning, if it has an Edit-2509 file) 8 steps without
true CFG; otherwise 40 steps with it.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image

MODEL = 'Qwen/Qwen-Image-Edit-2509'
LIGHTNING = 'lightx2v/Qwen-Image-Lightning'


def lightning_file(steps):
    try:
        files = [f for f in list_repo_files(LIGHTNING) if 'Edit-2509' in f and f.endswith('.safetensors') and 'bf16' in f]
    except Exception as e:  # noqa: BLE001 - no LoRA is a slower run, not a failure
        print('no Lightning LoRA list:', e, flush=True)
        return None
    want = [f for f in files if f'{steps}steps' in f] or files
    return sorted(want)[-1] if want else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('jobs')
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--seed', type=int, default=7)
    a = p.parse_args()
    jobs = json.loads(Path(a.jobs).read_text())
    from diffusers import QwenImageEditPlusPipeline
    t0 = time.time()
    pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to('cuda')
    lora = lightning_file(a.steps)
    if lora:
        pipe.load_lora_weights(hf_hub_download(LIGHTNING, lora))
        pipe.fuse_lora()
        steps, cfg = a.steps, 1.0
    else:
        steps, cfg = 40, 4.0
    print(f'loaded in {time.time() - t0:.0f} s; LoRA {lora}; {steps} steps, true CFG {cfg}', flush=True)
    for j in jobs:
        out = Path(j['out'])
        if out.exists():
            continue
        imgs = [Image.open(f).convert('RGB') for f in j['images']]
        t1 = time.time()
        img = pipe(image=imgs if len(imgs) > 1 else imgs[0], prompt=j['prompt'], negative_prompt=' ',
                   num_inference_steps=steps, true_cfg_scale=cfg,
                   generator=torch.Generator('cuda').manual_seed(j.get('seed', a.seed))).images[0]
        out.parent.mkdir(parents=True, exist_ok=True)
        img = img.convert('RGB').resize(imgs[0].size, Image.LANCZOS)
        if j.get('key'):
            src = np.asarray(imgs[0]).astype(np.int16)
            her = torch.from_numpy((src[..., 1] - np.maximum(src[..., 0], src[..., 2]) <= 64).astype(np.float32))
            her = torch.nn.functional.max_pool2d(her[None, None], 7, 1, 3)[0, 0].numpy() > 0
            img = Image.fromarray(np.where(her[..., None], np.asarray(img), src.astype(np.uint8)))
        img.save(out)
        print(f'{out} {img.size} in {time.time() - t1:.1f} s', flush=True)


if __name__ == '__main__':
    main()
