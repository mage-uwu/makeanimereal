#!/usr/bin/env python3
"""Restyle the whole coverage film with Wan2.1 VACE 14B, on the GPU pod.

    python3 vace_film.py --depth depth.mp4 --ref style-ref.png --out film [--steps 4] [--size 720x1280]
                         [--chunk 81] [--overlap 17] [--only 0]

The film is too long for one pass (the model makes 81 frames at a time), so it is made in chunks
that overlap: each chunk's first --overlap frames are the previous chunk's finished frames, held
fixed (black mask), and the rest are generated (white mask) from the rendered depth. With the same
style reference on every chunk, the look carries across the joins. --seed-frames opens the film
with frames that are already finished (another film's last frames), so a new camera path continues
that film's look. Writes <out>/chunk_NN.mp4 as
each chunk finishes (so a stopped run resumes where it left off) and <out>/film.mp4 at the end.
The lightx2v step-distill LoRA lets 4 steps without classifier-free guidance stand in for 50.
"""
import argparse
import gc
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanVACEPipeline, WanVACETransformer3DModel
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download
from PIL import Image

MODEL = 'Wan-AI/Wan2.1-VACE-14B-diffusers'
LORA = ('Kijai/WanVideo_comfy', 'Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors')
PROMPT = (
    'Hand-drawn 2D anime cel animation of a girl with long straight silver hair and teal eyes, wearing '
    'a navy strapless dress with gold filigree and a gold satin sash with a white flower, standing still '
    'with her arms relaxed, as the camera circles slowly around her. Clean confident lineart, flat cel '
    'shading with one crisp shadow tone, consistent character design, plain flat green background.'
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--depth', required=True)
    p.add_argument('--ref', required=True)
    p.add_argument('--out', default='film')
    p.add_argument('--steps', type=int, default=4)
    p.add_argument('--size', default='720x1280')
    p.add_argument('--chunk', type=int, default=81)
    p.add_argument('--overlap', type=int, default=17)
    p.add_argument('--shift', type=float, default=5.0)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--only', type=int, help='make just this chunk (a test)')
    p.add_argument('--seed-frames', help='finished frames (a video) that open the film: held fixed, so this film continues that one')
    p.add_argument('--fp8', action='store_true', help='store the transformer weights in fp8 (half the memory)')
    a = p.parse_args()
    w, h = map(int, a.size.split('x'))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    depth = [Image.fromarray(f).convert('RGB').resize((w, h), Image.BICUBIC) for f in iio.imread(a.depth)]
    n = len(depth)
    ref = Image.open(a.ref).convert('RGB').resize((w, h), Image.LANCZOS)
    stride = a.chunk - a.overlap
    starts = sorted({min(i * stride, n - a.chunk) for i in range(-(-(n - a.overlap) // stride))})
    print(f'{n} frames, {len(starts)} chunks at {starts}', flush=True)

    t0 = time.time()
    transformer = WanVACETransformer3DModel.from_pretrained(MODEL, subfolder='transformer', torch_dtype=torch.bfloat16, device_map='cuda')
    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder='vae', torch_dtype=torch.float32)
    pipe = WanVACEPipeline.from_pretrained(MODEL, transformer=transformer, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=a.shift)
    pipe.load_lora_weights(hf_hub_download(*LORA), adapter_name='lightx2v')
    pipe.fuse_lora(lora_scale=1.0)
    pipe.unload_lora_weights()
    pipe.text_encoder.to('cuda')
    pipe.vae.to('cuda')
    with torch.no_grad():
        embeds, _ = pipe.encode_prompt(PROMPT, do_classifier_free_guidance=False, device='cuda')
    # The text encoder is done with: drop it so the pipeline runs (and places tensors) on the GPU.
    pipe.register_modules(text_encoder=None)
    gc.collect()
    torch.cuda.empty_cache()
    # 720p frames through the fp32 VAE don't fit beside the 14B transformer in one piece.
    pipe.vae.enable_tiling()
    if a.fp8:
        pipe.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
        torch.cuda.empty_cache()
    print(f'loaded in {time.time() - t0:.0f}s, {torch.cuda.memory_allocated() / 1e9:.1f} GB on GPU', flush=True)

    frames = [None] * n  # finished frames, uint8 arrays
    if a.seed_frames:
        for j, f in enumerate(iio.imread(a.seed_frames)):
            frames[j] = np.asarray(Image.fromarray(f).convert('RGB').resize((w, h), Image.LANCZOS))
    black, white = Image.new('L', (w, h), 0), Image.new('L', (w, h), 255)
    for k, s in enumerate(starts):
        path = out / f'chunk_{k:02d}.mp4'
        if path.exists():
            for j, f in enumerate(iio.imread(path)):
                frames[s + j] = f
            print(f'chunk {k}: reused {path}', flush=True)
            continue
        if a.only is not None and k != a.only:
            continue
        keep = sum(frames[s + j] is not None for j in range(a.chunk))
        video = [Image.fromarray(frames[s + j]) if j < keep else depth[s + j] for j in range(a.chunk)]
        mask = [black if j < keep else white for j in range(a.chunk)]
        t = time.time()
        result = pipe(
            video=video, mask=mask, reference_images=[ref], prompt_embeds=embeds,
            height=h, width=w, num_frames=a.chunk, num_inference_steps=a.steps, guidance_scale=1.0,
            generator=torch.Generator('cuda').manual_seed(a.seed + k), output_type='np',
        ).frames[0]
        chunk = (np.clip(result, 0, 1) * 255).round().astype(np.uint8)
        for j in range(keep, a.chunk):
            frames[s + j] = chunk[j]
        export_to_video([c / 255.0 for c in chunk], str(path), fps=16)
        print(f'chunk {k}: frames {s}-{s + a.chunk - 1}, kept {keep}, {time.time() - t:.0f}s', flush=True)

    if all(f is not None for f in frames):
        iio.imwrite(out / 'film.mp4', np.stack(frames), fps=16, codec='libx264', quality=9)
        print(f'film: {n} frames -> {out}/film.mp4', flush=True)


if __name__ == '__main__':
    main()
