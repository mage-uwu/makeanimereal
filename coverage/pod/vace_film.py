#!/usr/bin/env python3
"""Restyle the whole coverage film with Wan2.1 VACE 14B, on the GPU pod.

    python3 vace_film.py --depth depth.mp4 --ref style-ref.png --out film [--steps 4] [--size 720x1280]
                         [--chunk 81] [--overlap 17] [--only 0]

With --anchors (keyframes drawn independently, every 80 frames) and --overlap 1, each chunk is
pinned between two anchors and nothing generated is ever fed forward, so the look cannot drift;
this is the way to run it. Without anchors, the film is made in chunks
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

import cv2
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


class Cleaner:
    """Stops drift in the frames a chunk carries over from the last one. Chained chunks feed their
    own output forward, so small errors in colour and background grow from chunk to chunk. Before a
    frame is carried, its background (outside the rendered silhouette) goes back to the clean green
    of the calibration frames, and each part of her (found by clustering the raw render's colours:
    dress, sash, skin, hair...) goes back to the colour that part had in the calibration frames.
    The correction is a smooth per-pixel offset, so the drawing's own lines and shading stay."""

    K = 12

    def __init__(self, frames, render, depth, calib):
        self.render, self.depth = render, depth
        px_r, px_s, bg = [], [], []
        for i in calib:
            m = self.mask(i, 0)
            px_r.append(self.lab(render[i])[m][::7])
            px_s.append(self.lab(frames[i])[m][::7])
            bg.append(frames[i][~self.mask(i, 25)])
        px_r, px_s = np.concatenate(px_r), np.concatenate(px_s)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        _, labels, self.centers = cv2.kmeans(px_r, self.K, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        labels = labels.ravel()
        self.want = np.stack([np.median(px_s[labels == k], 0) for k in range(self.K)])
        self.bg = np.median(np.concatenate(bg), 0).astype(np.uint8)
        print(f'cleaning: {self.K} parts, background {self.bg}', flush=True)

    @staticmethod
    def lab(rgb):
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    def mask(self, i, grow=9):
        m = (np.asarray(self.depth[i])[..., 0] > 6).astype(np.uint8)
        return cv2.dilate(m, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0 if grow else m > 0

    def __call__(self, frame, i):
        m0, m = self.mask(i, 0), self.mask(i)
        lr, ls = self.lab(self.render[i]), self.lab(frame)
        part = np.argmin(((lr[m0][:, None, :] - self.centers[None]) ** 2).sum(-1), 1)
        # Per part: how far this frame's colour has wandered from the calibrated colour.
        offset = np.zeros_like(ls)
        weight = np.zeros(ls.shape[:2], np.float32)
        cur = ls[m0]
        for k in range(self.K):
            sel = part == k
            if sel.sum() < 50:
                continue
            d = self.want[k] - np.median(cur[sel], 0)
            idx = np.nonzero(m0)
            offset[idx[0][sel], idx[1][sel]] = d
            weight[idx[0][sel], idx[1][sel]] = 1
        blur = lambda x: cv2.GaussianBlur(x, (0, 0), 6)
        wsm = blur(weight)[..., None]
        offset = blur(offset) / np.maximum(wsm, 1e-3)
        ls = ls + offset * np.clip(wsm * 2, 0, 1)
        out = cv2.cvtColor(ls.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
        # A touch of softening, so the model's sharpening can't compound through the chain either.
        out = cv2.GaussianBlur(out, (0, 0), 0.6)
        soft = cv2.GaussianBlur(m.astype(np.float32), (7, 7), 0)[..., None]
        return (out * soft + self.bg * (1 - soft)).astype(np.uint8)


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
    p.add_argument('--only', help='make just these chunks, e.g. 3 or 0,1,2 (a test, or a film whose render is still being drawn)')
    p.add_argument('--seed-frames', help='finished frames (a video) that open the film: held fixed, so this film continues that one')
    p.add_argument('--anchors', help='a folder of finished keyframes NNNN.png: held fixed wherever they fall (use with --overlap 1)')
    p.add_argument('--render', help='the raw render of this film (a video): turns on cleaning of the carried-over frames')
    p.add_argument('--calib', default='0:80', help='finished frames a:b that set the look the cleaning holds to')
    p.add_argument('--prompt', help='the character, in place of the built-in one (magical_girl_02): '
                   'what she looks like and wears, e.g. "a girl with long pink hair and white rabbit ears, wearing ..."')
    p.add_argument('--fp8', action='store_true', help='store the transformer weights in fp8 (half the memory)')
    a = p.parse_args()
    w, h = map(int, a.size.split('x'))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prompt = PROMPT if not a.prompt else (
        f'Hand-drawn 2D anime cel animation of {a.prompt}, standing still with her arms relaxed, as the camera '
        'circles slowly around her. Clean confident lineart, flat cel shading with one crisp shadow tone, '
        'consistent character design, plain flat green background.')

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
        embeds, _ = pipe.encode_prompt(prompt, do_classifier_free_guidance=False, device='cuda')
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

    render = [np.asarray(Image.fromarray(f).convert('RGB').resize((w, h), Image.BICUBIC)) for f in iio.imread(a.render)] if a.render else None
    clean = None
    frames = [None] * n  # finished frames, uint8 arrays
    if a.seed_frames:
        for j, f in enumerate(iio.imread(a.seed_frames)):
            frames[j] = np.asarray(Image.fromarray(f).convert('RGB').resize((w, h), Image.LANCZOS))
    if a.anchors:
        for f in sorted(Path(a.anchors).glob('*.png')):
            if int(f.stem) < n:
                frames[int(f.stem)] = np.asarray(Image.open(f).convert('RGB').resize((w, h), Image.LANCZOS))
        print(f'anchors at {[i for i in range(n) if frames[i] is not None]}', flush=True)
    black, white = Image.new('L', (w, h), 0), Image.new('L', (w, h), 255)
    for k, s in enumerate(starts):
        path = out / f'chunk_{k:02d}.mp4'
        if path.exists():
            for j, f in enumerate(iio.imread(path)):
                frames[s + j] = f
            print(f'chunk {k}: reused {path}', flush=True)
            continue
        if a.only is not None and k not in {int(x) for x in a.only.split(',')}:
            continue
        # Every frame of this window that is already finished (anchors, the last chunk's frames) is
        # held fixed; the rest is generated from the rendered depth.
        held = [frames[s + j] is not None for j in range(a.chunk)]
        keep = sum(held)
        if render and clean is None:
            c0, c1 = map(int, a.calib.split(':'))
            clean = Cleaner(frames, render, depth, range(c0, c1 + 1))
        video = [Image.fromarray(clean(frames[s + j], s + j) if clean else frames[s + j]) if held[j] else depth[s + j] for j in range(a.chunk)]
        mask = [black if held[j] else white for j in range(a.chunk)]
        t = time.time()
        result = pipe(
            video=video, mask=mask, reference_images=[ref], prompt_embeds=embeds,
            height=h, width=w, num_frames=a.chunk, num_inference_steps=a.steps, guidance_scale=1.0,
            generator=torch.Generator('cuda').manual_seed(a.seed + k), output_type='np',
        ).frames[0]
        chunk = (np.clip(result, 0, 1) * 255).round().astype(np.uint8)
        for j in range(a.chunk):
            if not held[j]:
                frames[s + j] = chunk[j]
        export_to_video([c / 255.0 for c in chunk], str(path), fps=16)
        print(f'chunk {k}: frames {s}-{s + a.chunk - 1}, kept {keep}, {time.time() - t:.0f}s', flush=True)

    if all(f is not None for f in frames):
        iio.imwrite(out / 'film.mp4', np.stack(frames), fps=16, codec='libx264', quality=9)
        print(f'film: {n} frames -> {out}/film.mp4', flush=True)


if __name__ == '__main__':
    main()
