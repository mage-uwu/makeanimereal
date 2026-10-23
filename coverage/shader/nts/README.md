# Shader training on nanotss (CPU)

`shader_nts.c` trains the neural shader with the nanotss engine (`nanotss_blt.c`, included with
`NANOTSS_NO_MAIN`): neural texture (fine 512² + coarse 64², 8 channels, 18 parts) → dense embed →
8 blocks of the engine's ternary stencil mixer (`tc_*`, dilated 3×3) + ternary FFN (int8 VNNI) → RGBA.
Muon on the ternary FFN masters, Adam on dense parameters, sparse Adam on the texture.

    python3 pack_data.py ../../../out/shader-data/w1 ../../../out/shader-data/bottom --out all.bin
    cc -O3 -march=native -fopenmp shader_nts.c -lm -o shader_nts
    ./shader_nts --check                       # gradient check
    ./shader_nts --data all.bin --seconds 2400 --batch 16 --crop 96 --threads 16 --profile

Measured on a 16-vCPU EPYC 4564P (Zen 4): ~0.5 M training px/s after tiling the FFNs (from ~0.3 M),
vs ~7.8 M px/s for the PyTorch U-Net on an A6000. Training is memory-bandwidth bound: per-token
kernels are fast in isolation, but full-resolution activations for backward stream through DRAM.

## In the viewer (CPU mode, `#cpu`)

    ./shader_nts --data held.bin --load model.ntsw --iters 0 --eval-all --tex8   # held-out L1, 8-bit texture
    python3 export_nts.py model.ntsw ../../live/nts --parts ../../../out/shader-data/w1/meta.json
    clang --target=wasm32 -O3 -msimd128 -mbulk-memory -nostdlib -Wl,--no-entry -Wl,--export-dynamic \
          -Wl,--initial-memory=33554432 -o ../../live/nts/infer.wasm infer.c        # and -mrelaxed-simd -o infer-relaxed.wasm
    ./shader_nts --data held.bin --load model.ntsw --iters 0 --eval-all --tex8 --ref ref8.bin
    node check_infer.mjs ref8.bin fine.raw 4                                         # the WASM kernel against the trainer

`infer.c` is the forward pass in freestanding C for WebAssembly SIMD (relaxed SIMD's 8-bit dot products
where the browser has them). `coverage/live/nts.js` decodes the G-buffer and samples the texture on the
main thread; Web Workers (`nts-worker.js`) each run a band of rows plus the network's 38-pixel reach.
Background skipping: a token only runs where it can differ from the background and still reach a pixel
near her, which is about a third of the work and gives the same drawing (check_infer.mjs compares both).

run1 (40 minutes on 16 vCPUs): held-out L1 0.0114 over all 144 held-out frames, the same with the
8-bit texture. The browser kernel matches the trainer to a mean 2e-5 over white.
