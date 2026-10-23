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
