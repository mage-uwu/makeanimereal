#!/bin/sh
# Build the NPMP layers to WebAssembly with 128-bit SIMD (no libc, no emscripten): web/npmp.wasm.
set -e
cd "$(dirname "$0")/.."
clang --target=wasm32 -O3 -msimd128 -nostdlib -fvisibility=hidden \
  -Wl,--no-entry -Wl,--export-dynamic -Wl,--initial-memory=27262976 -Wl,--strip-all \
  mocapact/npmp.c -o web/npmp.wasm
echo "web/npmp.wasm: $(wc -c < web/npmp.wasm) bytes"
