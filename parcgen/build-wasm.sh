#!/bin/sh
# Build the parcgen runtime to WebAssembly with 128-bit SIMD (no libc, no emscripten): web/parcgen.wasm.
set -e
cd "$(dirname "$0")/.."
clang --target=wasm32 -O3 -msimd128 -nostdlib -fvisibility=hidden \
  -Wl,--no-entry -Wl,--export-dynamic -Wl,--initial-memory=50331648 -Wl,--strip-all \
  parcgen/parcgen.c -o web/parcgen.wasm
echo "web/parcgen.wasm: $(wc -c < web/parcgen.wasm) bytes"
