#!/bin/sh
# Build the vrmanet runtime to WebAssembly (no libc, no emscripten): web/vrmanet.wasm.
set -e
cd "$(dirname "$0")/.."
clang --target=wasm32 -O3 -nostdlib -fvisibility=hidden \
  -Wl,--no-entry -Wl,--export-dynamic -Wl,--initial-memory=54525952 -Wl,--strip-all \
  vrmanet/vrmanet.c -o web/vrmanet.wasm
echo "web/vrmanet.wasm: $(wc -c < web/vrmanet.wasm) bytes"
