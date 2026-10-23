// CPU mode: the nanotss shader (coverage/shader/nts: trained on the CPU with the nanotss engine) run in
// WebAssembly. The main thread turns the G-buffer tiles into the network's inputs (decode and the
// neural texture, infer.c nts_prepare); Web Workers each run the network on a band of rows
// (nts-worker.js), the band plus the network's reach on both sides, so the bands join seamlessly.
// infer-relaxed.wasm (8-bit dot products) where the browser has relaxed SIMD, else infer.wasm.

export async function createCpuShader({ base = 'nts/', maxPixels, threads }) {
  const get = async f => { const r = await fetch(base + f); if (!r.ok) throw new Error(`${base}${f}: ${r.status}`); return r; };
  let module = null, relaxed = true;
  try { module = await WebAssembly.compile(await (await get('infer-relaxed.wasm')).arrayBuffer()); }
  catch { relaxed = false; module = await WebAssembly.compile(await (await get('infer.wasm')).arrayBuffer()); }
  const [meta, net] = await Promise.all([get('meta.json').then(r => r.json()), get('net.bin').then(r => r.arrayBuffer())]);

  // The main thread's instance: inputs and the neural texture.
  const k = (await WebAssembly.instantiate(module)).exports;
  const put = bytes => { const p = k.nts_alloc(bytes.byteLength); new Uint8Array(k.memory.buffer).set(new Uint8Array(bytes), p); return p; };
  if (!k.nts_load(put(net), net.byteLength)) throw new Error('nts/net.bin did not load');
  // The fine texture: fine0..2.png hold channels 3k..3k+2, part p in tile (p % cols, p / cols).
  const P = meta.parts.length, T = meta.tile;
  const fineP = k.nts_alloc(P * T * T * 8);
  const imgs = await Promise.all([0, 1, 2].map(async i => {
    const bmp = await createImageBitmap(await (await get(`fine${i}.png`)).blob(), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
    const c = new OffscreenCanvas(bmp.width, bmp.height), ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bmp, 0, 0);
    return ctx.getImageData(0, 0, bmp.width, bmp.height);
  }));
  const fine = new Uint8Array(k.memory.buffer, fineP, P * T * T * 8);
  for (let p = 0; p < P; p++) {
    const x0 = (p % meta.cols) * T, y0 = Math.floor(p / meta.cols) * T;
    for (let ch = 0; ch < 8; ch++) {
      const src = imgs[ch / 3 | 0], W = src.width, d = src.data, j = ch % 3;
      for (let y = 0; y < T; y++) {
        let s = ((y0 + y) * W + x0) * 4 + j, o = ((p * T + y) * T) * 8 + ch;
        for (let x = 0; x < T; x++, s += 4, o += 8) fine[o] = d[s];
      }
    }
  }
  k.nts_set_fine(fineP);
  const tilesP = k.nts_alloc(maxPixels * 20), inP = k.nts_alloc(maxPixels * 72);
  const reach = k.nts_reach();

  const n = Math.max(1, Math.min(8, threads ?? (navigator.hardwareConcurrency || 4) - 1));
  const workers = await Promise.all(Array.from({ length: n }, () => new Promise((resolve, reject) => {
    const w = new Worker(new URL('./nts-worker.js', import.meta.url));
    w.onmessage = e => { if (e.data.ready) resolve(w); };
    w.onerror = reject;
    w.postMessage({ init: true, module, net }, []);
  })));

  let busy = false, ms = null;
  // tiles: the five G-buffer tiles (colour, normal, depth, uv-a, uv-b), each w x h RGBA, rows
  // bottom-up as readPixels gives them. Resolves to straight RGBA, w x h, rows top-down.
  async function frame({ tiles, w, h, pitch }) {
    busy = true;
    const t0 = performance.now();
    new Uint8Array(k.memory.buffer, tilesP, tiles.length).set(tiles);
    k.nts_prepare(tilesP, w, h, Math.sin(pitch), Math.cos(pitch), inP);
    const all = new Float32Array(k.memory.buffer, inP, w * h * 18);
    // Bands of equal height; each worker gets its band's inputs plus the reach above and below.
    const out = new Uint8ClampedArray(w * h * 4);
    await Promise.all(workers.map((wk, i) => new Promise(resolve => {
      const r0 = Math.round(h * i / n), r1 = Math.round(h * (i + 1) / n);
      if (r1 <= r0) { resolve(); return; }
      const a = Math.max(0, r0 - reach), b = Math.min(h, r1 + reach);
      const input = all.slice(a * w * 18, b * w * 18);
      wk.onmessage = e => { if (e.data.error) console.error('[cpu worker]', e.data.error); else out.set(new Uint8Array(e.data.rgba), r0 * w * 4); resolve(); };
      wk.postMessage({ input, w, h: b - a, r0: r0 - a, r1: r1 - a }, [input.buffer]);
    })));
    const dt = performance.now() - t0;
    if (globalThis.cpuLog) console.log(`[cpu] ${w}x${h} in ${dt.toFixed(0)} ms`);
    ms = ms == null ? dt : ms * 0.8 + dt * 0.2;
    busy = false;
    return out;
  }
  // For checks: the fine texture as decoded here, sampled every 4099th byte.
  const fineSample = () => Array.from({ length: 2048 }, (_, i) => new Uint8Array(k.memory.buffer, fineP, P * T * T * 8)[i * 4099]);
  return { frame, fineSample, parts: meta.parts, threads: n, relaxed, get busy() { return busy; }, get ms() { return ms; } };
}
