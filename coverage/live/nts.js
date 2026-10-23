// CPU mode: the nanotss shader (coverage/shader/nts: trained on the CPU with the nanotss engine) run in
// WebAssembly. The main thread turns the G-buffer tiles into the network's inputs (decode and the
// neural texture, infer.c nts_prepare); Web Workers each run the network on a band of rows
// (nts-worker.js), the band plus the network's reach on both sides, so the bands join seamlessly.
// infer-relaxed.wasm (8-bit dot products) where the browser has relaxed SIMD, else infer.wasm.

// net.png: export_nts.py's net.bin packed 3 bytes a pixel into an opaque RGB PNG after a 4-byte length.
async function unpackPng(blob) {
  const bmp = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  const c = new OffscreenCanvas(bmp.width, bmp.height), ctx = c.getContext('2d');
  ctx.drawImage(bmp, 0, 0);
  const px = ctx.getImageData(0, 0, bmp.width, bmp.height).data, rgb = new Uint8Array(bmp.width * bmp.height * 3);
  for (let i = 0, j = 0; i < px.length; i += 4) { rgb[j++] = px[i]; rgb[j++] = px[i + 1]; rgb[j++] = px[i + 2]; }
  const n = rgb[0] | (rgb[1] << 8) | (rgb[2] << 16) | (rgb[3] << 24);
  return rgb.buffer.slice(4, 4 + n);
}

export async function createCpuShader({ base = 'nts/', maxPixels, threads }) {
  const get = async f => { const r = await fetch(base + f); if (!r.ok) throw new Error(`${base}${f}: ${r.status}`); return r; };
  let module = null, relaxed = true;
  try { module = await WebAssembly.compile(await (await get('infer-relaxed.wasm')).arrayBuffer()); }
  catch { relaxed = false; module = await WebAssembly.compile(await (await get('infer.wasm')).arrayBuffer()); }
  const [meta, net] = await Promise.all([get('meta.json').then(r => r.json()), get('net.png').then(r => r.blob()).then(unpackPng)]);

  // The main thread's instance: inputs and the neural texture.
  const k = (await WebAssembly.instantiate(module)).exports;
  const put = bytes => { const p = k.nts_alloc(bytes.byteLength); new Uint8Array(k.memory.buffer).set(new Uint8Array(bytes), p); return p; };
  if (!k.nts_load(put(net), net.byteLength)) throw new Error('nts/net.png did not load');
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

  // Buffers go to the workers and come back (transferred, never copied or dropped), and the output
  // array is kept: no per-frame garbage, which on phones turns into collector pauses.
  const bufs = workers.map(() => ({ input: new ArrayBuffer(0), rgba: new ArrayBuffer(0) }));
  let out = new Uint8ClampedArray(0);
  let busy = false, ms = null;
  // Band cuts as fractions of the crop's height. Background skipping makes a band's cost follow how
  // much of her it holds, so equal heights leave workers idle while one finishes (1.5x slower on a
  // standing frame). The first split puts equal shares of her pixels in each band; after that each
  // frame's band times move the cuts toward equal time (half-way, to stay steady).
  let cuts = null;
  function firstCuts(all, w, h) {
    const cum = new Float64Array(h + 1);
    for (let r = 0; r < h; r++) { let m = w * 0.02; for (let c = 0; c < w; c++) m += all[(r * w + c) * 18 + 7]; cum[r + 1] = cum[r] + m; }
    return Array.from({ length: n + 1 }, (_, i) => { let r = 0; while (r < h && cum[r + 1] < cum[h] * i / n) r++; return i === n ? 1 : r / h; });
  }
  function adjust(times) {
    // cost spread evenly over each band's rows, then cut at equal shares of the total
    const total = times.reduce((a, b) => a + b, 0); if (!(total > 0)) return;
    const next = [0]; let band = 0, acc = 0;
    for (let i = 1; i < n; i++) {
      const goal = total * i / n;
      while (band < n - 1 && acc + times[band] < goal) acc += times[band++];
      const f = times[band] > 0 ? (goal - acc) / times[band] : 0;
      next.push(cuts[band] + f * (cuts[band + 1] - cuts[band]));
    }
    next.push(1);
    cuts = cuts.map((c, i) => (c + next[i]) / 2);
  }
  // tiles: the five G-buffer tiles (colour, normal, depth, uv-a, uv-b), each w x h RGBA, rows
  // bottom-up as readPixels gives them. Resolves to straight RGBA, w x h, rows top-down (valid until
  // the next frame).
  async function frame({ tiles, w, h, pitch }) {
    busy = true;
    const t0 = performance.now();
    new Uint8Array(k.memory.buffer, tilesP, tiles.length).set(tiles);
    k.nts_prepare(tilesP, w, h, Math.sin(pitch), Math.cos(pitch), inP);
    const all = new Float32Array(k.memory.buffer, inP, w * h * 18);
    if (out.length < w * h * 4) out = new Uint8ClampedArray(Math.ceil(w * h * 4 * 1.3));
    if (!cuts) cuts = firstCuts(all, w, h);
    const times = new Array(n).fill(0), rows = cuts.map(c => Math.round(c * h));
    // Each worker gets its band's inputs plus the reach above and below.
    await Promise.all(workers.map((wk, i) => new Promise(resolve => {
      const r0 = rows[i], r1 = rows[i + 1];
      if (r1 <= r0) { resolve(); return; }
      const a = Math.max(0, r0 - reach), b = Math.min(h, r1 + reach), len = (b - a) * w * 18, B = bufs[i];
      if (B.input.byteLength < len * 4) B.input = new ArrayBuffer(Math.ceil(len * 4 * 1.3));
      new Float32Array(B.input, 0, len).set(all.subarray(a * w * 18, b * w * 18));
      wk.onmessage = e => {
        const d = e.data;
        if (d.error) console.error('[cpu worker]', d.error);
        else out.set(new Uint8Array(d.rgba, 0, (r1 - r0) * w * 4), r0 * w * 4);
        if (d.input) { B.input = d.input; B.rgba = d.rgba; times[i] = d.ms; }
        resolve();
      };
      wk.postMessage({ input: B.input, rgba: B.rgba, w, h: b - a, r0: r0 - a, r1: r1 - a }, [B.input, B.rgba]);
    })));
    adjust(times);
    const dt = performance.now() - t0;
    if (globalThis.cpuLog) console.log(`[cpu] ${w}x${h} in ${dt.toFixed(0)} ms, bands ${times.map(t => t.toFixed(0)).join('/')} ms`);
    ms = ms == null ? dt : ms * 0.8 + dt * 0.2;
    busy = false;
    return out.subarray(0, w * h * 4);
  }
  // For checks: the fine texture as decoded here, sampled every 4099th byte.
  const fineSample = () => Array.from({ length: 2048 }, (_, i) => new Uint8Array(k.memory.buffer, fineP, P * T * T * 8)[i * 4099]);
  return { frame, fineSample, parts: meta.parts, threads: n, relaxed, get busy() { return busy; }, get ms() { return ms; } };
}
