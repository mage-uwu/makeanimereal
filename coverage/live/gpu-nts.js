// WebGPU compute path for the nanotss shader: infer.c in WGSL, the same frame({ tiles, w, h, pitch })
// -> RGBA interface as nts.js (createCpuShader), which it falls back to when WebGPU is missing (or the
// net is one it doesn't run: the ink branch, NTS2).
//
// Timing: ms (wall, upload to readback, smoothed) and gpuMs (the passes, timestamp queries where the
// device has them). On SwiftShader (no GPU) a 360 x 640 frame takes seconds; that's CPU emulation.
// Inputs: the five G-buffer tiles (colour, normal, depth, uv-a, uv-b), each w x h RGBA bytes with rows
// bottom-up as readPixels gives them, and the camera pitch (radians). Output: straight RGBA bytes, w x h,
// rows top-down (valid until the next frame). The same net files as nts.js: net.png, fine0..2.png, meta.json.
//
// Per frame, compute passes in order (infer.c's run(), background skipping included):
//   prepare   decode + the neural texture: 18 floats a pixel (nts_prepare)
//   mask      which pixels' inputs differ from pixel 0's (the background, by construction)
//   dist ×2   chessboard distance to her, capped past the reach any block needs
//   chain     the background's state after each block (one invocation: token() on itself)
//   embed     18 -> 64 for her pixels
//   block ×NB the stencil mixer, then the dense FFN (int8 codes x ternary rows, dot4I8Packed: the same
//             integer sums) or the trees' walk, on pixels within the block's reach of her
//   head      64 -> 4, sigmoid, bytes; the background's colour beyond the margin
// Activations are float32, channel-major (adjacent pixels adjacent in memory). Float steps follow
// infer.c's order and its maths (exp4, Abramowitz-Stegun erf); WGSL's division is not correctly
// rounded, so an int8 rounding can rarely land the other way: within check_infer.mjs's tolerance.
import { createCpuShader } from './nts.js';

export const DW = 64, KT = 9, CIN = 18, FCH = 8, FINE = 512, COARSE = 64, MARGIN = 6;

export async function unpackPng(blob) {   // net.png: the bytes 3 a pixel after a 4-byte length (as nts.js)
  const bmp = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  const c = new OffscreenCanvas(bmp.width, bmp.height), ctx = c.getContext('2d');
  ctx.drawImage(bmp, 0, 0);
  const px = ctx.getImageData(0, 0, bmp.width, bmp.height).data, rgb = new Uint8Array(bmp.width * bmp.height * 3);
  for (let i = 0, j = 0; i < px.length; i += 4) { rgb[j++] = px[i]; rgb[j++] = px[i + 1]; rgb[j++] = px[i + 2]; }
  const n = rgb[0] | (rgb[1] << 8) | (rgb[2] << 16) | (rgb[3] << 24);
  return rgb.buffer.slice(4, 4 + n);
}

// net.bin (export_nts.py): NTSV (dense FFNs) or NTS3 (trees).
export function parseNet(buf) {
  const dv = new DataView(buf); let o = 4;
  const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 4));
  if (magic !== 'NTSV' && magic !== 'NTS3') throw new Error(`gpu-nts: ${magic} nets run on the CPU`);
  const i32 = () => { const v = dv.getInt32(o, true); o += 4; return v; };
  const f32 = n => { const a = new Float32Array(buf.slice(o, o + 4 * n)); o += 4 * n; return a; };
  const i8 = n => { const a = new Int8Array(buf.slice(o, o + n)); o += n; return a; };
  const NB = i32(), P = i32(), T = magic === 'NTS3' ? i32() : 0, V = T ? i32() : 0, TN = T ? (1 << V) - 1 : 0;
  const DIL = Array.from({ length: NB }, i32);
  const We = f32(DW * CIN), be = f32(DW), blocks = [];
  for (let b = 0; b < NB; b++) blocks.push(T
    ? { q: i8((KT + 4) * DW), gate: f32(DW), b2: f32(DW), bias: f32(T * TN), sin: f32(T * TN), sout: f32(T * TN), cin: i8(T * TN * DW), cout: i8(T * TN * DW) }
    : { q: i8((KT + 4) * DW), gate: f32(DW), b1: f32(DW), b2: f32(DW), s1: f32(DW), s2: f32(DW), c1: i8(DW * DW), c2: i8(DW * DW) });
  const Wh = f32(4 * DW), bh = f32(4), lo = f32(2 * FCH), st = f32(2 * FCH);
  const coarse = new Uint8Array(buf.slice(o, o + P * COARSE * COARSE * FCH)); o += P * COARSE * COARSE * FCH;
  if (o !== buf.byteLength) throw new Error('gpu-nts: net.bin has the wrong length');
  return { NB, P, T, V, TN, DIL, We, be, blocks, Wh, bh, lo, st, coarse };
}

// The parameters: one float array, one array of int8 rows packed 4 to a u32, with each block's offsets.
export function layout(net) {
  const { NB, T, TN } = net, f = [], u = [], fo = {}, uo = [];
  const put = (name, a) => { fo[name] = f.length; for (const v of a) f.push(v); };
  put('We', net.We); put('be', net.be); put('Wh', net.Wh); put('bh', net.bh); put('lo', net.lo); put('st', net.st);
  const pack = c => { for (let i = 0; i < c.length; i += 4) u.push(((c[i] & 255) | ((c[i + 1] & 255) << 8) | ((c[i + 2] & 255) << 16) | ((c[i + 3] & 255) << 24)) >>> 0); };
  const blk = [];
  net.blocks.forEach(k => {
    const o = { q: f.length }; for (const v of k.q) f.push(v);
    o.gate = f.length; for (const v of k.gate) f.push(v);
    o.b2 = f.length; for (const v of k.b2) f.push(v);
    o.w1 = u.length;
    if (T) {
      o.bias = f.length; for (const v of k.bias) f.push(v);
      o.sin = f.length; for (const v of k.sin) f.push(v);
      o.wout = f.length; for (let r = 0; r < T * TN; r++) for (let i = 0; i < DW; i++) f.push(Math.fround(k.cout[r * DW + i] * k.sout[r]));
      o.sout = f.length; for (const v of k.sout) f.push(v);
      pack(k.cin);
      // w_out as ternary bitmasks (fused-nts.js): per row the +1 channels (2 words), then the -1 ones
      o.wob = u.length;
      for (let r = 0; r < T * TN; r++) {
        const m = [0, 0, 0, 0];
        for (let i = 0; i < DW; i++) { const c = k.cout[r * DW + i]; if (c) m[(c < 0 ? 2 : 0) + (i >> 5)] |= 1 << (i & 31); }
        for (const w of m) u.push(w >>> 0);
      }
    } else {
      o.b1 = f.length; for (const v of k.b1) f.push(v);
      o.s1 = f.length; for (const v of k.s1) f.push(v);
      o.s2 = f.length; for (const v of k.s2) f.push(v);
      pack(k.c1); o.w2 = u.length; pack(k.c2);
    }
    blk.push(o);
  });
  return { f: new Float32Array(f), u: new Uint32Array(u), fo, blk };
}

// ---------------- WGSL ----------------
export function wgsl(net, L, packed) {
  const { NB, T, V, TN } = net, B = L.blk;
  const arr = (name, key) => `const ${name} = array<u32, ${NB}>(${B.map(o => `${o[key] ?? 0}u`).join(', ')});`;
  // dot4I8Packed / pack4xI8 where the language feature is there, else the same sums by hand
  const dot = packed ? '' : `fn dot4I8Packed(a: u32, b: u32) -> i32 {
  var s = 0; for (var j = 0u; j < 32u; j += 8u) { s += extractBits(i32(a), j, 8u) * extractBits(i32(b), j, 8u); } return s; }
fn pack4xI8(v: vec4<i32>) -> u32 { return (u32(v.x) & 255u) | ((u32(v.y) & 255u) << 8u) | ((u32(v.z) & 255u) << 16u) | ((u32(v.w) & 255u) << 24u); }`;
  return `
struct Frame { w: u32, h: u32, np: u32, parts: u32, sinp: f32, cosp: f32, pad0: u32, pad1: u32 };
struct Blk { b: u32, dil: u32, need: u32, prev: u32 };
${arr('QB', 'q')} ${arr('GATE', 'gate')} ${arr('B1', 'b1')} ${arr('B2', 'b2')} ${arr('S1', 's1')} ${arr('S2', 's2')}
${arr('W1', 'w1')} ${arr('W2', 'w2')} ${arr('BIAS', 'bias')} ${arr('SIN', 'sin')} ${arr('WOUT', 'wout')}
const WE = ${L.fo.We}u; const BE = ${L.fo.be}u; const WH = ${L.fo.Wh}u; const BH = ${L.fo.bh}u; const LO = ${L.fo.lo}u; const ST = ${L.fo.st}u;
const MAGIC = 12582912.0;
// a / b correctly rounded (WGSL's own division may be 2.5 ulp off): one Newton correction of the quotient
fn dv(a: f32, b: f32) -> f32 { let q = a / b; return q + fma(-q, b, a) / b; }
${dot}
fn exp4(xi: f32) -> f32 {   // infer.c exp4: round by the magic constant, Cephes polynomial, 2^n by bits
  let x = max(xi, -87.0);
  let nr = x * 1.44269504 + MAGIC; let n = nr - MAGIC;
  var r = x - n * 0.693359375; r = r - n * -2.12194440e-4;
  var p = 1.9875691500e-4;
  p = p * r + 1.3981999507e-3; p = p * r + 8.3334519073e-3; p = p * r + 4.1665795894e-2;
  p = p * r + 1.6666665459e-1; p = p * r + 5.0000001201e-1;
  p = p * (r * r) + (r + 1.0);
  return p * bitcast<f32>((bitcast<i32>(nr) - 0x4B400000 + 127) << 23u);
}
fn gelu(v: f32) -> f32 {   // infer.c gelu4: Abramowitz-Stegun erf sharing one exp
  let z = abs(v * 0.70710678); let e = exp4((-0.5 * v) * v); let t = dv(1.0, z * 0.3275911 + 1.0);
  var p = 1.061405429; p = p * t - 1.453152027; p = p * t + 1.421413741; p = p * t - 0.284496736; p = p * t + 0.254829592;
  var er = 1.0 - (p * t) * e; if (v < 0.0) { er = -er; }
  return v * (0.5 * (1.0 + er));
}
`;
}
// The per-token network in parts: quant (quant16), token (the stencil, then the FFN or the trees),
// embed and head; each kernel takes the parts it calls.
export function wgslParts(net) {
  const { T, V, TN } = net;
  const ffn = T ? `
  // the trees: t + b2 + each visited node's gelu(s) w_out; s = integer sum x row scale x xs + bias
  var y: array<f32, 64>;
  for (var c = 0u; c < 64u; c++) { y[c] = t[c] + P[B2[b] + c]; }
  var a = array<u32, 4>(0u, 0u, 0u, 0u);
  for (var j = 0u; j < ${V}u; j++) {
    for (var tr = 0u; tr < ${T}u; tr++) {
      let row = tr * ${TN}u + a[tr];
      var sum = 0;
      for (var k = 0u; k < 16u; k++) { sum += dot4I8Packed(codes[k], U32[W1[b] + row * 16u + k]); }
      let sv = (f32(sum) * P[SIN[b] + row]) * xs + P[BIAS[b] + row];
      let g = gelu(sv);
      for (var c = 0u; c < 64u; c++) { y[c] = P[WOUT[b] + row * 64u + c] * g + y[c]; }
      a[tr] = 2u * a[tr] + 1u + select(0u, 1u, sv > 0.0);
    }
  }
  return y;` : `
  // FFN1: integer sums of int8 codes x ternary rows, then ((sum x row scale) x xs) + b1
  var g: array<f32, 64>;
  for (var r = 0u; r < 64u; r++) {
    var sum = 0;
    for (var k = 0u; k < 16u; k++) { sum += dot4I8Packed(codes[k], U32[W1[b] + r * 16u + k]); }
    g[r] = gelu((f32(sum) * P[S1[b] + r]) * xs + P[B1[b] + r]);
  }
  let xs2 = quant(&g, &codes);
  var y: array<f32, 64>;
  for (var r = 0u; r < 64u; r++) {
    var sum = 0;
    for (var k = 0u; k < 16u; k++) { sum += dot4I8Packed(codes[k], U32[W2[b] + r * 16u + k]); }
    y[r] = ((f32(sum) * P[S2[b] + r]) * xs2 + P[B2[b] + r]) + t[r];
  }
  return y;`;
  const quant = `
fn quant(v: ptr<function, array<f32, 64>>, codes: ptr<function, array<u32, 16>>) -> f32 {   // quant16
  var mx = 0.0;
  for (var c = 0u; c < 64u; c++) { mx = max(mx, abs((*v)[c])); }
  mx = max(mx, 1e-5);
  let sc = dv(127.0, mx);
  for (var k = 0u; k < 16u; k++) {
    (*codes)[k] = pack4xI8(vec4<i32>(i32(round((*v)[4u * k] * sc)), i32(round((*v)[4u * k + 1u] * sc)),
                                      i32(round((*v)[4u * k + 2u] * sc)), i32(round((*v)[4u * k + 3u] * sc))));
  }
  return dv(1.0, sc);
}`;
  const token = `
fn token(b: u32, taps: array<i32, 9>, ctr: i32) -> array<f32, 64> {
  // the ternary stencil: depthwise 3 x 3 (x 1/3), the 4-channel group mix (x 0.5), gate, residual
  var dw: array<f32, 64>;
  for (var c = 0u; c < 64u; c++) {
    var z = 0.0;
    for (var q = 0u; q < 9u; q++) { z = P[QB[b] + q * 64u + c] * rd(b, taps[q], c) + z; }
    dw[c] = z * 0.33333334;
  }
  var t: array<f32, 64>;
  for (var c = 0u; c < 64u; c++) {
    let g0 = (c / 4u) * 4u; var m = 0.0;
    for (var j = 0u; j < 4u; j++) { m = P[QB[b] + (9u + j) * 64u + c] * dw[g0 + j] + m; }
    t[c] = P[GATE[b] + c] * (m * 0.5) + rd(b, ctr, c);
  }
  var codes: array<u32, 16>;
  let xs = quant(&t, &codes);
  ${ffn}
}`;
  const embed = `
fn embed(f: array<f32, 18>) -> array<f32, 64> {
  var x: array<f32, 64>;
  for (var j = 0u; j < 64u; j++) {
    var s = 0.0;
    for (var i = 0u; i < 18u; i++) { s = P[WE + j * 18u + i] * f[i] + s; }
    x[j] = P[BE + j] + s;
  }
  return x;
}`;
  const head = `
fn head(v: array<f32, 64>) -> vec4<f32> {   // four lanes of partial sums, then (s0 + s1) + (s2 + s3), as infer.c
  var p: vec4<f32>;
  for (var k = 0u; k < 4u; k++) {
    var s = vec4<f32>(0.0);
    for (var i = 0u; i < 64u; i += 4u) {
      s = vec4<f32>(P[WH + k * 64u + i], P[WH + k * 64u + i + 1u], P[WH + k * 64u + i + 2u], P[WH + k * 64u + i + 3u])
          * vec4<f32>(v[i], v[i + 1u], v[i + 2u], v[i + 3u]) + s;
    }
    let z = P[BH + k] + ((s.x + s.y) + (s.z + s.w));
    p[k] = dv(1.0, 1.0 + exp4(min(-z, 88.0)));
  }
  return p;
}`;
  return { quant, token, embed, head };
}

// The neural texture as one byte array: fine0..2.png hold channels 3k..3k+2, part p at tile
// (p % cols, p / cols) -> [part][y][x][8] (nts.js), then the coarse level from net.bin.
export async function netTexture(get, meta, net) {
  const { P } = net;
  const T = meta.tile, imgs = await Promise.all([0, 1, 2].map(async i => {
    const bmp = await createImageBitmap(await (await get(`fine${i}.png`)).blob(), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
    const c = new OffscreenCanvas(bmp.width, bmp.height), ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bmp, 0, 0); return ctx.getImageData(0, 0, bmp.width, bmp.height);
  }));
  const fineN = P * T * T * FCH, coarseN = P * COARSE * COARSE * FCH;
  const texb = new Uint8Array(Math.ceil((fineN + coarseN) / 4) * 4);
  for (let p = 0; p < P; p++) {
    const x0 = (p % meta.cols) * T, y0 = Math.floor(p / meta.cols) * T;
    for (let ch = 0; ch < 8; ch++) {
      const src = imgs[ch / 3 | 0], W = src.width, d = src.data, j = ch % 3;
      for (let y = 0; y < T; y++) {
        let s = ((y0 + y) * W + x0) * 4 + j, o = ((p * T + y) * T) * 8 + ch;
        for (let x = 0; x < T; x++, s += 4, o += 8) texb[o] = d[s];
      }
    }
  }
  texb.set(net.coarse, fineN);
  return texb;
}

// stats: per frame, the crop, the pixels each pass computes (the distance buffer read back) and each
// pass's GPU time, in .stats (diagnostics; it costs a readback a frame).
export async function createGpuShader({ base = 'nts/', maxPixels, threads, fallback = true, stats = false } = {}) {
  try {
    if (!navigator.gpu) throw new Error('no WebGPU');
    return await create(base, maxPixels, stats);
  } catch (e) {
    if (!fallback) throw e;
    console.warn('[gpu-nts] using the CPU shader (nts.js):', e.message);
    const cpu = await createCpuShader({ base, maxPixels, threads });
    return Object.assign(cpu, { gpu: false });
  }
}

async function create(base, maxPixels, wantStats) {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter');
  const get = async f => { const r = await fetch(base + f); if (!r.ok) throw new Error(`${base}${f}: ${r.status}`); return r; };
  const [meta, raw] = await Promise.all([get('meta.json').then(r => r.json()), get('net.png').then(r => r.blob()).then(unpackPng)]);
  const net = parseNet(raw), { NB, P, DIL } = net;
  const texb = await netTexture(get, meta, net);
  const fineN = P * meta.tile * meta.tile * FCH;

  const packed = navigator.gpu.wgslLanguageFeatures?.has('packed_4x8_integer_dot_product');
  // the largest buffers: a frame's activations (64 floats a pixel) and the texture
  const big = Math.max(texb.byteLength, maxPixels * DW * 4);
  if (big > adapter.limits.maxStorageBufferBindingSize || big > adapter.limits.maxBufferSize) throw new Error(`needs ${big} byte buffers, this GPU allows ${adapter.limits.maxStorageBufferBindingSize}`);
  const ts = adapter.features.has('timestamp-query');   // the GPU's own time for the passes, where it tells
  const device = await adapter.requestDevice({
    requiredFeatures: ts ? ['timestamp-query'] : [],
    requiredLimits: { maxStorageBufferBindingSize: Math.max(big, 134217728), maxBufferSize: Math.max(big, 268435456), maxStorageBuffersPerShaderStage: Math.min(8, adapter.limits.maxStorageBuffersPerShaderStage) },
  });
  const L = layout(net);
  // how far from her each block must run, and how far the distance needs to be exact
  const RF = [], need = [];
  let S = DIL.reduce((a, b) => a + b, 0), r = 0;
  for (let b = 0; b < NB; b++) { r += DIL[b]; S -= DIL[b]; RF.push(r); need.push(Math.min(r, S + MARGIN)); }
  const WIN = Math.max(MARGIN, ...need) + 1;

  const ST = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
  const buffer = (size, usage, data) => {
    const b = device.createBuffer({ size: Math.ceil(size / 4) * 4, usage, mappedAtCreation: !!data });
    if (data) { new data.constructor(b.getMappedRange()).set(data); b.unmap(); }
    return b;
  };
  const pBuf = buffer(L.f.byteLength, ST, L.f), uBuf = buffer(Math.max(4, L.u.byteLength), ST, L.u.length ? L.u : new Uint32Array(1));
  const texBuf = buffer(texb.byteLength, ST, new Uint32Array(texb.buffer));
  const tilesBuf = buffer(maxPixels * 20, ST | CD), inBuf = buffer(maxPixels * CIN * 4, ST);
  const maskBuf = buffer(maxPixels * 4, ST), hBuf = buffer(maxPixels * 4, ST), distBuf = buffer(maxPixels * 4, ST | CS);   // CS: the stats read the distance back
  const X = [buffer(maxPixels * DW * 4, ST), buffer(maxPixels * DW * 4, ST)];
  const bgBuf = buffer((NB + 1) * DW * 4, ST), outBuf = buffer(maxPixels * 4, ST | CS), readBuf = buffer(maxPixels * 4, GPUBufferUsage.MAP_READ | CD);
  const frameBuf = buffer(32, GPUBufferUsage.UNIFORM | CD);
  const NQ = 2 * (NB + 8);   // stats: a pair of timestamps a pass
  const qs = ts ? device.createQuerySet({ type: 'timestamp', count: NQ }) : null;
  const qBuf = ts ? buffer(8 * NQ, GPUBufferUsage.QUERY_RESOLVE | CS) : null, qRead = ts ? buffer(8 * NQ, GPUBufferUsage.MAP_READ | CD) : null;
  const blkBufs = Array.from({ length: NB }, (_, b) => buffer(16, GPUBufferUsage.UNIFORM, new Uint32Array([b, DIL[b], need[b], b ? need[b - 1] : 0])));

  // ---- kernels ----
  const common = wgsl(net, L, packed), W = wgslParts(net);
  const binds = (decls) => decls.map(([i, kind, name, type]) => `@group(0) @binding(${i}) var<${kind}> ${name}: ${type};`).join('\n');
  const F = [0, 'uniform', 'U', 'Frame'], Pf = [1, 'storage, read', 'P', 'array<f32>'], Pu = [2, 'storage, read', 'U32', 'array<u32>'];
  const rdFn = (bgName, xName) => `fn rd(b: u32, idx: i32, c: u32) -> f32 { if (idx < 0) { return ${bgName}[b * 64u + c]; } return ${xName}[c * U.np + u32(idx)]; }`;
  const kernels = {
    prepare: [binds([F, Pf, [2, 'storage, read', 'TILES', 'array<u32>'], [3, 'storage, read', 'TEX', 'array<u32>'], [4, 'storage, read_write', 'IN', 'array<f32>']]) + `
fn byteAt(i: u32) -> u32 { return (TEX[i >> 2u] >> ((i & 3u) * 8u)) & 255u; }
fn tile(k: u32, s: u32) -> vec4<u32> { let v = TILES[k * U.np + s]; return vec4<u32>(v & 255u, (v >> 8u) & 255u, (v >> 16u) & 255u, v >> 24u); }
fn texOne(base: u32, S: u32, lev: u32, u: f32, v: f32, f: ptr<function, array<f32, 18>>) {
  let px = u * f32(S) - 0.5; let py = v * f32(S) - 0.5;
  let fx0 = floor(px); let fy0 = floor(py); let x0 = i32(fx0); let y0 = i32(fy0); let fx = px - fx0; let fy = py - fy0;
  for (var j = 0; j < 2; j++) { for (var i = 0; i < 2; i++) {
    let x = x0 + i; let y = y0 + j;
    if (x < 0 || y < 0 || x >= i32(S) || y >= i32(S)) { continue; }
    let w = select(1.0 - fx, fx, i == 1) * select(1.0 - fy, fy, j == 1);
    let t = base + (u32(y) * S + u32(x)) * 8u;
    for (var c = 0u; c < 8u; c++) { (*f)[8u + c] += w * (P[LO + lev * 8u + c] + f32(byteAt(t + c)) * P[ST + lev * 8u + c]); }
  } }
}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let p = id.x; if (p >= U.np) { return; }
  let y = p / U.w; let x = p % U.w; let s = (U.h - 1u - y) * U.w + x;   // tiles are bottom-up
  let col = tile(0u, s); let nrm = tile(1u, s); let dep = tile(2u, s); let ua = tile(3u, s); let ub = tile(4u, s);
  var f: array<f32, 18>;
  for (var c = 0u; c < 3u; c++) { f[c] = dv(f32(col[c]), 255.0); f[3u + c] = dv(f32(nrm[c]), 255.0) * 2.0 - 1.0; }
  f[6] = dv(f32(dep.x), 255.0);
  let mask = ub.z > 127u;
  f[7] = select(0.0, 1.0, mask);
  if (mask) {
    let u = dv(f32(ua.x) + dv(f32(ua.y), 255.0), 255.0); let v = dv(f32(ua.z) + dv(f32(ub.x), 255.0), 255.0);
    let part = select(0u, ub.y, ub.y < U.parts);
    texOne(part * ${FINE * FINE * FCH}u, ${FINE}u, 0u, u, v, &f);
    texOne(${fineN}u + part * ${COARSE * COARSE * FCH}u, ${COARSE}u, 1u, u, v, &f);
  }
  f[16] = U.sinp; f[17] = U.cosp;
  for (var i = 0u; i < 18u; i++) { IN[p * 18u + i] = f[i]; }
}`],
    mask: [binds([F, [1, 'storage, read', 'IN', 'array<f32>'], [2, 'storage, read_write', 'M', 'array<u32>']]) + `
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let p = id.x; if (p >= U.np) { return; }
  var same = true;
  for (var i = 0u; i < 16u; i++) { same = same && IN[p * 18u + i] == IN[i]; }
  M[p] = select(1u, 0u, same);
}`],
    distH: [binds([F, [1, 'storage, read', 'M', 'array<u32>'], [2, 'storage, read_write', 'H', 'array<u32>']]) + `
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let p = id.x; if (p >= U.np) { return; }
  let y = i32(p / U.w); let x = i32(p % U.w); var d = 255;
  for (var k = -${WIN}; k <= ${WIN}; k++) { let q = x + k; if (q >= 0 && q < i32(U.w) && M[u32(y) * U.w + u32(q)] == 1u) { d = min(d, abs(k)); } }
  H[p] = u32(d);
}`],
    distV: [binds([F, [1, 'storage, read', 'H', 'array<u32>'], [2, 'storage, read_write', 'D', 'array<u32>']]) + `
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let p = id.x; if (p >= U.np) { return; }
  let y = i32(p / U.w); let x = p % U.w; var d = 255u;
  for (var k = -${WIN}; k <= ${WIN}; k++) { let q = y + k; if (q >= 0 && q < i32(U.h)) { d = min(d, max(u32(abs(k)), H[u32(q) * U.w + x])); } }
  D[p] = select(255u, d, d <= ${WIN}u);
}`],
    chain: [binds([F, Pf, Pu, [3, 'storage, read_write', 'BG', 'array<f32>'], [4, 'storage, read', 'IN', 'array<f32>'], [5, 'storage, read', 'XI', 'array<f32>']]) + rdFn('BG', 'XI') + W.quant + W.token + W.embed + `
@compute @workgroup_size(1) fn main() {   // the background (pixel 0's inputs) through the blocks, on itself
  var f: array<f32, 18>;
  for (var i = 0u; i < 18u; i++) { f[i] = IN[i]; }
  let x = embed(f);
  for (var c = 0u; c < 64u; c++) { BG[c] = x[c]; }
  var taps: array<i32, 9>;
  for (var q = 0u; q < 9u; q++) { taps[q] = -1; }
  for (var b = 0u; b < ${NB}u; b++) {
    let y = token(b, taps, -1);
    for (var c = 0u; c < 64u; c++) { BG[(b + 1u) * 64u + c] = y[c]; }
  }
}`],
    embed: [binds([F, Pf, [2, 'storage, read', 'IN', 'array<f32>'], [3, 'storage, read', 'D', 'array<u32>'], [4, 'storage, read_write', 'XO', 'array<f32>']]) + W.embed + `
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let p = id.x; if (p >= U.np || D[p] > 0u) { return; }
  var f: array<f32, 18>;
  for (var i = 0u; i < 18u; i++) { f[i] = IN[p * 18u + i]; }
  let x = embed(f);
  for (var c = 0u; c < 64u; c++) { XO[c * U.np + p] = x[c]; }
}`],
    block: [binds([F, Pf, Pu, [3, 'storage, read', 'BG', 'array<f32>'], [4, 'storage, read', 'D', 'array<u32>'], [5, 'uniform', 'K', 'Blk'], [6, 'storage, read', 'XI', 'array<f32>'], [7, 'storage, read_write', 'XO', 'array<f32>']]) + rdFn('BG', 'XI') + W.quant + W.token + `
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let p = id.x; if (p >= U.np || D[p] > K.need) { return; }
  let r = i32(p / U.w); let c = i32(p % U.w); let dil = i32(K.dil);
  var taps: array<i32, 9>; var k = 0u;
  for (var dy = -1; dy <= 1; dy++) { for (var dx = -1; dx <= 1; dx++) {
    let rr = r + dy * dil; let cc = c + dx * dil; var idx = -1;
    if (rr >= 0 && rr < i32(U.h) && cc >= 0 && cc < i32(U.w)) { let q = u32(rr) * U.w + u32(cc); if (D[q] <= K.prev) { idx = i32(q); } }
    taps[k] = idx; k++;
  } }
  let y = token(K.b, taps, select(-1, i32(p), D[p] <= K.prev));
  for (var ch = 0u; ch < 64u; ch++) { XO[ch * U.np + p] = y[ch]; }
}`],
    head: [binds([F, Pf, [2, 'storage, read', 'BG', 'array<f32>'], [3, 'storage, read', 'D', 'array<u32>'], [4, 'storage, read', 'XI', 'array<f32>'], [5, 'storage, read_write', 'OUT', 'array<u32>']]) + W.head + `
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let p = id.x; if (p >= U.np) { return; }
  var v: array<f32, 64>;
  if (D[p] <= ${MARGIN}u) { for (var c = 0u; c < 64u; c++) { v[c] = XI[c * U.np + p]; } }
  else { for (var c = 0u; c < 64u; c++) { v[c] = BG[${NB}u * 64u + c]; } }
  let q = head(v);
  var o = 0u;
  for (var k = 0u; k < 4u; k++) { let b = q[k] * 255.0 + 0.5; o |= select(select(u32(b), 255u, b > 255.0), 0u, b < 0.0) << (8u * k); }
  OUT[p] = o;
}`],
  };
  const pipes = {};
  for (const [name, [src]] of Object.entries(kernels)) {
    const code = common + src;
    const module = device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    const errs = info.messages.filter(m => m.type === 'error');
    if (errs.length) throw new Error(`gpu-nts ${name}: ${errs.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join('; ')}`);
    pipes[name] = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  }
  const group = (name, bufs) => device.createBindGroup({ layout: pipes[name].getBindGroupLayout(0), entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
  const g = {
    prepare: group('prepare', [frameBuf, pBuf, tilesBuf, texBuf, inBuf]),
    mask: group('mask', [frameBuf, inBuf, maskBuf]),
    distH: group('distH', [frameBuf, maskBuf, hBuf]),
    distV: group('distV', [frameBuf, hBuf, distBuf]),
    chain: group('chain', [frameBuf, pBuf, uBuf, bgBuf, inBuf, X[1]]),
    embed: group('embed', [frameBuf, pBuf, inBuf, distBuf, X[0]]),
    blocks: Array.from({ length: NB }, (_, b) => group('block', [frameBuf, pBuf, uBuf, bgBuf, distBuf, blkBufs[b], X[b % 2], X[(b + 1) % 2]])),
    head: group('head', [frameBuf, pBuf, bgBuf, distBuf, X[NB % 2], outBuf]),
  };

  let busy = false, ms = null, gpuMs = null, out = new Uint8ClampedArray(0), stats = null;
  const distRead = wantStats ? buffer(maxPixels * 4, GPUBufferUsage.MAP_READ | CD) : null;
  async function frame({ tiles, w, h, pitch }) {
    busy = true;
    const t0 = performance.now(), np = w * h;
    device.queue.writeBuffer(tilesBuf, 0, tiles.buffer, tiles.byteOffset, np * 20);
    const fr = new ArrayBuffer(32), fu = new Uint32Array(fr), ff = new Float32Array(fr);
    fu[0] = w; fu[1] = h; fu[2] = np; fu[3] = P; ff[4] = Math.sin(pitch); ff[5] = Math.cos(pitch);
    device.queue.writeBuffer(frameBuf, 0, fr);
    const enc = device.createCommandEncoder(), groups = Math.ceil(np / 64);
    // timestamps: the first pass's start and the last's end; with stats, every pass's
    const names = [];
    const run = (name, bg, n = groups) => {
      const i = names.length, first = i === 0, last = name === 'head', each = ts && wantStats;
      names.push(name);
      const tw = each ? { querySet: qs, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 }
        : ts && first ? { querySet: qs, beginningOfPassWriteIndex: 0 } : ts && last ? { querySet: qs, endOfPassWriteIndex: 1 } : null;
      const p = enc.beginComputePass(tw ? { timestampWrites: tw } : {});
      p.setPipeline(pipes[name]); p.setBindGroup(0, bg); p.dispatchWorkgroups(n); p.end();
    };
    run('prepare', g.prepare); run('mask', g.mask); run('distH', g.distH); run('distV', g.distV);
    run('chain', g.chain, 1); run('embed', g.embed);
    for (let b = 0; b < NB; b++) run('block', g.blocks[b]);
    run('head', g.head);
    enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, np * 4);
    const nq = ts && wantStats ? 2 * names.length : 2;
    if (ts) { enc.resolveQuerySet(qs, 0, nq, qBuf, 0); enc.copyBufferToBuffer(qBuf, 0, qRead, 0, 8 * nq); }
    if (wantStats) enc.copyBufferToBuffer(distBuf, 0, distRead, 0, np * 4);
    device.queue.submit([enc.finish()]);
    await Promise.all([readBuf.mapAsync(GPUMapMode.READ, 0, np * 4), ts ? qRead.mapAsync(GPUMapMode.READ, 0, 8 * nq) : null,
      wantStats ? distRead.mapAsync(GPUMapMode.READ, 0, np * 4) : null]);
    if (ts) {
      const q = new BigUint64Array(qRead.getMappedRange(0, 8 * nq).slice(0)); qRead.unmap();
      const g = Number((wantStats ? q[nq - 1] : q[1]) - q[0]) / 1e6;
      if (g > 0 && g < 1e4) gpuMs = gpuMs == null ? g : gpuMs * 0.8 + g * 0.2;
      if (wantStats) stats = { ...stats, gpuMs: g, passMs: names.map((n, i) => [n === 'block' ? `block ${i - 6}` : n, Number(q[2 * i + 1] - q[2 * i]) / 1e6]) };
    }
    if (wantStats) {   // how many pixels each pass computed: the distance against each block's reach
      const d = new Uint32Array(distRead.getMappedRange(0, np * 4)), hist = new Uint32Array(257);
      for (let i = 0; i < np; i++) hist[Math.min(d[i], 256)]++;
      distRead.unmap();
      const upto = n => { let c = 0; for (let k = 0; k <= n; k++) c += hist[k]; return c; };
      stats = { ...stats, w, h, np, her: hist[0], active: need.map(upto), corner: hist[0] === np ? 'on her' : 'background' };
    }
    if (out.length < np * 4) out = new Uint8ClampedArray(Math.ceil(np * 4 * 1.3));
    out.set(new Uint8Array(readBuf.getMappedRange(0, np * 4)));
    readBuf.unmap();
    const dt = performance.now() - t0;
    ms = ms == null ? dt : ms * 0.8 + dt * 0.2;
    busy = false;
    return out.subarray(0, np * 4);
  }
  const kind = net.T ? `${net.T} trees × ${net.V} levels` : 'dense FFNs';
  return {
    frame, parts: meta.parts, gpu: true, threads: 1, relaxed: false,
    kernel: `WebGPU compute (${kind}${packed ? ', dot4I8Packed' : ''})`,
    adapter: adapter.info ? `${adapter.info.vendor} ${adapter.info.architecture}`.trim() : '',
    get busy() { return busy; }, get ms() { return ms; },   // ms: wall time a frame (upload to readback)
    get gpuMs() { return gpuMs; },                           // the passes on the GPU's clock (timestamp queries), or null
    get stats() { return stats; },
  };
}
