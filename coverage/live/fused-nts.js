// The fused WebGPU engine (cpu.html #fused): the nanotss tree shader with everything on the GPU.
//   in    the five G-buffer tiles as three.js drew them side by side on its canvas, her crop copied
//         GPU to GPU into a texture array (copyExternalImageToTexture: no readPixels, no upload)
//   net   per block, one compute kernel a pixel: the stencil, int8 rounding, the trees' walk
//         (dot4I8Packed: int8 codes x ternary w_in, four at a time) and their outputs (w_out as
//         ternary bitmasks and a row scale: adds and subtracts), the activations kept as half floats
//         where the GPU has shader-f16 (#fp32: 32-bit), in strips of rows as the GL engine (each with
//         the network's reach of extra rows, so a strip's own rows come out as from the whole crop)
//   out   a texture, copied to a WebGPU canvas and drawn onto the page's 2D canvas, no readback
// The maths is gpu-nts.js's (infer.c's order, its exp and erf); the inputs are decoded where they're
// used rather than kept (18 floats a pixel), the background test reads the tiles' bytes.
import { parseNet, layout, wgsl, wgslParts, unpackPng, netTexture } from './gpu-nts.js';

const DW = 64, FCH = 8, FINE = 512, COARSE = 64, MARGIN = 6;

export async function createFusedShader({ base = 'nts/', NW, NH, maxStrip = Infinity, precision = 'half', target, stats = false }) {
  if (!navigator.gpu) throw new Error('no WebGPU');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter');
  const get = async f => { const r = await fetch(base + f); if (!r.ok) throw new Error(`${base}${f}: ${r.status}`); return r; };
  const [meta, raw] = await Promise.all([get('meta.json').then(r => r.json()), get('net.png').then(r => r.blob()).then(unpackPng)]);
  const net = parseNet(raw), { NB, P, DIL, T, V, TN } = net;
  if (!T) throw new Error('fused-nts: tree nets only');
  const texb = await netTexture(get, meta, net), fineN = P * meta.tile * meta.tile * FCH;
  const packed = navigator.gpu.wgslLanguageFeatures?.has('packed_4x8_integer_dot_product');
  const half = precision !== 'full' && adapter.features.has('shader-f16');
  const ts = adapter.features.has('timestamp-query');

  // the strips: SH rows of activations (2 buffers of 64 values a pixel), the reach on each side
  // (and no more than the device lets one storage buffer hold)
  const REACH = DIL.reduce((a, b) => a + b, 0), xpx = DW * (half ? 2 : 4);
  const SH = Math.min(NH, Math.max(2 * REACH + 64, Math.floor(Math.min(maxStrip, adapter.limits.maxStorageBufferBindingSize / xpx) / NW))), STEP = SH - 2 * REACH;
  const NS = Math.ceil(NH / STEP) + 1, SPX = NW * SH, xBytes = SPX * xpx;
  const device = await adapter.requestDevice({
    requiredFeatures: [...(half ? ['shader-f16'] : []), ...(ts ? ['timestamp-query'] : [])],
    requiredLimits: { maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, Math.max(xBytes, texb.byteLength, 134217728)),
      maxBufferSize: Math.min(adapter.limits.maxBufferSize, Math.max(xBytes, texb.byteLength, 268435456)),
      maxStorageBuffersPerShaderStage: Math.min(8, adapter.limits.maxStorageBuffersPerShaderStage) },
  });
  if (xBytes > device.limits.maxStorageBufferBindingSize) throw new Error(`fused-nts: a strip needs ${xBytes} bytes`);
  const L = layout(net);
  const RF = [], need = [];
  let S = REACH, r = 0;
  for (let b = 0; b < NB; b++) { r += DIL[b]; S -= DIL[b]; RF.push(r); need.push(Math.min(r, S + MARGIN)); }
  const WIN = Math.max(MARGIN, ...need) + 1;

  // ---- buffers and textures ----
  const SB = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, UB = GPUBufferUsage.UNIFORM;
  const buffer = (size, usage, data) => {
    const b = device.createBuffer({ size: Math.ceil(size / 4) * 4, usage, mappedAtCreation: !!data });
    if (data) { new data.constructor(b.getMappedRange()).set(data); b.unmap(); }
    return b;
  };
  const pBuf = buffer(L.f.byteLength, SB, L.f), uBuf = buffer(L.u.byteLength, SB, L.u);
  const texBuf = buffer(texb.byteLength, SB, new Uint32Array(texb.buffer));
  const maskBuf = buffer(NW * NH * 4, SB), hBuf = buffer(NW * NH * 4, SB), distBuf = buffer(NW * NH * 4, SB);
  const X = [buffer(xBytes, SB), buffer(xBytes, SB)], bgBuf = buffer((NB + 1) * DW * 4, SB);
  const frameBuf = buffer(32, UB | CD);
  const stripBufs = Array.from({ length: NS }, () => buffer(16, UB | CD));
  const blkBufs = Array.from({ length: NB }, (_, b) => buffer(16, UB, new Uint32Array([b, DIL[b], need[b], b ? need[b - 1] : 0])));
  const TU = GPUTextureUsage;
  const tilesTex = device.createTexture({ size: [NW, NH, 5], format: 'rgba8unorm', usage: TU.TEXTURE_BINDING | TU.COPY_DST | TU.RENDER_ATTACHMENT });
  const outTex = device.createTexture({ size: [NW, NH], format: 'rgba8unorm', usage: TU.STORAGE_BINDING | TU.COPY_SRC | TU.RENDER_ATTACHMENT });
  const canvas = document.createElement('canvas'); canvas.width = NW; canvas.height = NH;
  const cx = canvas.getContext('webgpu');
  cx.configure({ device, format: 'rgba8unorm', alphaMode: 'premultiplied', usage: TU.COPY_DST | TU.RENDER_ATTACHMENT });
  const qs = ts ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null;
  const qBuf = ts ? buffer(16, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC) : null, qRead = ts ? buffer(16, GPUBufferUsage.MAP_READ | CD) : null;

  // ---- kernels ----
  const XT = half ? 'f16' : 'f32';
  const common = (half ? 'enable f16;\n' : '') + wgsl(net, L, packed) + `
const WOB = array<u32, ${NB}>(${L.blk.map(o => `${o.wob}u`).join(', ')});
const SOUT = array<u32, ${NB}>(${L.blk.map(o => `${o.sout}u`).join(', ')});`;
  const W = wgslParts(net);
  // the trees' outputs from the bitmasks: + or - g x the row's scale, as gpu-nts.js's float rows give
  const token = W.token.replace(/      for \(var c = 0u; c < 64u; c\+\+\) \{ y\[c\] = P\[WOUT\[b\] \+ row \* 64u \+ c\] \* g \+ y\[c\]; \}/, `      let gs = g * P[SOUT[b] + row]; let mb = WOB[b] + row * 4u;
      let pl = U32[mb]; let ph = U32[mb + 1u]; let nl = U32[mb + 2u]; let nh = U32[mb + 3u];
      for (var c = 0u; c < 64u; c++) {
        let bit = 1u << (c & 31u); let hi = c >= 32u;
        if ((select(pl, ph, hi) & bit) != 0u) { y[c] = gs + y[c]; } else if ((select(nl, nh, hi) & bit) != 0u) { y[c] = -gs + y[c]; }
      }`);
  if (token === W.token) throw new Error('fused-nts: the trees\' output loop has changed');
  // (Frame and Blk are gpu-nts.js's; Frame's pad0, pad1 carry the crop's corner here)
  const F = `struct Strip { o: u32, top: u32, r0: u32, r1: u32 };
@group(0) @binding(0) var<uniform> U: Frame;`;
  // the 18 inputs at crop pixel (x, y), top-down, from the tiles (gpu-nts.js prepare)
  const inputs = `
fn byteAt(i: u32) -> u32 { return (TEX[i >> 2u] >> ((i & 3u) * 8u)) & 255u; }
fn tileB(k: u32, x: u32, y: u32) -> vec4<u32> { return vec4<u32>(round(textureLoad(TT, vec2<u32>(x, y), k, 0) * 255.0)); }
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
fn inputsAt(x: u32, y: u32) -> array<f32, 18> {
  let col = tileB(0u, x, y); let nrm = tileB(1u, x, y); let dep = tileB(2u, x, y); let ua = tileB(3u, x, y); let ub = tileB(4u, x, y);
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
  return f;
}`;
  const rdFn = `fn rd(b: u32, idx: i32, c: u32) -> f32 { if (idx < 0) { return BG[b * 64u + c]; } return f32(XI[c * ${SPX}u + u32(idx)]); }`;
  const kernels = {
    // the background test against the crop's bottom-left pixel (outside her by construction)
    mask: `${F}
@group(0) @binding(1) var TT: texture_2d_array<f32>;
@group(0) @binding(2) var<storage, read_write> M: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let p = id.x; if (p >= U.np) { return; }
  let x = p % U.w; let y = p / U.w; var same = true;
  for (var k = 0u; k < 5u; k++) { same = same && all(textureLoad(TT, vec2<u32>(x, y), k, 0) == textureLoad(TT, vec2<u32>(0u, U.h - 1u), k, 0)); }
  M[p] = select(1u, 0u, same);
}`,
    distH: `${F}
@group(0) @binding(1) var<storage, read> M: array<u32>;
@group(0) @binding(2) var<storage, read_write> H: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let p = id.x; if (p >= U.np) { return; }
  let y = i32(p / U.w); let x = i32(p % U.w); var d = 255;
  for (var k = -${WIN}; k <= ${WIN}; k++) { let q = x + k; if (q >= 0 && q < i32(U.w) && M[u32(y) * U.w + u32(q)] == 1u) { d = min(d, abs(k)); } }
  H[p] = u32(d);
}`,
    distV: `${F}
@group(0) @binding(1) var<storage, read> H: array<u32>;
@group(0) @binding(2) var<storage, read_write> D: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let p = id.x; if (p >= U.np) { return; }
  let y = i32(p / U.w); let x = p % U.w; var d = 255u;
  for (var k = -${WIN}; k <= ${WIN}; k++) { let q = y + k; if (q >= 0 && q < i32(U.h)) { d = min(d, max(u32(abs(k)), H[u32(q) * U.w + x])); } }
  D[p] = select(255u, d, d <= ${WIN}u);
}`,
    // the background through the blocks, on itself (one invocation)
    chain: `${F}
@group(0) @binding(1) var<storage, read> P: array<f32>;
@group(0) @binding(2) var<storage, read> U32: array<u32>;
@group(0) @binding(3) var<storage, read> TEX: array<u32>;
@group(0) @binding(4) var TT: texture_2d_array<f32>;
@group(0) @binding(5) var<storage, read_write> BG: array<f32>;
@group(0) @binding(6) var<storage, read> XI: array<${XT}>;
${inputs}
${rdFn}${W.quant}${token}${W.embed}
@compute @workgroup_size(1) fn main() {
  let x = embed(inputsAt(0u, U.h - 1u));
  for (var c = 0u; c < 64u; c++) { BG[c] = x[c]; }
  var taps: array<i32, 9>;
  for (var q = 0u; q < 9u; q++) { taps[q] = -1; }
  for (var b = 0u; b < ${NB}u; b++) {
    let y = token(b, taps, -1);
    for (var c = 0u; c < 64u; c++) { BG[(b + 1u) * 64u + c] = y[c]; }
  }
}`,
    // a strip's rows o .. top: the inputs and the embedding where she is
    embed: `${F}
@group(0) @binding(1) var<storage, read> P: array<f32>;
@group(0) @binding(2) var<storage, read> TEX: array<u32>;
@group(0) @binding(3) var TT: texture_2d_array<f32>;
@group(0) @binding(4) var<storage, read> D: array<u32>;
@group(0) @binding(5) var<uniform> SS: Strip;
@group(0) @binding(6) var<storage, read_write> XO: array<${XT}>;
${inputs}${W.embed}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x; if (i >= (SS.top - SS.o) * U.w) { return; }
  let y = SS.o + i / U.w; let x = i % U.w;
  if (D[y * U.w + x] > 0u) { return; }
  let e = embed(inputsAt(x, y));
  for (var c = 0u; c < 64u; c++) { XO[c * ${SPX}u + i] = ${XT}(e[c]); }
}`,
    block: `${F}
@group(0) @binding(1) var<storage, read> P: array<f32>;
@group(0) @binding(2) var<storage, read> U32: array<u32>;
@group(0) @binding(3) var<storage, read> BG: array<f32>;
@group(0) @binding(4) var<storage, read> D: array<u32>;
@group(0) @binding(5) var<uniform> K: Blk;
@group(0) @binding(6) var<uniform> SS: Strip;
@group(0) @binding(7) var<storage, read> XI: array<${XT}>;
@group(0) @binding(8) var<storage, read_write> XO: array<${XT}>;
${rdFn}${W.quant}${token}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x; if (i >= (SS.top - SS.o) * U.w) { return; }
  let r = i32(SS.o + i / U.w); let c = i32(i % U.w); let p = u32(r) * U.w + u32(c);
  if (D[p] > K.need) { return; }
  let dil = i32(K.dil); let o = i32(SS.o);
  var taps: array<i32, 9>; var k = 0u;
  for (var dy = -1; dy <= 1; dy++) { for (var dx = -1; dx <= 1; dx++) {
    let rr = r + dy * dil; let cc = c + dx * dil; var idx = -1;
    if (rr >= o && rr < i32(SS.top) && cc >= 0 && cc < i32(U.w)) { let q = u32(rr) * U.w + u32(cc); if (D[q] <= K.prev) { idx = (rr - o) * i32(U.w) + cc; } }
    taps[k] = idx; k++;
  } }
  let y = token(K.b, taps, select(-1, i32(i), D[p] <= K.prev));
  for (var ch = 0u; ch < 64u; ch++) { XO[ch * ${SPX}u + i] = ${XT}(y[ch]); }
}`,
    // a strip's own rows r0 .. r1: 64 -> 4, premultiplied into the output texture
    head: `${F}
@group(0) @binding(1) var<storage, read> P: array<f32>;
@group(0) @binding(2) var<storage, read> BG: array<f32>;
@group(0) @binding(3) var<storage, read> D: array<u32>;
@group(0) @binding(4) var<uniform> SS: Strip;
@group(0) @binding(5) var<storage, read> XI: array<${XT}>;
@group(0) @binding(6) var OUT: texture_storage_2d<rgba8unorm, write>;
${W.head}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let j = id.x; if (j >= (SS.r1 - SS.r0) * U.w) { return; }
  let y = SS.r0 + j / U.w; let x = j % U.w; let p = y * U.w + x; let i = (y - SS.o) * U.w + x;
  var v: array<f32, 64>;
  if (D[p] <= ${MARGIN}u) { for (var c = 0u; c < 64u; c++) { v[c] = f32(XI[c * ${SPX}u + i]); } }
  else { for (var c = 0u; c < 64u; c++) { v[c] = BG[${NB}u * 64u + c]; } }
  let q = clamp(round(head(v) * 255.0), vec4<f32>(0.0), vec4<f32>(255.0)) / 255.0;   // the bytes gpu-nts.js gives
  textureStore(OUT, vec2<u32>(U.pad0 + x, U.pad1 + y), vec4<f32>(q.rgb * q.a, q.a));
}`,
  };
  const pipes = {};
  for (const [name, src] of Object.entries(kernels)) {
    const module = device.createShaderModule({ code: common + src });
    const errs = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
    if (errs.length) throw new Error(`fused-nts ${name}: ${errs.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join('; ')}`);
    pipes[name] = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  }
  const res = x => x instanceof GPUBuffer ? { buffer: x } : x;
  const group = (name, list) => device.createBindGroup({ layout: pipes[name].getBindGroupLayout(0), entries: list.map((x, i) => ({ binding: i, resource: res(x) })) });
  const tilesView = tilesTex.createView({ dimension: '2d-array' }), outView = outTex.createView();
  const g = {
    mask: group('mask', [frameBuf, tilesView, maskBuf]),
    distH: group('distH', [frameBuf, maskBuf, hBuf]),
    distV: group('distV', [frameBuf, hBuf, distBuf]),
    chain: group('chain', [frameBuf, pBuf, uBuf, texBuf, tilesView, bgBuf, X[1]]),
    embed: stripBufs.map(s => group('embed', [frameBuf, pBuf, texBuf, tilesView, distBuf, s, X[0]])),
    blocks: stripBufs.map(s => Array.from({ length: NB }, (_, b) => group('block', [frameBuf, pBuf, uBuf, bgBuf, distBuf, blkBufs[b], s, X[b % 2], X[(b + 1) % 2]]))),
    head: stripBufs.map(s => group('head', [frameBuf, pBuf, bgBuf, distBuf, s, X[NB % 2], outView])),
  };

  let busy = false, ms = null, gpuMs = null;
  // source: the canvas holding the tiles side by side (each NW x NH); crop: [x0, y0, x1, y1] network
  // pixels, top-down. Draws her into target (a 2D context the size of the network frame).
  async function frame({ source, crop, pitch }) {
    busy = true;
    const t0 = performance.now();
    const [x0, y0, x1, y1] = crop, w = x1 - x0, h = y1 - y0, np = w * h;
    for (let k = 0; k < 5; k++)
      device.queue.copyExternalImageToTexture({ source, origin: { x: k * NW + x0, y: y0 } }, { texture: tilesTex, origin: { x: 0, y: 0, z: k } }, { width: w, height: h });
    const fr = new ArrayBuffer(32), fu = new Uint32Array(fr), ff = new Float32Array(fr);
    fu[0] = w; fu[1] = h; fu[2] = np; fu[3] = P; ff[4] = Math.sin(pitch); ff[5] = Math.cos(pitch); fu[6] = x0; fu[7] = y0;
    device.queue.writeBuffer(frameBuf, 0, fr);
    // the strips (crop rows, top-down): own rows [r0, r1), activations from row o
    const one = h <= SH, step = one ? SH : STEP, strips = [];
    for (let r0 = 0; r0 < h; r0 += step) {
      const r1 = Math.min(h, r0 + step), o = one ? 0 : Math.max(0, r0 - REACH), top = Math.min(h, o + SH);
      device.queue.writeBuffer(stripBufs[strips.length], 0, new Uint32Array([o, top, r0, r1]));
      strips.push({ o, top, r0, r1 });
    }
    const enc = device.createCommandEncoder();
    enc.beginRenderPass({ colorAttachments: [{ view: outView, loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }] }).end();
    const pass = enc.beginComputePass(ts ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
    const run = (name, bg, n) => { pass.setPipeline(pipes[name]); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(Math.ceil(n / 64)); };
    run('mask', g.mask, np); run('distH', g.distH, np); run('distV', g.distV, np);
    pass.setPipeline(pipes.chain); pass.setBindGroup(0, g.chain); pass.dispatchWorkgroups(1);
    strips.forEach(({ o, top, r0, r1 }, s) => {
      run('embed', g.embed[s], (top - o) * w);
      for (let b = 0; b < NB; b++) run('block', g.blocks[s][b], (top - o) * w);
      run('head', g.head[s], (r1 - r0) * w);
    });
    pass.end();
    enc.copyTextureToTexture({ texture: outTex }, { texture: cx.getCurrentTexture() }, [NW, NH]);
    if (ts) { enc.resolveQuerySet(qs, 0, 2, qBuf, 0); if (qRead.mapState === 'unmapped') enc.copyBufferToBuffer(qBuf, 0, qRead, 0, 16); }
    device.queue.submit([enc.finish()]);
    // onto the page's canvas now, while the WebGPU canvas's texture is this frame's
    target.clearRect(0, 0, NW, NH);
    target.drawImage(canvas, x0, y0, w, h, x0, y0, w, h);
    await device.queue.onSubmittedWorkDone();
    if (ts && qRead.mapState === 'unmapped') {
      await qRead.mapAsync(GPUMapMode.READ);
      const q = new BigUint64Array(qRead.getMappedRange().slice(0)); qRead.unmap();
      const gm = Number(q[1] - q[0]) / 1e6;
      if (gm > 0 && gm < 1e4) gpuMs = gpuMs == null ? gm : gpuMs * 0.8 + gm * 0.2;
    }
    const dt = performance.now() - t0;
    ms = ms == null ? dt : ms * 0.8 + dt * 0.2;
    busy = false;
  }
  return {
    frame, parts: meta.parts, gpu: true, fused: true, threads: 1, relaxed: false,
    kernel: `fused WebGPU (${T} trees × ${V} levels, ${half ? 'f16' : 'f32'} activations${packed ? ', dot4I8Packed' : ''}, ${SH < NH ? `strips of ${SH} rows` : 'whole crop'})`,
    get busy() { return busy; }, get ms() { return ms; }, get gpuMs() { return gpuMs; }, get stats() { return null; },
  };
}
