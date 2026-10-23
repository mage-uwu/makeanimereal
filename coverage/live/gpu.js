// WebGPU compute path for the learned shader (coverage/live/index.html).
//
// three.js still renders the G-buffer, but into a hidden WebGL canvas laid out as five tiles in
// exactly the 8-bit encodings the shader was trained on (colour, normal, depth, UV high/low bytes
// with the part id and coverage). copyExternalImageToTexture moves that canvas to the GPU device,
// and compute shaders run the decoder on storage buffers. Each convolution thread computes
// PX pixels x G channel groups, so every weight it loads is used PX times and every input it loads
// G x 4 times; all passes go out in one command buffer and only cover the box around the character.
// Tensors are float16 where the GPU has shader-f16, accumulating in float32.

const PX = 4, G = 4, TILES = 5;

export async function createGpuShader({ net, feats, canvas, NW, NH, forceF32 = false, skipEmpty = true, profile = false, math = 'auto' }) {
  if (!navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return null;
  const f16 = !forceF32 && adapter.features.has('shader-f16');
  const timestamps = adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice({
    requiredFeatures: [...(f16 ? ['shader-f16'] : []), ...(timestamps ? ['timestamp-query'] : [])],
    // Large frames make full-resolution tensors over the default 128 MB binding: ask for what the adapter has.
    requiredLimits: { maxStorageBuffersPerShaderStage: Math.min(8, adapter.limits.maxStorageBuffersPerShaderStage),
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize },
  });
  device.lost.then(info => console.error('WebGPU device lost:', info.reason, info.message));
  device.addEventListener('uncapturederror', e => console.error('WebGPU error:', e.error.message));
  const ctx = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'opaque' });

  const S = f16 ? 'f16' : 'f32', bytes = f16 ? 8 : 16;
  const head = `${f16 ? 'enable f16;' : ''}\nalias S = ${S};\nconst NWF = ${NW}.0;\n`;
  const buf = (size, usage) => device.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage });
  const STORE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

  // ---- tensors ----
  const div = { x: 1, d0: 1, d1: 2, d2: 4, d3: 8, u0: 4, u1: 2, u2: 1, rgba: 1 };
  const chans = { ...net.channels, rgba: 4 };
  for (const l of net.layers) chans[l.out] = l.cout;
  const chansOf = n => chans[n.replace(/_pool$/, '')] ?? chans[n];
  const T = {};
  const tensor = (w, h, layers) => ({ w, h, layers, buf: buf(w * h * layers * bytes, STORE) });
  // Sizes come from shader.json where it has them; a sharp model also has a full-resolution input
  // (xf) and draws at full resolution.
  // shader.json's sizes are for its training frame (net.size); the page's frame may be larger.
  const [sx, sy] = net.size ? [NW / net.size[0], NH / net.size[1]] : [1, 1];
  const dimsOf = name => net.dims?.[name] ? [net.dims[name][0] * sx, net.dims[name][1] * sy] : [NW / div[name.replace(/_mid$/, '')], NH / div[name.replace(/_mid$/, '')]];
  const [GW, GH] = net.sharp ? [net.full[0] * sx, net.full[1] * sy] : [NW, NH];  // the G-buffer's size
  const LIT = net.light || 0;  // light channels after the 18 inputs (train_light.py)
  const XL = Math.ceil((18 + LIT) / 4);
  T.x = tensor(NW, NH, XL);
  if (net.sharp) T.xf = tensor(GW, GH, XL);
  for (const l of net.layers) { const [w, h] = dimsOf(l.out); T[l.out] = tensor(w, h, Math.ceil(l.cout / 4)); }
  for (const [pool, src] of Object.entries(net.pools)) T[pool] = tensor(T[src].w / 2, T[src].h / 2, T[src].layers);
  const [OW, OH] = [T.rgba.w, T.rgba.h];

  // ---- weights: per layer, per output group g, input layer l, tap t: 4 vec4 (one per output j) ----
  let wTotal = 0, rows = 0;
  for (const l of net.layers) {
    l.ins = l.inputs.map(n => ({ n, c: chansOf(n) }));
    l.L = l.ins.reduce((s, i) => s + Math.ceil(i.c / 4), 0);
    l.T = l.k * l.k; l.groups = Math.ceil(l.cout / 4); l.la = Math.ceil(l.ins[0].c / 4);
    l.wbase = wTotal; l.row = rows;
    wTotal += l.groups * l.L * l.T * 4; rows += l.groups;
  }
  const wData = f16 ? new Uint16Array(wTotal * 4) : new Float32Array(wTotal * 4);
  const aff = new Float32Array(rows * 12);
  for (const l of net.layers) {
    const raw = new Uint16Array(Uint8Array.from(atob(l.w), c => c.charCodeAt(0)).buffer);
    let cOff = 0, lOff = 0;
    for (const inp of l.ins) {
      for (let c = 0; c < inp.c; c++) {
        const layer = lOff + (c >> 2), comp = c & 3, tc = cOff + c;
        for (let o = 0; o < l.cout; o++) for (let t = 0; t < l.T; t++) {
          const i = (l.wbase + (((o >> 2) * l.L + layer) * l.T + t) * 4 + (o & 3)) * 4 + comp;
          const h = raw[(o * l.cin + tc) * l.T + t];
          wData[i] = f16 ? h : halfToFloat(h);
        }
      }
      cOff += inp.c; lOff += Math.ceil(inp.c / 4);
    }
    for (let o = 0; o < l.groups * 4; o++) {
      const r = l.row + (o >> 2), j = o & 3;
      aff[r * 12 + j] = l.b[o] ?? 0;
      aff[r * 12 + 4 + j] = l.affine ? (l.affine[0][o] ?? 1) : 1;
      aff[r * 12 + 8 + j] = l.affine ? (l.affine[1][o] ?? 0) : 0;
    }
  }
  const wBuf = buf(wData.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(wBuf, 0, wData);
  const affBuf = buf(aff.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(affBuf, 0, aff);

  // ---- textures: the G-buffer atlas and the neural texture ----
  const gbuf = device.createTexture({ size: [GW * TILES, GH], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
  const featTex = feats.map(bmp => {
    const t = device.createTexture({ size: [bmp.width, bmp.height], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
    device.queue.copyExternalImageToTexture({ source: bmp }, { texture: t }, [bmp.width, bmp.height]);
    return t;
  });
  const linear = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });

  const cropBuf = buf(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const uniform = data => { const b = buf(data.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST); device.queue.writeBuffer(b, 0, data); return b; };
  const pipe = (code, entry = 'main') => device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: head + code }), entryPoint: entry } });

  const CROP = `
fn cropOf(w: i32, h: i32) -> vec4<i32> {
  let s = f32(w) / NWF;
  return vec4<i32>(max(0, i32(floor(f32(crop.x) * s))), max(0, i32(floor(f32(crop.y) * s))),
                   min(w, i32(ceil(f32(crop.z) * s)) + 1), min(h, i32(ceil(f32(crop.w) * s)) + 1));
}`;

  // ---- pack: G-buffer atlas -> an input tensor, at the G-buffer's size (down 1) or half of it (down 2:
  // 2 x 2 averages for colour, normal and depth, the top-left texel's UV, part and coverage, as train_sr.py) ----
  const packPipeFor = (down) => pipe(`
struct PP { sinP: f32, cosP: f32, tile: f32, cols: f32, rows: f32, pad0: f32, pad1: f32, pad2: f32, lo0: vec4<f32>, lo1: vec4<f32>, hi0: vec4<f32>, hi1: vec4<f32>,
  ldir: vec4<f32>, key: vec4<f32>, amb: vec4<f32> };
@group(0) @binding(0) var<uniform> pp: PP;
@group(0) @binding(1) var<uniform> crop: vec4<i32>;
@group(0) @binding(2) var gb: texture_2d<f32>;
@group(0) @binding(3) var f0: texture_2d<f32>;
@group(0) @binding(4) var f1: texture_2d<f32>;
@group(0) @binding(5) var f2: texture_2d<f32>;
@group(0) @binding(6) var smp: sampler;
@group(0) @binding(7) var<storage, read_write> X: array<vec4<S>>;
${CROP}
const D = ${down}; const OWP = ${GW / down}; const OHP = ${GH / down};
fn tileAt(k: i32, x: i32, y: i32) -> vec4<f32> { return textureLoad(gb, vec2<i32>(x + k * ${GW}, y), 0); }
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = cropOf(OWP, OHP);
  let x = c.x + i32(id.x); let y = c.y + i32(id.y);
  if (x >= c.z || y >= c.w) { return; }
  var col = vec3<f32>(0.0); var nrm = vec3<f32>(0.0); var d = 0.0;
  for (var j = 0; j < D; j++) {
    for (var i = 0; i < D; i++) {
      col += tileAt(0, x * D + i, y * D + j).rgb; nrm += tileAt(1, x * D + i, y * D + j).rgb; d += tileAt(2, x * D + i, y * D + j).r;
    }
  }
  let inv = 1.0 / f32(D * D);
  col *= inv; d *= inv;
  let n = nrm * inv * 2.0 - 1.0;
  let a = tileAt(3, x * D, y * D);
  let b = tileAt(4, x * D, y * D);
  let u = a.r + a.g / 255.0; let v = a.b + b.r / 255.0;
  let part = round(b.g * 255.0);
  let mask = select(0.0, 1.0, b.b > 0.5);
  let cell = vec2<f32>(part % pp.cols, floor(part / pp.cols));
  let uv = (cell + clamp(vec2<f32>(u, v), vec2<f32>(0.5 / pp.tile), vec2<f32>(1.0 - 0.5 / pp.tile))) / vec2<f32>(pp.cols, pp.rows);
  let s0 = textureSampleLevel(f0, smp, uv, 0.0).rgb;
  let s1 = textureSampleLevel(f1, smp, uv, 0.0).rgb;
  let s2 = textureSampleLevel(f2, smp, uv, 0.0).rgb;
  let i = y * OWP + x; let L = OWP * OHP;
  X[i] = vec4<S>(vec4<f32>(col, n.x));
  X[i + L] = vec4<S>(vec4<f32>(n.y, n.z, d, mask));
  X[i + 2 * L] = vec4<S>((pp.lo0 + vec4<f32>(s0, s1.r) * (pp.hi0 - pp.lo0)) * mask);
  X[i + 3 * L] = vec4<S>((pp.lo1 + vec4<f32>(s1.g, s1.b, s2.r, s2.g) * (pp.hi1 - pp.lo1)) * mask);
${LIT ? `  // Light: cos(light, surface), key colour, ambient colour, on her (train_light.py light_channels).
  let nn = n / max(length(n), 1e-3);
  let ndl = dot(nn, pp.ldir.xyz) * mask;
  X[i + 4 * L] = vec4<S>(vec4<f32>(pp.sinP, pp.cosP, ndl, pp.key.r * mask));
  X[i + 5 * L] = vec4<S>(vec4<f32>(pp.key.g, pp.key.b, pp.amb.r, pp.amb.g) * mask);
  X[i + 6 * L] = vec4<S>(vec4<f32>(pp.amb.b * mask, 0.0, 0.0, 0.0));` : `  X[i + 4 * L] = vec4<S>(vec4<f32>(pp.sinP, pp.cosP, 0.0, 0.0));`}
}`);
  const packParams = new Float32Array(36);
  packParams.set([0, 1, net.tile, net.cols, net.rows, 0, 0, 0], 0);
  packParams.set(net.feat_lo.slice(0, 4), 8); packParams.set(net.feat_lo.slice(4, 8), 12);
  packParams.set(net.feat_hi.slice(0, 4), 16); packParams.set(net.feat_hi.slice(4, 8), 20);
  const packBuf = buf(packParams.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const packs = (net.sharp ? [[2, T.x], [1, T.xf]] : [[1, T.x]]).map(([down, out]) => {
    const pp = packPipeFor(down);
    return { pipe: pp, out, group: device.createBindGroup({ layout: pp.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: packBuf } }, { binding: 1, resource: { buffer: cropBuf } },
      { binding: 2, resource: gbuf.createView() }, { binding: 3, resource: featTex[0].createView() },
      { binding: 4, resource: featTex[1].createView() }, { binding: 5, resource: featTex[2].createView() },
      { binding: 6, resource: linear }, { binding: 7, resource: { buffer: out.buf } }] }) };
  });

  // ---- convolution ----
  // math: 'f32' multiplies in f32; 'f16' (with shader-f16) multiplies in f16, summing each input
  // chunk's 9 taps x 4 channels in f16 and the chunks in f32; 'emu' is the f16 scheme in f32 with
  // every value rounded to f16 where 'f16' would round it (to check the precision without f16).
  const convCode = math => `
alias M = ${math === 'f16' ? 'f16' : 'f32'};
${math === 'emu' ? 'fn Q(x: vec4<f32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(pack2x16float(x.xy)), unpack2x16float(pack2x16float(x.zw))); }' : ''}
struct P { W: i32, H: i32, Wa: i32, Ha: i32, la: i32, L: i32, T: i32, K: i32, wbase: i32, row0: i32, groups: i32, act: i32, skip: i32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> crop: vec4<i32>;
@group(0) @binding(2) var<storage, read> A: array<vec4<S>>;
@group(0) @binding(3) var<storage, read> B: array<vec4<S>>;
@group(0) @binding(4) var<storage, read_write> O: array<vec4<S>>;
@group(0) @binding(5) var<storage, read> Wt: array<vec4<S>>;
@group(0) @binding(6) var<storage, read> aff: array<vec4<f32>>;
${CROP}
fn loadA(l: i32, x: i32, y: i32) -> vec4<f32> {
  if (p.Wa == p.W) { return vec4<f32>(A[(l * p.Ha + y) * p.Wa + x]); }
  // Bilinear upsample (align_corners = false), as the decoder's F.interpolate.
  let sx = max((f32(x) + 0.5) * f32(p.Wa) / f32(p.W) - 0.5, 0.0);
  let sy = max((f32(y) + 0.5) * f32(p.Ha) / f32(p.H) - 0.5, 0.0);
  let x0 = i32(floor(sx)); let y0 = i32(floor(sy));
  let x1 = min(x0 + 1, p.Wa - 1); let y1 = min(y0 + 1, p.Ha - 1);
  let fx = sx - f32(x0); let fy = sy - f32(y0);
  let base = l * p.Ha;
  let a00 = vec4<f32>(A[(base + y0) * p.Wa + x0]); let a10 = vec4<f32>(A[(base + y0) * p.Wa + x1]);
  let a01 = vec4<f32>(A[(base + y1) * p.Wa + x0]); let a11 = vec4<f32>(A[(base + y1) * p.Wa + x1]);
  return mix(mix(a00, a10, fx), mix(a01, a11, fx), fy);
}
fn loadIn(l: i32, x: i32, y: i32) -> vec4<f32> {
  if (l < p.la) { return loadA(l, x, y); }
  return vec4<f32>(B[((l - p.la) * p.H + y) * p.W + x]);
}
fn gelu(x: vec4<f32>) -> vec4<f32> {
  return 0.5 * x * (1.0 + tanh(clamp(0.7978845608 * (x + 0.044715 * x * x * x), vec4<f32>(-10.0), vec4<f32>(10.0))));
}
// Shared-memory tiling: a workgroup of 8 x 8 threads makes a 32 x 8 output tile. For each input
// layer it loads the tile's 34 x 10 input region once (upsampling there, when the input is smaller),
// then every thread reads its taps from workgroup memory.
const TW = ${8 * PX + 2};
const TH = 10;
var<workgroup> tile: array<vec4<M>, ${(8 * PX + 2) * 10}>;
var<workgroup> anyMask: u32;
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let c = cropOf(p.W, p.H);
  let ox0 = c.x + i32(wid.x) * ${8 * PX}; let oy0 = c.y + i32(wid.y) * 8;
  let lx = i32(lid.x); let ly = i32(lid.y);
  let x0 = ox0 + lx * ${PX}; let y = oy0 + ly;
  let inside = x0 < c.z && y < c.w;
  let g0 = i32(wid.z) * ${G};
  let ng = min(${G}, p.groups - g0);
  // Full-resolution layers (1-pixel reach): a tile with none of her coverage within 8 pixels draws
  // nothing there, so it skips the convolution (the head writes transparent; its 3 x 3 input layer
  // makes the same call, so what it leaves unwritten is never read). Coverage: B's 8th channel.
  if (p.skip > 0) {
    if (lid.x == 0u && lid.y == 0u) { anyMask = 0u; }
    workgroupBarrier();
    for (var i = i32(lid.y) * 8 + i32(lid.x); i < ${8 * PX + 16} * 24; i += 64) {
      let gx = ox0 - 8 + i % ${8 * PX + 16}; let gy = oy0 - 8 + i / ${8 * PX + 16};
      if (gx >= 0 && gx < p.W && gy >= 0 && gy < p.H && f32(B[(p.H + gy) * p.W + gx].w) > 0.5) { anyMask = 1u; }
    }
    if (workgroupUniformLoad(&anyMask) == 0u) {
      if (p.act == 2 && inside) {
        for (var k = 0; k < ng; k++) {
          for (var q = 0; q < ${PX}; q++) { if (x0 + q < c.z) { O[((g0 + k) * p.H + y) * p.W + x0 + q] = vec4<S>(0.0); } }
        }
      }
      return;
    }
  }
  var acc: array<array<vec4<f32>, ${G}>, ${PX}>;
  for (var k = 0; k < ng; k++) {
    let b = aff[(p.row0 + g0 + k) * 3];
    for (var q = 0; q < ${PX}; q++) { acc[q][k] = b; }
  }
  let r = p.K / 2;
  let li = ly * 8 + lx;
  for (var l = 0; l < p.L; l++) {
    for (var i = li; i < TW * TH; i += 64) {
      let gx = ox0 - 1 + i % TW; let gy = oy0 - 1 + i / TW;
      if (gx >= 0 && gx < p.W && gy >= 0 && gy < p.H) { tile[i] = vec4<M>(${math === 'emu' ? 'Q(loadIn(l, gx, gy))' : 'loadIn(l, gx, gy)'}); } else { tile[i] = vec4<M>(0.0); }
    }
    workgroupBarrier();
    if (inside) {
      // This input chunk's taps, summed in M, then added to the f32 sums.
      var part: array<array<vec4<M>, ${G}>, ${PX}>;
      for (var k = 0; k < ${G}; k++) { for (var q = 0; q < ${PX}; q++) { part[q][k] = vec4<M>(0.0); } }
      for (var t = 0; t < p.T; t++) {
        let dy = t / p.K - r; let dx = t % p.K - r;
        let rowBase = (ly + 1 + dy) * TW + lx * ${PX} + 1 + dx;
        var v: array<vec4<M>, ${PX}>;
        for (var q = 0; q < ${PX}; q++) { v[q] = tile[rowBase + q]; }
        for (var k = 0; k < ng; k++) {
          let wi = p.wbase + (((g0 + k) * p.L + l) * p.T + t) * 4;
          let w0 = vec4<M>(Wt[wi]); let w1 = vec4<M>(Wt[wi + 1]); let w2 = vec4<M>(Wt[wi + 2]); let w3 = vec4<M>(Wt[wi + 3]);
          for (var q = 0; q < ${PX}; q++) {
            ${math === 'emu' ? 'part[q][k] = Q(part[q][k] + Q(vec4<f32>(dot(w0, v[q]), dot(w1, v[q]), dot(w2, v[q]), dot(w3, v[q]))));'
              : 'part[q][k] += vec4<M>(dot(w0, v[q]), dot(w1, v[q]), dot(w2, v[q]), dot(w3, v[q]));'}
          }
        }
      }
      for (var k = 0; k < ng; k++) { for (var q = 0; q < ${PX}; q++) { acc[q][k] += vec4<f32>(part[q][k]); } }
    }
    workgroupBarrier();
  }
  if (!inside) { return; }
  for (var k = 0; k < ng; k++) {
    let row = (p.row0 + g0 + k) * 3;
    let sc = aff[row + 1]; let sh = aff[row + 2];
    for (var q = 0; q < ${PX}; q++) {
      let x = x0 + q;
      if (x >= c.z) { break; }
      var a = acc[q][k];
      if (p.act == 1) { a = gelu(a); } else { a = 1.0 / (1.0 + exp(-a)); }
      O[((g0 + k) * p.H + y) * p.W + x] = vec4<S>(a * sc + sh);
    }
  }
}`;
  const convPipes = {};
  const mathModes = math === 'emu' ? ['emu', 'f32'] : f16 && math !== 'f32' ? ['f16', 'f32'] : ['f32'];
  for (const m of mathModes) convPipes[m] = pipe(convCode(m));
  let mathMode = mathModes[0];

  const poolPipe = pipe(`
struct Q { W: i32, H: i32, layers: i32, pad: i32 };
@group(0) @binding(0) var<uniform> p: Q;
@group(0) @binding(1) var<uniform> crop: vec4<i32>;
@group(0) @binding(2) var<storage, read> Sx: array<vec4<S>>;
@group(0) @binding(3) var<storage, read_write> O: array<vec4<S>>;
${CROP}
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = cropOf(p.W, p.H);
  let x = c.x + i32(id.x); let y = c.y + i32(id.y); let l = i32(id.z);
  if (x >= c.z || y >= c.w || l >= p.layers) { return; }
  let W2 = p.W * 2; let base = l * p.H * 2;
  let s = vec4<f32>(Sx[(base + 2 * y) * W2 + 2 * x]) + vec4<f32>(Sx[(base + 2 * y) * W2 + 2 * x + 1])
        + vec4<f32>(Sx[(base + 2 * y + 1) * W2 + 2 * x]) + vec4<f32>(Sx[(base + 2 * y + 1) * W2 + 2 * x + 1]);
  O[(l * p.H + y) * p.W + x] = vec4<S>(0.25 * s);
}`);

  // ---- the pass list, with static bind groups ----
  const steps = [];
  for (const l of net.layers) {
    const out = T[l.out], A = T[l.inputs[0]], B = T[l.inputs[1] ?? l.inputs[0]];
    const skip = skipEmpty && net.sharp && (l.name === 'srh' || l.name === 'srhead') ? 1 : 0;
    const params = uniform(new Int32Array([out.w, out.h, A.w, A.h, l.la, l.L, l.T, l.k, l.wbase, l.row, l.groups, l.act === 'gelu' ? 1 : 2, skip]));
    const groups = {};
    for (const m of mathModes) groups[m] = device.createBindGroup({ layout: convPipes[m].getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: params } }, { binding: 1, resource: { buffer: cropBuf } },
      { binding: 2, resource: { buffer: A.buf } }, { binding: 3, resource: { buffer: B.buf } },
      { binding: 4, resource: { buffer: out.buf } }, { binding: 5, resource: { buffer: wBuf } },
      { binding: 6, resource: { buffer: affBuf } }] });
    steps.push({ name: l.name, conv: true, pipes: convPipes, groups, out, z: Math.ceil(l.groups / G), px: PX,
      get pipe() { return this.pipes[mathMode]; }, get group() { return this.groups[mathMode]; } });
    const pooled = Object.entries(net.pools).find(([, src]) => src === l.out);
    if (pooled) {
      const dst = T[pooled[0]];
      const pg = device.createBindGroup({ layout: poolPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: uniform(new Int32Array([dst.w, dst.h, dst.layers, 0])) } },
        { binding: 1, resource: { buffer: cropBuf } }, { binding: 2, resource: { buffer: out.buf } },
        { binding: 3, resource: { buffer: dst.buf } }] });
      steps.push({ name: pooled[0], pipe: poolPipe, group: pg, out: dst, z: dst.layers, px: 1 });
    }
  }

  // ---- display ----
  const showMod = device.createShaderModule({ code: head + `
struct V { box: vec4<f32>, crop: vec4<f32>, top: vec4<f32>, bottom: vec4<f32>, view: vec2<f32>, mode: f32, pad: f32 };
@group(0) @binding(0) var<uniform> v: V;
@group(0) @binding(1) var<storage, read> R: array<vec4<S>>;
@group(0) @binding(2) var gb: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}
fn px(x: i32, y: i32) -> vec4<f32> { return vec4<f32>(R[clamp(y, 0, ${OH - 1}) * ${OW} + clamp(x, 0, ${OW - 1})]); }
@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = pos.xy / v.view;
  let bg = mix(v.top.rgb, v.bottom.rgb, uv.y);
  let p = (uv - v.box.xy) / v.box.zw;
  if (p.x < 0.0 || p.y < 0.0 || p.x > 1.0 || p.y > 1.0) { return vec4<f32>(bg, 1.0); }
  let q = p * vec2<f32>(${OW}.0, ${OH}.0) - 0.5;
  let qc = p * vec2<f32>(${NW}.0, ${NH}.0);
  if (v.mode > 0.5) {
    let g = p * vec2<f32>(${GW}.0, ${GH}.0);
    let c = textureLoad(gb, vec2<i32>(clamp(i32(g.x), 0, ${GW - 1}), clamp(i32(g.y), 0, ${GH - 1})), 0).rgb;
    let green = step(0.25, c.g - max(c.r, c.b));
    return vec4<f32>(mix(c, bg, green), 1.0);
  }
  if (qc.x < v.crop.x || qc.y < v.crop.y || qc.x > v.crop.z || qc.y > v.crop.w) { return vec4<f32>(bg, 1.0); }
  let i = vec2<i32>(floor(q)); let f = q - floor(q);
  let d = mix(mix(px(i.x, i.y), px(i.x + 1, i.y), f.x), mix(px(i.x, i.y + 1), px(i.x + 1, i.y + 1), f.x), f.y);
  return vec4<f32>(mix(bg, d.rgb, d.a), 1.0);  // the canvas is not sRGB: values go out as drawn
}` });
  const showPipe = device.createRenderPipeline({ layout: 'auto', vertex: { module: showMod, entryPoint: 'vs' },
    fragment: { module: showMod, entryPoint: 'fs', targets: [{ format }] }, primitive: { topology: 'triangle-list' } });
  const showBuf = buf(96, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const showGroup = device.createBindGroup({ layout: showPipe.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: showBuf } }, { binding: 1, resource: { buffer: T.rgba.buf } },
    { binding: 2, resource: gbuf.createView() }] });

  const query = timestamps ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null;
  // #gpuprof: every dispatch in its own pass with timestamps, averaged per stage (layerTimes()).
  const prof = profile && timestamps ? (() => {
    const names = [...packs.map((pk, i) => `pack ${i ? 'xf' : 'x'}`), ...steps.map(st => st.name)];
    const n = names.length;
    return { names, n, q: device.createQuerySet({ type: 'timestamp', count: 2 * n }), qb: buf(16 * n, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC),
      rb: buf(16 * n, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST), busy: false, sum: new Float64Array(n), count: 0 };
  })() : null;
  const qBuf = timestamps ? buf(16, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC) : null;
  const qRead = timestamps ? buf(16, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST) : null;
  let qBusy = false, gpuMs = null;
  // Frames in flight: the page starts a frame only while fewer than MAX_IN_FLIGHT are unfinished, so
  // work can't queue up behind a GPU that takes longer than the display interval (without this the
  // latency, then the frame rate, snowballs). Two lets the CPU prepare one frame while the GPU draws one.
  const MAX_IN_FLIGHT = 2;
  let inFlight = 0;

  // One frame: copy the G-buffer canvas in, run the network on the crop, draw.
  // lighting (models with light channels): 9 floats, the light's view-space direction, key colour x strength
  // and ambient colour x strength relative to the films' light (train_light.py light_params).
  // rect: the part of each G-buffer tile to copy in ([x, y, w, h], top-down); the whole tile by default.
  // view: [zoom, x, y], the drawing magnified about the canvas's top-left and moved by (x, y) canvas
  // widths/heights; the display pass does it, so zoom costs nothing and redraw() can redo just that.
  let lastShow = null;
  function show({ crop, mode, top, bottom, view = [1, 0, 0] }, enc = null) {
    lastShow = { crop, mode, top, bottom, view };
    const own = !enc; if (own) enc = device.createCommandEncoder();
    const W = canvas.width, H = canvas.height;
    const s = Math.min(W / NW, H / NH), bw = NW * s / W, bh = NH * s / H, [z, vx, vy] = view;
    device.queue.writeBuffer(showBuf, 0, new Float32Array([vx + z * (1 - bw) / 2, vy + z * (1 - bh) / 2, z * bw, z * bh, ...crop, ...top, 1, ...bottom, 1, W, H, mode, 0]));
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
    rp.setPipeline(showPipe); rp.setBindGroup(0, showGroup); rp.draw(3); rp.end();
    if (own) device.queue.submit([enc.finish()]);
  }
  // The last frame again under a new view (zoom, pan, resize): the display pass only.
  function redraw(view) { if (lastShow) show({ ...lastShow, view }); }
  function frame({ source, crop, rect = null, pitch, mode, top, bottom, view, lighting = null, limit = Infinity, noCopy = false, noShow = false }) {
    if (!noCopy) {
      if (!rect) device.queue.copyExternalImageToTexture({ source }, { texture: gbuf }, [GW * TILES, GH]);
      else for (let k = 0; k < TILES; k++) {
        const o = { x: k * GW + rect[0], y: rect[1] };
        device.queue.copyExternalImageToTexture({ source, origin: o }, { texture: gbuf, origin: o }, [rect[2], rect[3]]);
      }
    }
    device.queue.writeBuffer(cropBuf, 0, new Int32Array(crop));
    packParams[0] = Math.sin(pitch); packParams[1] = Math.cos(pitch);
    if (lighting) { packParams.set(lighting.slice(0, 3), 24); packParams.set(lighting.slice(3, 6), 28); packParams.set(lighting.slice(6, 9), 32); }
    device.queue.writeBuffer(packBuf, 0, packParams);
    const enc = device.createCommandEncoder();
    const span = (w, scale, px) => {
      const s = w / NW;
      return [Math.ceil((Math.ceil(crop[2] * s) + 1 - Math.floor(crop[0] * s)) / px / 8), Math.ceil((Math.ceil(crop[3] * s) + 1 - Math.floor(crop[1] * s)) / 8)];
    };
    if (mode === 0 && prof && !prof.busy) {
      const list = [...packs.map(pk => [pk, span(pk.out.w, 1, 1), 1]), ...steps.map(st => [st, span(st.out.w, 1, st.px), st.z])];
      list.forEach(([st, xy, z], i) => {
        const pass = enc.beginComputePass({ timestampWrites: { querySet: prof.q, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1 } });
        pass.setPipeline(st.pipe); pass.setBindGroup(0, st.group); pass.dispatchWorkgroups(...xy, z); pass.end();
      });
      enc.resolveQuerySet(prof.q, 0, 2 * prof.n, prof.qb, 0); enc.copyBufferToBuffer(prof.qb, 0, prof.rb, 0, 16 * prof.n);
      prof.pending = true;
    } else if (mode === 0) {
      const pass = enc.beginComputePass(query && !qBusy ? { timestampWrites: { querySet: query, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : undefined);
      if (limit > 0) for (const pk of packs) {
        pass.setPipeline(pk.pipe); pass.setBindGroup(0, pk.group);
        pass.dispatchWorkgroups(...span(pk.out.w, 1, 1), 1);
      }
      for (const st of steps.slice(0, Math.max(0, limit - 1))) {
        pass.setPipeline(st.pipe); pass.setBindGroup(0, st.group);
        pass.dispatchWorkgroups(...span(st.out.w, 1, st.px), st.z);
      }
      pass.end();
      if (query && !qBusy) { enc.resolveQuerySet(query, 0, 2, qBuf, 0); enc.copyBufferToBuffer(qBuf, 0, qRead, 0, 16); }
    }
    const readProf = () => {
      if (!prof?.pending) return;
      prof.pending = false; prof.busy = true;
      prof.rb.mapAsync(GPUMapMode.READ).then(() => {
        const t = new BigInt64Array(prof.rb.getMappedRange());
        let total = 0;
        for (let i = 0; i < prof.n; i++) { const ms = Number(t[2 * i + 1] - t[2 * i]) / 1e6; prof.sum[i] += ms; total += ms; }
        prof.rb.unmap(); prof.count++; prof.busy = false;
        gpuMs = gpuMs == null ? total : gpuMs * 0.8 + total * 0.2;
      }).catch(() => { prof.busy = false; });
    };
    const done = () => { inFlight++; device.queue.onSubmittedWorkDone().then(() => { inFlight--; }); };
    if (noShow) { device.queue.submit([enc.finish()]); done(); readProf(); return; }  // checks: headless browsers can't present
    show({ crop, mode, top, bottom, view }, enc);
    device.queue.submit([enc.finish()]);
    done();
    readProf();
    if (query && !qBusy && mode === 0 && !prof) {
      qBusy = true;
      qRead.mapAsync(GPUMapMode.READ).then(() => {
        const t = new BigInt64Array(qRead.getMappedRange());
        const ms = Number(t[1] - t[0]) / 1e6;
        qRead.unmap(); qBusy = false;
        if (ms > 0 && ms < 1000) gpuMs = gpuMs == null ? ms : gpuMs * 0.8 + ms * 0.2;
      }).catch(() => { qBusy = false; });
    }
  }

  // For checks: one layer of a tensor as float32, row 0 at the top.
  async function readTensor(name, layer) {
    const t = T[name], n = t.w * t.h;
    const read = buf(n * bytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(t.buf, layer * n * bytes, read, 0, n * bytes);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const raw = read.getMappedRange();
    const out = f16 ? Float32Array.from(new Uint16Array(raw), halfToFloat) : new Float32Array(raw.slice(0));
    read.unmap(); read.destroy();
    return { w: t.w, h: t.h, data: out };
  }

  // The display pass drawn offscreen and read back (checks: headless browsers can't present), after a
  // frame(): W must be a multiple of 64. Returns RGBA bytes.
  async function readShow({ crop, mode = 0, top = [1, 1, 1], bottom = [1, 1, 1], W = 512, H = 910, view = [1, 0, 0] }) {
    const tex = device.createTexture({ size: [W, H], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const s = Math.min(W / NW, H / NH), bw = NW * s / W, bh = NH * s / H, [z, vx, vy] = view;
    device.queue.writeBuffer(showBuf, 0, new Float32Array([vx + z * (1 - bw) / 2, vy + z * (1 - bh) / 2, z * bw, z * bh, ...crop, ...top, 1, ...bottom, 1, W, H, mode, 0]));
    const read = buf(W * H * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    const enc = device.createCommandEncoder();
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: tex.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
    rp.setPipeline(showPipe); rp.setBindGroup(0, showGroup); rp.draw(3); rp.end();
    enc.copyTextureToBuffer({ texture: tex }, { buffer: read, bytesPerRow: W * 4 }, [W, H]);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const px = new Uint8Array(read.getMappedRange().slice(0));
    read.unmap(); read.destroy(); tex.destroy();
    if (format === 'bgra8unorm') for (let i = 0; i < px.length; i += 4) [px[i], px[i + 2]] = [px[i + 2], px[i]];
    return { w: W, h: H, data: px };
  }

  // Per-stage GPU ms averaged since the last call (#gpuprof), slowest first; resets.
  function layerTimes() {
    if (!prof || !prof.count) return [];
    const rows = prof.names.map((name, i) => ({ stage: name, ms: +(prof.sum[i] / prof.count).toFixed(2) }));
    const total = rows.reduce((a, r) => a + r.ms, 0);
    rows.forEach(r => { r.share = `${(100 * r.ms / total).toFixed(0)}%`; });
    prof.sum.fill(0); prof.count = 0;
    return rows.sort((a, b) => b.ms - a.ms);
  }

  return { frame, redraw, readTensor, readShow, layerTimes, light: LIT, mathModes,
    get math() { return mathMode; }, set math(m) { if (convPipes[m]) mathMode = m; }, get busy() { return inFlight >= MAX_IN_FLIGHT; }, steps: steps.length, gbufSize: [GW, GH], outSize: [OW, OH], f16, get gpuMs() { return gpuMs; }, device };
}

function halfToFloat(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + f / 1024);
}
