// GPU fast path for the nanotss (Tree-SSM) shader: the same network as infer.c, in WebGL2 fragment
// shaders. WebGL2 with float render targets is all it needs (no WebGPU, no compute), so it runs on
// iPhones as well as desktops. The G-buffer tiles stay on the GPU: three.js draws them into render
// targets and the passes below read them directly.
//
// Per frame, inside the crop box: a background test and two passes of the chessboard distance to
// her (the CPU kernel's background skipping, window 16), the embedding (two passes of 8 layers),
// per block three passes, the head, then the display pass onto the canvas.
//   block pass A: the ternary stencil on all 64 channels, quant8, FFN1 (ternary weights from a
//     uniform buffer, integer sums), GELU, quant8 again -> 64 int8 codes packed 4 to a uint, + scale
//   block passes B0, B1: FFN2 on those codes for 32 channels each, plus the residual (the stencil
//     output again: it's channel-local, so 8 layers cost little and no tensor holds it)
// Activations are float16 textures, 64 channels as 16 RGBA layers; arithmetic is float32. A token
// whose receptive field holds only background has a per-block constant (computed here in JS from
// the weights, per frame since the camera pitch is an input) and isn't drawn.

// net.png: export_nts.py's net.bin packed 3 bytes a pixel into an opaque RGB PNG after a 4-byte length.
async function loadImage(url) {
  const img = new Image();
  img.src = url;
  await img.decode();
  return img;
}
function pixelsOf(img) {
  const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
  const g = c.getContext('2d', { willReadFrequently: true });
  g.drawImage(img, 0, 0);
  return g.getImageData(0, 0, img.width, img.height).data;
}
function unpackNet(img) {
  const px = pixelsOf(img), rgb = new Uint8Array(img.width * img.height * 3);
  for (let i = 0, j = 0; i < px.length; i += 4) { rgb[j++] = px[i]; rgb[j++] = px[i + 1]; rgb[j++] = px[i + 2]; }
  const n = rgb[0] | (rgb[1] << 8) | (rgb[2] << 16) | (rgb[3] << 24);
  return rgb.buffer.slice(4, 4 + n);
}

const DW = 64, KT = 9, CIN = 18, FCH = 8, FINE = 512, COARSE = 64, MARGIN = 6, WIN = 16;

// net.bin (export_nts.py) into arrays.
function parseNet(buf) {
  const dv = new DataView(buf); let o = 4;
  const i32 = () => { const v = dv.getInt32(o, true); o += 4; return v; };
  const f32 = n => { const a = new Float32Array(buf.slice(o, o + 4 * n)); o += 4 * n; return a; };
  const i8 = n => { const a = new Int8Array(buf.slice(o, o + n)); o += n; return a; };
  if (new TextDecoder().decode(new Uint8Array(buf, 0, 4)) !== 'NTSV') throw new Error('net: not NTSV');
  const NB = i32(), P = i32(), DIL = Array.from({ length: NB }, i32);
  const We = f32(DW * CIN), be = f32(DW), blocks = [];
  for (let b = 0; b < NB; b++) blocks.push({ q: i8((KT + 4) * DW), gate: f32(DW), b1: f32(DW), b2: f32(DW), s1: f32(DW), s2: f32(DW), c1: i8(DW * DW), c2: i8(DW * DW) });
  const Wh = f32(4 * DW), bh = f32(4), lo = f32(2 * FCH), st = f32(2 * FCH);
  const coarse = new Uint8Array(buf.slice(o, o + P * COARSE * COARSE * FCH));
  return { NB, P, DIL, We, be, blocks, Wh, bh, lo, st, coarse };
}

// The background's value after each block, in JS: infer.c's token() with every neighbour the same.
function backgroundStates(net, input) {
  const { NB, We, be, blocks, Wh, bh } = net, f = Math.fround;
  const roundEven = x => { const r = Math.round(x); return Math.abs(x - Math.trunc(x)) === 0.5 && r % 2 !== 0 ? r - 1 : r; };
  const quant = v => {
    let mx = 0; for (const x of v) mx = Math.max(mx, Math.abs(x)); mx = Math.max(mx, 1e-5);
    const sc = f(127 / mx); return { c: v.map(x => Math.min(127, Math.max(-128, roundEven(f(x * sc))))), xs: f(1 / sc) };
  };
  const gelu = v => { const z = Math.abs(v * 0.70710678), e = Math.exp(-0.5 * v * v), t = 1 / (1 + 0.3275911 * z);
    let p = 1.061405429; p = p * t - 1.453152027; p = p * t + 1.421413741; p = p * t - 0.284496736; p = p * t + 0.254829592;
    let er = 1 - p * t * e; if (v < 0) er = -er; return v * (0.5 * (1 + er)); };
  const states = [];
  let x = Array.from({ length: DW }, (_, j) => { let s = 0; for (let i = 0; i < CIN; i++) s += We[j * CIN + i] * input[i]; return be[j] + s; });
  states.push(x);
  for (let b = 0; b < NB; b++) {
    const k = blocks[b], t = new Array(DW);
    for (let L = 0; L < 16; L++) {
      const dw = [0, 1, 2, 3].map(e => { let z = 0; for (let q = 0; q < KT; q++) z += k.q[q * DW + 4 * L + e] * x[4 * L + e]; return z / 3; });
      for (let e = 0; e < 4; e++) {
        const i = 4 * L + e; let m = 0;
        for (let j = 0; j < 4; j++) m += k.q[(KT + j) * DW + i] * dw[j];
        t[i] = k.gate[i] * (m * 0.5) + x[i];
      }
    }
    const q1 = quant(t), g = new Array(DW);
    for (let r = 0; r < DW; r++) { let a = 0; for (let i = 0; i < DW; i++) a += q1.c[i] * k.c1[r * DW + i]; g[r] = gelu(a * k.s1[r] * q1.xs + k.b1[r]); }
    const q2 = quant(g);
    x = Array.from({ length: DW }, (_, r) => { let a = 0; for (let i = 0; i < DW; i++) a += q2.c[i] * k.c2[r * DW + i]; return a * k.s2[r] * q2.xs + k.b2[r] + t[r]; });
    states.push(x);
  }
  const out = [0, 1, 2, 3].map(c => { let z = bh[c]; for (let i = 0; i < DW; i++) z += Wh[c * DW + i] * x[i]; return 1 / (1 + Math.exp(-z)); });
  return { states, out };
}

const HEAD = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D; precision highp sampler2DArray; precision highp usampler2DArray;
uniform ivec4 crop;   // x0, y0, x1, y1 (GL pixels, y up, end exclusive)
uniform sampler2D distT;
int distAt(ivec2 p) { return int(texelFetch(distT, p, 0).r * 255.0 + 0.5); }
bool inCrop(ivec2 p) { return p.x >= crop.x && p.y >= crop.y && p.x < crop.z && p.y < crop.w; }
`;
const outs = (n, type = 'vec4') => Array.from({ length: n }, (_, k) => `layout(location = ${k}) out ${type} o${k};`).join('\n');
const VS = `#version 300 es
void main() { vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }`;

// The ternary stencil (tc_one) for one layer of 4 channels; taps in training order, rows top-down.
const BLOCK_COMMON = `
layout(std140) uniform Blk { vec4 Q[208]; vec4 gate[16]; vec4 b1[16]; vec4 b2[16]; vec4 s1[16]; vec4 s2[16]; vec4 bg[16]; };
layout(std140) uniform Wt { vec4 W[1024]; };
uniform sampler2DArray X; uniform int dil, needPrev, needCur;
vec4 xin(ivec2 p, int L) { return (!inCrop(p) || distAt(p) > needPrev) ? bg[L] : texelFetch(X, ivec3(p, L), 0); }
vec4 stencil(ivec2 p, int L) {
  vec4 z = vec4(0.0);
  for (int k = 0; k < 9; k++) z = Q[k * 16 + L] * xin(p + ivec2(k % 3 - 1, 1 - k / 3) * dil, L) + z;
  vec4 dw = z * (1.0 / 3.0), m = vec4(0.0);
  m = Q[9 * 16 + L] * dw.x + m; m = Q[10 * 16 + L] * dw.y + m; m = Q[11 * 16 + L] * dw.z + m; m = Q[12 * 16 + L] * dw.w + m;
  return gate[L] * (m * 0.5) + xin(p, L);
}
float quant(inout vec4 v[16]) {   // v := its int8 codes; returns their scale
  float mx = 0.0;
  for (int i = 0; i < 16; i++) { vec4 a = abs(v[i]); mx = max(mx, max(max(a.x, a.y), max(a.z, a.w))); }
  mx = max(mx, 1e-5);
  float sc = 127.0 / mx;
  for (int i = 0; i < 16; i++) v[i] = clamp(roundEven(v[i] * sc), -128.0, 127.0);
  return 1.0 / sc;
}
float gelu(float v) {
  float z = abs(v * 0.70710678), e = exp(-0.5 * v * v), t = 1.0 / (1.0 + 0.3275911 * z);
  float p = 1.061405429; p = p * t - 1.453152027; p = p * t + 1.421413741; p = p * t - 0.284496736; p = p * t + 0.254829592;
  float er = 1.0 - p * t * e; if (v < 0.0) er = -er;
  return v * (0.5 * (1.0 + er));
}
float rowdot(vec4 c[16], int r) { float a = 0.0; for (int k = 0; k < 16; k++) a += dot(c[k], W[r * 16 + k]); return a; }
`;
const PASS_A = (scaleOnly, M) => HEAD + BLOCK_COMMON + `${M >= 5 ? outs(5, 'uvec4') : scaleOnly ? outs(1, 'uvec4') : outs(4, 'uvec4')}
uint pack4(vec4 c) { uvec4 i = uvec4(ivec4(c) & 255); return i.x | (i.y << 8u) | (i.z << 16u) | (i.w << 24u); }
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  if (distAt(p) > needCur) discard;
  vec4 v[16];
  for (int L = 0; L < 16; L++) v[L] = stencil(p, L);
  float xs = quant(v);
  vec4 g[16];
  for (int L = 0; L < 16; L++) {
    vec4 h;
    for (int e = 0; e < 4; e++) h[e] = rowdot(v, 4 * L + e) * s1[L][e] * xs + b1[L][e];
    g[L] = vec4(gelu(h.x), gelu(h.y), gelu(h.z), gelu(h.w));
  }
  float xs2 = quant(g);
  ${M >= 5 || !scaleOnly ? `o0 = uvec4(pack4(g[0]), pack4(g[1]), pack4(g[2]), pack4(g[3]));
  o1 = uvec4(pack4(g[4]), pack4(g[5]), pack4(g[6]), pack4(g[7]));
  o2 = uvec4(pack4(g[8]), pack4(g[9]), pack4(g[10]), pack4(g[11]));
  o3 = uvec4(pack4(g[12]), pack4(g[13]), pack4(g[14]), pack4(g[15]));` : ''}
  ${M >= 5 ? 'o4' : scaleOnly ? 'o0' : '//'} = uvec4(floatBitsToUint(xs2), 0u, 0u, 0u);
}`;
const PASS_B = M => HEAD + BLOCK_COMMON + `${outs(M)}
uniform highp usampler2DArray C; uniform int hf;
vec4 unpack4(uint u) { ivec4 i = ivec4(int(u << 24u), int(u << 16u), int(u << 8u), int(u)) >> 24; return vec4(i); }
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  if (distAt(p) > needCur) discard;
  vec4 c[16];
  for (int j = 0; j < 4; j++) { uvec4 u = texelFetch(C, ivec3(p, j), 0); c[4 * j] = unpack4(u.x); c[4 * j + 1] = unpack4(u.y); c[4 * j + 2] = unpack4(u.z); c[4 * j + 3] = unpack4(u.w); }
  float xs = uintBitsToFloat(texelFetch(C, ivec3(p, 4), 0).x);
  vec4 y[${M}];
  for (int j = 0; j < ${M}; j++) {
    int L = ${M} * hf + j;
    vec4 o;
    for (int e = 0; e < 4; e++) o[e] = rowdot(c, 4 * L + e) * s2[L][e] * xs + b2[L][e];
    y[j] = o + stencil(p, L);
  }
  ${Array.from({ length: M }, (_, j) => `o${j} = y[${j}];`).join(' ')}
}`;
// The tiles: colour (sRGB texture, read back to its bytes), normal, depth, uv-a, uv-b.
const TILES = `uniform sampler2D T0, T1, T2, T3, T4;
vec3 toSrgb(vec3 c) { return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c)); }
vec3 bytes(sampler2D t, ivec2 p) { return floor(texelFetch(t, p, 0).rgb * 255.0 + 0.5); }
vec3 colourBytes(ivec2 p) { return floor(toSrgb(texelFetch(T0, p, 0).rgb) * 255.0 + 0.5); }
`;
const EMBED = M => HEAD + TILES + `${outs(M)}
layout(std140) uniform Emb { vec4 WeT[288]; vec4 be[16]; vec4 lo[4]; vec4 st[4]; };
uniform sampler2D F0, F1, F2, C0, C1; uniform int parts, cols, hf; uniform vec2 pitch;
float fineAt(int part, ivec2 t, int c) {
  ivec2 q = ivec2(part % cols, part / cols) * ${FINE} + t;
  vec3 v = c < 3 ? texelFetch(F0, q, 0).rgb : c < 6 ? texelFetch(F1, q, 0).rgb : texelFetch(F2, q, 0).rgb;
  float b = floor((c % 3 == 0 ? v.r : c % 3 == 1 ? v.g : v.b) * 255.0 + 0.5);
  return lo[c / 4][c % 4] + b * st[c / 4][c % 4];
}
float coarseAt(int part, ivec2 t, int c) {
  ivec2 q = ivec2(part * ${COARSE} + t.x, t.y);
  vec4 v = c < 4 ? texelFetch(C0, q, 0) : texelFetch(C1, q, 0);
  float b = floor(v[c % 4] * 255.0 + 0.5);
  return lo[2 + c / 4][c % 4] + b * st[2 + c / 4][c % 4];
}
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  if (distAt(p) > 0) discard;
  vec3 col = colourBytes(p), nrm = bytes(T1, p), ua = bytes(T3, p), ub = bytes(T4, p);
  float dep = bytes(T2, p).r;
  float f[18];
  f[0] = col.r / 255.0; f[1] = col.g / 255.0; f[2] = col.b / 255.0;
  f[3] = nrm.r / 255.0 * 2.0 - 1.0; f[4] = nrm.g / 255.0 * 2.0 - 1.0; f[5] = nrm.b / 255.0 * 2.0 - 1.0;
  f[6] = dep / 255.0;
  bool mask = ub.b > 127.0;
  f[7] = mask ? 1.0 : 0.0;
  for (int c = 0; c < 8; c++) f[8 + c] = 0.0;
  if (mask) {
    float u = (ua.r + ua.g / 255.0) / 255.0, v = (ua.b + ub.r / 255.0) / 255.0;
    int part = int(ub.g) < parts ? int(ub.g) : 0;
    for (int lev = 0; lev < 2; lev++) {
      float S = lev == 0 ? ${FINE}.0 : ${COARSE}.0;
      float px = u * S - 0.5, py = v * S - 0.5, x0 = floor(px), y0 = floor(py), fx = px - x0, fy = py - y0;
      for (int j = 0; j < 2; j++) for (int i = 0; i < 2; i++) {
        ivec2 t = ivec2(int(x0) + i, int(y0) + j);
        if (t.x < 0 || t.y < 0 || float(t.x) >= S || float(t.y) >= S) continue;
        float w = (i == 1 ? fx : 1.0 - fx) * (j == 1 ? fy : 1.0 - fy);
        for (int c = 0; c < 8; c++) f[8 + c] += w * (lev == 0 ? fineAt(part, t, c) : coarseAt(part, t, c));
      }
    }
  }
  f[16] = pitch.x; f[17] = pitch.y;
  vec4 y[${M}];
  for (int j = 0; j < ${M}; j++) {
    int L = ${M} * hf + j;
    vec4 s = vec4(0.0);
    for (int i = 0; i < 18; i++) s = WeT[i * 16 + L] * f[i] + s;
    y[j] = be[L] + s;
  }
  ${Array.from({ length: M }, (_, j) => `o${j} = y[${j}];`).join(' ')}
}`;
// Background test against the crop's corner (outside her by construction), then the distance.
const DIFF = HEAD + TILES + `out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy), r = crop.xy;
  bool same = colourBytes(p) == colourBytes(r) && bytes(T1, p) == bytes(T1, r) && bytes(T2, p) == bytes(T2, r)
           && bytes(T3, p) == bytes(T3, r) && bytes(T4, p) == bytes(T4, r);
  o = vec4(same ? 0.0 : 1.0);
}`;
const DIST_H = HEAD + `uniform sampler2D M; out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy); int d = 255;
  for (int k = -${WIN}; k <= ${WIN}; k++) { ivec2 q = p + ivec2(k, 0); if (inCrop(q) && texelFetch(M, q, 0).r > 0.5) d = min(d, abs(k)); }
  o = vec4(float(d) / 255.0);
}`;
const DIST_V = HEAD + `uniform sampler2D M; out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy); int d = 255;
  for (int k = -${WIN}; k <= ${WIN}; k++) { ivec2 q = p + ivec2(0, k); if (inCrop(q)) d = min(d, max(abs(k), int(texelFetch(M, q, 0).r * 255.0 + 0.5))); }
  o = vec4(float(d) / 255.0);
}`;
const HEAD_PASS = HEAD + `out vec4 o;
uniform sampler2DArray X; uniform vec4 Wh[64]; uniform vec4 bh, pbg;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec4 q = pbg;
  if (distAt(p) <= ${MARGIN}) {
    for (int c = 0; c < 4; c++) {
      vec4 s = vec4(0.0);
      for (int L = 0; L < 16; L++) s = Wh[c * 16 + L] * texelFetch(X, ivec3(p, L), 0) + s;
      q[c] = 1.0 / (1.0 + exp(-(bh[c] + ((s.x + s.y) + (s.z + s.w)))));
    }
  }
  o = vec4(q.rgb * q.a, q.a);   // premultiplied, for filtering when it's magnified
}`;
const SHOW = `#version 300 es
precision highp float;
uniform sampler2D R, T0; uniform vec4 box; uniform vec3 top, bottom; uniform vec2 view; uniform int raw;
out vec4 o;
vec3 toSrgb(vec3 c) { return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c)); }
void main() {
  vec2 q = vec2(gl_FragCoord.x, view.y - gl_FragCoord.y);   // canvas pixels, top-down
  vec3 bg = mix(top, bottom, q.y / view.y);
  vec2 p = (q - box.xy) / box.zw;
  if (p.x < 0.0 || p.y < 0.0 || p.x > 1.0 || p.y > 1.0) { o = vec4(bg, 1.0); return; }
  vec2 uv = vec2(p.x, 1.0 - p.y);
  if (raw == 1) { vec3 c = toSrgb(texture(T0, uv).rgb); o = vec4(mix(c, bg, step(0.25, c.g - max(c.r, c.b))), 1.0); return; }
  vec4 d = texture(R, uv);
  o = vec4(bg * (1.0 - d.a) + d.rgb, 1.0);
}`;

export async function createGlShader({ THREE, renderer, NW, NH, base = 'nts/' }) {
  const gl = renderer.getContext();
  if (!(gl instanceof WebGL2RenderingContext)) throw new Error('needs WebGL2');
  if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('needs float render targets (EXT_color_buffer_float)');
  // Layers written per pass: 8 where the device has 8 draw buffers (iPhones, desktops), else 4.
  const M = gl.getParameter(gl.MAX_DRAW_BUFFERS) >= 8 ? 8 : 4;
  if (gl.getParameter(gl.MAX_DRAW_BUFFERS) < 4) throw new Error('needs 4 draw buffers');
  const [meta, netImg, ...fineImgs] = await Promise.all([fetch(base + 'meta.json').then(r => r.json()), loadImage(base + 'net.png'),
    ...[0, 1, 2].map(k => loadImage(`${base}fine${k}.png`))]);
  const net = parseNet(unpackNet(netImg));
  const { NB, P, DIL } = net;

  // ---- programs ----
  function program(fs) {
    const p = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, VS], [gl.FRAGMENT_SHADER, fs]]) {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) + '\n' + src.split('\n').map((l, i) => `${i + 1}: ${l}`).join('\n'));
      gl.attachShader(p, s);
    }
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const u = {};
    for (let i = 0, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i < n; i++) { const name = gl.getActiveUniform(p, i).name.replace('[0]', ''); u[name] = gl.getUniformLocation(p, name); }
    const blocks = {};
    for (const b of ['Blk', 'Wt', 'Emb']) { const i = gl.getUniformBlockIndex(p, b); if (i !== gl.INVALID_INDEX) { const point = { Blk: 0, Wt: 1, Emb: 2 }[b]; gl.uniformBlockBinding(p, i, point); blocks[b] = point; } }
    return { p, u, blocks };
  }
  const prog = { A: program(PASS_A(false, M)), A2: M < 5 ? program(PASS_A(true, M)) : null, B: program(PASS_B(M)), embed: program(EMBED(M)), diff: program(DIFF), distH: program(DIST_H), distV: program(DIST_V), head: program(HEAD_PASS), show: program(SHOW) };

  // ---- textures ----
  function tex2d(w, h, internal, format, type, data = null, filter = gl.NEAREST) {
    const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false); gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false); gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    if (data && data instanceof HTMLImageElement) gl.texImage2D(gl.TEXTURE_2D, 0, internal, format, type, data);
    else gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, data);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, filter], [gl.TEXTURE_MAG_FILTER, filter], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
    return t;
  }
  function array(layers, internal) {
    const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D_ARRAY, t);
    gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, internal, NW, NH, layers);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST]]) gl.texParameteri(gl.TEXTURE_2D_ARRAY, k, v);
    return { tex: t, fbos: new Map() };
  }
  const fine = fineImgs.map(img => tex2d(0, 0, gl.RGB8, gl.RGB, gl.UNSIGNED_BYTE, img));
  // coarse: [part][64][64][8] -> two RGBA8 atlases, part p at x = p * 64
  const cw = P * COARSE, cdata = [new Uint8Array(cw * COARSE * 4), new Uint8Array(cw * COARSE * 4)];
  for (let p = 0; p < P; p++) for (let y = 0; y < COARSE; y++) for (let x = 0; x < COARSE; x++) for (let c = 0; c < 8; c++)
    cdata[c >> 2][((y * cw) + p * COARSE + x) * 4 + (c & 3)] = net.coarse[((p * COARSE + y) * COARSE + x) * 8 + c];
  const coarse = cdata.map(d => tex2d(cw, COARSE, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, d));
  const X = array(16, gl.RGBA16F), Y = array(16, gl.RGBA16F), Cq = array(5, gl.RGBA32UI);
  const diffT = tex2d(NW, NH, gl.R8, gl.RED, gl.UNSIGNED_BYTE), hT = tex2d(NW, NH, gl.R8, gl.RED, gl.UNSIGNED_BYTE), distT = tex2d(NW, NH, gl.R8, gl.RED, gl.UNSIGNED_BYTE);
  const outT = tex2d(NW, NH, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, null, gl.LINEAR);
  function fbo(target, first = 0, m = 1) {
    const key = `${first}:${m}`;
    if (target.fbos?.has(key)) return target.fbos.get(key);
    const f = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    for (let k = 0; k < m; k++) {
      if (target.fbos) gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + k, target.tex, 0, first + k);
      else gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, target, 0);
    }
    gl.drawBuffers(Array.from({ length: m }, (_, k) => gl.COLOR_ATTACHMENT0 + k));
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error('framebuffer incomplete');
    if (target.fbos) target.fbos.set(key, f); else (target.__fbo = f);
    return f;
  }
  const fbo2d = t => t.__fbo || fbo(t);

  // ---- the G-buffer tiles: render targets three.js draws into (colour sRGB, multisampled like the
  // coverage renders) ----
  const tiles = [0, 1, 2, 3, 4].map(k => {
    const rt = new THREE.WebGLRenderTarget(NW, NH, { samples: 4, depthBuffer: true, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
    if (k === 0) { rt.texture.colorSpace = THREE.SRGBColorSpace; rt.texture.minFilter = rt.texture.magFilter = THREE.LinearFilter; }
    return rt;
  });
  const glTex = rt => renderer.properties.get(rt.texture).__webglTexture;

  // ---- uniform buffers: per block (stencil codes, gates, biases, scales, its background input),
  // its two weight matrices, the embedding ----
  const BLK = 5120, WT = 16384;   // bytes, multiples of the binding alignment
  const blkData = new Float32Array(NB * BLK / 4), wtData = new Float32Array(2 * NB * WT / 4);
  net.blocks.forEach((k, b) => {
    const o = b * BLK / 4;
    for (let i = 0; i < 13 * DW; i++) blkData[o + i] = k.q[i];   // Q[t * 16 + L] = codes of tap t, channels 4L..4L+3
    [k.gate, k.b1, k.b2, k.s1, k.s2].forEach((a, j) => blkData.set(a, o + 13 * DW + j * DW));
    for (const [m, c] of [[0, k.c1], [1, k.c2]]) { const w = (2 * b + m) * WT / 4; for (let i = 0; i < DW * DW; i++) wtData[w + i] = c[i]; }
  });
  const bgOff = 13 * DW + 5 * DW;   // floats into a block's slot: its background input
  const ubo = data => { const b = gl.createBuffer(); gl.bindBuffer(gl.UNIFORM_BUFFER, b); gl.bufferData(gl.UNIFORM_BUFFER, data, gl.DYNAMIC_DRAW); return b; };
  const blkBuf = ubo(blkData), wtBuf = ubo(wtData);
  const emb = new Float32Array(312 * 4);
  for (let i = 0; i < CIN; i++) for (let j = 0; j < DW; j++) emb[i * DW + j] = net.We[j * CIN + i];
  emb.set(net.be, 288 * 4); emb.set(net.lo.subarray(0, 8), 304 * 4); emb.set(net.lo.subarray(8, 16), 306 * 4);
  emb.set(net.st.subarray(0, 8), 308 * 4); emb.set(net.st.subarray(8, 16), 310 * 4);
  const embBuf = ubo(emb);
  const WhT = new Float32Array(64 * 4);
  for (let c = 0; c < 4; c++) for (let L = 0; L < 16; L++) for (let e = 0; e < 4; e++) WhT[(c * 16 + L) * 4 + e] = net.Wh[c * DW + 4 * L + e];
  // how far from her each block must run: min(what it can see, what still reaches her + margin)
  const RF = [], need = [];
  let S = DIL.reduce((a, b) => a + b, 0), r = 0;
  for (let b = 0; b < NB; b++) { r += DIL[b]; S -= DIL[b]; RF.push(r); need.push(Math.min(r, S + MARGIN)); }

  const vao = gl.createVertexArray();
  const bindTex = (unit, target, t, u, loc) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(target, t); gl.uniform1i(loc, unit); };
  let lastPitch = null, pbg = [0, 0, 0, 0];
  let crop = [0, 0, NW, NH];

  // three.js draws the tiles; call before frame(). cropTD: [x0, y0, x1, y1] network pixels, top-down.
  function renderTiles(scene, camera, passes, cropTD) {
    const [x0, y0, x1, y1] = cropTD;
    crop = [x0, NH - y1, x1, NH - y0];
    passes.forEach(([bg, mat, show], k) => {
      const rt = tiles[k];
      rt.scissor.set(crop[0], crop[1], crop[2] - crop[0], crop[3] - crop[1]); rt.scissorTest = true;
      show(); scene.background = bg; scene.overrideMaterial = mat;
      renderer.setRenderTarget(rt); renderer.render(scene, camera);
    });
    scene.overrideMaterial = null; renderer.setRenderTarget(null);
  }

  function pass(pr, target, first, m, setup) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbos ? fbo(target, first, m) : fbo2d(target));
    gl.useProgram(pr.p);
    gl.uniform4i(pr.u.crop, ...crop);
    setup(pr.u);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  // The network on the crop, then the display pass. view: { W, H, box: [x, y, w, h] in canvas pixels,
  // top: [r, g, b], bottom: [r, g, b], raw, showOnly (the last drawing again, e.g. zoomed) }.
  function frame({ pitch, view }) {
    renderer.resetState();
    gl.bindVertexArray(vao);
    gl.disable(gl.DEPTH_TEST); gl.disable(gl.BLEND); gl.disable(gl.CULL_FACE); gl.disable(gl.STENCIL_TEST);
    gl.viewport(0, 0, NW, NH);
    gl.enable(gl.SCISSOR_TEST); gl.scissor(crop[0], crop[1], crop[2] - crop[0], crop[3] - crop[1]);
    const sinp = Math.sin(pitch), cosp = Math.cos(pitch);
    if (!view.raw && !view.showOnly) {
      if (pitch !== lastPitch) {   // the background's states: the green screen's inputs at this pitch
        const bgIn = [0, 177 / 255, 64 / 255, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, Math.fround(sinp), Math.fround(cosp)];
        const { states, out } = backgroundStates(net, bgIn);
        for (let b = 0; b < NB; b++) blkData.set(states[b], b * BLK / 4 + bgOff);
        gl.bindBuffer(gl.UNIFORM_BUFFER, blkBuf); gl.bufferSubData(gl.UNIFORM_BUFFER, 0, blkData);
        pbg = out; lastPitch = pitch;
      }
      const T = tiles.map(glTex);
      const bindTiles = u => T.forEach((t, k) => bindTex(k, gl.TEXTURE_2D, t, null, u[`T${k}`]));
      pass(prog.diff, diffT, 0, 1, bindTiles);
      pass(prog.distH, hT, 0, 1, u => bindTex(0, gl.TEXTURE_2D, diffT, null, u.M));
      pass(prog.distV, distT, 0, 1, u => bindTex(0, gl.TEXTURE_2D, hT, null, u.M));
      gl.bindBufferBase(gl.UNIFORM_BUFFER, 2, embBuf);
      for (let h = 0; h < 16 / M; h++) pass(prog.embed, X, M * h, M, u => {
        bindTiles(u); bindTex(5, gl.TEXTURE_2D, distT, null, u.distT);
        fine.forEach((t, k) => bindTex(6 + k, gl.TEXTURE_2D, t, null, u[`F${k}`]));
        coarse.forEach((t, k) => bindTex(9 + k, gl.TEXTURE_2D, t, null, u[`C${k}`]));
        gl.uniform1i(u.parts, P); gl.uniform1i(u.cols, meta.cols); gl.uniform1i(u.hf, h); gl.uniform2f(u.pitch, Math.fround(sinp), Math.fround(cosp));
      });
      let x = X, y = Y;
      for (let b = 0; b < NB; b++) {
        gl.bindBufferRange(gl.UNIFORM_BUFFER, 0, blkBuf, b * BLK, BLK);
        const common = u => {
          bindTex(0, gl.TEXTURE_2D_ARRAY, x.tex, null, u.X); bindTex(1, gl.TEXTURE_2D, distT, null, u.distT);
          gl.uniform1i(u.dil, DIL[b]); gl.uniform1i(u.needPrev, b ? need[b - 1] : 0); gl.uniform1i(u.needCur, need[b]);
        };
        gl.bindBufferRange(gl.UNIFORM_BUFFER, 1, wtBuf, (2 * b) * WT, WT);
        if (M >= 5) pass(prog.A, Cq, 0, 5, common);
        else { pass(prog.A, Cq, 0, 4, common); pass(prog.A2, Cq, 4, 1, common); }   // 4 draw buffers: the scale on its own
        gl.bindBufferRange(gl.UNIFORM_BUFFER, 1, wtBuf, (2 * b + 1) * WT, WT);
        for (let h = 0; h < 16 / M; h++) pass(prog.B, y, M * h, M, u => { common(u); bindTex(2, gl.TEXTURE_2D_ARRAY, Cq.tex, null, u.C); gl.uniform1i(u.hf, h); });
        [x, y] = [y, x];
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo2d(outT));
      gl.disable(gl.SCISSOR_TEST); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); gl.enable(gl.SCISSOR_TEST);
      pass(prog.head, outT, 0, 1, u => {
        bindTex(0, gl.TEXTURE_2D_ARRAY, x.tex, null, u.X); bindTex(1, gl.TEXTURE_2D, distT, null, u.distT);
        gl.uniform4fv(u.Wh, WhT); gl.uniform4fv(u.bh, net.bh); gl.uniform4fv(u.pbg, pbg);
      });
    }
    // display
    gl.disable(gl.SCISSOR_TEST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, view.W, view.H);
    const s = prog.show; gl.useProgram(s.p);
    bindTex(0, gl.TEXTURE_2D, outT, null, s.u.R); bindTex(1, gl.TEXTURE_2D, glTex(tiles[0]), null, s.u.T0);
    gl.uniform4f(s.u.box, ...view.box); gl.uniform3f(s.u.top, ...view.top); gl.uniform3f(s.u.bottom, ...view.bottom);
    gl.uniform2f(s.u.view, view.W, view.H); gl.uniform1i(s.u.raw, view.raw ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
    renderer.resetState();
  }
  // For checks: the drawing (premultiplied RGBA bytes, rows bottom-up) and the five tiles.
  function read() {
    const px = new Uint8Array(NW * NH * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo2d(outT)); gl.readPixels(0, 0, NW, NH, gl.RGBA, gl.UNSIGNED_BYTE, px);
    const t = tiles.map(rt => { const b = new Uint8Array(NW * NH * 4); renderer.readRenderTargetPixels(rt, 0, 0, NW, NH, b); return b; });
    renderer.resetState();
    return { px, tiles: t };
  }
  return { renderTiles, frame, read, parts: meta.parts, blocks: NB };
}
