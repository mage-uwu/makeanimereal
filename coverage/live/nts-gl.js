// GPU fast path for the nanotss ternary tree shader: the same network as infer.c, in WebGL2 fragment
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
// A net with an ink branch (NTS2) takes its colour head after block NBC - 1; the branch's blocks then
// run on and the ink head writes black line art's opacity to its own texture, laid over the colour in
// the display pass (shader_nts.c --lines, trained on the film's strokes along her geometry's edges).
// Activations are float16 textures, 64 channels as 16 RGBA layers; arithmetic is float32. A token
// whose receptive field holds only background has a per-block constant (computed here in JS from
// the weights, per frame since the camera pitch is an input) and isn't drawn.

import { BACKDROP_GLSL, setBackdropUniforms } from './backdrop.js';

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
  // NTSV: colour only. NTS2: an ink branch too; the colour head reads block NBC's output, the ink head
  // (black line art's opacity) the last block's.
  // NTS3: each block's FFN is T binary trees of V levels (shader_nts.c --tree)
  const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 4));
  if (magic !== 'NTSV' && magic !== 'NTS2' && magic !== 'NTS3') throw new Error('net: not NTSV, NTS2 or NTS3');
  const NB = i32(), P = i32(), NBC = magic === 'NTS2' ? i32() : NB;
  const T = magic === 'NTS3' ? i32() : 0, V = T ? i32() : 0, TN = (1 << V) - 1, DIL = Array.from({ length: NB }, i32);
  const We = f32(DW * CIN), be = f32(DW), blocks = [];
  for (let b = 0; b < NB; b++) blocks.push(T
    ? { q: i8((KT + 4) * DW), gate: f32(DW), b2: f32(DW), bias: f32(T * TN), sin: f32(T * TN), sout: f32(T * TN), cin: i8(T * TN * DW), cout: i8(T * TN * DW) }
    : { q: i8((KT + 4) * DW), gate: f32(DW), b1: f32(DW), b2: f32(DW), s1: f32(DW), s2: f32(DW), c1: i8(DW * DW), c2: i8(DW * DW) });
  const Wh = f32(4 * DW), bh = f32(4), Wl = NBC < NB ? f32(DW) : null, bl = NBC < NB ? f32(1)[0] : 0, lo = f32(2 * FCH), st = f32(2 * FCH);
  const coarse = new Uint8Array(buf.slice(o, o + P * COARSE * COARSE * FCH));
  return { NB, NBC, P, DIL, T, V, TN, We, be, blocks, Wh, bh, Wl, bl, lo, st, coarse };
}

// ssm.bin (export_ssm.py): the SSM tree blocks; their w_in, w_out, scales and node biases take the place
// of the blocks' trees, the rest goes in block.ssm.
function parseSSM(buf, net) {
  const dv = new DataView(buf); let o = 4;
  if (new TextDecoder().decode(new Uint8Array(buf, 0, 4)) !== 'SSM1') throw new Error('ssm.bin: not SSM1');
  const i32 = () => { const v = dv.getInt32(o, true); o += 4; return v; };
  const f32 = n => { const a = new Float32Array(buf.slice(o, o + 4 * n)); o += 4 * n; return a; };
  const i8 = n => { const a = new Int8Array(buf.slice(o, o + n)); o += n; return a; };
  const R = net.T * net.TN, PD = 36, NS = 64;
  for (let s = 0, n = i32(); s < n; s++) {
    const k = net.blocks[i32()];
    const pc = i8(PD * DW), ps = f32(PD);
    k.cin = i8(R * DW); k.sin = f32(R); k.bias = f32(R);
    k.cout = i8(R * DW); k.sout = f32(R);
    k.ssm = { slot: s, pc, ps, M0: f32(R * NS), H1: f32(R * NS), uc: i8(DW * net.T * 8), us: f32(DW) };
  }
  if (o !== buf.byteLength) throw new Error('ssm.bin has the wrong length');
}

// The background's value after each block, in JS: infer.c's token() with every neighbour the same.
function backgroundStates(net, input) {
  const { NB, NBC, We, be, blocks, Wh, bh, Wl, bl } = net, f = Math.fround;
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
    const q1 = quant(t);
    if (net.T && k.ssm) {   // an SSM tree block (PASS_TA_SSM, PASS_TB's ssm part), its states fixed
      const { T, V, TN } = net, S = k.ssm, y = t.map((v, r) => v + k.b2[r]), Racc = new Array(T * 8).fill(0);
      const proj = r => { let a = 0; for (let i = 0; i < DW; i++) a += q1.c[i] * S.pc[r * DW + i]; return a * S.ps[r] * q1.xs; };
      const C = [0, 1, 2, 3, 4, 5, 6, 7].map(i => proj(8 + i)), Z = [0, 1, 2, 3, 4, 5, 6, 7].map(i => proj(24 + i));
      const read = (H, row) => [0, 1, 2, 3, 4, 5, 6, 7].map(v => { let r = 0; for (let n = 0; n < 8; n++) r += C[n] * H[row * 64 + n * 8 + v]; return r; });
      for (let tr = 0; tr < T; tr++) for (let j = 0, a = 0; j < V; j++) {
        const row = tr * TN + a; let sum = 0;
        for (let i = 0; i < DW; i++) sum += q1.c[i] * k.cin[row * DW + i];
        const sin = sum * k.sin[row] * q1.xs + k.bias[row];
        const r1 = read(S.M0, row), r2 = read(S.H1, row);
        const s1 = sin + r1.reduce((acc, r, v) => acc + r * Z[v], 0), s2 = sin + r2.reduce((acc, r, v) => acc + r * Z[v], 0), g = gelu(s2);
        for (let r = 0; r < DW; r++) y[r] += g * k.cout[row * DW + r] * k.sout[row];
        for (let v = 0; v < 8; v++) Racc[tr * 8 + v] += r2[v];
        a = 2 * a + 1 + (s1 > 0 ? 1 : 0);
      }
      for (let r = 0; r < DW; r++) { let u = 0; for (let q = 0; q < T * 8; q++) u += Racc[q] * S.uc[r * T * 8 + q]; y[r] += u * S.us[r]; }
      x = y; states.push(x); continue;
    }
    if (net.T) {   // the trees (infer.c walk): t + b2 + each visited node's gelu(s) w_out
      const { T, V, TN } = net, y = t.map((v, r) => v + k.b2[r]);
      for (let tr = 0; tr < T; tr++) for (let j = 0, a = 0; j < V; j++) {
        const row = tr * TN + a; let sum = 0;
        for (let i = 0; i < DW; i++) sum += q1.c[i] * k.cin[row * DW + i];
        const sv = f(f(f(sum * k.sin[row]) * q1.xs) + k.bias[row]), g = gelu(sv);
        for (let r = 0; r < DW; r++) y[r] += g * k.cout[row * DW + r] * k.sout[row];
        a = 2 * a + 1 + (sv > 0 ? 1 : 0);
      }
      x = y; states.push(x); continue;
    }
    const g = new Array(DW);
    for (let r = 0; r < DW; r++) { let a = 0; for (let i = 0; i < DW; i++) a += q1.c[i] * k.c1[r * DW + i]; g[r] = gelu(a * k.s1[r] * q1.xs + k.b1[r]); }
    const q2 = quant(g);
    x = Array.from({ length: DW }, (_, r) => { let a = 0; for (let i = 0; i < DW; i++) a += q2.c[i] * k.c2[r * DW + i]; return a * k.s2[r] * q2.xs + k.b2[r] + t[r]; });
    states.push(x);
  }
  const sig = z => 1 / (1 + Math.exp(-z));
  const out = [0, 1, 2, 3].map(c => { let z = bh[c]; for (let i = 0; i < DW; i++) z += Wh[c * DW + i] * states[NBC][i]; return sig(z); });
  let ink = 0;
  if (Wl) { let z = bl; for (let i = 0; i < DW; i++) z += Wl[i] * x[i]; const t = Math.min(1, Math.max(0, (sig(z) - 0.4) / 0.3)); ink = t * t * (3 - 2 * t); }
  return { states, out, ink };
}

const HEAD = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D; precision highp sampler2DArray; precision highp usampler2DArray;
uniform ivec4 crop;   // x0, y0, x1, y1 (GL pixels, y up, end exclusive)
uniform ivec2 org;    // the strip's origin: the activations (X, C) hold the frame's pixels from here
uniform sampler2D distT;
int distAt(ivec2 p) { return int(texelFetch(distT, p, 0).r * 255.0 + 0.5); }
bool inCrop(ivec2 p) { return p.x >= crop.x && p.y >= crop.y && p.x < crop.z && p.y < crop.w; }
`;
const outs = (n, type = 'vec4') => Array.from({ length: n }, (_, k) => `layout(location = ${k}) out ${type} o${k};`).join('\n');
const VS = `#version 300 es
void main() { vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }`;

// The ternary stencil (tc_one) for one layer of 4 channels; taps in training order, rows top-down.
// Half precision (createGlShader's precision): MP marks the blocks' arithmetic that may run in
// mediump (16-bit floats on phone GPUs, twice the rate; desktops ignore it): the stencil and the
// trees' outputs. The tree walk stays exact: a code (|c| <= 128) times ternary weights, four at a time,
// is an integer below 2048, exact in a half float; the sums of those and the int8 rounding run in highp.
// HALF_EMU rounds each MP step to a half float, to measure on a desktop what a phone would draw.
const BLOCK_COMMON = `
#if defined(HALF)
#define MP mediump
#else
#define MP highp
#endif
vec4 H(vec4 x) { return vec4(unpackHalf2x16(packHalf2x16(x.xy)), unpackHalf2x16(packHalf2x16(x.zw))); }
#if defined(HALF_EMU) || defined(HALF_EMU_S)
#define R(x) H(x)
#else
#define R(x) (x)
#endif
#if defined(HALF_EMU) || defined(HALF_EMU_B)
#define RB(x) H(x)
#else
#define RB(x) (x)
#endif
layout(std140) uniform Blk { vec4 Q[208]; vec4 gate[16]; vec4 b1[16]; vec4 b2[16]; vec4 s1[16]; vec4 s2[16]; vec4 bg[16]; };
layout(std140) uniform Wt { vec4 W[1024]; };
uniform sampler2DArray X; uniform int dil, needPrev, needCur;
vec4 xin(ivec2 p, int L) { return (!inCrop(p) || distAt(p) > needPrev) ? bg[L] : texelFetch(X, ivec3(p - org, L), 0); }
// the taps' pixels and whether each holds an activation (else the background's), found once a pixel
// (taps(p)) rather than once a tap and layer
ivec2 tapAt[9]; bool tapOn[9];
void taps(ivec2 p) {
  for (int k = 0; k < 9; k++) { ivec2 q = p + ivec2(k % 3 - 1, 1 - k / 3) * dil; tapAt[k] = q - org; tapOn[k] = inCrop(q) && distAt(q) <= needPrev; }
}
MP vec4 tap(int k, int L) { return tapOn[k] ? texelFetch(X, ivec3(tapAt[k], L), 0) : R(bg[L]); }
MP vec4 stencil(ivec2 p, int L) {   // after taps(p)
  MP vec4 z = vec4(0.0);
  for (int k = 0; k < 9; k++) { MP vec4 q = Q[k * 16 + L]; z = R(q * tap(k, L) + z); }
  MP vec4 dw = R(z * (1.0 / 3.0)), m = vec4(0.0);
  for (int j = 0; j < 4; j++) { MP vec4 q = Q[(9 + j) * 16 + L]; m = R(q * dw[j] + m); }
  MP vec4 gt = R(gate[L]);
  return R(gt * R(m * 0.5) + tap(4, L));
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
// The trees' weights, packed: each pixel walks its own rows, which a GPU reads far faster from a
// texture than from a uniform buffer, and a ternary row of 64 is 128 bits, one texel: x, y the
// channels that are +1 (bits 0-31, 32-63), z, w those that are -1. Texel 1 of the row: its scale and
// (w_in) its node bias, as float bits. Row r of the matrix at texture row wrow + r.
uniform highp usampler2D Wp; uniform int wrow;
uvec4 rowBits(int r) { return texelFetch(Wp, ivec2(0, wrow + r), 0); }
vec2 rowScale(int r) { return uintBitsToFloat(texelFetch(Wp, ivec2(1, wrow + r), 0).xy); }
vec4 tern(uvec4 m, int k) {   // channels 4k .. 4k + 3: +1, 0 or -1
  uint s = uint(4 * (k & 7)), pos = (k < 8 ? m.x : m.y) >> s, neg = (k < 8 ? m.z : m.w) >> s;
  return vec4(uvec4(pos, pos >> 1u, pos >> 2u, pos >> 3u) & 1u) - vec4(uvec4(neg, neg >> 1u, neg >> 2u, neg >> 3u) & 1u);
}
#ifdef UNPACKED
// (#unpacked: the rows as 32-bit floats, 16 texels a row, to compare)
uniform highp sampler2D Wf;
#define TERN(m, r, k) texelFetch(Wf, ivec2(k, wrow + r), 0)
#else
#define TERN(m, r, k) tern(m, k)
#endif
float rowdotT(vec4 c[16], uvec4 m, int r) {   // c: int8 codes; exact (integers)
  float a = 0.0;
  for (int k = 0; k < 16; k++) { MP vec4 ck = c[k], wk = TERN(m, r, k); a += float(dot(ck, wk)); }
  return a;
}
`;
// ablate (profiling only, benchA): steps to leave out, each replaced by the cheapest thing that keeps
// the data flowing (so the compiler can't drop the rest): STENCIL (the pixel's own input), QUANT
// (scale 1, no rounding), FFN (each output its own input), GELU (the identity).
const PASS_A = (scaleOnly, M, ablate = []) => HEAD.replace('\n', '\n' + ablate.map(a => `#define NO_${a}\n`).join('')) + BLOCK_COMMON + `${M >= 5 ? outs(5, 'uvec4') : scaleOnly ? outs(1, 'uvec4') : outs(4, 'uvec4')}
uint pack4(vec4 c) { uvec4 i = uvec4(ivec4(c) & 255); return i.x | (i.y << 8u) | (i.z << 16u) | (i.w << 24u); }
#ifdef NO_QUANT
#define QUANT(v) 1.0
#else
#define QUANT(v) quant(v)
#endif
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy) + org;   // drawn into the strip
  if (distAt(p) > needCur) discard;
  taps(p);
  vec4 v[16];
#ifdef NO_STENCIL
  for (int L = 0; L < 16; L++) v[L] = xin(p, L);
#else
  for (int L = 0; L < 16; L++) v[L] = stencil(p, L);
#endif
  float xs = QUANT(v);
  vec4 g[16];
  for (int L = 0; L < 16; L++) {
    vec4 h;
#ifdef NO_FFN
    for (int e = 0; e < 4; e++) h[e] = v[L][e] * s1[L][e] * xs + b1[L][e];
#else
    for (int e = 0; e < 4; e++) h[e] = rowdot(v, 4 * L + e) * s1[L][e] * xs + b1[L][e];
#endif
#ifdef NO_GELU
    g[L] = h;
#else
    g[L] = vec4(gelu(h.x), gelu(h.y), gelu(h.z), gelu(h.w));
#endif
  }
  float xs2 = QUANT(g);
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
  ivec2 p = ivec2(gl_FragCoord.xy) + org;   // drawn into the strip
  if (distAt(p) > needCur) discard;
  taps(p);
  vec4 c[16];
  for (int j = 0; j < 4; j++) { uvec4 u = texelFetch(C, ivec3(p - org, j), 0); c[4 * j] = unpack4(u.x); c[4 * j + 1] = unpack4(u.y); c[4 * j + 2] = unpack4(u.z); c[4 * j + 3] = unpack4(u.w); }
  float xs = uintBitsToFloat(texelFetch(C, ivec3(p - org, 4), 0).x);
  vec4 y[${M}];
  for (int j = 0; j < ${M}; j++) {
    int L = ${M} * hf + j;
    vec4 o;
    for (int e = 0; e < 4; e++) o[e] = rowdot(c, 4 * L + e) * s2[L][e] * xs + b2[L][e];
    y[j] = o + stencil(p, L);
  }
  ${Array.from({ length: M }, (_, j) => `o${j} = y[${j}];`).join(' ')}
}`;
// Tree blocks (NTS3): the FFN is T binary trees of V levels. Pass TA: the stencil, quant8, then each
// tree's walk: at each level the node's s = w_in . codes (w_in from the Wt buffer, its row scale and
// bias in Blk's s1 and b1 slots), gelu(s) kept and the child sign(s) picks. Out: the visited nodes
// (a tree's in one uint, 8 bits a level) and their GELUs (half floats, two to a uint). Pass TB:
// x = stencil + b2 + each visited node's gelu x its w_out row (the next Wt buffer), M layers a pass.
// The stencil is worked out once, by pass TS (M layers a pass, half floats, as the activations are
// kept), and both read it from St.
const PASS_TS = M => HEAD + BLOCK_COMMON + `${outs(M)}
uniform int hf;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy) + org;   // drawn into the strip
  if (distAt(p) > needCur) discard;
  taps(p);
  ${Array.from({ length: M }, (_, j) => `o${j} = stencil(p, ${M} * hf + ${j});`).join(' ')}
}`;
// cached: the stencil from St; else each pass works it out (taps(p) by prep(p))
const STENCIL_IN = cached => cached ? `uniform highp sampler2DArray St;
MP vec4 stencilAt(ivec2 p, int L) { return texelFetch(St, ivec3(p - org, L), 0); }
void prep(ivec2 p) {}
` : `MP vec4 stencilAt(ivec2 p, int L) { return stencil(p, L); }
void prep(ivec2 p) { taps(p); }
`;
const PASS_TA = (T, V, TN, cached) => HEAD + BLOCK_COMMON + STENCIL_IN(cached) + `${outs(3, 'uvec4')}
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy) + org;   // drawn into the strip
  if (distAt(p) > needCur) discard;
  prep(p);
  vec4 v[16];
  for (int L = 0; L < 16; L++) v[L] = stencilAt(p, L);
  float xs = quant(v);
  uvec4 nodes = uvec4(0u);
  float g[16];
  for (int k = 0; k < 16; k++) g[k] = 0.0;
  for (int tr = 0; tr < ${T}; tr++) {
    int a = 0;
    for (int j = 0; j < ${V}; j++) {
      int row = tr * ${TN} + a;
      vec2 sb = rowScale(row);   // its w_in scale and node bias
      float sv = rowdotT(v, rowBits(row), row) * sb.x * xs + sb.y;
      g[tr * ${V} + j] = gelu(sv);
      nodes[tr] |= uint(row) << (8u * uint(j));
      a = 2 * a + 1 + (sv > 0.0 ? 1 : 0);
    }
  }
  o0 = nodes;
  o1 = uvec4(packHalf2x16(vec2(g[0], g[1])), packHalf2x16(vec2(g[2], g[3])), packHalf2x16(vec2(g[4], g[5])), packHalf2x16(vec2(g[6], g[7])));
  o2 = uvec4(packHalf2x16(vec2(g[8], g[9])), packHalf2x16(vec2(g[10], g[11])), packHalf2x16(vec2(g[12], g[13])), packHalf2x16(vec2(g[14], g[15])));
}`;
// SSM tree blocks (style_torch.py / ssm_tree.py; meta.json "ssm"): each visited node has a state h (8 x 8),
// read here as fixed (export_ssm.py): the walk routes on s = w_in.x + bias + (C . dec h0) . Z, the outputs
// read the frozen state h1: gelu(w_in.x + bias + (C . h1) . Z) for w_out, and the sum of those reads over a
// tree's path for U. C and Z (8 each) from the block's ternary projection (Wp's third slot, prow); dec h0,
// h1 and U from Sf (per block 184 rows of 16 texels: dec h0, then h1, a node a row, n x pv; then U, a
// channel a row). Out: the visited nodes and their gelus as pass TA, and each tree's sum of reads (8 half
// floats, a layer a tree, layers 3..6).
// part: -1 all 3 + T layers in one pass (8 draw buffers); 0 the nodes and gelus, 1 the sums (4 draw buffers: two passes)
const PASS_TA_SSM = (T, V, TN, cached, part = -1) => HEAD + BLOCK_COMMON + STENCIL_IN(cached) + `${outs(part < 0 ? 3 + T : part === 0 ? 3 : T, 'uvec4')}
uniform int prow, sbase; uniform highp sampler2D Sf;
float projRow(vec4 c[16], int k) {   // the projection's row k: int8 codes . ternary, x its scale
  uvec4 m = texelFetch(Wp, ivec2(0, prow + k), 0);
  float a = 0.0;
  for (int kk = 0; kk < 16; kk++) { MP vec4 ck = c[kk], wk = tern(m, kk); a += float(dot(ck, wk)); }
  return a * uintBitsToFloat(texelFetch(Wp, ivec2(1, prow + k), 0).x);
}
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy) + org;   // drawn into the strip
  if (distAt(p) > needCur) discard;
  prep(p);
  vec4 v[16];
  for (int L = 0; L < 16; L++) v[L] = stencilAt(p, L);
  float xs = quant(v);
  float Cc[8];
  for (int i = 0; i < 8; i++) Cc[i] = projRow(v, 8 + i) * xs;
  vec4 Za = vec4(projRow(v, 24), projRow(v, 25), projRow(v, 26), projRow(v, 27)) * xs;
  vec4 Zb = vec4(projRow(v, 28), projRow(v, 29), projRow(v, 30), projRow(v, 31)) * xs;
  uvec4 nodes = uvec4(0u);
  float g[16];
  vec4 RA[${T}], RB_[${T}];
  for (int tr = 0; tr < ${T}; tr++) {
    int a = 0;
    RA[tr] = vec4(0.0); RB_[tr] = vec4(0.0);
    for (int j = 0; j < ${V}; j++) {
      int row = tr * ${TN} + a;
      vec2 sb = rowScale(row);
      float sin_ = rowdotT(v, rowBits(row), row) * sb.x * xs + sb.y;
      vec4 r1a = vec4(0.0), r1b = vec4(0.0), r2a = vec4(0.0), r2b = vec4(0.0);
      for (int n = 0; n < 8; n++) {
        r1a += Cc[n] * texelFetch(Sf, ivec2(2 * n, sbase + row), 0); r1b += Cc[n] * texelFetch(Sf, ivec2(2 * n + 1, sbase + row), 0);
        r2a += Cc[n] * texelFetch(Sf, ivec2(2 * n, sbase + ${T * TN} + row), 0); r2b += Cc[n] * texelFetch(Sf, ivec2(2 * n + 1, sbase + ${T * TN} + row), 0);
      }
      float s1 = sin_ + dot(r1a, Za) + dot(r1b, Zb), s2 = sin_ + dot(r2a, Za) + dot(r2b, Zb);
      g[tr * ${V} + j] = gelu(s2);
      nodes[tr] |= uint(row) << (8u * uint(j));
      RA[tr] += r2a; RB_[tr] += r2b;
      a = 2 * a + 1 + (s1 > 0.0 ? 1 : 0);
    }
  }
  ${part === 1 ? '' : `o0 = nodes;
  o1 = uvec4(packHalf2x16(vec2(g[0], g[1])), packHalf2x16(vec2(g[2], g[3])), packHalf2x16(vec2(g[4], g[5])), packHalf2x16(vec2(g[6], g[7])));
  o2 = uvec4(packHalf2x16(vec2(g[8], g[9])), packHalf2x16(vec2(g[10], g[11])), packHalf2x16(vec2(g[12], g[13])), packHalf2x16(vec2(g[14], g[15])));`}
  ${part === 0 ? '' : Array.from({ length: T }, (_, t) => `o${(part < 0 ? 3 : 0) + t} = uvec4(packHalf2x16(RA[${t}].xy), packHalf2x16(RA[${t}].zw), packHalf2x16(RB_[${t}].xy), packHalf2x16(RB_[${t}].zw));`).join(' ')}
}`;
const PASS_TB = (M, T, V, cached, ssm = false) => HEAD + BLOCK_COMMON + STENCIL_IN(cached) + `${outs(M)}${ssm ? `
uniform int sbase; uniform highp sampler2D Sf;` : ''}
uniform highp usampler2DArray C; uniform int hf;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy) + org;   // drawn into the strip
  if (distAt(p) > needCur) discard;
  prep(p);
  uvec4 nodes = texelFetch(C, ivec3(p - org, 0), 0), h0 = texelFetch(C, ivec3(p - org, 1), 0), h1 = texelFetch(C, ivec3(p - org, 2), 0);
  MP float g[16];   // half floats already
  for (int i = 0; i < 4; i++) {
    vec2 a = unpackHalf2x16(h0[i]), b = unpackHalf2x16(h1[i]);
    g[2 * i] = a.x; g[2 * i + 1] = a.y; g[8 + 2 * i] = b.x; g[8 + 2 * i + 1] = b.y;
  }
  MP vec4 y[${M}];
  for (int j = 0; j < ${M}; j++) y[j] = RB(stencilAt(p, ${M} * hf + j) + RB(b2[${M} * hf + j]));
  // each visited node: its gelu x its w_out scale, added, subtracted or not per channel (ternary)
  for (int tr = 0; tr < ${T}; tr++) for (int k = 0; k < ${V}; k++) {
    int row = int((nodes[tr] >> (8u * uint(k))) & 255u);
    uvec4 m = rowBits(row);
    MP float gs = RB(vec4(g[tr * ${V} + k] * RB(vec4(rowScale(row).x)).x)).x;
    for (int j = 0; j < ${M}; j++) y[j] = RB(gs * TERN(m, row, ${M} * hf + j) + y[j]);
  }${ssm ? `
  // SSM: + U . (each tree's sum of reads), U a channel a row of Sf after the states
  vec4 Rs[${2 * T}];
  for (int tr = 0; tr < ${T}; tr++) {
    uvec4 u = texelFetch(C, ivec3(p - org, 3 + tr), 0);
    Rs[2 * tr] = vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y)); Rs[2 * tr + 1] = vec4(unpackHalf2x16(u.z), unpackHalf2x16(u.w));
  }
  for (int j = 0; j < ${M}; j++) for (int e = 0; e < 4; e++) {
    int ch = 4 * (${M} * hf + j) + e; float a = 0.0;
    for (int kk = 0; kk < ${2 * T}; kk++) a += dot(Rs[kk], texelFetch(Sf, ivec2(kk, sbase + ${2 * T * ((1 << V) - 1)} + ch), 0));
    y[j][e] += a;
  }` : ''}
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
  ivec2 p = ivec2(gl_FragCoord.xy) + org;   // drawn into the strip
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
      for (int L = 0; L < 16; L++) s = Wh[c * 16 + L] * texelFetch(X, ivec3(p - org, L), 0) + s;
      q[c] = 1.0 / (1.0 + exp(-(bh[c] + ((s.x + s.y) + (s.z + s.w)))));
    }
  }
  o = vec4(q.rgb * q.a, q.a);   // premultiplied, for filtering when it's magnified
}`;
// The ink head: black line art's opacity, kept apart from the colour so the two filter separately.
const INK_PASS = HEAD + `out vec4 o;
uniform sampler2DArray X; uniform vec4 Wl[16]; uniform float bl, kbg;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  float k = kbg;
  if (distAt(p) <= ${MARGIN}) {
    vec4 s = vec4(0.0);
    for (int L = 0; L < 16; L++) s = Wl[L] * texelFetch(X, ivec3(p - org, L), 0) + s;
    k = 1.0 / (1.0 + exp(-(bl + ((s.x + s.y) + (s.z + s.w)))));
    k = smoothstep(0.4, 0.7, k);   // drawn crisp (infer.c ink)
  }
  o = vec4(k);
}`;
const SHOW = `#version 300 es
precision highp float;
uniform sampler2D R, K, T0, TT, TD; uniform vec4 box; uniform vec3 top, bottom; uniform vec2 view; uniform int raw, ink, toon;
uniform int bgOn;   // the backdrop (backdrop.js) instead of the stage gradient
${BACKDROP_GLSL}
uniform vec3 toonP;   // near, far, line width (canvas pixels)
uniform sampler2D G, T1, T2, T4; uniform int sharp; uniform vec2 lowRes;
uniform sampler2D RAW;   // raw == 2: Raw 3D at the canvas's resolution (guide.js createRawView), premultiplied
out vec4 o;
// The edge-aware upscale (joint bilateral): the drawing is the network's resolution, guide.js's G is
// her geometry at the canvas's. A canvas pixel takes the drawing's pixels around it (4 x 4) weighted by
// bilinear's weights and by how well their geometry (the tiles' normal, depth, mask) matches its own, so her
// edges and the edges between her parts land where the canvas-sized geometry puts them, and only
// colour from the same surface crosses into a pixel. Pixels the drawing half-covers count by their
// alpha; her outline is G's coverage. Where nothing around matches (a strand
// thinner than the drawing's pixels), the plain bilinear sample.
vec4 upscale(vec2 uv, vec4 bil) {
  vec4 g = texelFetch(G, ivec2(gl_FragCoord.xy), 0);
  float cov = g.a;
  if (cov < 0.004) return vec4(0.0);
  vec2 gxy = g.xy / cov;
  vec3 gn = vec3(gxy, sqrt(max(0.0, 1.0 - dot(gxy, gxy))));
  float gd = g.z / cov;
  vec2 P = uv * lowRes - 0.5;
  ivec2 b = ivec2(floor(P));
  vec3 srgb = vec3(0.0); float sa = 0.0, saa = 0.0;
  for (int j = -1; j <= 2; j++) for (int i = -1; i <= 2; i++) {
    ivec2 q = clamp(b + ivec2(i, j), ivec2(0), ivec2(lowRes) - 1);
    vec4 c = texelFetch(R, q, 0);
    if (c.a < 0.004) continue;
    vec3 n = normalize(texelFetch(T1, q, 0).rgb * 2.0 - 1.0);
    float dz = (texelFetch(T2, q, 0).r - gd) * 80.0;   // in 2 cm steps (the depth tile spans 1.6 m)
    vec2 dd = P - vec2(q);
    // bilinear's own weights (so a surface with no edge in it is exactly the bilinear sample) plus a
    // faint wider tail, for an edge pixel whose nearest four are all across the edge
    vec2 t = max(vec2(0.0), 1.0 - abs(dd));
    float w = (t.x * t.y + 0.02 * exp(-0.5 * dot(dd, dd))) * exp(-0.5 * dz * dz) * exp(-8.0 * (1.0 - dot(n, gn)));
    w *= texelFetch(T4, q, 0).b > 0.5 ? 1.0 : 0.05;
    srgb += w * c.a * c.rgb; sa += w * c.a; saa += w * c.a * c.a;   // colour by alpha^2: her edge pixels (the film's green spill) count little
  }
  // her outline from the canvas-sized geometry: where the drawing has her colour nearby, its coverage is
  // G's (crisp and multisampled at the screen's resolution), not the network's coarser alpha, which
  // broke strands thinner than its pixels into dots
  // (where it hardly drew her at all, a strand it missed, its colour is noise: fade it out with its alpha)
  if (saa < 1e-4) return vec4(0.0);
  vec3 rgb = srgb / saa;
  float a = cov * smoothstep(0.03, 0.3, saa / sa);
  return vec4(rgb * a, a);
}
float viewZ(ivec2 p) {   // distance along the view axis; far away where nothing was drawn
  p = clamp(p, ivec2(0), ivec2(view) - 1);
  float d = texelFetch(TD, p, 0).r;
  if (d >= 1.0) return 1e4;
  float n = toonP.x, f = toonP.y;
  return 2.0 * n * f / (f + n - (d * 2.0 - 1.0) * (f - n));
}
// toon.js's contours (canvas-sized), plus the near side of every step in depth of more than 3 cm from
// one of her surfaces to another (the silhouette is the hull's): sixteen taps about half a line-width
// away, thinner than the outline as inner lines are drawn, a partial count giving the soft edge.
vec4 toonOver(vec4 c) {
  if (toon == 0) return c;
  ivec2 p = ivec2(gl_FragCoord.xy);
  float k = texelFetch(TT, p, 0).a, z = viewZ(p);
  if (z < 1e3) {
    float n = 0.0;
    for (int i = 0; i < 16; i++) {
      float a = float(i) * 0.3927 + 0.2, r = max(1.0, toonP.z * (i % 2 == 0 ? 0.45 : 0.8));
      float zn = viewZ(p + ivec2(round(vec2(cos(a), sin(a)) * r)));
      if (zn < 1e3 && zn - z > 0.03) n += 1.0;
    }
    k = max(k, smoothstep(0.5, 3.5, n));
  }
  return vec4(mix(c.rgb, vec3(0.012, 0.010, 0.022), k), 1.0);
}
vec3 toSrgb(vec3 c) { return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c)); }
void main() {
  vec2 q = vec2(gl_FragCoord.x, view.y - gl_FragCoord.y);   // canvas pixels, top-down
  vec3 bg = bgOn == 1 ? backdrop((q - box.xy) / box.zw) : mix(top, bottom, q.y / view.y);
  if (raw == 2) { vec4 c = texelFetch(RAW, ivec2(gl_FragCoord.xy), 0); o = vec4(c.a < 0.004 ? bg : mix(bg, toSrgb(c.rgb / c.a), c.a), 1.0); return; }
  vec2 p = (q - box.xy) / box.zw;
  if (p.x < 0.0 || p.y < 0.0 || p.x > 1.0 || p.y > 1.0) { o = toonOver(vec4(bg, 1.0)); return; }
  vec2 uv = vec2(p.x, 1.0 - p.y);
  if (raw == 1) { vec3 c = toSrgb(texture(T0, uv).rgb); o = vec4(mix(c, bg, step(0.25, c.g - max(c.r, c.b))), 1.0); return; }
  vec4 d = texture(R, uv);
  if (sharp == 1) d = upscale(uv, d);
  float k = ink == 1 ? texture(K, uv).r : 0.0;   // ink over the colour: alpha a + k (1 - a), colour scaled by (1 - k)
  d = vec4(d.rgb * (1.0 - k), d.a + k * (1.0 - d.a));
  o = toonOver(vec4(bg * (1.0 - d.a) + d.rgb, 1.0));
}`;

// maxStrip: the most pixels the network's activations may hold at once (a phone's GPU memory: 464
// bytes a pixel). A taller crop than fits runs in horizontal strips, each with the network's reach of
// extra rows above and below, so every strip's own rows come out exactly as from the whole crop.
// cacheStencil: the trees' blocks work their stencil out once (pass TS, kept as half floats, 128 bytes a
// pixel more) instead of in both of their passes.
// precision: 'half' (the blocks' stencil and tree outputs in 16-bit floats where the GPU has them),
// 'full', or 'emulate' (32-bit, rounding each of those steps to 16 bits: a phone's result on a desktop).
// dilScale: run the network on a grid this many times finer than it trained on, its stencils' dilations
// scaled to match (each still reaches as far on her): the same picture, drawn at that resolution.
export async function createGlShader({ THREE, renderer, NW, NH, base = 'nts/', maxStrip = Infinity, cacheStencil = true, precision = 'half', packed = true, dilScale = 1 }) {
  const gl = renderer.getContext();
  if (!(gl instanceof WebGL2RenderingContext)) throw new Error('needs WebGL2');
  if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('needs float render targets (EXT_color_buffer_float)');
  // Layers written per pass: 8 where the device has 8 draw buffers (iPhones, desktops), else 4.
  const M = gl.getParameter(gl.MAX_DRAW_BUFFERS) >= 8 ? 8 : 4;
  if (gl.getParameter(gl.MAX_DRAW_BUFFERS) < 4) throw new Error('needs 4 draw buffers');
  const [meta, netImg, ...fineImgs] = await Promise.all([fetch(base + 'meta.json').then(r => r.json()), loadImage(base + 'net.png'),
    ...[0, 1, 2].map(k => loadImage(`${base}fine${k}.png`))]);
  const net = parseNet(unpackNet(netImg));
  net.DIL = net.DIL.map(d => d * dilScale);
  const { NB, NBC, P, DIL } = net;
  if (meta.ssm) parseSSM(await (await fetch(base + 'ssm.bin')).arrayBuffer(), net);
  const SSM = net.blocks.filter(k => k.ssm);


  // ---- programs ----
  const defs = (packed ? '' : '#define UNPACKED\n') + { half: '#define HALF\n', emulate: '#define HALF_EMU\n', emulateS: '#define HALF_EMU_S\n', emulateB: '#define HALF_EMU_B\n', full: '' }[precision];
  function program(fs) {
    fs = fs.replace('#version 300 es\n', '#version 300 es\n' + defs);
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
  const prog = { TS: net.T && cacheStencil ? program(PASS_TS(M)) : null, TA: net.T ? program(PASS_TA(net.T, net.V, net.TN, cacheStencil)) : null, TB: net.T ? program(PASS_TB(M, net.T, net.V, cacheStencil)) : null,
    TAs: SSM.length && M >= 8 ? program(PASS_TA_SSM(net.T, net.V, net.TN, cacheStencil)) : null,
    TAs0: SSM.length && M < 8 ? program(PASS_TA_SSM(net.T, net.V, net.TN, cacheStencil, 0)) : null, TAs1: SSM.length && M < 8 ? program(PASS_TA_SSM(net.T, net.V, net.TN, cacheStencil, 1)) : null, TBs: SSM.length ? program(PASS_TB(M, net.T, net.V, cacheStencil, true)) : null,
    A: program(PASS_A(false, M)), A2: M < 5 ? program(PASS_A(true, M)) : null, B: program(PASS_B(M)), embed: program(EMBED(M)), diff: program(DIFF), distH: program(DIST_H), distV: program(DIST_V), head: program(HEAD_PASS), ink: program(INK_PASS), show: program(SHOW) };

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
    gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, internal, NW, SH, layers);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST]]) gl.texParameteri(gl.TEXTURE_2D_ARRAY, k, v);
    return { tex: t, fbos: new Map() };
  }
  // the strips: SH rows of activations; a strip computes its own rows plus REACH (all the blocks'
  // dilations: how far a pixel's result can see) on each side
  const REACH = DIL.reduce((a, b) => a + b, 0);
  const SH = Math.min(NH, Math.max(2 * REACH + 64, Math.floor(maxStrip / NW)));
  const fine = fineImgs.map(img => tex2d(0, 0, gl.RGB8, gl.RGB, gl.UNSIGNED_BYTE, img));
  // coarse: [part][64][64][8] -> two RGBA8 atlases, part p at x = p * 64
  const cw = P * COARSE, cdata = [new Uint8Array(cw * COARSE * 4), new Uint8Array(cw * COARSE * 4)];
  for (let p = 0; p < P; p++) for (let y = 0; y < COARSE; y++) for (let x = 0; x < COARSE; x++) for (let c = 0; c < 8; c++)
    cdata[c >> 2][((y * cw) + p * COARSE + x) * 4 + (c & 3)] = net.coarse[((p * COARSE + y) * COARSE + x) * 8 + c];
  const coarse = cdata.map(d => tex2d(cw, COARSE, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, d));
  const X = array(16, gl.RGBA16F), Y = array(16, gl.RGBA16F), Cq = array(SSM.length ? 3 + net.T : 5, gl.RGBA32UI), St = net.T && cacheStencil ? array(16, gl.RGBA16F) : null;
  const diffT = tex2d(NW, NH, gl.R8, gl.RED, gl.UNSIGNED_BYTE), hT = tex2d(NW, NH, gl.R8, gl.RED, gl.UNSIGNED_BYTE), distT = tex2d(NW, NH, gl.R8, gl.RED, gl.UNSIGNED_BYTE);
  const outT = tex2d(NW, NH, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, null, gl.LINEAR), inkT = tex2d(NW, NH, gl.R8, gl.RED, gl.UNSIGNED_BYTE, null, gl.LINEAR);
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
  // coverage renders). The colour has its own; the other four take turns in one multisampled target,
  // each copied out to a plain texture (one set of multisampled buffers instead of four) ----
  const rtColour = new THREE.WebGLRenderTarget(NW, NH, { samples: 4, depthBuffer: true, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter });
  rtColour.texture.colorSpace = THREE.SRGBColorSpace;
  const rtG = new THREE.WebGLRenderTarget(NW, NH, { samples: 4, depthBuffer: true, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
  const glTex = rt => renderer.properties.get(rt.texture).__webglTexture;
  const gTex = [1, 2, 3, 4].map(() => tex2d(NW, NH, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE));
  const tileTex = k => k ? gTex[k - 1] : glTex(rtColour);

  // ---- uniform buffers: per block (stencil codes, gates, biases, scales, its background input),
  // its two weight matrices, the embedding ----
  const BLK = 5120, WT = 16384;   // bytes, multiples of the binding alignment
  const blkData = new Float32Array(NB * BLK / 4), wtData = new Float32Array(2 * NB * WT / 4);
  net.blocks.forEach((k, b) => {
    const o = b * BLK / 4;
    for (let i = 0; i < 13 * DW; i++) blkData[o + i] = k.q[i];   // Q[t * 16 + L] = codes of tap t, channels 4L..4L+3
    if (net.T) {   // trees: node biases and w_in row scales in the b1 and s1 slots; w_in codes, then w_out (x its scale)
      [k.gate, k.bias, k.b2, k.sin].forEach((a, j) => blkData.set(a, o + 13 * DW + j * DW));
      const n = net.T * net.TN, wi = 2 * b * WT / 4, wo = (2 * b + 1) * WT / 4;
      for (let i = 0; i < n * DW; i++) { wtData[wi + i] = k.cin[i]; wtData[wo + i] = k.cout[i] * k.sout[i / DW | 0]; }
      return;
    }
    [k.gate, k.b1, k.b2, k.s1, k.s2].forEach((a, j) => blkData.set(a, o + 13 * DW + j * DW));
    for (const [m, c] of [[0, k.c1], [1, k.c2]]) { const w = (2 * b + m) * WT / 4; for (let i = 0; i < DW * DW; i++) wtData[w + i] = c[i]; }
  });
  const bgOff = 13 * DW + 5 * DW;   // floats into a block's slot: its background input
  const ubo = data => { const b = gl.createBuffer(); gl.bindBuffer(gl.UNIFORM_BUFFER, b); gl.bufferData(gl.UNIFORM_BUFFER, data, gl.DYNAMIC_DRAW); return b; };
  const blkBuf = ubo(blkData), wtBuf = ubo(wtData);
  // the same weights as a texture for the trees: 16 texels (64 floats) a row, each matrix 64 rows (WT bytes)
  // the same, packed for the shaders (rowBits, rowScale): per matrix 64 rows of 2 RGBA32UI texels
  const wp = new Uint32Array(3 * NB * 64 * 8), wpf = new Float32Array(wp.buffer);
  if (net.T) net.blocks.forEach((k, b) => {
    // (an SSM block: its projection's 36 rows in a third slot)
    for (const [m, codes, scale, bias] of [[0, k.cin, k.sin, k.bias], [1, k.cout, k.sout, null], ...(k.ssm ? [[2, k.ssm.pc, k.ssm.ps, null]] : [])]) {
      for (let r = 0; r < codes.length / DW; r++) {
        const o = ((3 * b + m) * 64 + r) * 8;
        for (let i = 0; i < DW; i++) {
          const c = codes[r * DW + i], w = i >> 5, bit = (1 << (i & 31)) >>> 0;
          if (c > 0) wp[o + w] |= bit; else if (c < 0) wp[o + 2 + w] |= bit;
        }
        wpf[o + 4] = scale[r]; if (bias) wpf[o + 5] = bias[r];
      }
    }
  });
  const wTex = tex2d(2, 3 * NB * 64, gl.RGBA32UI, gl.RGBA_INTEGER, gl.UNSIGNED_INT, wp);
  // the SSM blocks' fixed states and U (PASS_TA_SSM): per block 184 rows of 16 RGBA32F texels
  const SR = 2 * net.T * net.TN + DW, sf = new Float32Array(Math.max(1, SSM.length) * SR * 64);
  for (const k of SSM) {
    const o = k.ssm.slot * SR * 64, R = net.T * net.TN, Q = net.T * 8;
    sf.set(k.ssm.M0, o); sf.set(k.ssm.H1, o + R * 64);
    for (let ch = 0; ch < DW; ch++) for (let q = 0; q < Q; q++) sf[o + 2 * R * 64 + ch * 64 + q] = k.ssm.uc[ch * Q + q] * k.ssm.us[ch];
  }
  const sfTex = SSM.length ? tex2d(16, SSM.length * SR, gl.RGBA32F, gl.RGBA, gl.FLOAT, sf) : null;
  // (#unpacked) the rows as floats: w_in codes, w_out codes (x 1: the scale is applied as packed)
  const wf = new Float32Array(3 * NB * 64 * 64);
  if (net.T && !packed) net.blocks.forEach((k, b) => [k.cin, k.cout].forEach((c, m) => wf.set(Float32Array.from(c), (3 * b + m) * 64 * 64)));
  const wfTex = packed ? null : tex2d(16, 3 * NB * 64, gl.RGBA32F, gl.RGBA, gl.FLOAT, wf);
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
  let lastPitch = null, pbg = [0, 0, 0, 0], kbg = 0;
  let crop = [0, 0, NW, NH];

  // three.js draws the tiles; call before frame(). cropTD: [x0, y0, x1, y1] network pixels, top-down.
  function renderTiles(scene, camera, passes, cropTD) {
    const [x0, y0, x1, y1] = cropTD;
    crop = [x0, NH - y1, x1, NH - y0];
    passes.forEach(([bg, mat, show], k) => {
      const rt = k ? rtG : rtColour;
      rt.scissor.set(crop[0], crop[1], crop[2] - crop[0], crop[3] - crop[1]); rt.scissorTest = true;
      show(); scene.background = bg; scene.overrideMaterial = mat;
      renderer.setRenderTarget(rt); renderer.render(scene, camera);   // three.js resolves the samples into rt.texture
      if (k) {   // the crop, out to the tile's own texture
        const dst = fbo2d(gTex[k - 1]);   // (made first: making one binds it)
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, renderer.properties.get(rtG).__webglFramebuffer);
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, dst);
        gl.disable(gl.SCISSOR_TEST);
        gl.blitFramebuffer(crop[0], crop[1], crop[2], crop[3], crop[0], crop[1], crop[2], crop[3], gl.COLOR_BUFFER_BIT, gl.NEAREST);
        renderer.resetState();
      }
    });
    scene.overrideMaterial = null; renderer.setRenderTarget(null);
  }

  function clear(t) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo2d(t));
    gl.disable(gl.SCISSOR_TEST); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); gl.enable(gl.SCISSOR_TEST);
  }
  let org = [0, 0];   // the strip's origin in the frame (the activations' pixel (0, 0))
  function pass(pr, target, first, m, setup) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbos ? fbo(target, first, m) : fbo2d(target));
    gl.useProgram(pr.p);
    gl.uniform4i(pr.u.crop, ...crop);
    if (pr.u.org) gl.uniform2i(pr.u.org, ...org);
    setup(pr.u);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  // The network on the crop, then the display pass. view: { W, H, box: [x, y, w, h] in canvas pixels,
  // top: [r, g, b], bottom: [r, g, b], raw, ink (false: the drawing without its line art), toon
  // (toon.js's render: contour mask and depth, or null), guide (guide.js's texture: the edge-aware
  // upscale; null: bilinear), rawTex (guide.js createRawView's render: Raw 3D at the canvas's resolution;
  // null: the colour tile), showOnly (the last drawing again, e.g. zoomed) }.
  // Profiling (cpu.html #prof): each stage ends by reading back one pixel of a private 1 x 1 target,
  // which can't return before every command before it has run (gl.finish() can, in some browsers),
  // timed on the CPU clock. That works on any device (iPhones have no GPU timer queries); the stalls
  // make frames slower, but the shares hold.
  const syncT = tex2d(1, 1, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE), syncPx = new Uint8Array(4);
  const sync = () => {
    const prev = gl.getParameter(gl.FRAMEBUFFER_BINDING);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo2d(syncT)); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, syncPx);
    gl.bindFramebuffer(gl.FRAMEBUFFER, prev);
  };
  const prof = {
    on: false, t: 0, frames: 0, acc: new Map(),
    start() { if (!this.on) return; sync(); this.t = performance.now(); },
    mark(name) {
      if (!this.on) return;
      sync(); const t = performance.now();
      this.acc.set(name, (this.acc.get(name) || 0) + t - this.t); this.t = t;
    },
    end() { if (this.on) this.frames++; },
    // [[stage, ms a frame]], largest first; then cleared
    report() {
      const r = [...this.acc].map(([k, v]) => [k, v / Math.max(1, this.frames)]).sort((a, b) => b[1] - a[1]);
      this.acc.clear(); this.frames = 0; return r;
    },
  };
  const mark = n => prof.mark(n);

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
        const { states, out, ink } = backgroundStates(net, bgIn);
        for (let b = 0; b < NB; b++) blkData.set(states[b], b * BLK / 4 + bgOff);
        gl.bindBuffer(gl.UNIFORM_BUFFER, blkBuf); gl.bufferSubData(gl.UNIFORM_BUFFER, 0, blkData);
        pbg = out; kbg = ink; lastPitch = pitch;
      }
      const T = [0, 1, 2, 3, 4].map(tileTex);
      const bindTiles = u => T.forEach((t, k) => bindTex(k, gl.TEXTURE_2D, t, null, u[`T${k}`]));
      pass(prog.diff, diffT, 0, 1, bindTiles);
      pass(prog.distH, hT, 0, 1, u => bindTex(0, gl.TEXTURE_2D, diffT, null, u.M));
      pass(prog.distV, distT, 0, 1, u => bindTex(0, gl.TEXTURE_2D, hT, null, u.M));
      mark('background test + distance');
      clear(outT); clear(inkT);
      gl.bindBufferBase(gl.UNIFORM_BUFFER, 2, embBuf);
      // the strips, bottom up (GL rows): own rows [r0, r1), activations from row o = r0 - REACH
      const one = crop[3] - crop[1] <= SH, step = one ? SH : SH - 2 * REACH;
      for (let r0 = crop[1]; r0 < crop[3]; r0 += step) {
      const r1 = Math.min(crop[3], r0 + step), o = one ? crop[1] : Math.max(crop[1], r0 - REACH), top = Math.min(crop[3], o + SH);
      org = [0, o];
      // the strip's passes: into the activations (rows o..top), then its own rows into the drawing
      const inStrip = () => { gl.viewport(0, 0, NW, SH); gl.scissor(crop[0], 0, crop[2] - crop[0], top - o); };
      const inFrame = () => { gl.viewport(0, 0, NW, NH); gl.scissor(crop[0], r0, crop[2] - crop[0], r1 - r0); };
      inStrip();
      for (let h = 0; h < 16 / M; h++) pass(prog.embed, X, M * h, M, u => {
        bindTiles(u); bindTex(5, gl.TEXTURE_2D, distT, null, u.distT);
        fine.forEach((t, k) => bindTex(6 + k, gl.TEXTURE_2D, t, null, u[`F${k}`]));
        coarse.forEach((t, k) => bindTex(9 + k, gl.TEXTURE_2D, t, null, u[`C${k}`]));
        gl.uniform1i(u.parts, P); gl.uniform1i(u.cols, meta.cols); gl.uniform1i(u.hf, h); gl.uniform2f(u.pitch, Math.fround(sinp), Math.fround(cosp));
      });
      mark('embed (inputs + neural texture)');
      let x = X, y = Y;
      for (let b = 0; b < NB; b++) {
        gl.bindBufferRange(gl.UNIFORM_BUFFER, 0, blkBuf, b * BLK, BLK);
        const common = u => {
          bindTex(0, gl.TEXTURE_2D_ARRAY, x.tex, null, u.X); bindTex(1, gl.TEXTURE_2D, distT, null, u.distT);
          gl.uniform1i(u.dil, DIL[b]); gl.uniform1i(u.needPrev, b ? need[b - 1] : 0); gl.uniform1i(u.needCur, need[b]);
        };
        gl.bindBufferRange(gl.UNIFORM_BUFFER, 1, wtBuf, (2 * b) * WT, WT);
        if (net.T) {   // trees: the walk, then the visited nodes' outputs
          const wt = (u, m) => { bindTex(3, gl.TEXTURE_2D, wTex, null, u.Wp); if (wfTex) bindTex(5, gl.TEXTURE_2D, wfTex, null, u.Wf); gl.uniform1i(u.wrow, (3 * b + m) * 64); };
          const st = u => St && bindTex(4, gl.TEXTURE_2D_ARRAY, St.tex, null, u.St);
          if (St) for (let h = 0; h < 16 / M; h++) pass(prog.TS, St, M * h, M, u => { common(u); gl.uniform1i(u.hf, h); });
          const ks = net.blocks[b].ssm;
          const sfu = u => { bindTex(6, gl.TEXTURE_2D, sfTex, null, u.Sf); gl.uniform1i(u.sbase, ks.slot * SR); };
          const sa = u => { common(u); wt(u, 0); st(u); sfu(u); gl.uniform1i(u.prow, (3 * b + 2) * 64); };
          if (ks && M >= 8) pass(prog.TAs, Cq, 0, 3 + net.T, sa);
          else if (ks) { pass(prog.TAs0, Cq, 0, 3, sa); pass(prog.TAs1, Cq, 3, net.T, sa); }   // 4 draw buffers: the walk twice
          else pass(prog.TA, Cq, 0, 3, u => { common(u); wt(u, 0); st(u); });
          mark(`block ${b} A: stencil, tree walk`);   // (the stencil pass counted in A)
          gl.bindBufferRange(gl.UNIFORM_BUFFER, 1, wtBuf, (2 * b + 1) * WT, WT);
          for (let h = 0; h < 16 / M; h++) pass(ks ? prog.TBs : prog.TB, y, M * h, M, u => { common(u); wt(u, 1); st(u); bindTex(2, gl.TEXTURE_2D_ARRAY, Cq.tex, null, u.C); gl.uniform1i(u.hf, h); if (ks) sfu(u); });
          mark(`block ${b} B: tree outputs, residual`);
        } else {
          if (M >= 5) pass(prog.A, Cq, 0, 5, common);
          else { pass(prog.A, Cq, 0, 4, common); pass(prog.A2, Cq, 4, 1, common); }   // 4 draw buffers: the scale on its own
          mark(`block ${b} A: stencil, FFN1, GELU`);
          gl.bindBufferRange(gl.UNIFORM_BUFFER, 1, wtBuf, (2 * b + 1) * WT, WT);
          for (let h = 0; h < 16 / M; h++) pass(prog.B, y, M * h, M, u => { common(u); bindTex(2, gl.TEXTURE_2D_ARRAY, Cq.tex, null, u.C); gl.uniform1i(u.hf, h); });
          mark(`block ${b} B: FFN2, residual`);
        }
        [x, y] = [y, x];
        if (b === NBC - 1) {   // the colour, before the ink branch's blocks (if any) go on
          inFrame();
          pass(prog.head, outT, 0, 1, u => {
            bindTex(0, gl.TEXTURE_2D_ARRAY, x.tex, null, u.X); bindTex(1, gl.TEXTURE_2D, distT, null, u.distT);
            gl.uniform4fv(u.Wh, WhT); gl.uniform4fv(u.bh, net.bh); gl.uniform4fv(u.pbg, pbg);
          });
          mark('head');
          inStrip();
        }
      }
      inFrame();
      if (net.Wl) pass(prog.ink, inkT, 0, 1, u => {
        bindTex(0, gl.TEXTURE_2D_ARRAY, x.tex, null, u.X); bindTex(1, gl.TEXTURE_2D, distT, null, u.distT);
        gl.uniform4fv(u.Wl, net.Wl); gl.uniform1f(u.bl, net.bl); gl.uniform1f(u.kbg, kbg);
      });
      }
    }
    // display
    gl.disable(gl.SCISSOR_TEST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, view.W, view.H);
    const s = prog.show; gl.useProgram(s.p);
    bindTex(0, gl.TEXTURE_2D, outT, null, s.u.R); bindTex(1, gl.TEXTURE_2D, tileTex(0), null, s.u.T0); bindTex(2, gl.TEXTURE_2D, inkT, null, s.u.K);
    gl.uniform4f(s.u.box, ...view.box); gl.uniform3f(s.u.top, ...view.top); gl.uniform3f(s.u.bottom, ...view.bottom);
    gl.uniform2f(s.u.view, view.W, view.H); gl.uniform1i(s.u.raw, view.raw ? (view.rawTex ? 2 : 1) : 0);
    if (view.rawTex) bindTex(9, gl.TEXTURE_2D, view.rawTex, null, s.u.RAW); gl.uniform1i(s.u.ink, view.ink === false ? 0 : 1);
    gl.uniform1i(s.u.toon, view.toon && !view.raw ? 1 : 0);
    gl.uniform1i(s.u.bgOn, view.backdrop ? 1 : 0);
    if (view.backdrop) setBackdropUniforms(gl, s.u, view.backdrop);
    gl.uniform1i(s.u.sharp, view.guide && !view.raw ? 1 : 0);
    if (view.guide) {
      bindTex(5, gl.TEXTURE_2D, view.guide, null, s.u.G);
      [1, 2, 4].forEach((k, i) => bindTex(6 + i, gl.TEXTURE_2D, tileTex(k), null, s.u[`T${k}`]));
      gl.uniform2f(s.u.lowRes, NW, NH);
    }
    if (view.toon) { bindTex(3, gl.TEXTURE_2D, view.toon.ink, null, s.u.TT); bindTex(4, gl.TEXTURE_2D, view.toon.depth, null, s.u.TD); gl.uniform3f(s.u.toonP, view.toon.near, view.toon.far, view.toon.width); }
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    mark(view.guide ? 'display (edge-aware upscale)' : 'display');
    gl.bindVertexArray(null);
    renderer.resetState();
  }
  // For checks: the drawing with its ink (premultiplied RGBA bytes, rows bottom-up) and the five tiles.
  function read() {
    const px = new Uint8Array(NW * NH * 4), ink = new Uint8Array(NW * NH * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo2d(outT)); gl.readPixels(0, 0, NW, NH, gl.RGBA, gl.UNSIGNED_BYTE, px);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo2d(inkT)); gl.readPixels(0, 0, NW, NH, gl.RGBA, gl.UNSIGNED_BYTE, ink);
    for (let i = 0; i < px.length; i += 4) {
      const k = ink[i] / 255;
      for (let c = 0; c < 3; c++) px[i + c] = Math.round(px[i + c] * (1 - k));
      px[i + 3] = Math.round(px[i + 3] + k * (255 - px[i + 3]));
    }
    const t = [0, 1, 2, 3, 4].map(k => {
      const b = new Uint8Array(NW * NH * 4);
      if (k) { gl.bindFramebuffer(gl.FRAMEBUFFER, fbo2d(gTex[k - 1])); gl.readPixels(0, 0, NW, NH, gl.RGBA, gl.UNSIGNED_BYTE, b); renderer.resetState(); }
      else renderer.readRenderTargetPixels(rtColour, 0, 0, NW, NH, b);
      return b;
    });
    renderer.resetState();
    return { px, tiles: t };
  }
  // Pass A by ablation (cpu.html #profA): one block's pass A alone, reps times, with each step left out
  // in turn; a step's cost is the time it saves. Run after a frame (it reads that frame's distances
  // and activations; the values don't matter, only the work). Each run closed by a readback, as prof.
  const variants = new Map();
  function benchA({ block = 3, reps = 8 } = {}) {
    const sets = { full: [], 'no stencil': ['STENCIL'], 'no int8 rounding': ['QUANT'], 'no FFN1': ['FFN'], 'no GELU': ['GELU'],
      'none (fetch, pack, write)': ['STENCIL', 'QUANT', 'FFN', 'GELU'] };
    const out = {};
    renderer.resetState(); gl.bindVertexArray(vao);
    gl.disable(gl.DEPTH_TEST); gl.disable(gl.BLEND); gl.disable(gl.CULL_FACE); gl.disable(gl.STENCIL_TEST);
    org = [0, crop[1]];   // the last frame's activations: its first strip
    gl.viewport(0, 0, NW, SH); gl.enable(gl.SCISSOR_TEST); gl.scissor(crop[0], 0, crop[2] - crop[0], Math.min(SH, crop[3] - crop[1]));
    gl.bindBufferRange(gl.UNIFORM_BUFFER, 0, blkBuf, block * BLK, BLK);
    gl.bindBufferRange(gl.UNIFORM_BUFFER, 1, wtBuf, (2 * block) * WT, WT);
    const common = u => {
      bindTex(0, gl.TEXTURE_2D_ARRAY, X.tex, null, u.X); bindTex(1, gl.TEXTURE_2D, distT, null, u.distT);
      gl.uniform1i(u.dil, DIL[block]); gl.uniform1i(u.needPrev, block ? need[block - 1] : 0); gl.uniform1i(u.needCur, need[block]);
    };
    for (const [name, ab] of Object.entries(sets)) {
      const key = ab.join();
      if (!variants.has(key)) variants.set(key, program(PASS_A(false, M, ab)));
      const pr = variants.get(key), m = M >= 5 ? 5 : 4;
      pass(pr, Cq, 0, m, common); sync();   // warm: the first draw compiles
      const t0 = performance.now();
      for (let r = 0; r < reps; r++) pass(pr, Cq, 0, m, common);
      sync();
      out[name] = (performance.now() - t0) / reps;
    }
    gl.bindVertexArray(null); renderer.resetState();
    return out;
  }
  return { renderTiles, frame, read, prof, benchA, parts: meta.parts, blocks: NB, ink: !!net.Wl };
}
