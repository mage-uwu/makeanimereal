// npmp.c: the two layer kinds of MoCapAct's NPMP for web/npmp.js, in WebAssembly SIMD (no libc).
//
//   wasm:  mocapact/build-wasm.sh  ->  web/npmp.wasm
//
// The host allocates float arrays in this module's memory with np_alloc (the network once, then
// the activations), and runs each layer: np_linear (y = W x + b, W row-major out x in) and
// np_norm_elu (LayerNorm with gain and bias, eps 1e-5, then ELU), in place.

#define EXPORT __attribute__((visibility("default")))
typedef float v4 __attribute__((vector_size(16), aligned(4)));   // 4 lanes, unaligned loads

static unsigned char heap[24 << 20] __attribute__((aligned(16)));
static unsigned long used;
EXPORT float *np_alloc(int n) {
  float *p = (float *)(heap + used);
  used += ((unsigned long)n * 4 + 15) & ~15ul;
  return used <= sizeof heap ? p : 0;
}

static float hsum(v4 a) { return a[0] + a[1] + a[2] + a[3]; }

// Four outputs at a time, so each load of x feeds four rows of W.
EXPORT void np_linear(float *y, const float *x, const float *W, const float *b, int in, int out) {
  int o = 0;
  for (; o + 4 <= out; o += 4) {
    const float *w0 = W + o * in, *w1 = w0 + in, *w2 = w1 + in, *w3 = w2 + in;
    v4 a0 = { 0 }, a1 = { 0 }, a2 = { 0 }, a3 = { 0 };
    int i = 0;
    for (; i + 4 <= in; i += 4) {
      v4 u = *(const v4 *)(x + i);
      a0 += u * *(const v4 *)(w0 + i); a1 += u * *(const v4 *)(w1 + i);
      a2 += u * *(const v4 *)(w2 + i); a3 += u * *(const v4 *)(w3 + i);
    }
    float s0 = hsum(a0), s1 = hsum(a1), s2 = hsum(a2), s3 = hsum(a3);
    for (; i < in; i++) { s0 += x[i] * w0[i]; s1 += x[i] * w1[i]; s2 += x[i] * w2[i]; s3 += x[i] * w3[i]; }
    y[o] = s0 + b[o]; y[o + 1] = s1 + b[o + 1]; y[o + 2] = s2 + b[o + 2]; y[o + 3] = s3 + b[o + 3];
  }
  for (; o < out; o++) {
    float s = b[o];
    for (int i = 0; i < in; i++) s += x[i] * W[o * in + i];
    y[o] = s;
  }
}

static float expf_(float x) {
  if (x < -87.f) return 0.f;
  float k = __builtin_floorf(x * 1.44269504f + 0.5f);
  float r = (x - k * 0.693145752f) - k * 1.42860677e-6f;
  float p = 1.f + r * (1.f + r * (0.5f + r * (1.f / 6 + r * (1.f / 24 + r * (1.f / 120 + r * (1.f / 720))))));
  union { float f; int i; } s = { .i = ((int)k + 127) << 23 };
  return p * s.f;
}

EXPORT void np_norm_elu(float *y, const float *g, const float *beta, int n) {
  float m = 0, v = 0;
  for (int i = 0; i < n; i++) m += y[i];
  m /= n;
  for (int i = 0; i < n; i++) v += (y[i] - m) * (y[i] - m);
  float inv = 1 / __builtin_sqrtf(v / n + 1e-5f);
  for (int i = 0; i < n; i++) {
    float t = (y[i] - m) * inv * g[i] + beta[i];
    y[i] = t > 0 ? t : expf_(t) - 1;
  }
}
