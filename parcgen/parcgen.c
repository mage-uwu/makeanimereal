// parcgen.c: PARC's kinematic motion generator (parcgen/model.py, operation for operation) in C.
//
//   native:  cc -O2 -shared -fPIC parcgen/parcgen.c -o parcgen/libparcgen.so
//   wasm:    parcgen/build-wasm.sh  (clang --target=wasm32 -msimd128, no libc)
//
// The host writes the weights (parcgen/export_weights.py's .bin, float32, in WEIGHTS order) into
// pg_weights(), the world heightfield into pg_terrain(), the last two frames into pg_prev() and
// Gaussian noise into pg_noise(), then calls pg_generate(): pg_out() holds the next 15 frames.
// A frame is 78 floats in PARC's frame (z up): root position 3, root rotation 4 (x, y, z, w),
// 14 joint rotations 56, 15 contacts.

#define EXPORT __attribute__((visibility("default")))
typedef float v4 __attribute__((vector_size(16), aligned(4)));   // 4 lanes, unaligned loads

enum { D = 256, NH = 8, DH = 32, L = 4, S = 15, P = 2, F = 91, NTOK = 82, NCOND = 67, FRAME = 78, T = 21 };
enum { GX = 31, GY = 31 };                                        // heightmap: x 2 m back to 4 m ahead, y +-3 m
static const float CELL = 0.2f, MAX_H = 3.0f, NXN = 10, NYN = 15;

// --- memory ---------------------------------------------------------------------------------------

static unsigned char arena[40 << 20] __attribute__((aligned(16)));
static unsigned long used;
static void *take(unsigned long n) {
  void *p = arena + used;
  used += (n + 15) & ~15ul;
  return used <= sizeof arena ? p : 0;
}
#ifdef __wasm__
void *memset(void *d, int c, unsigned long n) { unsigned char *p = d; while (n--) *p++ = (unsigned char)c; return d; }
void *memcpy(void *d, const void *s, unsigned long n) { unsigned char *p = d; const unsigned char *q = s; while (n--) *p++ = *q++; return d; }
#endif

// --- maths (no libc) -----------------------------------------------------------------------------

static const float PI = 3.14159265358979f;
static float sqrtf_(float x) { return __builtin_sqrtf(x); }
static float absf(float x) { return x < 0 ? -x : x; }
static float expf_(float x) {
  if (x < -87.f) return 0.f;
  if (x > 88.f) x = 88.f;
  float k = __builtin_floorf(x * 1.44269504f + 0.5f);
  float r = (x - k * 0.693145752f) - k * 1.42860677e-6f;
  float p = 1.f + r * (1.f + r * (0.5f + r * (1.f / 6 + r * (1.f / 24 + r * (1.f / 120 + r * (1.f / 720))))));
  union { float f; int i; } s = { .i = ((int)k + 127) << 23 };
  return p * s.f;
}
static void sincosf_(float x, float *s, float *c) {
  float k = __builtin_rintf(x * (2 / PI));
  float r = (x - k * 1.5707963705062866f) + k * 4.37113883e-8f, r2 = r * r;
  float sn = r * (1 + r2 * (-1.f / 6 + r2 * (1.f / 120 + r2 * (-1.f / 5040 + r2 * (1.f / 362880)))));
  float cs = 1 + r2 * (-0.5f + r2 * (1.f / 24 + r2 * (-1.f / 720 + r2 * (1.f / 40320))));
  int q = (int)k & 3;
  *s = q == 0 ? sn : q == 1 ? cs : q == 2 ? -sn : -cs;
  *c = q == 0 ? cs : q == 1 ? -sn : q == 2 ? -cs : sn;
}
static float atanf_(float x) {                                   // Cephes atanf
  float sign = x < 0 ? -1.f : 1.f, y = 0;
  x = absf(x);
  if (x > 2.414213562f) { y = PI / 2; x = -1 / x; }
  else if (x > 0.414213562f) { y = PI / 4; x = (x - 1) / (x + 1); }
  float z = x * x;
  y += (((8.05374449538e-2f * z - 1.38776856032e-1f) * z + 1.99777106478e-1f) * z - 3.33329491539e-1f) * z * x + x;
  return sign * y;
}
static float atan2f_(float y, float x) {
  if (x > 0) return atanf_(y / x);
  if (x < 0) return atanf_(y / x) + (y >= 0 ? PI : -PI);
  return y > 0 ? PI / 2 : y < 0 ? -PI / 2 : 0;
}
static float erff_(float x) {                                    // Abramowitz-Stegun 7.1.26
  float s = x < 0 ? -1.f : 1.f;
  x = absf(x);
  float t = 1 / (1 + 0.3275911f * x);
  float y = 1 - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * expf_(-x * x);
  return s * y;
}

// --- quaternions (x, y, z, w) ---------------------------------------------------------------------

static void qmul(float *o, const float *a, const float *b) {
  float r[4] = { a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1], a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
                 a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3], a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2] };
  for (int i = 0; i < 4; i++) o[i] = r[i];
}
static void qrot(float *o, const float *q, const float *v) {
  float t[3] = { 2 * (q[1] * v[2] - q[2] * v[1]), 2 * (q[2] * v[0] - q[0] * v[2]), 2 * (q[0] * v[1] - q[1] * v[0]) };
  float r[3] = { v[0] + q[3] * t[0] + q[1] * t[2] - q[2] * t[1], v[1] + q[3] * t[1] + q[2] * t[0] - q[0] * t[2],
                 v[2] + q[3] * t[2] + q[0] * t[1] - q[1] * t[0] };
  for (int i = 0; i < 3; i++) o[i] = r[i];
}
static void axis_angle_to_quat(float *q, const float *axis, float angle) {
  float n = sqrtf_(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]), s, c;
  if (n < 1e-9f) n = 1e-9f;
  sincosf_(angle / 2, &s, &c);
  q[0] = axis[0] / n * s; q[1] = axis[1] / n * s; q[2] = axis[2] / n * s; q[3] = c;
  float m = sqrtf_(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  for (int i = 0; i < 4; i++) q[i] /= m > 1e-9f ? m : 1e-9f;
}
static float quat_to_axis_angle(const float *q0, float *axis) {
  float q[4], sg = q0[3] < 0 ? -1.f : 1.f;
  for (int i = 0; i < 4; i++) q[i] = sg * q0[i];
  float len = sqrtf_(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
  if (len <= 1e-5f) { axis[0] = axis[1] = 0; axis[2] = 1; return 0; }
  float l = len > 1e-6f ? len : 1e-6f;
  for (int i = 0; i < 3; i++) axis[i] = q[i] / l;
  return 2 * atan2f_(len, q[3]);
}
static void quat_to_exp_map(float *e, const float *q) {
  float axis[3], a = quat_to_axis_angle(q, axis);
  for (int i = 0; i < 3; i++) e[i] = axis[i] * a;
}
static void exp_map_to_quat(float *q, const float *e) {
  float a = sqrtf_(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]), axis[3] = { 0, 0, 1 }, s, c;
  if (a > 0) for (int i = 0; i < 3; i++) axis[i] = e[i] / a;
  sincosf_(a, &s, &c);
  a = atan2f_(s, c);
  if (absf(a) <= 1e-5f) { a = 0; axis[0] = axis[1] = 0; axis[2] = 1; }
  axis_angle_to_quat(q, axis, a);
}
static float heading(const float *q) {
  float x[3] = { 1, 0, 0 }, d[3];
  qrot(d, q, x);
  return atan2f_(d[1], d[0]);
}
static void heading_quat(float *q, float h) { float z[3] = { 0, 0, 1 }; axis_angle_to_quat(q, z, h); }

// --- the character ------------------------------------------------------------------------------

static const int PARENT[15] = { -1, 0, 1, 1, 3, 4, 1, 6, 7, 0, 9, 10, 0, 12, 13 };
static const float OFFSET[15][3] = { { 0, 0, 0 }, { 0, 0, .236151f }, { 0, 0, .223894f }, { -.02405f, -.18311f, .2435f },
  { 0, -.274788f, 0 }, { 0, -.258947f, 0 }, { -.02405f, .18311f, .2435f }, { 0, .274788f, 0 }, { 0, .258947f, 0 },
  { 0, -.084887f, 0 }, { 0, 0, -.421546f }, { 0, 0, -.40987f }, { 0, .084887f, 0 }, { 0, 0, -.421546f }, { 0, 0, -.40987f } };
static const int DOFS[15] = { 0, 3, 3, 3, 1, 0, 3, 1, 0, 3, 1, 3, 3, 1, 3 };
static const float HINGE[15][3] = { [4] = { 0, 0, 1 }, [7] = { 0, 0, 1 }, [10] = { 0, 1, 0 }, [13] = { 0, 1, 0 } };

static void fk(float pos[15][3], const float *rp, const float *rr, const float *jr) {
  float rot[15][4];
  for (int i = 0; i < 3; i++) pos[0][i] = rp[i];
  for (int i = 0; i < 4; i++) rot[0][i] = rr[i];
  for (int k = 1; k < 15; k++) {
    int p = PARENT[k];
    float o[3];
    qrot(o, rot[p], OFFSET[k]);
    for (int i = 0; i < 3; i++) pos[k][i] = pos[p][i] + o[i];
    qmul(rot[k], rot[p], jr + 4 * (k - 1));
  }
}

// Canonical frame -> standardized features (row f of the mean/std): model.py's assemble().
static float *W_mean, *W_std;
static void assemble(float *out, int f, const float *rp, const float *rr, const float *jr, const float *ct) {
  float pos[15][3], x[F];
  fk(pos, rp, rr, jr);
  for (int i = 0; i < 3; i++) x[i] = rp[i];
  quat_to_exp_map(x + 3, rr);
  for (int k = 1; k < 15; k++) for (int i = 0; i < 3; i++) x[6 + 3 * (k - 1) + i] = pos[k][i];
  int c = 48;
  for (int k = 1; k < 15; k++) {
    if (DOFS[k] == 3) { quat_to_exp_map(x + c, jr + 4 * (k - 1)); c += 3; }
    else if (DOFS[k] == 1) {
      float axis[3], a = quat_to_axis_angle(jr + 4 * (k - 1), axis);
      x[c++] = axis[0] * HINGE[k][0] + axis[1] * HINGE[k][1] + axis[2] * HINGE[k][2] < 0 ? -a : a;
    }
  }
  for (int i = 0; i < 15; i++) x[76 + i] = ct[i];
  for (int i = 0; i < F; i++) out[i] = (x[i] - W_mean[f * F + i]) / W_std[f * F + i];
}

// Standardized features (row f) -> canonical frame: model.py's extract().
static void extract(const float *in, int f, float *rp, float *rr, float *jr, float *ct) {
  float x[F];
  for (int i = 0; i < F; i++) x[i] = in[i] * W_std[f * F + i] + W_mean[f * F + i];
  for (int i = 0; i < 3; i++) rp[i] = x[i];
  exp_map_to_quat(rr, x + 3);
  int c = 48;
  for (int k = 1; k < 15; k++) {
    float *q = jr + 4 * (k - 1);
    if (DOFS[k] == 3) { exp_map_to_quat(q, x + c); c += 3; }
    else if (DOFS[k] == 1) { axis_angle_to_quat(q, HINGE[k], x[c]); c += 1; }
    else { q[0] = q[1] = q[2] = 0; q[3] = 1; }
  }
  for (int i = 0; i < 15; i++) ct[i] = x[76 + i];
}

// --- the network ---------------------------------------------------------------------------------

// Weights, in the order export_weights.py writes them.
static struct {
  float *conv_w[4], *conv_b[4], *obs_w, *obs_b, *tgt_w0, *tgt_b0, *tgt_w1, *tgt_b1, *flag, *pos_pe;
  float *qkv_w[L], *qkv_b[L], *out_w[L], *out_b[L], *l1_w[L], *l1_b[L], *l2_w[L], *l2_b[L], *n1_g[L], *n1_b[L], *n2_g[L], *n2_b[L];
  float *outm_w0, *outm_b0, *outm_w1, *outm_b1, *t_pe, *t_w0, *t_b0, *t_w1, *t_b1, *in_w0, *in_b0, *in_w1, *in_b1, *mean, *std;
} w;
static const int CONV_C[5] = { 1, 32, 32, 32, 64 }, CONV_K[4] = { 5, 5, 5, 4 };

#define NWEIGHTS 2206405
static float *weights;
static void bind_weights(void) {
  float *p = weights;
#define TAKE(dst, n) (dst = p, p += (n))
  for (int i = 0; i < 4; i++) { TAKE(w.conv_w[i], CONV_C[i + 1] * CONV_C[i] * CONV_K[i] * CONV_K[i]); TAKE(w.conv_b[i], CONV_C[i + 1]); }
  TAKE(w.obs_w, D * D); TAKE(w.obs_b, D);
  TAKE(w.tgt_w0, 512 * 2); TAKE(w.tgt_b0, 512); TAKE(w.tgt_w1, D * 512); TAKE(w.tgt_b1, D);
  TAKE(w.flag, 2 * D); TAKE(w.pos_pe, NTOK * D);
  for (int l = 0; l < L; l++) {
    TAKE(w.qkv_w[l], 3 * D * D); TAKE(w.qkv_b[l], 3 * D); TAKE(w.out_w[l], D * D); TAKE(w.out_b[l], D);
    TAKE(w.l1_w[l], D * D); TAKE(w.l1_b[l], D); TAKE(w.l2_w[l], D * D); TAKE(w.l2_b[l], D);
    TAKE(w.n1_g[l], D); TAKE(w.n1_b[l], D); TAKE(w.n2_g[l], D); TAKE(w.n2_b[l], D);
  }
  TAKE(w.outm_w0, D * D); TAKE(w.outm_b0, D); TAKE(w.outm_w1, F * D); TAKE(w.outm_b1, F);
  TAKE(w.t_pe, T * D); TAKE(w.t_w0, D * D); TAKE(w.t_b0, D); TAKE(w.t_w1, D * D); TAKE(w.t_b1, D);
  TAKE(w.in_w0, D * F); TAKE(w.in_b0, D); TAKE(w.in_w1, D * D); TAKE(w.in_b1, D);
  TAKE(w.mean, S * F); TAKE(w.std, S * F);
#undef TAKE
  W_mean = w.mean; W_std = w.std;
}

static float dot(const float *a, const float *b, int n) {
  v4 s0 = { 0 }, s1 = { 0 };
  int i = 0;
  for (; i + 8 <= n; i += 8) { s0 += *(const v4 *)(a + i) * *(const v4 *)(b + i); s1 += *(const v4 *)(a + i + 4) * *(const v4 *)(b + i + 4); }
  s0 += s1;
  float s = s0[0] + s0[1] + s0[2] + s0[3];
  for (; i < n; i++) s += a[i] * b[i];
  return s;
}

// y[r][o] = x[r] . W[o] + b[o], blocked 2 rows x 4 outputs so each load feeds several products
// (4 x 4 runs out of SIMD registers and is slower).
static void linear(float *y, const float *x, int rows, int in, const float *W, const float *b, int out) {
  int r = 0;
  for (; r + 2 <= rows; r += 2) {
    const float *x0 = x + r * in, *x1 = x0 + in;
    int o = 0;
    for (; o + 4 <= out; o += 4) {
      const float *w0 = W + o * in, *w1 = w0 + in, *w2 = w1 + in, *w3 = w2 + in;
      v4 a00 = { 0 }, a01 = { 0 }, a02 = { 0 }, a03 = { 0 }, a10 = { 0 }, a11 = { 0 }, a12 = { 0 }, a13 = { 0 };
      int i = 0;
      for (; i + 4 <= in; i += 4) {
        v4 u = *(const v4 *)(x0 + i), v = *(const v4 *)(x1 + i);
        v4 k0 = *(const v4 *)(w0 + i), k1 = *(const v4 *)(w1 + i), k2 = *(const v4 *)(w2 + i), k3 = *(const v4 *)(w3 + i);
        a00 += u * k0; a01 += u * k1; a02 += u * k2; a03 += u * k3;
        a10 += v * k0; a11 += v * k1; a12 += v * k2; a13 += v * k3;
      }
      float s[8] = { a00[0] + a00[1] + a00[2] + a00[3], a01[0] + a01[1] + a01[2] + a01[3], a02[0] + a02[1] + a02[2] + a02[3],
                     a03[0] + a03[1] + a03[2] + a03[3], a10[0] + a10[1] + a10[2] + a10[3], a11[0] + a11[1] + a11[2] + a11[3],
                     a12[0] + a12[1] + a12[2] + a12[3], a13[0] + a13[1] + a13[2] + a13[3] };
      for (; i < in; i++) {
        s[0] += x0[i] * w0[i]; s[1] += x0[i] * w1[i]; s[2] += x0[i] * w2[i]; s[3] += x0[i] * w3[i];
        s[4] += x1[i] * w0[i]; s[5] += x1[i] * w1[i]; s[6] += x1[i] * w2[i]; s[7] += x1[i] * w3[i];
      }
      for (int j = 0; j < 4; j++) { y[r * out + o + j] = s[j] + b[o + j]; y[(r + 1) * out + o + j] = s[4 + j] + b[o + j]; }
    }
    for (; o < out; o++) { y[r * out + o] = dot(x0, W + o * in, in) + b[o]; y[(r + 1) * out + o] = dot(x1, W + o * in, in) + b[o]; }
  }
  for (; r < rows; r++) for (int o = 0; o < out; o++) y[r * out + o] = dot(x + r * in, W + o * in, in) + b[o];
}

static void silu(float *x, int n) { for (int i = 0; i < n; i++) x[i] = x[i] / (1 + expf_(-x[i])); }
static void gelu(float *x, int n) { for (int i = 0; i < n; i++) x[i] = 0.5f * x[i] * (1 + erff_(x[i] * 0.70710678f)); }
static void add_norm(float *h, const float *a, const float *g, const float *b, int rows) {
  for (int r = 0; r < rows; r++) {
    float *x = h + r * D, m = 0, v = 0;
    for (int i = 0; i < D; i++) { x[i] += a[r * D + i]; m += x[i]; }
    m /= D;
    for (int i = 0; i < D; i++) v += (x[i] - m) * (x[i] - m);
    float inv = 1 / sqrtf_(v / D + 1e-5f);
    for (int i = 0; i < D; i++) x[i] = (x[i] - m) * inv * g[i] + b[i];
  }
}

// Scratch buffers.
static float *buf_h, *buf_qkv, *buf_a, *buf_f, *buf_col, *buf_c0, *buf_c1, *buf_tok, *buf_cond, *buf_cond_free;

// Terrain tokens (64 x D) from the 31 x 31 normalized heights: 4 valid convolutions (as matrix
// products over patches) with LeakyReLU, 2 x 2 patches of the 64 x 16 x 16 map, a linear embedding.
static void terrain_tokens(float *tok, const float *obs) {
  const float *in = obs;
  float *out = buf_c0;
  int H = GX, Wd = GY;
  for (int l = 0; l < 4; l++) {
    int C = CONV_C[l], O = CONV_C[l + 1], k = CONV_K[l], Ho = H - k + 1, Wo = Wd - k + 1, KK = C * k * k;
    for (int y = 0; y < Ho; y++) for (int x = 0; x < Wo; x++) {
      float *col = buf_col + (y * Wo + x) * KK;
      for (int c = 0; c < C; c++) for (int i = 0; i < k; i++) for (int j = 0; j < k; j++) col[(c * k + i) * k + j] = in[(c * H + y + i) * Wd + x + j];
    }
    linear(buf_f, buf_col, Ho * Wo, KK, w.conv_w[l], w.conv_b[l], O);       // [Ho*Wo][O]
    for (int o = 0; o < O; o++) for (int p = 0; p < Ho * Wo; p++) {        // -> [O][Ho][Wo], LeakyReLU
      float v = buf_f[p * O + o];
      out[o * Ho * Wo + p] = v > 0 ? v : 0.01f * v;
    }
    in = out; out = out == buf_c0 ? buf_c1 : buf_c0; H = Ho; Wd = Wo;
  }
  // in: 64 x 16 x 16 -> 64 tokens (8 x 8 patches) of 64 x 2 x 2.
  for (int bi = 0; bi < 8; bi++) for (int bj = 0; bj < 8; bj++) for (int c = 0; c < 64; c++)
    for (int ki = 0; ki < 2; ki++) for (int kj = 0; kj < 2; kj++)
      buf_col[(bi * 8 + bj) * D + c * 4 + ki * 2 + kj] = in[(c * 16 + bi * 2 + ki) * 16 + bj * 2 + kj];
  linear(tok, buf_col, 64, D, w.obs_w, w.obs_b, D);
}

// Condition tokens [64 terrain, target, flag] into cond (66 x D), from terrain tokens.
static void cond_tokens(float *cond, const float *tok, const float *target, int use_prev) {
  for (int i = 0; i < 64 * D; i++) cond[i] = tok[i];
  float hid[512];
  linear(hid, target, 1, 2, w.tgt_w0, w.tgt_b0, 512);
  silu(hid, 512);
  linear(cond + 64 * D, hid, 1, 512, w.tgt_w1, w.tgt_b1, D);
  for (int i = 0; i < D; i++) cond[65 * D + i] = w.flag[use_prev * D + i];
}

// One denoiser pass: x [15][F] at step t -> x0 [15][F].
static void denoise(float *x0, const float *x, int t, const float *cond, int mask_prev) {
  float *h = buf_h, hid[D];
  linear(hid, w.t_pe + t * D, 1, D, w.t_w0, w.t_b0, D);
  silu(hid, D);
  linear(h, hid, 1, D, w.t_w1, w.t_b1, D);
  for (int i = 0; i < 66 * D; i++) h[D + i] = cond[i];
  linear(buf_f, x, S, F, w.in_w0, w.in_b0, D);
  silu(buf_f, S * D);
  linear(h + NCOND * D, buf_f, S, D, w.in_w1, w.in_b1, D);
  for (int i = 0; i < NTOK * D; i++) h[i] += w.pos_pe[i];
  float scale = 1 / sqrtf_((float)DH);
  for (int l = 0; l < L; l++) {
    linear(buf_qkv, h, NTOK, D, w.qkv_w[l], w.qkv_b[l], 3 * D);
    for (int hd = 0; hd < NH; hd++) {
      for (int i = 0; i < NTOK; i++) {
        const float *q = buf_qkv + i * 3 * D + hd * DH;
        float s[NTOK], mx = -1e30f, tot = 0;
        for (int j = 0; j < NTOK; j++) {
          if (mask_prev && j >= NCOND && j < NCOND + P) { s[j] = -1e30f; continue; }
          s[j] = dot(q, buf_qkv + j * 3 * D + D + hd * DH, DH) * scale;
          if (s[j] > mx) mx = s[j];
        }
        for (int j = 0; j < NTOK; j++) { s[j] = s[j] <= -1e29f ? 0 : expf_(s[j] - mx); tot += s[j]; }
        v4 acc[DH / 4] = { { 0 } };
        for (int j = 0; j < NTOK; j++) {
          if (s[j] == 0) continue;
          v4 p = { s[j], s[j], s[j], s[j] };
          const float *v = buf_qkv + j * 3 * D + 2 * D + hd * DH;
          for (int d = 0; d < DH / 4; d++) acc[d] += p * *(const v4 *)(v + 4 * d);
        }
        for (int d = 0; d < DH / 4; d++) for (int c = 0; c < 4; c++) buf_f[i * D + hd * DH + 4 * d + c] = acc[d][c] / tot;
      }
    }
    linear(buf_a, buf_f, NTOK, D, w.out_w[l], w.out_b[l], D);
    add_norm(h, buf_a, w.n1_g[l], w.n1_b[l], NTOK);
    linear(buf_f, h, NTOK, D, w.l1_w[l], w.l1_b[l], D);
    gelu(buf_f, NTOK * D);
    linear(buf_a, buf_f, NTOK, D, w.l2_w[l], w.l2_b[l], D);
    add_norm(h, buf_a, w.n2_g[l], w.n2_b[l], NTOK);
  }
  // Only the frame tokens are read out.
  linear(buf_f, h + NCOND * D, S, D, w.outm_w0, w.outm_b0, D);
  silu(buf_f, S * D);
  linear(x0, buf_f, S, D, w.outm_w1, w.outm_b1, F);
}

// PARC's project_dofs: rebuild each frame's features through forward kinematics, clamp contacts.
static void project(float *x) {
  for (int f = 0; f < S; f++) {
    float rp[3], rr[4], jr[56], ct[15];
    extract(x + f * F, f, rp, rr, jr, ct);
    for (int i = 0; i < 15; i++) ct[i] = ct[i] < 0 ? 0 : ct[i] > 1 ? 1 : ct[i];
    assemble(x + f * F, f, rp, rr, jr, ct);
  }
}

// --- world, state and generation ------------------------------------------------------------------

static float *terrain, terrain_min[2], terrain_cell;
static int terrain_nx, terrain_ny;
static float *prev, *noise, *out, *x_t, *x_0, *x_free, *obs;
static double acp[T];

EXPORT int pg_create(void) {
  used = 0;
  if (!(weights = take(NWEIGHTS * 4))) return -1;
  buf_h = take(NTOK * D * 4); buf_qkv = take(NTOK * 3 * D * 4); buf_a = take(NTOK * D * 4);
  buf_f = take(27 * 27 * 64 * 4 > NTOK * D * 4 ? 27 * 27 * 64 * 4 : NTOK * D * 4);
  buf_col = take(23 * 23 * 32 * 25 * 4); buf_c0 = take(32 * 27 * 27 * 4); buf_c1 = take(32 * 27 * 27 * 4);
  buf_tok = take(64 * D * 4); buf_cond = take(66 * D * 4); buf_cond_free = take(66 * D * 4);
  prev = take(2 * FRAME * 4); noise = take(S * F * 4); out = take(S * FRAME * 4);
  x_t = take(S * F * 4); x_0 = take(S * F * 4); x_free = take(S * F * 4); obs = take(GX * GY * 4);
  if (!obs) return -1;
  // Cosine schedule, as diffusion_util.cosine_beta_schedule computes it (float32, s = 0.008).
  float a[T + 1];
  for (int i = 0; i <= T; i++) {
    float s, c;
    sincosf_((((float)i / T) + 0.008f) / 1.008f * 3.1415927410125732f * 0.5f, &s, &c);
    a[i] = c * c;
  }
  double cum = 1;
  for (int i = 0; i < T; i++) {
    float beta = 1 - (a[i + 1] / a[0]) / (a[i] / a[0]);
    beta = beta < 0.0001f ? 0.0001f : beta > 0.9999f ? 0.9999f : beta;
    cum *= 1 - (double)beta;
    acp[i] = cum;
  }
  return 0;
}

EXPORT float *pg_weights(void) { return weights; }
EXPORT int pg_num_weights(void) { return NWEIGHTS; }
EXPORT void pg_bind(void) { bind_weights(); }
EXPORT float *pg_prev(void) { return prev; }
EXPORT float *pg_noise(void) { return noise; }
EXPORT float *pg_out(void) { return out; }
EXPORT float *pg_obs(void) { return obs; }

// The world heightfield: heights[i * ny + j] at x = min_x + i * cell, y = min_y + j * cell.
EXPORT float *pg_terrain(int nx, int ny, float min_x, float min_y, float cell) {
  terrain_nx = nx; terrain_ny = ny; terrain_min[0] = min_x; terrain_min[1] = min_y; terrain_cell = cell;
  return terrain = take((unsigned long)nx * ny * 4);
}

static float ground(float x, float y) {
  int i = (int)__builtin_rintf((x - terrain_min[0]) / terrain_cell), j = (int)__builtin_rintf((y - terrain_min[1]) / terrain_cell);
  i = i < 0 ? 0 : i >= terrain_nx ? terrain_nx - 1 : i;
  j = j < 0 ? 0 : j >= terrain_ny ? terrain_ny - 1 : j;
  return terrain[i * terrain_ny + j];
}
EXPORT float pg_ground(float x, float y) { return ground(x, y); }

// The next 15 frames from pg_prev(), heading for (tx, ty) (world xy; (0, 0) = stand), DDIM at
// stride (10: steps 20, 10, 0), with CFG at cfg_scale (< 0: off).
EXPORT void pg_generate(float tx, float ty, int stride, float cfg_scale) {
  const float *p1 = prev + FRAME;
  float cxy[2] = { p1[0], p1[1] }, cz = p1[2], h = heading(p1 + 3), hq[4], hq_inv[4], c, s;
  heading_quat(hq, h); heading_quat(hq_inv, -h);
  sincosf_(h, &s, &c);
  // The previous frames in the canonical frame, as features.
  float prev_f[P * F];
  for (int f = 0; f < P; f++) {
    const float *fr = prev + f * FRAME;
    float rp[3] = { fr[0] - cxy[0], fr[1] - cxy[1], fr[2] }, rr[4];
    qrot(rp, hq_inv, rp);
    rp[2] -= cz;
    qmul(rr, hq_inv, fr + 3);
    assemble(prev_f + f * F, f, rp, rr, fr + 7, fr + 63);
  }
  // The terrain around the root, relative to its height, normalized.
  for (int i = 0; i < GX; i++) for (int j = 0; j < GY; j++) {
    float gx = -NXN * CELL + i * CELL, gy = -NYN * CELL + j * CELL;
    float v = ground(gx * c - gy * s + cxy[0], gx * s + gy * c + cxy[1]) - cz;
    v = v < -MAX_H ? -MAX_H : v > MAX_H ? MAX_H : v;
    obs[i * GY + j] = v / MAX_H;
  }
  float target[2] = { tx * c + ty * s, -tx * s + ty * c };
  terrain_tokens(buf_tok, obs);
  cond_tokens(buf_cond, buf_tok, target, 1);
  if (cfg_scale >= 0) cond_tokens(buf_cond_free, buf_tok, target, 0);
  for (int i = 0; i < S * F; i++) x_t[i] = noise[i];
  for (int t = (T - 1) / stride * stride; t >= 0; t -= stride) {          // PARC: reversed(range(0, T, stride))
    for (int i = 0; i < P * F; i++) x_t[i] = prev_f[i];
    denoise(x_0, x_t, t, buf_cond, 0);
    if (cfg_scale >= 0) {
      denoise(x_free, x_t, t, buf_cond_free, 1);
      for (int i = 0; i < S * F; i++) x_0[i] += cfg_scale * (x_free[i] - x_0[i]);
    }
    project(x_0);
    if (t - stride < 0) break;
    int n = t - stride;
    double a_t = __builtin_sqrt(acp[t]), a_n = __builtin_sqrt(acp[n]), s_t = __builtin_sqrt(1 - acp[t]), s_n = __builtin_sqrt(1 - acp[n]);
    float k0 = (float)(a_n - a_t * s_n / s_t), kt = (float)(s_n / s_t);
    for (int i = 0; i < S * F; i++) x_t[i] = k0 * x_0[i] + kt * x_t[i];
  }
  // Back to the world.
  for (int f = 0; f < S; f++) {
    float *fr = out + f * FRAME, rp[3], rr[4];
    extract(x_0 + f * F, f, rp, rr, fr + 7, fr + 63);
    qmul(fr + 3, hq, rr);
    qrot(rp, hq, rp);
    fr[0] = rp[0] + cxy[0]; fr[1] = rp[1] + cxy[1]; fr[2] = rp[2] + cz;
    for (int i = 0; i < 15; i++) fr[63 + i] = fr[63 + i] < 0 ? 0 : fr[63 + i] > 1 ? 1 : fr[63 + i];
  }
}
