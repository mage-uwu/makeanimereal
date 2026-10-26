// vrmanet.c: the vrmanet forward pass (model.py's step(), line for line) in plain C.
//
//   native:  cc -O2 -shared -fPIC vrmanet/vrmanet.c -o vrmanet/libvrmanet.so -lm
//   wasm:    see vrmanet/build-wasm.sh (clang --target=wasm32, no libc)
//
// The host parses the .vrma and writes the model into the arrays vn_f32()/vn_i32() point at; this
// file has no parser, no libc and no allocator beyond one static arena. Per frame the host writes the
// control input (vn_f32(VN_U)), calls vn_step(dt) and reads vn_f32(VN_POSE): J quaternions (x, y, z,
// w) for the normalized humanoid bones in the file's bone order, then the hips' position.

#define EXPORT __attribute__((visibility("default")))

enum { VN_W, VN_PERIOD, VN_A_LOG, VN_B, VN_C, VN_D, VN_REST_HIPS, VN_WIDTH, VN_U, VN_X, VN_Y, VN_GATES, VN_RATE, VN_POSE, VN_F32 };
enum { VN_KEY_OFF, VN_KEY_CNT, VN_BONE_PART, VN_I32 };

static struct {
  int E, J, P, N, U, O, D, K, hips;
  float *f[VN_F32], *row;
  int *i[VN_I32];
  double *phase;  // the clocks run in double so long sessions don't drift
} m;

static unsigned char arena[48 << 20] __attribute__((aligned(16)));
static unsigned long used;

static void *take(unsigned long n) {
  void *p = arena + used;
  used += (n + 15) & ~15ul;
  if (used > sizeof arena) return 0;
  for (unsigned long k = 0; k < n; k++) ((unsigned char *)p)[k] = 0;
  return p;
}

#ifdef __wasm__
// clang may lower loops to these; there is no libc to provide them.
void *memset(void *d, int c, unsigned long n) { unsigned char *p = d; while (n--) *p++ = (unsigned char)c; return d; }
void *memcpy(void *d, const void *s, unsigned long n) { unsigned char *p = d; const unsigned char *q = s; while (n--) *p++ = *q++; return d; }
#endif

static float absf(float x) { return x < 0 ? -x : x; }

// e^x for the SSM's discretization (|relative error| < 2e-7 over the range used).
static float expf_(float x) {
  if (x < -87.f) return 0.f;
  if (x > 88.f) x = 88.f;
  float k = __builtin_floorf(x * 1.44269504f + 0.5f);
  float r = (x - k * 0.693145752f) - k * 1.42860677e-6f;
  float p = 1.f + r * (1.f + r * (0.5f + r * (1.f / 6 + r * (1.f / 24 + r * (1.f / 120 + r * (1.f / 720))))));
  union { float f; int i; } s = { .i = ((int)k + 127) << 23 };
  return p * s.f;
}

// Size the model: E experts, J bones, P body parts, an SSM of N states, U inputs and O outputs, K
// keyframes in all; hips = the hips' bone index. Returns 0, or -1 if it does not fit the arena.
EXPORT int vn_create(int E, int J, int P, int N, int U, int O, int K, int hips) {
  used = 0;
  m.E = E; m.J = J; m.P = P; m.N = N; m.U = U; m.O = O; m.K = K; m.hips = hips;
  m.D = 4 * J + 3;
  unsigned long fs[VN_F32] = {
    [VN_W] = (unsigned long)K * m.D, [VN_PERIOD] = E, [VN_A_LOG] = N, [VN_B] = (unsigned long)N * U,
    [VN_C] = (unsigned long)O * N, [VN_D] = (unsigned long)O * U, [VN_REST_HIPS] = 3, [VN_WIDTH] = 1,
    [VN_U] = U, [VN_X] = N, [VN_Y] = O, [VN_GATES] = (unsigned long)E * P, [VN_RATE] = E, [VN_POSE] = m.D,
  };
  unsigned long is[VN_I32] = { [VN_KEY_OFF] = E, [VN_KEY_CNT] = E, [VN_BONE_PART] = J };
  for (int k = 0; k < VN_F32; k++) if (!(m.f[k] = take(fs[k] * 4))) return -1;
  for (int k = 0; k < VN_I32; k++) if (!(m.i[k] = take(is[k] * 4))) return -1;
  if (!(m.row = take(m.D * 4)) || !(m.phase = take(E * 8))) return -1;
  return 0;
}

EXPORT float *vn_f32(int which) { return m.f[which]; }
EXPORT int *vn_i32(int which) { return m.i[which]; }
EXPORT float vn_phase(int e) { return (float)m.phase[e]; }

EXPORT void vn_reset(void) {
  for (int k = 0; k < m.N; k++) m.f[VN_X][k] = 0;
  for (int e = 0; e < m.E; e++) m.phase[e] = 0;
}

// Jump the state to its steady state for the current input: x = B u / exp(A_log).
EXPORT void vn_steady(void) {
  const float *B = m.f[VN_B], *u = m.f[VN_U];
  for (int n = 0; n < m.N; n++) {
    float s = 0;
    for (int j = 0; j < m.U; j++) s += B[n * m.U + j] * u[j];
    m.f[VN_X][n] = s / expf_(m.f[VN_A_LOG][n]);
  }
}

EXPORT void vn_trigger(int e) { m.phase[e] = 0; }

// Expert e's row at its phase: the triangle kernel over its keys, sign-aligned per bone.
static void row(int e) {
  int K = m.i[VN_KEY_CNT][e], off = m.i[VN_KEY_OFF][e];
  float w = m.f[VN_WIDTH][0], x = (float)(m.phase[e] * K), tot = 0, *r = m.row;
  for (int d = 0; d < m.D; d++) r[d] = 0;
  for (int k = (int)__builtin_ceilf(x - w); k <= (int)__builtin_floorf(x + w); k++) {
    float a = 1 - absf(x - k) / w;
    if (a <= 0) continue;
    const float *v = m.f[VN_W] + (long)(off + ((k % K) + K) % K) * m.D;
    for (int j = 0; j < m.J; j++) {
      const float *q = v + 4 * j;
      float *rq = r + 4 * j, s = q[0] * rq[0] + q[1] * rq[1] + q[2] * rq[2] + q[3] * rq[3] < 0 ? -a : a;
      for (int c = 0; c < 4; c++) rq[c] += s * q[c];
    }
    for (int c = 0; c < 3; c++) r[4 * m.J + c] += a * v[4 * m.J + c];
    tot += a;
  }
  for (int d = 0; d < m.D; d++) r[d] /= tot;
}

static void decode(void) {
  float *pose = m.f[VN_POSE], *g = m.f[VN_GATES], h_w = 0;
  const int *part = m.i[VN_BONE_PART];
  for (int d = 0; d < m.D; d++) pose[d] = 0;
  for (int e = 0; e < m.E; e++) {
    float any = 0;
    for (int p = 0; p < m.P; p++) if (g[e * m.P + p] > any) any = g[e * m.P + p];
    if (any <= 0) continue;
    row(e);
    for (int j = 0; j < m.J; j++) {
      float *q = m.row + 4 * j, *o = pose + 4 * j, ge = g[e * m.P + part[j]];
      if (o[0] * q[0] + o[1] * q[1] + o[2] * q[2] + o[3] * q[3] < 0) ge = -ge;
      for (int c = 0; c < 4; c++) o[c] += ge * q[c];
    }
    float gh = g[e * m.P + part[m.hips]];
    for (int c = 0; c < 3; c++) pose[4 * m.J + c] += gh * m.row[4 * m.J + c];
    h_w += gh;
  }
  for (int j = 0; j < m.J; j++) {
    float *o = pose + 4 * j, n = __builtin_sqrtf(o[0] * o[0] + o[1] * o[1] + o[2] * o[2] + o[3] * o[3]);
    if (n > 1e-8f) for (int c = 0; c < 4; c++) o[c] /= n;
    else { o[0] = o[1] = o[2] = 0; o[3] = 1; }
  }
  if (h_w <= 1e-6f) for (int c = 0; c < 3; c++) pose[4 * m.J + c] = m.f[VN_REST_HIPS][c];
}

// Advance dt seconds: the SSM (zero-order hold), its gate and rate heads, the clocks, the decoder.
EXPORT void vn_step(float dt) {
  float *x = m.f[VN_X], *y = m.f[VN_Y], *u = m.f[VN_U], *g = m.f[VN_GATES];
  const float *B = m.f[VN_B], *C = m.f[VN_C], *Dm = m.f[VN_D];
  for (int n = 0; n < m.N; n++) {
    float A = -expf_(m.f[VN_A_LOG][n]), Ad = expf_(A * dt), s = 0;
    for (int j = 0; j < m.U; j++) s += B[n * m.U + j] * u[j];
    x[n] = Ad * x[n] + (Ad - 1) / A * s;
  }
  for (int o = 0; o < m.O; o++) {
    float s = 0;
    for (int n = 0; n < m.N; n++) s += C[o * m.N + n] * x[n];
    for (int j = 0; j < m.U; j++) s += Dm[o * m.U + j] * u[j];
    y[o] = s;
  }
  for (int p = 0; p < m.P; p++) {
    float tot = 0;
    for (int e = 0; e < m.E; e++) tot += y[e * m.P + p] > 0 ? y[e * m.P + p] : 0;
    for (int e = 0; e < m.E; e++) g[e * m.P + p] = tot > 1e-6f && y[e * m.P + p] > 0 ? y[e * m.P + p] / tot : 0;
  }
  for (int e = 0; e < m.E; e++) {
    float rate = 1 + y[m.E * m.P + e];
    m.f[VN_RATE][e] = rate;
    double ph = m.phase[e] + (double)rate * dt / m.f[VN_PERIOD][e];
    m.phase[e] = ph - __builtin_floor(ph);
  }
  decode();
}
