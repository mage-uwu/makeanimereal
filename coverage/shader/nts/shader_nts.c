/*
 * shader_nts.c -- the learned render shader, trained on the nanotss engine (nanotss_blt.c, included
 * whole with NANOTSS_NO_MAIN).
 *
 * Pixels are the tokens. A crop of the G-buffer is a "sequence" of h x w tokens of width 64:
 *   input  : colour 3, normal 3, depth, coverage, 8 neural-texture channels (bilinear in each part's
 *            slice of a 512^2 + 64^2 feature grid, as train.py's NeuralTexture), sin/cos pitch
 *   embed  : dense 18 -> 64
 *   blocks : the engine's ternary stencil mixer (tc_forward/tc_backward) over a 3 x 3 neighbour map
 *            at dilation 1, 2, 4, 8, 16, ... (receptive field of a small U-Net, no pooling), then a
 *            ternary FFN 64 -> 64 -> GELU -> 64 with a residual: quant8 activations, VNNI
 *            qrows_dot8_batch forward, qrows_axpy_batch / outer_rows backward (straight-through)
 *   head   : dense 64 -> 4, sigmoid: straight RGBA, as the PyTorch shader
 * Loss: L1 on coverage plus L1 on colour where the target has coverage (train.py's, without VGG).
 * One OpenMP team per step, as faithful_step: a crop per thread, per-thread gradients, a reduction,
 * unified clipping; Muon (muon_apply) on the ternary FFN matrices, AdamW (adam) on everything else,
 * the neural texture with its own learning rate (its gradients are added atomically).
 *
 * --tree T,V: each block's FFN is instead the engine's fast feedforward trees (fff_fwd / fff_bwd): T
 * binary trees of V levels; a token visits one node a level, s = w_in . x + b (ternary w_in, quant8 x),
 * adds gelu(s) w_out (ternary) and moves to the child sign(s) picks. Routing learns from the engine's
 * credit: each decision against the subtree it passed over (every alternative walked, V being small).
 * --distill (with --load of a dense model): each block's trees learn its dense FFN's output from the
 * same input, everything else as loaded; then train as usual with --load of the result.
 *
 * Build: cc -O3 -march=native -fopenmp shader_nts.c -lm -o shader_nts
 * Data : python3 pack_data.py <dataset dirs> --out data.bin
 * Check: ./shader_nts --check            (finite-difference gradients, continuous weights)
 * Train: ./shader_nts --data data.bin --iters 12000 --batch 32 --crop 96 --save model.ntsw
 */
#define NANOTSS_NO_MAIN
#include "nanotss_blt.c"

#define DW 64          /* token width */
#define KT 9           /* 3 x 3 stencil */
#define CIN 18
#define FCH 8
#define FINE 512
#define COARSE 64
#define MAXB 16

static int NB = 8;
static int DIL[MAXB] = {1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16};
/* --lines K: an ink branch. The first NBC blocks and the colour head are a trained model, frozen (the
 * colour head reads block NBC's output, so colours can't change); K more blocks continue from there
 * into a 1-channel head: the opacity of black line art drawn over the colour. 0: no branch. */
static int NBC = 0;
/* How the viewers draw the ink head's opacity: a smoothstep from 0.4 to 0.7, so confident strokes go
 * solid black and the haze of unsure ones (an average over where a stroke might fall) goes away. */
static float ink_style(float k) { float t = fminf(1, fmaxf(0, (k - .4f) / .3f)); return t * t * (3 - 2 * t); }
static int check_mode = 0;
static int TREES = 0, TV = 4, TN = 15, TAL = 6;   /* --tree: trees, levels, nodes a tree (2^V - 1), alternatives a tree (V(V-1)/2) */
static int tree_on = 0;                           /* the forward pass uses the trees (0: the dense FFN, the distillation teacher) */
static float tree_temp = 1;                       /* the engine's routing-credit temperature */

typedef struct { int N, H, W, P; float *pitch; uint8_t *px; } Data;

/* ---------------- parameters ---------------- */
static int P_PARTS = 18;
static size_t o_We, o_be, o_tc[MAXB], o_b1[MAXB], o_b2[MAXB], o_Wh, o_bh, o_Wl, o_bl, n_dense;
static float *dense, *dm, *dv;                 /* Adam: embedding, stencils, biases, head */
static float *W1[MAXB], *W2[MAXB];             /* Muon masters (ternary FFN) */
static float *Q1[MAXB], *Q2[MAXB];             /* their ternary values */
static float *M1[MAXB], *M2[MAXB];             /* Muon momentum */
static Plane PL1[MAXB], PL2[MAXB];
static Codes codes;
static TCCode *TCC[MAXB];
static float *tex, *gtex, *tm, *tv;            /* neural texture, its gradient and Adam moments */
static uint8_t *thit;                          /* texels touched this step (Adam updates only those, as the engine's hit rows) */
static size_t n_tex;
static float **gcz_t;  /* per-thread coarse texture gradients */
#define N_COARSE ((size_t)P_PARTS * COARSE * COARSE * FCH)
static int nthreads = 1;
static float **gd_t, ***gW1_t, ***gW2_t;       /* per-thread gradients */
static float *gd, *gW1[MAXB], *gW2[MAXB];
static float *TI[MAXB], *TO[MAXB], *QI[MAXB], *QO[MAXB];   /* trees: node rows w_in, w_out (Muon masters, ternary values) */
static float *MI[MAXB], *MO[MAXB], *gTI[MAXB], *gTO[MAXB], ***gTI_t, ***gTO_t;
static size_t o_tb[MAXB];                                   /* their node biases, in dense */
static int8_t *QIc[MAXB]; static float *QIs[MAXB];          /* w_in's codes and row scales: s exactly as infer.c computes it */

/* ---------------- narrow-row VNNI rows (d = 64) ----------------
 * qrows_dot8_batch reduces each output across 16 lanes, built for the engine's wide rows; at 64
 * channels that reduction is most of the work. Same arithmetic (quant8 codes, the 0x80 offset and its
 * 128 * sum(w) correction, per-row ternary scale), transposed: weights packed [input chunk][output][4],
 * 4 activation bytes broadcast, so one dpbusd makes 16 outputs and nothing is reduced across lanes. */
typedef struct { int8_t w[16 * DW * 4]; int32_t corr[DW]; float sc[DW]; } Rows64;
static Rows64 R1[MAXB], R2[MAXB];
static void rows64_pack(const float *Q, Rows64 *R) {
    for (int r = 0; r < DW; r++) {
        float m = 0; int sum = 0;
        for (int i = 0; i < DW; i++) {
            float q = Q[r * DW + i]; int c = (q > 0) - (q < 0);
            R->w[((i / 4) * DW + r) * 4 + (i % 4)] = (int8_t)c; sum += c; if (fabsf(q) > m) m = fabsf(q);
        }
        R->corr[r] = 128 * sum; R->sc[r] = m;
    }
}
static void rows64_dot8(const Rows64 *R, const int8_t *xq, const float *xs, float *out, int L) {
#if FAST && defined(__AVX512VNNI__)
    const __m512i flip = _mm512_set1_epi32((int)0x80808080);
    for (int t = 0; t < L; t++) {
        const int32_t *x = (const int32_t *)(xq + (size_t)t * DW);
        __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
        for (int kc = 0; kc < 16; kc++) {
            __m512i v = _mm512_xor_si512(_mm512_set1_epi32(x[kc]), flip);
            const int8_t *w = R->w + (size_t)kc * DW * 4;
            a0 = _mm512_dpbusd_epi32(a0, v, _mm512_loadu_si512(w));
            a1 = _mm512_dpbusd_epi32(a1, v, _mm512_loadu_si512(w + 64));
            a2 = _mm512_dpbusd_epi32(a2, v, _mm512_loadu_si512(w + 128));
            a3 = _mm512_dpbusd_epi32(a3, v, _mm512_loadu_si512(w + 192));
        }
        const __m512 s = _mm512_set1_ps(xs[t]);
        __m512i acc[4] = {a0, a1, a2, a3};
        for (int q = 0; q < 4; q++) {
            __m512i c = _mm512_sub_epi32(acc[q], _mm512_loadu_si512(R->corr + 16 * q));
            _mm512_storeu_ps(out + (size_t)t * DW + 16 * q, _mm512_mul_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(c), _mm512_loadu_ps(R->sc + 16 * q)), s));
        }
    }
#else
    for (int t = 0; t < L; t++) for (int r = 0; r < DW; r++) {
        int acc = 0; for (int i = 0; i < DW; i++) acc += xq[(size_t)t * DW + i] * R->w[((i / 4) * DW + r) * 4 + (i % 4)];
        out[(size_t)t * DW + r] = R->sc[r] * acc * xs[t];
    }
#endif
}

/* The engine's quant8 and gelu_array, specialised to 64-wide float tokens. quant8 rounds and clamps
 * exactly as the engine does; gelu shares one exp(-x^2/2) between erf (Abramowitz-Stegun 7.1.26,
 * |error| < 1.5e-7) and the derivative instead of two double-precision libm calls. */
#if FAST && defined(__AVX512F__)
static inline __m512 exp_ps(__m512 x) {
    x = _mm512_max_ps(x, _mm512_set1_ps(-87.f));
    __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693359375f), x);
    r = _mm512_fnmadd_ps(n, _mm512_set1_ps(-2.12194440e-4f), r);
    __m512 p = _mm512_set1_ps(1.9875691500e-4f);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.3981999507e-3f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(8.3334519073e-3f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(4.1665795894e-2f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.6666665459e-1f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(5.0000001201e-1f));
    p = _mm512_fmadd_ps(p, _mm512_mul_ps(r, r), _mm512_add_ps(r, _mm512_set1_ps(1.f)));
    return _mm512_scalef_ps(p, n);
}
#endif
static void quant8_64(const float *x, float *xq, int8_t *code, float *xs) {
#if FAST && defined(__AVX512F__)
    if (!continuous_check) {
        __m512 v[4], m = _mm512_setzero_ps();
        for (int q = 0; q < 4; q++) { v[q] = _mm512_loadu_ps(x + 16 * q); m = _mm512_max_ps(m, _mm512_abs_ps(v[q])); }
        float mx = _mm512_reduce_max_ps(m); if (mx < 1e-5f) mx = 1e-5f;
        const float scale = 127 / mx; const __m512 S = _mm512_set1_ps(scale);
        for (int q = 0; q < 4; q++) {
            __m512 r = _mm512_roundscale_ps(_mm512_mul_ps(v[q], S), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            r = _mm512_min_ps(_mm512_max_ps(r, _mm512_set1_ps(-128.f)), _mm512_set1_ps(127.f));
            _mm512_storeu_ps(xq + 16 * q, _mm512_div_ps(r, S));
            _mm_storeu_si128((__m128i *)(code + 16 * q), _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(r)));
        }
        *xs = 1 / scale; return;
    }
#endif
    quant8(x, xq, code, xs, DW);
}
static void gelu_n(const float *x, float *y, float *dy, size_t n) {
    size_t i = 0;
#if FAST && defined(__AVX512F__)
    const __m512 half = _mm512_set1_ps(.5f), one = _mm512_set1_ps(1.f);
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(x + i), z = _mm512_abs_ps(_mm512_mul_ps(v, _mm512_set1_ps(.70710678f)));
        __m512 e = exp_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(-.5f), v), v));
        __m512 t = _mm512_div_ps(one, _mm512_fmadd_ps(z, _mm512_set1_ps(.3275911f), one));
        __m512 p = _mm512_set1_ps(1.061405429f);
        p = _mm512_fmadd_ps(p, t, _mm512_set1_ps(-1.453152027f));
        p = _mm512_fmadd_ps(p, t, _mm512_set1_ps(1.421413741f));
        p = _mm512_fmadd_ps(p, t, _mm512_set1_ps(-.284496736f));
        p = _mm512_fmadd_ps(p, t, _mm512_set1_ps(.254829592f));
        __m512 er = _mm512_fnmadd_ps(_mm512_mul_ps(p, t), e, one);                 /* erf(|z|) */
        er = _mm512_mask_sub_ps(er, _mm512_cmp_ps_mask(v, _mm512_setzero_ps(), _CMP_LT_OQ), _mm512_setzero_ps(), er);
        __m512 c = _mm512_mul_ps(half, _mm512_add_ps(one, er));
        _mm512_storeu_ps(y + i, _mm512_mul_ps(v, c));
        _mm512_storeu_ps(dy + i, _mm512_fmadd_ps(_mm512_mul_ps(v, e), _mm512_set1_ps(.39894228f), c));
    }
#endif
    for (; i < n; i++) { y[i] = gelu(x[i]); dy[i] = dgelu(x[i]); }
}

static size_t tex_fine_off(int part) { return (size_t)part * FINE * FINE * FCH; }
static size_t tex_coarse_off(int part) { return (size_t)P_PARTS * FINE * FINE * FCH + (size_t)part * COARSE * COARSE * FCH; }

static void layout(void) {
    size_t o = 0;
    o_We = o; o += (size_t)DW * CIN; o_be = o; o += DW;
    for (int b = 0; b < NB; b++) {
        o_tc[b] = o; o += tc_size(DW, KT); o_b1[b] = o; o += DW; o_b2[b] = o; o += DW;
        if (TREES) { o_tb[b] = o; o += (size_t)TREES * TN; }
    }
    o_Wh = o; o += 4 * DW; o_bh = o; o += 4;
    if (NBC) { o_Wl = o; o += DW; o_bl = o; o += 4; }
    n_dense = o;
    n_tex = (size_t)P_PARTS * (FINE * FINE + COARSE * COARSE) * FCH;
}

static void requantize(void) {
    for (int b = 0; b < NB; b++) {
        if (check_mode) { memcpy(Q1[b], W1[b], sizeof(float) * DW * DW); memcpy(Q2[b], W2[b], sizeof(float) * DW * DW); }
        else {
            tss_ternary(W1[b], Q1[b], DW, DW, DW); tss_ternary(W2[b], Q2[b], DW, DW, DW);
            plane_rows(Q1[b], &PL1[b], 0, DW, DW, DW); plane_rows(Q2[b], &PL2[b], 0, DW, DW, DW);
            rows64_pack(Q1[b], &R1[b]); rows64_pack(Q2[b], &R2[b]);
        }
        if (TREES) {
            size_t n = (size_t)TREES * TN * DW;
            if (check_mode) { memcpy(QI[b], TI[b], sizeof(float) * n); memcpy(QO[b], TO[b], sizeof(float) * n); }
            else {
                tss_ternary(TI[b], QI[b], (long)TREES * TN, DW, DW); tss_ternary(TO[b], QO[b], (long)TREES * TN, DW, DW);
                for (int r = 0; r < TREES * TN; r++) {
                    float m = 0;
                    for (int i = 0; i < DW; i++) { float q = QI[b][r * DW + i]; QIc[b][r * DW + i] = (int8_t)((q > 0) - (q < 0)); if (fabsf(q) > m) m = fabsf(q); }
                    QIs[b][r] = m;
                }
            }
        }
        tc_pack(dense + o_tc[b], TCC[b]);
    }
}

static void net_alloc(int threads) {
    layout();
    dense = calloc(n_dense, 4); dm = calloc(n_dense, 4); dv = calloc(n_dense, 4); gd = calloc(n_dense, 4);
    for (int b = 0; b < NB; b++) {
        W1[b] = calloc(DW * DW, 4); W2[b] = calloc(DW * DW, 4); Q1[b] = calloc(DW * DW, 4); Q2[b] = calloc(DW * DW, 4);
        M1[b] = calloc(DW * DW, 4); M2[b] = calloc(DW * DW, 4); gW1[b] = calloc(DW * DW, 4); gW2[b] = calloc(DW * DW, 4);
        PL1[b] = plane_new(DW, DW, DW); PL2[b] = plane_new(DW, DW, DW);
        TCC[b] = tc_code_new(DW, KT);
        if (TREES) {
            size_t n = (size_t)TREES * TN * DW;
            TI[b] = calloc(n, 4); TO[b] = calloc(n, 4); QI[b] = calloc(n, 4); QO[b] = calloc(n, 4);
            MI[b] = calloc(n, 4); MO[b] = calloc(n, 4); gTI[b] = calloc(n, 4); gTO[b] = calloc(n, 4);
            QIc[b] = calloc(n, 1); QIs[b] = calloc((size_t)TREES * TN, 4);
        }
    }
    codes.ok = FAST;
    tex = calloc(n_tex, 4); gtex = calloc(n_tex, 4); tm = calloc(n_tex, 4); tv = calloc(n_tex, 4); thit = calloc(n_tex / FCH, 1);
    nthreads = threads;
    gd_t = malloc(threads * sizeof(float *)); gW1_t = malloc(threads * sizeof(float **)); gW2_t = malloc(threads * sizeof(float **));
    gcz_t = malloc(threads * sizeof(float *));
    for (int t = 0; t < threads; t++) {
        gcz_t[t] = calloc(N_COARSE, 4);
        gd_t[t] = calloc(n_dense, 4); gW1_t[t] = malloc(MAXB * sizeof(float *)); gW2_t[t] = malloc(MAXB * sizeof(float *));
        for (int b = 0; b < NB; b++) { gW1_t[t][b] = calloc(DW * DW, 4); gW2_t[t][b] = calloc(DW * DW, 4); }
    }
    if (TREES) {
        gTI_t = malloc(threads * sizeof(float **)); gTO_t = malloc(threads * sizeof(float **));
        for (int t = 0; t < threads; t++) {
            gTI_t[t] = malloc(MAXB * sizeof(float *)); gTO_t[t] = malloc(MAXB * sizeof(float *));
            for (int b = 0; b < NB; b++) { gTI_t[t][b] = calloc((size_t)TREES * TN * DW, 4); gTO_t[t][b] = calloc((size_t)TREES * TN * DW, 4); }
        }
    }
}

static void net_init(void) {
    for (size_t i = 0; i < (size_t)DW * CIN; i++) dense[o_We + i] = (float)(randn() / sqrt(CIN));
    for (int b = 0; b < NB; b++) {
        tc_init(dense + o_tc[b], DW, KT);
        for (int i = 0; i < DW * DW; i++) { W1[b][i] = (float)(randn() / sqrt(DW)); W2[b][i] = (float)(randn() * 0.02); }
        if (TREES) for (size_t i = 0; i < (size_t)TREES * TN * DW; i++) { TI[b][i] = (float)(randn() / sqrt(DW)); TO[b][i] = (float)(randn() * 0.02); }
    }
    for (int i = 0; i < 4 * DW; i++) dense[o_Wh + i] = (float)(randn() * 0.02);
    if (NBC) { for (int i = 0; i < DW; i++) dense[o_Wl + i] = (float)(randn() * 0.02); dense[o_bl] = -4; }   /* little ink to start */
    for (int p = 0; p < P_PARTS; p++) {
        for (size_t i = 0; i < (size_t)FINE * FINE * FCH; i++) tex[tex_fine_off(p) + i] = (float)(randn() * 0.01);
        for (size_t i = 0; i < (size_t)COARSE * COARSE * FCH; i++) tex[tex_coarse_off(p) + i] = (float)(randn() * 0.1);
    }
    requantize();
}

/* ---------------- pixels ---------------- */
typedef struct { float u, v; int part; float mask; } UV;
static inline void decode(const uint8_t *q, float pitch, float *in, UV *uv) {
    for (int c = 0; c < 3; c++) in[c] = q[c] / 255.f;
    for (int c = 0; c < 3; c++) in[3 + c] = q[3 + c] / 255.f * 2 - 1;
    in[6] = q[6] / 255.f;
    float mask = q[12] > 127 ? 1.f : 0.f;
    in[7] = mask;
    uv->u = (q[7] + q[8] / 255.f) / 255.f;
    uv->v = (q[9] + q[10] / 255.f) / 255.f;
    uv->part = q[11] < P_PARTS ? q[11] : 0;
    uv->mask = mask;
    float p = pitch * 0.017453292519943295f;
    in[16] = sinf(p); in[17] = cosf(p);
}
/* grid_sample, bilinear, align_corners=False, zero padding: one part's grid of side S. */
static inline void tex_one(const float *g, int S, float u, float v, float *out, float w0) {
    float px = u * S - 0.5f, py = v * S - 0.5f;
    int x0 = (int)floorf(px), y0 = (int)floorf(py);
    float fx = px - x0, fy = py - y0;
    for (int j = 0; j < 2; j++) for (int i = 0; i < 2; i++) {
        int x = x0 + i, y = y0 + j;
        if (x < 0 || y < 0 || x >= S || y >= S) continue;
        float w = w0 * (i ? fx : 1 - fx) * (j ? fy : 1 - fy);
        const float *t = g + ((size_t)y * S + x) * FCH;
        for (int c = 0; c < FCH; c++) out[c] += w * t[c];
    }
}
/* G's texel 0 is texel hit0 of the whole texture. The fine level (151 MB) is shared, with atomic adds;
 * the coarse level, which every crop hits, gets a private copy per thread (gcz_t), summed once per step. */
static inline void tex_one_bwd(float *G, size_t hit0, int S, float u, float v, const float *dout, int atomic) {
    float px = u * S - 0.5f, py = v * S - 0.5f;
    int x0 = (int)floorf(px), y0 = (int)floorf(py);
    float fx = px - x0, fy = py - y0;
    for (int j = 0; j < 2; j++) for (int i = 0; i < 2; i++) {
        int x = x0 + i, y = y0 + j;
        if (x < 0 || y < 0 || x >= S || y >= S) continue;
        float w = (i ? fx : 1 - fx) * (j ? fy : 1 - fy);
        size_t k = (size_t)y * S + x; float *t = G + k * FCH;
        thit[hit0 + k] = 1;
        if (atomic) for (int c = 0; c < FCH; c++) {
#pragma omp atomic
            t[c] += w * dout[c];
        }
        else for (int c = 0; c < FCH; c++) t[c] += w * dout[c];
    }
}

/* ---------------- neighbour maps ---------------- */
static int *neighbours(int h, int w, int dil) {
    int *m = malloc(sizeof(int) * (size_t)h * w * KT);
    for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) {
        int t = r * w + c, k = 0;
        for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++, k++) {
            int rr = r + dy * dil, cc = c + dx * dil;
            m[(size_t)t * KT + k] = (rr >= 0 && rr < h && cc >= 0 && cc < w) ? rr * w + cc : -1;
        }
    }
    return m;
}

/* ---------------- workspace ---------------- */
typedef struct {
    int L, keep;
    float *in; UV *uv;
    float *x[MAXB + 1];
    TCWork *tw[MAXB];
    float *xq1[MAXB], *xs1[MAXB], *dg[MAXB], *g[MAXB], *xs2[MAXB];
    int8_t *c1[MAXB], *c2[MAXB];
    float *z, *p, *zl, *pl;   /* colour head logits / RGBA; ink head (line mode) */
    float *dx, *dxt, *din;
    float *ta, *tb;  /* TT-token scratch: pre-GELU / FFN out forward, dgv / dh backward */
    int *tnode[MAXB], *tanode[MAXB]; float *ts[MAXB], *tas[MAXB];   /* trees: visited nodes and their s; the alternatives' (training) */
} WS;
/* Per-token work (the FFNs) runs over tiles of TT tokens, so its intermediates stay in L1/L2 instead
 * of streaming 2.4 MB arrays through DRAM, which is what limits 8 threads sharing one memory system. */
enum { TT = 128 };
static WS *ws_new(int L, int keep) {
    WS *w = calloc(1, sizeof(WS)); w->L = L; w->keep = keep;
    size_t LD = (size_t)L * DW;
    w->in = calloc((size_t)L * CIN, 4); w->uv = calloc(L, sizeof(UV));
    int nx = keep ? NB + 1 : 2;
    /* without backward, two buffers in turn; in line mode block NBC's output keeps its own (the colour head reads it after the branch) */
    for (int b = 0; b <= NB; b++) w->x[b] = b < nx || (NBC && b == NBC) ? calloc(LD, 4) : w->x[b % 2];
    for (int b = 0; b < NB; b++) {
        int own = keep || b == 0;
        w->tw[b] = own ? tc_work_new(L, DW) : w->tw[0];
        w->xq1[b] = own ? calloc(LD, 4) : w->xq1[0]; w->xs1[b] = own ? calloc(L, 4) : w->xs1[0];
        w->dg[b] = own ? calloc(LD, 4) : w->dg[0];
        w->g[b] = own ? calloc(LD, 4) : w->g[0]; w->xs2[b] = own ? calloc(L, 4) : w->xs2[0];
        w->c1[b] = own ? calloc(LD, 1) : w->c1[0]; w->c2[b] = own ? calloc(LD, 1) : w->c2[0];
        if (TREES) {
            size_t nv = (size_t)L * TREES * TV, na = (size_t)L * TREES * TAL;
            w->tnode[b] = own ? calloc(nv, 4) : w->tnode[0]; w->ts[b] = own ? calloc(nv, 4) : w->ts[0];
            w->tanode[b] = keep ? calloc(na, 4) : NULL; w->tas[b] = keep ? calloc(na, 4) : NULL;
        }
    }
    w->z = calloc((size_t)L * 4, 4); w->p = calloc((size_t)L * 4, 4); w->zl = calloc(L, 4); w->pl = calloc(L, 4);
    w->ta = calloc(TT * DW, 4); w->tb = calloc(TT * DW, 4);
    if (keep) { w->dx = calloc(LD, 4); w->dxt = calloc(LD, 4); w->din = calloc((size_t)L * CIN, 4); }
    return w;
}

/* ---------------- forward / backward ---------------- */
static double tphase[6]; static int profile = 0;
/* thread 0's time per stage: input, embed, stencil, ffn rows, gelu, head | head, ffn outer, ffn axpy, stencil, embed+texture */
static double tst[12]; static int prof_on = 0;  /* only training steps count, not evaluation */
#define TS(k, stmt) do { double _t = (profile && prof_on && omp_get_thread_num() == 0) ? wall() : 0; stmt; if (profile && prof_on && omp_get_thread_num() == 0) tst[k] += wall() - _t; } while (0)
static int engine_rows = 0;  /* --engine-rows: the engine's qrows_dot8_batch instead of rows64_dot8 */
static void ffn_rows(const float *Q, Plane *pl, const Rows64 *R, const float *xin, float *xq, int8_t *code, float *xs, float *out, int L) {
    /* check mode: the continuous path (no codes); the engine still reads a scale per token */
    if (check_mode) { for (int t = 0; t < L; t++) xs[t] = 1; qrows_dot8_batch(NULL, pl, Q, 0, xin, code, xs, out, L, DW, DW, DW); if (xq != xin) memcpy(xq, xin, sizeof(float) * L * DW); return; }
    for (int t = 0; t < L; t++) quant8_64(xin + (size_t)t * DW, xq + (size_t)t * DW, code + (size_t)t * DW, xs + t);
    if (engine_rows) qrows_dot8_batch(&codes, pl, Q, 0, xin, code, xs, out, L, DW, DW, DW);
    else rows64_dot8(R, code, xs, out, L);
}

/* ---------------- trees (fff_fwd / fff_bwd for one token) ----------------
 * x: the token's quant8 input (code, xs: its codes and scale; NULL in the gradient check's continuous
 * mode); y += the trees' path sum. s is the integer sum of codes x w_in's row scale x xs + the bias,
 * in that order, as infer.c and shader_torch.py compute it: a tie (an integer sum of 0 at a zero bias,
 * common in fresh trees) then goes the same way everywhere. node / sv: the visited rows and their s, a
 * tree's V in turn; anode / as (training): below each decision but the last, the walk down the other
 * child, (V - 1) + ... + 1 rows a tree. */
static inline float node_s(int b, int row, const float *x, const int8_t *code, float xs) {
    if (!code) return dotr(QI[b] + (size_t)row * DW, x, DW) + dense[o_tb[b] + row];
    const int8_t *w = QIc[b] + (size_t)row * DW; int sum = 0;
    for (int i = 0; i < DW; i++) sum += w[i] * code[i];
    return (float)sum * QIs[b][row] * xs + dense[o_tb[b] + row];
}
static void tree_fwd(int b, const float *x, const int8_t *code, float xs, float *y, int *node, float *sv, int *anode, float *as) {
    const float *QOb = QO[b];
    for (int tr = 0; tr < TREES; tr++) {
        int a = 0, *nd = node + tr * TV; float *sd = sv + tr * TV;
        for (int j = 0; j < TV; j++) {
            int row = tr * TN + a; float s = node_s(b, row, x, code, xs);
            nd[j] = row; sd[j] = s;
            axpy(y, gelu(s), QOb + (size_t)row * DW, DW);
            a = 2 * a + 1 + (s > 0);
        }
        if (!anode) continue;
        int k = tr * TAL;
        for (int j = 0; j + 1 < TV; j++) {
            int a2 = 2 * (nd[j] - tr * TN) + 2 - (sd[j] > 0);   /* the child not taken */
            for (int jj = j + 1; jj < TV; jj++, k++) {
                int row = tr * TN + a2; float s = node_s(b, row, x, code, xs);
                anode[k] = row; as[k] = s; a2 = 2 * a2 + 1 + (s > 0);
            }
        }
    }
}
/* gy: d/d the path sum. Gradients into thread th's buffers; dx (if any) += d/dx (straight through
 * quant8). anode NULL: no routing credit (the gradient check's fixed paths). */
static void tree_bwd(int b, int th, const float *x, const float *gy, float *dx, const int *node, const float *sv, const int *anode, const float *as) {
    float *GI = gTI_t[th][b], *GO = gTO_t[th][b], *Gb = gd_t[th] + o_tb[b];
    const float *QIb = QI[b], *QOb = QO[b];
    float on[16], ds[16];
    for (int tr = 0; tr < TREES; tr++) {
        const int *nd = node + tr * TV; const float *sd = sv + tr * TV;
        for (int j = 0; j < TV; j++) {
            float go = dotr(QOb + (size_t)nd[j] * DW, gy, DW);
            on[j] = gelu(sd[j]) * go; ds[j] = dgelu(sd[j]) * go;
        }
        if (anode) {   /* each decision: what its subtree gave against what the other one would have */
            int k = tr * TAL;
            for (int j = 0; j + 1 < TV; j++) {
                float o = 0, alt = 0;
                for (int jj = j + 1; jj < TV; jj++, k++) { o += on[jj]; alt += gelu(as[k]) * dotr(QOb + (size_t)anode[k] * DW, gy, DW); }
                float sg = 1 / (1 + expf(-fabsf(sd[j]) / tree_temp));
                ds[j] += (sd[j] > 0 ? 1 : -1) * (o - alt) * sg * (1 - sg) / tree_temp;
            }
        }
        for (int j = 0; j < TV; j++) {
            int row = nd[j];
            axpy(GO + (size_t)row * DW, gelu(sd[j]), gy, DW);
            axpy(GI + (size_t)row * DW, ds[j], x, DW); Gb[row] += ds[j];
            if (dx) axpy(dx, ds[j], QIb + (size_t)row * DW, DW);
        }
    }
}

/* src: packed pixel rows of the crop (h x w at stride Wf), pitch. Leaves straight RGBA in w->p. */
static void forward(WS *w, const uint8_t *src, int Wf, int h, int wd, float pitch, int *const *nb) {
    int L = h * wd;
    double _t0 = (profile && prof_on && omp_get_thread_num() == 0) ? wall() : 0;
    for (int r = 0; r < h; r++) for (int c = 0; c < wd; c++) {
        int t = r * wd + c; float *in = w->in + (size_t)t * CIN;
        decode(src + ((size_t)r * Wf + c) * 17, pitch, in, w->uv + t);
        float f[FCH] = {0};
        if (w->uv[t].mask > 0) {
            tex_one(tex + tex_fine_off(w->uv[t].part), FINE, w->uv[t].u, w->uv[t].v, f, 1);
            tex_one(tex + tex_coarse_off(w->uv[t].part), COARSE, w->uv[t].u, w->uv[t].v, f, 1);
        }
        for (int c2 = 0; c2 < FCH; c2++) in[8 + c2] = f[c2];
    }
    if (profile && prof_on && omp_get_thread_num() == 0) tst[0] += wall() - _t0;
    const float *We = dense + o_We, *be = dense + o_be;
    TS(1, for (int t = 0; t < L; t++) {
        float *x = w->x[0] + (size_t)t * DW; const float *in = w->in + (size_t)t * CIN;
        for (int j = 0; j < DW; j++) x[j] = be[j] + dotr(We + j * CIN, in, CIN);
    });
    for (int b = 0; b < NB; b++) {
        TCWork *tw = w->tw[b];
        TS(2, tc_forward(dense + o_tc[b], TCC[b], w->x[b], L, nb[b], tw));
        const float *b1 = dense + o_b1[b], *b2 = dense + o_b2[b];
        float *o = w->x[b + 1];
        if (tree_on) {   /* x[b+1] = t + b2 + the trees' path sum on quant8(t) */
            for (int t = 0; t < L; t++) {
                const float *ti = tw->out + (size_t)t * DW; float *xq = w->xq1[b] + (size_t)t * DW, *ot = o + (size_t)t * DW;
                if (check_mode) memcpy(xq, ti, sizeof(float) * DW); else quant8_64(ti, xq, w->c1[b] + (size_t)t * DW, w->xs1[b] + t);
                for (int j = 0; j < DW; j++) ot[j] = b2[j] + ti[j];
                size_t v = (size_t)t * TREES * TV, a = (size_t)t * TREES * TAL;
                tree_fwd(b, xq, check_mode ? NULL : w->c1[b] + (size_t)t * DW, w->xs1[b][t], ot, w->tnode[b] + v, w->ts[b] + v, w->tanode[b] && !check_mode ? w->tanode[b] + a : NULL, w->tas[b] ? w->tas[b] + a : NULL);
            }
            continue;
        }
        for (int t0 = 0; t0 < L; t0 += TT) {
            int n = L - t0 < TT ? L - t0 : TT; size_t off = (size_t)t0 * DW;
            TS(3, ffn_rows(Q1[b], &PL1[b], &R1[b], tw->out + off, w->xq1[b] + off, w->c1[b] + off, w->xs1[b] + t0, w->ta, n));
            for (int t = 0; t < n; t++) for (int j = 0; j < DW; j++) w->ta[t * DW + j] += b1[j];
            TS(4, gelu_n(w->ta, w->g[b] + off, w->dg[b] + off, (size_t)n * DW));
            /* g := its quantised self */
            TS(3, ffn_rows(Q2[b], &PL2[b], &R2[b], w->g[b] + off, w->g[b] + off, w->c2[b] + off, w->xs2[b] + t0, w->tb, n));
            for (int t = 0; t < n; t++) for (int j = 0; j < DW; j++) o[off + t * DW + j] = w->tb[t * DW + j] + b2[j] + tw->out[off + t * DW + j];
        }
    }
    const float *Wh = dense + o_Wh, *bh = dense + o_bh;
    const float *xc = w->x[NBC ? NBC : NB];   /* the colour head: after block NBC in line mode */
    for (int t = 0; t < L; t++) for (int c = 0; c < 4; c++) {
        float z = bh[c] + dotr(Wh + c * DW, xc + (size_t)t * DW, DW);
        w->z[t * 4 + c] = z; w->p[t * 4 + c] = 1 / (1 + expf(-z));
    }
    if (NBC) for (int t = 0; t < L; t++) {
        float z = dense[o_bl] + dotr(dense + o_Wl, w->x[NB] + (size_t)t * DW, DW);
        w->zl[t] = z; w->pl[t] = 1 / (1 + expf(-z));
    }
}

/* train.py's loss on straight RGBA against the target bytes; returns it and leaves d/dz in w->z. */
static double loss_grad(WS *w, const uint8_t *src, int Wf, int h, int wd, int grad) {
    int L = h * wd; double la = 0, lc = 0;
    for (int r = 0; r < h; r++) for (int c = 0; c < wd; c++) {
        int t = r * wd + c; const uint8_t *q = src + ((size_t)r * Wf + c) * 17 + 13;
        float ta = q[3] / 255.f, *p = w->p + t * 4, *z = w->z + t * 4;
        float da = p[3] - ta; la += fabsf(da);
        float g[4];
        g[3] = (da > 0 ? 1.f : da < 0 ? -1.f : 0.f) / L;
        for (int k = 0; k < 3; k++) {
            float e = (p[k] - q[k] / 255.f) * ta; lc += fabsf(e);
            g[k] = (e > 0 ? 1.f : e < 0 ? -1.f : 0.f) * ta / (3.f * L);
        }
        if (grad) for (int k = 0; k < 4; k++) z[k] = g[k] * p[k] * (1 - p[k]);
    }
    return la / L + lc / (3.0 * L);
}

/* Line mode: weighted cross-entropy of the ink against the target's red byte (lines are sparse, so ink
 * counts 1 + 6 x as much); returns the mean |ink - target| and leaves d/dz in w->zl. */
static double line_grad(WS *w, const uint8_t *src, int Wf, int h, int wd, int grad) {
    int L = h * wd; double err = 0;
    for (int r = 0; r < h; r++) for (int c = 0; c < wd; c++) {
        int t = r * wd + c; float k = src[((size_t)r * Wf + c) * 17 + 13] / 255.f, p = w->pl[t];
        err += fabsf(p - k);
        if (grad) w->zl[t] = (1 + 6 * k) * (p - k) / L;
    }
    return err / L;
}

static void backward(WS *w, int th, int h, int wd, int *const *nb) {
    int L = h * wd; size_t LD = (size_t)L * DW;
    float *gdt = gd_t[th];
    /* head */
    float *Gh = gdt + o_Wh, *Gbh = gdt + o_bh; const float *Wh = dense + o_Wh;
    memset(w->dx, 0, LD * 4);
    if (NBC) for (int t = 0; t < L; t++) {   /* line mode: the ink head only, then the branch's blocks */
        float dz = w->zl[t]; if (dz == 0) continue;
        gdt[o_bl] += dz; axpy(gdt + o_Wl, dz, w->x[NB] + (size_t)t * DW, DW); axpy(w->dx + (size_t)t * DW, dz, dense + o_Wl, DW);
    }
    else for (int t = 0; t < L; t++) {
        const float *x = w->x[NB] + (size_t)t * DW; float *dx = w->dx + (size_t)t * DW;
        for (int c = 0; c < 4; c++) {
            float dz = w->z[t * 4 + c]; if (dz == 0) continue;
            Gbh[c] += dz; axpy(Gh + c * DW, dz, x, DW); axpy(dx, dz, Wh + c * DW, DW);
        }
    }
    for (int b = NB - 1; b >= NBC; b--) {
        TCWork *tw = w->tw[b];
        /* x[b+1] = tw.out + Q2 g + b2, g = gelu(Q1 xq1 + b1) */
        float *Gb2 = gdt + o_b2[b], *Gb1 = gdt + o_b1[b];
        if (tree_on) for (int t = 0; t < L; t++) {
            const float *gy = w->dx + (size_t)t * DW; float *dxt = w->dxt + (size_t)t * DW;
            for (int j = 0; j < DW; j++) Gb2[j] += gy[j];
            memcpy(dxt, gy, sizeof(float) * DW);   /* residual */
            size_t v = (size_t)t * TREES * TV, a = (size_t)t * TREES * TAL;
            tree_bwd(b, th, w->xq1[b] + (size_t)t * DW, gy, dxt, w->tnode[b] + v, w->ts[b] + v, check_mode ? NULL : w->tanode[b] + a, check_mode ? NULL : w->tas[b] + a);
        }
        else for (int t0 = 0; t0 < L; t0 += TT) {
            int n = L - t0 < TT ? L - t0 : TT; size_t off = (size_t)t0 * DW;
            const float *dx = w->dx + off, *dg = w->dg[b] + off; float *dgv = w->ta, *dh = w->tb, *dxt = w->dxt + off;
            for (int t = 0; t < n; t++) for (int j = 0; j < DW; j++) Gb2[j] += dx[t * DW + j];
            TS(7, outer_rows(gW2_t[th][b], dx, w->g[b] + off, n, DW, DW));
            memset(dgv, 0, sizeof(float) * n * DW);
            TS(8, qrows_axpy_batch(check_mode ? NULL : &codes, &PL2[b], Q2[b], 0, dx, dgv, n, DW, DW, DW));
            for (int i = 0; i < n * DW; i++) dh[i] = dgv[i] * dg[i];
            for (int t = 0; t < n; t++) for (int j = 0; j < DW; j++) Gb1[j] += dh[t * DW + j];
            TS(7, outer_rows(gW1_t[th][b], dh, w->xq1[b] + off, n, DW, DW));
            memcpy(dxt, dx, sizeof(float) * n * DW);   /* residual */
            TS(8, qrows_axpy_batch(check_mode ? NULL : &codes, &PL1[b], Q1[b], 0, dh, dxt, n, DW, DW, DW));
        }
        /* stencil mixer: dx := d/d x[b] */
        TS(9, tc_backward(dense + o_tc[b], gdt + o_tc[b], TCC[b], w->x[b], w->dxt, w->dx, L, nb[b], tw));
    }
    if (NBC) return;   /* the frozen trunk takes no gradient */
    /* embedding and texture */
    float *GWe = gdt + o_We, *Gbe = gdt + o_be; const float *We = dense + o_We;
    double _t10 = (profile && prof_on && omp_get_thread_num() == 0) ? wall() : 0;
    for (int t = 0; t < L; t++) {
        const float *dx = w->dx + (size_t)t * DW, *in = w->in + (size_t)t * CIN;
        float din[CIN] = {0};
        for (int j = 0; j < DW; j++) { Gbe[j] += dx[j]; axpy(GWe + j * CIN, dx[j], in, CIN); axpy(din, dx[j], We + j * CIN, CIN); }
        const UV *uv = w->uv + t;
        if (uv->mask > 0) {
            tex_one_bwd(gtex + tex_fine_off(uv->part), tex_fine_off(uv->part) / FCH, FINE, uv->u, uv->v, din + 8, 1);
            size_t co = tex_coarse_off(uv->part) - tex_coarse_off(0);
            tex_one_bwd(gcz_t[th] + co, tex_coarse_off(uv->part) / FCH, COARSE, uv->u, uv->v, din + 8, 0);
        }
    }
    if (profile && prof_on && omp_get_thread_num() == 0) tst[10] += wall() - _t10;
}

/* --distill: after the dense forward pass, each block's trees against its dense FFN (Q2 g, without b2,
 * which the trees share) on the same quant8 input; squared error, mean per token. Returns it per value. */
static double distill_grad(WS *w, int th, int L) {
    double err = 0; float y[DW], gy[DW];
    for (int b = NBC; b < NB; b++) {
        const float *b2 = dense + o_b2[b];
        for (int t = 0; t < L; t++) {
            const float *ti = w->tw[b]->out + (size_t)t * DW, *xq = w->xq1[b] + (size_t)t * DW, *xo = w->x[b + 1] + (size_t)t * DW;
            size_t v = (size_t)t * TREES * TV, a = (size_t)t * TREES * TAL;
            memset(y, 0, sizeof y);
            tree_fwd(b, xq, w->c1[b] + (size_t)t * DW, w->xs1[b][t], y, w->tnode[b] + v, w->ts[b] + v, w->tanode[b] + a, w->tas[b] + a);
            for (int j = 0; j < DW; j++) { float e = y[j] - (xo[j] - ti[j] - b2[j]); err += (double)e * e; gy[j] = 2 * e / L; }
            tree_bwd(b, th, xq, gy, NULL, w->tnode[b] + v, w->ts[b] + v, w->tanode[b] + a, w->tas[b] + a);
        }
    }
    return err / ((double)L * DW * (NB - NBC));
}
/* What Adam may move: in --distill only the trees' node biases; in line mode the branch and its head. */
static int trainable(size_t i, int distill) {
    if (distill) { for (int b = 0; b < NB; b++) if (i >= o_tb[b] && i < o_tb[b] + (size_t)TREES * TN) return 1; return 0; }
    if (NBC) return (i >= o_tc[NBC] && i < o_Wh) || i >= o_Wl;
    return 1;
}

/* ---------------- data ---------------- */
static Data load_data(const char *path) {
    Data d = {0}; FILE *f = fopen(path, "rb"); char mg[4];
    if (!f || fread(mg, 1, 4, f) != 4 || memcmp(mg, "NTSD", 4)) { fprintf(stderr, "bad data %s\n", path); exit(1); }
    int hd[4]; if (fread(hd, 4, 4, f) != 4) exit(1);
    d.N = hd[0]; d.H = hd[1]; d.W = hd[2]; d.P = hd[3];
    d.pitch = malloc(4 * d.N); if (fread(d.pitch, 4, d.N, f) != (size_t)d.N) exit(1);
    size_t n = (size_t)d.N * d.H * d.W * 17; d.px = malloc(n);
    if (fread(d.px, 1, n, f) != n) { fprintf(stderr, "short data\n"); exit(1); }
    fclose(f); return d;
}
static const uint8_t *frame_px(const Data *d, int i) { return d->px + (size_t)i * d->H * d->W * 17; }

/* ---------------- one training step (faithful_step's shape) ---------------- */
   /* forward+backward, reduce, norm(tex), adam, muon, adam(tex) */
static double train_step(const Data *d, const int *train_ids, int ntrain, int B, int crop, int *const *nb, WS **wss,
                         const Hyper *h, const Hyper *ht, uint64_t seed, int distill) {
    double loss = 0, nrm = 0; real scale = 0;
    real c1 = 1 - pow(h->beta1, (real)h->step), c2 = 1 - pow(h->beta2, (real)h->step);
    int ch = crop < d->H ? crop : d->H, cw = crop < d->W ? crop : d->W;
    double tp[7]; tp[0] = wall(); prof_on = 1;
#pragma omp parallel num_threads(nthreads)
    {
        int th = omp_get_thread_num();
#pragma omp for schedule(dynamic, 1) reduction(+:loss)
        for (int b = 0; b < B; b++) {
            uint64_t s = seed * 0x9E3779B97F4A7C15ULL + b * 7919 + 1;
            int fi = train_ids[(int)(urand(&s) * ntrain)];
            int cy = (int)(urand(&s) * (d->H - ch + 1)), cx = (int)(urand(&s) * (d->W - cw + 1));
            const uint8_t *src = frame_px(d, fi) + ((size_t)cy * d->W + cx) * 17;
            forward(wss[th], src, d->W, ch, cw, d->pitch[fi], nb);
            if (distill) { loss += distill_grad(wss[th], th, ch * cw); continue; }
            loss += NBC ? line_grad(wss[th], src, d->W, ch, cw, 1) : loss_grad(wss[th], src, d->W, ch, cw, 1);
            backward(wss[th], th, ch, cw, nb);
        }
#pragma omp single
        tp[1] = wall();
#pragma omp for schedule(static)
        for (size_t i = 0; i < N_COARSE; i++) { float g = 0; for (int t = 0; t < nthreads; t++) { g += gcz_t[t][i]; gcz_t[t][i] = 0; } gtex[tex_coarse_off(0) + i] = g; }
#pragma omp for schedule(static) reduction(+:nrm)
        for (size_t i = 0; i < n_dense; i++) { float g = 0; for (int t = 0; t < nthreads; t++) { g += gd_t[t][i]; gd_t[t][i] = 0; } gd[i] = g; nrm += (double)g * g; }
        const int TM = TREES * TN * DW;   /* a block's tree matrix */
#pragma omp for schedule(static) reduction(+:nrm)
        for (int k = 0; k < (TREES ? 2 * NB * TM : 0); k++) {
            int m = k / (NB * TM), b = (k / TM) % NB, i = k % TM;
            float ***src_t = m ? gTO_t : gTI_t; float *dst = m ? gTO[b] : gTI[b];
            float g = 0; for (int t = 0; t < nthreads; t++) { g += src_t[t][b][i]; src_t[t][b][i] = 0; } dst[i] = g; nrm += (double)g * g;
        }
#pragma omp for schedule(static) reduction(+:nrm)
        for (int k = 0; k < (TREES ? 0 : 2 * NB * DW * DW); k++) {
            int m = k / (NB * DW * DW), b = (k / (DW * DW)) % NB, i = k % (DW * DW);
            float ***src_t = m ? gW2_t : gW1_t; float *dst = m ? gW2[b] : gW1[b];
            float g = 0; for (int t = 0; t < nthreads; t++) { g += src_t[t][b][i]; src_t[t][b][i] = 0; } dst[i] = g; nrm += (double)g * g;
        }
#pragma omp single
        tp[2] = wall();
#pragma omp for schedule(static) reduction(+:nrm)
        for (size_t i = 0; i < n_tex / FCH; i++) if (thit[i]) for (int c = 0; c < FCH; c++) nrm += (double)gtex[i * FCH + c] * gtex[i * FCH + c];
#pragma omp single
        tp[3] = wall();
#pragma omp single
        { scale = (real)1 / B; double norm = sqrt(nrm) * scale; if (h->clip > 0 && norm > h->clip) scale *= h->clip / (norm + 1e-6); }
#pragma omp for schedule(static)
        for (size_t off = 0; off < n_dense; off += 1024) {
            size_t n = n_dense - off < 1024 ? n_dense - off : 1024;
            if (!NBC && !distill) { adam(dense + off, gd + off, dm + off, dv + off, n, h, scale, c1, c2); continue; }
            /* the rest stays exactly as it is (Adam's decay would shift it) */
            for (size_t i = off; i < off + n; i++)
                if (trainable(i, distill)) adam(dense + i, gd + i, dm + i, dv + i, 1, h, scale, c1, c2);
                else gd[i] = 0;
        }
#pragma omp single
        tp[4] = wall();
#pragma omp for schedule(static)
        for (int j = 0; j < (TREES ? 2 * (NB - NBC) * TREES : 0); j++) {   /* trees: one Muon matrix a tree (TN rows) */
            int b = NBC + j / (2 * TREES), m = (j / TREES) % 2, tr = j % TREES; size_t o = (size_t)tr * TN * DW;
            MuonJob job = {.p = (m ? TO[b] : TI[b]) + o, .g = (m ? gTO[b] : gTI[b]) + o, .m = (m ? MO[b] : MI[b]) + o, .q = NULL, .hit = NULL, .model = NULL, .field = 0, .rows = TN, .d = DW, .row = 0, .v = NULL};
            muon_apply(&job, h, scale);
        }
#pragma omp for schedule(static)
        for (int j = TREES ? 4 * NB : 4 * NBC; j < 4 * NB; j++) {
            int b = j / 4, m = (j / 2) % 2, half = j % 2;
            float *p = (m ? W2[b] : W1[b]) + half * 32 * DW, *g = (m ? gW2[b] : gW1[b]) + half * 32 * DW, *mo = (m ? M2[b] : M1[b]) + half * 32 * DW;
            MuonJob job = {.p = p, .g = g, .m = mo, .q = NULL, .hit = NULL, .model = NULL, .field = 0, .rows = 32, .d = DW, .row = 0, .v = NULL};
            muon_apply(&job, h, scale);
        }
#pragma omp single
        tp[5] = wall();
#pragma omp for schedule(static)
        for (size_t i = 0; i < n_tex / FCH; i++) if (thit[i]) { thit[i] = 0; size_t o = i * FCH; adam(tex + o, gtex + o, tm + o, tv + o, FCH, ht, scale, c1, c2); }
    }
    tp[6] = wall();
    prof_on = 0;
    for (int k = 0; k < 6; k++) tphase[k] += tp[k + 1] - tp[k];
    requantize();
    return loss / B;
}

/* Held-out error over white (train.py's metric), code-free; full frames, a frame per thread. */
static int n_eval_ws = 1;
static double evaluate(const Data *d, const int *ids, int n, WS **ews, int *const *nbf) {
    double err = 0;
#pragma omp parallel for num_threads(n_eval_ws) schedule(dynamic, 1) reduction(+:err)
    for (int i = 0; i < n; i++) {
        WS *w = ews[omp_get_thread_num()]; const uint8_t *src = frame_px(d, ids[i]);
        forward(w, src, d->W, d->H, d->W, d->pitch[ids[i]], nbf);
        double e = 0; int L = d->H * d->W;
        if (NBC) { err += line_grad(w, src, d->W, d->H, d->W, 0); continue; }
        for (int t = 0; t < L; t++) {
            const uint8_t *q = src + (size_t)t * 17 + 13; float a = w->p[t * 4 + 3], ta = q[3] / 255.f;
            for (int k = 0; k < 3; k++) e += fabsf((w->p[t * 4 + k] * a + 1 - a) - (q[k] / 255.f * ta + 1 - ta));
        }
        err += e / (3.0 * L);
    }
    return err / n;
}

static void preview(const Data *d, const int *ids, int n, WS *w, int *const *nbf, const char *path) {
    FILE *f = fopen(path, "wb"); int W3 = d->W * 3; fprintf(f, "P6 %d %d 255\n", W3, d->H * n);
    unsigned char *row = malloc(W3 * 3);
    for (int i = 0; i < n; i++) {
        const uint8_t *src = frame_px(d, ids[i]);
        forward(w, src, d->W, d->H, d->W, d->pitch[ids[i]], nbf);
        for (int y = 0; y < d->H; y++) {
            for (int x = 0; x < d->W; x++) {
                const uint8_t *q = src + ((size_t)y * d->W + x) * 17; const float *p = w->p + ((size_t)y * d->W + x) * 4;
                float ta = q[16] / 255.f;
                if (NBC) {   /* line mode: the drawing with its ink (as drawn), the ink, the target ink */
                    float k = w->pl[(size_t)y * d->W + x], kt = q[13] / 255.f, ks = ink_style(k);
                    for (int c = 0; c < 3; c++) {
                        row[x * 3 + c] = (unsigned char)fminf(255, fmaxf(0, 255 * ((p[c] * p[3] + 1 - p[3]) * (1 - ks))));
                        row[(d->W + x) * 3 + c] = (unsigned char)(255 * (1 - k));
                        row[(2 * d->W + x) * 3 + c] = (unsigned char)(255 * (1 - kt));
                    }
                    continue;
                }
                for (int k = 0; k < 3; k++) {
                    row[(x) * 3 + k] = q[k];
                    row[(d->W + x) * 3 + k] = (unsigned char)fminf(255, fmaxf(0, 255 * (p[k] * p[3] + 1 - p[3])));
                    row[(2 * d->W + x) * 3 + k] = (unsigned char)(255 * (q[13 + k] / 255.f * ta + 1 - ta));
                }
            }
            fwrite(row, 1, W3 * 3, f);
        }
    }
    free(row); fclose(f);
}

static int save_net(const char *path) {
    FILE *f = fopen(path, "wb"); if (!f) return 0;
    int hd[4] = {NB, DW, P_PARTS, KT};
    if (TREES) {   /* NTST: trees, levels; dense (with the node biases); per block the trees' w_in, w_out; texture */
        int tv[2] = {TREES, TV};
        fwrite("NTST", 1, 4, f); fwrite(hd, 4, 4, f); fwrite(DIL, 4, MAXB, f); fwrite(tv, 4, 2, f);
        fwrite(dense, 4, n_dense, f);
        for (int b = 0; b < NB; b++) { fwrite(TI[b], 4, (size_t)TREES * TN * DW, f); fwrite(TO[b], 4, (size_t)TREES * TN * DW, f); }
        fwrite(tex, 4, n_tex, f); fclose(f); return 1;
    }
    fwrite(NBC ? "NTSL" : "NTSW", 1, 4, f); fwrite(hd, 4, 4, f); fwrite(DIL, 4, MAXB, f);
    if (NBC) fwrite(&NBC, 4, 1, f);   /* NTSL: the colour head reads block NBC; the ink head (o_Wl) is in dense */
    fwrite(dense, 4, n_dense, f);
    for (int b = 0; b < NB; b++) { fwrite(W1[b], 4, DW * DW, f); fwrite(W2[b], 4, DW * DW, f); }
    fwrite(tex, 4, n_tex, f); fclose(f); return 1;
}

/* --tree from a dense model (NTSW): its parameters into the tree layout, its FFNs kept (the teachers). */
static int load_dense_into_trees(FILE *f) {
    int keep = TREES; TREES = 0; layout();
    size_t on = n_dense, otc[MAXB], ob1[MAXB], ob2[MAXB], owh = o_Wh;
    memcpy(otc, o_tc, sizeof otc); memcpy(ob1, o_b1, sizeof ob1); memcpy(ob2, o_b2, sizeof ob2);
    TREES = keep; layout();
    float *old = malloc(on * 4); int ok = fread(old, 4, on, f) == on;
    memcpy(dense, old, (size_t)(DW * CIN + DW) * 4);
    for (int b = 0; b < NB; b++) {
        memcpy(dense + o_tc[b], old + otc[b], tc_size(DW, KT) * 4);
        memcpy(dense + o_b1[b], old + ob1[b], DW * 4); memcpy(dense + o_b2[b], old + ob2[b], DW * 4);
    }
    memcpy(dense + o_Wh, old + owh, (4 * DW + 4) * 4);
    free(old);
    for (int b = 0; b < NB; b++) ok = ok && fread(W1[b], 4, DW * DW, f) == DW * DW && fread(W2[b], 4, DW * DW, f) == DW * DW;
    ok = ok && fread(tex, 4, n_tex, f) == n_tex;
    fclose(f); requantize(); return ok;
}

static int load_net(const char *path) {
    FILE *f = fopen(path, "rb"); char mg[4]; int hd[4], dil[MAXB], nbc = 0;
    if (!f || fread(mg, 1, 4, f) != 4 || (memcmp(mg, "NTSW", 4) && memcmp(mg, "NTSL", 4) && memcmp(mg, "NTST", 4)) || fread(hd, 4, 4, f) != 4 || fread(dil, 4, MAXB, f) != MAXB) return 0;
    if (hd[0] != NB || hd[1] != DW || hd[2] != P_PARTS || hd[3] != KT) { if (TREES || !memcmp(mg, "NTST", 4)) { fprintf(stderr, "%s: other shape\n", path); return 0; } }
    if (!memcmp(mg, "NTST", 4)) {
        int tv[2];
        if (fread(tv, 4, 2, f) != 2 || tv[0] != TREES || tv[1] != TV || memcmp(dil, DIL, sizeof(int) * NB)) { fprintf(stderr, "%s: other trees (use --tree %d,%d)\n", path, tv[0], tv[1]); return 0; }
        int ok = fread(dense, 4, n_dense, f) == n_dense;
        for (int b = 0; b < NB; b++) ok = ok && fread(TI[b], 4, (size_t)TREES * TN * DW, f) == (size_t)TREES * TN * DW && fread(TO[b], 4, (size_t)TREES * TN * DW, f) == (size_t)TREES * TN * DW;
        ok = ok && fread(tex, 4, n_tex, f) == n_tex;
        fclose(f); requantize(); return ok;
    }
    if (TREES) { if (memcmp(mg, "NTSW", 4) || memcmp(dil, DIL, sizeof(int) * NB)) { fprintf(stderr, "%s: --tree starts from a dense NTSW model\n", path); return 0; } return load_dense_into_trees(f); }
    if (!memcmp(mg, "NTSL", 4) && fread(&nbc, 4, 1, f) != 1) return 0;
    /* line mode from a plain model: it becomes the frozen trunk (its blocks, then its colour head) */
    int base = NBC && !nbc && hd[0] == NBC, nb = base ? NBC : NB;
    if (hd[0] != nb || nbc != (base ? 0 : NBC) || hd[1] != DW || hd[2] != P_PARTS || hd[3] != KT || memcmp(dil, DIL, sizeof(int) * nb)) { fprintf(stderr, "%s: other shape\n", path); return 0; }
    int ok;
    if (base) {
        size_t pre = o_tc[NBC];   /* the embedding and the trunk's blocks: the same offsets in both layouts */
        ok = fread(dense, 4, pre, f) == pre && fread(dense + o_Wh, 4, 4 * DW + 4, f) == 4 * DW + 4;
    } else ok = fread(dense, 4, n_dense, f) == n_dense;
    for (int b = 0; b < nb; b++) ok = ok && fread(W1[b], 4, DW * DW, f) == DW * DW && fread(W2[b], 4, DW * DW, f) == DW * DW;
    ok = ok && fread(tex, 4, n_tex, f) == n_tex;
    fclose(f); requantize(); return ok;
}
/* The texture as the viewer gets it (export_nts.py): 8 bits per channel between each level's channel min and max. */
static void tex_to_8bit(void) {
    for (int lev = 0; lev < 2; lev++) for (int c = 0; c < FCH; c++) {
        size_t S = lev ? COARSE : FINE, n = S * S; float lo = INFINITY, hi = -INFINITY;
        for (int p = 0; p < P_PARTS; p++) { const float *t = tex + (lev ? tex_coarse_off(p) : tex_fine_off(p)); for (size_t i = 0; i < n; i++) { lo = fminf(lo, t[i * FCH + c]); hi = fmaxf(hi, t[i * FCH + c]); } }
        float st = (hi - lo) / 255; if (st <= 0) st = 1;
        for (int p = 0; p < P_PARTS; p++) { float *t = tex + (lev ? tex_coarse_off(p) : tex_fine_off(p)); for (size_t i = 0; i < n; i++) t[i * FCH + c] = lo + nearbyintf((t[i * FCH + c] - lo) / st) * st; }
    }
}

/* ---------------- gradient check (continuous weights, as nanotss --check) ---------------- */
static uint64_t routes(const WS *w, int L) {   /* a hash of every node every token visited */
    uint64_t hsh = 1469598103934665603ULL;
    if (TREES) for (int b = 0; b < NB; b++) for (size_t i = 0; i < (size_t)L * TREES * TV; i++) hsh = (hsh ^ (uint64_t)w->tnode[b][i]) * 1099511628211ULL;
    return hsh;
}
static int run_check(void) {
    check_mode = 1; continuous_check = 1; tc_continuous = 1; NB = 3; P_PARTS = 2; tree_on = TREES > 0;
    net_alloc(1); net_init();
    for (int b = 0; b < NB; b++) {   /* nonzero gates and FFNs, so every path carries gradient */
        for (int i = 0; i < DW; i++) dense[o_tc[b] + (size_t)(KT + TC_GROUP) * DW + i] = (float)(randn() * 0.5);
        for (int i = 0; i < DW * DW; i++) W2[b][i] = (float)(randn() * 0.1);
        if (TREES) for (size_t i = 0; i < (size_t)TREES * TN * DW; i++) TO[b][i] = (float)(randn() * 0.1);
    }
    for (int i = 0; i < 4 * DW; i++) dense[o_Wh + i] = (float)(randn() * 0.3);
    requantize();
    int h = 5, w = 6, L = h * w;
    uint8_t *px = malloc((size_t)L * 17); uint64_t s = 12345;
    for (int t = 0; t < L; t++) {
        for (int k = 0; k < 17; k++) px[t * 17 + k] = (uint8_t)(urand(&s) * 256);
        px[t * 17 + 11] = (uint8_t)(urand(&s) * P_PARTS); px[t * 17 + 12] = t % 3 ? 255 : 0;
    }
    int *nb[MAXB]; for (int b = 0; b < NB; b++) nb[b] = neighbours(h, w, DIL[b]);
    WS *ws = ws_new(L, 1);
    /* the L1 loss's kinks make finite differences unreliable: check a smooth surrogate, sum of z . r */
    float *R = malloc(sizeof(float) * L * 4); for (int i = 0; i < L * 4; i++) R[i] = (float)randn();
    #define LOSS() ({ forward(ws, px, w, h, w, 10.f, nb); double acc = 0; for (int i = 0; i < L * 4; i++) acc += (double)ws->z[i] * R[i]; acc; })
    LOSS();
    uint64_t r0 = routes(ws, L); int flipped = 0;
    for (int i = 0; i < L * 4; i++) ws->z[i] = R[i];
    backward(ws, 0, h, w, nb);
    for (size_t i = 0; i < N_COARSE; i++) { gtex[tex_coarse_off(0) + i] += gcz_t[0][i]; gcz_t[0][i] = 0; }
    struct { const char *name; float *p; float *g; size_t n; } groups[] = {
        {"embed", dense + o_We, gd_t[0] + o_We, (size_t)DW * CIN},
        {"stencil0", dense + o_tc[0], gd_t[0] + o_tc[0], tc_size(DW, KT)},
        {"stencil2", dense + o_tc[2], gd_t[0] + o_tc[2], tc_size(DW, KT)},
        {"ffn W1[1]", W1[1], gW1_t[0][1], DW * DW}, {"ffn W2[0]", W2[0], gW2_t[0][0], DW * DW},
        {"bias b1[2]", dense + o_b1[2], gd_t[0] + o_b1[2], DW}, {"head", dense + o_Wh, gd_t[0] + o_Wh, 4 * DW},
        {"texture", tex, gtex, n_tex},
        {"tree w_in[1]", TREES ? TI[1] : NULL, TREES ? gTI_t[0][1] : NULL, (size_t)TREES * TN * DW},
        {"tree w_out[0]", TREES ? TO[0] : NULL, TREES ? gTO_t[0][0] : NULL, (size_t)TREES * TN * DW},
        {"tree bias[2]", TREES ? dense + o_tb[2] : NULL, TREES ? gd_t[0] + o_tb[2] : NULL, (size_t)TREES * TN},
    };
    int bad = 0; double worst_all = 0;
    for (size_t gi = 0; gi < sizeof(groups) / sizeof(groups[0]); gi++) {
        double worst = 0; int tried = 0;
        if (!groups[gi].p) continue;
        /* trees: only the rows some token visited have gradient (fixed paths: a nudge that changed a
         * route would make the difference meaningless); pick among those */
        int tree_group = TREES && gi >= 8;
        /* the texture: only the texels this crop samples have gradient; pick among those */
        size_t *nz = NULL, nnz = 0;
        if (groups[gi].p == tex || tree_group) { nz = malloc(sizeof(size_t) * 4096); for (size_t i = 0; i < groups[gi].n && nnz < 4096; i++) if (groups[gi].g[i] != 0) nz[nnz++] = i; }
        for (int k = 0; k < 400 && tried < 24; k++) {
            size_t i = nz ? (nnz ? nz[(size_t)(urand(&s) * nnz)] : 0) : (size_t)(urand(&s) * groups[gi].n);
            double an = groups[gi].g[i];
            if ((groups[gi].p == tex || tree_group) && an == 0) continue;   /* untouched texels, unvisited nodes */
            float keep = groups[gi].p[i], e = 1e-2f;
            groups[gi].p[i] = keep + e; requantize(); double lp = LOSS(); uint64_t rp = routes(ws, L);
            groups[gi].p[i] = keep - e; requantize(); double lm = LOSS(); uint64_t rm = routes(ws, L);
            groups[gi].p[i] = keep; requantize();
            if (TREES && (rp != r0 || rm != r0)) { flipped++; continue; }   /* a route changed: no derivative there */
            double num = (lp - lm) / (2 * e), err = fabs(num - an) / fmax(1, fmax(fabs(num), fabs(an)));
            if (err > worst) worst = err;
            if (err > 2e-2) bad++;
            tried++;
        }
        printf("  %-12s %2d coordinates, worst scaled error %.2e\n", groups[gi].name, tried, worst);
        if (worst > worst_all) worst_all = worst;
    }
    if (TREES) printf("  (%d nudges changed a route and were skipped)\n", flipped);
    printf("shader gradient check: worst %.2e, bad %d -> %s\n", worst_all, bad, bad ? "FAILED" : "PASSED");
    return bad ? 1 : 0;
}

int main(int argc, char **argv) {
    const char *data = NULL, *save = "model.ntsw", *prev = "preview.ppm", *load = NULL, *ref = NULL, *dump = NULL, *save_init = NULL;
    int tex8 = 0, eval_all = 0, lines = 0, distill = 0;
    int iters = 12000, B = 0, crop = 96, threads = 0, eval_every = 500, check = 0;
    double lr = 0.003, lr_tex = 0.01, seconds = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : "";
        if (!strcmp(a, "--check")) check = 1;
        else if (!strcmp(a, "--data")) { data = v; i++; }
        else if (!strcmp(a, "--save")) { save = v; i++; }
        else if (!strcmp(a, "--preview")) { prev = v; i++; }
        else if (!strcmp(a, "--iters")) { iters = atoi(v); i++; }
        else if (!strcmp(a, "--seconds")) { seconds = atof(v); i++; }
        else if (!strcmp(a, "--batch")) { B = atoi(v); i++; }
        else if (!strcmp(a, "--crop")) { crop = atoi(v); i++; }
        else if (!strcmp(a, "--threads")) { threads = atoi(v); i++; }
        else if (!strcmp(a, "--blocks")) { NB = atoi(v); i++; }
        else if (!strcmp(a, "--lines")) { lines = atoi(v); i++; }   /* an ink branch of K blocks on the (frozen) model --load gives */
        else if (!strcmp(a, "--tree")) { if (sscanf(v, "%d,%d", &TREES, &TV) != 2 || TREES < 1 || TV < 2 || TV > 5) { fprintf(stderr, "--tree T,V (V 2..5)\n"); return 2; } TN = (1 << TV) - 1; TAL = TV * (TV - 1) / 2; i++; }
        else if (!strcmp(a, "--distill")) distill = 1;           /* the trees learn the loaded dense model's FFNs */
        else if (!strcmp(a, "--tree-temp")) { tree_temp = atof(v); i++; }
        else if (!strcmp(a, "--lr")) { lr = atof(v); i++; }
        else if (!strcmp(a, "--lr-tex")) { lr_tex = atof(v); i++; }
        else if (!strcmp(a, "--eval-every")) { eval_every = atoi(v); i++; }
        else if (!strcmp(a, "--profile")) profile = 1;
        else if (!strcmp(a, "--engine-rows")) engine_rows = 1;
        else if (!strcmp(a, "--load")) { load = v; i++; }        /* start from a saved model */
        else if (!strcmp(a, "--tex8")) tex8 = 1;                  /* the viewer's 8-bit texture */
        else if (!strcmp(a, "--eval-all")) eval_all = 1;          /* every frame is held out (a pack_data.py --held file), all evaluated */
        else if (!strcmp(a, "--ref")) { ref = v; i++; }           /* first held-out frame's input and output, for infer.c */
        else if (!strcmp(a, "--dump-grads")) { dump = v; i++; }
        else if (!strcmp(a, "--save-init")) { save_init = v; i++; }   /* the model as training would start it (net_init, then --load), and stop */   /* step 1's batch: its summed raw gradients, for shader_torch.py */
        else { fprintf(stderr, "unknown %s\n", a); return 2; }
    }
    if (check) return run_check();
    if (lines) { NBC = NB; for (int k = 0; k < lines; k++) DIL[NB + k] = k % 2 ? 2 : 1; NB += lines; }   /* fine branch: dilations 1, 2, ... */
    if (!data || NB < 1 || NB > MAXB) { fprintf(stderr, "need --data (blocks 1..%d)\n", MAXB); return 2; }
    if (!threads) threads = omp_get_max_threads();
    if (!B) B = threads;
    Data d = load_data(data); P_PARTS = d.P;
    int *train_ids = malloc(4 * d.N), *held = malloc(4 * d.N), ntr = 0, nh = 0;
    for (int i = 0; i < d.N; i++) { if (i % 10 == 5 || eval_all) held[nh++] = i; else train_ids[ntr++] = i; }
    int neval = 0; int *eval_ids = malloc(4 * nh); for (int i = 0; i < nh; i += eval_all ? 1 : 4) eval_ids[neval++] = held[i];
    net_alloc(threads); net_init();
    if (distill && (!TREES || !load || lines)) { fprintf(stderr, "--distill needs --tree and --load of a dense model\n"); return 2; }
    if (TREES && lines) { fprintf(stderr, "--tree with --lines is not supported\n"); return 2; }
    if (load && !load_net(load)) { fprintf(stderr, "could not load %s\n", load); return 1; }
    tree_on = TREES && !distill;
    if (save_init) { printf(save_net(save_init) ? "initial model -> %s\n" : "could not save %s\n", save_init); return 0; }
    if (tex8) tex_to_8bit();
    int ch = crop < d.H ? crop : d.H, cw = crop < d.W ? crop : d.W;
    int *nb[MAXB], *nbf[MAXB];
    for (int b = 0; b < NB; b++) { nb[b] = neighbours(ch, cw, DIL[b]); nbf[b] = neighbours(d.H, d.W, DIL[b]); }
    WS **wss = malloc(threads * sizeof(WS *)), **ews = malloc(threads * sizeof(WS *));
    /* full-frame workspaces are ~0.6 GB each: at most 8 */
    n_eval_ws = threads < 8 ? threads : 8;
    for (int t = 0; t < threads; t++) wss[t] = ws_new(ch * cw, 1);
    for (int t = 0; t < n_eval_ws; t++) ews[t] = ws_new(d.H * d.W, 0);
    size_t ternary = TREES ? (size_t)NB * 2 * TREES * TN * DW : (size_t)NB * 2 * DW * DW;
    if (TREES) printf("trees: %d a block, %d levels (%d nodes each); a token visits %d of %d nodes (dense FFN: 64 hidden)%s\n",
                      TREES, TV, TN, TREES * TV, TREES * TN, distill ? "; distilling the dense FFNs" : "");
    printf("%d frames %dx%d (%d train, %d held out, %d evaluated), %d parts; %d blocks x %d wide, dilations",
           d.N, d.W, d.H, ntr, nh, neval, d.P, NB, DW);
    for (int b = 0; b < NB; b++) printf(" %d", DIL[b]);
    printf("; %zu dense + %zu ternary weights, texture %zu; %d threads, batch %d of %dx%d crops; %s kernels\n",
           n_dense, ternary, n_tex, threads, B, cw, ch, FAST ? "AVX-512 VNNI" : "scalar");
    fflush(stdout);
    Hyper h = {.lr = lr, .beta1 = .9, .beta2 = .98, .eps = 1e-8, .weight_decay = .01, .clip = 1};
    Hyper ht = {.lr = lr_tex, .beta1 = .9, .beta2 = .98, .eps = 1e-8, .weight_decay = 0, .clip = 1};
    double t0 = wall(), ema = 0;
    if (TREES) tree_on = 1;   /* what's evaluated, previewed and referenced is the tree model */
    if (ref) {   /* 'NTSR', H, W, pitch, the 17-byte pixels, then the straight RGBA output (float32) */
        int fi = held[0]; forward(ews[0], frame_px(&d, fi), d.W, d.H, d.W, d.pitch[fi], nbf);
        FILE *f = fopen(ref, "wb"); int hw[2] = {d.H, d.W};
        fwrite("NTSR", 1, 4, f); fwrite(hw, 4, 2, f); fwrite(d.pitch + fi, 4, 1, f);
        if (NBC) for (size_t t = 0; t < (size_t)d.H * d.W; t++) {   /* line mode: the ink over the colour, as the viewers draw it */
            float *q = ews[0]->p + t * 4, k = ink_style(ews[0]->pl[t]), al = q[3] + k * (1 - q[3]), fa = al > 0 ? q[3] * (1 - k) / al : 0;
            q[0] *= fa; q[1] *= fa; q[2] *= fa; q[3] = al;
        }
        fwrite(frame_px(&d, fi), 17, (size_t)d.H * d.W, f); fwrite(ews[0]->p, 4, (size_t)d.H * d.W * 4, f); fclose(f);
        printf("reference frame %d -> %s\n", fi, ref);
    }
    if (dump) {   /* 'NTSG', B, loss, then the gradients summed over step 1's crops (as train_step draws them), before scaling */
        tree_on = TREES && !distill; double loss = 0;
        for (int b = 0; b < B; b++) {
            uint64_t sd = 1 * 0x9E3779B97F4A7C15ULL + b * 7919 + 1;
            int fi = train_ids[(int)(urand(&sd) * ntr)];
            int cy = (int)(urand(&sd) * (d.H - ch + 1)), cx = (int)(urand(&sd) * (d.W - cw + 1));
            const uint8_t *src = frame_px(&d, fi) + ((size_t)cy * d.W + cx) * 17;
            forward(wss[0], src, d.W, ch, cw, d.pitch[fi], nb);
            if (distill) loss += distill_grad(wss[0], 0, ch * cw);
            else { loss += loss_grad(wss[0], src, d.W, ch, cw, 1); backward(wss[0], 0, ch, cw, nb); }
        }
        for (size_t i = 0; i < N_COARSE; i++) { gtex[tex_coarse_off(0) + i] += gcz_t[0][i]; gcz_t[0][i] = 0; }
        FILE *f = fopen(dump, "wb"); int hd[2] = {B, TREES}; fwrite("NTSG", 1, 4, f); fwrite(hd, 4, 2, f); fwrite(&loss, 8, 1, f);
        fwrite(gd_t[0], 4, n_dense, f);
        for (int b = 0; b < NB; b++) {
            if (TREES) { fwrite(gTI_t[0][b], 4, (size_t)TREES * TN * DW, f); fwrite(gTO_t[0][b], 4, (size_t)TREES * TN * DW, f); }
            else { fwrite(gW1_t[0][b], 4, DW * DW, f); fwrite(gW2_t[0][b], 4, DW * DW, f); }
        }
        int nh = 0; for (size_t i = 0; i < n_tex / FCH; i++) nh += thit[i];
        fwrite(&nh, 4, 1, f);
        for (size_t i = 0; i < n_tex / FCH; i++) if (thit[i]) { int k = (int)i; fwrite(&k, 4, 1, f); fwrite(gtex + i * FCH, 4, FCH, f); }
        fclose(f); printf("step 1 gradients (%d crops, loss %.6f, %d texels hit) -> %s\n", B, loss / B, nh, dump);
        return 0;
    }
    if (iters == 0) {   /* evaluation only: the error, and the preview of up to 6 held-out frames */
        printf("held-out L1 %.4f over %d frames\n", evaluate(&d, eval_ids, neval, ews, nbf), neval);
        int pick[6], np = nh < 6 ? nh : 6; for (int i = 0; i < np; i++) pick[i] = held[np > 1 ? (long)i * (nh - 1) / (np - 1) : 0];
        preview(&d, pick, np, ews[0], nbf, prev);
        return 0;
    }
    for (int it = 1; it <= iters; it++) {
        double el = wall() - t0, prog = seconds > 0 ? fmin(1, el / seconds) : (double)it / iters;
        double sched = fmin(1, it / 300.0) * (.1 + .9 * .5 * (1 + cos(M_PI * prog)));
        h.step = ht.step = it; h.lr = lr * sched; ht.lr = lr_tex * sched;
        tree_on = TREES && !distill;
        double loss = train_step(&d, train_ids, ntr, B, crop, nb, wss, &h, &ht, (uint64_t)it, distill);
        if (TREES) tree_on = 1;
        ema = it == 1 ? loss : .98 * ema + .02 * loss;
        int last = it == iters || (seconds > 0 && wall() - t0 >= seconds);
        if (it % eval_every == 0 || last) {
            double e = evaluate(&d, eval_ids, neval, ews, nbf);
            printf("it %d  loss %.4f  held-out L1 %.4f  %.0fs\n", it, ema, e, wall() - t0);
            if (!last) save_net(save);   /* a run cut short keeps its latest evaluation */
            if (profile) printf("  thread-0 ms/step: input %.1f embed %.1f stencil %.1f ffn-rows %.1f gelu %.1f | ffn-outer %.1f ffn-axpy %.1f stencil-bwd %.1f embed+tex-bwd %.1f\n",
                1e3 * tst[0] / it, 1e3 * tst[1] / it, 1e3 * tst[2] / it, 1e3 * tst[3] / it, 1e3 * tst[4] / it, 1e3 * tst[7] / it, 1e3 * tst[8] / it, 1e3 * tst[9] / it, 1e3 * tst[10] / it);
            if (profile) printf("  ms/step: forward+backward %.1f, reduce %.1f, texture norm %.1f, adam %.1f, muon %.1f, texture adam %.1f\n",
                1e3 * tphase[0] / it, 1e3 * tphase[1] / it, 1e3 * tphase[2] / it, 1e3 * tphase[3] / it, 1e3 * tphase[4] / it, 1e3 * tphase[5] / it);
            fflush(stdout);
        }
        if (last) break;
    }
    int pick[6]; for (int i = 0; i < 6; i++) pick[i] = held[(long)i * (nh - 1) / 5];
    preview(&d, pick, 6, ews[0], nbf, prev);
    printf(save_net(save) ? "saved %s, preview %s\n" : "could not save %s\n", save, prev);
    return 0;
}
