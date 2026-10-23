/* The nanotss shader's forward pass for the browser: freestanding C compiled to WebAssembly SIMD128
 * (coverage/live/nts.js runs it on the CPU, one band of rows per Web Worker).
 *
 *   clang --target=wasm32 -O3 -msimd128 -mbulk-memory -nostdlib -Wl,--no-entry -Wl,--export-dynamic \
 *         -Wl,--initial-memory=33554432 -o ../../live/nts/infer.wasm infer.c
 *   (the same with -mrelaxed-simd -o infer-relaxed.wasm: relaxed swizzle; and both again with -DNO_LUT,
 *   -o infer-dot.wasm and infer-relaxed-dot.wasm: the dot-product kernel, 8-bit dots with relaxed SIMD.
 *   nts.js times both kernels on the device and keeps the faster: on x86 the lookup tables win by a
 *   little, while an ARM core whose browser turns the relaxed dot into sdot may favour the dots.)
 *
 * It computes what shader_nts.c's forward() computes, from export_nts.py's net.bin: the input decode
 * and neural texture, the embedding, per block the engine's ternary stencil mixer (tc_one) and the
 * two ternary FFNs on int8 activation codes (quant8 + rows64_dot8: the same integer sums, here by
 * table lookup, T-MAC style, or with -DNO_LUT dot products) with the same GELU, then the head. A
 * band computes each block only on the rows later blocks still read, so overlapping bands give the
 * whole-frame result.
 */
#include <stddef.h>
#include <stdint.h>
#include <wasm_simd128.h>

#define EXPORT __attribute__((visibility("default")))
#define DW 64
#define KT 9
#define CIN 18
#define FCH 8
#define FINE 512
#define COARSE 64
#define TCG 4
#define MAXB 16

/* ---- memory: a bump allocator over the module's linear memory ---- */
extern unsigned char __heap_base;
static uintptr_t heap_top;
EXPORT void *nts_alloc(int n) {
    if (!heap_top) heap_top = (uintptr_t)&__heap_base;
    uintptr_t p = (heap_top + 63) & ~(uintptr_t)63, end = p + (uintptr_t)n;
    uintptr_t have = (uintptr_t)__builtin_wasm_memory_size(0) * 65536;
    if (end > have && __builtin_wasm_memory_grow(0, (end - have + 65535) / 65536) < 0) return 0;
    heap_top = end;
    return (void *)p;
}
static void copy(void *d, const void *s, int n) { uint8_t *a = d; const uint8_t *b = s; while (n--) *a++ = *b++; }

/* ---- the network ---- */
typedef struct {
    float q[KT + TCG][DW];          /* stencil codes {-1, 0, 1} as floats */
    float gate[DW], b1[DW], b2[DW], s1[DW], s2[DW];
    v128_t w1[32 * 16], w2[32 * 16];  /* [input group][4 outputs]: 4 outputs x 2 inputs as i16 (relaxed: x 4 inputs as i8) */
} Block;
static int NB, NBC, P, DIL[MAXB];   /* NBC: the block the colour head reads (NB without an ink branch) */
static float We[DW * CIN], WeT[CIN * DW], be[DW], Wh[4 * DW], bh[4], Wl[DW], bl, lo[2][FCH], st[2][FCH];
static Block *blk;
static const uint8_t *coarse, *fine;

#ifndef NO_LUT
#define LUT          /* -DNO_LUT: the dot-product kernel instead (same sums) */
#endif
#if defined(LUT)
#define NG 22       /* groups of 3 inputs (the last padded) */
typedef int16_t code_t;
#elif defined(__wasm_relaxed_simd__)
#define GROUPS 16   /* inputs per dot product: 4 */
typedef int8_t code_t;
#else
#define GROUPS 32   /* 2 */
typedef int16_t code_t;
#endif
#ifdef LUT
/* Lookup-table form of a ternary row block (T-MAC style): per group of 3 inputs and 8 outputs, the
 * byte-shuffle indices that pick each output's subset sum of the +1 inputs, then of the -1 inputs,
 * from the group's table of 8 int16 subset sums. */
static void pack_rows(const int8_t *c, v128_t *w) {
    for (int g = 0; g < NG; g++) for (int q = 0; q < 8; q++) for (int s = 0; s < 2; s++) {
        uint8_t v[16];
        for (int o = 0; o < 8; o++) {
            int idx = 0;
            for (int j = 0; j < 3; j++) { int i = 3 * g + j; if (i < DW && c[(8 * q + o) * DW + i] == (s ? -1 : 1)) idx |= 1 << j; }
            v[2 * o] = (uint8_t)(2 * idx); v[2 * o + 1] = (uint8_t)(2 * idx + 1);
        }
        w[(g * 8 + q) * 2 + s] = wasm_v128_load(v);
    }
}
#else
static void pack_rows(const int8_t *c, v128_t *w) {
    const int n = DW / GROUPS;
    for (int p = 0; p < GROUPS; p++) for (int q = 0; q < 16; q++) {
        code_t v[16 / sizeof(code_t)];
        for (int o = 0; o < 4; o++) for (int j = 0; j < n; j++) v[n * o + j] = c[(4 * q + o) * DW + n * p + j] + (GROUPS == 16);
        w[p * 16 + q] = wasm_v128_load(v);
    }
}
#endif
/* net: export_nts.py's net.bin, which must stay alive (the coarse texture is read in place). */
EXPORT int nts_load(const uint8_t *net, int len) {
    const uint8_t *p = net;
    /* NTSV: colour only; NTS2: also NBC, and the ink head after the colour head */
    if (len < 12 || p[0] != 'N' || p[1] != 'T' || p[2] != 'S' || (p[3] != 'V' && p[3] != '2')) return 0;
    int lines = p[3] == '2', hd[3]; copy(hd, p + 4, lines ? 12 : 8); NB = hd[0]; P = hd[1]; NBC = lines ? hd[2] : NB; p += lines ? 16 : 12;
    if (NB < 1 || NB > MAXB || NBC < 1 || NBC > NB) return 0;
    copy(DIL, p, 4 * NB); p += 4 * NB;
    copy(We, p, sizeof We); p += sizeof We;
    copy(be, p, sizeof be); p += sizeof be;
    for (int j = 0; j < DW; j++) for (int i = 0; i < CIN; i++) WeT[i * DW + j] = We[j * CIN + i];
    blk = nts_alloc(NB * (int)sizeof(Block));
    for (int b = 0; b < NB; b++) {
        Block *k = blk + b;
        for (int r = 0; r < KT + TCG; r++) for (int i = 0; i < DW; i++) k->q[r][i] = (float)(int8_t)p[r * DW + i];
        p += (KT + TCG) * DW;
        copy(k->gate, p, 4 * DW); p += 4 * DW;
        copy(k->b1, p, 4 * DW); p += 4 * DW;
        copy(k->b2, p, 4 * DW); p += 4 * DW;
        copy(k->s1, p, 4 * DW); p += 4 * DW;
        copy(k->s2, p, 4 * DW); p += 4 * DW;
        pack_rows((const int8_t *)p, k->w1); p += DW * DW;
        pack_rows((const int8_t *)p, k->w2); p += DW * DW;
    }
    copy(Wh, p, sizeof Wh); p += sizeof Wh;
    copy(bh, p, sizeof bh); p += sizeof bh;
    if (NBC < NB) { copy(Wl, p, sizeof Wl); p += sizeof Wl; copy(&bl, p, 4); p += 4; }
    copy(lo, p, sizeof lo); p += sizeof lo;
    copy(st, p, sizeof st); p += sizeof st;
    coarse = p; p += (size_t)P * COARSE * COARSE * FCH;
    return p - net == len;
}
EXPORT void nts_set_fine(const uint8_t *f) { fine = f; }
EXPORT int nts_parts(void) { return P; }
EXPORT int nts_reach(void) { int r = 0; for (int b = 0; b < NB; b++) r += DIL[b]; return r; }

/* ---- inputs: the five G-buffer tiles to 18 floats a pixel ---- */
/* grid_sample, bilinear, align_corners=False, zero padding, on 8-bit texels dequantised per channel. */
static void tex_one(const uint8_t *g, int S, int lev, float u, float v, float *out) {
    float px = u * S - 0.5f, py = v * S - 0.5f;
    float fx0 = __builtin_floorf(px), fy0 = __builtin_floorf(py);
    int x0 = (int)fx0, y0 = (int)fy0; float fx = px - fx0, fy = py - fy0;
    for (int j = 0; j < 2; j++) for (int i = 0; i < 2; i++) {
        int x = x0 + i, y = y0 + j;
        if (x < 0 || y < 0 || x >= S || y >= S) continue;
        float w = (i ? fx : 1 - fx) * (j ? fy : 1 - fy);
        const uint8_t *t = g + ((uint32_t)y * S + x) * FCH;
        for (int c = 0; c < FCH; c++) out[c] += w * (lo[lev][c] + t[c] * st[lev][c]);
    }
}
/* tiles: colour, normal, depth, uv-a, uv-b, each w x h RGBA with rows bottom-up (readPixels).
 * in: w x h x 18 floats, rows top-down: shader_nts.c's decode() plus the texture features. */
EXPORT void nts_prepare(const uint8_t *tiles, int w, int h, float sinp, float cosp, float *in) {
    uint32_t plane = (uint32_t)w * h * 4;
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
        uint32_t s = ((uint32_t)(h - 1 - y) * w + x) * 4;
        const uint8_t *col = tiles + s, *nrm = col + plane, *dep = nrm + plane, *ua = dep + plane, *ub = ua + plane;
        float *f = in + ((uint32_t)y * w + x) * CIN;
        for (int c = 0; c < 3; c++) f[c] = col[c] / 255.f;
        for (int c = 0; c < 3; c++) f[3 + c] = nrm[c] / 255.f * 2 - 1;
        f[6] = dep[0] / 255.f;
        int mask = ub[2] > 127;
        f[7] = mask ? 1.f : 0.f;
        for (int c = 0; c < FCH; c++) f[8 + c] = 0;
        if (mask && fine) {
            float u = (ua[0] + ua[1] / 255.f) / 255.f, v = (ua[2] + ub[0] / 255.f) / 255.f;
            int part = ub[1] < P ? ub[1] : 0;
            tex_one(fine + (size_t)part * FINE * FINE * FCH, FINE, 0, u, v, f + 8);
            tex_one(coarse + (size_t)part * COARSE * COARSE * FCH, COARSE, 1, u, v, f + 8);
        }
        f[16] = sinp; f[17] = cosp;
    }
}

/* ---- maths (shader_nts.c's float GELU: Abramowitz-Stegun erf, one exp) ---- */
static inline v128_t splat(float x) { return wasm_f32x4_splat(x); }
static inline v128_t madd(v128_t a, v128_t b, v128_t c) { return wasm_f32x4_add(wasm_f32x4_mul(a, b), c); }
static inline v128_t exp4(v128_t x) {
    x = wasm_f32x4_max(x, splat(-87.f));
    v128_t n = wasm_f32x4_nearest(wasm_f32x4_mul(x, splat(1.44269504f)));
    v128_t r = wasm_f32x4_sub(x, wasm_f32x4_mul(n, splat(0.693359375f)));
    r = wasm_f32x4_sub(r, wasm_f32x4_mul(n, splat(-2.12194440e-4f)));
    v128_t p = splat(1.9875691500e-4f);
    p = madd(p, r, splat(1.3981999507e-3f));
    p = madd(p, r, splat(8.3334519073e-3f));
    p = madd(p, r, splat(4.1665795894e-2f));
    p = madd(p, r, splat(1.6666665459e-1f));
    p = madd(p, r, splat(5.0000001201e-1f));
    p = madd(p, wasm_f32x4_mul(r, r), wasm_f32x4_add(r, splat(1.f)));
    v128_t e = wasm_i32x4_shl(wasm_i32x4_add(wasm_i32x4_trunc_sat_f32x4(n), wasm_i32x4_splat(127)), 23);
    return wasm_f32x4_mul(p, e);
}
static inline v128_t gelu4(v128_t v) {
    v128_t one = splat(1.f), z = wasm_f32x4_abs(wasm_f32x4_mul(v, splat(.70710678f)));
    v128_t e = exp4(wasm_f32x4_mul(wasm_f32x4_mul(splat(-.5f), v), v));
    v128_t t = wasm_f32x4_div(one, madd(z, splat(.3275911f), one));
    v128_t p = splat(1.061405429f);
    p = madd(p, t, splat(-1.453152027f));
    p = madd(p, t, splat(1.421413741f));
    p = madd(p, t, splat(-.284496736f));
    p = madd(p, t, splat(.254829592f));
    v128_t er = wasm_f32x4_sub(one, wasm_f32x4_mul(wasm_f32x4_mul(p, t), e));
    er = wasm_v128_bitselect(wasm_f32x4_neg(er), er, wasm_f32x4_lt(v, wasm_f32x4_splat(0)));
    return wasm_f32x4_mul(v, wasm_f32x4_mul(splat(.5f), wasm_f32x4_add(one, er)));
}
static float exp1(float x) {   /* scalar exp for the head's sigmoid */
    float v[4] = {x < 88.f ? x : 88.f}; wasm_v128_store(v, exp4(wasm_v128_load(v))); return v[0];
}

/* quant8 (round to nearest even, clamp to int8) into 64 codes; returns the code's scale. */
static inline float quant(const v128_t *x, code_t *code, int *sum) {
    v128_t m = wasm_f32x4_abs(x[0]);
    for (int i = 1; i < 16; i++) m = wasm_f32x4_max(m, wasm_f32x4_abs(x[i]));
    float mx = wasm_f32x4_extract_lane(m, 0);
    for (int l = 1; l < 4; l++) { float v = l == 1 ? wasm_f32x4_extract_lane(m, 1) : l == 2 ? wasm_f32x4_extract_lane(m, 2) : wasm_f32x4_extract_lane(m, 3); if (v > mx) mx = v; }
    if (mx < 1e-5f) mx = 1e-5f;
    float scale = 127 / mx; v128_t S = splat(scale);
    v128_t h[8];
    for (int i = 0; i < 16; i += 2) {
        v128_t a = wasm_f32x4_min(wasm_f32x4_max(wasm_f32x4_nearest(wasm_f32x4_mul(x[i], S)), splat(-128.f)), splat(127.f));
        v128_t b = wasm_f32x4_min(wasm_f32x4_max(wasm_f32x4_nearest(wasm_f32x4_mul(x[i + 1], S)), splat(-128.f)), splat(127.f));
        h[i / 2] = wasm_i16x8_narrow_i32x4(wasm_i32x4_trunc_sat_f32x4(a), wasm_i32x4_trunc_sat_f32x4(b));
    }
#if defined(__wasm_relaxed_simd__) && !defined(LUT)
    for (int i = 0; i < 4; i++) wasm_v128_store(code + 16 * i, wasm_i8x16_narrow_i16x8(h[2 * i], h[2 * i + 1]));
    v128_t a = wasm_i32x4_extadd_pairwise_i16x8(h[0]);
    for (int i = 1; i < 8; i++) a = wasm_i32x4_add(a, wasm_i32x4_extadd_pairwise_i16x8(h[i]));
    *sum = wasm_i32x4_extract_lane(a, 0) + wasm_i32x4_extract_lane(a, 1) + wasm_i32x4_extract_lane(a, 2) + wasm_i32x4_extract_lane(a, 3);
#else
    *sum = 0;
    for (int i = 0; i < 8; i++) wasm_v128_store(code + 8 * i, h[i]);
#endif
    return 1 / scale;
}
/* out[r] = sum_i code[i] * w[r][i] * s[r] * xs (+ add[r]): rows64_dot8's integer sum, then its scaling. */
/* Two passes of 8 accumulators: 16 would not stay in the 16 vector registers V8 has on x86-64.
 * Relaxed: the weights are stored + 1 (0..2: x86 runs the 7-bit operand as unsigned), so each sum
 * starts at -sum(codes), the engine's 0x80-offset correction the other way round. */
#ifdef LUT
/* The same integer sums by table lookup: no multiplies. Per group, its 8 subset sums of 3 codes as
 * one int16 vector; per 8 outputs, two byte shuffles (the +1 and the -1 inputs) and an add and a
 * subtract cover 24 multiply-adds. Sums stay within int16 (64 x 128). code: 66 entries, the last 2 zero. */
#ifdef __wasm_relaxed_simd__
#define SWZ wasm_i8x16_relaxed_swizzle   /* indices are always in range: the same result, one instruction on x86 */
#else
#define SWZ wasm_i8x16_swizzle
#endif
static inline void rows(const v128_t *w, const code_t *code, int sum, float xs, const float *s, const float *add, v128_t *out) {
    (void)sum;
    const v128_t M0 = wasm_i16x8_make(0, -1, 0, -1, 0, -1, 0, -1), M1 = wasm_i16x8_make(0, 0, -1, -1, 0, 0, -1, -1),
                 M2 = wasm_i16x8_make(0, 0, 0, 0, -1, -1, -1, -1);
    v128_t acc[8];
    for (int q = 0; q < 8; q++) acc[q] = wasm_i16x8_splat(0);
    for (int g = 0; g < NG; g++) {
        v128_t T = wasm_i16x8_add(wasm_i16x8_add(wasm_v128_and(wasm_i16x8_splat(code[3 * g]), M0),
                                                 wasm_v128_and(wasm_i16x8_splat(code[3 * g + 1]), M1)),
                                  wasm_v128_and(wasm_i16x8_splat(code[3 * g + 2]), M2));
        const v128_t *wg = w + g * 16;
        for (int q = 0; q < 8; q++)
            acc[q] = wasm_i16x8_sub(wasm_i16x8_add(acc[q], SWZ(T, wg[2 * q])), SWZ(T, wg[2 * q + 1]));
    }
    v128_t X = splat(xs);
    for (int q = 0; q < 8; q++) for (int h = 0; h < 2; h++) {
        v128_t a = h ? wasm_i32x4_extend_high_i16x8(acc[q]) : wasm_i32x4_extend_low_i16x8(acc[q]);
        int o = 8 * q + 4 * h;
        out[2 * q + h] = wasm_f32x4_add(wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(a), wasm_v128_load(s + o)), X), wasm_v128_load(add + o));
    }
}
#else
static inline void rows(const v128_t *w, const code_t *code, int sum, float xs, const float *s, const float *add, v128_t *out) {
    const int32_t *grp = (const int32_t *)code;
    v128_t X = splat(xs);
    for (int half = 0; half < 16; half += 8) {
        v128_t acc[8];
        for (int q = 0; q < 8; q++) acc[q] = wasm_i32x4_splat(-sum);
        for (int p = 0; p < GROUPS; p++) {
            v128_t a = wasm_i32x4_splat(grp[p]);
            const v128_t *wp = w + p * 16 + half;
#ifdef __wasm_relaxed_simd__
            for (int q = 0; q < 8; q++) acc[q] = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(a, wp[q], acc[q]);
#else
            for (int q = 0; q < 8; q++) acc[q] = wasm_i32x4_add(acc[q], wasm_i32x4_dot_i16x8(a, wp[q]));
#endif
        }
        for (int q = 0; q < 8; q++)
            out[half + q] = wasm_f32x4_add(wasm_f32x4_mul(wasm_f32x4_mul(wasm_f32x4_convert_i32x4(acc[q]), wasm_v128_load(s + 4 * (half + q))), X), wasm_v128_load(add + 4 * (half + q)));
    }
}
#endif

/* One token of a block: src, its 3 x 3 dilated neighbours' block inputs (0: outside, zero padding). */
static void token(const Block *k, const float *const *src, const float *xc, float *yo) {
    const v128_t invk = splat(1.f / 3.f), half = splat(.5f);   /* tc_code_new: 1/sqrt(K), K = 9 */
    v128_t t[16], hb[16], g[16];
    for (int i = 0; i < 16; i++) {
        v128_t z = wasm_f32x4_splat(0);
        for (int q = 0; q < KT; q++) if (src[q]) z = madd(wasm_v128_load(&k->q[q][4 * i]), wasm_v128_load(src[q] + 4 * i), z);
        v128_t dw = wasm_f32x4_mul(z, invk), m = wasm_f32x4_splat(0);
        m = madd(wasm_v128_load(&k->q[KT + 0][4 * i]), wasm_i32x4_shuffle(dw, dw, 0, 0, 0, 0), m);
        m = madd(wasm_v128_load(&k->q[KT + 1][4 * i]), wasm_i32x4_shuffle(dw, dw, 1, 1, 1, 1), m);
        m = madd(wasm_v128_load(&k->q[KT + 2][4 * i]), wasm_i32x4_shuffle(dw, dw, 2, 2, 2, 2), m);
        m = madd(wasm_v128_load(&k->q[KT + 3][4 * i]), wasm_i32x4_shuffle(dw, dw, 3, 3, 3, 3), m);
        t[i] = madd(wasm_v128_load(k->gate + 4 * i), wasm_f32x4_mul(m, half), wasm_v128_load(xc + 4 * i));
    }
    code_t code[DW + 8] __attribute__((aligned(16))) = {0};   /* past DW: the lookup kernel's zero padding */
    int sum;
    float xs = quant(t, code, &sum);
    rows(k->w1, code, sum, xs, k->s1, k->b1, hb);
    for (int i = 0; i < 16; i++) g[i] = gelu4(hb[i]);
    xs = quant(g, code, &sum);
    rows(k->w2, code, sum, xs, k->s2, k->b2, hb);
    for (int i = 0; i < 16; i++) wasm_v128_store(yo + 4 * i, wasm_f32x4_add(hb[i], t[i]));
}
static void embed(const float *f, float *x) {
    for (int j = 0; j < DW; j += 4) {
        v128_t s = wasm_f32x4_splat(0);
        for (int i = 0; i < CIN; i++) s = madd(wasm_v128_load(WeT + i * DW + j), splat(f[i]), s);
        wasm_v128_store(x + j, wasm_f32x4_add(wasm_v128_load(be + j), s));
    }
}
static float ink(const float *v) {   /* the ink head: opacity of black line art */
    v128_t s = wasm_f32x4_splat(0);
    for (int i = 0; i < DW; i += 4) s = madd(wasm_v128_load(Wl + i), wasm_v128_load(v + i), s);
    float z = bl + (wasm_f32x4_extract_lane(s, 0) + wasm_f32x4_extract_lane(s, 1)) + (wasm_f32x4_extract_lane(s, 2) + wasm_f32x4_extract_lane(s, 3));
    /* drawn crisp: a smoothstep from 0.4 to 0.7 (shader_nts.c ink_style) */
    float k = 1 / (1 + exp1(-z)), t = (k - .4f) / .3f; t = t < 0 ? 0 : t > 1 ? 1 : t;
    return t * t * (3 - 2 * t);
}
static void head(const float *v, float *p) {
    for (int k = 0; k < 4; k++) {
        v128_t s = wasm_f32x4_splat(0);
        for (int i = 0; i < DW; i += 4) s = madd(wasm_v128_load(Wh + k * DW + i), wasm_v128_load(v + i), s);
        float z = bh[k] + (wasm_f32x4_extract_lane(s, 0) + wasm_f32x4_extract_lane(s, 1)) + (wasm_f32x4_extract_lane(s, 2) + wasm_f32x4_extract_lane(s, 3));
        p[k] = 1 / (1 + exp1(-z));
    }
}

/* Background skipping. Every token whose receptive field so far (RF[b] = DIL[0] + .. + DIL[b] pixels)
 * holds only background has one value per block, bg[b + 1] (bg[0]: the embedded background input).
 * And block b's output only reaches outputs within S[b] = DIL[b + 1] + .. pixels. So block b runs on
 * the tokens within min(RF[b], S[b] + MARGIN) of a pixel that isn't background, reads bg[b] for the
 * rest, and outputs further than MARGIN from her get the background's own output: the drawing is the
 * whole frame's wherever she is (MARGIN covers what she casts on the background), at a fraction of
 * the work. Outside the region counts as background too (the frame goes on), not as zero padding. */
#define MARGIN 6
static float bg[MAXB + 1][DW] __attribute__((aligned(16)));

/* in: a w x h region's inputs (nts_prepare). Writes straight RGBA for rows [r0, r1) to out (bytes) or
 * outf (floats): the colour head on block NBC's output, then (with an ink branch) black ink over it. xa, xb: w x h x 64 floats each; dist: w x h bytes. sparse = 0: every token, with zero
 * padding outside the region, exactly shader_nts.c's forward() on the region. */
static void run(const float *in, int w, int h, int r0, int r1, float *xa, float *xb, uint8_t *dist, int sparse, uint8_t *out, float *outf) {
    int S[MAXB + 1], RF[MAXB], need[MAXB]; S[NB] = 0;
    for (int b = NB - 1; b >= 0; b--) S[b] = S[b + 1] + DIL[b];
    for (int b = 0; b < NB; b++) {
        RF[b] = (b ? RF[b - 1] : 0) + DIL[b];
        need[b] = sparse ? (RF[b] < S[b + 1] + MARGIN ? RF[b] : S[b + 1] + MARGIN) : 255;
    }
    #define LO(s) (r0 - (s) > 0 ? r0 - (s) : 0)
    #define HI(s) (r1 + (s) < h ? r1 + (s) : h)
    #define AT(a, r, c) ((a) + ((uint32_t)(r) * w + (c)) * DW)
    /* Chebyshev distance (capped at 255) to the nearest pixel whose inputs aren't the background's,
     * which is pixel (0, 0): the region always starts outside her. Two raster passes over the 8
     * neighbours give the chessboard distance exactly. */
    const float *bgin = in;
    if (sparse) {
        for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) {
            const float *f = in + ((uint32_t)r * w + c) * CIN; int same = 1;
            for (int i = 0; i < 16; i++) same &= f[i] == bgin[i];
            dist[(uint32_t)r * w + c] = same ? 255 : 0;
        }
        #define D(r, c) dist[(uint32_t)(r) * w + (c)]
        #define RELAX(r, c, rr, cc) do { int v_ = D(rr, cc) + 1; if (v_ < D(r, c)) D(r, c) = (uint8_t)v_; } while (0)
        for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) {
            if (c) RELAX(r, c, r, c - 1);
            if (r) { RELAX(r, c, r - 1, c); if (c) RELAX(r, c, r - 1, c - 1); if (c + 1 < w) RELAX(r, c, r - 1, c + 1); }
        }
        for (int r = h - 1; r >= 0; r--) for (int c = w - 1; c >= 0; c--) {
            if (c + 1 < w) RELAX(r, c, r, c + 1);
            if (r + 1 < h) { RELAX(r, c, r + 1, c); if (c + 1 < w) RELAX(r, c, r + 1, c + 1); if (c) RELAX(r, c, r + 1, c - 1); }
        }
        #undef RELAX
        #undef D
        /* the background's value after each block */
        embed(bgin, bg[0]);
        for (int b = 0; b < NB; b++) { const float *src[KT]; for (int q = 0; q < KT; q++) src[q] = bg[b]; token(blk + b, src, bg[b], bg[b + 1]); }
    }
    #define ACTIVE(r, c, n) (!sparse || dist[(uint32_t)(r) * w + (c)] <= (n))
    for (int r = LO(S[0]); r < HI(S[0]); r++) for (int c = 0; c < w; c++)
        if (ACTIVE(r, c, 0)) embed(in + ((uint32_t)r * w + c) * CIN, AT(xa, r, c));
    float *x = xa, *y = xb;
    for (int b = 0; b < NB; b++) {
        int dil = DIL[b], prev = b ? need[b - 1] : 0;
        for (int r = LO(S[b + 1]); r < HI(S[b + 1]); r++) for (int c = 0; c < w; c++) {
            if (!ACTIVE(r, c, need[b])) continue;
            const float *src[KT];
            for (int dy = -1, t = 0; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++, t++) {
                int rr = r + dy * dil, cc = c + dx * dil, in_region = rr >= 0 && rr < h && cc >= 0 && cc < w;
                src[t] = in_region && ACTIVE(rr, cc, prev) ? AT(x, rr, cc) : sparse ? bg[b] : 0;
            }
            token(blk + b, src, ACTIVE(r, c, prev) ? AT(x, r, c) : bg[b], AT(y, r, c));
        }
        float *t = x; x = y; y = t;
        if (b == NBC - 1) {   /* the colour, before the ink branch's blocks overwrite this buffer */
            float pbg[4]; head(bg[NBC], pbg);
            for (int r = r0; r < r1; r++) for (int c = 0; c < w; c++) {
                float p[4];
                if (ACTIVE(r, c, MARGIN)) head(AT(x, r, c), p); else for (int k = 0; k < 4; k++) p[k] = pbg[k];
                uint32_t o = ((uint32_t)(r - r0) * w + c) * 4;
                for (int k = 0; k < 4; k++) {
                    if (outf) outf[o + k] = p[k];
                    else { float v = p[k] * 255 + .5f; out[o + k] = v < 0 ? 0 : v > 255 ? 255 : (uint8_t)v; }
                }
            }
        }
    }
    if (NBC < NB) {   /* black line art over the colour: alpha a + k (1 - a), colour scaled by (1 - k) */
        float kbg = ink(bg[NB]);
        for (int r = r0; r < r1; r++) for (int c = 0; c < w; c++) {
            float k = ACTIVE(r, c, MARGIN) ? ink(AT(x, r, c)) : kbg, p[4];
            uint32_t o = ((uint32_t)(r - r0) * w + c) * 4;
            for (int i = 0; i < 4; i++) p[i] = outf ? outf[o + i] : out[o + i] / 255.f;
            float a = p[3] + k * (1 - p[3]), f = a > 0 ? p[3] * (1 - k) / a : 0;
            p[0] *= f; p[1] *= f; p[2] *= f; p[3] = a;
            for (int i = 0; i < 4; i++) {
                if (outf) outf[o + i] = p[i];
                else { float v = p[i] * 255 + .5f; out[o + i] = v < 0 ? 0 : v > 255 ? 255 : (uint8_t)v; }
            }
        }
    }
    #undef LO
    #undef HI
    #undef AT
    #undef ACTIVE
}
/* For picking a kernel on the device (nts.js): n tokens through the blocks, on a made-up input. */
EXPORT float nts_bench(int n) {
    static float x[DW] __attribute__((aligned(16))), y[DW] __attribute__((aligned(16)));
    for (int i = 0; i < DW; i++) x[i] = (float)((i * 37) % 23 - 11) * 0.13f;
    const float *src[KT]; for (int q = 0; q < KT; q++) src[q] = x;
    for (int i = 0; i < n; i++) { token(blk + i % NB, src, x, y); x[i % DW] += y[i % DW] * 1e-3f; }
    return y[0];
}
EXPORT void nts_run(const float *in, int w, int h, int r0, int r1, float *xa, float *xb, uint8_t *dist, uint8_t *out) { run(in, w, h, r0, r1, xa, xb, dist, 1, out, 0); }
/* For checks: the output as floats (w x (r1 - r0) x 4), background skipping or not. */
EXPORT void nts_run_f(const float *in, int w, int h, int r0, int r1, float *xa, float *xb, uint8_t *dist, int sparse, float *out) { run(in, w, h, r0, r1, xa, xb, dist, sparse, 0, out); }
