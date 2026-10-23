/*
 * nanotss_blt.c -- standalone protected plastic / FFF / SSM / FFF stack.
 * Single-file C implementation: training, streaming inference, checkpoints,
 * neural-entropy BLT byte frontend, and ternary convolution mixer included.
 * No companion source or external ML framework is required.
 *
 * Build:
 *   cc -O3 -march=native -fopenmp nanotss_blt.c -lm -o nanotss_blt
 * Check:
 *   ./nanotss_blt --check
 * Train on TinyShakespeare (save the corpus as input.txt). Default architecture: byte mode,
 * pure SSM trees (no plastic layer) with per-tree, L2-normalized, rotary SSM keys/queries,
 * trained with blockwise Muon:
 *   ./nanotss_blt --data input.txt --seconds 300 --save model.bin
 * Previous BLT stack (patching, plastic layer, conv mixer): add --mode blt
 * Session (read, adapt online with TTT + match model, generate on demand):
 *   ./nanotss_blt --load model.bin --session 1   (stdin; @@gen N [temp] [cands], @@file, @@stats)
 * Sample:
 *   ./nanotss_blt --load model.bin --sample 600 --prompt "ROMEO:"
 * Stream a context file without resetting recurrent memory:
 *   ./nanotss_blt --load model.bin --context-file context.txt --sample 200
 * Define NANOTSS_NO_MAIN to include this file in your own C translation unit.
 *
 * GLOBAL STACK (default fw_mask=1):
 *   protected plastic tree -> stateless gated FFF -> original SSM tree -> FFF
 * Each block retains its local causal convolution/residual machinery.
 * The optional BLT ternary mixer precedes the global stack (conv-mix=2).
 * Local BLT encoder/decoder and the entropy model retain SSM/FFF blocks.
 *
 * PLASTIC MEMORY, per visited tree node (H has shape n x p):
 *   g = clip((gate_logit + 1)/2, 0, 1)
 *   d = 1 - 0.1*g*(1-sigmoid(-A))
 *   H <- d*H + g*k*(v - (d*H)^T*k)^T
 * Query q reads H^T*q. Keys/queries are learned normalized features.
 * g=0 preserves H exactly: neither writes nor decay occur.
 * Slow parameters train with backpropagation and Muon/AdamW; fast state updates online.
 * Original SSM dynamics and coupled-routing FFF gates are retained.
 *
 * CORE API (types and functions below are translation-unit local):
 *   Cfg / model_new / params_init / model_requantize
 *   tss_train_step / eval_seq / gen_new / gen_step / save / load
 * gen_step retains state in Gen across calls. Create a fresh Gen per sequence.
 * BLT API: faithful_new / faithful_step / faithful_eval / fstream_new /
 *          fstream_step / faithful_save / faithful_load.
 *
 * Default width 128; four global layers. Optimized AVX-512 kernels are used
 * when available, with scalar fallback. REAL=double is also supported.
 * CPU acceleration: cached signed-byte ternary rows for AVX-512 VNNI,
 * glibc vector libm with double intermediates, and cached FFF gate derivatives.
 * The checkpoint layout and all learned patching/model equations are retained.
 * --cpu-reference disables the new VNNI/vector math/batched FFF paths.
 * Protected checkpoint formats: TFP1 (core), TPC1 (BLT). Checkpoints store
 * configuration and slow weights, not live memory or optimizer moments.
 * Loaded models retain their saved stack mask. Old unprotected plastic
 * checkpoints are rejected by this protected build.
 *
 * BLT uses a separately trained, then frozen neural entropy predictor,
 * learned patch cross-attention, and 3..8-byte hash embeddings. The default
 * 300-second budget includes 60 seconds of entropy pretraining/calibration.
 * Hash embeddings are excluded from the reported capacity budget only;
 * they still consume memory/compute. Default total inference weights:
 * 1,428,864 including the 1,024-weight ternary mixer.
 *
 * Evidence: the small symbolic core achieved 87.5% mean four-binding recall
 * across three seeds after 1,048,576 distractors (96 queries total), training
 * on gaps up to 256. One seed still leaked writes. This is not a guarantee
 * of language-quality improvement or million-fact capacity. BLT's upstream
 * recurrent encoder is absent from that symbolic experiment.
 */
#ifndef PROTECTED_FW
#define PROTECTED_FW 1
#endif
// REAL=double automatically disables float-only intrinsics.
#define TSS_CAT_(a,b) a##b
#define TSS_CAT(a,b) TSS_CAT_(a,b)
#define TSS_IS_DOUBLE_double 1
#if defined(REAL) && TSS_CAT(TSS_IS_DOUBLE_, REAL)
#ifndef TSS_DOUBLE
#define TSS_DOUBLE 1
#endif
#endif
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifndef REAL
#define REAL float
#endif
typedef REAL real;
typedef struct {
    int vocab,d,n_layer,T,D,n,p,K,group,walks;
    real temp;
    int fw_mask,fff_gate; /* bit l selects fast-weight tree at even layer l */
} Cfg;
typedef struct {int vocab,d,n_layer,T,D,n,p,K,group,walks;real temp;} LegacyCfg;
static int read_cfg(FILE*fp,Cfg*c,int legacy){memset(c,0,sizeof(*c));if(!legacy)return fread(c,sizeof(*c),1,fp)==1;LegacyCfg old;if(fread(&old,sizeof(old),1,fp)!=1)return 0;memcpy(c,&old,sizeof(old));return 1;}
typedef struct {
    real *embed,*norm_f,*norm,*conv,*w_in,*bias,*w_out,*A_log,*proj,*dt_bias,*U;
} Params;
#define NODES(c) ((1 << ((c)->D + 1)) - 1)
/* ssm_groups (GQA analogue): the SSM key B and query C are computed per group of trees instead of
 * once for all trees (1 = shared, as before; T = one per tree). Values V, read-out Z and steps stay as they were. */
static int ssm_groups=1;
/* n_loops (universal-transformer style): the layer stack is applied n_loops times per position with
 * shared weights. Virtual layer v uses the parameters of layer v % n_layer and its own buffers/state. */
static int n_loops=1;
/* ssm_qknorm: L2-normalize each group's SSM key B and query C. ssm_rope: rotate coordinate pairs of
 * B and C by angle pos*theta_k (theta_k = 10000^(-2k/n)), so a state written at s and read at t
 * responds to t-s. Both act on the projections right after they are computed. */
static int ssm_qknorm=0,ssm_rope=0;
/* pos_feat>0: add fixed sinusoidal codes of (bytes since the last space/newline) to dims 0..31 and
 * (bytes since the last newline) to dims 32..63 of the input embedding, amplitude pos_feat. No parameters. */
static real pos_feat=0;
static void pos_feat_add(real*x,int kw,int kl,int d){
    if(pos_feat<=0||d<64)return;
    for(int j=0;j<16;j++){double w=pow(1.5,-(double)j);x[2*j]+=pos_feat*(real)sin(kw*w);x[2*j+1]+=pos_feat*(real)cos(kw*w);
        double wl=pow(1.7,-(double)j);x[32+2*j]+=pos_feat*(real)sin(kl*wl);x[32+2*j+1]+=pos_feat*(real)cos(kl*wl);}
}
static inline void pos_feat_next(int tok,int*kw,int*kl){if(tok=='\n'){*kw=0;*kl=0;}else if(tok==' '){*kw=0;(*kl)++;}else{(*kw)++;(*kl)++;}}
#define BASE_PDIM(c) (2*(c)->n*ssm_groups+2*(c)->p+(c)->T)
#define PDIM(c) BASE_PDIM(c)
/* Nine leaves per FFF tree gain two children: 273 rather than 255 nodes.
 * Two TSS + two FFF blocks: 1,155,584 active parameters at width 128. */
static int fff_nodes(const Cfg*c){int leaves=1<<c->D;return NODES(c)+2*(leaves<9?leaves:9);}
static size_t node_off(const Cfg*c,int l){return (size_t)c->T*((l+1)/2*NODES(c)+l/2*fff_nodes(c));}
static size_t node_rows(const Cfg*c){return node_off(c,c->n_layer);}
static real gelu(real x) {
    return (real)0.5*x*(1+erf(x*(real)0.70710678118654752440));
}
static real dgelu(real x) {
    return (real)0.5*(1+erf(x*(real)0.70710678118654752440))+x*exp((real)-0.5*x*x)*(real)0.39894228040143267794;
}
static real softplus(real x) {
    return x>20?x:log1p(exp(x));
}
static real sigm(real x) {
    return 1/(1+exp(-x));
}
/* Reference switch retains original reduction order for validation. */
static int runtime_fast=1;
static real ssm_dec_cap=1;
/* ssm_ltc=1: SSM tree nodes use the liquid time-constant update (Hasani et al. 2020, fused
 * semi-implicit Euler, unit step): dec = 1/(1 + 1/tau + f), gain = f*dec, with 1/tau = exp(A_log)
 * per node and the input-dependent gate f = softplus(dt preactivation). Must match at train and load. */
static int ssm_ltc=0;
/* ssm_ltc=2 ("true liquid"): the gate also depends on the state. Per visited node,
 * f = softplus(z + Z . (C^T H_prev)), z the input gate preactivation, Z the input-dependent read-out
 * vector and C the query: the time constant follows how strongly the node's memory answers the query. */
static inline real ltc2_gate(real z,const real*Z,const real*C,const real*H,int n,int pv){
    real a=z;for(int q=0;q<pv;q++){real gq=0;for(int j=0;j<n;j++)gq+=C[j]*H[j*pv+q];a+=Z[q]*gq;}return a;
}
static inline real _cvtsh_ss_bf(uint16_t b){uint32_t u=(uint32_t)b<<16;float f;memcpy(&f,&u,4);return f;}
/* Rotary angles for position pos: one cos/sin per coordinate pair, shared by every key and query. */
static void rope_angles(long pos,int n,real*cs,real*sn){
    static double th[32];static int tn=0;if(tn!=n){for(int k=0;k+1<n&&k/2<32;k+=2)th[k/2]=pow(10000.,-(double)k/n);tn=n;}
    for(int k=0;k+1<n;k+=2){double a=pos*th[k/2];cs[k/2]=(real)cos(a);sn[k/2]=(real)sin(a);}
}
static void ssm_prep(real*pt,int n,int G,long pos,real*inv){
    real cs[32],sn[32];if(ssm_rope)rope_angles(pos,n,cs,sn);
    for(int part=0;part<2;part++)for(int g=0;g<G;g++){real*v=pt+(size_t)part*G*n+(size_t)g*n;
        if(ssm_qknorm){real ss=0;for(int j=0;j<n;j++)ss+=v[j]*v[j];real r=1/sqrt(ss+(real)1e-6);inv[part*G+g]=r;for(int j=0;j<n;j++)v[j]*=r;}
        if(ssm_rope)for(int k=0;k+1<n;k+=2){real c_=cs[k/2],s_=sn[k/2],a=v[k],b=v[k+1];v[k]=a*c_-b*s_;v[k+1]=a*s_+b*c_;}
    }
}
/* Gradient through ssm_prep: dpt holds d/d(prepared values); pt the prepared values. */
static void ssm_prep_bwd(const real*pt,real*dpt,int n,int G,long pos,const real*inv){
    real cs[32],sn[32];if(ssm_rope)rope_angles(pos,n,cs,sn);
    for(int part=0;part<2;part++)for(int g=0;g<G;g++){const real*v=pt+(size_t)part*G*n+(size_t)g*n;real*dv=dpt+(size_t)part*G*n+(size_t)g*n;real u[64];
        memcpy(u,v,n*sizeof(real));
        if(ssm_rope)for(int k=0;k+1<n;k+=2){real c_=cs[k/2],s_=sn[k/2],a=dv[k],b=dv[k+1];dv[k]=a*c_+b*s_;dv[k+1]=-a*s_+b*c_;
            real x=u[k],y=u[k+1];u[k]=x*c_+y*s_;u[k+1]=-x*s_+y*c_;}
        if(ssm_qknorm){real r=inv[part*G+g],dot=0;for(int j=0;j<n;j++)dot+=u[j]*dv[j];for(int j=0;j<n;j++)dv[j]=r*(dv[j]-u[j]*dot);}
    }
}
static inline void ssm_coef(real ea,real dt,real*dec,real*gain){if(ssm_ltc){real q=1/(1+ea+dt);*dec=q;*gain=dt*q;}else{*dec=exp(-ea*dt);*gain=dt;}} /* <1 caps SSM retention per step: horizon ~ 1/(1-cap) steps */
#if defined(__AVX512F__) && !defined(TSS_DOUBLE)
#include <immintrin.h>
#endif
static inline __attribute__((always_inline)) real dotr(const real *a,const real *b,int n) {
#if defined(__AVX512F__) && !defined(TSS_DOUBLE)
    if(runtime_fast && n>=16){
        __m512 acc=_mm512_setzero_ps();int i=0;
        for(;i+16<=n;i+=16)acc=_mm512_fmadd_ps(_mm512_loadu_ps(a+i),_mm512_loadu_ps(b+i),acc);
        real sum=_mm512_reduce_add_ps(acc);for(;i<n;i++)sum+=a[i]*b[i];return sum;
    }
#endif
    real s=0;
    for(int i=0;i<n;i++) s+=a[i]*b[i];
    return s;
}
void tss_ternary(const real *w,real *out,long rows,int d,int g) {
    for(long r=0;r<rows;r++) for(int j=0;j<d;j+=g) {
        const real *wr=w+r*d+j;
        real s=0;
        for(int i=0;i<g;i++) s+=fabs(wr[i]);
        s=s/g;
        if(s<(real)1e-8)s=(real)1e-8;
        for(int i=0;i<g;i++) {
            real v=wr[i]/s;
            v=v>1?1:(v< -1?-1:v);
            out[r*d+j+i]=nearbyint(v)*s;
        }
    }
}
void tss_quantize(const Cfg *c,const Params *P,Params *Q) {
    const long N=NODES(c),L=c->n_layer;
    tss_ternary(P->w_in,Q->w_in,node_rows(c),c->d,c->group);
    tss_ternary(P->w_out,Q->w_out,node_rows(c),c->d,c->group);
    tss_ternary(P->proj,Q->proj,L*PDIM(c),c->d,c->group);
    tss_ternary(P->U,Q->U,L*c->T*c->p,c->d,c->group);
    memcpy(Q->A_log,P->A_log,sizeof(real)*node_rows(c));
}
static int continuous_check=0; /* numerical gradient check only */
static void quant8(const real *x,real *xq,int8_t *code,real *xs,int d) {
    if(continuous_check){memcpy(xq,x,sizeof(real)*d);memset(code,0,d);*xs=1;return;}
    real m=0;
    for(int i=0;i<d;i++)m=fabs(x[i])>m?fabs(x[i]):m;
    if(m<(real)1e-5)m=(real)1e-5;
    const real scale=127/m;
    for(int i=0;i<d;i++) {
        real v=nearbyint(x[i]*scale);
        v=v>127?127:(v< -128?-128:v);
        xq[i]=v/scale;
        code[i]=(int8_t)v;
    }*xs=1/scale;
}
static void rmsnorm_fwd(const real *x,const real *w,real *out,int d,real *r_out) {
    const real r=1/sqrt(dotr(x,x,d)/d+(real)1e-5);
    for(int i=0;i<d;i++)out[i]=x[i]*r*w[i];
    *r_out=r;
}
static void rmsnorm_bwd(const real *x,const real *w,real r,const real *du,real *dx,real *dw,int d) {
    real s=0;
    for(int i=0;i<d;i++)s+=w[i]*du[i]*x[i];
    const real k=r*r*r*s/d;
    for(int i=0;i<d;i++) {
        dx[i]+=r*w[i]*du[i]-x[i]*k;
        dw[i]+=du[i]*x[i]*r;
    }
}
#if defined(__AVX512BW__) && defined(__AVX512F__) && !defined(TSS_DOUBLE)
#define FAST 1
#include <immintrin.h>
#else
#define FAST 0
#endif
/* glibc's vector libm retains double evaluation before float rounding. */
#if FAST && defined(__GLIBC__)
#if __GLIBC_PREREQ(2,35)
#define VMATH 1
extern __m512d _ZGVeN8v_tanh(__m512d);
extern __m512d _ZGVeN8v_exp(__m512d);
extern __m512d _ZGVeN8v_erf(__m512d);
#else
#define VMATH 0
#endif
#else
#define VMATH 0
#endif
static int cpu_fast=1;
static void tanh_array(real*x,int n){
 int i=0;
#if VMATH
 if(cpu_fast)for(;i+8<=n;i+=8){__m512d a=_mm512_cvtps_pd(_mm256_loadu_ps(x+i));_mm256_storeu_ps(x+i,_mm512_cvtpd_ps(_ZGVeN8v_tanh(a)));}
#endif
 for(;i<n;i++)x[i]=tanh(x[i]);
}
static void head_exponentials(const real*x,real mx,double*out,int n){
 int i=0;
#if VMATH
 if(cpu_fast)for(;i+8<=n;i+=8){__m256 z=_mm256_sub_ps(_mm256_loadu_ps(x+i),_mm256_set1_ps(mx));_mm512_storeu_pd(out+i,_ZGVeN8v_exp(_mm512_cvtps_pd(z)));}
#endif
 for(;i<n;i++)out[i]=exp(x[i]-mx);
}
static void gelu_array(const real*x,real*y,real*dy,int n,int training){
 double ee[n],de[n];int i=0;
#if VMATH
 if(cpu_fast)for(;i+8<=n;i+=8){
  __m256 v=_mm256_loadu_ps(x+i),z=_mm256_mul_ps(v,_mm256_set1_ps((real).70710678118654752440));
  _mm512_storeu_pd(ee+i,_ZGVeN8v_erf(_mm512_cvtps_pd(z)));
  if(training){z=_mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(-.5f),v),v);_mm512_storeu_pd(de+i,_ZGVeN8v_exp(_mm512_cvtps_pd(z)));}
 }
#endif
 for(int j=0;j<i;j++){y[j]=(real).5*x[j]*(1+ee[j]);if(training)dy[j]=(real).5*(1+ee[j])+x[j]*de[j]*(real).39894228040143267794;}
 for(;i<n;i++){y[i]=gelu(x[i]);if(training)dy[i]=dgelu(x[i]);}
}
static void sigmoid_array(real*x,int n){
 int i=0;
#if VMATH
 if(cpu_fast)for(;i+8<=n;i+=8){__m512d v=_mm512_cvtps_pd(_mm256_sub_ps(_mm256_setzero_ps(),_mm256_loadu_ps(x+i)));
  v=_ZGVeN8v_exp(v);v=_mm512_div_pd(_mm512_set1_pd(1),_mm512_add_pd(_mm512_set1_pd(1),v));_mm256_storeu_ps(x+i,_mm512_cvtpd_ps(v));}
#endif
 for(;i<n;i++)x[i]=sigm(x[i]);
}
typedef struct {
    uint64_t *bits;
    real *sc;
    int8_t *i8;
    int32_t *correction;
} Plane;
typedef struct {
    int ok;
    Plane w_in,w_out,proj,U;
    double *decay_rates;
}
Codes;
static void plane_rows(const real *q,Plane *pl,long r0,long r1,int d,int g) {
    for(long r=r0;r<r1;r++) {
        const real *qr=q+r*d;
        uint64_t *b=pl->bits+r*(d/32);
        for(int j=0;j<d;j+=64) {
            uint64_t P=0,Nm=0;
            for(int i=0;i<64;i++) {
                P|=(uint64_t)(qr[j+i]>0)<<i;
                Nm|=(uint64_t)(qr[j+i]<0)<<i;
                if(pl->i8)pl->i8[r*d+j+i]=(qr[j+i]>0)-(qr[j+i]<0);
            }b[(j/64)*2]=P;
            b[(j/64)*2+1]=Nm;
        }
        for(int j=0;j<d;j+=g) {
            real m=0;
            for(int i=0;i<g;i++)m=fabs(qr[j+i])>m?fabs(qr[j+i]):m;
            pl->sc[r*(d/g)+j/g]=m;
            if(pl->correction){int sum=0;for(int i=j;i<j+g;i+=64)sum+=__builtin_popcountll(b[(i/64)*2])-__builtin_popcountll(b[(i/64)*2+1]);pl->correction[r*(d/g)+j/g]=128*sum;}
        }
    }
}
static Plane plane_new(long rows,int d,int g) {
    Plane p= {
        calloc((size_t)rows*(d/32),sizeof(uint64_t)),calloc((size_t)rows*(d/g),sizeof(real))
    };
#if FAST && defined(__AVX512VNNI__)
    p.i8=calloc((size_t)rows*d,sizeof(int8_t));p.correction=calloc((size_t)rows*(d/g),sizeof(int32_t));
#endif
    return p;
}
Codes *tss_codes_new(const Cfg *c) {
    Codes *k=calloc(1,sizeof(Codes));
    k->decay_rates=calloc(node_rows(c),sizeof(double));
    k->ok=FAST&&c->d%64==0&&c->group%64==0&&c->p%16==0;
    if(!k->ok)return k;
    const long N=NODES(c),L=c->n_layer;
    k->w_in=plane_new(node_rows(c),c->d,c->group);
    k->w_out=plane_new(node_rows(c),c->d,c->group);
    k->proj=plane_new(L*PDIM(c),c->d,c->group);
    k->U=plane_new(L*c->T*c->p,c->d,c->group);
    return k;
}
void tss_codes_all(const Cfg *c,const Params *Q,Codes *k) {
    if(!k)return;
    const long N=NODES(c),L=c->n_layer;
    for(long r=0;r<node_rows(c);r++)k->decay_rates[r]=exp(Q->A_log[r]);
    if(!k->ok)return;
    plane_rows(Q->w_in,&k->w_in,0,node_rows(c),c->d,c->group);
    plane_rows(Q->w_out,&k->w_out,0,node_rows(c),c->d,c->group);
    plane_rows(Q->proj,&k->proj,0,L*PDIM(c),c->d,c->group);
    plane_rows(Q->U,&k->U,0,L*c->T*c->p,c->d,c->group);
}
#if FAST
static inline float pdot8(const Plane *pl,size_t r,const int8_t *xq,float xs,int d,int g) {
#if defined(__AVX512VNNI__)
    if(cpu_fast&&pl->i8){
        float acc=0;const __m512i offset=_mm512_set1_epi8((char)0x80);
        for(int j=0;j<d;j+=g){__m512i s=_mm512_setzero_si512();
            for(int i=j;i<j+g;i+=64){__m512i x=_mm512_xor_si512(_mm512_loadu_si512(xq+i),offset);s=_mm512_dpbusd_epi32(s,x,_mm512_loadu_si512(pl->i8+r*d+i));}
            acc+=pl->sc[r*(d/g)+j/g]*(float)(_mm512_reduce_add_epi32(s)-pl->correction[r*(d/g)+j/g]);
        }return acc*xs;
    }
#endif
    const uint64_t *b=pl->bits+r*(d/32);
    const float *sc=pl->sc+r*(d/g);
    const __m512i ones8=_mm512_set1_epi8(1),ones16=_mm512_set1_epi16(1);
    float acc=0;
    for(int j=0;j<d;j+=g) {
        __m512i s32=_mm512_setzero_si512();
        for(int i=j;i<j+g;i+=64) {
            const __m512i x=_mm512_loadu_si512((const void *)(xq+i));
            __m512i v=_mm512_maskz_mov_epi8(b[(i/64)*2],x);
            v=_mm512_mask_sub_epi8(v,b[(i/64)*2+1],v,x);
            s32=_mm512_add_epi32(s32,_mm512_madd_epi16(_mm512_maddubs_epi16(ones8,v),ones16));
        }acc+=sc[j/g]*(float)_mm512_reduce_add_epi32(s32);
    }return acc*xs;
}
static inline float pdotf(const Plane *pl,size_t r,const float *v,int d,int g) {
    const uint64_t *b=pl->bits+r*(d/32);
    const float *sc=pl->sc+r*(d/g);
    float acc=0;
    for(int j=0;j<d;j+=g) {
        __m512 s=_mm512_setzero_ps();
        for(int i=j;i<j+g;i+=64) {
            const uint64_t P=b[(i/64)*2],Nm=b[(i/64)*2+1];
            for(int q=0;q<4;q++) {
                const __m512 x=_mm512_loadu_ps(v+i+16*q);
                s=_mm512_mask_add_ps(s,(__mmask16)(P>>(16*q)),s,x);
                s=_mm512_mask_sub_ps(s,(__mmask16)(Nm>>(16*q)),s,x);
            }
        }acc+=sc[j/g]*_mm512_reduce_add_ps(s);
    }return acc;
}
static inline void paxpyf(float *y,float a,const Plane *pl,size_t r,int d,int g) {
    const uint64_t *b=pl->bits+r*(d/32);
    const float *sc=pl->sc+r*(d/g);
    for(int j=0;j<d;j+=g) {
        const __m512 as=_mm512_set1_ps(a*sc[j/g]);
        for(int i=j;i<j+g;i+=64) {
            const uint64_t P=b[(i/64)*2],Nm=b[(i/64)*2+1];
            for(int q=0;q<4;q++) {
                __m512 yv=_mm512_loadu_ps(y+i+16*q);
                yv=_mm512_mask_add_ps(yv,(__mmask16)(P>>(16*q)),yv,as);
                yv=_mm512_mask_sub_ps(yv,(__mmask16)(Nm>>(16*q)),yv,as);
                _mm512_storeu_ps(y+i+16*q,yv);
            }
        }
    }
}
#endif
// Preserve double precision of the original nested exp(-exp(A_log)*dt).
static inline double decay_rate(const Codes *k,size_t row,real a) {
    if(k)return k->decay_rates[row];
    return exp(a);
}
static inline real qdotx(const Codes *k,const Plane *pl,const real *qm,size_t r,const real *x,const int8_t *xq,real xs,int d,int g) {
#if FAST
    if(k&&k->ok)return pdot8(pl,r,xq,xs,d,g);
#endif
    (void)k;
    (void)pl;
    (void)xq;
    (void)xs;
    (void)g;
    return dotr(qm+r*d,x,d);
}
static inline real qdotf(const Codes *k,const Plane *pl,const real *qm,size_t r,const real *v,int d,int g) {
#if FAST
    if(k&&k->ok)return pdotf(pl,r,v,d,g);
#endif
    (void)k;
    (void)pl;
    (void)g;
    return dotr(qm+r*d,v,d);
}
static inline void qaxpy(const Codes *k,const Plane *pl,const real *qm,size_t r,real *y,real a,int d,int g) {
#if FAST
    if(k&&k->ok) {
        paxpyf(y,a,pl,r,d,g);
        return;
    }
#endif
    (void)k;
    (void)pl;
    (void)g;
    const real *row=qm+r*d;
    for(int i=0;i<d;i++)y[i]+=a*row[i];
}
static inline void axpy(real *y,real a,const real *x,int d) {
    for(int i=0;i<d;i++)y[i]+=a*x[i];
}
typedef struct {
    real *xin,*u,*rn,*x,*p,*dt,*xs,*fw_raw;
    int8_t *xq;
    int *node;
    real *s,*R,*hprev,*decay;
    int *anode;
    real *as,*aR,*bw,*gg,*dg,*fg_raw,*fg_gate,*fg_low,*fg_gg,*fg_dg;
} Tape;
// Reuse the same erf evaluation for GELU and its derivative, retaining the
// original arithmetic grouping. Evaluation records values but skips derivatives.
static inline real record_gelu(real s,Tape *tp,size_t v,int training) {
    const double e=erf(s*(real)0.70710678118654752440);
    const real y=(real)0.5*s*(1+e);
    tp->gg[v]=y;
    if(training)tp->dg[v]=(real)0.5*(1+e)+s*exp((real)-0.5*s*s)*(real)0.39894228040143267794;
    return y;
}
static inline real tape_gelu(const Tape *tp,size_t v) {
    return tp->gg[v];
}
static inline real tape_dgelu(const Tape *tp,size_t v) {
    return tp->dg[v];
}
static size_t tape_reals(const Cfg *c,int L) {
    const size_t V=(size_t)L*c->T*(c->D+2),A=(size_t)L*c->T*(c->D+1)*(c->D+2)/2;
    return (size_t)L*PDIM(c)+(size_t)L*(5*c->d+3*c->n+2+PDIM(c)+c->T)+V*(1+c->p+c->n*c->p+1+1+2*1)+A*(1+c->p);
}
static void tape_carve(const Cfg *c,int L,real *buf,int *ibuf,Tape *tp) {
    const size_t V=(size_t)L*c->T*(c->D+2),A=(size_t)L*c->T*(c->D+1)*(c->D+2)/2;
    tp->fw_raw=buf;buf+=(size_t)L*PDIM(c);
    tp->xin=buf;
    buf+=(size_t)L*c->d;
    tp->u=buf;
    buf+=(size_t)L*c->d;
    tp->x=buf;
    buf+=(size_t)L*c->d;
    tp->rn=buf;
    buf+=L;
    tp->xs=buf;
    buf+=L;
    tp->p=buf;
    buf+=(size_t)L*PDIM(c);
    tp->dt=buf;
    buf+=(size_t)L*c->T;
    tp->s=buf;
    buf+=V;
    tp->R=buf;
    buf+=V*c->p;
    tp->hprev=buf;
    buf+=V*c->n*c->p;
    tp->decay=buf;
    buf+=V;
    tp->as=buf;
    buf+=A;
    tp->aR=buf;
    buf+=A*c->p;
    tp->bw=buf;
    buf+=V;
    tp->gg=buf;
    buf+=V;
    tp->dg=buf;
    buf+=V;
    tp->fg_raw=buf;buf+=(size_t)L*c->d;
    tp->fg_gate=buf;buf+=(size_t)L*c->d;
    tp->fg_low=buf;buf+=(size_t)L*c->n;
    tp->fg_gg=buf;buf+=(size_t)L*c->n;tp->fg_dg=buf;buf+=(size_t)L*c->n;
    tp->node=ibuf;
    ibuf+=V;
    tp->anode=ibuf;
}
typedef struct {
    int Lmax;
    real *tapes,*x,*h,*g,*xn,*dxn,*lg,*lam;
    real *ds,*dR,*dp,*dx,*du,*gUr,*on,*Racc,*mx,*y;
    int *itapes;
    int8_t *xq8;
    int *rcnt,*rord,*walk;
    uint16_t *hp16;
    int tape_bf16;
    Tape *tp;
    const Codes *k;
    uint64_t rng;
    int no_alt;
    real *ys,*rs,*drs;real *head;
    double *head_exp;
    real *stream_hist;
    real *carry_h,*carry_hist; /* training state carry: per even layer h, per layer K-1 conv inputs */
    real *qkn; long pos; int lv; real *mtp_g,*lat_g,*lat_u,*pos_loss,*lat_l1; int *lat_b; uint64_t lat_rng; /* per-thread MTP / latent-thought gradients; latent inputs */ /* per-thread gradient of the MTP head matrix */ /* lv: current virtual layer (buffer index when looping) */ /* SSM QK-norm inverse norms per (layer,t,group,B/C); stream position for rotary */
    void (*prepare_state)(void*,real*,size_t,int,int);
    void *prepare_state_user;
    const Tape *reuse_fw_input;
}
Work;
/* Global-stack state carried across consecutive training windows (one slot per batch row). */
static real*carry_bank_h=NULL,*carry_bank_hist=NULL;static size_t carry_sz_h=0,carry_sz_hist=0;
/* Seed an even (stateful) layer's state from the carry slot, or zero it. */
static void carry_pre(const Cfg*c,Work*w,int l,real*h){size_t S=(size_t)c->T*NODES(c)*c->n*c->p;
    if(w->carry_h)memcpy(h,w->carry_h+(size_t)(l/2)*S,S*sizeof(real));else memset(h,0,S*sizeof(real));}
/* Save an even layer's final state and its last K-1 conv inputs into the carry slot. */
static void carry_post(const Cfg*c,Work*w,int l,int L,const real*h){if(!w->carry_h)return;size_t S=(size_t)c->T*NODES(c)*c->n*c->p;
    memcpy(w->carry_h+(size_t)(l/2)*S,h,S*sizeof(real));int K=c->K,d=c->d;
    if(w->carry_hist&&K>1){real*hb=w->carry_hist+(size_t)l*(K-1)*d;
        for(int i=K-2;i>=0;i--){int t=L-1-i;if(t>=0)memcpy(hb+(size_t)i*d,w->tp[l].u+(size_t)t*d,d*sizeof(real));else if(i-L>=0)memcpy(hb+(size_t)i*d,hb+(size_t)(i-L)*d,d*sizeof(real));}}}
static inline real urand(uint64_t *s) {
    *s^=*s>>12;
    *s^=*s<<25;
    *s^=*s>>27;
    return (real)((*s*2685821657736338717ULL)>>11)*(real)(1.0/9007199254740992.0);
}
void tss_work_seed(Work *w,uint64_t seed) {
    w->rng=seed*0x9E3779B97F4A7C15ULL+1;
}
// CPU microkernels. Tiles hold four/eight tokens or four gradient rows,
// never a dense expansion of the sparse forest. Packed weights stay packed.
static void qrows_dot8_batch(const Codes *k,const Plane *pl,const real *qm,size_t r0,
                            const real *x,const int8_t *xq,const real *xs,real *out,
                            int L,int rows,int d,int group) {
#if FAST
#if defined(__AVX512VNNI__)
    if(cpu_fast&&k&&k->ok&&pl->i8){
        const __m512i offset=_mm512_set1_epi8((char)0x80);
        for(int t=0;t<L;t+=4){const int nt=L-t<4?L-t:4;
            for(int r=0;r<rows;r++){float a[4]={0};
                for(int j=0;j<d;j+=group){__m512i sum[4];for(int z=0;z<nt;z++)sum[z]=_mm512_setzero_si512();
                    for(int i=j;i<j+group;i+=64){__m512i w=_mm512_loadu_si512(pl->i8+(r0+r)*d+i);
                        for(int z=0;z<nt;z++){__m512i v=_mm512_xor_si512(_mm512_loadu_si512(xq+(size_t)(t+z)*d+i),offset);sum[z]=_mm512_dpbusd_epi32(sum[z],v,w);}}
                    for(int z=0;z<nt;z++)a[z]+=pl->sc[(r0+r)*(d/group)+j/group]*(float)(_mm512_reduce_add_epi32(sum[z])-pl->correction[(r0+r)*(d/group)+j/group]);
                }
                for(int z=0;z<nt;z++)out[(size_t)(t+z)*rows+r]=a[z]*xs[t+z];
            }
        }return;
    }
#endif
    if(k&&k->ok) {
        const __m512i ones8=_mm512_set1_epi8(1),ones16=_mm512_set1_epi16(1);
        for(int t=0;t<L;t+=4) {
            const int nt=L-t<4?L-t:4;
            for(int r=0;r<rows;r++) {
                const uint64_t *bits=pl->bits+(r0+r)*(d/32);
                const real *sc=pl->sc+(r0+r)*(d/group);
                float a[4]={0};
                for(int j=0;j<d;j+=group) {
                    __m512i sum[4];for(int z=0;z<nt;z++)sum[z]=_mm512_setzero_si512();
                    for(int i=j;i<j+group;i+=64) {
                        const __mmask64 pos=bits[(i/64)*2],neg=bits[(i/64)*2+1];
                        for(int z=0;z<nt;z++) {
                            __m512i v=_mm512_loadu_si512(xq+(size_t)(t+z)*d+i);
                            __m512i s=_mm512_maskz_mov_epi8(pos,v);
                            s=_mm512_mask_sub_epi8(s,neg,s,v);
                            sum[z]=_mm512_add_epi32(sum[z],_mm512_madd_epi16(_mm512_maddubs_epi16(ones8,s),ones16));
                        }
                    }
                    for(int z=0;z<nt;z++)a[z]+=sc[j/group]*(float)_mm512_reduce_add_epi32(sum[z]);
                }
                for(int z=0;z<nt;z++)out[(size_t)(t+z)*rows+r]=a[z]*xs[t+z];
            }
        }
        return;
    }
#endif
    for(int t=0;t<L;t++)for(int r=0;r<rows;r++)out[(size_t)t*rows+r]=qdotx(k,pl,qm,r0+r,x+(size_t)t*d,xq+(size_t)t*d,xs[t],d,group);
}
static void qrows_dotf_batch(const Codes *k,const Plane *pl,const real *qm,size_t r0,
                            const real *x,real *out,int L,int rows,int d,int group) {
#if FAST
    if(k&&k->ok) {
        for(int t=0;t<L;t+=4) {
            const int nt=L-t<4?L-t:4;
            for(int r=0;r<rows;r++) {
                const uint64_t *bits=pl->bits+(r0+r)*(d/32);
                const real *sc=pl->sc+(r0+r)*(d/group);float a[4]={0};
                for(int j=0;j<d;j+=group) {
                    __m512 sum[4];for(int z=0;z<nt;z++)sum[z]=_mm512_setzero_ps();
                    for(int i=j;i<j+group;i+=16) {
                        const int shift=i%64;
                        const __mmask16 pos=(__mmask16)(bits[(i/64)*2]>>shift),neg=(__mmask16)(bits[(i/64)*2+1]>>shift);
                        for(int z=0;z<nt;z++) {
                            __m512 v=_mm512_loadu_ps(x+(size_t)(t+z)*d+i);
                            sum[z]=_mm512_mask_add_ps(sum[z],pos,sum[z],v);
                            sum[z]=_mm512_mask_sub_ps(sum[z],neg,sum[z],v);
                        }
                    }
                    for(int z=0;z<nt;z++)a[z]+=sc[j/group]*_mm512_reduce_add_ps(sum[z]);
                }
                for(int z=0;z<nt;z++)out[(size_t)(t+z)*rows+r]=a[z];
            }
        }
        return;
    }
#endif
    for(int t=0;t<L;t++)for(int r=0;r<rows;r++)out[(size_t)t*rows+r]=qdotf(k,pl,qm,r0+r,x+(size_t)t*d,d,group);
}
// Y += A Q. Keep each output slice in registers across all ternary rows;
// its additions retain row order, with the same mask-add/subtract arithmetic.
static void qrows_axpy_batch(const Codes *k,const Plane *pl,const real *qm,size_t r0,
                            const real *a,real *y,int L,int rows,int d,int group) {
#if FAST
    if(k&&k->ok) {
        for(int t=0;t<L;t+=8) {
            const int nt=L-t<8?L-t:8;
            for(int i=0;i<d;i+=16) {
                __m512 acc[8];for(int z=0;z<nt;z++)acc[z]=_mm512_loadu_ps(y+(size_t)(t+z)*d+i);
                for(int r=0;r<rows;r++) {
                    const uint64_t *bits=pl->bits+(r0+r)*(d/32);const int shift=i%64;
                    const __mmask16 pos=(__mmask16)(bits[(i/64)*2]>>shift),neg=(__mmask16)(bits[(i/64)*2+1]>>shift);
                    const float sc=pl->sc[(r0+r)*(d/group)+i/group];
                    for(int z=0;z<nt;z++) {
                        const __m512 v=_mm512_set1_ps(a[(size_t)(t+z)*rows+r]*sc);
                        acc[z]=_mm512_mask_add_ps(acc[z],pos,acc[z],v);
                        acc[z]=_mm512_mask_sub_ps(acc[z],neg,acc[z],v);
                    }
                }
                for(int z=0;z<nt;z++)_mm512_storeu_ps(y+(size_t)(t+z)*d+i,acc[z]);
            }
        }
        return;
    }
#endif
    for(int t=0;t<L;t++)for(int r=0;r<rows;r++)qaxpy(k,pl,qm,r0+r,y+(size_t)t*d,a[(size_t)t*rows+r],d,group);
}
// G += A^T X. Four gradient rows share each loaded activation vector.
static void outer_rows(real *g,const real *a,const real *x,int L,int rows,int d) {
#if FAST
    int r=0;
    for(;r+4<=rows;r+=4) {
        int i=0;
        for(;i+16<=d;i+=16) {
            __m512 v0=_mm512_loadu_ps(g+(size_t)(r+0)*d+i),v1=_mm512_loadu_ps(g+(size_t)(r+1)*d+i);
            __m512 v2=_mm512_loadu_ps(g+(size_t)(r+2)*d+i),v3=_mm512_loadu_ps(g+(size_t)(r+3)*d+i);
            for(int t=0;t<L;t++) {
                const __m512 xv=_mm512_loadu_ps(x+(size_t)t*d+i);const real *ar=a+(size_t)t*rows+r;
                v0=_mm512_fmadd_ps(_mm512_set1_ps(ar[0]),xv,v0);v1=_mm512_fmadd_ps(_mm512_set1_ps(ar[1]),xv,v1);
                v2=_mm512_fmadd_ps(_mm512_set1_ps(ar[2]),xv,v2);v3=_mm512_fmadd_ps(_mm512_set1_ps(ar[3]),xv,v3);
            }
            _mm512_storeu_ps(g+(size_t)(r+0)*d+i,v0);_mm512_storeu_ps(g+(size_t)(r+1)*d+i,v1);
            _mm512_storeu_ps(g+(size_t)(r+2)*d+i,v2);_mm512_storeu_ps(g+(size_t)(r+3)*d+i,v3);
        }
        for(;i<d;i++)for(int t=0;t<L;t++)for(int z=0;z<4;z++)g[(size_t)(r+z)*d+i]+=a[(size_t)t*rows+r+z]*x[(size_t)t*d+i];
    }
    for(;r<rows;r++)for(int t=0;t<L;t++)axpy(g+(size_t)r*d,a[(size_t)t*rows+r],x+(size_t)t*d,d);
#else
    for(int t=0;t<L;t++)for(int r=0;r<rows;r++)axpy(g+(size_t)r*d,a[(size_t)t*rows+r],x+(size_t)t*d,d);
#endif
}

/* Stateless FFF: bounded, identity-initialized channel gate. Reuses dormant
 * odd-layer proj/U storage: rank n down projection and n-to-d up projection.
 * FFF remains stateless; its sigmoid gate starts at unit gain.
 */
static void fff_gate_forward(const Cfg*c,const Params*Q,int l,const real*x,real*y,real*raw,real*gate,real*low,real*gg,real*dg,int training,const Codes*k){
    int d=c->d,n=c->n;size_t rp=(size_t)l*PDIM(c),ru=(size_t)l*c->T*c->p;
    for(int j=0;j<n;j++)low[j]=qdotf(k,k?&k->proj:NULL,Q->proj,rp+j,x,d,c->group);
    gelu_array(low,gg,dg,n,training);
    memset(gate,0,d*sizeof(real));
    for(int j=0;j<n;j++)qaxpy(k,k?&k->U:NULL,Q->U,ru+j,gate,gg[j],d,c->group);
    sigmoid_array(gate,d);
    for(int i=0;i<d;i++){raw[i]=y[i];gate[i]=2*gate[i];y[i]*=gate[i];}
}
static void fff_gate_backward(const Cfg*c,const Params*Q,Params*G,int l,const real*x,const real*gy,const real*raw,const real*gate,const real*low,const real*gg,const real*dg,real*treegy,real*dx,const Codes*k){
    int d=c->d,n=c->n;size_t rp=(size_t)l*PDIM(c),ru=(size_t)l*c->T*c->p;real dz[d];
    for(int i=0;i<d;i++){treegy[i]=gy[i]*gate[i];dz[i]=gy[i]*raw[i]*gate[i]*(1-gate[i]/2);}
    for(int j=0;j<n;j++){
        real dl=qdotf(k,k?&k->U:NULL,Q->U,ru+j,dz,d,c->group)*dg[j];
        axpy(G->U+(ru+j)*d,gg[j],dz,d);axpy(G->proj+(rp+j)*d,dl,x,d);
        qaxpy(k,k?&k->proj:NULL,Q->proj,rp+j,dx,dl,d,c->group);
    }
}

/* Stateless FFF-style path-sum MLP. Odd layers: RMSNorm, quant8, trees,
 * residual. No convolution, projection, readout, decay, or recurrent state. */
static void fff_fwd(const Cfg*c,const Params*P,const Params*Q,int l,int L,real*xres,Tape*tp,Work*w){
 const int d=c->d,T=c->T,D=c->D+1,N=fff_nodes(c),gs=c->group,AL=D*(D+1)/2;
 const size_t off=node_off(c,l);const Codes*k=w->k;
 const Plane*pi=k?&k->w_in:NULL,*po=k?&k->w_out:NULL;
 for(int t=0;t<L;t++){
  real*xx=xres+(size_t)t*d,*x=tp->x+(size_t)t*d,*u=tp->u+(size_t)t*d;int8_t*xq=tp->xq+(size_t)t*d;
  memcpy(tp->xin+(size_t)t*d,xx,d*sizeof(real));rmsnorm_fwd(xx,P->norm+(size_t)l*d,u,d,tp->rn+t);
  quant8(u,x,xq,tp->xs+t,d);real xs=tp->xs[t];memset(w->y,0,d*sizeof(real));
  for(int tr=0;tr<T;tr++){
   size_t vb=((size_t)t*T+tr)*(D+1),ab=((size_t)t*T+tr)*AL;int a=0;
   for(int j=0;j<=D;j++){tp->node[vb+j]=-1;if(cpu_fast)tp->s[vb+j]=0;}
   for(int j=0;j<AL;j++)tp->anode[ab+j]=-1;
   for(int j=0;j<=D;j++){
    if(a>=N)break;
    size_t row=(size_t)tr*N+a,v=vb+j;
    real s=qdotx(k,pi,Q->w_in,off+row,x,xq,xs,d,gs)+P->bias[off+row];
    tp->node[v]=row;tp->s[v]=s;
    if(!cpu_fast)qaxpy(k,po,Q->w_out,off+row,w->y,record_gelu(s,tp,v,!w->no_alt),d,gs);
    a=2*a+1+(s>0);
   }
   if(!w->no_alt){
    real tot=0;for(int j=0;j<D;j++){if(tp->node[vb+j]<0||2*(tp->node[vb+j]-tr*N)+1>=N){tp->bw[vb+j]=0;continue;}real sg=sigm(fabs(tp->s[vb+j])/c->temp);tp->bw[vb+j]=sg*(1-sg);tot+=tp->bw[vb+j];}
    int ai=0;
    for(int j=0;j<D;j++){
     if(tp->node[vb+j]<0||2*(tp->node[vb+j]-tr*N)+1>=N){tp->bw[vb+j]=0;ai+=D-j;continue;}
     real q=c->walks<=0?1:(tot>0?fmin(1,c->walks*tp->bw[vb+j]/tot):1);
     real bw=c->walks<=0?1:(urand(&w->rng)<q?1/q:0);tp->bw[vb+j]=bw;
     if(!bw){ai+=D-j;continue;}
     int a=2*(tp->node[vb+j]-tr*N)+2-(tp->s[vb+j]>0);
     for(int jj=j+1;jj<=D;jj++,ai++){
      if(a>=N){ai+=D-jj+1;break;}
      size_t row=(size_t)tr*N+a;
      real s=qdotx(k,pi,Q->w_in,off+row,x,xq,xs,d,gs)+P->bias[off+row];
      tp->anode[ab+ai]=row;tp->as[ab+ai]=s;a=2*a+1+(s>0);
     }
    }
   }
  }
  if(cpu_fast){size_t base=(size_t)t*T*(D+1);int count=T*(D+1);
   gelu_array(tp->s+base,tp->gg+base,tp->dg+base,count,!w->no_alt);
   for(int j=0;j<count;j++)if(tp->node[base+j]>=0)qaxpy(k,po,Q->w_out,off+tp->node[base+j],w->y,tp->gg[base+j],d,gs);
  }
  if(c->fff_gate)fff_gate_forward(c,Q,l,x,w->y,tp->fg_raw+(size_t)t*d,tp->fg_gate+(size_t)t*d,tp->fg_low+(size_t)t*c->n,tp->fg_gg+(size_t)t*c->n,tp->fg_dg+(size_t)t*c->n,!w->no_alt,k);
  for(int i=0;i<d;i++)xx[i]+=w->y[i];
 }
}
static void fff_bwd(const Cfg*c,const Params*P,const Params*Q,Params*G,int l,int L,real*g,unsigned char*hit,const Tape*tp,Work*w){
 const int d=c->d,T=c->T,D=c->D+1,N=fff_nodes(c),gs=c->group,AL=D*(D+1)/2;
 const size_t off=node_off(c,l);const Codes*k=w->k;
 const Plane*pi=k?&k->w_in:NULL,*po=k?&k->w_out:NULL;
 for(int t=0;t<L;t++){
  real*gy=g+(size_t)t*d,*dx=w->dx;const real*x=tp->x+(size_t)t*d;
  memset(dx,0,d*sizeof(real));
  if(c->fff_gate){fff_gate_backward(c,Q,G,l,x,gy,tp->fg_raw+(size_t)t*d,tp->fg_gate+(size_t)t*d,tp->fg_low+(size_t)t*c->n,tp->fg_gg+(size_t)t*c->n,tp->fg_dg+(size_t)t*c->n,w->y,dx,k);gy=w->y;}
  for(int tr=0;tr<T;tr++){
   size_t vb=((size_t)t*T+tr)*(D+1),ab=((size_t)t*T+tr)*AL;
   for(int j=0;j<=D;j++){
    size_t v=vb+j;if(tp->node[v]<0){w->on[j]=w->ds[j]=0;continue;}size_t row=tp->node[v];real go=qdotf(k,po,Q->w_out,off+row,gy,d,gs);
    w->on[j]=tape_gelu(tp,v)*go;w->ds[j]=tape_dgelu(tp,v)*go;
   }
   int ai=0;
   for(int j=0;j<D;j++){
    real bw=tp->bw[vb+j];if(!bw){ai+=D-j;continue;}
    real on=0,alt=0;for(int jj=j+1;jj<=D;jj++,ai++){
     on+=w->on[jj];if(tp->anode[ab+ai]<0)continue;size_t row=tp->anode[ab+ai];alt+=gelu(tp->as[ab+ai])*qdotf(k,po,Q->w_out,off+row,gy,d,gs);
    }
    real s=tp->s[vb+j],sg=sigm(fabs(s)/c->temp);
    w->ds[j]+=bw*(s>0?1:-1)*(on-alt)*sg*(1-sg)/c->temp;
   }
   for(int j=0;j<=D;j++){
    size_t v=vb+j;if(tp->node[v]<0)continue;size_t row=tp->node[v];real ds=w->ds[j];hit[row]=1;
    axpy(G->w_out+(off+row)*d,tape_gelu(tp,v),gy,d);
    axpy(G->w_in+(off+row)*d,ds,x,d);G->bias[off+row]+=ds;
    qaxpy(k,pi,Q->w_in,off+row,dx,ds,d,gs);
   }
  }
  rmsnorm_bwd(tp->xin+(size_t)t*d,P->norm+(size_t)l*d,tp->rn[t],dx,g+(size_t)t*d,G->norm+(size_t)l*d,d);
 }
}

/* AVX-512 lanes span the contiguous 16-value dimension of each fast matrix.
 * No approximation to tanh, no BF16 tape, and no change to the delta rule. */
static int fw_fast=1;
static int credit_turbo=0;
/* Forward-credit experiment: preserve H between loss-driven corrections when 0. */
static int fw_associative_writes=1;
#if FAST
static inline void fw_state16(real*H,const real*B,const real*C,const real*V,
                              real*R,int n,real decay,real eta){
    __m512 read=_mm512_setzero_ps(),pred=_mm512_setzero_ps();
    for(int j=0;j<n;j++){
        __m512 h=_mm512_loadu_ps(H+j*16);
        read=_mm512_add_ps(read,_mm512_mul_ps(_mm512_set1_ps(C[j]),h));
        pred=_mm512_add_ps(pred,_mm512_mul_ps(_mm512_set1_ps(B[j]),h));
    }
    _mm512_storeu_ps(R,read);tanh_array(R,16);
    if(credit_turbo && eta==0 && decay==1)return;
    __m512 vd=_mm512_set1_ps(decay);
    __m512 err=_mm512_sub_ps(_mm512_loadu_ps(V),_mm512_mul_ps(vd,pred));
    for(int j=0;j<n;j++){
        __m512 h=_mm512_loadu_ps(H+j*16);
        _mm512_storeu_ps(H+j*16,_mm512_add_ps(_mm512_mul_ps(vd,h),_mm512_mul_ps(_mm512_set1_ps(eta*B[j]),err)));
    }
}
static inline real fw_adjoint16(real*A,const real*H,const real*B,const real*C,
    const real*V,const real*R,const real*dR,real*dB,real*dC,real*dV,real*deta,
    int n,real decay,real eta){
    __m512 mk=_mm512_setzero_ps(),ak=_mm512_setzero_ps();
    for(int j=0;j<n;j++){
        __m512 b=_mm512_set1_ps(B[j]);
        mk=_mm512_add_ps(mk,_mm512_mul_ps(b,_mm512_loadu_ps(H+j*16)));
        ak=_mm512_add_ps(ak,_mm512_mul_ps(b,_mm512_loadu_ps(A+j*16)));
    }
    __m512 vd=_mm512_set1_ps(decay),ve=_mm512_set1_ps(eta);
    __m512 err=_mm512_sub_ps(_mm512_loadu_ps(V),_mm512_mul_ps(vd,mk));
    __m512 de=_mm512_mul_ps(ve,ak),r=_mm512_loadu_ps(R);
    __m512 dr=_mm512_mul_ps(_mm512_loadu_ps(dR),_mm512_sub_ps(_mm512_set1_ps(1),_mm512_mul_ps(r,r)));
    _mm512_storeu_ps(dV,_mm512_add_ps(_mm512_loadu_ps(dV),de));
    *deta+=_mm512_reduce_add_ps(_mm512_mul_ps(ak,err));
    __m512 dl=_mm512_setzero_ps();
    for(int j=0;j<n;j++){
        __m512 h=_mm512_loadu_ps(H+j*16),a=_mm512_loadu_ps(A+j*16);
        __m512 inner=_mm512_sub_ps(a,_mm512_mul_ps(_mm512_set1_ps(B[j]),de));
        dB[j]+=_mm512_reduce_add_ps(_mm512_sub_ps(_mm512_mul_ps(_mm512_mul_ps(ve,a),err),_mm512_mul_ps(_mm512_mul_ps(vd,h),de)));
        dC[j]+=_mm512_reduce_add_ps(_mm512_mul_ps(h,dr));
        dl=_mm512_add_ps(dl,_mm512_mul_ps(h,inner));
        _mm512_storeu_ps(A+j*16,_mm512_add_ps(_mm512_mul_ps(vd,inner),_mm512_mul_ps(_mm512_set1_ps(C[j]),dr)));
    }
    return _mm512_reduce_add_ps(dl);
}
#endif

static void fw_fwd(const Cfg *c,const Params *P,const Params *Q,int l,int L,real *xres,real *h,Tape *tp,Work *w) {


    const int d=c->d,T=c->T,D=c->D,n=c->n,pv=c->p,K=c->K,Pd=PDIM(c),N=NODES(c),g=c->group;
    const real *norm=P->norm+(size_t)l*d,*conv=P->conv+(size_t)l*K*d;
    const real *bias=P->bias+node_off(c,l),*A_log=P->A_log+node_off(c,l),*dt_bias=P->dt_bias+(size_t)l*T;
    const size_t r_node=node_off(c,l),r_proj=(size_t)l*Pd,r_U=(size_t)l*T*pv;
    const Codes *k=w->k;
    const Plane *pin=k?&k->w_in:NULL,*pout=k?&k->w_out:NULL,*pproj=k?&k->proj:NULL,*pU=k?&k->U:NULL;
    const int AL=D*(D+1)/2;
#if FAST
    const int vec=fw_fast&&pv==16;
#endif
    real *mx=w->mx,*y=w->y,*Racc=w->Racc;
    // The layer's input sequence is already available. No recurrent outputs
    // enter this preprocessing; only the subsequent tree walk is sequential.
    if(credit_turbo && l==0 && L==1 && w->reuse_fw_input){
        const Tape*a=w->reuse_fw_input;
        memcpy(tp->xin,a->xin,d*sizeof(real));memcpy(tp->u,a->u,d*sizeof(real));tp->rn[0]=a->rn[0];
        memcpy(tp->x,a->x,d*sizeof(real));memcpy(tp->xq,a->xq,d);tp->xs[0]=a->xs[0];
        memcpy(tp->p,a->p,Pd*sizeof(real));memcpy(tp->fw_raw,a->fw_raw,Pd*sizeof(real));
    }else {
    memcpy(tp->xin,xres,sizeof(real)*(size_t)L*d);
    for(int t=0;t<L;t++)rmsnorm_fwd(xres+(size_t)t*d,norm,tp->u+(size_t)t*d,d,tp->rn+t);
    for(int t=0;t<L;t++) {
        const real *ut=tp->u+(size_t)t*d;
        for(int i=0;i<d;i++)mx[i]=conv[i]*ut[i];
        for(int j=1;j<K;j++) {
            const real *past=t>=j?tp->u+(size_t)(t-j)*d:(w->stream_hist?w->stream_hist+(size_t)w->lv*(K-1)*d+(size_t)(j-t-1)*d:w->carry_hist?w->carry_hist+(size_t)w->lv*(K-1)*d+(size_t)(j-t-1)*d:NULL);
            if(past)for(int i=0;i<d;i++)mx[i]+=conv[j*d+i]*past[i];
        }
        quant8(mx,tp->x+(size_t)t*d,tp->xq+(size_t)t*d,tp->xs+t,d);
    }
    qrows_dot8_batch(k,pproj,Q->proj,r_proj,tp->x,tp->xq,tp->xs,tp->p,L,Pd,d,g);
    /* Nonlinear feature layer is slow; each routed node's output weights
     * are fast. Cache nonlinear features for BPTT (raw preactivations
     * in reference mode); the normalized feature itself is unchanged. */
    memcpy(tp->fw_raw,tp->p,(size_t)L*Pd*sizeof(real));
    for(int t=0;t<L;t++){
        real*pt=tp->p+(size_t)t*Pd;
        for(int part=0;part<2;part++){
            real*z=pt+part*n;tanh_array(z,n);
            if(fw_fast)memcpy(tp->fw_raw+(size_t)t*Pd+part*n,z,n*sizeof(real));
            real inv=1/sqrt(dotr(z,z,n)+(real)1e-6);
            for(int j=0;j<n;j++)z[j]*=inv;
        }
        tanh_array(pt+2*n,pv);
    }
    }
    for(int t=0;t<L;t++) {
        real *xt=tp->x+(size_t)t*d,*ut=tp->u+(size_t)t*d,*pt=tp->p+(size_t)t*Pd,*dtt=tp->dt+(size_t)t*T;
        int8_t *xq=tp->xq+(size_t)t*d;
        const real xs=tp->xs[t];y=w->ys+(size_t)t*d;
        const real *B=pt,*C=pt+n,*V=pt+2*n,*Z=pt+2*n+pv;
        for(int tr=0;tr<T;tr++)dtt[tr]=PROTECTED_FW?fmax(0,fmin(1,(pt[2*n+2*pv+tr]+dt_bias[tr]+1)*(real).5)):sigm(pt[2*n+2*pv+tr]+dt_bias[tr]);
        memset(y,0,sizeof(real)*d);
        memset(Racc,0,sizeof(real)*T*pv);
        int *node=w->walk;
        for(int tr=0;tr<T;tr++)node[tr]=0;
        for(int kk=0;kk<=D;kk++)for(int tr=0;tr<T;tr++) {
            const size_t row=(size_t)tr*N+node[tr],v=((size_t)t*T+tr)*(D+1)+kk;
            real *ha=h+row*n*pv,*R=tp->R+v*pv;
            if(w->prepare_state)w->prepare_state(w->prepare_state_user,ha,r_node+row,n,pv);
            const real keep=fw_fast?(real)(1/(1+decay_rate(k,r_node+row,A_log[row]))):sigm(-A_log[row]),eta=fw_associative_writes?dtt[tr]:0,dec=PROTECTED_FW?1-eta*(real).1*(1-keep):(real).9+(real).1*keep;
            if(!fw_fast||!w->no_alt)memcpy(tp->hprev+v*n*pv,ha,sizeof(real)*n*pv);
#if FAST
            if(vec)fw_state16(ha,B,C,V,R,n,dec,eta);else
#endif
            {
            real pred[pv];
            for(int q=0;q<pv;q++){
                real read=0,write=0;
                for(int j=0;j<n;j++){read+=C[j]*ha[j*pv+q];write+=B[j]*ha[j*pv+q];}
                R[q]=tanh(read);pred[q]=dec*write;
            }
            for(int j=0;j<n;j++)for(int q=0;q<pv;q++)
                ha[j*pv+q]=dec*ha[j*pv+q]+eta*B[j]*(V[q]-pred[q]);
            }
            const real s=qdotx(k,pin,Q->w_in,r_node+row,xt,xq,xs,d,g)+bias[row]+dotr(R,Z,pv);
            qaxpy(k,pout,Q->w_out,r_node+row,y,record_gelu(s,tp,v,!w->no_alt),d,g);
            for(int q=0;q<pv;q++)Racc[tr*pv+q]+=R[q];
            tp->node[v]=(int)row;
            tp->s[v]=s;
            tp->decay[v]=dec;
            node[tr]=2*node[tr]+1+(s>0);
            if(kk<D) {
                const size_t nr=(size_t)tr*N+node[tr];
                const char *hs=(const char *)(h+nr*n*pv);
                for(int o=0;o<n*pv*(int)sizeof(real);o+=64)__builtin_prefetch(hs+o);
#if FAST
                if(k&&k->ok)__builtin_prefetch(pin->bits+(r_node+nr)*(d/32));
#endif
            }
        }
        memcpy(w->rs+(size_t)t*T*pv,Racc,sizeof(real)*T*pv);
        int *cn=w->walk;
        for(int tr=0;tr<T&&!w->no_alt;tr++) {
            const size_t vb=((size_t)t*T+tr)*(D+1);
            real tot=0;
            for(int kk=0;kk<D;kk++) {
                const size_t v=vb+kk;
                cn[tr*(D+1)+kk]=2*(tp->node[v]-tr*N)+2-(tp->s[v]>0);
                const real sg=sigm(fabs(tp->s[v])/c->temp);
                tp->bw[v]=sg*(1-sg);
                tot+=tp->bw[v];
            }
            for(int kk=0;kk<D;kk++) {
                const size_t v=vb+kk;
                if(c->walks<=0) {
                    tp->bw[v]=1;
                    continue;
                }const real q=tot>0?fmin(1,c->walks*tp->bw[v]/tot):1;
                tp->bw[v]=urand(&w->rng)<q?1/q:0;
            }
        }
        for(int j=1;j<=D&&!w->no_alt;j++)for(int tr=0;tr<T;tr++) {
            const size_t ab=((size_t)t*T+tr)*AL;
            for(int kk=0,base=0;kk<j;base+=D-kk,kk++) {
                if(tp->bw[((size_t)t*T+tr)*(D+1)+kk]==0)continue;
                const size_t ai=ab+base+(j-kk-1),row=(size_t)tr*N+cn[tr*(D+1)+kk];
                const real dec=1,*hr=h+row*n*pv;
                real *R=tp->aR+ai*pv;
#if FAST
                if(vec) {
                    __m512 vR=_mm512_setzero_ps();
                    for(int jj=0;jj<n;jj++)vR=_mm512_fmadd_ps(_mm512_set1_ps(C[jj]),_mm512_loadu_ps(hr+jj*16),vR);
                    _mm512_storeu_ps(R,_mm512_mul_ps(vR,_mm512_set1_ps(dec)));
                }else
#endif
               {
                    for(int q=0;q<pv;q++)R[q]=0;
                    for(int jj=0;jj<n;jj++)for(int q=0;q<pv;q++)R[q]+=C[jj]*dec*hr[jj*pv+q];
                }
                tanh_array(R,pv);
                const real s=qdotx(k,pin,Q->w_in,r_node+row,xt,xq,xs,d,g)+bias[row]+dotr(R,Z,pv);
                tp->anode[ai]=(int)row;
                tp->as[ai]=s;
                const int nx=2*cn[tr*(D+1)+kk]+1+(s>0);
                cn[tr*(D+1)+kk]=nx;
                if(j<D) {
                    const size_t nr=(size_t)tr*N+nx;
                    const char *hs=(const char *)(h+nr*n*pv);
                    for(int o=0;o<n*pv*(int)sizeof(real);o+=64)__builtin_prefetch(hs+o);
#if FAST
                    if(k&&k->ok)__builtin_prefetch(pin->bits+(r_node+nr)*(d/32));
#endif
                }
            }
        }
    }
    qrows_axpy_batch(k,pU,Q->U,r_U,w->rs,w->ys,L,T*pv,d,g);
    for(size_t i=0;i<(size_t)L*d;i++)xres[i]+=w->ys[i];
    if(w->stream_hist&&K>1) {
        if(L!=1)abort();
        real *hb=w->stream_hist+(size_t)w->lv*(K-1)*d;
        memmove(hb+d,hb,sizeof(real)*(size_t)(K-2)*d);
        memcpy(hb,tp->u,sizeof(real)*d);
    }
}
static void fw_bwd(const Cfg *c,const Params *P,const Params *Q,Params *G,int l,int L,real *g,real *lam,unsigned char *hit,const Tape *tp,Work *w) {


    const int d=c->d,T=c->T,D=c->D,n=c->n,pv=c->p,K=c->K,Pd=PDIM(c),N=NODES(c),gs=c->group,AL=D*(D+1)/2;
    const real *norm=P->norm+(size_t)l*d,*conv=P->conv+(size_t)l*K*d,*A_log=P->A_log+node_off(c,l),*dt_bias=P->dt_bias+(size_t)l*T;
    const size_t r_node=node_off(c,l),r_proj=(size_t)l*Pd,r_U=(size_t)l*T*pv;
    real *gnorm=G->norm+(size_t)l*d,*gconv=G->conv+(size_t)l*K*d,*gw_in=G->w_in+r_node*d,*gw_out=G->w_out+r_node*d,*gbias=G->bias+r_node,*gA=G->A_log+r_node;
    real *gproj=G->proj+r_proj*d,*gdtb=G->dt_bias+(size_t)l*T,*gU=G->U+r_U*d;
    const Codes *k=w->k;
    const Plane *pin=k?&k->w_in:NULL,*pout=k?&k->w_out:NULL,*pproj=k?&k->proj:NULL,*pU=k?&k->U:NULL;
#if FAST
    const int vec=fw_fast&&pv==16;
#endif
    const size_t V=(size_t)L*T*(D+1);
    real *ds=w->ds,*dR=w->dR,*dp=w->dp,*dx=w->dx,*gUr=w->gUr,*on=w->on,*Racc=w->Racc;
    memset(ds,0,sizeof(real)*V);
    memset(dR,0,sizeof(real)*V*pv);
    memset(dp,0,sizeof(real)*(size_t)L*Pd);
    memset(dx,0,sizeof(real)*(size_t)L*d);
    // Readout sums/adjoints and weight gradients are independent of state BPTT.
    for(int t=0;t<L;t++)for(int tr=0;tr<T;tr++) {
        real *rs=w->rs+((size_t)t*T+tr)*pv;
        memset(rs,0,sizeof(real)*pv);
        for(int kk=0;kk<=D;kk++)for(int q=0;q<pv;q++)rs[q]+=tp->R[(((size_t)t*T+tr)*(D+1)+kk)*pv+q];
    }
    qrows_dotf_batch(k,pU,Q->U,r_U,g,w->drs,L,T*pv,d,gs);
    outer_rows(gU,w->rs,g,L,T*pv,d);
    // A: local output and branch gradients.
    for(int t=0;t<L;t++) {
        const real *gy=g+(size_t)t*d,*pt=tp->p+(size_t)t*Pd,*Z=pt+2*n+pv;
        real *dZ=dp+(size_t)t*Pd+2*n+pv,*dxt=dx+(size_t)t*d;
        for(int tr=0;tr<T;tr++) {
            const size_t vb=((size_t)t*T+tr)*(D+1),ab=((size_t)t*T+tr)*AL;
            gUr=w->drs+((size_t)t*T+tr)*pv;
            for(int kk=0;kk<=D;kk++) {
                const size_t v=vb+kk,row=tp->node[v];
                const real s=tp->s[v],go=qdotf(k,pout,Q->w_out,r_node+row,gy,d,gs);
                on[kk]=tape_gelu(tp,v)*go+dotr(gUr,tp->R+v*pv,pv);
                ds[v]+=tape_dgelu(tp,v)*go;
                for(int q=0;q<pv;q++)dR[v*pv+q]+=gUr[q];
                hit[row]=1;
            }
            int ai=0;
            for(int kk=0;kk<D;kk++) {
                const real bwk=tp->bw[vb+kk];
                if(bwk==0) {
                    ai+=D-kk;
                    continue;
                }real onk=0,alt=0;
                for(int j=kk+1;j<=D;j++)onk+=on[j];
                for(int j=kk+1;j<=D;j++,ai++) {
                    const size_t a=ab+ai,row=tp->anode[a];
                    alt+=gelu(tp->as[a])*qdotf(k,pout,Q->w_out,r_node+row,gy,d,gs)+dotr(gUr,tp->aR+a*pv,pv);
                }
                const real s=tp->s[vb+kk],sg=sigm(fabs(s)/c->temp);
                ds[vb+kk]+=bwk*(s>0?1:-1)*(onk-alt)*sg*(1-sg)/c->temp;
            }
            for(int kk=0;kk<=D;kk++) {
                const size_t v=vb+kk,row=tp->node[v];
                const real dsv=ds[v];
                for(int q=0;q<pv;q++) {
                    dZ[q]+=dsv*tp->R[v*pv+q];
                    dR[v*pv+q]+=dsv*Z[q];
                }qaxpy(k,pin,Q->w_in,r_node+row,dxt,dsv,d,gs);
                gbias[row]+=dsv;
            }
        }
    }
    // A': stable counting sort of visits by node; cnt stores row ends.
   {
        const int TN=T*N,VD=T*(D+1);
        int *cnt=w->rcnt,*ord=w->rord;
        memset(cnt,0,sizeof(int)*(TN+1));
        for(size_t v=0;v<V;v++)cnt[tp->node[v]+1]++;
        for(int r=0;r<TN;r++)cnt[r+1]+=cnt[r];
        for(size_t v=0;v<V;v++)ord[cnt[tp->node[v]]++]=(int)v;
        for(int r=0,start=0;r<TN;start=cnt[r],r++) {
            if(start==cnt[r])continue;
            real *go=gw_out+(size_t)r*d,*gi=gw_in+(size_t)r*d;
            for(int e=start;e<cnt[r];e++) {
                const int v=ord[e],t=v/VD;
                axpy(go,tape_gelu(tp,v),g+(size_t)t*d,d);
                axpy(gi,ds[v],tp->x+(size_t)t*d,d);
            }
        }
    }
    // Exact adjoint of error-correcting fast weights, reverse time.
    for(int t=L-1;t>=0;t--){
        const real*pt=tp->p+(size_t)t*Pd,*B=pt,*C=pt+n,*Vv=pt+2*n;
        real*dpt=dp+(size_t)t*Pd,*dB=dpt,*dC=dpt+n,*dV=dpt+2*n,*deta=dpt+2*n+2*pv;
        for(int tr=0;tr<T;tr++){
            real eta=tp->dt[(size_t)t*T+tr];
            for(int kk=0;kk<=D;kk++){
                size_t v=((size_t)t*T+tr)*(D+1)+kk,row=tp->node[v];
                real*la=lam+row*n*pv;const real*hp=tp->hprev+v*n*pv;
                real keep=fw_fast?(real)(1/(1+decay_rate(k,r_node+row,A_log[row]))):sigm(-A_log[row]),dec=tp->decay[v],ddec=0;
                #if FAST
                if(vec)ddec=fw_adjoint16(la,hp,B,C,Vv,tp->R+v*pv,dR+v*pv,dB,dC,dV,deta+tr,n,dec,eta);
                else
#endif
                {
                real err[pv],derr[pv],dr[pv];
                for(int q=0;q<pv;q++){
                    real mk=0,ak=0;for(int j=0;j<n;j++){mk+=hp[j*pv+q]*B[j];ak+=la[j*pv+q]*B[j];}
                    err[q]=Vv[q]-dec*mk;derr[q]=eta*ak;
                    dr[q]=dR[v*pv+q]*(1-tp->R[v*pv+q]*tp->R[v*pv+q]);
                    dV[q]+=derr[q];deta[tr]+=ak*err[q];
                }
                for(int j=0;j<n;j++)for(int q=0;q<pv;q++){
                    real old=hp[j*pv+q],adj=la[j*pv+q];
                    dB[j]+=eta*adj*err[q]-dec*old*derr[q];
                    dC[j]+=old*dr[q];ddec+=old*(adj-B[j]*derr[q]);
                    la[j*pv+q]=dec*(adj-B[j]*derr[q])+C[j]*dr[q];
                }
                }
                gA[row]-=ddec*(real).1*keep*(1-keep)*(PROTECTED_FW?eta:1);
                if(PROTECTED_FW)deta[tr]-=ddec*(real).1*(1-keep);
            }
        }
    }
    // C: dense projections, convolution, norm.
    real *du=w->du;
    memset(du,0,sizeof(real)*(size_t)L*d);
    for(int t=0;t<L;t++) {
        const real *pt=tp->p+(size_t)t*Pd,*xt=tp->x+(size_t)t*d;
        real *dpt=dp+(size_t)t*Pd,*dxt=dx+(size_t)t*d;
        for(int tr=0;tr<T;tr++) {
            const real z=pt[2*n+2*pv+tr]+dt_bias[tr];
            const real eta=fw_fast?tp->dt[(size_t)t*T+tr]:sigm(z),dz=dpt[2*n+2*pv+tr]*(PROTECTED_FW?((z>-1&&z<1)?(real).5:0):eta*(1-eta));
            dpt[2*n+2*pv+tr]=dz;
            gdtb[tr]+=dz;
        }
    }
    for(int t=0;t<L;t++){
        const real*pt=tp->p+(size_t)t*Pd,*raw=tp->fw_raw+(size_t)t*Pd;real*dpt=dp+(size_t)t*Pd;
        for(int part=0;part<2;part++){
            real sq=(real)1e-6,dot=dotr(dpt+part*n,pt+part*n,n);
            for(int j=0;j<n;j++){real u=fw_fast?raw[part*n+j]:tanh(raw[part*n+j]);sq+=u*u;}
            real inv=1/sqrt(sq);
            for(int j=0;j<n;j++){int ix=part*n+j;real u=fw_fast?raw[ix]:tanh(raw[ix]);dpt[ix]=inv*(dpt[ix]-pt[ix]*dot)*(1-u*u);}
        }
        for(int q=0;q<pv;q++){real v=pt[2*n+q];dpt[2*n+q]*=1-v*v;}
    }
    outer_rows(gproj,dp,tp->x,L,Pd,d);
    qrows_axpy_batch(k,pproj,Q->proj,r_proj,dp,dx,L,Pd,d,gs);
    for(int t=0;t<L;t++) {
        real *dxt=dx+(size_t)t*d;
        for(int j=0;j<K;j++)if(t-j>=0)for(int i=0;i<d;i++) {
            gconv[j*d+i]+=dxt[i]*tp->u[(size_t)(t-j)*d+i];
            du[(size_t)(t-j)*d+i]+=conv[j*d+i]*dxt[i];
        }
    }
    for(int t=0;t<L;t++)rmsnorm_bwd(tp->xin+(size_t)t*d,norm,tp->rn[t],du+(size_t)t*d,g+(size_t)t*d,gnorm,d);
}
static void layer_fwd(const Cfg *c,const Params *P,const Params *Q,int l,int L,real *xres,real *h,Tape *tp,Work *w) {
    w->lv=l;l%=c->n_layer;
    if(l&1){fff_fwd(c,P,Q,l,L,xres,tp,w);return;}
    if((c->fw_mask>>l)&1){fw_fwd(c,P,Q,l,L,xres,h,tp,w);return;}

    const int d=c->d,T=c->T,D=c->D,n=c->n,pv=c->p,K=c->K,Pd=PDIM(c),N=NODES(c),g=c->group;
    const real *norm=P->norm+(size_t)l*d,*conv=P->conv+(size_t)l*K*d;
    const real *bias=P->bias+node_off(c,l),*A_log=P->A_log+node_off(c,l),*dt_bias=P->dt_bias+(size_t)l*T;
    const size_t r_node=node_off(c,l),r_proj=(size_t)l*Pd,r_U=(size_t)l*T*pv;
    const Codes *k=w->k;
    const Plane *pin=k?&k->w_in:NULL,*pout=k?&k->w_out:NULL,*pproj=k?&k->proj:NULL,*pU=k?&k->U:NULL;
    const int AL=D*(D+1)/2;
#if FAST
    const int vec=k&&k->ok&&pv==16;
    uint16_t *hp16=w->hp16+(size_t)w->lv*w->Lmax*T*(D+1)*n*pv;
    const int bf=vec&&w->tape_bf16;
#endif
    real *mx=w->mx,*y=w->y,*Racc=w->Racc;
    // The layer's input sequence is already available. No recurrent outputs
    // enter this preprocessing; only the subsequent tree walk is sequential.
    memcpy(tp->xin,xres,sizeof(real)*(size_t)L*d);
    for(int t=0;t<L;t++)rmsnorm_fwd(xres+(size_t)t*d,norm,tp->u+(size_t)t*d,d,tp->rn+t);
    for(int t=0;t<L;t++) {
        const real *ut=tp->u+(size_t)t*d;
        for(int i=0;i<d;i++)mx[i]=conv[i]*ut[i];
        for(int j=1;j<K;j++) {
            const real *past=t>=j?tp->u+(size_t)(t-j)*d:(w->stream_hist?w->stream_hist+(size_t)w->lv*(K-1)*d+(size_t)(j-t-1)*d:w->carry_hist?w->carry_hist+(size_t)w->lv*(K-1)*d+(size_t)(j-t-1)*d:NULL);
            if(past)for(int i=0;i<d;i++)mx[i]+=conv[j*d+i]*past[i];
        }
        quant8(mx,tp->x+(size_t)t*d,tp->xq+(size_t)t*d,tp->xs+t,d);
    }
    qrows_dot8_batch(k,pproj,Q->proj,r_proj,tp->x,tp->xq,tp->xs,tp->p,L,Pd,d,g);
    if(ssm_qknorm||ssm_rope)for(int t=0;t<L;t++)ssm_prep(tp->p+(size_t)t*Pd,n,ssm_groups,w->pos+t,w->qkn+((size_t)w->lv*w->Lmax+t)*2*16);
    for(int t=0;t<L;t++) {
        real *xt=tp->x+(size_t)t*d,*ut=tp->u+(size_t)t*d,*pt=tp->p+(size_t)t*Pd,*dtt=tp->dt+(size_t)t*T;
        int8_t *xq=tp->xq+(size_t)t*d;
        const real xs=tp->xs[t];y=w->ys+(size_t)t*d;
        const int G=ssm_groups;const real *B0=pt,*C0=pt+G*n,*V=pt+2*G*n,*Z=pt+2*G*n+pv;
        for(int tr=0;tr<T;tr++)dtt[tr]=softplus(pt[2*G*n+2*pv+tr]+dt_bias[tr]);
        memset(y,0,sizeof(real)*d);
        memset(Racc,0,sizeof(real)*T*pv);
        int *node=w->walk;
        for(int tr=0;tr<T;tr++)node[tr]=0;
        for(int kk=0;kk<=D;kk++)for(int tr=0;tr<T;tr++) {
            const size_t row=(size_t)tr*N+node[tr],v=((size_t)t*T+tr)*(D+1)+kk;
            const real*B=B0+(size_t)(tr*G/T)*n,*C=C0+(size_t)(tr*G/T)*n;
            real *ha=h+row*n*pv,*R=tp->R+v*pv;
            if(w->prepare_state)w->prepare_state(w->prepare_state_user,ha,r_node+row,n,pv);
            real dec,wdt;ssm_coef(decay_rate(k,r_node+row,A_log[row]),ssm_ltc==2?softplus(ltc2_gate(pt[2*G*n+2*pv+tr]+dt_bias[tr],Z,C,ha,n,pv)):dtt[tr],&dec,&wdt);if(dec>ssm_dec_cap)dec=ssm_dec_cap;
#if FAST
            if(!bf)
#endif
            if(!credit_turbo||!w->no_alt)memcpy(tp->hprev+v*n*pv,ha,sizeof(real)*n*pv);
#if FAST
            if(vec) {
                const __m512 vd=_mm512_set1_ps(dec),vV=_mm512_loadu_ps(V);
                __m512 vR=_mm512_setzero_ps();
                uint16_t *hb=hp16+v*n*pv;
                for(int j=0;j<n;j++) {
                    const __m512 h0=_mm512_loadu_ps(ha+j*16);
                    if(bf) {
                        const __m512i u=_mm512_castps_si512(h0);
                        const __m512i r=_mm512_add_epi32(u,_mm512_add_epi32(_mm512_set1_epi32(0x7fff),_mm512_and_si512(_mm512_srli_epi32(u,16),_mm512_set1_epi32(1))));
                        _mm256_storeu_si256((__m256i *)(hb+j*16),_mm512_cvtepi32_epi16(_mm512_srli_epi32(r,16)));
                    }
                    const __m512 hv=_mm512_mul_ps(h0,vd);
                    vR=_mm512_fmadd_ps(_mm512_set1_ps(C[j]),hv,vR);
                    _mm512_storeu_ps(ha+j*16,_mm512_fmadd_ps(_mm512_set1_ps(wdt*B[j]),vV,hv));
                }_mm512_storeu_ps(R,vR);
            }else
#endif
           {
                for(int q=0;q<pv;q++)R[q]=0;
                for(int j=0;j<n;j++)for(int q=0;q<pv;q++) {
                    ha[j*pv+q]*=dec;
                    R[q]+=C[j]*ha[j*pv+q];
                }for(int j=0;j<n;j++)for(int q=0;q<pv;q++)ha[j*pv+q]+=wdt*B[j]*V[q];
            }
            const real s=qdotx(k,pin,Q->w_in,r_node+row,xt,xq,xs,d,g)+bias[row]+dotr(R,Z,pv);
            qaxpy(k,pout,Q->w_out,r_node+row,y,record_gelu(s,tp,v,!w->no_alt),d,g);
            for(int q=0;q<pv;q++)Racc[tr*pv+q]+=R[q];
            tp->node[v]=(int)row;
            tp->s[v]=s;
            tp->decay[v]=dec;
            node[tr]=2*node[tr]+1+(s>0);
            if(kk<D) {
                const size_t nr=(size_t)tr*N+node[tr];
                const char *hs=(const char *)(h+nr*n*pv);
                for(int o=0;o<n*pv*(int)sizeof(real);o+=64)__builtin_prefetch(hs+o);
#if FAST
                if(k&&k->ok)__builtin_prefetch(pin->bits+(r_node+nr)*(d/32));
#endif
            }
        }
        memcpy(w->rs+(size_t)t*T*pv,Racc,sizeof(real)*T*pv);
        int *cn=w->walk;
        for(int tr=0;tr<T&&!w->no_alt;tr++) {
            const size_t vb=((size_t)t*T+tr)*(D+1);
            real tot=0;
            for(int kk=0;kk<D;kk++) {
                const size_t v=vb+kk;
                cn[tr*(D+1)+kk]=2*(tp->node[v]-tr*N)+2-(tp->s[v]>0);
                const real sg=sigm(fabs(tp->s[v])/c->temp);
                tp->bw[v]=sg*(1-sg);
                tot+=tp->bw[v];
            }
            for(int kk=0;kk<D;kk++) {
                const size_t v=vb+kk;
                if(c->walks<=0) {
                    tp->bw[v]=1;
                    continue;
                }const real q=tot>0?fmin(1,c->walks*tp->bw[v]/tot):1;
                tp->bw[v]=urand(&w->rng)<q?1/q:0;
            }
        }
        for(int j=1;j<=D&&!w->no_alt;j++)for(int tr=0;tr<T;tr++) {
            const size_t ab=((size_t)t*T+tr)*AL;
            for(int kk=0,base=0;kk<j;base+=D-kk,kk++) {
                if(tp->bw[((size_t)t*T+tr)*(D+1)+kk]==0)continue;
                const size_t ai=ab+base+(j-kk-1),row=(size_t)tr*N+cn[tr*(D+1)+kk];
                const real*C=C0+(size_t)(tr*G/T)*n;
                const real*hr=h+row*n*pv;real dec,gn_;ssm_coef(decay_rate(k,r_node+row,A_log[row]),ssm_ltc==2?softplus(ltc2_gate(pt[2*G*n+2*pv+tr]+dt_bias[tr],Z,C,hr,n,pv)):dtt[tr],&dec,&gn_);
                real *R=tp->aR+ai*pv;
#if FAST
                if(vec) {
                    __m512 vR=_mm512_setzero_ps();
                    for(int jj=0;jj<n;jj++)vR=_mm512_fmadd_ps(_mm512_set1_ps(C[jj]),_mm512_loadu_ps(hr+jj*16),vR);
                    _mm512_storeu_ps(R,_mm512_mul_ps(vR,_mm512_set1_ps(dec)));
                }else
#endif
               {
                    for(int q=0;q<pv;q++)R[q]=0;
                    for(int jj=0;jj<n;jj++)for(int q=0;q<pv;q++)R[q]+=C[jj]*dec*hr[jj*pv+q];
                }
                const real s=qdotx(k,pin,Q->w_in,r_node+row,xt,xq,xs,d,g)+bias[row]+dotr(R,Z,pv);
                tp->anode[ai]=(int)row;
                tp->as[ai]=s;
                const int nx=2*cn[tr*(D+1)+kk]+1+(s>0);
                cn[tr*(D+1)+kk]=nx;
                if(j<D) {
                    const size_t nr=(size_t)tr*N+nx;
                    const char *hs=(const char *)(h+nr*n*pv);
                    for(int o=0;o<n*pv*(int)sizeof(real);o+=64)__builtin_prefetch(hs+o);
#if FAST
                    if(k&&k->ok)__builtin_prefetch(pin->bits+(r_node+nr)*(d/32));
#endif
                }
            }
        }
    }
    qrows_axpy_batch(k,pU,Q->U,r_U,w->rs,w->ys,L,T*pv,d,g);
    for(size_t i=0;i<(size_t)L*d;i++)xres[i]+=w->ys[i];
    if(w->stream_hist&&K>1) {
        if(L!=1)abort();
        real *hb=w->stream_hist+(size_t)w->lv*(K-1)*d;
        memmove(hb+d,hb,sizeof(real)*(size_t)(K-2)*d);
        memcpy(hb,tp->u,sizeof(real)*d);
    }
}
static void layer_bwd(const Cfg *c,const Params *P,const Params *Q,Params *G,int l,int L,real *g,real *lam,unsigned char *hit,const Tape *tp,Work *w) {
    w->lv=l;l%=c->n_layer;
    if(l&1){fff_bwd(c,P,Q,G,l,L,g,hit,tp,w);return;}
    if((c->fw_mask>>l)&1){fw_bwd(c,P,Q,G,l,L,g,lam,hit,tp,w);return;}

    const int d=c->d,T=c->T,D=c->D,n=c->n,pv=c->p,K=c->K,Pd=PDIM(c),N=NODES(c),gs=c->group,AL=D*(D+1)/2;
    const real *norm=P->norm+(size_t)l*d,*conv=P->conv+(size_t)l*K*d,*A_log=P->A_log+node_off(c,l),*dt_bias=P->dt_bias+(size_t)l*T;
    const size_t r_node=node_off(c,l),r_proj=(size_t)l*Pd,r_U=(size_t)l*T*pv;
    real *gnorm=G->norm+(size_t)l*d,*gconv=G->conv+(size_t)l*K*d,*gw_in=G->w_in+r_node*d,*gw_out=G->w_out+r_node*d,*gbias=G->bias+r_node,*gA=G->A_log+r_node;
    real *gproj=G->proj+r_proj*d,*gdtb=G->dt_bias+(size_t)l*T,*gU=G->U+r_U*d;
    const Codes *k=w->k;
    const Plane *pin=k?&k->w_in:NULL,*pout=k?&k->w_out:NULL,*pproj=k?&k->proj:NULL,*pU=k?&k->U:NULL;
#if FAST
    const int vec=k&&k->ok&&pv==16;
    const uint16_t *hp16=w->hp16+(size_t)w->lv*w->Lmax*T*(D+1)*n*pv;
    const int bf=vec&&w->tape_bf16;
#endif
    const size_t V=(size_t)L*T*(D+1);
    real *ds=w->ds,*dR=w->dR,*dp=w->dp,*dx=w->dx,*gUr=w->gUr,*on=w->on,*Racc=w->Racc;
    memset(ds,0,sizeof(real)*V);
    memset(dR,0,sizeof(real)*V*pv);
    memset(dp,0,sizeof(real)*(size_t)L*Pd);
    memset(dx,0,sizeof(real)*(size_t)L*d);
    // Readout sums/adjoints and weight gradients are independent of state BPTT.
    for(int t=0;t<L;t++)for(int tr=0;tr<T;tr++) {
        real *rs=w->rs+((size_t)t*T+tr)*pv;
        memset(rs,0,sizeof(real)*pv);
        for(int kk=0;kk<=D;kk++)for(int q=0;q<pv;q++)rs[q]+=tp->R[(((size_t)t*T+tr)*(D+1)+kk)*pv+q];
    }
    qrows_dotf_batch(k,pU,Q->U,r_U,g,w->drs,L,T*pv,d,gs);
    outer_rows(gU,w->rs,g,L,T*pv,d);
    // A: local output and branch gradients.
    for(int t=0;t<L;t++) {
        const real *gy=g+(size_t)t*d,*pt=tp->p+(size_t)t*Pd,*Z=pt+2*ssm_groups*n+pv;
        real *dZ=dp+(size_t)t*Pd+2*ssm_groups*n+pv,*dxt=dx+(size_t)t*d;
        for(int tr=0;tr<T;tr++) {
            const size_t vb=((size_t)t*T+tr)*(D+1),ab=((size_t)t*T+tr)*AL;
            gUr=w->drs+((size_t)t*T+tr)*pv;
            for(int kk=0;kk<=D;kk++) {
                const size_t v=vb+kk,row=tp->node[v];
                const real s=tp->s[v],go=qdotf(k,pout,Q->w_out,r_node+row,gy,d,gs);
                on[kk]=tape_gelu(tp,v)*go+dotr(gUr,tp->R+v*pv,pv);
                ds[v]+=tape_dgelu(tp,v)*go;
                for(int q=0;q<pv;q++)dR[v*pv+q]+=gUr[q];
                hit[row]=1;
            }
            int ai=0;
            for(int kk=0;kk<D;kk++) {
                const real bwk=tp->bw[vb+kk];
                if(bwk==0) {
                    ai+=D-kk;
                    continue;
                }real onk=0,alt=0;
                for(int j=kk+1;j<=D;j++)onk+=on[j];
                for(int j=kk+1;j<=D;j++,ai++) {
                    const size_t a=ab+ai,row=tp->anode[a];
                    alt+=gelu(tp->as[a])*qdotf(k,pout,Q->w_out,r_node+row,gy,d,gs)+dotr(gUr,tp->aR+a*pv,pv);
                }
                const real s=tp->s[vb+kk],sg=sigm(fabs(s)/c->temp);
                ds[vb+kk]+=bwk*(s>0?1:-1)*(onk-alt)*sg*(1-sg)/c->temp;
            }
            for(int kk=0;kk<=D;kk++) {
                const size_t v=vb+kk,row=tp->node[v];
                const real dsv=ds[v];
                for(int q=0;q<pv;q++) {
                    dZ[q]+=dsv*tp->R[v*pv+q];
                    dR[v*pv+q]+=dsv*Z[q];
                }qaxpy(k,pin,Q->w_in,r_node+row,dxt,dsv,d,gs);
                gbias[row]+=dsv;
            }
        }
    }
    // A': stable counting sort of visits by node; cnt stores row ends.
   {
        const int TN=T*N,VD=T*(D+1);
        int *cnt=w->rcnt,*ord=w->rord;
        memset(cnt,0,sizeof(int)*(TN+1));
        for(size_t v=0;v<V;v++)cnt[tp->node[v]+1]++;
        for(int r=0;r<TN;r++)cnt[r+1]+=cnt[r];
        for(size_t v=0;v<V;v++)ord[cnt[tp->node[v]]++]=(int)v;
        for(int r=0,start=0;r<TN;start=cnt[r],r++) {
            if(start==cnt[r])continue;
            real *go=gw_out+(size_t)r*d,*gi=gw_in+(size_t)r*d;
            for(int e=start;e<cnt[r];e++) {
                const int v=ord[e],t=v/VD;
                axpy(go,tape_gelu(tp,v),g+(size_t)t*d,d);
                axpy(gi,ds[v],tp->x+(size_t)t*d,d);
            }
        }
    }
    // B: recurrent state adjoints, reverse time.
    for(int t=L-1;t>=0;t--) {
        const int G=ssm_groups;const real *pt=tp->p+(size_t)t*Pd,*B0=pt,*C0=pt+G*n,*Vv=pt+2*G*n;
        real *dpt=dp+(size_t)t*Pd,*dB0=dpt,*dC0=dpt+G*n,*dV=dpt+2*G*n,*ddt=dpt+2*G*n+2*pv;
        for(int tr=0;tr<T;tr++) {
            const real dtv=tp->dt[(size_t)t*T+tr];
            const real*B=B0+(size_t)(tr*G/T)*n,*C=C0+(size_t)(tr*G/T)*n;real*dB=dB0+(size_t)(tr*G/T)*n,*dC=dC0+(size_t)(tr*G/T)*n;
            for(int kk=0;kk<=D;kk++) {
                const size_t v=((size_t)t*T+tr)*(D+1)+kk,row=tp->node[v];
                real *la=lam+row*n*pv;
                const real *hp=tp->hprev+v*n*pv,dec=tp->decay[v],*dRv=dR+v*pv;
                const real ea_=decay_rate(k,r_node+row,A_log[row]),fv=ssm_ltc==2?1/dec-1-ea_:dtv,gv=ssm_ltc?fv*dec:dtv;
                real ddec=0,dgain=0;
#if FAST
                if(vec) {
                    const __m512 vV=_mm512_loadu_ps(Vv),vdR=_mm512_loadu_ps(dRv),vd=_mm512_set1_ps(dec);
                    __m512 vBl=_mm512_setzero_ps(),vdec=_mm512_setzero_ps();
                    const uint16_t *hb=hp16+v*n*pv;
                    for(int j=0;j<n;j++) {
                        const __m512 lv=_mm512_loadu_ps(la+j*16);
                        const __m512 hv=bf?_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(hb+j*16))),16)):_mm512_loadu_ps(hp+j*16);
                        dB[j]+=gv*_mm512_reduce_add_ps(_mm512_mul_ps(lv,vV));
                        vBl=_mm512_fmadd_ps(_mm512_set1_ps(B[j]),lv,vBl);
                        dC[j]+=dec*_mm512_reduce_add_ps(_mm512_mul_ps(hv,vdR));
                        const __m512 dht=_mm512_fmadd_ps(_mm512_set1_ps(C[j]),vdR,lv);
                        vdec=_mm512_fmadd_ps(dht,hv,vdec);
                        _mm512_storeu_ps(la+j*16,_mm512_mul_ps(vd,dht));
                    } _mm512_storeu_ps(dV,_mm512_fmadd_ps(_mm512_set1_ps(gv),vBl,_mm512_loadu_ps(dV)));
                    dgain=_mm512_reduce_add_ps(_mm512_mul_ps(vBl,vV));
                    ddec=_mm512_reduce_add_ps(vdec);
                }else
#endif
               {
                    for(int j=0;j<n;j++)for(int q=0;q<pv;q++) {
                        const real lv=la[j*pv+q];
                        dB[j]+=gv*lv*Vv[q];
                        dV[q]+=gv*lv*B[j];
                        dgain+=lv*B[j]*Vv[q];
                        dC[j]+=dec*hp[j*pv+q]*dRv[q];
                        const real dht=lv+C[j]*dRv[q];
                        ddec+=dht*hp[j*pv+q];
                        la[j*pv+q]=dec*dht;
                    }
                }
                if(ssm_ltc){const real d2=dec*dec; /* dec=1/(1+ea+f), gain=f*dec */
                    const real df=dgain*(1+ea_)*d2-ddec*d2;gA[row]+=(-dgain*fv*d2-ddec*d2)*ea_;
                    if(ssm_ltc==2){ /* f=softplus(a), a=z+Z.(C^T H_prev): chain into z, Z, C and H_prev */
                        real hq[64*64];for(int i=0;i<n*pv;i++){
#if FAST
                            hq[i]=bf?_cvtsh_ss_bf(hp16[v*n*pv+i]):hp[i];
#else
                            hq[i]=hp[i];
#endif
                        }
                        const real*Zv=Vv+pv;real a=ltc2_gate(pt[2*ssm_groups*n+2*pv+tr]+dt_bias[tr],Zv,C,hq,n,pv),da=df*sigm(a);
                        real*dZ=dpt+2*G*n+pv;ddt[tr]+=da;
                        for(int q=0;q<pv;q++){real gq=0;for(int j=0;j<n;j++)gq+=C[j]*hq[j*pv+q];dZ[q]+=da*gq;const real dg=da*Zv[q];
                            for(int j=0;j<n;j++){dC[j]+=dg*hq[j*pv+q];la[j*pv+q]+=C[j]*dg;}}
                    }else ddt[tr]+=df;}
                else{ddt[tr]+=dgain;gA[row]+=ddec*dec*-ea_*dtv;ddt[tr]+=ddec*dec*-ea_;}
            }
        }
    }
    // C: dense projections, convolution, norm.
    real *du=w->du;
    memset(du,0,sizeof(real)*(size_t)L*d);
    for(int t=0;t<L;t++) {
        const real *pt=tp->p+(size_t)t*Pd,*xt=tp->x+(size_t)t*d;
        real *dpt=dp+(size_t)t*Pd,*dxt=dx+(size_t)t*d;
        for(int tr=0;tr<T;tr++) {
            const real z=pt[2*ssm_groups*n+2*pv+tr]+dt_bias[tr];
            const real dz=ssm_ltc==2?dpt[2*ssm_groups*n+2*pv+tr]:dpt[2*ssm_groups*n+2*pv+tr]*(z>20?1:sigm(z));
            dpt[2*ssm_groups*n+2*pv+tr]=dz;
            gdtb[tr]+=dz;
        }
    }
    if(ssm_qknorm||ssm_rope)for(int t=0;t<L;t++)ssm_prep_bwd(tp->p+(size_t)t*Pd,dp+(size_t)t*Pd,n,ssm_groups,w->pos+t,w->qkn+((size_t)w->lv*w->Lmax+t)*2*16);
    outer_rows(gproj,dp,tp->x,L,Pd,d);
    qrows_axpy_batch(k,pproj,Q->proj,r_proj,dp,dx,L,Pd,d,gs);
    for(int t=0;t<L;t++) {
        real *dxt=dx+(size_t)t*d;
        for(int j=0;j<K;j++)if(t-j>=0)for(int i=0;i<d;i++) {
            gconv[j*d+i]+=dxt[i]*tp->u[(size_t)(t-j)*d+i];
            du[(size_t)(t-j)*d+i]+=conv[j*d+i]*dxt[i];
        }
    }
    for(int t=0;t<L;t++)rmsnorm_bwd(tp->xin+(size_t)t*d,norm,tp->rn[t],du+(size_t)t*d,g+(size_t)t*d,gnorm,d);
}
Work *tss_work_new(const Cfg *c,int Lmax) {
    const int d=c->d,N=NODES(c),T=c->T,nl=c->n_layer*n_loops;
    const size_t tr=tape_reals(c,Lmax),ti=(size_t)Lmax*T*(c->D+2+(c->D+1)*(c->D+2)/2);
    const size_t V=(size_t)Lmax*T*(c->D+1),S=(size_t)T*N*c->n*c->p;
    Work *w=calloc(1,sizeof(Work));
    w->Lmax=Lmax;w->qkn=calloc((size_t)nl*Lmax*2*16,sizeof(real));
    w->ys=calloc((size_t)Lmax*d,sizeof(real));w->rs=calloc((size_t)Lmax*T*c->p,sizeof(real));w->drs=calloc((size_t)Lmax*T*c->p,sizeof(real));
    w->head=calloc((size_t)8*(2*d+c->vocab+1),sizeof(real));
    w->head_exp=calloc(c->vocab,sizeof(double));
    w->tapes=calloc(tr*nl,sizeof(real));
    w->itapes=calloc(ti*nl,sizeof(int));
    w->tp=calloc(nl,sizeof(Tape));
    w->x=calloc((size_t)Lmax*d,sizeof(real));
    w->g=calloc((size_t)Lmax*d,sizeof(real));
    w->h=calloc(S,sizeof(real));
    w->lam=calloc(S,sizeof(real));
    w->xn=calloc(d,sizeof(real));
    w->dxn=calloc(d,sizeof(real));
    w->lg=calloc(c->vocab,sizeof(real));
    w->ds=calloc(V,sizeof(real));
    w->dR=calloc(V*c->p,sizeof(real));
    w->dp=calloc((size_t)Lmax*PDIM(c),sizeof(real));
    w->dx=calloc((size_t)Lmax*d,sizeof(real));
    w->du=calloc((size_t)Lmax*d,sizeof(real));
    w->gUr=calloc(c->p,sizeof(real));
    w->on=calloc(c->D+2,sizeof(real));
    w->Racc=calloc((size_t)T*c->p,sizeof(real));
    w->walk=calloc((size_t)T*(c->D+1),sizeof(int));
    w->mx=calloc(d,sizeof(real));
    w->y=calloc(d,sizeof(real));
    w->xq8=calloc((size_t)nl*Lmax*d,1);
    w->rcnt=calloc((size_t)T*N+1,sizeof(int));
    w->rord=calloc((size_t)Lmax*T*(c->D+1),sizeof(int));
    w->hp16=calloc((size_t)nl*Lmax*T*(c->D+1)*c->n*c->p,sizeof(uint16_t));
    for(int l=0;l<nl;l++) {
        tape_carve(c,Lmax,w->tapes+tr*l,w->itapes+ti*l,w->tp+l);
        w->tp[l].xq=w->xq8+(size_t)l*Lmax*d;
    }return w;
}
void tss_work_free(Work *w) {
    free(w->tapes);
    free(w->itapes);
    free(w->tp);
    free(w->x);
    free(w->g);
    free(w->h);
    free(w->lam);
    free(w->xn);
    free(w->dxn);
    free(w->lg);
    free(w->ds);
    free(w->dR);
    free(w->dp);
    free(w->dx);
    free(w->du);
    free(w->gUr);
    free(w->on);
    free(w->Racc);
    free(w->mx);
    free(w->y);
    free(w->xq8);
    free(w->hp16);
    free(w->rcnt);
    free(w->rord);
    free(w->walk);
    free(w->ys);free(w->rs);free(w->drs);free(w->head);
    free(w->head_exp);
    free(w);
}
void tss_work_bf16(Work *w,int on) {
    w->tape_bf16=on;
}
// Tiled tied head, eight tokens at a time. No nested OpenMP and no BLAS dependency.
// dX = dlogits E; dE += dlogits^T X. Reuse each embedding row across the tile.
/* Multi-token prediction (training only): an auxiliary head predicts the byte two ahead from the
 * same final state, logits2 = E (W rmsnorm(x_t)) with the tied embedding E and a d x d matrix W
 * (identity init), weighted by mtp_lambda. The head is not saved; inference is unchanged. */
static real mtp_lambda=0;static real*mtp_W=NULL,*mtp_m=NULL,*mtp_v=NULL;static int mtp_d=0;
static void mtp_head(const Cfg*c,const Params*P,Params*G,const real*x,const int*labels,int L,real*g,Work*w){
    const int d=c->d,V=c->vocab;if(!mtp_W||mtp_d!=d)return;if(!w->mtp_g)w->mtp_g=calloc((size_t)d*d,sizeof(real));
    real u[512],z[512],dz[512],du[512],lg[256];double pe[256];
    for(int t=0;t+1<L;t++){if(labels[t]<0||labels[t+1]<0)continue;int y=labels[t+1];real r;
        rmsnorm_fwd(x+(size_t)t*d,P->norm_f,u,d,&r);
        for(int i=0;i<d;i++)z[i]=dotr(mtp_W+(size_t)i*d,u,d);
        real mx=-INFINITY;for(int v=0;v<V;v++){lg[v]=dotr(P->embed+(size_t)v*d,z,d);mx=lg[v]>mx?lg[v]:mx;}
        double zs=0;for(int v=0;v<V;v++){pe[v]=exp(lg[v]-mx);zs+=pe[v];}
        memset(dz,0,d*sizeof(real));
        for(int v=0;v<V;v++){real dl=mtp_lambda*(real)(pe[v]/zs-(v==y));if(dl==0)continue;
            axpy(G->embed+(size_t)v*d,dl,z,d);axpy(dz,dl,P->embed+(size_t)v*d,d);}
        memset(du,0,d*sizeof(real));
        for(int i=0;i<d;i++){axpy(w->mtp_g+(size_t)i*d,dz[i],u,d);axpy(du,dz[i],mtp_W+(size_t)i*d,d);}
        rmsnorm_bwd(x+(size_t)t*d,P->norm_f,r,du,g+(size_t)t*d,G->norm_f,d);
    }
}
static real head_train(const Cfg *c,const Params *P,Params *G,const real *x, const int *labels,int L,real *g,int *ns,Work *w) {
    const int d=c->d,V=c->vocab,BT=8;
    real *X=w->head,*dX=X+BT*d,*pr=dX+BT*d,*rn=pr+BT*V;
    real loss=0;
    for(int t0=0;t0<L;t0+=BT) {
        const int nt=L-t0<BT?L-t0:BT;
        for(int t=0;t<nt;t++)rmsnorm_fwd(x+(size_t)(t0+t)*d,P->norm_f,X+t*d,d,rn+t);
        // Dot-product microkernel: load each embedding vector once for four tokens.
        for(int v=0;v<V;v++) {
            const real *er=P->embed+(size_t)v*d;
            int t=0;
#if FAST
            for(;t+4<=nt;t+=4) {
                __m512 a0=_mm512_setzero_ps(),a1=a0,a2=a0,a3=a0;
                int i=0;
                for(;i+16<=d;i+=16) {
                    const __m512 ev=_mm512_loadu_ps(er+i);
                    a0=_mm512_fmadd_ps(ev,_mm512_loadu_ps(X+(t+0)*d+i),a0);
                    a1=_mm512_fmadd_ps(ev,_mm512_loadu_ps(X+(t+1)*d+i),a1);
                    a2=_mm512_fmadd_ps(ev,_mm512_loadu_ps(X+(t+2)*d+i),a2);
                    a3=_mm512_fmadd_ps(ev,_mm512_loadu_ps(X+(t+3)*d+i),a3);
                }
                real z0=_mm512_reduce_add_ps(a0),z1=_mm512_reduce_add_ps(a1);
                real z2=_mm512_reduce_add_ps(a2),z3=_mm512_reduce_add_ps(a3);
                for(;i<d;i++) {
                    z0+=er[i]*X[(t+0)*d+i];
                    z1+=er[i]*X[(t+1)*d+i];
                    z2+=er[i]*X[(t+2)*d+i];
                    z3+=er[i]*X[(t+3)*d+i];
                } pr[(t+0)*V+v]=z0;
                pr[(t+1)*V+v]=z1;
                pr[(t+2)*V+v]=z2;
                pr[(t+3)*V+v]=z3;
            }
#endif
            for(;t<nt;t++)pr[t*V+v]=dotr(er,X+t*d,d);
        }
        for(int t=0;t<nt;t++) {
            real *pt=pr+t*V;
            if(labels[t0+t]<0) {
                memset(pt,0,sizeof(real)*V);
                continue;
            }
            real mx=-INFINITY,z=0;
            for(int v=0;v<V;v++)mx=pt[v]>mx?pt[v]:mx;
            // Cache double exponentials, preserving original summation/rounding.
            head_exponentials(pt,mx,w->head_exp,V);
            for(int v=0;v<V;v++)z+=w->head_exp[v]; loss+=log(z)+mx-pt[labels[t0+t]];
            if(w->pos_loss)w->pos_loss[t0+t]=log(z)+mx-pt[labels[t0+t]];
            (*ns)++;
            for(int v=0;v<V;v++)pt[v]=w->head_exp[v]/z-(v==labels[t0+t]);
        }
        memset(dX,0,sizeof(real)*nt*d);
        for(int v=0;v<V;v++) {
            const real *er=P->embed+(size_t)v*d;
            real *ge=G->embed+(size_t)v*d;
            int i=0;
#if FAST
            for(;i+16<=d;i+=16) {
                const __m512 ev=_mm512_loadu_ps(er+i);
                __m512 eg=_mm512_loadu_ps(ge+i);
                for(int t=0;t<nt;t++) {
                    const __m512 pv=_mm512_set1_ps(pr[t*V+v]);
                    eg=_mm512_fmadd_ps(pv,_mm512_loadu_ps(X+t*d+i),eg);
                    const __m512 dx=_mm512_fmadd_ps(pv,ev,_mm512_loadu_ps(dX+t*d+i));
                    _mm512_storeu_ps(dX+t*d+i,dx);
                } _mm512_storeu_ps(ge+i,eg);
            }
#endif
            for(;i<d;i++) {
                real eg=ge[i];
                for(int t=0;t<nt;t++) {
                    const real p=pr[t*V+v];
                    eg+=p*X[t*d+i];
                    dX[t*d+i]+=p*er[i];
                } ge[i]=eg;
            }
        }
        for(int t=0;t<nt;t++)if(labels[t0+t]>=0) rmsnorm_bwd(x+(size_t)(t0+t)*d,P->norm_f,rn[t],dX+t*d,g+(size_t)(t0+t)*d,G->norm_f,d);
    }
    return loss;
}
/* bwd_layers>0: backpropagate only through the top bwd_layers layers (cheaper test-time training);
 * lower layers and the input embedding get no gradient. The tied output head still learns. */
static int bwd_layers=0;
/* Thoughts in pause slots (--pause K). A first forward pass runs the window with plain pause slots
 * (only the bottom --latent-layers M layers when M>0). Its state u_t = rmsnorm(x_{t-1}) at each slot
 * then fills the slot for the second, trained pass:
 *   --latent 1  dense:    input = embed[pause] + Wl u_t        (Wl d x d, zero init)
 *   --latent 2  discrete: input = embed[pause] + embed[b_t], b_t ~ softmax(E Wt u_t), Wt identity init.
 * Pass 1 is a constant (stop-gradient). Wl learns from the input gradient. Wt is trained by REINFORCE:
 * the reward of a word's thoughts is (pass-1 loss - pass-2 loss) on the next rl_n real bytes, so the
 * no-thought baseline comes free from pass 1. Wl/Wt are training-time state only for now. */
static int latent_on=0,latent_layers=0,rl_n=4;static real rl_beta=1;static real*lat_W=NULL,*lat_m=NULL,*lat_v=NULL;static int lat_d=0;
static double rl_rsum=0;static long rl_rcount=0;
static void thought_logits(const Cfg*c,const Params*P,const real*u,real*z,real*q){const int d=c->d;
    for(int i=0;i<d;i++)z[i]=dotr(lat_W+(size_t)i*d,u,d);for(int v=0;v<c->vocab;v++)q[v]=v?dotr(P->embed+(size_t)v*d,z,d):-INFINITY;}
static real pos_ce(const Cfg*c,const Params*P,const real*x,int y){const int d=c->d;real u[512],r,mx=-INFINITY,lg[256];double z=0;
    rmsnorm_fwd(x,P->norm_f,u,d,&r);for(int v=0;v<c->vocab;v++){lg[v]=dotr(P->embed+(size_t)v*d,u,d);mx=lg[v]>mx?lg[v]:mx;}
    for(int v=0;v<c->vocab;v++)z+=exp(lg[v]-mx);return (real)(log(z)+mx-lg[y]);}
static void latent_inputs(const Cfg*c,const Params*P,const Params*Q,const int*tok,const int*lab,int L,real*x,real*h,Tape*tp,Work*w){
    const int d=c->d;if(!lat_W||lat_d!=d)return;int any=0;for(int t=1;t<L;t++)if(tok[t]==0){any=1;break;}if(!any)return;
    if(!w->lat_u){w->lat_u=calloc((size_t)w->Lmax*d,sizeof(real));w->lat_b=calloc(w->Lmax,sizeof(int));w->lat_l1=calloc(w->Lmax,sizeof(real));w->lat_rng=0x9E3779B97F4A7C15ULL^(uint64_t)(uintptr_t)w;}
    real*x0=malloc((size_t)L*d*sizeof(real));memcpy(x0,x,(size_t)L*d*sizeof(real));
    int nv=c->n_layer*n_loops,M=latent_on==1&&latent_layers>0&&latent_layers<nv?latent_layers:nv;
    for(int l=0;l<M;l++){if(!(l&1))memset(h,0,sizeof(real)*(size_t)c->T*NODES(c)*c->n*c->p);layer_fwd(c,P,Q,l,L,x,h,tp+l,w);}
    if(latent_on==2&&lab)for(int t=0;t<L;t++)w->lat_l1[t]=lab[t]>=0?pos_ce(c,P,x+(size_t)t*d,lab[t]):0;
    real z[512],q[256];
    for(int t=0;t<L;t++){real*u=w->lat_u+(size_t)t*d;w->lat_b[t]=-1;if(t==0||tok[t]!=0){memset(u,0,d*sizeof(real));continue;}real r;
        rmsnorm_fwd(x+(size_t)(t-1)*d,P->norm_f,u,d,&r);
        if(latent_on==1){for(int i=0;i<d;i++)x0[(size_t)t*d+i]+=dotr(lat_W+(size_t)i*d,u,d);continue;}
        thought_logits(c,P,u,z,q);real mx=-INFINITY;for(int v=1;v<c->vocab;v++)mx=q[v]>mx?q[v]:mx;
        double zs=0;for(int v=1;v<c->vocab;v++)zs+=exp(q[v]-mx);double a=urand(&w->lat_rng)*zs;int b=c->vocab-1;
        for(int v=1;v<c->vocab;v++){a-=exp(q[v]-mx);if(a<=0){b=v;break;}}
        w->lat_b[t]=b;axpy(x0+(size_t)t*d,1,P->embed+(size_t)b*d,d);}
    memcpy(x,x0,(size_t)L*d*sizeof(real));free(x0);
}
/* After the trained pass: Wl gradient (dense) or the thought-embedding gradient plus REINFORCE on Wt. */
static void latent_grad(const Cfg*c,const Params*P,Params*G,const int*tok,const int*lab,int L,const real*g,Work*w){
    const int d=c->d;if(!lat_W||!w->lat_u)return;if(!w->lat_g)w->lat_g=calloc((size_t)d*d,sizeof(real));
    if(latent_on==1){for(int t=1;t<L;t++)if(tok[t]==0)for(int i=0;i<d;i++)axpy(w->lat_g+(size_t)i*d,g[(size_t)t*d+i],w->lat_u+(size_t)t*d,d);return;}
    real z[512],q[256],dz[512];double pe[256];
    for(int t=1;t<L;t++){if(w->lat_b[t]<0)continue;axpy(G->embed+(size_t)w->lat_b[t]*d,1,g+(size_t)t*d,d);
        int e=t;while(e<L&&tok[e]==0)e++;e--; /* last slot of this word's run predicts its first real byte */
        double R=0;int cnt=0;for(int p=e;p<L&&p<e+rl_n;p++)if(lab[p]>=0&&w->pos_loss){R+=w->lat_l1[p]-w->pos_loss[p];cnt++;}
        if(!cnt)continue;R/=cnt;
#pragma omp atomic
        rl_rsum+=R;
#pragma omp atomic
        rl_rcount++;
        thought_logits(c,P,w->lat_u+(size_t)t*d,z,q);real mx=-INFINITY;for(int v=1;v<c->vocab;v++)mx=q[v]>mx?q[v]:mx;
        double zs=0;for(int v=1;v<c->vocab;v++){pe[v]=exp(q[v]-mx);zs+=pe[v];}
        memset(dz,0,d*sizeof(real));real A=(real)(rl_beta*R);if(A>2)A=2;if(A<-2)A=-2;
        for(int v=1;v<c->vocab;v++){real dq=A*(real)(pe[v]/zs-(v==w->lat_b[t]));axpy(dz,dq,P->embed+(size_t)v*d,d);}
        for(int i=0;i<d;i++)axpy(w->lat_g+(size_t)i*d,dz[i],w->lat_u+(size_t)t*d,d);}
}
real tss_train_seq(const Cfg *c,const Params *P,const Params *Q,Params *G,const int *tokens,const int *labels,int L,unsigned char *hit,int *n_scored,Work *w,const Codes *k) {
    const int d=c->d,N=NODES(c),T=c->T,nl=c->n_layer;
    if(L>w->Lmax)return NAN;
    w->k=k;
    Tape *tp=w->tp;
    real *x=w->x,*h=w->h;
    for(int t=0;t<L;t++)memcpy(x+(size_t)t*d,P->embed+(size_t)tokens[t]*d,sizeof(real)*d);
    if(pos_feat>0){int kw=0,kl=0;for(int t=0;t<L;t++){pos_feat_next(tokens[t],&kw,&kl);pos_feat_add(x+(size_t)t*d,kw,kl,d);}}
    if(latent_on)latent_inputs(c,P,Q,tokens,labels,L,x,h,tp,w);
    for(int l=0;l<nl*n_loops;l++) {
        if(!(l&1))carry_pre(c,w,l,h);
        layer_fwd(c,P,Q,l,L,x,h,tp+l,w);
        if(!(l&1))carry_post(c,w,l,L,h);
    }
    (void)N;(void)T;
    if(continuous_check)for(int l=0;l<nl*n_loops;l++)memset(tp[l].bw,0,(size_t)L*T*(c->D+2)*sizeof(real)); /* fixed-path check: no routing credit */
    real *g=w->g;
    int ns=0;
    memset(g,0,sizeof(real)*(size_t)L*d);
    if(latent_on==2&&!w->pos_loss)w->pos_loss=calloc(w->Lmax,sizeof(real));
    real loss=head_train(c,P,G,x,labels,L,g,&ns,w);
    if(mtp_lambda>0)mtp_head(c,P,G,x,labels,L,g,w);
    real *lam=w->lam;const int nv=nl*n_loops;int lstop=bwd_layers>0&&bwd_layers<nv?nv-bwd_layers:0;
    for(int l=nv-1;l>=lstop;l--) {
        if(!(l&1))memset(lam,0,sizeof(real)*(size_t)T*N*c->n*c->p);
        layer_bwd(c,P,Q,G,l,L,g,lam,hit+node_off(c,l%nl),tp+l,w);
    }
    if(!lstop)for(int t=0;t<L;t++)for(int i=0;i<d;i++)G->embed[(size_t)tokens[t]*d+i]+=g[(size_t)t*d+i];
    if(latent_on&&!lstop)latent_grad(c,P,G,tokens,labels,L,g,w);
    *n_scored=ns;
    return loss;
}
typedef struct {
    real lr,beta1,beta2,eps,weight_decay,clip;
    long step;
} Hyper;
#define N_FIELDS 11
static void fields(const Params *p,real **f) {
    f[0]=p->embed;
    f[1]=p->norm_f;
    f[2]=p->norm;
    f[3]=p->conv;
    f[4]=p->w_in;
    f[5]=p->bias;
    f[6]=p->w_out;
    f[7]=p->A_log;
    f[8]=p->proj;
    f[9]=p->dt_bias;
    f[10]=p->U;
}
static void sizes(const Cfg *c,size_t *z) {
    const size_t nl=c->n_layer,d=c->d,rows=node_rows(c);
    z[0]=(size_t)c->vocab*d;
    z[1]=d;
    z[2]=nl*d;
    z[3]=nl*c->K*d;
    z[4]=rows*d;
    z[5]=rows;
    z[6]=rows*d;
    z[7]=rows;
    z[8]=nl*PDIM(c)*d;
    z[9]=nl*c->T;
    z[10]=nl*c->T*c->p*d;
}
static int node_field(int f) {
    return f==4||f==5||f==6||f==7;
}
static void adam(real *p,real *g,real *m,real *v,size_t n,const Hyper *h,real scale,real cached_c1,real cached_c2) {
    const real b1=h->beta1,b2=h->beta2;
    const real c1=cached_c1,c2=cached_c2;
    for(size_t i=0;i<n;i++) {
        const real gi=g[i]*scale;
        m[i]=b1*m[i]+(1-b1)*gi;
        v[i]=b2*v[i]+(1-b2)*gi*gi;
        p[i]-=h->lr*(h->weight_decay*p[i]+(m[i]/c1)/(sqrt(v[i]/c2)+h->eps));
        g[i]=0;
    }
}
// One CPU team for the batch, reduction/norm, and row-local update/packing.
// The clipping barrier is retained. No worker touches another worker's row.
static double reduce_span(real *dst, real *tf[64][N_FIELDS], int f,
                          size_t off, size_t len, int threads) {
    for(int th=0;th<threads;th++) {
        real *src=tf[th][f]+off;
        real *out=dst+off;
        for(size_t i=0;i<len;i++){out[i]+=src[i];src[i]=0;}
    }
    double norm=0;
    for(size_t i=0;i<len;i++)norm+=(double)dst[off+i]*dst[off+i];
    return norm;
}
/* Byte-mode Muon: when byte_muon is set, the matrix fields (tree-node w_in/w_out rows, proj, U) of
 * byte_muon_model skip AdamW here and are updated by blockwise Muon (NorMuon when normuon is set). */
static int byte_muon=0,normuon=0;static void*byte_muon_model=NULL;
static void byte_muon_update(const Hyper*h,real scale,int threads);
/* Sequence hook: when set, tss_train_step trains sequence b through this function instead of tss_train_seq
 * (the byte LM), with the same parallel reduction, clipping, Adam/Muon and requantisation. Used by
 * body/ppo_tss.c, which feeds observation vectors and backpropagates a policy loss. */
typedef real (*SeqHook)(void *user,int b,const Cfg *c,const Params *P,const Params *Q,Params *G,unsigned char *hit,int *n_scored,Work *w,const Codes *k);
static SeqHook seq_hook=NULL;static void *seq_hook_user=NULL;
real tss_train_step(const Cfg *c,Params *P,Params *Q,Params *G,unsigned char *hit,
                    Params *M,Params *Vm,Params *Gt,unsigned char *hits,Work **ws,
                    int threads,const int *tokens,const int *labels,int B,int L,
                    const Hyper *h,Codes *k) {
    if(threads<1||threads>64)return NAN;
    const size_t rows=node_rows(c);
    const int d=c->d;
    size_t z[N_FIELDS],ends[N_FIELDS+1]={0},chunk[N_FIELDS];sizes(c,z);
    real *gf[N_FIELDS],*pf[N_FIELDS],*mf[N_FIELDS],*vf[N_FIELDS],*tf[64][N_FIELDS];
    fields(G,gf);fields(P,pf);fields(M,mf);fields(Vm,vf);
    for(int th=0;th<threads;th++)fields(Gt+th,tf[th]);
    for(int f=0;f<N_FIELDS;f++) {
        chunk[f]=(f==8||f==10)?(size_t)d:4096;
        ends[f+1]=ends[f]+(node_field(f)?0:(z[f]+chunk[f]-1)/chunk[f]);
    }
    const long jobs=(long)(rows+ends[N_FIELDS]);
    const real c1=1-pow(h->beta1,(real)h->step),c2=1-pow(h->beta2,(real)h->step);
    real loss=0,scale=0;int scored=0;double nrm=0;
#pragma omp parallel num_threads(threads)
    {
#ifdef _OPENMP
        const int th=omp_get_thread_num();
#else
        const int th=0;
#endif
#pragma omp for schedule(static) reduction(+:loss,scored)
        for(int b=0;b<B;b++) {
            int ns=0;tss_work_seed(ws[th],(uint64_t)h->step*1000003ULL+(uint64_t)b);
            ws[th]->carry_h=carry_bank_h?carry_bank_h+(size_t)b*carry_sz_h:NULL;ws[th]->carry_hist=carry_bank_hist?carry_bank_hist+(size_t)b*carry_sz_hist:NULL;
            loss+=seq_hook?seq_hook(seq_hook_user,b,c,P,Q,Gt+th,hits+rows*th,&ns,ws[th],k)
                          :tss_train_seq(c,P,Q,Gt+th,tokens+(size_t)b*L,labels+(size_t)b*L,L,
                               hits+rows*th,&ns,ws[th],k);
            ws[th]->carry_h=ws[th]->carry_hist=NULL;
            scored+=ns;
        }
#pragma omp for schedule(static,16) reduction(+:nrm)
        for(long job=0;job<jobs;job++) {
            if((size_t)job<rows) {
                const size_t r=(size_t)job;unsigned char any=0;
                for(int j=0;j<threads;j++){any|=hits[rows*j+r];hits[rows*j+r]=0;}
                hit[r]=any;if(!any)continue;
                for(int f=4;f<=7;f++) {
                    const size_t width=(f==4||f==6)?(size_t)d:1;
                    nrm+=reduce_span(gf[f],tf,f,r*width,width,threads);
                }
            } else {
                const size_t dj=(size_t)job-rows;int f=0;
                while(dj>=ends[f+1])f++;
                const size_t off=(dj-ends[f])*chunk[f],len=z[f]-off<chunk[f]?z[f]-off:chunk[f];
                nrm+=reduce_span(gf[f],tf,f,off,len,threads);
            }
        }
#pragma omp single
        {
            const real inv=scored?(real)1/scored:0;
            const double norm=sqrt(nrm)*inv;
            scale=inv;if(h->clip>0&&norm>h->clip)scale*=h->clip/(norm+(real)1e-6);
        }
#pragma omp for schedule(static,16) nowait
        for(long job=0;job<jobs;job++) {
            if((size_t)job<rows) {
                const size_t r=(size_t)job;if(!hit[r])continue;
                for(int f=4;f<=7;f++) {
                    if(byte_muon&&(f==4||f==6))continue;
                    const size_t width=(f==4||f==6)?(size_t)d:1,off=r*width;
                    adam(pf[f]+off,gf[f]+off,mf[f]+off,vf[f]+off,width,h,scale,c1,c2);
                }
                Q->A_log[r]=P->A_log[r];if(k)k->decay_rates[r]=exp(P->A_log[r]);
                if(!byte_muon){
                tss_ternary(P->w_in+r*d,Q->w_in+r*d,1,d,c->group);
                tss_ternary(P->w_out+r*d,Q->w_out+r*d,1,d,c->group);
                if(k&&k->ok){plane_rows(Q->w_in,&k->w_in,r,r+1,d,c->group);plane_rows(Q->w_out,&k->w_out,r,r+1,d,c->group);}}
            } else {
                const size_t dj=(size_t)job-rows;int f=0;
                while(dj>=ends[f+1])f++;
                const size_t off=(dj-ends[f])*chunk[f],len=z[f]-off<chunk[f]?z[f]-off:chunk[f];
                if(byte_muon&&(f==8||f==10))continue;
                adam(pf[f]+off,gf[f]+off,mf[f]+off,vf[f]+off,len,h,scale,c1,c2);
                if(f==8||f==10) {
                    real *qm=f==8?Q->proj:Q->U;const size_t r=off/d;
                    tss_ternary(pf[f]+off,qm+off,1,d,c->group);
                    if(k&&k->ok)plane_rows(qm,f==8?&k->proj:&k->U,r,r+1,d,c->group);
                }
            }
        }
    }
    if(byte_muon)byte_muon_update(h,scale,threads);
    if(mtp_lambda>0&&mtp_W){const int dd=mtp_d;size_t n=(size_t)dd*dd;real*gsum=calloc(n,sizeof(real));
        for(int th=0;th<threads;th++)if(ws[th]->mtp_g){for(size_t i=0;i<n;i++){gsum[i]+=ws[th]->mtp_g[i];ws[th]->mtp_g[i]=0;}}
        Hyper hh=*h;hh.weight_decay=0;adam(mtp_W,gsum,mtp_m,mtp_v,n,&hh,scale,c1,c2);free(gsum);}
    if(latent_on&&lat_W){size_t n=(size_t)lat_d*lat_d;real*gsum=calloc(n,sizeof(real));
        for(int th=0;th<threads;th++)if(ws[th]->lat_g){for(size_t i=0;i<n;i++){gsum[i]+=ws[th]->lat_g[i];ws[th]->lat_g[i]=0;}}
        Hyper hh=*h;hh.weight_decay=0;adam(lat_W,gsum,lat_m,lat_v,n,&hh,scale,c1,c2);free(gsum);}
    return scored?loss/scored:0;
}

static uint64_t g_rng=88172645463325252ULL;
static double rand01(void) {
    g_rng^=g_rng>>12;
    g_rng^=g_rng<<25;
    g_rng^=g_rng>>27;
    return (double)((g_rng*2685821657736338717ULL)>>11)*(1.0/9007199254740992.0);
}
static double randn(void) {
    double u=rand01(),v=rand01();
    if(u<1e-300)u=1e-300;
    return sqrt(-2*log(u))*cos(6.283185307179586*v);
}
static double wall(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec+ts.tv_nsec*1e-9;
}
static void params_alloc(const Cfg *c,Params *P) {
    size_t z[N_FIELDS];
    sizes(c,z);
    real *f[N_FIELDS];
    for(int i=0;i<N_FIELDS;i++)f[i]=calloc(z[i]?z[i]:1,sizeof(real));
    P->embed=f[0];
    P->norm_f=f[1];
    P->norm=f[2];
    P->conv=f[3];
    P->w_in=f[4];
    P->bias=f[5];
    P->w_out=f[6];
    P->A_log=f[7];
    P->proj=f[8];
    P->dt_bias=f[9];
    P->U=f[10];
}
static void params_init(const Cfg *c,Params *P) {
    const int d=c->d,T=c->T,D=c->D,N=NODES(c),Pd=PDIM(c),K=c->K,pv=c->p;
    const real k_in=1/sqrt((real)d),k_out=1/sqrt((real)(T*(D+1))),k_u=k_out/sqrt((real)pv);
#define UNI(k) ((real)((2*rand01()-1)*(k)))
    for(size_t i=0;i<(size_t)c->vocab*d;i++)P->embed[i]=(real)(0.02*randn());
    for(int i=0;i<d;i++)P->norm_f[i]=1;
    for(int l=0;l<c->n_layer;l++) {
        for(int i=0;i<d;i++)P->norm[(size_t)l*d+i]=1;
        for(int i=0;i<K*d;i++)P->conv[(size_t)l*K*d+i]=i<d;
        const size_t r0=node_off(c,l);
        for(size_t r=r0;r<node_off(c,l+1);r++) {
            for(int i=0;i<d;i++)P->w_in[r*d+i]=UNI(k_in);
            P->bias[r]=UNI(k_in);
            for(int i=0;i<d;i++)P->w_out[r*d+i]=UNI(k_out);
            P->A_log[r]=ssm_ltc?(real)(-rand01()*log(1000.)):(real)log(1+15*rand01()); /* LTC: tau log-uniform in [1,1000] */
        }
        for(size_t i=0;i<(size_t)BASE_PDIM(c)*d;i++)P->proj[(size_t)l*Pd*d+i]=UNI(k_in);
        for(int t=0;t<T;t++) {
            double dt=exp(rand01()*(log(0.1)-log(0.001))+log(0.001));
            if(dt<1e-4)dt=1e-4;
            P->dt_bias[(size_t)l*T+t]=(real)(dt+log(-expm1(-dt)));
        }
        for(size_t i=0;i<(size_t)T*pv*d;i++)P->U[(size_t)l*T*pv*d+i]=UNI(k_u);
        if((c->fw_mask>>l)&1){
            for(size_t r=r0;r<node_off(c,l+1);r++)P->A_log[r]=-2;
            for(int tr=0;tr<T;tr++)P->dt_bias[(size_t)l*T+tr]=0;
        }
        if((l&1)&&c->fff_gate)memset(P->U+(size_t)l*T*pv*d,0,(size_t)c->n*d*sizeof(real));
    }
#undef UNI
}
typedef struct {
    Cfg c;
    Params P,Q,G,M,V,*Gt;
    unsigned char *hit,*hits;
    Work **ws;
    Codes *k;
    int threads,Lmax;
} Model;
static Model *model_new(const Cfg *c,int threads,int Lmax) {
    Model *m=calloc(1,sizeof(Model));
    m->c=*c;
    m->threads=threads;
    m->Lmax=Lmax;
    params_alloc(c,&m->P);
    params_alloc(c,&m->Q);
    params_alloc(c,&m->G);
    params_alloc(c,&m->M);
    params_alloc(c,&m->V);
    m->Gt=calloc(threads,sizeof(Params));
    for(int t=0;t<threads;t++)params_alloc(c,m->Gt+t);
    const size_t rows=node_rows(c);
    m->hit=calloc(rows,1);
    m->hits=calloc(rows*threads,1);
    m->ws=calloc(threads,sizeof(Work *));
    for(int t=0;t<threads;t++) {
        m->ws[t]=tss_work_new(c,Lmax);
        tss_work_bf16(m->ws[t],1);
    }m->k=tss_codes_new(c);
    return m;
}
static void model_requantize(Model *m) {
    tss_quantize(&m->c,&m->P,&m->Q);
    tss_codes_all(&m->c,&m->Q,m->k);
}
static double eval_seq(Model *m,const int *tok,const int *lab,int L,Work *w,int *ns,real *logits) {
    const Cfg *c=&m->c;
    const int d=c->d,N=NODES(c),T=c->T;
    w->k=m->k;
    w->no_alt=1;
    real *x=w->x,*h=w->h,*xn=w->xn,*lg=w->lg;
    for(int t=0;t<L;t++)memcpy(x+(size_t)t*d,m->P.embed+(size_t)tok[t]*d,sizeof(real)*d);
    if(pos_feat>0){int kw=0,kl=0;for(int t=0;t<L;t++){pos_feat_next(tok[t],&kw,&kl);pos_feat_add(x+(size_t)t*d,kw,kl,d);}}
    if(latent_on)latent_inputs(c,&m->P,&m->Q,tok,NULL,L,x,h,w->tp,w);
    for(int l=0;l<c->n_layer*n_loops;l++) {
        if(!(l&1))memset(h,0,sizeof(real)*(size_t)T*N*c->n*c->p);
        layer_fwd(c,&m->P,&m->Q,l,L,x,h,w->tp+l,w);
    }
    double loss=0;
    *ns=0;
    for(int t=0;t<L;t++) {
        if(!logits&&lab[t]<0)continue;
        real r;
        rmsnorm_fwd(x+(size_t)t*d,m->P.norm_f,xn,d,&r);
        real mx=-INFINITY,z=0;
        for(int v=0;v<c->vocab;v++)lg[v]=dotr(m->P.embed+(size_t)v*d,xn,d),mx=lg[v]>mx?lg[v]:mx;
        if(logits)memcpy(logits+(size_t)t*c->vocab,lg,sizeof(real)*c->vocab);
        if(lab[t]<0)continue;
        for(int v=0;v<c->vocab;v++)z+=exp(lg[v]-mx);
        loss+=log(z)+mx-lg[lab[t]];
        (*ns)++;
    }w->no_alt=0;
    return loss;
}
typedef struct {
    real *h,*hist,*x,*u,*mx,*xt,*y,*p,*R,*Racc;
    int8_t *xq;
    Work *work;
    int kw,kl; /* structural position counters for pos_feat */
} Gen;
static Gen *gen_new(const Cfg *c) {
    Gen *g=calloc(1,sizeof(Gen));
    const int d=c->d;
    g->h=calloc(node_rows(c)*c->n*c->p*n_loops,sizeof(real));
    g->hist=calloc((size_t)c->n_layer*n_loops*(c->K>1?c->K-1:1)*d,sizeof(real));
    g->x=calloc(d,sizeof(real));
    g->u=calloc(d,sizeof(real));
    g->mx=calloc(d,sizeof(real));
    g->xt=calloc(d,sizeof(real));
    g->y=calloc(d,sizeof(real));
    g->p=calloc(PDIM(c),sizeof(real));
    g->R=calloc(c->p,sizeof(real));
    g->Racc=calloc(c->p,sizeof(real));
    g->xq=calloc(d,1);
    g->work=tss_work_new(c,1);g->work->stream_hist=g->hist;
    return g;
}
static void gen_free(Gen*g){free(g->h);free(g->hist);free(g->x);free(g->u);free(g->mx);free(g->xt);free(g->y);free(g->p);free(g->R);free(g->Racc);free(g->xq);tss_work_free(g->work);free(g);}
static void gen_core(Model*m,Gen*g,const real*input,real*logits) {
    const Cfg*c=&m->c;int d=c->d;Work*w=g->work;w->k=m->k;w->no_alt=1;
    memcpy(g->x,input,sizeof(real)*d);
    for(int l=0;l<c->n_layer*n_loops;l++)layer_fwd(c,&m->P,&m->Q,l,1,g->x,g->h+((size_t)(l/c->n_layer)*node_rows(c)+node_off(c,l%c->n_layer))*c->n*c->p,w->tp+l,w);
    w->pos++;
    if(logits){real rr;rmsnorm_fwd(g->x,m->P.norm_f,g->u,d,&rr);for(int v=0;v<c->vocab;v++)logits[v]=dotr(m->P.embed+(size_t)v*d,g->u,d);}
}
static void gen_step(Model*m,Gen*g,int token,real*logits){
    if(pos_feat<=0){gen_core(m,g,m->P.embed+(size_t)token*m->c.d,logits);return;}
    real x[512];int d=m->c.d;memcpy(x,m->P.embed+(size_t)token*d,d*sizeof(real));pos_feat_next(token,&g->kw,&g->kl);pos_feat_add(x,g->kw,g->kl,d);gen_core(m,g,x,logits);
}
static int sample_logits(const real *lg,int V,double temp) {
    double mx=-1e300,z=0;
    for(int v=0;v<V;v++)mx=lg[v]>mx?lg[v]:mx;
    for(int v=0;v<V;v++)z+=exp((lg[v]-mx)/temp);
    double r=rand01()*z;
    for(int v=0;v<V;v++)if((r-=exp((lg[v]-mx)/temp))<=0)return v;
    return V-1;
}
static int save(const Model *m,const char *path) {
    FILE *f=fopen(path,"wb");
    if(!f)return -1;
    size_t z[N_FIELDS];
    sizes(&m->c,z);
    real *pf[N_FIELDS];
    fields(&m->P,pf);
    const int rs=sizeof(real);
    /* TFX1 adds an extension header after Cfg: {ssm_groups, ssm_ltc, 0, 0}. Older magics imply {1, 0}. */
    const int ext=ssm_groups!=1||ssm_ltc||ssm_qknorm||ssm_rope||pos_feat>0||n_loops>1;
    fwrite(ext?"TFX1":PROTECTED_FW&&m->c.fw_mask?"TFP1":"TFW1",1,4,f);
    fwrite(&rs,sizeof(int),1,f);
    fwrite(&m->c,sizeof(Cfg),1,f);
    if(ext){int e[4]={ssm_groups,ssm_ltc|(n_loops>1?n_loops<<4:0),ssm_qknorm|(ssm_rope<<1),(int)(pos_feat*1e6+.5)};fwrite(e,sizeof(int),4,f);}
    for(int i=0;i<N_FIELDS;i++)fwrite(pf[i],sizeof(real),z[i],f);
    return fclose(f);
}
static Model *load(const char *path,int threads,int Lmax) {
    FILE *f=fopen(path,"rb");
    if(!f)return NULL;
    char magic[4];
    int rs;
    Cfg c;
    if(fread(magic,1,4,f)!=4||(memcmp(magic,"TSFM",4)&&memcmp(magic,"TSFK",4)&&memcmp(magic,"TFW1",4)&&memcmp(magic,"TFP1",4)&&memcmp(magic,"TFX1",4))||fread(&rs,sizeof(int),1,f)!=1||rs!=(int)sizeof(real)||!read_cfg(f,&c,!memcmp(magic,"TSFM",4))) {
        fclose(f);
        return NULL;
    }
    const int ext=!memcmp(magic,"TFX1",4);int e[4]={1,0,0,0};
    if(ext&&(fread(e,sizeof(int),4,f)!=4||e[0]<1||e[0]>c.T||c.T%e[0]||e[1]<0||(e[1]&15)>2||(e[1]>>4)>16)){fclose(f);return NULL;}
    ssm_groups=e[0];ssm_ltc=e[1]&15;n_loops=(e[1]>>4)>1?(e[1]>>4):1;ssm_qknorm=e[2]&1;ssm_rope=(e[2]>>1)&1;pos_feat=(real)(e[3]/1e6);
    if(!ext&&((c.fw_mask && (PROTECTED_FW ? memcmp(magic,"TFP1",4) : !memcmp(magic,"TFP1",4))) || (memcmp(magic,"TFW1",4)&&memcmp(magic,"TFP1",4)&&c.fw_mask))){fclose(f);return NULL;}
    if((c.fw_mask&~0x5555)||c.n_layer<1||c.n_layer>16||(c.fw_mask>>c.n_layer)||c.fff_gate<0||c.fff_gate>1){fclose(f);return NULL;}
    Model *m=model_new(&c,threads,Lmax);
    size_t z[N_FIELDS];
    sizes(&c,z);
    real *pf[N_FIELDS];
    fields(&m->P,pf);
    for(int i=0;i<N_FIELDS;i++)if(fread(pf[i],sizeof(real),z[i],f)!=z[i]) {
        fclose(f);
        return NULL;
    }fclose(f);
    model_requantize(m);
    return m;
}
static void generate(Model *m,const char *prompt,int n,double temp) {
    Gen *g=gen_new(&m->c);
    real *lg=malloc(sizeof(real)*m->c.vocab);
    const char *pr=prompt&&*prompt?prompt:"\n";
    for(const char *q=pr;*q;q++) {
        gen_step(m,g,(unsigned char)*q,lg);
        fputc(*q,stdout);
    }for(int i=0;i<n;i++) {
        const int v=sample_logits(lg,m->c.vocab,temp);
        fputc(v,stdout);
        gen_step(m,g,v,lg);
    }fputc('\n',stdout);
    free(lg);
}


/* Frozen causal next-byte neural predictor. Entropy is computed from the
 * entire predicted distribution, never from the observed next-byte label. */
static Cfg neural_entropy_cfg(void){
    Cfg c={.vocab=256,.d=64,.n_layer=2,.T=2,.D=4,.n=8,.p=16,.K=3,.group=64,.walks=2,.temp=1};return c;
}
static real predictive_entropy(const real*logits){
    double mx=logits[0],z=0,weighted=0;
    for(int i=1;i<256;i++)if(logits[i]>mx)mx=logits[i];
    double ee[256];int v=0;
#if VMATH
    if(cpu_fast)for(;v<256;v+=8){__m512d shift=_mm512_sub_pd(_mm512_cvtps_pd(_mm256_loadu_ps(logits+v)),_mm512_set1_pd(mx));_mm512_storeu_pd(ee+v,_ZGVeN8v_exp(shift));}
#endif
    for(;v<256;v++)ee[v]=exp(logits[v]-mx);
    for(int i=0;i<256;i++){double shift=logits[i]-mx,e=ee[i];z+=e;weighted+=e*shift;}
    double h=log(z)-weighted/z;return (real)(h>0?h:0);
}
static int write_aux_model(FILE*fp,Model*m){
    int ok=fwrite(&m->c,sizeof(Cfg),1,fp)==1;size_t z[N_FIELDS];real*p[N_FIELDS];sizes(&m->c,z);fields(&m->P,p);
    for(int i=0;i<N_FIELDS;i++)ok&=fwrite(p[i],sizeof(real),z[i],fp)==z[i];return ok;
}
static Model*read_aux_model(FILE*fp,int threads,int L){
    Cfg c;if(fread(&c,sizeof(Cfg),1,fp)!=1)return NULL;
    if((c.fw_mask&~0x5555)||c.fff_gate<0||c.fff_gate>1||c.vocab!=256||c.d<1||c.d>512||c.n_layer<1||c.n_layer>16||c.D<1||c.D>10||c.T<1||c.T>16||c.n<1||c.n>64||c.p<1||c.p>64||c.K<1||c.K>16||c.group<1||c.d%c.group)return NULL;
    Model*m=model_new(&c,threads,L);size_t z[N_FIELDS];real*p[N_FIELDS];sizes(&c,z);fields(&m->P,p);
    for(int i=0;i<N_FIELDS;i++)if(fread(p[i],sizeof(real),z[i],fp)!=z[i])return NULL;
    model_requantize(m);return m;
}

/* Row-major learned projections with exact manual adjoints. */
static void lin_fwd(const real*W,const real*x,real*y,int N,int in,int out){
    for(int t=0;t<N;t++)for(int j=0;j<out;j++)y[(size_t)t*out+j]=dotr(W+(size_t)j*in,x+(size_t)t*in,in);
}
static void lin_bwd(const real*W,real*G,const real*x,const real*dy,real*dx,int N,int in,int out){
    outer_rows(G,dy,x,N,out,in);
#if FAST
    if(runtime_fast && in%16==0){
        /* Keep each input-gradient tile in registers across output rows. */
        for(int t=0;t<N;t++)for(int i=0;i<in;i+=16){
            __m512 acc=_mm512_loadu_ps(dx+(size_t)t*in+i);
            for(int j=0;j<out;j++)acc=_mm512_fmadd_ps(_mm512_set1_ps(dy[(size_t)t*out+j]),_mm512_loadu_ps(W+(size_t)j*in+i),acc);
            _mm512_storeu_ps(dx+(size_t)t*in+i,acc);
        }return;
    }
#endif
    for(int t=0;t<N;t++)for(int j=0;j<out;j++)axpy(dx+(size_t)t*in,dy[(size_t)t*out+j],W+(size_t)j*in,in);
}
static size_t ca_size(int d){return (size_t)4*d*d+2*d;}
typedef struct {int d,H,span;real *qn,*kn,*q,*k,*v,*z,*out,*a,*qr,*kr,*dq,*dk,*dv,*dz,*dnq,*dnk;} CA;
static CA*ca_new(int qmax,int kmax,int d,int H,int span){
    CA*c=calloc(1,sizeof(*c));c->d=d;c->H=H;c->span=span;
#define QA(name) c->name=calloc((size_t)qmax*d,sizeof(real))
#define KA(name) c->name=calloc((size_t)kmax*d,sizeof(real))
    QA(qn);QA(q);QA(z);QA(out);QA(dq);QA(dz);QA(dnq);KA(kn);KA(k);KA(v);KA(dk);KA(dv);KA(dnk);
#undef QA
#undef KA
    c->a=calloc((size_t)qmax*H*span,sizeof(real));c->qr=calloc(qmax,sizeof(real));c->kr=calloc(kmax,sizeof(real));return c;
}
static void ca_keys(const real*P,const real*x,int nk,CA*c){
    int d=c->d;const real*norm=P+(size_t)4*d*d+d;
    for(int t=0;t<nk;t++)rmsnorm_fwd(x+(size_t)t*d,norm,c->kn+(size_t)t*d,d,c->kr+t);
    lin_fwd(P+(size_t)d*d,c->kn,c->k,nk,d,d);lin_fwd(P+(size_t)2*d*d,c->kn,c->v,nk,d,d);
}
static void ca_queries_shared(const real*P,const real*x,int nq,const int*lo,const int*hi,CA*c,const CA*shared){
    int d=c->d,H=c->H,hd=d/H;const real*norm=P+(size_t)4*d*d;real scale=1/sqrt((real)hd);
    if(shared){memcpy(c->q,shared->q,(size_t)nq*d*sizeof(real));}
    else {for(int t=0;t<nq;t++)rmsnorm_fwd(x+(size_t)t*d,norm,c->qn+(size_t)t*d,d,c->qr+t);
    lin_fwd(P,c->qn,c->q,nq,d,d);}memset(c->z,0,(size_t)nq*d*sizeof(real));
    for(int t=0;t<nq;t++)for(int h=0;h<H;h++){
        real*aa=c->a+((size_t)t*H+h)*c->span;real mx=-INFINITY,total=0;
        for(int j=lo[t];j<hi[t];j++){real v=dotr(c->q+(size_t)t*d+h*hd,c->k+(size_t)j*d+h*hd,hd)*scale;aa[j-lo[t]]=v;if(v>mx)mx=v;}
        for(int j=lo[t];j<hi[t];j++){aa[j-lo[t]]=exp(aa[j-lo[t]]-mx);total+=aa[j-lo[t]];}
        for(int j=lo[t];j<hi[t];j++){aa[j-lo[t]]/=total;axpy(c->z+(size_t)t*d+h*hd,aa[j-lo[t]],c->v+(size_t)j*d+h*hd,hd);}
    }
    lin_fwd(P+(size_t)3*d*d,c->z,c->out,nq,d,d);
    for(size_t i=0;i<(size_t)nq*d;i++)c->out[i]+=x[i];
}
static void ca_queries(const real*P,const real*x,int nq,const int*lo,const int*hi,CA*c){ca_queries_shared(P,x,nq,lo,hi,c,NULL);}
static void ca_fwd(const real*P,const real*qsrc,int nq,const real*ksrc,int nk,const int*lo,const int*hi,CA*c){ca_keys(P,ksrc,nk,c);ca_queries(P,qsrc,nq,lo,hi,c);}
static void ca_bwd(const real*P,real*G,const real*qsrc,int nq,const real*ksrc,int nk,const int*lo,const int*hi,const real*dy,real*dqsrc,real*dksrc,CA*c){
    int d=c->d,H=c->H,hd=d/H;real scale=1/sqrt((real)hd);
    memset(c->dz,0,(size_t)nq*d*sizeof(real));memset(c->dq,0,(size_t)nq*d*sizeof(real));memset(c->dk,0,(size_t)nk*d*sizeof(real));memset(c->dv,0,(size_t)nk*d*sizeof(real));
    for(size_t i=0;i<(size_t)nq*d;i++)dqsrc[i]+=dy[i];
    lin_bwd(P+(size_t)3*d*d,G+(size_t)3*d*d,c->z,dy,c->dz,nq,d,d);
    for(int t=0;t<nq;t++)for(int h=0;h<H;h++){
        real*aa=c->a+((size_t)t*H+h)*c->span;real avg=0;
        for(int j=lo[t];j<hi[t];j++)avg+=aa[j-lo[t]]*dotr(c->dz+(size_t)t*d+h*hd,c->v+(size_t)j*d+h*hd,hd);
        for(int j=lo[t];j<hi[t];j++){
            real a=aa[j-lo[t]],ds=a*(dotr(c->dz+(size_t)t*d+h*hd,c->v+(size_t)j*d+h*hd,hd)-avg)*scale;
            axpy(c->dv+(size_t)j*d+h*hd,a,c->dz+(size_t)t*d+h*hd,hd);
            axpy(c->dq+(size_t)t*d+h*hd,ds,c->k+(size_t)j*d+h*hd,hd);
            axpy(c->dk+(size_t)j*d+h*hd,ds,c->q+(size_t)t*d+h*hd,hd);
        }
    }
    memset(c->dnq,0,(size_t)nq*d*sizeof(real));memset(c->dnk,0,(size_t)nk*d*sizeof(real));
    lin_bwd(P,G,c->qn,c->dq,c->dnq,nq,d,d);
    lin_bwd(P+(size_t)d*d,G+(size_t)d*d,c->kn,c->dk,c->dnk,nk,d,d);
    lin_bwd(P+(size_t)2*d*d,G+(size_t)2*d*d,c->kn,c->dv,c->dnk,nk,d,d);
    for(int t=0;t<nq;t++)rmsnorm_bwd(qsrc+(size_t)t*d,P+(size_t)4*d*d,c->qr[t],c->dnq+(size_t)t*d,dqsrc+(size_t)t*d,G+(size_t)4*d*d,d);
    for(int t=0;t<nk;t++)rmsnorm_bwd(ksrc+(size_t)t*d,P+(size_t)4*d*d+d,c->kr[t],c->dnk+(size_t)t*d,dksrc+(size_t)t*d,G+(size_t)4*d*d+d,d);
}

/* Small ternary separable mixer. Codes are {-1,0,+1}, row scales are fixed.
 * Batch neighbors[t*K+k]: source position or -1 for zero padding.
 * NULL neighbors means causal temporal offsets 0..K-1. Arbitrary 2-D
 * stencils use the same kernel; callers must supply geometrically valid maps.
 */
#define TC_GROUP 4
#define TC_TAPS 3
static int tc_continuous=0;
static size_t tc_size(int d,int K){return (size_t)(K+TC_GROUP+1)*d;}
typedef struct {int d,K;real*gate;real invk;int8_t*q;uint16_t*pos,*neg;} TCCode;
typedef struct {real*dw,*mid,*out,*grad;} TCWork;
static TCCode*tc_code_new(int d,int K){
    TCCode*c=calloc(1,sizeof(*c));c->d=d;c->K=K;c->gate=calloc(d,sizeof(real));c->invk=1/sqrt((real)K);c->q=calloc((size_t)(K+TC_GROUP)*d,1);
    c->pos=calloc((size_t)(K+TC_GROUP)*((d+15)/16),sizeof(uint16_t));c->neg=calloc((size_t)(K+TC_GROUP)*((d+15)/16),sizeof(uint16_t));return c;
}
static TCWork*tc_work_new(int L,int d){TCWork*w=calloc(1,sizeof(*w));w->dw=calloc((size_t)L*d,sizeof(real));w->mid=calloc((size_t)L*d,sizeof(real));w->out=calloc((size_t)L*d,sizeof(real));w->grad=calloc((size_t)L*d,sizeof(real));return w;}
static void tc_pack(const real*P,TCCode*c){
    int d=c->d,K=c->K,C=(d+15)/16;
    for(int i=0;i<d;i++)c->gate[i]=tanh(P[(K+TC_GROUP)*d+i]);
    memset(c->pos,0,(size_t)(K+TC_GROUP)*C*sizeof(uint16_t));memset(c->neg,0,(size_t)(K+TC_GROUP)*C*sizeof(uint16_t));
    for(int r=0;r<K+TC_GROUP;r++)for(int i=0;i<d;i++){
        real a=P[r*d+i];int q=a>(real).5?1:(a<(real)-.5?-1:0);c->q[r*d+i]=q;
        if(q>0)c->pos[r*C+i/16]|=(uint16_t)(1u<<(i%16));if(q<0)c->neg[r*C+i/16]|=(uint16_t)(1u<<(i%16));
    }
}
static void tc_init(real*P,int d,int K){
    for(int i=0;i<(K+TC_GROUP)*d;i++)P[i]=(2*rand01()-1);
    memset(P+(K+TC_GROUP)*d,0,d*sizeof(real));
}
static inline real tc_weight(const real*P,const TCCode*c,int r,int i){return tc_continuous?P[r*c->d+i]:(real)c->q[r*c->d+i];}
static void tc_one(const real*P,const TCCode*c,const real*const*src,const real*x,real*dw,real*mid,real*out){
    int d=c->d,K=c->K;real a=c->invk,b=(real).5;
#if FAST
    if(!tc_continuous&&d%16==0){
        int C=d/16;
        for(int i=0;i<d;i+=16){
            __m512 z=_mm512_setzero_ps();
            for(int k=0;k<K;k++)if(src[k]){
                __m512 v=_mm512_loadu_ps(src[k]+i);int r=k*C+i/16;
                z=_mm512_mask_add_ps(z,(__mmask16)c->pos[r],z,v);z=_mm512_mask_sub_ps(z,(__mmask16)c->neg[r],z,v);
            }
            _mm512_storeu_ps(dw+i,_mm512_mul_ps(z,_mm512_set1_ps(a)));
        }
        for(int i=0;i<d;i+=16){
            __m512 v=_mm512_loadu_ps(dw+i),z=_mm512_setzero_ps();
            const __m512i bases=_mm512_setr_epi32(0,0,0,0,4,4,4,4,8,8,8,8,12,12,12,12);
            for(int j=0;j<TC_GROUP;j++){
                __m512 q=_mm512_permutexvar_ps(_mm512_add_epi32(bases,_mm512_set1_epi32(j)),v);int r=(K+j)*C+i/16;
                z=_mm512_mask_add_ps(z,(__mmask16)c->pos[r],z,q);z=_mm512_mask_sub_ps(z,(__mmask16)c->neg[r],z,q);
            }
            _mm512_storeu_ps(mid+i,_mm512_mul_ps(z,_mm512_set1_ps(b)));
        }
    }else
#endif
    {
        for(int i=0;i<d;i++){real z=0;for(int k=0;k<K;k++)if(src[k])z+=tc_weight(P,c,k,i)*src[k][i];dw[i]=a*z;}
        for(int i=0;i<d;i++){real z=0;for(int j=0;j<TC_GROUP;j++)z+=tc_weight(P,c,K+j,i)*dw[(i/TC_GROUP)*TC_GROUP+j];mid[i]=b*z;}
    }
    if(tc_continuous){for(int i=0;i<d;i++)out[i]=x[i]+tanh(P[(K+TC_GROUP)*d+i])*mid[i];}
    else for(int i=0;i<d;i++)out[i]=x[i]+c->gate[i]*mid[i];
}
/* Noncausal 2-D five-point stencil: center, left, right, up, down.
 * Precompute once for a fixed environment grid; no wrap across row edges. */
static void tc_grid5_neighbors(int*map,int height,int width){
    for(int r=0;r<height;r++)for(int c=0;c<width;c++){
        int t=r*width+c;map[5*t]=t;map[5*t+1]=c?t-1:-1;
        map[5*t+2]=c+1<width?t+1:-1;map[5*t+3]=r?t-width:-1;
        map[5*t+4]=r+1<height?t+width:-1;
    }
}
static void tc_forward(const real*P,const TCCode*c,const real*x,int L,const int*neighbors,TCWork*w){
    int d=c->d,K=c->K;
    for(int t=0;t<L;t++){
        const real*src[K];for(int k=0;k<K;k++){int j=neighbors?neighbors[t*K+k]:t-k;src[k]=j>=0&&j<L?x+(size_t)j*d:NULL;}
        tc_one(P,c,src,x+(size_t)t*d,w->dw+(size_t)t*d,w->mid+(size_t)t*d,w->out+(size_t)t*d);
    }
}
static void tc_backward(const real*P,real*G,const TCCode*c,const real*x,const real*dy,real*dx,int L,const int*neighbors,const TCWork*w){
    int d=c->d,K=c->K;real a=c->invk,b=(real).5;
    memcpy(dx,dy,(size_t)L*d*sizeof(real));
    for(int t=0;t<L;t++){
        real dd[d];memset(dd,0,sizeof(dd));
        for(int i=0;i<d;i++){
            real gate=tc_continuous?tanh(P[(K+TC_GROUP)*d+i]):c->gate[i],dm=dy[(size_t)t*d+i]*gate*b;
            G[(K+TC_GROUP)*d+i]+=dy[(size_t)t*d+i]*w->mid[(size_t)t*d+i]*(1-gate*gate);
            for(int j=0;j<TC_GROUP;j++){
                int v=(i/TC_GROUP)*TC_GROUP+j;G[(K+j)*d+i]+=dm*w->dw[(size_t)t*d+v];dd[v]+=dm*tc_weight(P,c,K+j,i);
            }
        }
        for(int k=0;k<K;k++){
            int j=neighbors?neighbors[t*K+k]:t-k;if(j<0||j>=L)continue;
            for(int i=0;i<d;i++){
                real grad=a*dd[i];G[k*d+i]+=grad*x[(size_t)j*d+i];dx[(size_t)j*d+i]+=grad*tc_weight(P,c,k,i);
            }
        }
    }
}
/* Streaming history stores past UNMIXED inputs, newest first. */
static void tc_stream(const real*P,const TCCode*c,const real*x,real*history,TCWork*w){
    int d=c->d,K=c->K;const real*src[K];src[0]=x;for(int k=1;k<K;k++)src[k]=history+(k-1)*d;
    tc_one(P,c,src,x,w->dw,w->mid,w->out);
    if(K>1){memmove(history+d,history,(size_t)(K-2)*d*sizeof(real));memcpy(history,x,d*sizeof(real));}
}

#define NG 6

typedef struct {int width,heads,bins,maxpatch,enc_layers,dec_layers,local_depth,local_trees,local_state,neural;real threshold,target;int mixer;} FConfig;
typedef struct {int width,heads,bins,maxpatch,enc_layers,dec_layers,local_depth,local_trees,local_state,neural;real threshold,target;} OldFConfig;
typedef struct {
    TCWork *mix; CA *enc,*dec;real *mean,*init,*kv,*dinit,*dmean,*dkv,*eg;
    int *lo,*hi,*dlo,*dhi,*starts,*lens,*map,*hash;
    real *entropy_logits,*entropy;int *ignored;
    int M;
} FWork;
typedef struct {
    TCCode *mixcode; FConfig c;Model *global,*encoder,*decoder,*entropy;FWork **ws;
    size_t dense,er,dp,ce,cd,mixoff,hashrows,hashsize;
    real *p,*g,*m,*v,**tg;unsigned char *hit,**thit;
    int threads,Lmax;
    void*muon_jobs;int muon_njobs,muon_cached_block;
} Faithful;
static uint32_t ngram_id(const int*t,int pos,int n,int bins){
    uint64_t h=1469598103934665603ULL;
    for(int i=pos-n+1;i<=pos;i++){h^=(uint64_t)(t[i]+1);h*=1099511628211ULL;}
    return (uint32_t)(h%(uint64_t)bins);
}
static void hash_input(Faithful*f,Work*w,FWork*b,const int*t,int L){
    int d=f->c.width;
    for(int i=0;i<L;i++){
        int count=1;real*x=w->x+(size_t)i*d;memcpy(x,f->encoder->P.embed+(size_t)t[i]*d,d*sizeof(real));
        for(int z=0;z<NG;z++){
            int row=-1;if(i+1>=z+3&&f->c.bins){row=z*f->c.bins+ngram_id(t,i,z+3,f->c.bins);count++;axpy(x,1,f->p+f->dense+(size_t)row*d,d);}
            b->hash[i*NG+z]=row;
        }
        for(int j=0;j<d;j++)x[j]/=count;
    }
}
static void core_fwd(Model*m,int th,int L,int training){
    Work*w=m->ws[th];w->k=m->k;w->no_alt=!training;
    if(!L)return;
    for(int l=0;l<m->c.n_layer;l++){
        if(!(l&1))carry_pre(&m->c,w,l,w->h);
        layer_fwd(&m->c,&m->P,&m->Q,l,L,w->x,w->h,w->tp+l,w);
        if(!(l&1))carry_post(&m->c,w,l,L,w->h);
    }
}
static void core_bwd(Model*m,int th,int L){
    Work*w=m->ws[th];if(!L)return;
    for(int l=m->c.n_layer-1;l>=0;l--){
        if(!(l&1))memset(w->lam,0,(size_t)m->c.T*NODES(&m->c)*m->c.n*m->c.p*sizeof(real));
        layer_bwd(&m->c,&m->P,&m->Q,m->Gt+th,l,L,w->g,w->lam,m->hits+node_rows(&m->c)*th+node_off(&m->c,l),w->tp+l,w);
    }
}
static FWork*fw_new(int L,int d,int heads,int maxpatch){
    FWork*w=calloc(1,sizeof(*w));w->mix=tc_work_new(L,2*d);w->enc=ca_new(2*L,L,d,heads,maxpatch);w->dec=ca_new(L,2*L,d,heads,2);
#define R(name,n) w->name=calloc((size_t)(n),sizeof(real))
#define I(name,n) w->name=calloc((size_t)(n),sizeof(int))
    R(mean,L*d);R(init,2*L*d);R(kv,2*L*d);R(dinit,2*L*d);R(dmean,L*d);R(dkv,2*L*d);R(eg,L*d);
    R(entropy_logits,L*256);R(entropy,L);I(ignored,L);for(int i=0;i<L;i++)w->ignored[i]=-1;
    I(lo,2*L);I(hi,2*L);I(dlo,L);I(dhi,L);I(starts,L);I(lens,L);I(map,L);I(hash,NG*L);
#undef R
#undef I
    return w;
}
static Faithful*faithful_new(Cfg global,FConfig cfg,int threads,int L,int initialize){
    Faithful*f=calloc(1,sizeof(*f));f->c=cfg;f->threads=threads;f->Lmax=L;int d=cfg.width,g=global.d;
    if((global.fff_gate&&cfg.local_state>cfg.local_trees*16)||g!=2*d||d%cfg.heads){fprintf(stderr,"global width must equal twice local width; heads must divide width\n");exit(2);}
    Cfg local=global;local.fw_mask=0;local.d=d;local.n_layer=cfg.enc_layers;local.T=cfg.local_trees;local.D=cfg.local_depth;local.n=cfg.local_state;local.p=16;local.group=d;
    f->global=model_new(&global,threads,L);f->encoder=model_new(&local,threads,L);local.n_layer=cfg.dec_layers;f->decoder=model_new(&local,threads,L);
    f->er=0;f->dp=(size_t)g*d;f->ce=f->dp+(size_t)g*g;f->cd=f->ce+ca_size(d);f->mixoff=f->cd+ca_size(d);f->dense=f->mixoff+(cfg.mixer?tc_size(g,TC_TAPS):0);f->mixcode=tc_code_new(g,TC_TAPS);
    f->hashrows=(size_t)NG*cfg.bins;f->hashsize=f->hashrows*d;size_t n=f->dense+f->hashsize;
    f->p=calloc(n,sizeof(real));f->g=calloc(n,sizeof(real));f->m=calloc(n,sizeof(real));f->v=calloc(n,sizeof(real));f->hit=calloc(f->hashrows,1);
    f->tg=calloc(threads,sizeof(real*));f->thit=calloc(threads,sizeof(unsigned char*));f->ws=calloc(threads,sizeof(FWork*));
    for(int t=0;t<threads;t++){f->tg[t]=calloc(n,sizeof(real));f->thit[t]=calloc(f->hashrows,1);f->ws[t]=fw_new(L,d,cfg.heads,cfg.maxpatch);}
    if(initialize){
        params_init(&global,&f->global->P);params_init(&f->encoder->c,&f->encoder->P);params_init(&f->decoder->c,&f->decoder->P);
        for(size_t i=0;i<f->mixoff;i++)f->p[i]=(2*rand01()-1)/sqrt((real)d);
        for(int i=0;i<g*g;i++)f->p[f->dp+i]=(2*rand01()-1)/sqrt((real)g);
        for(int i=0;i<2*d;i++){f->p[f->ce+4*d*d+i]=1;f->p[f->cd+4*d*d+i]=1;}
        for(size_t i=f->dense;i<n;i++)f->p[i]=(real)(0.02*randn());
        model_requantize(f->global);model_requantize(f->encoder);model_requantize(f->decoder);
        if(cfg.neural){Cfg ec=neural_entropy_cfg();f->entropy=model_new(&ec,threads,L);params_init(&ec,&f->entropy->P);model_requantize(f->entropy);}
        if(cfg.mixer){tc_init(f->p+f->mixoff,g,TC_TAPS);tc_pack(f->p+f->mixoff,f->mixcode);}
    }return f;
}
static void faithful_forward(Faithful*f,int th,const int*t,int L,int training){
    int d=f->c.width,g=2*d;FWork*b=f->ws[th];Work*e=f->encoder->ws[th],*z=f->global->ws[th],*dec=f->decoder->ws[th];
    if(f->c.neural){int ns;eval_seq(f->entropy,t,b->ignored,L,f->entropy->ws[th],&ns,b->entropy_logits);for(int i=0;i<L;i++)b->entropy[i]=predictive_entropy(b->entropy_logits+(size_t)i*256);}
    hash_input(f,e,b,t,L);core_fwd(f->encoder,th,L,training);
    int M=0,start=0,last=-1;
    for(int i=0;i<L;i++){
        if(i-start+1>=f->c.maxpatch||b->entropy[i]>f->c.threshold){
            b->starts[M]=start;b->lens[M]=i-start+1;
            real*mean=b->mean+(size_t)M*d;memset(mean,0,d*sizeof(real));
            for(int j=start;j<=i;j++)axpy(mean,(real)1/b->lens[M],e->x+(size_t)j*d,d);
            for(int s=0;s<2;s++){b->lo[2*M+s]=start;b->hi[2*M+s]=i+1;}
            last=M++;start=i+1;
        }
        b->map[i]=last;b->dlo[i]=last<0?0:2*last;b->dhi[i]=last<0?0:2*last+2;
    }b->M=M;
    lin_fwd(f->p+f->er,b->mean,b->init,M,d,g);
    ca_fwd(f->p+f->ce,b->init,2*M,e->x,L,b->lo,b->hi,b->enc);
    if(f->c.mixer==2)tc_forward(f->p+f->mixoff,f->mixcode,b->enc->out,M,NULL,b->mix);
    memcpy(z->x,f->c.mixer==2?b->mix->out:b->enc->out,(size_t)M*g*sizeof(real));core_fwd(f->global,th,M,training);
    if(f->c.mixer==1)tc_forward(f->p+f->mixoff,f->mixcode,z->x,M,NULL,b->mix);
    lin_fwd(f->p+f->dp,f->c.mixer==1?b->mix->out:z->x,b->kv,M,g,g);
    ca_fwd(f->p+f->cd,e->x,L,b->kv,2*M,b->dlo,b->dhi,b->dec);
    memcpy(dec->x,b->dec->out,(size_t)L*d*sizeof(real));core_fwd(f->decoder,th,L,training);
}
static real faithful_train_seq(Faithful*f,int th,const int*t,const int*y,int L,int*ns){
    int d=f->c.width,g=2*d;FWork*b=f->ws[th];Work*e=f->encoder->ws[th],*z=f->global->ws[th],*dec=f->decoder->ws[th];
    faithful_forward(f,th,t,L,1);int M=b->M;
    Params hp=f->decoder->P,hg=f->decoder->Gt[th];hp.embed=f->encoder->P.embed;hg.embed=f->encoder->Gt[th].embed;
    memset(dec->g,0,(size_t)L*d*sizeof(real));*ns=0;real loss=head_train(&f->decoder->c,&hp,&hg,dec->x,y,L,dec->g,ns,dec);
    core_bwd(f->decoder,th,L);
    memset(b->eg,0,(size_t)L*d*sizeof(real));memset(b->dkv,0,(size_t)M*g*sizeof(real));
    ca_bwd(f->p+f->cd,f->tg[th]+f->cd,e->x,L,b->kv,2*M,b->dlo,b->dhi,dec->g,b->eg,b->dkv,b->dec);
    memset(z->g,0,(size_t)M*g*sizeof(real));memset(b->mix->grad,0,(size_t)M*g*sizeof(real));
    lin_bwd(f->p+f->dp,f->tg[th]+f->dp,f->c.mixer==1?b->mix->out:z->x,b->dkv,f->c.mixer==1?b->mix->grad:z->g,M,g,g);
    if(f->c.mixer==1)tc_backward(f->p+f->mixoff,f->tg[th]+f->mixoff,f->mixcode,z->x,b->mix->grad,z->g,M,NULL,b->mix);
    core_bwd(f->global,th,M);
    if(f->c.mixer==2){
        memcpy(b->mix->grad,z->g,(size_t)M*g*sizeof(real));
        tc_backward(f->p+f->mixoff,f->tg[th]+f->mixoff,f->mixcode,b->enc->out,b->mix->grad,z->g,M,NULL,b->mix);
    }
    memset(b->dinit,0,(size_t)M*g*sizeof(real));
    ca_bwd(f->p+f->ce,f->tg[th]+f->ce,b->init,2*M,e->x,L,b->lo,b->hi,z->g,b->dinit,b->eg,b->enc);
    memset(b->dmean,0,(size_t)M*d*sizeof(real));lin_bwd(f->p+f->er,f->tg[th]+f->er,b->mean,b->dinit,b->dmean,M,d,g);
    for(int p=0;p<M;p++)for(int j=b->starts[p];j<b->starts[p]+b->lens[p];j++)axpy(b->eg+(size_t)j*d,(real)1/b->lens[p],b->dmean+(size_t)p*d,d);
    memcpy(e->g,b->eg,(size_t)L*d*sizeof(real));core_bwd(f->encoder,th,L);
    for(int i=0;i<L;i++){
        int count=1;for(int k=0;k<NG;k++)count+=b->hash[i*NG+k]>=0;
        real scale=(real)1/count;axpy(f->encoder->Gt[th].embed+(size_t)t[i]*d,scale,e->g+(size_t)i*d,d);
        for(int k=0;k<NG;k++){int row=b->hash[i*NG+k];if(row>=0){f->thit[th][row]=1;axpy(f->tg[th]+f->dense+(size_t)row*d,scale,e->g+(size_t)i*d,d);}}
    }return loss;
}
static double faithful_eval(Faithful*f,int th,const int*t,const int*y,int L,int*ns,real*logits){
    faithful_forward(f,th,t,L,0);Work*w=f->decoder->ws[th];int d=f->c.width;double loss=0;*ns=0;
    for(int i=0;i<L;i++){
        real rr;rmsnorm_fwd(w->x+(size_t)i*d,f->decoder->P.norm_f,w->xn,d,&rr);real mx=-INFINITY;double sum=0;
        for(int v=0;v<256;v++){w->lg[v]=dotr(f->encoder->P.embed+(size_t)v*d,w->xn,d);mx=fmax(mx,w->lg[v]);}
        if(logits)memcpy(logits+(size_t)i*256,w->lg,256*sizeof(real));
        if(y[i]<0)continue;for(int v=0;v<256;v++)sum+=exp(w->lg[v]-mx);loss+=log(sum)+mx-w->lg[y[i]];(*ns)++;
    }return loss;
}

/* CPU blockwise Muon: Nesterov EMA, five quintic Newton-Schulz iterations.
 * Blocks stay within a tree/matrix. Inactive routed rows are untouched.
 * This is blockwise active-row Muon, not full-matrix Muon. */
static int use_muon=1,muon_steps=5,muon_block=16;
static int ttt_freeze=0; /* test-time training: 1 keeps ternary matrices fixed */
static real muon_scale=3;
typedef struct {real*p,*g,*m,*q;unsigned char*hit;Model*model;int field,rows,d;size_t row;real*v;} MuonJob;
static void muon_apply(MuonJob*j,const Hyper*h,real scale){
    if(j->rows>32||j->d>512){fprintf(stderr,"Muon block exceeds CPU workspace bounds\n");exit(2);}
    int map[32],r=0,d=j->d;real X[32*512],Y[32*512],A[32*32],B[32*32];double norm=0;
    for(int i=0;i<j->rows;i++)if(!j->hit||j->hit[i]){
        map[r]=i;for(int k=0;k<d;k++){size_t off=(size_t)i*d+k;real g=j->g[off]*scale;
            real m=(real).95*j->m[off]+(real).05*g;j->m[off]=m;
            real v=(real).95*m+(real).05*g;X[r*d+k]=v;norm+=(double)v*v;j->g[off]=0;
        }r++;
    }
    if(!r)return;
    real inv=(real)(1/(sqrt(norm)+1e-7));for(int i=0;i<r*d;i++)X[i]*=inv;
    for(int it=0;it<muon_steps;it++){
        for(int i=0;i<r;i++)for(int k=0;k<=i;k++)A[i*r+k]=A[k*r+i]=dotr(X+i*d,X+k*d,d);
        for(int i=0;i<r;i++)for(int k=0;k<r;k++)B[i*r+k]=(real)-4.775*A[i*r+k]+(real)2.0315*dotr(A+i*r,A+k*r,r);
        for(int i=0;i<r;i++){
            for(int k=0;k<d;k++)Y[i*d+k]=(real)3.4445*X[i*d+k];
            for(int k=0;k<r;k++)axpy(Y+i*d,B[i*r+k],X+k*d,d);
        }memcpy(X,Y,(size_t)r*d*sizeof(real));
    }
    /* NorMuon: per-row second moment of the orthogonalized update, normalized per row, then rescaled
     * to keep the block's Frobenius norm. Row moments live in v[row*d]. */
    if(normuon&&j->v){double before=0,after=0;for(int i=0;i<r*d;i++)before+=(double)X[i]*X[i];
        for(int i=0;i<r;i++){real*xi=X+i*d;double ms=0;for(int k=0;k<d;k++)ms+=(double)xi[k]*xi[k];ms/=d;
            real*vv=j->v+(size_t)map[i]*d;*vv=(real)(.95*(*vv)+.05*ms);real inv=(real)(1/sqrt(*vv/(1-pow(.95,(double)h->step))+1e-12));
            for(int k=0;k<d;k++){xi[k]*=inv;after+=(double)xi[k]*xi[k];}}
        real sc=(real)sqrt(before/(after+1e-30));for(int i=0;i<r*d;i++)X[i]*=sc;}
    /* AdamW-comparable update RMS convention; separate multiplier available. */
    real lr=h->lr*muon_scale*(real)(.2*sqrt(fmax(r,d)));
    for(int i=0;i<r;i++){
        int at=map[i];real*p=j->p+(size_t)at*d;
        for(int k=0;k<d;k++)p[k]-=h->lr*h->weight_decay*p[k]+lr*X[i*d+k];
        if(j->q){real*q=j->q+(size_t)at*d;tss_ternary(p,q,1,d,j->model->c.group);
            Codes*c=j->model->k;if(c->ok){Plane*pl=j->field==4?&c->w_in:j->field==6?&c->w_out:j->field==8?&c->proj:&c->U;
                real*base=j->field==4?j->model->Q.w_in:j->field==6?j->model->Q.w_out:j->field==8?j->model->Q.proj:j->model->Q.U;
                plane_rows(base,pl,j->row+at,j->row+at+1,d,j->model->c.group);
            }
        }
    }
}
static void muon_jobs_add(MuonJob*jobs,int*count,real*p,real*g,real*m,real*q,unsigned char*hit,Model*model,int field,size_t row,int rows,int d){
    for(int at=0;at<rows;at+=muon_block){int nr=rows-at<muon_block?rows-at:muon_block;
        jobs[(*count)++]=(MuonJob){p+(size_t)at*d,g+(size_t)at*d,m+(size_t)at*d,q?q+(size_t)at*d:NULL,hit?hit+at:NULL,model,field,nr,d,row+at,NULL};
    }
}
static int muon_build(Faithful*f,MuonJob*jobs){
    int count=0;Model*models[3]={f->global,f->encoder,f->decoder};
    for(int z=0;z<3;z++){
        Model*m=models[z];Cfg*c=&m->c;real*p[N_FIELDS],*g[N_FIELDS],*mom[N_FIELDS],*q[N_FIELDS];fields(&m->P,p);fields(&m->G,g);fields(&m->M,mom);fields(&m->Q,q);
        for(int l=0;l<c->n_layer;l++){
            int n=(l&1)?fff_nodes(c):NODES(c);
            for(int tr=0;tr<c->T;tr++)for(int field=4;field<=6;field+=2){size_t row=node_off(c,l)+(size_t)tr*n;
                muon_jobs_add(jobs,&count,p[field]+row*c->d,g[field]+row*c->d,mom[field]+row*c->d,q[field]+row*c->d,m->hit+row,m,field,row,n,c->d);
            }
            for(int field=8;field<=10;field+=2){int nrow=field==8?PDIM(c):c->T*c->p;size_t row=(size_t)l*nrow;
                int active=(l&1)?(c->fff_gate?c->n:0):nrow;
                muon_jobs_add(jobs,&count,p[field]+row*c->d,g[field]+row*c->d,mom[field]+row*c->d,q[field]+row*c->d,NULL,m,field,row,active,c->d);
            }
        }
    }
    int d=f->c.width,gd=2*d;
    muon_jobs_add(jobs,&count,f->p+f->er,f->g+f->er,f->m+f->er,NULL,NULL,NULL,0,0,gd,d);
    muon_jobs_add(jobs,&count,f->p+f->dp,f->g+f->dp,f->m+f->dp,NULL,NULL,NULL,0,0,gd,gd);
    for(int ca=0;ca<2;ca++)for(int a=0;a<4;a++){size_t off=(ca?f->cd:f->ce)+(size_t)a*d*d;
        muon_jobs_add(jobs,&count,f->p+off,f->g+off,f->m+off,NULL,NULL,NULL,0,0,d,d);
    }return count;
}

static MuonJob*bm_jobs=NULL;static int bm_n=0;
static void byte_muon_update(const Hyper*h,real scale,int threads){
    Model*m=(Model*)byte_muon_model;if(!m)return;Cfg*c=&m->c;
    if(!bm_jobs){size_t cap=4*node_rows(c)+4*c->n_layer*(PDIM(c)+c->T*c->p)+64;bm_jobs=malloc(cap*sizeof(MuonJob));
        real*p[N_FIELDS],*g[N_FIELDS],*mom[N_FIELDS],*q[N_FIELDS],*vv[N_FIELDS];fields(&m->P,p);fields(&m->G,g);fields(&m->M,mom);fields(&m->Q,q);fields(&m->V,vv);
        for(int l=0;l<c->n_layer;l++){
            int n=(l&1)?fff_nodes(c):NODES(c);
            for(int tr=0;tr<c->T;tr++)for(int field=4;field<=6;field+=2){size_t row=node_off(c,l)+(size_t)tr*n;int st=bm_n;
                muon_jobs_add(bm_jobs,&bm_n,p[field]+row*c->d,g[field]+row*c->d,mom[field]+row*c->d,q[field]+row*c->d,m->hit+row,m,field,row,n,c->d);
                for(int j=st;j<bm_n;j++)bm_jobs[j].v=vv[field]+(bm_jobs[j].p-p[field]);}
            for(int field=8;field<=10;field+=2){int nrow=field==8?PDIM(c):c->T*c->p;size_t row=(size_t)l*nrow;int st=bm_n;
                int active=(l&1)?(c->fff_gate?c->n:0):nrow;
                muon_jobs_add(bm_jobs,&bm_n,p[field]+row*c->d,g[field]+row*c->d,mom[field]+row*c->d,q[field]+row*c->d,NULL,m,field,row,active,c->d);
                for(int j=st;j<bm_n;j++)bm_jobs[j].v=vv[field]+(bm_jobs[j].p-p[field]);}
        }}
#pragma omp parallel for num_threads(threads) schedule(static,1)
    for(int j=0;j<bm_n;j++)muon_apply(bm_jobs+j,h,scale);
}
/* One OpenMP team: sequence work, unified clipping, sparse row updates.
 * Free-budget n-gram weights still participate in clipping and AdamW. */
typedef struct {
    Model*m;size_t z[N_FIELDS],ends[N_FIELDS+1],chunk[N_FIELDS],rows;long jobs;
    real *gf[N_FIELDS],*pf[N_FIELDS],*mf[N_FIELDS],*vf[N_FIELDS],*tf[64][N_FIELDS];
} UpdateView;
static void update_view(Model*m,UpdateView*v){
    memset(v,0,sizeof(*v));v->m=m;v->rows=node_rows(&m->c);sizes(&m->c,v->z);
    fields(&m->G,v->gf);fields(&m->P,v->pf);fields(&m->M,v->mf);fields(&m->V,v->vf);
    for(int t=0;t<m->threads;t++)fields(m->Gt+t,v->tf[t]);
    for(int j=0;j<N_FIELDS;j++){v->chunk[j]=(j==8||j==10)?(size_t)m->c.d:4096;v->ends[j+1]=v->ends[j]+(node_field(j)?0:(v->z[j]+v->chunk[j]-1)/v->chunk[j]);}
    v->jobs=v->rows+v->ends[N_FIELDS];
}
static double reduce_job(UpdateView*v,long job){
    Model*m=v->m;int d=m->c.d;double nrm=0;
    if((size_t)job<v->rows){
        size_t r=job;unsigned char any=0;for(int j=0;j<m->threads;j++){any|=m->hits[v->rows*j+r];m->hits[v->rows*j+r]=0;}m->hit[r]=any;if(!any)return 0;
        for(int j=4;j<=7;j++){size_t width=(j==4||j==6)?d:1;nrm+=reduce_span(v->gf[j],v->tf,j,r*width,width,m->threads);}
    }else{
        size_t dj=job-v->rows;int j=0;while(dj>=v->ends[j+1])j++;size_t off=(dj-v->ends[j])*v->chunk[j],len=v->z[j]-off<v->chunk[j]?v->z[j]-off:v->chunk[j];nrm+=reduce_span(v->gf[j],v->tf,j,off,len,m->threads);
    }return nrm;
}
static void apply_job(UpdateView*v,long job,const Hyper*h,real scale,real c1,real c2){
    Model*m=v->m;int d=m->c.d;Codes*k=m->k;
    if((size_t)job<v->rows){
        size_t r=job;if(!m->hit[r])return;
        for(int j=4;j<=7;j++){if((use_muon||ttt_freeze)&&(j==4||j==6)){if(ttt_freeze)memset(v->gf[j]+r*d,0,d*sizeof(real));continue;}size_t width=(j==4||j==6)?d:1,off=r*width;adam(v->pf[j]+off,v->gf[j]+off,v->mf[j]+off,v->vf[j]+off,width,h,scale,c1,c2);}
        m->Q.A_log[r]=m->P.A_log[r];k->decay_rates[r]=exp(m->P.A_log[r]);
        if(!use_muon){tss_ternary(m->P.w_in+r*d,m->Q.w_in+r*d,1,d,m->c.group);tss_ternary(m->P.w_out+r*d,m->Q.w_out+r*d,1,d,m->c.group);
        if(k->ok){plane_rows(m->Q.w_in,&k->w_in,r,r+1,d,m->c.group);plane_rows(m->Q.w_out,&k->w_out,r,r+1,d,m->c.group);}}
    }else{
        size_t dj=job-v->rows;int j=0;while(dj>=v->ends[j+1])j++;size_t off=(dj-v->ends[j])*v->chunk[j],len=v->z[j]-off<v->chunk[j]?v->z[j]-off:v->chunk[j];
        if((use_muon||ttt_freeze)&&(j==8||j==10)){if(ttt_freeze)memset(v->gf[j]+off,0,len*sizeof(real));return;}
        adam(v->pf[j]+off,v->gf[j]+off,v->mf[j]+off,v->vf[j]+off,len,h,scale,c1,c2);
        if(j==8||j==10){real*qm=j==8?m->Q.proj:m->Q.U;size_t r=off/d;tss_ternary(v->pf[j]+off,qm+off,1,d,m->c.group);if(k->ok)plane_rows(qm,j==8?&k->proj:&k->U,r,r+1,d,m->c.group);}
    }
}
static real faithful_step(Faithful*f,const int*t,const int*y,int B,int L,const Hyper*h,long long*patches){
    UpdateView views[3];update_view(f->global,views);update_view(f->encoder,views+1);update_view(f->decoder,views+2);
    if(use_muon&&(!f->muon_jobs||f->muon_cached_block!=muon_block)){
        size_t cap=1024;for(int j=0;j<3;j++)cap+=4*views[j].rows+4*views[j].m->c.n_layer*(PDIM(&views[j].m->c)+views[j].m->c.T*views[j].m->c.p);
        free(f->muon_jobs);f->muon_jobs=malloc(cap*sizeof(MuonJob));f->muon_njobs=muon_build(f,f->muon_jobs);f->muon_cached_block=muon_block;
    }
    MuonJob*jobs=f->muon_jobs;int nj=f->muon_njobs;
    int scored=0,d=f->c.width;double nrm=0;real loss=0,scale=0;long long np=0;
    real c1=1-pow(h->beta1,(real)h->step),c2=1-pow(h->beta2,(real)h->step);
#pragma omp parallel num_threads(f->threads)
    {
#ifdef _OPENMP
        int th=omp_get_thread_num();
#else
        int th=0;
#endif
#pragma omp for schedule(static) reduction(+:loss,scored,np)
        for(int b=0;b<B;b++){
            for(int j=0;j<3;j++)tss_work_seed(views[j].m->ws[th],(uint64_t)h->step*1000003ULL+b+97*j);
            Work*gw=f->global->ws[th];gw->carry_h=carry_bank_h?carry_bank_h+(size_t)b*carry_sz_h:NULL;gw->carry_hist=carry_bank_hist?carry_bank_hist+(size_t)b*carry_sz_hist:NULL;
            int ns;loss+=faithful_train_seq(f,th,t+(size_t)b*L,y+(size_t)b*L,L,&ns);scored+=ns;np+=f->ws[th]->M;
            gw->carry_h=gw->carry_hist=NULL;
        }
        for(int j=0;j<3;j++){
#pragma omp for schedule(static,16) reduction(+:nrm)
            for(long job=0;job<views[j].jobs;job++)nrm+=reduce_job(views+j,job);
        }
#pragma omp for schedule(static) reduction(+:nrm)
        for(size_t i=0;i<f->dense;i++){
            real g=0;for(int j=0;j<f->threads;j++){g+=f->tg[j][i];f->tg[j][i]=0;}f->g[i]=g;nrm+=(double)g*g;
        }
#pragma omp for schedule(static,16) reduction(+:nrm)
        for(size_t r=0;r<f->hashrows;r++){
            unsigned char hit=0;for(int j=0;j<f->threads;j++){hit|=f->thit[j][r];f->thit[j][r]=0;}f->hit[r]=hit;if(!hit)continue;
            real*gg=f->g+f->dense+r*d;memset(gg,0,d*sizeof(real));
            for(int j=0;j<f->threads;j++){real*tg=f->tg[j]+f->dense+r*d;for(int k=0;k<d;k++){gg[k]+=tg[k];tg[k]=0;}}
            for(int k=0;k<d;k++)nrm+=(double)gg[k]*gg[k];
        }
#pragma omp single
        {scale=scored?(real)1/scored:0;double norm=sqrt(nrm)*scale;if(h->clip>0&&norm>h->clip)scale*=h->clip/(norm+1e-6);}
        for(int j=0;j<3;j++){
#pragma omp for schedule(static,16) nowait
            for(long job=0;job<views[j].jobs;job++)apply_job(views+j,job,h,scale,c1,c2);
        }
        if(use_muon){
#pragma omp for schedule(static) nowait
            for(int part=0;part<3;part++){
                size_t off=part==0?f->ce+4*d*d:part==1?f->cd+4*d*d:f->mixoff;
                size_t len=part<2?2*d:f->dense-f->mixoff;
                adam(f->p+off,f->g+off,f->m+off,f->v+off,len,h,scale,c1,c2);
            }
#pragma omp for schedule(static,1) nowait
            for(int j=0;j<nj;j++){if(ttt_freeze&&jobs[j].q)memset(jobs[j].g,0,(size_t)jobs[j].rows*jobs[j].d*sizeof(real));else muon_apply(jobs+j,h,scale);}
        }else{
#pragma omp for schedule(static) nowait
            for(size_t off=0;off<f->dense;off+=1024){size_t n=f->dense-off<1024?f->dense-off:1024;adam(f->p+off,f->g+off,f->m+off,f->v+off,n,h,scale,c1,c2);}
        }
#pragma omp for schedule(static,16)
        for(size_t r=0;r<f->hashrows;r++)if(f->hit[r]){size_t off=f->dense+r*d;adam(f->p+off,f->g+off,f->m+off,f->v+off,d,h,scale,c1,c2);}
    }
    if(f->c.mixer)tc_pack(f->p+f->mixoff,f->mixcode);
    if(patches)*patches+=np;return loss/scored;
}

typedef struct {TCWork*mix;real*mixhist;Gen*enc,*global,*dec,*entropy;CA*ce,*cd;real *input,*bytes,*mean,*init,*kv;int history[8],len;uint64_t seen,patches;} FStream;
static FStream*fstream_new(Faithful*f){
    int d=f->c.width;FStream*s=calloc(1,sizeof(*s));s->mix=tc_work_new(1,2*d);s->mixhist=calloc((TC_TAPS-1)*2*d,sizeof(real));s->enc=gen_new(&f->encoder->c);s->global=gen_new(&f->global->c);s->dec=gen_new(&f->decoder->c);
    if(f->c.neural)s->entropy=gen_new(&f->entropy->c);
    s->ce=ca_new(2,f->c.maxpatch,d,f->c.heads,f->c.maxpatch);s->cd=ca_new(1,2,d,f->c.heads,2);
    s->input=calloc(d,sizeof(real));s->bytes=calloc((size_t)f->c.maxpatch*d,sizeof(real));s->mean=calloc(d,sizeof(real));s->init=calloc(2*d,sizeof(real));s->kv=calloc(2*d,sizeof(real));return s;
}
static void fstream_logits(Faithful*f,FStream*s,real*logits){
    int d=f->c.width;real rr;rmsnorm_fwd(s->dec->x,f->decoder->P.norm_f,s->input,d,&rr);
    for(int v=0;v<256;v++)logits[v]=dotr(f->encoder->P.embed+(size_t)v*d,s->input,d);
}
static void fstream_step(Faithful*f,FStream*s,int token,real*logits){
    real nh=0;if(f->c.neural){real el[256];gen_step(f->entropy,s->entropy,token,el);nh=predictive_entropy(el);}
    int d=f->c.width,g=2*d;
    memmove(s->history,s->history+1,7*sizeof(int));s->history[7]=token;s->seen++;
    memcpy(s->input,f->encoder->P.embed+(size_t)token*d,d*sizeof(real));int count=1;
    for(int z=0;z<NG;z++)if(s->seen>=z+3&&f->c.bins){size_t row=z*f->c.bins+ngram_id(s->history,7,z+3,f->c.bins);axpy(s->input,1,f->p+f->dense+row*d,d);count++;}
    for(int i=0;i<d;i++)s->input[i]/=count;
    gen_core(f->encoder,s->enc,s->input,NULL);
    memcpy(s->bytes+(size_t)s->len*d,s->enc->x,d*sizeof(real));s->len++;
    if(s->len>=f->c.maxpatch||nh>f->c.threshold){
        memset(s->mean,0,d*sizeof(real));for(int i=0;i<s->len;i++)axpy(s->mean,(real)1/s->len,s->bytes+(size_t)i*d,d);
        lin_fwd(f->p+f->er,s->mean,s->init,1,d,g);int lo[2]={0,0},hi[2]={s->len,s->len};
        ca_fwd(f->p+f->ce,s->init,2,s->bytes,s->len,lo,hi,s->ce);
        if(f->c.mixer==2)tc_stream(f->p+f->mixoff,f->mixcode,s->ce->out,s->mixhist,s->mix);
        gen_core(f->global,s->global,f->c.mixer==2?s->mix->out:s->ce->out,NULL);
        if(f->c.mixer==1)tc_stream(f->p+f->mixoff,f->mixcode,s->global->x,s->mixhist,s->mix);
        lin_fwd(f->p+f->dp,f->c.mixer==1?s->mix->out:s->global->x,s->kv,1,g,g);
        ca_keys(f->p+f->cd,s->kv,2,s->cd);s->len=0;s->patches++;
    }
    int lo=0,hi=s->patches?2:0;ca_queries(f->p+f->cd,s->enc->x,1,&lo,&hi,s->cd);
    gen_core(f->decoder,s->dec,s->cd->out,NULL);
    if(logits)fstream_logits(f,s,logits);
}
static Faithful*faithful_load(const char*path,int threads,int L);
/* Streaming input chunks are I/O boundaries only: recurrent state persists. */
static int context_generate(const char*checkpoint,const char*path,const char*prompt,int count,real temp){
    FILE*fp=fopen(path,"rb");if(!fp){perror(path);return 1;}
    Faithful*f=faithful_load(checkpoint,1,8);
    Model*m=f?NULL:load(checkpoint,1,8);
    if(!f&&(!m||m->c.vocab!=256)){fprintf(stderr,"context-file requires a byte-vocabulary checkpoint\n");fclose(fp);return 2;}
    FStream*fs=f?fstream_new(f):NULL;Gen*gs=m?gen_new(&m->c):NULL;
    unsigned char buf[4096];real logits[256];uint64_t bytes=0;size_t n;double start=wall();
    while((n=fread(buf,1,sizeof(buf),fp))){for(size_t i=0;i<n;i++){
        if(f)fstream_step(f,fs,buf[i],NULL);else gen_step(m,gs,buf[i],NULL);
    }bytes+=n;}
    if(ferror(fp)){perror("context read");fclose(fp);return 1;}fclose(fp);
    fprintf(stderr,"{\"event\":\"context_ingested\",\"bytes\":%llu,\"seconds\":%.6f,\"state_reset_at_chunks\":false}\n",(unsigned long long)bytes,wall()-start);
    if(bytes){
        if(f)fstream_logits(f,fs,logits);
        else {real rr;rmsnorm_fwd(gs->x,m->P.norm_f,gs->u,m->c.d,&rr);for(int v=0;v<256;v++)logits[v]=dotr(m->P.embed+(size_t)v*m->c.d,gs->u,m->c.d);}
    }
    for(const unsigned char*q=(const unsigned char*)prompt;*q;q++){
        if(f)fstream_step(f,fs,*q,logits);else gen_step(m,gs,*q,logits);putchar(*q);
    }
    if(!bytes&&!*prompt){if(f)fstream_step(f,fs,'\n',logits);else gen_step(m,gs,'\n',logits);}
    for(int i=0;i<count;i++){int v=sample_logits(logits,256,temp);putchar(v);if(f)fstream_step(f,fs,v,logits);else gen_step(m,gs,v,logits);}
    putchar('\n');return 0;
}
static int faithful_save(Faithful*f,const char*path){
    FILE*fp=fopen(path,"wb");if(!fp)return -1;int rs=sizeof(real);int ok=fwrite(PROTECTED_FW&&f->global->c.fw_mask?"TPC1":"TCM1",1,4,fp)==4&&fwrite(&rs,sizeof(int),1,fp)==1&&fwrite(&f->global->c,sizeof(Cfg),1,fp)==1&&fwrite(&f->c,sizeof(FConfig),1,fp)==1;
    Model*mods[3]={f->global,f->encoder,f->decoder};
    for(int j=0;j<3;j++){size_t z[N_FIELDS];real*p[N_FIELDS];sizes(&mods[j]->c,z);fields(&mods[j]->P,p);for(int k=0;k<N_FIELDS;k++)ok&=fwrite(p[k],sizeof(real),z[k],fp)==z[k];}
    size_t total=f->dense+f->hashsize;ok&=fwrite(f->p,sizeof(real),total,fp)==total;if(f->c.neural)ok&=write_aux_model(fp,f->entropy);
    if(fclose(fp))ok=0;return ok?0:-1;
}
static Faithful*faithful_load(const char*path,int threads,int L){
    FILE*fp=fopen(path,"rb");if(!fp)return NULL;ssm_groups=1; /* BLT checkpoints always use shared SSM projections */char magic[4];int rs;Cfg gc;FConfig c;
    if(fread(magic,1,4,fp)!=4||(memcmp(magic,"TFB1",4)&&memcmp(magic,"TCM1",4)&&memcmp(magic,"TPC1",4))||fread(&rs,sizeof(int),1,fp)!=1||rs!=sizeof(real)||fread(&gc,sizeof(Cfg),1,fp)!=1){fclose(fp);return NULL;}
    if(gc.fw_mask && (PROTECTED_FW ? memcmp(magic,"TPC1",4) : !memcmp(magic,"TPC1",4))){fclose(fp);return NULL;}
    memset(&c,0,sizeof(c));
    if(!memcmp(magic,"TFB1",4)){OldFConfig old;if(fread(&old,sizeof(old),1,fp)!=1){fclose(fp);return NULL;}memcpy(&c,&old,sizeof(old));}
    else if(fread(&c,sizeof(c),1,fp)!=1){fclose(fp);return NULL;}
    if(c.mixer<0||c.mixer>2){fclose(fp);return NULL;}
    if((gc.fw_mask&~0x5555)||gc.fff_gate<0||gc.fff_gate>1||c.neural!=1||c.width<1||c.heads<1||c.maxpatch<1||c.maxpatch>1024||c.bins<0||c.bins>1048576||gc.d!=2*c.width){fclose(fp);return NULL;}
    Faithful*f=faithful_new(gc,c,threads,L,0);Model*mods[3]={f->global,f->encoder,f->decoder};int ok=1;
    for(int j=0;j<3;j++){size_t z[N_FIELDS];real*p[N_FIELDS];sizes(&mods[j]->c,z);fields(&mods[j]->P,p);for(int k=0;k<N_FIELDS;k++)ok&=fread(p[k],sizeof(real),z[k],fp)==z[k];model_requantize(mods[j]);}
    size_t total=f->dense+f->hashsize;ok&=fread(f->p,sizeof(real),total,fp)==total;if(c.neural){f->entropy=read_aux_model(fp,threads,L);ok&=f->entropy!=NULL;}fclose(fp);if(ok&&c.mixer)tc_pack(f->p+f->mixoff,f->mixcode);return ok?f:NULL;
}
static size_t core_parameters(const Cfg*c){size_t z[N_FIELDS],n=0;sizes(c,z);for(int i=0;i<N_FIELDS;i++)n+=z[i];n-=(size_t)(c->n_layer/2)*(c->K*c->d+c->T*fff_nodes(c)+PDIM(c)*c->d+c->T+c->T*c->p*c->d);if(c->fff_gate)n+=(size_t)(c->n_layer/2)*2*c->n*c->d;return n-(size_t)c->vocab*c->d-c->d;}

/* Pause tokens (--pause K): K think bytes (NUL) are inserted after every space in train and val text,
 * giving the model K extra recurrent steps before predicting each word's first byte. Predictions whose
 * target is a pause byte are not scored, so BPB counts real bytes only. */
static int pause_k=0;
#define PLAB(v) (pause_k&&(v)==0?-1:(int)(v))
static unsigned char*pause_expand(const unsigned char*r,long n,long*out){long m=0;for(long i=0;i<n;i++)m+=1+(r[i]==' ')*pause_k;
    unsigned char*o=malloc(m);long j=0;for(long i=0;i<n;i++){o[j++]=r[i];if(r[i]==' ')for(int k=0;k<pause_k;k++)o[j++]=0;}*out=m;return o;}
static double score_data(Faithful*f,Model*m,const unsigned char*raw,long start,long end,int L,int burn,long*count){
    int threads=f?f->threads:m->threads;long nw=(end-start-1)/L,total=0;double loss=0;
#pragma omp parallel for num_threads(threads) reduction(+:loss,total) schedule(static)
    for(long w=0;w<nw;w++){
#ifdef _OPENMP
        int th=omp_get_thread_num();
#else
        int th=0;
#endif
        int*t=malloc(L*sizeof(int)),*y=malloc(L*sizeof(int));for(int i=0;i<L;i++){t[i]=raw[start+w*L+i];y[i]=i<burn?-1:PLAB(raw[start+w*L+i+1]);}
        int ns;loss+=f?faithful_eval(f,th,t,y,L,&ns,NULL):eval_seq(m,t,y,L,m->ws[th],&ns,NULL);total+=ns;free(t);free(y);
    }*count=total;return loss/total/log(2.);
}


static double train_entropy_model(Model*m,const unsigned char*raw,long split,int B,int L,double seconds,int seed){
    int*t=malloc((size_t)B*L*sizeof(int)),*y=malloc((size_t)B*L*sizeof(int));uint64_t rng=((uint64_t)seed+19)*0xD1B54A32D192ED03ULL;
    Hyper h={.beta1=.9,.beta2=.98,.eps=1e-8,.weight_decay=.01,.clip=1};double elapsed=0,loss=0;long long seen=0;int step=0;
    while(elapsed<seconds){
        h.step=++step;h.lr=.003*fmin(1,(elapsed+.1)/fmin(5,seconds*.1))*(.1+.9*.5*(1+cos(3.141592653589793*fmin(1,elapsed/seconds))));
        for(int b=0;b<B;b++){long at=(long)(urand(&rng)*(split-L-1));for(int i=0;i<L;i++){t[b*L+i]=raw[at+i];y[b*L+i]=raw[at+i+1];}}
        double st=wall();loss=tss_train_step(&m->c,&m->P,&m->Q,&m->G,m->hit,&m->M,&m->V,m->Gt,m->hits,m->ws,m->threads,t,y,B,L,&h,m->k);elapsed+=wall()-st;seen+=(long long)B*L;
    }
    printf("{\"event\":\"entropy_training\",\"steps\":%d,\"train_seconds\":%.6f,\"train_bytes\":%lld,\"bytes_per_second\":%.3f,\"last_batch_bpc\":%.6f}\n",step,elapsed,seen,seen/elapsed,loss/log(2.));fflush(stdout);free(t);free(y);return elapsed;
}
static double entropy_patch_average(const real*h,int windows,int L,real threshold,int cap){
    long n=0;for(int w=0;w<windows;w++){int len=0;for(int i=0;i<L;i++){len++;if(len>=cap||h[(size_t)w*L+i]>threshold){n++;len=0;}}n+=len>0;}return (double)windows*L/n;
}
static double calibrate_neural(Faithful*f,const unsigned char*raw,long split,int L,double target){
    const int nw=128;real*h=malloc((size_t)nw*L*sizeof(real));
#pragma omp parallel for num_threads(f->threads) schedule(static)
    for(int w=0;w<nw;w++){
#ifdef _OPENMP
        int th=omp_get_thread_num();
#else
        int th=0;
#endif
        int*t=malloc(L*sizeof(int)),ns;long start=(long)(((double)w+.5)/nw*(split-L-1));for(int i=0;i<L;i++)t[i]=raw[start+i];FWork*b=f->ws[th];
        eval_seq(f->entropy,t,b->ignored,L,f->entropy->ws[th],&ns,b->entropy_logits);for(int i=0;i<L;i++)h[(size_t)w*L+i]=predictive_entropy(b->entropy_logits+(size_t)i*256);free(t);
    }
    real lo=0,hi=log(256);for(int j=0;j<28;j++){real mid=(lo+hi)/2;if(entropy_patch_average(h,nw,L,mid,f->c.maxpatch)<target)lo=mid;else hi=mid;}
    f->c.threshold=(lo+hi)/2;double avg=entropy_patch_average(h,nw,L,f->c.threshold,f->c.maxpatch);free(h);return avg;
}
static uint64_t param_fingerprint(Model*m){
    size_t z[N_FIELDS];real*p[N_FIELDS];sizes(&m->c,z);fields(&m->P,p);uint64_t h=1469598103934665603ULL;
    for(int f=0;f<N_FIELDS;f++){unsigned char*b=(unsigned char*)p[f];for(size_t j=0;j<z[f]*sizeof(real);j++){h^=b[j];h*=1099511628211ULL;}}return h;
}
static int check_neural(void){
    Cfg gc={.vocab=256,.d=128,.n_layer=2,.T=2,.D=3,.n=4,.p=16,.K=3,.group=128,.walks=2,.temp=1};
    FConfig cfg={.width=64,.heads=4,.bins=64,.maxpatch=5,.enc_layers=2,.dec_layers=2,.local_depth=3,.local_trees=2,.local_state=4,.neural=1,.mixer=2,.threshold=5.54,.target=4};
    int L=48,t[48],y[48],ns;for(int i=0;i<L;i++){t[i]=(13*i+7)%256;y[i]=(17*i+3)%256;}
    Faithful*f=faithful_new(gc,cfg,1,L,1);real*a=calloc(L*256,sizeof(real)),*b=calloc(L*256,sizeof(real)),lg[256];
    faithful_eval(f,0,t,y,L,&ns,a);double mean=0;for(int i=0;i<L;i++)mean+=f->ws[0]->entropy[i]/L;f->c.threshold=mean;
    faithful_eval(f,0,t,y,L,&ns,a);real original_h[48];memcpy(original_h,f->ws[0]->entropy,sizeof(original_h));int original_map[48];memcpy(original_map,f->ws[0]->map,sizeof(original_map));
    FStream*s=fstream_new(f);Gen*eg=gen_new(&f->entropy->c);double he=0,le=0;int disagree=0,closed=0,len=0;
    for(int i=0;i<L;i++){gen_step(f->entropy,eg,t[i],lg);real h=predictive_entropy(lg);he=fmax(he,fabs(h-original_h[i]));len++;if(len>=cfg.maxpatch||h>f->c.threshold){closed++;len=0;}disagree+=closed-1!=original_map[i];fstream_step(f,s,t[i],lg);for(int j=0;j<256;j++)le=fmax(le,fabs(lg[j]-a[(size_t)i*256+j]));}
    for(int i=24;i<L;i++)t[i]^=63;faithful_eval(f,0,t,y,L,&ns,b);double future=0;for(int i=0;i<24*256;i++)future=fmax(future,fabs(a[i]-b[i]));for(int i=0;i<24;i++)disagree+=f->ws[0]->map[i]!=original_map[i];
    int same_tail[48];memcpy(same_tail,t,sizeof(t));for(int i=0;i<L-2;i++)same_tail[i]^=127;
    faithful_eval(f,0,t,y,L,&ns,a);real h0=f->ws[0]->entropy[L-1];faithful_eval(f,0,same_tail,y,L,&ns,b);double contextual=fabs(h0-f->ws[0]->entropy[L-1]);
    uint64_t before=param_fingerprint(f->entropy);Hyper hp={.lr=.003,.beta1=.9,.beta2=.98,.eps=1e-8,.weight_decay=.01,.clip=1,.step=1};faithful_step(f,t,y,1,L,&hp,NULL);int frozen=before==param_fingerprint(f->entropy);
    faithful_eval(f,0,t,y,L,&ns,a);int saved=faithful_save(f,"/tmp/nanotss_neural_check.bin")==0;Faithful*r=faithful_load("/tmp/nanotss_neural_check.bin",1,L);double rt=INFINITY;int same=0;
    if(r){same=param_fingerprint(r->entropy)==before;faithful_eval(r,0,t,y,L,&ns,b);rt=0;for(int i=0;i<L*256;i++)rt=fmax(rt,fabs(a[i]-b[i]));}remove("/tmp/nanotss_neural_check.bin");
    printf("neural entropy streaming max_abs %.9g; boundary mismatches %d; main logits %.9g; future leakage %.9g\n",he,disagree,le,future);
    printf("same final two bytes, different earlier context: entropy delta %.9g\n",contextual);
    printf("frozen after main optimizer: %d; saved auxiliary weights identical: %d; checkpoint logits %.9g\n",frozen,same,rt);
    int ok=he<1e-4&&!disagree&&le<1e-3&&future<1e-6&&contextual>0&&frozen&&saved&&same&&rt==0;puts(ok?"NEURAL CHECK PASSED":"NEURAL CHECK FAILED");return !ok;
}
/* Finite differences on continuous weights/activations and fixed hard paths.
 * The production quantizers and sampled branch estimator are tested separately
 * by train/stream equivalence; their STE is intentionally not an exact derivative.
 */
static double check_objective(const Cfg*c,Params*P,int l,int L,Work*w,const real*x,const real*h,const real*gy,const real*lh){
    size_t S=(size_t)c->T*NODES(c)*c->n*c->p;memcpy(w->x,x,L*c->d*sizeof(real));memcpy(w->h,h,S*sizeof(real));w->k=NULL;w->no_alt=0;tss_work_seed(w,7);
    layer_fwd(c,P,P,l,L,w->x,w->h,w->tp+l,w);double s=0;for(int i=0;i<L*c->d;i++)s+=(double)gy[i]*w->x[i];
    if(!(l&1))for(size_t i=0;i<S;i++)s+=(double)lh[i]*w->h[i];return s;
}
static int check_block_gradient(int l){
    Cfg c={.vocab=16,.d=8,.n_layer=2,.T=2,.D=2,.n=3,.p=4,.K=3,.group=4,.walks=0,.temp=1,.fw_mask=0,.fff_gate=1};
    int L=4;size_t S=(size_t)c.T*NODES(&c)*c.n*c.p,z[N_FIELDS];sizes(&c,z);Params P,G;params_alloc(&c,&P);params_alloc(&c,&G);params_init(&c,&P);
    real *pf[N_FIELDS],*gf[N_FIELDS];fields(&P,pf);fields(&G,gf);
    /* Nonzero gates exercise every chain, rather than only identity init. */
    for(size_t i=0;i<z[8];i++)P.proj[i]=.09*sin((double)i+.3);
    for(size_t i=0;i<z[10];i++)P.U[i]=.07*cos((double)i+.4);
    for(size_t i=0;i<z[5];i++)P.bias[i]=.6+.1*sin((double)i);
    for(size_t i=0;i<z[9];i++)P.dt_bias[i]=-.7;
    real x[L*c.d],gy[L*c.d],h[S],lh[S];
    for(int i=0;i<L*c.d;i++){x[i]=.3+.2*sin(i*.41);gy[i]=.1*cos(i*.7);}
    for(size_t i=0;i<S;i++){h[i]=.2*sin(i*.3);lh[i]=.1*cos(i*.2);}
    Work*w=tss_work_new(&c,L);continuous_check=1;check_objective(&c,&P,l,L,w,x,h,gy,lh);
    memcpy(w->g,gy,sizeof(gy));memcpy(w->lam,lh,sizeof(lh));
    size_t V=(size_t)L*c.T*(c.D+2);memset(w->tp[l].bw,0,V*sizeof(real));
    unsigned char*hit=calloc(node_rows(&c),1);layer_bwd(&c,&P,&P,&G,l,L,w->g,w->lam,hit+node_off(&c,l),w->tp+l,w);
    real analytic_x[L*c.d],analytic_h[S];memcpy(analytic_x,w->g,sizeof(analytic_x));memcpy(analytic_h,w->lam,sizeof(analytic_h));
    double worst=0;int tested=0,bad=0;real eps=sizeof(real)==8?1e-5:1e-3;
    for(int f=2;f<N_FIELDS;f++)for(size_t i=0;i<z[f];i++){
        if(gf[f][i]==0)continue;real old=pf[f][i];pf[f][i]=old+eps;double a=check_objective(&c,&P,l,L,w,x,h,gy,lh);pf[f][i]=old-eps;double b=check_objective(&c,&P,l,L,w,x,h,gy,lh);pf[f][i]=old;
        double num=(a-b)/(2*eps),err=fabs(num-gf[f][i])/(1+fabs(num)+fabs(gf[f][i]));worst=fmax(worst,err);tested++;bad+=err>(sizeof(real)==8?1e-6:2e-3);
    }
    for(int i=0;i<L*c.d;i++){
        real old=x[i];x[i]=old+eps;double a=check_objective(&c,&P,l,L,w,x,h,gy,lh);x[i]=old-eps;double b=check_objective(&c,&P,l,L,w,x,h,gy,lh);x[i]=old;
        double num=(a-b)/(2*eps),err=fabs(num-analytic_x[i])/(1+fabs(num)+fabs(analytic_x[i]));worst=fmax(worst,err);tested++;bad+=err>(sizeof(real)==8?1e-6:2e-3);
    }
    if(!(l&1))for(size_t i=0;i<S;i++){
        real old=h[i];h[i]=old+eps;double a=check_objective(&c,&P,l,L,w,x,h,gy,lh);h[i]=old-eps;double b=check_objective(&c,&P,l,L,w,x,h,gy,lh);h[i]=old;
        double num=(a-b)/(2*eps),err=fabs(num-analytic_h[i])/(1+fabs(num)+fabs(analytic_h[i]));worst=fmax(worst,err);tested++;bad+=err>(sizeof(real)==8?1e-6:2e-3);
    }
    continuous_check=0;printf("%s continuous fixed-path gradient: %d coordinates; worst scaled %.9g; bad %d\n",l?"FFF gate":"Tree SSM",tested,worst,bad);return bad!=0;
}
/* Whole-model gradient check with looping: tss_train_seq gradients vs central differences of its loss. */
static int check_loops(int loops){
    int saved=n_loops;n_loops=loops;
    Cfg c={.vocab=16,.d=8,.n_layer=2,.T=2,.D=2,.n=3,.p=4,.K=3,.group=4,.walks=0,.temp=1,.fw_mask=0,.fff_gate=1};
    size_t z[N_FIELDS];sizes(&c,z);Params P,G,G2;params_alloc(&c,&P);params_alloc(&c,&G);params_alloc(&c,&G2);params_init(&c,&P);
    real*pf[N_FIELDS],*gf[N_FIELDS],*g2[N_FIELDS];fields(&P,pf);fields(&G,gf);fields(&G2,g2);
    for(size_t i=0;i<z[8];i++)P.proj[i]=.09*sin((double)i+.3);for(size_t i=0;i<z[10];i++)P.U[i]=.07*cos((double)i+.4);
    for(size_t i=0;i<z[5];i++)P.bias[i]=.6+.1*sin((double)i);for(size_t i=0;i<z[9];i++)P.dt_bias[i]=-.7;
    int L=6,tok[6]={1,5,3,7,2,9},lab[6]={5,3,7,2,9,4},ns;Work*w=tss_work_new(&c,L);unsigned char*hit=calloc(node_rows(&c),1);
    continuous_check=1;tss_work_seed(w,1);tss_train_seq(&c,&P,&P,&G,tok,lab,L,hit,&ns,w,NULL);
    double worst=0;int tested=0,bad=0;real eps=sizeof(real)==8?1e-6:1e-3;
    for(int f=0;f<N_FIELDS;f++)for(size_t i=0;i<z[f];i+=1+z[f]/40){
        real old=pf[f][i];pf[f][i]=old+eps;tss_work_seed(w,1);double a=tss_train_seq(&c,&P,&P,&G2,tok,lab,L,hit,&ns,w,NULL);
        pf[f][i]=old-eps;tss_work_seed(w,1);double b=tss_train_seq(&c,&P,&P,&G2,tok,lab,L,hit,&ns,w,NULL);pf[f][i]=old;
        double num=(a-b)/(2*eps),err=fabs(num-gf[f][i])/(1+fabs(num)+fabs(gf[f][i]));worst=fmax(worst,err);tested++;bad+=err>(sizeof(real)==8?1e-5:2e-2);
    }
    continuous_check=0;n_loops=saved;printf("loops=%d whole-model gradient: %d coordinates; worst scaled %.3g; bad %d\n",loops,tested,worst,bad);return bad!=0;
}
static int check_gated_fff(void){
    int bad=check_block_gradient(1);
    Cfg gc={.vocab=256,.d=128,.n_layer=2,.T=2,.D=3,.n=4,.p=16,.K=3,.group=128,.walks=2,.temp=1,.fw_mask=0,.fff_gate=1};
    FConfig cfg={.width=64,.heads=4,.bins=64,.maxpatch=5,.enc_layers=2,.dec_layers=2,.local_depth=3,.local_trees=2,.local_state=4,.neural=1,.mixer=2,.threshold=5.54,.target=4};
    int L=48,t[48],y[48],ns;for(int i=0;i<L;i++){t[i]=(13*i+7)%256;y[i]=(17*i+3)%256;}
    Faithful*f=faithful_new(gc,cfg,1,L,1);uint64_t frozen=param_fingerprint(f->entropy);
    const char*phrase="the tree state space learns. ";for(int i=0;i<L;i++){t[i]=(unsigned char)phrase[i%29];y[i]=(unsigned char)phrase[(i+1)%29];}
    Hyper hp={.lr=.003,.beta1=.9,.beta2=.98,.eps=1e-8,.weight_decay=.01,.clip=1};double first=0,last=0;
    for(int j=1;j<=60;j++){hp.step=j;last=faithful_step(f,t,y,1,L,&hp,NULL);if(j==1)first=last;}
    real*a=calloc(L*256,sizeof(real)),*b=calloc(L*256,sizeof(real)),lg[256];faithful_eval(f,0,t,y,L,&ns,a);FStream*s=fstream_new(f);double e=0;
    for(int i=0;i<L;i++){fstream_step(f,s,t[i],lg);for(int v=0;v<256;v++)e=fmax(e,fabs(lg[v]-a[i*256+v]));}
    for(int i=L/2;i<L;i++)t[i]^=63;faithful_eval(f,0,t,y,L,&ns,b);double future=0;for(int i=0;i<L/2*256;i++)future=fmax(future,fabs(a[i]-b[i]));
    for(int i=L/2;i<L;i++)t[i]^=63;
    int saved=faithful_save(f,"/tmp/nanotss_fff_check.bin")==0;Faithful*r=faithful_load("/tmp/nanotss_fff_check.bin",1,L);double rt=INFINITY;
    if(r){faithful_eval(r,0,t,y,L,&ns,b);rt=0;for(int i=0;i<L*256;i++)rt=fmax(rt,fabs(a[i]-b[i]));}remove("/tmp/nanotss_fff_check.bin");
    int frozen_ok=param_fingerprint(f->entropy)==frozen&&!f->entropy->c.fw_mask&&!f->entropy->c.fff_gate;
    printf("gated FFF BLT repeated phrase loss %.6f -> %.6f; trained stream max %.9g; future %.9g; checkpoint %.9g; entropy frozen %d\n",first,last,e,future,rt,frozen_ok);
    bad+=!(last<first*.6&&e<1e-4&&future==0&&saved&&rt==0&&frozen_ok);puts(bad?"GATED FFF CHECK FAILED":"GATED FFF CHECK PASSED");return !!bad;
}

static int check_fw_gradient(int l){
    Cfg c={.vocab=16,.d=8,.n_layer=2,.T=2,.D=2,.n=3,.p=4,.K=3,.group=4,.walks=0,.temp=1,.fw_mask=1,.fff_gate=1};
    int L=4;size_t S=(size_t)c.T*NODES(&c)*c.n*c.p,z[N_FIELDS];sizes(&c,z);Params P,G;params_alloc(&c,&P);params_alloc(&c,&G);params_init(&c,&P);
    real *pf[N_FIELDS],*gf[N_FIELDS];fields(&P,pf);fields(&G,gf);
    /* Nonzero gates exercise every chain, rather than only identity init. */
    for(size_t i=0;i<z[8];i++)P.proj[i]=.09*sin((double)i+.3);
    for(size_t i=0;i<z[10];i++)P.U[i]=.07*cos((double)i+.4);
    for(size_t i=0;i<z[5];i++)P.bias[i]=.6+.1*sin((double)i);
    for(size_t i=0;i<z[9];i++)P.dt_bias[i]=-.7;
    real x[L*c.d],gy[L*c.d],h[S],lh[S];
    for(int i=0;i<L*c.d;i++){x[i]=.3+.2*sin(i*.41);gy[i]=.1*cos(i*.7);}
    for(size_t i=0;i<S;i++){h[i]=.2*sin(i*.3);lh[i]=.1*cos(i*.2);}
    Work*w=tss_work_new(&c,L);continuous_check=1;check_objective(&c,&P,l,L,w,x,h,gy,lh);
    memcpy(w->g,gy,sizeof(gy));memcpy(w->lam,lh,sizeof(lh));
    size_t V=(size_t)L*c.T*(c.D+2);memset(w->tp[l].bw,0,V*sizeof(real));
    unsigned char*hit=calloc(node_rows(&c),1);layer_bwd(&c,&P,&P,&G,l,L,w->g,w->lam,hit+node_off(&c,l),w->tp+l,w);
    real analytic_x[L*c.d],analytic_h[S];memcpy(analytic_x,w->g,sizeof(analytic_x));memcpy(analytic_h,w->lam,sizeof(analytic_h));
    double worst=0;int tested=0,bad=0;real eps=sizeof(real)==8?1e-5:1e-3;
    for(int f=2;f<N_FIELDS;f++)for(size_t i=0;i<z[f];i++){
        if(gf[f][i]==0)continue;real old=pf[f][i];pf[f][i]=old+eps;double a=check_objective(&c,&P,l,L,w,x,h,gy,lh);pf[f][i]=old-eps;double b=check_objective(&c,&P,l,L,w,x,h,gy,lh);pf[f][i]=old;
        double num=(a-b)/(2*eps),err=fabs(num-gf[f][i])/(1+fabs(num)+fabs(gf[f][i]));worst=fmax(worst,err);tested++;bad+=err>(sizeof(real)==8?1e-6:2e-3);
    }
    for(int i=0;i<L*c.d;i++){
        real old=x[i];x[i]=old+eps;double a=check_objective(&c,&P,l,L,w,x,h,gy,lh);x[i]=old-eps;double b=check_objective(&c,&P,l,L,w,x,h,gy,lh);x[i]=old;
        double num=(a-b)/(2*eps),err=fabs(num-analytic_x[i])/(1+fabs(num)+fabs(analytic_x[i]));worst=fmax(worst,err);tested++;bad+=err>(sizeof(real)==8?1e-6:2e-3);
    }
    if(!(l&1))for(size_t i=0;i<S;i++){
        real old=h[i];h[i]=old+eps;double a=check_objective(&c,&P,l,L,w,x,h,gy,lh);h[i]=old-eps;double b=check_objective(&c,&P,l,L,w,x,h,gy,lh);h[i]=old;
        double num=(a-b)/(2*eps),err=fabs(num-analytic_h[i])/(1+fabs(num)+fabs(analytic_h[i]));worst=fmax(worst,err);tested++;bad+=err>(sizeof(real)==8?1e-6:2e-3);
    }
    continuous_check=0;printf("%s continuous fixed-path gradient: %d coordinates; worst scaled %.9g; bad %d\n","Fast-weight tree",tested,worst,bad);return bad!=0;
}
static int check_tree_fw(void){
    int bad=check_fw_gradient(0);
    Cfg gc={.vocab=256,.d=128,.n_layer=4,.T=2,.D=3,.n=4,.p=16,.K=3,.group=128,.walks=2,.temp=1,.fw_mask=1,.fff_gate=1};
    FConfig cfg={.width=64,.heads=4,.bins=64,.maxpatch=5,.enc_layers=2,.dec_layers=2,.local_depth=3,.local_trees=2,.local_state=4,.neural=1,.mixer=2,.threshold=5.54,.target=4};
    int L=48,t[48],y[48],ns;for(int i=0;i<L;i++){t[i]=(13*i+7)%256;y[i]=(17*i+3)%256;}
    Faithful*f=faithful_new(gc,cfg,1,L,1);uint64_t frozen=param_fingerprint(f->entropy);
    const char*phrase="the tree state space learns. ";for(int i=0;i<L;i++){t[i]=(unsigned char)phrase[i%29];y[i]=(unsigned char)phrase[(i+1)%29];}
    Hyper hp={.lr=.003,.beta1=.9,.beta2=.98,.eps=1e-8,.weight_decay=.01,.clip=1};double first=0,last=0;
    for(int j=1;j<=60;j++){hp.step=j;last=faithful_step(f,t,y,1,L,&hp,NULL);if(j==1)first=last;}
    real*a=calloc(L*256,sizeof(real)),*b=calloc(L*256,sizeof(real)),lg[256];faithful_eval(f,0,t,y,L,&ns,a);FStream*s=fstream_new(f);double e=0;
    for(int i=0;i<L;i++){fstream_step(f,s,t[i],lg);for(int v=0;v<256;v++)e=fmax(e,fabs(lg[v]-a[i*256+v]));}
    for(int i=L/2;i<L;i++)t[i]^=63;faithful_eval(f,0,t,y,L,&ns,b);double future=0;for(int i=0;i<L/2*256;i++)future=fmax(future,fabs(a[i]-b[i]));
    for(int i=L/2;i<L;i++)t[i]^=63;
    int saved=faithful_save(f,"/tmp/tree_fw_check.bin")==0;Faithful*r=faithful_load("/tmp/tree_fw_check.bin",1,L);double rt=INFINITY;
    if(r){faithful_eval(r,0,t,y,L,&ns,b);rt=0;for(int i=0;i<L*256;i++)rt=fmax(rt,fabs(a[i]-b[i]));}remove("/tmp/tree_fw_check.bin");
    int frozen_ok=param_fingerprint(f->entropy)==frozen&&!f->entropy->c.fw_mask&&!f->entropy->c.fff_gate;
    printf("three-block stack repeated phrase loss %.6f -> %.6f; trained stream max %.9g; future %.9g; checkpoint %.9g; entropy frozen %d\n",first,last,e,future,rt,frozen_ok);
    bad+=!(last<first*.6&&e<1e-4&&future==0&&saved&&rt==0&&frozen_ok);puts(bad?"TREE FW CHECK FAILED":"TREE FW CHECK PASSED");return !!bad;
}

/* ---------------- Test-time training: the context as a training set ----------------
 * The recurrent state carries what the context says; TTT also writes it into the
 * slow weights. Each chunk is scored first with the current weights, then the
 * weights take --ttt-steps optimizer steps on --ttt-batch windows: the newest
 * window plus uniform replay from everything already seen. No future byte is
 * ever trained on before it is scored. */
typedef struct {double lr,decay,clip,beta2;int steps,batch,chunk,ctx,stream,freeze,optim,rnd;} TTT;
typedef struct {real*p[3*N_FIELDS+1];size_t n[3*N_FIELDS+1];int k;} PSet;
/* TTT works on a BLT model, or on the byte-mode model ttt_byte when the Faithful pointer is NULL. */
static Model*ttt_byte=NULL;
static void fparams(Faithful*f,PSet*s){
    if(!f){size_t z[N_FIELDS];real*p[N_FIELDS];sizes(&ttt_byte->c,z);fields(&ttt_byte->P,p);s->k=0;for(int i=0;i<N_FIELDS;i++){s->p[s->k]=p[i];s->n[s->k++]=z[i];}return;}
    Model*mods[3]={f->global,f->encoder,f->decoder};s->k=0;
    for(int j=0;j<3;j++){size_t z[N_FIELDS];real*p[N_FIELDS];sizes(&mods[j]->c,z);fields(&mods[j]->P,p);for(int i=0;i<N_FIELDS;i++){s->p[s->k]=p[i];s->n[s->k++]=z[i];}}
    s->p[s->k]=f->p;s->n[s->k++]=f->dense+f->hashsize;
}
static real**snap_take(Faithful*f){
    PSet s;fparams(f,&s);real**b=calloc(s.k,sizeof(real*));
    for(int i=0;i<s.k;i++){b[i]=malloc(s.n[i]*sizeof(real));memcpy(b[i],s.p[i],s.n[i]*sizeof(real));}return b;
}
static void faithful_requantize(Faithful*f){
    if(!f){model_requantize(ttt_byte);return;}
    model_requantize(f->global);model_requantize(f->encoder);model_requantize(f->decoder);if(f->c.mixer)tc_pack(f->p+f->mixoff,f->mixcode);
}
static void opt_reset(Faithful*f){
    if(!f){size_t z[N_FIELDS];real*m[N_FIELDS],*v[N_FIELDS];sizes(&ttt_byte->c,z);fields(&ttt_byte->M,m);fields(&ttt_byte->V,v);
        for(int i=0;i<N_FIELDS;i++){memset(m[i],0,z[i]*sizeof(real));memset(v[i],0,z[i]*sizeof(real));}return;}
    Model*mods[3]={f->global,f->encoder,f->decoder};
    for(int j=0;j<3;j++){size_t z[N_FIELDS];real*m[N_FIELDS],*v[N_FIELDS];sizes(&mods[j]->c,z);fields(&mods[j]->M,m);fields(&mods[j]->V,v);
        for(int i=0;i<N_FIELDS;i++){memset(m[i],0,z[i]*sizeof(real));memset(v[i],0,z[i]*sizeof(real));}}
    size_t n=f->dense+f->hashsize;memset(f->m,0,n*sizeof(real));memset(f->v,0,n*sizeof(real));
}
static void snap_restore(Faithful*f,real**b){
    PSet s;fparams(f,&s);for(int i=0;i<s.k;i++)memcpy(s.p[i],b[i],s.n[i]*sizeof(real));faithful_requantize(f);opt_reset(f);
}
/* Pull the slow weights a fraction lambda back toward the snapshot (anchored dynamic evaluation). */
static void snap_pull(Faithful*f,real**b,real lambda){
    PSet s;fparams(f,&s);for(int i=0;i<s.k;i++)for(size_t j=0;j<s.n[i];j++)s.p[i][j]+=lambda*(b[i][j]-s.p[i][j]);faithful_requantize(f);
}
/* ---------------- Retrieval over the scratchpad ----------------
 * Chunks of RC bytes at stride RS are indexed by hashed word unigrams and bigrams.
 * Chunks join the index in order (retr_admit), so online use never sees unread text;
 * IDF comes from admitted chunks only. A query (recent text or a draft) ranks chunks by
 * sum over its features of tf_query * idf, divided by sqrt(chunk feature count). */
#define RC 256
#define RS 128
#define RB (1<<18)
typedef struct {long nch,admitted;long*start;int*nf;int*feat;long*foff;int*df;long*head,*next;int*pchunk;long np;float*acc;} Retr;
static int word_feats(const unsigned char*x,long n,int*out,int cap){
    int k=0;uint32_t prev=0;int have=0;long i=0;
    while(i<n&&k<cap-1){while(i<n&&!(isalpha(x[i])||x[i]=='\''))i++;if(i>=n)break;uint32_t h=2166136261u;long j=i;
        while(j<n&&(isalpha(x[j])||x[j]=='\'')){h=(h^(uint32_t)tolower(x[j]))*16777619u;j++;}
        out[k++]=(int)(h&(RB-1));if(have)out[k++]=(int)((h*31+prev*0x9E3779B1u)&(RB-1));prev=h;have=1;i=j;}
    return k;
}
static int cmp_int(const void*a,const void*b){return *(const int*)a-*(const int*)b;}
static Retr*retr_new(const unsigned char*raw,long n){
    Retr*r=calloc(1,sizeof(*r));r->nch=n<RC?1:(n-RC)/RS+1;r->start=malloc(r->nch*sizeof(long));r->nf=malloc(r->nch*sizeof(int));r->foff=malloc((r->nch+1)*sizeof(long));
    int buf[4*RC];long tot=0;int*tmp=malloc((size_t)r->nch*2*RC*sizeof(int));
    for(long c=0;c<r->nch;c++){r->start[c]=c*RS;long len=r->start[c]+RC<=n?RC:n-r->start[c];int k=word_feats(raw+r->start[c],len,buf,4*RC);
        qsort(buf,k,sizeof(int),cmp_int);int u=0;for(int i=0;i<k;i++)if(!u||buf[i]!=buf[u-1])buf[u++]=buf[i];r->foff[c]=tot;r->nf[c]=u;memcpy(tmp+tot,buf,u*sizeof(int));tot+=u;}
    r->foff[r->nch]=tot;r->feat=tmp;r->df=calloc(RB,sizeof(int));r->head=malloc(RB*sizeof(long));for(int i=0;i<RB;i++)r->head[i]=-1;
    r->next=malloc(tot*sizeof(long));r->pchunk=malloc(tot*sizeof(int));r->acc=calloc(r->nch,sizeof(float));return r;
}
/* Admit every chunk that lies entirely inside raw[0..e]. */
static void retr_admit(Retr*r,long e){
    while(r->admitted<r->nch&&r->start[r->admitted]+RC<=e+1){long c=r->admitted++;
        for(long i=r->foff[c];i<r->foff[c+1];i++){int f=r->feat[i];r->df[f]++;r->pchunk[r->np]=c;r->next[r->np]=r->head[f];r->head[f]=r->np++;}}
}
/* Top-m admitted chunks for the query; returns count written to ids. */
static int retr_query(Retr*r,const unsigned char*q,long qn,int m,long*ids,long excl){
    if(!r->admitted)return 0;static int qf[1<<14];int k=word_feats(q,qn,qf,1<<14);if(!k)return 0;qsort(qf,k,sizeof(int),cmp_int);
    memset(r->acc,0,r->admitted*sizeof(float));
    for(int i=0;i<k;){int j=i;while(j<k&&qf[j]==qf[i])j++;int f=qf[i];if(r->df[f]){float w=(float)(j-i)*logf(1+(float)r->admitted/r->df[f]);
        for(long p=r->head[f];p>=0;p=r->next[p])r->acc[r->pchunk[p]]+=w;}i=j;}
    int got=0;for(long c=0;c<r->admitted;c++){if(excl>=0&&r->start[c]+RC>excl){r->acc[c]=0;continue;}float s=r->acc[c]/sqrtf((float)(r->nf[c]+1));r->acc[c]=s;if(s<=0)continue;
        if(got<m){int j=got++;while(j>0&&r->acc[ids[j-1]]<s){ids[j]=ids[j-1];j--;}ids[j]=c;}
        else if(s>r->acc[ids[m-1]]){int j=m-1;while(j>0&&r->acc[ids[j-1]]<s){ids[j]=ids[j-1];j--;}ids[j]=c;}}
    return got;
}

static Retr*ttt_retr=NULL;static int ttt_retr_m=16;static double ttt_retr_mix=.5;static long ttt_qlen=512;
static long ttt_ids[1024];static int ttt_nids=0;
static int ttt_use_retr=0;
static long ttt_step_count=0;static int ttt_win=256;
/* ttt_adaptive>0: steps per chunk follow surprise. With r = chunk loss / running mean of chunk
 * losses, steps = clamp(round(ttt_adaptive * r^2), 0, 2*ttt_adaptive): well-predicted chunks get
 * fewer steps, surprising ones more. */
static double ttt_adaptive=0; /* TTT training window length */
/* ttt_only bitmask of groups TTT may change (0 = all): 1 global stack, 2 encoder, 4 decoder,
 * 8 byte embedding/head, 16 hashed n-gram rows, 32 dense cross-attention/projection/mixer. */
static int ttt_only=0;static real**ttt_hold=NULL;static double ttt_lex_mult=1; /* scales the TTT update of byte embedding + hash rows */
static void ttt_hold_take(Faithful*f){PSet s;fparams(f,&s);if(!ttt_hold){ttt_hold=calloc(s.k,sizeof(real*));for(int i=0;i<s.k;i++)ttt_hold[i]=malloc(s.n[i]*sizeof(real));}
    for(int i=0;i<s.k;i++)memcpy(ttt_hold[i],s.p[i],s.n[i]*sizeof(real));}
static void ttt_hold_restore(Faithful*f){PSet s;fparams(f,&s);
    for(int i=0;i<s.k-1;i++){int grp=i<N_FIELDS?1:i<2*N_FIELDS?2:4;if(i==N_FIELDS)grp=8;if(!(ttt_only&grp))memcpy(s.p[i],ttt_hold[i],s.n[i]*sizeof(real));}
    int k=s.k-1;if(!(ttt_only&32))memcpy(s.p[k],ttt_hold[k],f->dense*sizeof(real));if(!(ttt_only&16))memcpy(s.p[k]+f->dense,ttt_hold[k]+f->dense,f->hashsize*sizeof(real));
    faithful_requantize(f);}
static void ttt_lex_scale(Faithful*f){PSet s;fparams(f,&s);real a=(real)ttt_lex_mult;
    real*e=s.p[N_FIELDS],*he=ttt_hold[N_FIELDS];for(size_t j=0;j<s.n[N_FIELDS];j++)e[j]=he[j]+a*(e[j]-he[j]);
    int k=s.k-1;real*hp=s.p[k]+f->dense,*hh=ttt_hold[k]+f->dense;for(size_t j=0;j<f->hashsize;j++)hp[j]=hh[j]+a*(hp[j]-hh[j]);}
static long ttt_reset=0;static int ttt_reset_mask=0; /* streaming diagnostics: zero chosen states every ttt_reset bytes */
static void gen_reset(const Cfg*c,Gen*g){memset(g->h,0,node_rows(c)*c->n*c->p*sizeof(real));memset(g->hist,0,(size_t)c->n_layer*(c->K>1?c->K-1:1)*c->d*sizeof(real));memset(g->R,0,c->p*sizeof(real));memset(g->Racc,0,c->p*sizeof(real));}
static void fstream_reset(Faithful*f,FStream*s,int mask){
    if((mask&1)&&s->entropy)gen_reset(&f->entropy->c,s->entropy);if(mask&2)gen_reset(&f->encoder->c,s->enc);
    if(mask&4){gen_reset(&f->global->c,s->global);memset(s->mixhist,0,(TC_TAPS-1)*2*f->c.width*sizeof(real));}if(mask&8)gen_reset(&f->decoder->c,s->dec);
    for(int l=0;l<f->global->c.n_layer;l+=2)if(mask&(16<<l)){const Cfg*c=&f->global->c;size_t a=node_off(c,l)*c->n*c->p,z=(node_off(c,l+1)-node_off(c,l))*c->n*c->p;memset(s->global->h+a,0,z*sizeof(real));}
}
static const unsigned char*ttt_ctrl=NULL;static long ttt_ctrl_n=0;static long ttt_ranges[64];static int ttt_nranges=0; /* control: train on this corpus instead of the context */
/* Train on windows whose labels lie inside raw[0..e]. */
static double ttt_train(Faithful*f,const unsigned char*raw,long e,const TTT*o,uint64_t*rng,int*t,int*y){
    int Lm=f?f->Lmax:ttt_byte->Lmax,L=Lm<ttt_win?Lm:ttt_win;if(e<L)return 0;double loss=0;
    int save_muon=use_muon,save_freeze=ttt_freeze;use_muon=o->optim;ttt_freeze=o->freeze;
    if(ttt_retr&&!ttt_ctrl){retr_admit(ttt_retr,e);long q0=e-ttt_qlen<0?0:e-ttt_qlen;ttt_nids=retr_query(ttt_retr,raw+q0,e-q0,ttt_retr_m,ttt_ids,q0);}
    for(int it=0;it<o->steps;it++){
        for(int b=0;b<o->batch;b++){
            if(ttt_ctrl){long at=(long)(urand(rng)*(ttt_ctrl_n-L-1));for(int i=0;i<L;i++){t[b*L+i]=ttt_ctrl[at+i];y[b*L+i]=ttt_ctrl[at+i+1];}continue;}
            long at=b==0&&!o->rnd?e-L:(long)(urand(rng)*(e-L+1));
            if(ttt_retr&&ttt_nids&&!(b==0&&!o->rnd)&&urand(rng)<ttt_retr_mix)at=ttt_retr->start[ttt_ids[(int)(urand(rng)*ttt_nids)%ttt_nids]];
            if(at>e-L)at=e-L;
            for(int i=0;i<L;i++){t[b*L+i]=raw[at+i];y[b*L+i]=raw[at+i+1];}}
        Hyper h={.lr=o->lr,.beta1=.9,.beta2=o->beta2,.eps=1e-8,.weight_decay=0,.clip=o->clip,.step=++ttt_step_count};
        if(!f){Model*m=ttt_byte;loss+=tss_train_step(&m->c,&m->P,&m->Q,&m->G,m->hit,&m->M,&m->V,m->Gt,m->hits,m->ws,m->threads,t,y,o->batch,L,&h,m->k);continue;}
        if(ttt_only||ttt_lex_mult!=1)ttt_hold_take(f);
        loss+=faithful_step(f,t,y,o->batch,L,&h,NULL);
        if(ttt_lex_mult!=1)ttt_lex_scale(f);
        if(ttt_only)ttt_hold_restore(f);
    }
    use_muon=save_muon;ttt_freeze=save_freeze;return loss/o->steps;
}
/* ---------------- Match model (PAQ-style long-range copy) ----------------
 * Hashes the last MM_MIN bytes to the most recent earlier position with the same bytes, verifies
 * the match backwards, and predicts the byte that followed it. The prediction is mixed into the
 * model's distribution as p = lam*[x==pred] + (1-lam)*p_lm, with lam learned online per
 * match-length bucket by gradient steps on the log loss. Strictly causal: when predicting
 * raw[i+1] it has seen raw[0..i] only. */
#define MM_MIN 5
#define MM_BITS 22
typedef struct{int on,bits;const unsigned char*buf;long*tab;long ptr;int len;double lam[16];long learn_from;double lr;} Match;
static Match mm={.on=0,.bits=MM_BITS,.lr=.02};
static int mm_bucket(int len){int b=0;while(len>0&&b<15){len>>=1;b++;}return b;}
static void mmx_reset(Match*m,const unsigned char*buf,int keep_lam){
    if(!m->bits)m->bits=MM_BITS;if(!m->tab)m->tab=malloc(sizeof(long)<<m->bits);memset(m->tab,0,sizeof(long)<<m->bits);m->buf=buf;m->ptr=0;m->len=0;
    if(!keep_lam)for(int b=0;b<16;b++)m->lam[b]=b<3?0:b<4?.1:b<5?.3:.6;m->learn_from=0;
}
static void mm_reset(const unsigned char*buf){mmx_reset(&mm,buf,0);}
static uint32_t mm_hash(const unsigned char*x,int bits){uint32_t h=2166136261u;for(int k=0;k<MM_MIN;k++)h=(h^x[k])*16777619u;return h>>(32-bits);}
/* Incorporate byte buf[i]; afterwards the match predicts buf[i+1]. Call for every i in order. */
static void mmx_advance(Match*m,long i){
    const unsigned char*x=m->buf;
    if(m->len>0&&x[m->ptr]==x[i]){m->ptr++;m->len++;}else m->len=0;
    if(i>=MM_MIN-1){uint32_t h=mm_hash(x+i-MM_MIN+1,m->bits);
        if(m->len==0){long c=m->tab[h];if(c>0){int k=0;while(k<400&&c-1-k>=0&&x[c-1-k]==x[i-k])k++;if(k>=MM_MIN){m->len=k;m->ptr=c;}}}
        m->tab[h]=i+1;}
}
static void mm_advance(long i){mmx_advance(&mm,i);}
static int mmx_pred(const Match*m){return m->len>0?m->buf[m->ptr]:-1;}
/* Loss (nats) of target y under the mixture, given model logits; learns lam when pos>=learn_from. */
static double mm_mix_loss(const real*l,int y,long pos){
    double mx=-1e30,z=0;for(int v=0;v<256;v++)mx=l[v]>mx?l[v]:mx;for(int v=0;v<256;v++)z+=exp(l[v]-mx);
    double py=exp(l[y]-mx)/z;int pr=mm.on?mmx_pred(&mm):-1;if(pr<0)return -log(py);
    int b=mm_bucket(mm.len);double lam=mm.lam[b],d=pr==y,p=lam*d+(1-lam)*py;if(p<1e-12)p=1e-12;
    if(pos>=mm.learn_from){lam+=mm.lr*(d-py)/p;mm.lam[b]=lam<0?0:lam>.999?.999:lam;}
    return -log(p);
}
/* Mix a match prediction into a logits vector (used when sampling). */
static void mmx_mix_logits(const Match*m,real*l){
    int pr=mmx_pred(m);if(pr<0)return;double lam=m->lam[mm_bucket(m->len)];if(lam<=0)return;
    double mx=-1e30,z=0;for(int v=0;v<256;v++)mx=l[v]>mx?l[v]:mx;for(int v=0;v<256;v++)z+=exp(l[v]-mx);
    for(int v=0;v<256;v++){double p=(1-lam)*exp(l[v]-mx)/z+(v==pr?lam:0);l[v]=(real)log(p>1e-30?p:1e-30);}
}
/* Online score-then-adapt over raw[0..n). Fills per-position byte losses (nats) if loss!=NULL.
 * Returns bits per byte over positions >= skip. */
static double ttt_run(Faithful*f,const unsigned char*raw,long n,const TTT*o,real**anchor,float*loss,long skip,double*train_seconds){
    int C=o->chunk,W=o->ctx+C;if(W>(f?f->Lmax:ttt_byte->Lmax)){fprintf(stderr,"ttt: ctx+chunk exceeds Lmax\n");exit(2);}
    int*t=malloc((size_t)(W>256?W:256)*(o->batch>1?o->batch:1)*sizeof(int)),*y=malloc((size_t)(W>256?W:256)*(o->batch>1?o->batch:1)*sizeof(int));
    real*lg=malloc((size_t)W*256*sizeof(real));uint64_t rng=0x9E3779B97F4A7C15ULL;double tot=0,tsec=0;long cnt=0;
    if(mm.on){mm_reset(raw);mm.learn_from=o->lr>0?0:skip;}
    FStream*fs=o->stream&&f?fstream_new(f):NULL;Gen*bg=o->stream&&!f?gen_new(&ttt_byte->c):NULL;real slg[256];int have=0;
    double ema=0;
    for(long s=0;s<n-1;s+=C){
        long e=s+C<n-1?s+C:n-1;double ctot=tot;long ccnt=cnt;
        if(fs||bg){
            for(long i=s;i<e;i++){
                if(!have){if(fs)fstream_step(f,fs,raw[i],slg);else gen_step(ttt_byte,bg,raw[i],slg);}
                if(mm.on)mm_advance(i);
                double li=mm_mix_loss(slg,raw[i+1],i+1);if(loss)loss[i+1]=li;if(i+1>=skip){tot+=li;cnt++;}
                if(fs&&ttt_reset&&(i+1)%ttt_reset==0)fstream_reset(f,fs,ttt_reset_mask);
                if(fs)fstream_step(f,fs,raw[i+1],slg);else gen_step(ttt_byte,bg,raw[i+1],slg);have=1;
            }
        }else{
            long a=s-o->ctx<0?0:s-o->ctx;int Lw=(int)(e-a);
            for(int i=0;i<Lw;i++){t[i]=raw[a+i];y[i]=a+i<s?-1:raw[a+i+1];}
            int ns;if(f)faithful_eval(f,0,t,y,Lw,&ns,lg);else eval_seq(ttt_byte,t,y,Lw,ttt_byte->ws[0],&ns,lg);
            for(long i=s;i<e;i++){real*l=lg+(size_t)(i-a)*256;
                if(mm.on)mm_advance(i);
                double li=mm_mix_loss(l,raw[i+1],i+1);if(loss)loss[i+1]=li;if(i+1>=skip){tot+=li;cnt++;}}
        }
        if(o->lr>0){double st=wall();TTT oo=*o;
            if(ttt_adaptive>0&&cnt>ccnt){double cl=(tot-ctot)/(cnt-ccnt);ema=ema>0?.95*ema+.05*cl:cl;double r=cl/ema;
                int k=(int)floor(ttt_adaptive*r*r+.5);int mx=(int)(2*ttt_adaptive+.5);oo.steps=k<0?0:k>mx?mx:k;}
            if(oo.steps>0)ttt_train(f,raw,e,&oo,&rng,t,y);if(anchor&&o->decay>0)snap_pull(f,anchor,o->decay);tsec+=wall()-st;}
    }
    free(t);free(y);free(lg);if(train_seconds)*train_seconds=tsec;return cnt?tot/cnt/log(2.):0;
}
static unsigned char*read_file(const char*path,long*n){
    FILE*fp=fopen(path,"rb");if(!fp){perror(path);return NULL;}fseek(fp,0,SEEK_END);*n=ftell(fp);rewind(fp);unsigned char*r=malloc(*n+1);
    if(fread(r,1,*n,fp)!=(size_t)*n){fclose(fp);free(r);return NULL;}fclose(fp);return r;
}
/* --ttt-eval FILE: score a document online. Prints overall bpc and a log2 position curve. */
static int ttt_eval_main(const char*ckpt,const char*path,long off,long len,const TTT*o,int threads,int docs,long docstride,const char*ctrl){
    long n;unsigned char*all=read_file(path,&n);if(!all)return 1;
    if(ctrl){unsigned char*c=read_file(ctrl,&ttt_ctrl_n);if(!c)return 1;ttt_ctrl_n=(long)(ttt_ctrl_n*.9);ttt_ctrl=c;}
    if(off<0)off=(long)(n*.9);if(len<=0||off+len>n)len=n-off;
    int W=o->ctx+o->chunk;Faithful*f=faithful_load(ckpt,threads,W>256?W:256);
    if(!f){ttt_byte=load(ckpt,threads,W>256?W:256);if(!ttt_byte||ttt_byte->c.vocab!=256){fprintf(stderr,"cannot load checkpoint\n");return 1;}}
    real**base=snap_take(f);int nb=0;while((1L<<nb)<len)nb++;double*bs=calloc(nb+1,sizeof(double));long*bc=calloc(nb+1,sizeof(long));
    float*loss=malloc((size_t)len*sizeof(float));double total=0,tsec=0;long tc=0;
    for(int d=0;d<docs;d++){
        long start=off+d*docstride;if(start+len>n)break;
        snap_restore(f,base);double ts,st=wall();for(long i=0;i<len;i++)loss[i]=-1;
        if(ttt_retr_m>0&&ttt_retr_mix>0&&ttt_use_retr){ttt_retr=retr_new(all+start,len);}
        double bpc=ttt_run(f,all+start,len,o,base,loss,0,&ts);tsec+=ts;
        for(long i=1;i<len;i++){int b=0;while((2L<<b)<=i)b++;bs[b]+=loss[i];bc[b]++;total+=loss[i];tc++;}
        for(int r=0;r<ttt_nranges;r++){double rs=0;long rc=0;for(long i=ttt_ranges[2*r];i<ttt_ranges[2*r+1]&&i<len;i++)if(loss[i]>=0){rs+=loss[i];rc++;}
            printf("{\"event\":\"ttt_range\",\"doc\":%d,\"from\":%ld,\"to\":%ld,\"bpc\":%.6f}\n",d,ttt_ranges[2*r],ttt_ranges[2*r+1],rc?rs/rc/log(2.):0);}
        printf("{\"event\":\"ttt_doc\",\"doc\":%d,\"start\":%ld,\"bytes\":%ld,\"bpc\":%.6f,\"seconds\":%.3f,\"train_seconds\":%.3f}\n",d,start,len,bpc,wall()-st,ts);fflush(stdout);
    }
    for(int b=0;b<=nb;b++)if(bc[b])printf("{\"event\":\"ttt_curve\",\"from\":%ld,\"to\":%ld,\"bytes\":%ld,\"bpc\":%.6f}\n",b?1L<<b:1,2L<<b,bc[b],bs[b]/bc[b]/log(2.));
    printf("{\"event\":\"ttt_eval\",\"lr\":%g,\"steps\":%d,\"batch\":%d,\"chunk\":%d,\"ctx\":%d,\"stream\":%d,\"freeze\":%d,\"optim\":\"%s\",\"decay\":%g,\"control\":%d,\"bytes\":%ld,\"bpc\":%.6f,\"train_seconds\":%.3f}\n",
        o->lr,o->steps,o->batch,o->chunk,o->ctx,o->stream,o->freeze,o->optim?"muon":"adamw",o->decay,ttt_ctrl!=NULL,tc,total/tc/log(2.),tsec);
    return 0;
}

/* --ttt-fit FILE: read the first fraction of each document as context for E epochs of TTT,
 * then score the held-out remainder (windowed, weights fixed) after every epoch. */
static long ttt_ctxlen=0; /* >0: use only the last ttt_ctxlen context bytes before the query */
/* Held-out query BPC. With the match model on, score sequentially so the match table holds the
 * whole context (lam learns on the query only); otherwise the parallel windowed scorer. */
static double fit_score(Faithful*f,const unsigned char*raw,long c,long n,long*cnt){
    if(!mm.on)return score_data(f,ttt_byte,raw,c,n,256,64,cnt);
    TTT e={.lr=0,.ctx=64,.chunk=192,.steps=1,.batch=1};*cnt=n-1-c;return ttt_run(f,raw,n,&e,NULL,NULL,c,NULL);
}
static int ttt_fit_main(const char*ckpt,const char*paths,double frac,int epochs,const TTT*o0,int threads){
    Faithful*f=faithful_load(ckpt,threads,256);if(!f){ttt_byte=load(ckpt,threads,256);if(!ttt_byte){fprintf(stderr,"cannot load checkpoint\n");return 1;}}
    real**base=snap_take(f);TTT o=*o0;o.rnd=1;int L=256;
    int*t=malloc((size_t)L*o.batch*sizeof(int)),*y=malloc((size_t)L*o.batch*sizeof(int));
    char buf[4096];strncpy(buf,paths,sizeof(buf)-1);buf[sizeof(buf)-1]=0;
    for(char*path=strtok(buf,",");path;path=strtok(NULL,",")){
        long n;unsigned char*full=read_file(path,&n);if(!full)return 1;long c=(long)(n*frac);unsigned char*raw=full;
        if(ttt_ctxlen>0&&ttt_ctxlen<c){raw=full+(c-ttt_ctxlen);n-=c-ttt_ctxlen;c=ttt_ctxlen;}
        snap_restore(f,base);uint64_t rng=0x243F6A8885A308D3ULL;long cnt;double tsec=0;
        double q=fit_score(f,raw,c,n,&cnt);
        printf("{\"event\":\"ttt_fit\",\"doc\":\"%s\",\"context_bytes\":%ld,\"query_bytes\":%ld,\"epoch\":0,\"query_bpc\":%.6f,\"train_seconds\":0}\n",path,c,cnt,q);fflush(stdout);
        long per=c/((long)L*o.batch);if(per<1)per=1;o.steps=per;
        for(int e=1;e<=epochs;e++){
            double st=wall();ttt_train(f,raw,c,&o,&rng,t,y);if(o.decay>0)snap_pull(f,base,o.decay);tsec+=wall()-st;
            if((e&(e-1))==0||e==epochs){q=fit_score(f,raw,c,n,&cnt);
            printf("{\"event\":\"ttt_fit\",\"doc\":\"%s\",\"context_bytes\":%ld,\"query_bytes\":%ld,\"epoch\":%d,\"query_bpc\":%.6f,\"train_seconds\":%.3f}\n",path,c,cnt,e,q,tsec);fflush(stdout);}
        }free(full);
    }return 0;
}
/* ---------------- Iterative refinement: drafts as test-time training data ----------------
 * 1. Optionally read a context file for E epochs of TTT (the scratchpad lives in the weights).
 * 2. Each round: sample K drafts after (context tail + prompt), score each by mean byte NLL
 *    under the current weights plus a repetition penalty, keep the best m, and take TTT steps
 *    on windows ending in those drafts mixed with replay windows from the context.
 * 3. Optionally polish the final best draft by local search: resample a short span starting
 *    at a high-surprisal byte, keep the edit only if the whole-draft score improves. */
typedef struct {int rounds,cands,keep,len,steps,epochs,edits,span,tail,retr,retr_m,match;double temp,rep,final_temp;} Refine;
static int sample_rng(const real*lg,double temp,uint64_t*rng){
    double mx=-1e300,z=0,p[256];for(int v=0;v<256;v++)mx=lg[v]>mx?lg[v]:mx;
    for(int v=0;v<256;v++){p[v]=exp((lg[v]-mx)/temp);z+=p[v];}double r=urand(rng)*z;
    for(int v=0;v<256;v++)if((r-=p[v])<=0)return v;return 255;
}
/* Sample n bytes after prefix into out. */
static void draft_sample(Faithful*f,const unsigned char*pre,int np,unsigned char*out,int n,double temp,uint64_t*rng){
    real lg[256];
    if(!f){Gen*g=gen_new(&ttt_byte->c);for(int i=0;i<np;i++)gen_step(ttt_byte,g,pre[i],lg);
        for(int i=0;i<n;i++){int v=sample_rng(lg,temp,rng);out[i]=v;gen_step(ttt_byte,g,v,lg);}gen_free(g);return;}
    FStream*s=fstream_new(f);for(int i=0;i<np;i++)fstream_step(f,s,pre[i],lg);
    for(int i=0;i<n;i++){int v=sample_rng(lg,temp,rng);out[i]=v;fstream_step(f,s,v,lg);}
}
/* Draft sampling with the match model over the full context + prompt + draft so far; the model
 * itself is fed only pre (context tail + prompt). lam: calibrated match trust per length bucket. */
static void draft_sample_mm(Faithful*f,const unsigned char*pre,int np,unsigned char*out,int n,double temp,uint64_t*rng,
                            const unsigned char*ctx,long nc,const unsigned char*prompt,int pl,const double*lam){
    long base=nc+pl;unsigned char*sb=malloc(base+n+1);memcpy(sb,ctx,nc);memcpy(sb+nc,prompt,pl);
    Match m={.on=1,.bits=base>(1L<<19)?22:20};mmx_reset(&m,sb,0);memcpy(m.lam,lam,sizeof(m.lam));for(long i=0;i<base;i++)mmx_advance(&m,i);
    real lg[256];Gen*g=f?NULL:gen_new(&ttt_byte->c);FStream*s=f?fstream_new(f):NULL;
    for(int i=0;i<np;i++){if(f)fstream_step(f,s,pre[i],lg);else gen_step(ttt_byte,g,pre[i],lg);}
    for(int i=0;i<n;i++){mmx_mix_logits(&m,lg);int v=sample_rng(lg,temp,rng);out[i]=v;sb[base+i]=v;mmx_advance(&m,base+i);
        if(f)fstream_step(f,s,v,lg);else gen_step(ttt_byte,g,v,lg);}
    if(g)gen_free(g);free(m.tab);free(sb);
}
/* One TTT optimizer step on B windows, for either model kind. */
static void lm_step(Faithful*f,const int*t,const int*y,int B,int L,const Hyper*h){
    if(f){faithful_step(f,t,y,B,L,h,NULL);return;}
    Model*m=ttt_byte;tss_train_step(&m->c,&m->P,&m->Q,&m->G,m->hit,&m->M,&m->V,m->Gt,m->hits,m->ws,m->threads,t,y,B,L,h,m->k);
}
/* Mean NLL (nats) of seq[from..n) given seq[0..from), windowed with 64 bytes of left context. Thread th. */
static double seq_nll(Faithful*f,int th,const unsigned char*seq,int n,int from,float*per){
    int t[256],y[256];static __thread real*lg=NULL;if(!lg)lg=malloc(256*256*sizeof(real));double tot=0;int cnt=0;
    for(int s=from;s<n-1+1&&s<n;s+=192){int e=s+192<n?s+192:n;int a=s-64<0?0:s-64;int Lw=e-a;
        for(int i=0;i<Lw;i++){t[i]=seq[a+i];y[i]=-1;}int ns;if(f)faithful_eval(f,th,t,y,Lw,&ns,lg);else eval_seq(ttt_byte,t,y,Lw,ttt_byte->ws[th],&ns,lg);
        for(int i=s;i<e;i++){if(i==0)continue;real*l=lg+(size_t)(i-1-a)*256;if(i-1<a)continue;
            double mx=-1e30,z=0;for(int v=0;v<256;v++)mx=l[v]>mx?l[v]:mx;for(int v=0;v<256;v++)z+=exp(l[v]-mx);
            double li=log(z)+mx-l[seq[i]];if(per)per[i-from]=li;tot+=li;cnt++;}}
    return cnt?tot/cnt:0;
}
/* Fraction of repeated 6-byte grams: 0 for no repetition. */
static double rep_frac(const unsigned char*x,int n){
    if(n<7)return 0;int dup=0;uint64_t*h=malloc((size_t)n*sizeof(uint64_t));
    for(int i=0;i+6<=n;i++){uint64_t v=0;for(int k=0;k<6;k++)v=v*257+x[i+k];h[i]=v;int seen=0;for(int j=0;j<i;j++)if(h[j]==v){seen=1;break;}dup+=seen;}
    free(h);return (double)dup/(n-5);
}
static void json_str(FILE*o,const unsigned char*x,int n){fputc('"',o);for(int i=0;i<n;i++){int c=x[i];if(c=='"'||c=='\\')fprintf(o,"\\%c",c);else if(c=='\n')fputs("\\n",o);else if(c<32||c>126)fprintf(o,"\\u%04x",c);else fputc(c,o);}fputc('"',o);}
static int refine_main(const char*ckpt,const char*ctxpath,const char*prompt,const Refine*R,const TTT*o0,int threads,int seed){
    Faithful*f=faithful_load(ckpt,threads,256);if(!f){ttt_byte=load(ckpt,threads,256);if(!ttt_byte){fprintf(stderr,"cannot load checkpoint\n");return 1;}}
    long nc=0;unsigned char*ctx=ctxpath?read_file(ctxpath,&nc):NULL;if(ctxpath&&!ctx)return 1;
    TTT o=*o0;o.rnd=1;int L=256;uint64_t rng=0x9E3779B97F4A7C15ULL^((uint64_t)seed*0xD1B54A32D192ED03ULL);
    int*t=malloc((size_t)L*(o.batch>8?o.batch:8)*sizeof(int)),*y=malloc((size_t)L*(o.batch>8?o.batch:8)*sizeof(int));double t0=wall();
    if(ctx&&R->epochs>0&&o.lr>0){long per=nc/((long)L*o.batch);if(per<1)per=1;TTT oc=o;oc.steps=per;
        for(int e=0;e<R->epochs;e++)ttt_train(f,ctx,nc,&oc,&rng,t,y);}
    fprintf(stderr,"{\"event\":\"refine_context\",\"bytes\":%ld,\"epochs\":%d,\"seconds\":%.3f}\n",nc,R->epochs,wall()-t0);
    double mlam[16]={0};if(R->match&&ctx){long cl=nc<32768?nc:32768;TTT e={.lr=0,.ctx=64,.chunk=192,.steps=1,.batch=1};int was=mm.on;mm.on=1;
        double cb=ttt_run(f,ctx+nc-cl,cl,&e,NULL,NULL,0,NULL);memcpy(mlam,mm.lam,sizeof(mlam));mm.on=was;
        fprintf(stderr,"{\"event\":\"refine_match_calibration\",\"bytes\":%ld,\"bpc\":%.4f,\"lam8\":%.3f,\"lam32\":%.3f}\n",cl,cb,mlam[4],mlam[6]);}
    int tail=ctx?(nc<R->tail?(int)nc:R->tail):0,pl=strlen(prompt),np=tail+pl,N=R->len,K=R->cands;
    unsigned char*pre=malloc(np+1);if(tail)memcpy(pre,ctx+nc-tail,tail);memcpy(pre+tail,prompt,pl);
    unsigned char*cand=malloc((size_t)K*N),*seq=malloc((size_t)K*(np+N));double*score=malloc(K*sizeof(double)),*nll=malloc(K*sizeof(double));int*ord=malloc(K*sizeof(int));
    uint64_t*rs=malloc(K*sizeof(uint64_t));unsigned char*best=malloc(N);double bests=1e30;
    for(int r=0;r<=R->rounds;r++){
        double temp=r==R->rounds?R->final_temp:R->temp;
        for(int k=0;k<K;k++)rs[k]=rng^((uint64_t)(r*1000+k+1)*0x9E3779B97F4A7C15ULL);
#pragma omp parallel for num_threads(threads) schedule(dynamic,1)
        for(int k=0;k<K;k++){
#ifdef _OPENMP
            int th=omp_get_thread_num();
#else
            int th=0;
#endif
            if(R->match&&ctx)draft_sample_mm(f,pre,np,cand+(size_t)k*N,N,temp,rs+k,ctx,nc,(const unsigned char*)prompt,pl,mlam);
            else draft_sample(f,pre,np,cand+(size_t)k*N,N,temp,rs+k);unsigned char*sq=seq+(size_t)k*(np+N);memcpy(sq,pre,np);memcpy(sq+np,cand+(size_t)k*N,N);
            nll[k]=seq_nll(f,th,sq,np+N,np,NULL);score[k]=nll[k]+R->rep*rep_frac(cand+(size_t)k*N,N);
        }
        for(int k=0;k<K;k++)ord[k]=k;for(int i=1;i<K;i++){int v=ord[i],j=i;while(j>0&&score[ord[j-1]]>score[v]){ord[j]=ord[j-1];j--;}ord[j]=v;}
        double mean=0,mrep=0;for(int k=0;k<K;k++){mean+=nll[k];mrep+=rep_frac(cand+(size_t)k*N,N);}
        printf("{\"event\":\"refine_round\",\"round\":%d,\"temp\":%.3f,\"mean_nll_bpc\":%.4f,\"best_nll_bpc\":%.4f,\"mean_rep\":%.4f,\"best\":",r,temp,mean/K/log(2.),nll[ord[0]]/log(2.),mrep/K);
        json_str(stdout,cand+(size_t)ord[0]*N,N);printf(",\"cands\":[");for(int k=0;k<K;k++){if(k)putchar(',');json_str(stdout,cand+(size_t)k*N,N);}printf("]}\n");fflush(stdout);
        memcpy(best,cand+(size_t)ord[0]*N,N);bests=score[ord[0]];
        if(r==R->rounds||o.lr<=0)continue;
        /* retr=1: the kept drafts only choose which context passages to train on.
         * retr=0: self-training on windows ending inside kept drafts, half replay from context. */
        int save_muon=use_muon,save_freeze=ttt_freeze;use_muon=o.optim;ttt_freeze=o.freeze;
        long rid[1024];int nr=0;
        if(R->retr&&ctx){static Retr*rx=NULL;if(!rx){rx=retr_new(ctx,nc);retr_admit(rx,nc);}
            int per=R->retr_m/R->keep>0?R->retr_m/R->keep:1;
            for(int j=0;j<R->keep&&nr<1000;j++){unsigned char*sq=seq+(size_t)ord[j]*(np+N);int got=retr_query(rx,sq+tail,pl+N,per,rid+nr,-1);nr+=got;}
            if(r==0)printf("{\"event\":\"refine_retrieval\",\"chunks\":%d}\n",nr);
            for(int it=0;it<R->steps&&nr;it++){
                for(int b=0;b<o.batch;b++){long at=(b&1)?(long)(urand(&rng)*(nc-L-1)):rx->start[rid[(int)(urand(&rng)*nr)%nr]];if(at>nc-L-1)at=nc-L-1;
                    for(int i=0;i<L;i++){t[b*L+i]=ctx[at+i];y[b*L+i]=ctx[at+i+1];}}
                Hyper h={.lr=o.lr,.beta1=.9,.beta2=o.beta2,.eps=1e-8,.weight_decay=0,.clip=o.clip,.step=++ttt_step_count};lm_step(f,t,y,o.batch,L,&h);
            }
        }
        for(int it=0;it<R->steps&&!(R->retr&&ctx);it++){
            for(int b=0;b<o.batch;b++){
                if(ctx&&(b&1)&&nc>L+1){long at=(long)(urand(&rng)*(nc-L-1));for(int i=0;i<L;i++){t[b*L+i]=ctx[at+i];y[b*L+i]=ctx[at+i+1];}continue;}
                int k=ord[(int)(urand(&rng)*R->keep)%R->keep];unsigned char*sq=seq+(size_t)k*(np+N);int n=np+N;
                int lo=np-L+1>0?np-L+1:0,hi=n-L-1;if(hi<lo)hi=lo;int at=lo+(int)(urand(&rng)*(hi-lo+1));if(at+L+1>n)at=n-L-1;if(at<0)at=0;
                for(int i=0;i<L;i++){int a=at+i;t[b*L+i]=a<n?sq[a]:'\n';y[b*L+i]=a+1<n?sq[a+1]:-1;}
            }
            Hyper h={.lr=o.lr,.beta1=.9,.beta2=o.beta2,.eps=1e-8,.weight_decay=0,.clip=o.clip,.step=++ttt_step_count};lm_step(f,t,y,o.batch,L,&h);
        }use_muon=save_muon;ttt_freeze=save_freeze;
    }
    /* Local search on the best draft. */
    if(R->edits>0){
        unsigned char*cur=malloc(np+N),*prop=malloc((size_t)threads*(np+N));float*per=malloc(N*sizeof(float));memcpy(cur,pre,np);memcpy(cur+np,best,N);
        double cs=seq_nll(f,0,cur,np+N,np,per)+R->rep*rep_frac(cur+np,N);int acc=0;
        for(int e=0;e<R->edits;e+=threads){
            double ps[64];
#pragma omp parallel for num_threads(threads) schedule(static,1)
            for(int j=0;j<threads;j++){
#ifdef _OPENMP
                int th=omp_get_thread_num();
#else
                int th=0;
#endif
                uint64_t lr=rng^((uint64_t)(e+j+7)*0xA24BAED4963EE407ULL);unsigned char*p=prop+(size_t)j*(np+N);memcpy(p,cur,np+N);
                double tot=0;for(int i=0;i<N;i++)tot+=per[i];double u=urand(&lr)*tot;int pos=0;for(;pos<N-1;pos++)if((u-=per[pos])<=0)break;
                int w=R->span;if(pos+w>N)w=N-pos;
                {unsigned char tmp[1024];int ww=w<1024?w:1024;draft_sample(f,p,np+pos,tmp,ww,R->temp,&lr);memcpy(p+np+pos,tmp,ww);}
                ps[j]=seq_nll(f,th,p,np+N,np,NULL)+R->rep*rep_frac(p+np,N);
            }
            int bj=-1;for(int j=0;j<threads;j++)if(ps[j]<cs&&(bj<0||ps[j]<ps[bj]))bj=j;
            if(bj>=0){memcpy(cur,prop+(size_t)bj*(np+N),np+N);cs=seq_nll(f,0,cur,np+N,np,per)+R->rep*rep_frac(cur+np,N);acc++;}
        }
        printf("{\"event\":\"refine_edit\",\"edits\":%d,\"accepted\":%d,\"score\":%.4f,\"text\":",R->edits,acc,cs/log(2.));json_str(stdout,cur+np,N);printf("}\n");
        memcpy(best,cur+np,N);
    }
    printf("{\"event\":\"refine_final\",\"seconds\":%.3f,\"text\":",wall()-t0);json_str(stdout,best,N);printf("}\n");return 0;
}
/* ---------------- --session: one tool that reads, adapts online and generates on demand ----------------
 * stdin protocol. A line that does not start with "@@" is context: it is appended (with its newline),
 * learned online by TTT one chunk at a time, and added to the match model. Commands:
 *   @@gen N [temp] [cands]  generate N bytes after the current context (best of cands drafts)
 *   @@file PATH             ingest a file as context
 *   @@keep 0|1              whether generated text is appended to the context (default 0)
 *   @@stats                 report bytes read, online BPC and time
 *   @@reset                 restore the loaded weights and clear the context
 * Each chunk is scored before it is trained on (when scoring is on), so the reported BPC is honest. */
typedef struct {Faithful*f;real**base;unsigned char*buf;long len,cap,done;TTT o;int score,keep,tail;double loss,secs,tsecs;long scored;uint64_t rng;int*t,*y;real*lg;} Session;
static void sess_append(Session*S,const unsigned char*x,long n){
    if(S->len+n+1>S->cap){S->cap=(S->len+n+1)*2;S->buf=realloc(S->buf,S->cap);}memcpy(S->buf+S->len,x,n);S->len+=n;mm.buf=S->buf;
}
/* Learn every complete chunk; with flush, also the trailing partial one. */
static void sess_learn(Session*S,int flush){
    int C=S->o.chunk;
    while(S->len-1-S->done>=C||(flush&&S->len-1>S->done)){
        long s=S->done,e=s+C<S->len-1?s+C:S->len-1;double st=wall();
        if(S->score){long a=s-S->o.ctx<0?0:s-S->o.ctx;int Lw=(int)(e-a);
            for(int i=0;i<Lw;i++){S->t[i]=S->buf[a+i];S->y[i]=-1;}int ns;
            if(S->f)faithful_eval(S->f,0,S->t,S->y,Lw,&ns,S->lg);else eval_seq(ttt_byte,S->t,S->y,Lw,ttt_byte->ws[0],&ns,S->lg);
            for(long i=s;i<e;i++){mm_advance(i);S->loss+=mm_mix_loss(S->lg+(size_t)(i-a)*256,S->buf[i+1],i+1);S->scored++;}}
        else for(long i=s;i<e;i++)mm_advance(i);
        double tt=wall();if(S->o.lr>0)ttt_train(S->f,S->buf,e,&S->o,&S->rng,S->t,S->y);S->tsecs+=wall()-tt;
        S->done=e;S->secs+=wall()-st;
    }
}
static void sess_reset(Session*S){snap_restore(S->f,S->base);S->len=S->done=0;S->loss=S->secs=S->tsecs=0;S->scored=0;mm_reset(S->buf);}
static void sess_generate(Session*S,int n,double temp,int cands,int threads){
    sess_learn(S,1);if(n<1)return;if(cands<1)cands=1;if(cands>64)cands=64;
    long tl=S->len<S->tail?S->len:S->tail;const unsigned char*pre=S->buf+S->len-tl;
    unsigned char*out=malloc((size_t)cands*n);double sc[64];uint64_t rs[64];for(int k=0;k<cands;k++)rs[k]=S->rng^((uint64_t)(k+1)*0x9E3779B97F4A7C15ULL^(uint64_t)S->len);S->rng+=0x632BE59BD9B4E019ULL;
#pragma omp parallel for num_threads(threads) schedule(dynamic,1)
    for(int k=0;k<cands;k++){
#ifdef _OPENMP
        int th=omp_get_thread_num();
#else
        int th=0;
#endif
        draft_sample_mm(S->f,pre,(int)tl,out+(size_t)k*n,n,temp,rs+k,S->buf,S->len,(const unsigned char*)"",0,mm.lam);
        if(cands>1){unsigned char*sq=malloc(tl+n);memcpy(sq,pre,tl);memcpy(sq+tl,out+(size_t)k*n,n);sc[k]=seq_nll(S->f,th,sq,(int)(tl+n),(int)tl,NULL);free(sq);}else sc[k]=0;
    }
    int b=0;for(int k=1;k<cands;k++)if(sc[k]<sc[b])b=k;
    fwrite(out+(size_t)b*n,1,n,stdout);fputs("\n@@end\n",stdout);fflush(stdout);
    if(S->keep)sess_append(S,out+(size_t)b*n,n);free(out);
}
static int session_main(const char*ckpt,const TTT*o0,int threads,int score,int tail){
    Session S={0};S.f=faithful_load(ckpt,threads,256);if(!S.f){ttt_byte=load(ckpt,threads,256);if(!ttt_byte){fprintf(stderr,"cannot load checkpoint\n");return 1;}}
    S.base=snap_take(S.f);S.o=*o0;S.o.rnd=0;S.score=score;S.tail=tail;S.rng=0x243F6A8885A308D3ULL;S.cap=1<<16;S.buf=malloc(S.cap);
    int W=S.o.ctx+S.o.chunk;if(W<256)W=256;S.t=malloc((size_t)W*(S.o.batch>1?S.o.batch:1)*sizeof(int)*2);S.y=malloc((size_t)W*(S.o.batch>1?S.o.batch:1)*sizeof(int)*2);S.lg=malloc((size_t)W*256*sizeof(real));
    mm.on=1;mm_reset(S.buf);
    fprintf(stderr,"{\"event\":\"session_ready\",\"model\":\"%s\",\"lr\":%g,\"steps\":%d,\"window\":%d,\"chunk\":%d,\"score\":%d}\n",S.f?"blt":"byte",S.o.lr,S.o.steps,ttt_win,S.o.chunk,score);
    char*line=NULL;size_t cap=0;ssize_t r;
    while((r=getline(&line,&cap,stdin))>0){
        if(!strncmp(line,"@@",2)){
            char cmd[32]={0};int n=0;double temp=.7;int cands=1;char path[1024]={0};
            if(sscanf(line+2,"%31s",cmd)!=1)continue;
            if(!strcmp(cmd,"gen")){sscanf(line+2,"%*s %d %lf %d",&n,&temp,&cands);sess_generate(&S,n,temp,cands,threads);}
            else if(!strcmp(cmd,"file")){if(sscanf(line+2,"%*s %1023s",path)==1){long fn;unsigned char*fb=read_file(path,&fn);if(fb){sess_append(&S,fb,fn);free(fb);sess_learn(&S,0);}}}
            else if(!strcmp(cmd,"keep")){sscanf(line+2,"%*s %d",&S.keep);}
            else if(!strcmp(cmd,"reset")){sess_reset(&S);}
            else if(!strcmp(cmd,"stats")){sess_learn(&S,1);}
            if(!strcmp(cmd,"stats")||!strcmp(cmd,"file")||!strcmp(cmd,"reset"))
                fprintf(stderr,"{\"event\":\"session_stats\",\"bytes\":%ld,\"learned\":%ld,\"bpc\":%.4f,\"seconds\":%.2f,\"ttt_seconds\":%.2f,\"bytes_per_second\":%.0f}\n",
                    S.len,S.done,S.scored?S.loss/S.scored/log(2.):0,S.secs,S.tsecs,S.secs>0?S.done/S.secs:0);
            continue;
        }
        sess_append(&S,(unsigned char*)line,r);sess_learn(&S,0);
    }
    sess_learn(&S,1);fprintf(stderr,"{\"event\":\"session_end\",\"bytes\":%ld,\"bpc\":%.4f,\"seconds\":%.2f}\n",S.len,S.scored?S.loss/S.scored/log(2.):0,S.secs);
    return 0;
}
static void usage(void){
    puts("nanotss-plastic: protected plastic -> FFF -> SSM -> FFF\n"
         "  --data FILE [--save FILE] [--seconds 300] [--entropy-seconds 60]\n"
         "  --load FILE --sample 600 [--prompt TEXT] [--temperature 0.8]\n"
         "  --load FILE --data FILE --score-only [--burn 0]\n"
         "  --entropy-load FILE | --entropy-save FILE\n"
         "  --context-file FILE (stream entire context with persistent state; requires --load)\n"
         "  --fixed-steps N (step-based schedule, ignores wall-time stopping)\n"
         "  --optimizer adamw|muon --muon-block 16 --muon-steps 5 --muon-scale 3 (BLT default: muon)\n"
         "  --threads 4 --batch 16 --seq 256 --seed 0 --lr 0.003\n"
         "  --runtime-reference (disable earlier runtime SIMD reductions)\n"
         "  --cpu-reference (disable VNNI, vector libm, and batched FFF activations)\n"
         "  --conv-mix 2 (before global stack; 1 legacy after-stack; 0 off)\n"
         "  --fw-scalar (reference kernel; default is optimized)\n"
         "  --fw-mask 0 (byte default: SSM only; BLT default 1 = plastic first; 4=plastic third, 5=both)\n"
         "  --fff-gate 1 --ngram-bins 256 --patch-size 4 --max-patch 8\n"
         "  --mode byte|blt (default byte) --ssm-groups G (default 4) --ssm-qknorm 0|1 --ssm-rope 0|1 (default 1) --ssm-ltc 0|1|2\n"
         "  --optimizer muon|normuon|adamw (byte default: muon; TTT always uses AdamW)\n"
         "  --session 1 [--session-speed fast|balanced|quality] --match 1 --ttt-eval/--ttt-fit/--refine (see report)\n"
         "  --check --check-tree-fw --check-neural --check-ssm --help\n"
         "New runs enable FFF gates; loaded checkpoints retain their saved setting.");
}


#ifndef NANOTSS_NO_MAIN
int main(int argc,char**argv){
    int optimizer_explicit=0;
    const char*data=NULL,*out=NULL,*in=NULL,*mode="byte",*prompt="ROMEO:",*entropy_in=NULL,*entropy_out=NULL,*context_file=NULL;int threads=4,B=16,L=256,burn=64,sample=0,seed=0,steps=100000000,scoreonly=0,fixedsteps=0;
    double seconds=300,entropy_seconds=60,entropy_charge=0,lr=.003,target=4,temp=.8;
    Cfg gc={.vocab=256,.d=128,.n_layer=4,.T=4,.D=7,.n=16,.p=16,.K=3,.group=128,.walks=2,.temp=1,.fff_gate=1,.fw_mask=-1};int groups_arg=-1,qk_arg=-1,rope_arg=-1,want_byte_muon=0;
    double ssm_horizon=0;int carry_windows=0;long*carry_pos=NULL;int*carry_left=NULL;unsigned char*carry_buf=NULL;double repeat_p=0;long repeat_maxgap=0;
    int session=0,sess_score=1;const char*sess_speed="balanced";
    int refine=0;const char*refine_ctx=NULL;Refine rf={.match=0,.retr=0,.retr_m=32,.rounds=3,.cands=16,.keep=4,.len=400,.steps=8,.epochs=8,.edits=0,.span=12,.tail=512,.temp=1,.rep=2,.final_temp=1};
    const char*ttt_file=NULL,*ttt_control=NULL,*ttt_fit=NULL;double ttt_frac=.5;int ttt_epochs=8;long ttt_off=-1,ttt_len=0,ttt_stride=0;int ttt_docs=1;
    TTT tt={.rnd=0,.lr=0,.decay=0,.clip=1,.beta2=.99,.steps=1,.batch=4,.chunk=192,.ctx=64,.stream=0,.freeze=0,.optim=0};
    FConfig cfg={.width=64,.heads=4,.bins=256,.maxpatch=8,.enc_layers=2,.dec_layers=4,.local_depth=4,.local_trees=2,.local_state=8,.neural=1,.mixer=-1,.target=4};
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--cpu-reference")){cpu_fast=0;continue;}
        if(!strcmp(argv[i],"--fw-scalar")){fw_fast=0;continue;}
        if(!strcmp(argv[i],"--help")){usage();return 0;}
        if(!strcmp(argv[i],"--check-neural"))return check_neural();
        if(!strcmp(argv[i],"--check-loops")){if(qk_arg>=0)ssm_qknorm=qk_arg;if(rope_arg>=0)ssm_rope=rope_arg;int bad=0;for(int r=1;r<=4;r++)bad|=check_loops(r);return bad;}
        if(!strcmp(argv[i],"--check-ssm")){if(groups_arg>0)ssm_groups=groups_arg;if(qk_arg>=0)ssm_qknorm=qk_arg;if(rope_arg>=0)ssm_rope=rope_arg;return check_block_gradient(0);}
        if(!strcmp(argv[i],"--check-tree-fw"))return check_tree_fw();
        if(!strcmp(argv[i],"--runtime-reference")){runtime_fast=0;continue;}
        if(!strcmp(argv[i],"--check"))return check_gated_fff()||check_tree_fw();
        if(!strcmp(argv[i],"--score-only")){scoreonly=1;continue;}
        const char*a=argv[i],*v=i+1<argc?argv[++i]:NULL;if(!v)return 2;
        if(!strcmp(a,"--ttt-range")){if(ttt_nranges<32&&sscanf(v,"%ld:%ld",ttt_ranges+2*ttt_nranges,ttt_ranges+2*ttt_nranges+1)==2)ttt_nranges++;else return 2;continue;}
        if(!strcmp(a,"--ttt-optim")){if(!strcmp(v,"muon"))tt.optim=1;else if(!strcmp(v,"adamw"))tt.optim=0;else return 2;continue;}
        if(!strcmp(a,"--optimizer")){optimizer_explicit=1;if(!strcmp(v,"muon"))use_muon=1;else if(!strcmp(v,"normuon")){use_muon=1;normuon=1;}else if(!strcmp(v,"adamw"))use_muon=0;else return 2;continue;}
        if(!strcmp(a,"--patcher")){if(strcmp(v,"neural")){fprintf(stderr,"Only the neural entropy patcher is supported.\n");return 2;}continue;}
#define OPT(n,x,cv) if(!strcmp(a,n)){x=cv(v);continue;}
        OPT("--ssm-horizon",ssm_horizon,atof) OPT("--ttt-reset",ttt_reset,atol) OPT("--ttt-reset-mask",ttt_reset_mask,atoi) OPT("--session",session,atoi) OPT("--session-score",sess_score,atoi) OPT("--session-speed",sess_speed,(const char*)) OPT("--refine",refine,atoi) OPT("--refine-context",refine_ctx,(const char*)) OPT("--refine-rounds",rf.rounds,atoi) OPT("--refine-cands",rf.cands,atoi) OPT("--refine-keep",rf.keep,atoi)
        OPT("--refine-len",rf.len,atoi) OPT("--refine-steps",rf.steps,atoi) OPT("--refine-epochs",rf.epochs,atoi) OPT("--refine-edits",rf.edits,atoi) OPT("--refine-span",rf.span,atoi) OPT("--refine-tail",rf.tail,atoi)
        OPT("--refine-match",rf.match,atoi) OPT("--refine-retr",rf.retr,atoi) OPT("--refine-retr-m",rf.retr_m,atoi) OPT("--refine-temp",rf.temp,atof) OPT("--refine-rep",rf.rep,atof) OPT("--refine-final-temp",rf.final_temp,atof)
        OPT("--mtp",mtp_lambda,atof) OPT("--pause",pause_k,atoi) OPT("--latent",latent_on,atoi) OPT("--latent-layers",latent_layers,atoi) OPT("--rl-n",rl_n,atoi) OPT("--rl-beta",rl_beta,atof) OPT("--loops",n_loops,atoi) OPT("--layers",gc.n_layer,atoi) OPT("--ssm-groups",groups_arg,atoi) OPT("--pos-feat",pos_feat,atof) OPT("--ssm-qknorm",qk_arg,atoi) OPT("--ssm-rope",rope_arg,atoi) OPT("--ttt-adaptive",ttt_adaptive,atof) OPT("--ttt-win",ttt_win,atoi) OPT("--ttt-bwd-layers",bwd_layers,atoi) OPT("--ssm-ltc",ssm_ltc,atoi) OPT("--match",mm.on,atoi) OPT("--match-lr",mm.lr,atof) OPT("--fw-writes",fw_associative_writes,atoi) OPT("--depth",gc.D,atoi) OPT("--state-n",gc.n,atoi) OPT("--carry-windows",carry_windows,atoi) OPT("--repeat-p",repeat_p,atof) OPT("--repeat-maxgap",repeat_maxgap,atol) OPT("--width",cfg.width,atoi) OPT("--ttt-ctxlen",ttt_ctxlen,atol) OPT("--ttt-lex-mult",ttt_lex_mult,atof) OPT("--ttt-only",ttt_only,atoi) OPT("--ttt-retrieve",ttt_use_retr,atoi) OPT("--ttt-retr-m",ttt_retr_m,atoi) OPT("--ttt-retr-mix",ttt_retr_mix,atof) OPT("--ttt-qlen",ttt_qlen,atol) OPT("--ttt-fit",ttt_fit,(const char*)) OPT("--ttt-frac",ttt_frac,atof) OPT("--ttt-epochs",ttt_epochs,atoi) OPT("--ttt-eval",ttt_file,(const char*)) OPT("--ttt-control",ttt_control,(const char*)) OPT("--ttt-offset",ttt_off,atol) OPT("--ttt-len",ttt_len,atol) OPT("--ttt-docs",ttt_docs,atoi) OPT("--ttt-docstride",ttt_stride,atol)
        OPT("--ttt-lr",tt.lr,atof) OPT("--ttt-decay",tt.decay,atof) OPT("--ttt-clip",tt.clip,atof) OPT("--ttt-beta2",tt.beta2,atof) OPT("--ttt-steps",tt.steps,atoi) OPT("--ttt-batch",tt.batch,atoi)
        OPT("--ttt-chunk",tt.chunk,atoi) OPT("--ttt-ctx",tt.ctx,atoi) OPT("--ttt-stream",tt.stream,atoi) OPT("--ttt-freeze",tt.freeze,atoi)
        OPT("--context-file",context_file,(const char*)) OPT("--data",data,(const char*)) OPT("--save",out,(const char*)) OPT("--load",in,(const char*)) OPT("--mode",mode,(const char*)) OPT("--prompt",prompt,(const char*))
        OPT("--muon-scale",muon_scale,atof) OPT("--muon-steps",muon_steps,atoi) OPT("--muon-block",muon_block,atoi)
        OPT("--entropy-charge",entropy_charge,atof) OPT("--entropy-load",entropy_in,(const char*)) OPT("--entropy-save",entropy_out,(const char*)) OPT("--entropy-seconds",entropy_seconds,atof)
        OPT("--fixed-steps",fixedsteps,atoi) OPT("--threads",threads,atoi) OPT("--batch",B,atoi) OPT("--seq",L,atoi) OPT("--burn",burn,atoi) OPT("--sample",sample,atoi) OPT("--seed",seed,atoi) OPT("--steps",steps,atoi)
        OPT("--conv-mix",cfg.mixer,atoi) OPT("--fw-mask",gc.fw_mask,atoi) OPT("--fff-gate",gc.fff_gate,atoi) OPT("--seconds",seconds,atof) OPT("--lr",lr,atof) OPT("--patch-size",target,atof) OPT("--temperature",temp,atof)
        OPT("--ngram-bins",cfg.bins,atoi) OPT("--max-patch",cfg.maxpatch,atoi) OPT("--encoder-layers",cfg.enc_layers,atoi) OPT("--decoder-layers",cfg.dec_layers,atoi) OPT("--local-depth",cfg.local_depth,atoi) OPT("--local-trees",cfg.local_trees,atoi)
#undef OPT
        fprintf(stderr,"unknown argument %s\n",a);return 2;
    }
    /* Defaults: byte mode, pure SSM (no plastic layer), per-tree SSM keys/queries, no conv mixer.
     * --mode blt keeps the BLT defaults (plastic layer, mixer 2, shared projections). */
    {int byte=!strcmp(mode,"byte");if(gc.fw_mask<0)gc.fw_mask=byte?0:1;if(cfg.mixer<0)cfg.mixer=byte?0:2;
     ssm_groups=groups_arg>0?groups_arg:byte?gc.T:1;
     if(ssm_groups<1||ssm_groups>gc.T||gc.T%ssm_groups){fprintf(stderr,"--ssm-groups must divide the tree count\n");return 2;}
     if(!byte&&ssm_groups!=1){fprintf(stderr,"BLT mode supports only --ssm-groups 1\n");return 2;}
     /* Byte defaults also: SSM QK-norm and rotary keys/queries; Muon for training (TTT keeps AdamW). */
     ssm_qknorm=qk_arg>=0?qk_arg:byte;ssm_rope=rope_arg>=0?rope_arg:byte;
     if(n_loops<1||n_loops>16||(n_loops>1&&(!byte||gc.n_layer%2||carry_windows))){fprintf(stderr,"--loops needs byte mode, an even --layers and no --carry-windows\n");return 2;}
     if(!byte&&(ssm_qknorm||ssm_rope)){fprintf(stderr,"BLT mode does not support --ssm-qknorm/--ssm-rope\n");return 2;}}
    if(muon_steps<1||muon_steps>10||muon_block<1||muon_block>32||muon_scale<=0)return 2;
    if(cfg.mixer<0||cfg.mixer>2)return 2;
    if(fixedsteps<0)return 2;
    if(fixedsteps)steps=fixedsteps;
    if((gc.fw_mask&~0x5)||gc.fff_gate<0||gc.fff_gate>1)return 2;
    if(threads<1||threads>64||B<1||L<8||burn<0||burn>=L||seconds<=0||temp<=0||cfg.bins<0||cfg.bins>1048576||cfg.maxpatch<1||cfg.maxpatch>1024||target<1||target>cfg.maxpatch||cfg.enc_layers<0||cfg.dec_layers<0||cfg.local_depth<1||cfg.local_depth>12||cfg.local_trees<1)return 2;
    if(strcmp(mode,"byte")&&strcmp(mode,"blt"))return 2;
    if(strcmp(mode,"blt")){want_byte_muon=optimizer_explicit?use_muon:1;use_muon=0;} /* byte mode: Muon unless --optimizer adamw */
    if(!strcmp(mode,"byte")&&cfg.mixer){fprintf(stderr,"Standalone byte mode requires --conv-mix 0; the integrated mixer acts on global BLT states.\n");return 2;}
    g_rng^=((uint64_t)seed+1)*0x9E3779B97F4A7C15ULL;
    gc.d=2*cfg.width;
    if(ssm_horizon>0)ssm_dec_cap=(real)(1-1/ssm_horizon);
    if(session){if(!in)return 2;
        if(!strcmp(sess_speed,"fast")){tt.steps=1;ttt_win=128;}else if(!strcmp(sess_speed,"quality")){tt.steps=2;ttt_win=256;}else{tt.steps=1;ttt_win=256;}
        if(tt.lr==0)tt.lr=2e-4;return session_main(in,&tt,threads,sess_score,512);}
    if(refine){if(!in||rf.cands<1||rf.keep<1||rf.keep>rf.cands||rf.len<1||threads>64)return 2;return refine_main(in,refine_ctx,prompt,&rf,&tt,threads,seed);}
    if(ttt_fit){if(!in)return 2;return ttt_fit_main(in,ttt_fit,ttt_frac,ttt_epochs,&tt,threads);}
    if(ttt_file){if(!in)return 2;return ttt_eval_main(in,ttt_file,ttt_off,ttt_len,&tt,threads,ttt_docs,ttt_stride?ttt_stride:ttt_len,ttt_control);}
    if(context_file){if(!in||data||scoreonly||sample<0){fprintf(stderr,"--context-file requires --load and generation mode\n");return 2;}return context_generate(in,context_file,prompt,sample,temp);}
    if(in&&!data){
        Faithful*f=faithful_load(in,1,L);if(!f){Model*m=load(in,1,L);if(!m){fprintf(stderr,"cannot load checkpoint\n");return 1;}generate(m,prompt,sample,temp);return 0;}
        FStream*s=fstream_new(f);real lg[256];const char*p=*prompt?prompt:"\n";for(const char*q=p;*q;q++){fstream_step(f,s,(unsigned char)*q,lg);putchar(*q);}for(int i=0;i<sample;i++){int v=sample_logits(lg,256,temp);putchar(v);fstream_step(f,s,v,lg);}putchar('\n');return 0;
    }
    if(!data){usage();return 2;}
    FILE*fp=fopen(data,"rb");if(!fp)return 1;fseek(fp,0,SEEK_END);long n=ftell(fp);rewind(fp);unsigned char*raw=malloc(n);if(fread(raw,1,n,fp)!=(size_t)n)return 1;fclose(fp);long split=(long)(n*.9);
    if(pause_k>0){if(memchr(raw,0,n)){fprintf(stderr,"--pause needs text without NUL bytes\n");return 2;}long a,b;unsigned char*tr=pause_expand(raw,split,&a),*va=pause_expand(raw+split,n-split,&b);
        free(raw);raw=malloc(a+b);memcpy(raw,tr,a);memcpy(raw+a,va,b);free(tr);free(va);split=a;n=a+b;}
    if(split<L+2||n-split<L+2)return 2;
    Faithful*f=NULL;Model*m=NULL;double aux_seconds=0,calibration_seconds=0,avg=1,setup0=wall();int blt=!strcmp(mode,"blt");
    if(in){f=faithful_load(in,threads,L);if(!f)m=load(in,threads,L);if(!f&&!m)return 1;blt=f!=NULL;}
    else if(blt){cfg.target=target;f=faithful_new(gc,cfg,threads,L,1);}
    else {m=model_new(&gc,threads,L);params_init(&gc,&m->P);model_requantize(m);}
    byte_muon=m&&want_byte_muon;if(byte_muon)byte_muon_model=m;
    if(latent_on&&(!m||pause_k<1||carry_windows>0||bwd_layers>0)){fprintf(stderr,"--latent needs byte mode, --pause K, no carry and full backward\n");return 2;}
    if(latent_on){lat_d=m->c.d;size_t n=(size_t)lat_d*lat_d;lat_W=calloc(n,sizeof(real));lat_m=calloc(n,sizeof(real));lat_v=calloc(n,sizeof(real));if(latent_on==2)for(int i=0;i<lat_d;i++)lat_W[(size_t)i*lat_d+i]=1;}
    if(m&&mtp_lambda>0){mtp_d=m->c.d;size_t n=(size_t)mtp_d*mtp_d;mtp_W=calloc(n,sizeof(real));mtp_m=calloc(n,sizeof(real));mtp_v=calloc(n,sizeof(real));for(int i=0;i<mtp_d;i++)mtp_W[(size_t)i*mtp_d+i]=1;}
    if(scoreonly){if(!in)return 2;long scored;double bpc=score_data(f,m,raw,split,n,L,burn,&scored);printf("{\"event\":\"score\",\"val_bpc\":%.6f,\"scored_bytes\":%ld,\"burn\":%d}\n",bpc,scored,burn);return 0;}
    if(blt&&!in){
        if(f->c.neural){
            if(entropy_in){Model*em=load(entropy_in,threads,L);if(!em||em->c.vocab!=256)return 2;f->entropy=em;aux_seconds=entropy_charge;}
            else {if(entropy_seconds<=0||entropy_seconds>=seconds){fprintf(stderr,"entropy pretraining must be positive and shorter than total budget\n");return 2;}aux_seconds=train_entropy_model(f->entropy,raw,split,B,L,entropy_seconds,seed);}
            if(entropy_out&&save(f->entropy,entropy_out))return 1;
            double st=wall();avg=calibrate_neural(f,raw,split,L,target);calibration_seconds=wall()-st;
        }
    }
    printf("{\"event\":\"architecture\",\"fw_mask\":%d,\"fff_gate\":%d}\n",blt?f->global->c.fw_mask:m->c.fw_mask,blt?f->global->c.fff_gate:m->c.fff_gate);
    printf("{\"event\":\"ternary_mixer\",\"enabled\":%d,\"taps\":%d,\"group_size\":%d,\"parameters\":%zu}\n",blt?f->c.mixer:0,TC_TAPS,TC_GROUP,blt&&f->c.mixer?tc_size(2*f->c.width,TC_TAPS):0);
    size_t core=blt?core_parameters(&f->global->c):core_parameters(&m->c),enc=blt?core_parameters(&f->encoder->c):0,dec=blt?core_parameters(&f->decoder->c):0;
    size_t base=blt?core+enc+dec+f->dense+(size_t)257*f->c.width:core+257*m->c.d,ng=blt?f->hashsize:0,ep=blt&&f->c.neural?core_parameters(&f->entropy->c)+(size_t)257*f->entropy->c.d:0;
    printf("{\"event\":\"config\",\"mode\":\"%s\",\"patcher\":\"%s\",\"main_budgeted_parameters\":%zu,\"frozen_entropy_parameters\":%zu,\"budgeted_parameters\":%zu,\"free_ngram_parameters\":%zu,\"trainable_main_parameters\":%zu,\"total_inference_parameters\":%zu,\"threads\":%d,\"seq_bytes\":%d,\"batch\":%d,\"seed\":%d,\"avg_patch_calibration\":%.6f,\"threshold_nats\":%.7f,\"entropy_training_seconds\":%.6f,\"calibration_seconds\":%.6f,\"setup_wall_seconds\":%.6f}\n",blt?"blt":"byte",blt?"neural":"none",base,ep,base+ep,ng,base+ng,base+ng+ep,threads,L,B,seed,avg,blt?f->c.threshold:0,aux_seconds,calibration_seconds,wall()-setup0);fflush(stdout);
    printf("{\"event\":\"optimizer\",\"name\":\"%s\",\"block_rows\":%d,\"ns_steps\":%d,\"muon_scale\":%.6f}\n",use_muon?"cpu_block_muon":byte_muon?(normuon?"byte_block_normuon":"byte_block_muon"):"adamw",muon_block,muon_steps,muon_scale);fflush(stdout);
    double overhead=aux_seconds+calibration_seconds,budget=seconds-overhead;if(budget<=0)return 2;
    uint64_t frozen=ep?param_fingerprint(f->entropy):0,data_rng=((uint64_t)seed+1)*0xD1B54A32D192ED03ULL;
    int*bt=malloc((size_t)B*L*sizeof(int)),*by=malloc((size_t)B*L*sizeof(int));
    Hyper hp={.beta1=.9,.beta2=.98,.eps=1e-8,.weight_decay=.01,.clip=1};double trainsec=0,evalsec=0,nextlog=fmin(60,budget),loss=0;long long seen=0,patches=0;int step=0;
    double loop_start=wall(),max_step_seconds=0;
    while((fixedsteps||wall()-loop_start+max_step_seconds*1.25<budget)&&step<steps){
        trainsec=wall()-loop_start;
        step++;hp.step=step;hp.lr=lr*fmin(1,(trainsec+.1)/fmin(10,budget*.1))*(.1+.9*.5*(1+cos(3.141592653589793*fmin(1,trainsec/budget))));
        if(fixedsteps){double progress=steps>1?(double)(step-1)/(steps-1):1;hp.lr=lr*fmin(1,(double)step/100)*(.1+.9*.5*(1+cos(3.141592653589793*progress)));}
        for(int b=0;b<B;b++){long at;
            if(carry_windows>0){if(!carry_pos){carry_pos=calloc(B,sizeof(long));carry_left=calloc(B,sizeof(int));
                    Cfg*gcp=blt?&f->global->c:&m->c;carry_sz_h=(size_t)(gcp->n_layer/2)*gcp->T*NODES(gcp)*gcp->n*gcp->p;carry_sz_hist=(size_t)gcp->n_layer*(gcp->K-1)*gcp->d;
                    carry_bank_h=calloc((size_t)B*carry_sz_h,sizeof(real));carry_bank_hist=calloc((size_t)B*carry_sz_hist,sizeof(real));}
                long need=(long)L*carry_windows+1;
                if(carry_left[b]<=0){carry_left[b]=carry_windows;carry_pos[b]=0;
                    memset(carry_bank_h+(size_t)b*carry_sz_h,0,carry_sz_h*sizeof(real));memset(carry_bank_hist+(size_t)b*carry_sz_hist,0,carry_sz_hist*sizeof(real));
                    /* Build this row's stream: contiguous text, with pieces that re-copy earlier stream text. */
                    if(!carry_buf){carry_buf=calloc((size_t)B*need,1);}unsigned char*sb=carry_buf+(size_t)b*need;
                    long cur=(long)(urand(&data_rng)*(split-need-1)),len=0;
                    while(len<need){
                        if(len>512&&urand(&data_rng)<repeat_p){int pl=64+(int)(urand(&data_rng)*193);long lo=repeat_maxgap>0&&len-pl-repeat_maxgap>0?len-pl-repeat_maxgap:0;long gap=lo+(long)(urand(&data_rng)*(len-pl-lo));if(gap<0)gap=0;
                            for(int i=0;i<pl&&len<need;i++)sb[len++]=sb[gap+i];}
                        else{for(int i=0;i<128&&len<need;i++){sb[len++]=raw[cur++];if(cur>=split)cur=0;}}
                    }}
                unsigned char*sb=carry_buf+(size_t)b*need;at=carry_pos[b];carry_pos[b]+=L;carry_left[b]--;
                for(int i=0;i<L;i++){bt[b*L+i]=sb[at+i];by[b*L+i]=sb[at+i+1];}continue;
            }else at=(long)(urand(&data_rng)*(split-L-1));
            for(int i=0;i<L;i++){bt[b*L+i]=raw[at+i];by[b*L+i]=PLAB(raw[at+i+1]);}}
        double st=wall();if(blt)loss=faithful_step(f,bt,by,B,L,&hp,&patches);else{loss=tss_train_step(&m->c,&m->P,&m->Q,&m->G,m->hit,&m->M,&m->V,m->Gt,m->hits,m->ws,threads,bt,by,B,L,&hp,m->k);patches+=(long long)B*L;}
        double step_seconds=wall()-st;if(step_seconds>max_step_seconds)max_step_seconds=step_seconds;
        trainsec=wall()-loop_start;seen+=(long long)B*L;
        if(trainsec>=nextlog){printf("{\"event\":\"progress\",\"step\":%d,\"main_train_seconds\":%.6f,\"train_bytes\":%lld,\"train_bpc\":%.6f}\n",step,trainsec,seen,loss/log(2.));fflush(stdout);nextlog+=60;}
    }
    trainsec=wall()-loop_start;
    if(latent_on&&lat_W){double q=0;for(size_t i=0;i<(size_t)lat_d*lat_d;i++)q+=(double)lat_W[i]*lat_W[i];printf("{\"event\":\"latent\",\"W_frobenius\":%.6f,\"mean_reward_nats\":%.6f,\"rewards\":%ld}\n",sqrt(q),rl_rcount?rl_rsum/rl_rcount:0,rl_rcount);}
    long final_scored;double eval_start=wall();double final_bpc=score_data(f,m,raw,split,n,L,burn,&final_scored);
    printf("{\"event\":\"eval\",\"step\":%d,\"main_train_seconds\":%.6f,\"charged_seconds\":%.6f,\"train_bytes\":%lld,\"global_steps\":%lld,\"online_bytes_per_second\":%.3f,\"train_bpc\":%.6f,\"val_bpc\":%.6f,\"val_scored_bytes\":%ld,\"eval_seconds\":%.6f}\n",step,trainsec,trainsec+overhead,seen,patches,seen/trainsec,loss/log(2.),final_bpc,final_scored,wall()-eval_start);fflush(stdout);
    if(ep&&frozen!=param_fingerprint(f->entropy)){fprintf(stderr,"frozen entropy model changed\n");return 3;}
    if(out&&(blt?faithful_save(f,out):save(m,out))){fprintf(stderr,"save failed\n");return 1;}
    long scored;double all=score_data(f,m,raw,split,n,L,0,&scored);printf("{\"event\":\"all_position_score\",\"val_bpc\":%.6f,\"scored_bytes\":%ld,\"entropy_frozen_verified\":%d}\n",all,scored,ep!=0);
    return 0;
}

#endif /* NANOTSS_NO_MAIN */
