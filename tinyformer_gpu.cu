// ============================================================================
// tinyformer_gpu.cu — tinyformer with the standard library and a CUDA path
//
// Same model, tokenizer, training schedule and checkpoint format as
// tinyformer.cpp, so checkpoints move freely between the two programs. The
// differences:
//   - ordinary #includes (libc math, stdio, chrono) instead of hand-rolled ones
//   - training runs on the GPU: a whole batch of sequences goes through each
//     layer at once, matmuls via cuBLAS, everything else in simple kernels
//   - sample and chat stay on the CPU (KV-cached, one token at a time)
//
// The same file also builds with a plain C++ compiler: kernels then run as
// serial loops on the CPU. That build is slow, but it is the reference the
// GPU build is checked against, and it lets sample/chat work on any machine.
//
// build (GPU):  nvcc -O3 -arch=native -o tinyformer_gpu tinyformer_gpu.cu -lcublas
// build (CPU):  g++ -O3 -march=native -x c++ -o tinyformer_gpu tinyformer_gpu.cu
//
// usage: as tinyformer.cpp, plus an optional batch size for training modes
//   ./tinyformer_gpu train    [steps] [corpus.txt] [-b batch]
//   ./tinyformer_gpu resume   [steps] corpus.txt   [-b batch]
//   ./tinyformer_gpu finetune [steps] dialog.txt   [-b batch]
//   ./tinyformer_gpu sample   [nchars]
//   ./tinyformer_gpu chat
// ============================================================================

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <vector>

// ---------------------------------------------------------------------------
// device abstraction: CUDA when built by nvcc, serial loops otherwise
//   KERNEL declares a kernel, KID is the flat thread index inside it, and
//   LAUNCH(k, n, args...) runs k for KID = 0..n-1 (kernels bounds-check KID)
// ---------------------------------------------------------------------------
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define KERNEL static __global__ void
#define HD     __host__ __device__
#define KID    ((long)blockIdx.x * blockDim.x + threadIdx.x)
#define LAUNCH(k, n, ...) k<<<(unsigned)(((n) + 255) / 256), 256>>>(__VA_ARGS__)
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("cuda error %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)
#define CKB(x) do { cublasStatus_t e_ = (x); if (e_ != CUBLAS_STATUS_SUCCESS) { \
    printf("cublas error %d at line %d\n", (int)e_, __LINE__); exit(1); } } while (0)
static cublasHandle_t g_blas;
#else
static long g_kid;
#define KERNEL static void
#define HD
#define KID g_kid
#define LAUNCH(k, n, ...) for (g_kid = 0; g_kid < (long)(n); g_kid++) k(__VA_ARGS__)
static inline void atomicAdd(float* p, float v) { *p += v; }
#endif

template <class X> static X* dalloc(long n) {         // zeroed device buffer
    X* p;
#ifdef __CUDACC__
    CK(cudaMalloc((void**)&p, n * sizeof(X)));
    CK(cudaMemset(p, 0, n * sizeof(X)));
#else
    p = (X*)calloc(n, sizeof(X));
    if (!p) { printf("FATAL: out of memory\n"); exit(1); }
#endif
    return p;
}
static void to_dev(void* dst, const void* src, long bytes) {
#ifdef __CUDACC__
    CK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
#else
    memcpy(dst, src, bytes);
#endif
}
static void to_host(void* dst, const void* src, long bytes) {
#ifdef __CUDACC__
    CK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
#else
    memcpy(dst, src, bytes);
#endif
}
static void dzero(void* p, long bytes) {
#ifdef __CUDACC__
    CK(cudaMemset(p, 0, bytes));
#else
    memset(p, 0, bytes);
#endif
}

typedef unsigned int       u32;
typedef unsigned long long u64;

// ---------------------------------------------------------------------------
// random numbers (same generator as tinyformer.cpp)
// ---------------------------------------------------------------------------
static u64 g_rng = 0x853c49e6748fea9bULL;

static u32 rnd_u32() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (u32)(g_rng >> 32);
}
static float rnd_uniform() { return (float)(rnd_u32() >> 8) / 16777216.0f; }
static float rnd_normal() {              // Irwin-Hall: 12 uniforms - 6 ~ N(0,1)
    float s = 0.0f;
    for (int i = 0; i < 12; i++) s += rnd_uniform();
    return s - 6.0f;
}

// ---------------------------------------------------------------------------
// model configuration (must match tinyformer.cpp to share checkpoints)
// ---------------------------------------------------------------------------
enum {
    T  = 128,       // context length
    C  = 256,       // embedding width
    L  = 6,         // transformer layers
    H  = 8,         // attention heads
    HS = C / H,     // head size
    F  = 4 * C,     // MLP hidden width
    MAXV = 1024,    // vocab cap: specials + chars + learned BPE merges
    CORPUS_MAX = 8388608
};

#ifdef __CUDACC__
static int B = 16;                       // sequences per optimizer step (-b)
#else
static int B = 4;
#endif

enum { TOK_USER = 0, TOK_BOT = 1, TOK_END = 2, N_SPECIAL = 3 };

// ---------------------------------------------------------------------------
// built-in training text (used when no corpus file is given)
// ---------------------------------------------------------------------------
#define TXT \
"the little model reads the text one letter at a time. " \
"attention looks back over the letters and picks out what matters. " \
"the loss goes down as the weights learn the shape of the words. " \
"gradient by gradient the machine finds the pattern in the stream. " \
"the fans hum in the dark and the numbers flow through the layers. " \
"query meets key and the softmax shares out the focus. " \
"value vectors ride the residual stream up through the blocks. " \
"the little model dreams in plain text and speaks one letter at a time. " \
"deep in the pool of floats the adam optimizer walks down the slope. " \
"the fox taps the keys and watches the loss fall line by line. "

static const char* g_builtin = TXT TXT TXT;

static long read_file(const char* path, char* buf, long maxn) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    long n = (long)fread(buf, 1, maxn, f);
    fclose(f);
    return n;
}

// ---------------------------------------------------------------------------
// tokenizer: characters + byte-pair merges + special turn tokens
//   ids 0..2 are specials, then one id per character, then merged tokens.
//   a merged token m stands for the pair (g_left[m], g_right[m]); chars and
//   specials have g_left = -1. merges are applied in id order to encode.
// ---------------------------------------------------------------------------
static int  g_stoi[256];
static char g_itos[MAXV];                // char for base tokens, 0 for merges
static int  g_left[MAXV], g_right[MAXV];
static char g_first[MAXV];               // first char of each token's text
static int  V = 0;

static int* g_data = 0;                  // tokenized training stream
static int* g_mask = 0;                  // 1 = compute loss on this target
static int  g_datalen = 0;

static char g_text[CORPUS_MAX];          // file loading scratch

static void vocab_reset() {
    for (int i = 0; i < 256; i++) g_stoi[i] = -1;
    V = 0;
    for (int s = 0; s < N_SPECIAL; s++) {
        g_itos[V] = g_first[V] = (char)(s + 1); g_stoi[s + 1] = V;
        g_left[V] = g_right[V] = -1;
        V++;
    }
}

static int char_to_tok(unsigned char ch) {          // -1 if unknown
    if (ch >= 'A' && ch <= 'Z') ch = (unsigned char)(ch - 'A' + 'a');
    return g_stoi[ch];
}

static void vocab_add_text(const char* text, long n) {
    for (long i = 0; i < n; i++) {
        unsigned char ch = (unsigned char)text[i];
        if (ch >= 'A' && ch <= 'Z') ch = (unsigned char)(ch - 'A' + 'a');
        if (ch < 32 && ch != '\n') continue;
        if (ch == '\n') ch = ' ';
        if (g_stoi[ch] < 0) {
            if (V >= MAXV) continue;
            g_itos[V] = g_first[V] = (char)ch;
            g_left[V] = g_right[V] = -1;
            g_stoi[ch] = V;
            V++;
        }
    }
}

// specials never merge, and a space may only start a token (" the", not "the ")
static bool mergeable(int a, int b) {
    return a >= N_SPECIAL && b >= N_SPECIAL && g_first[b] != ' ';
}

// replace each (a, b) in the stream with m; text runs share one mask value
static long merge_pass(int* tok, int* msk, long n, int a, int b, int m) {
    long w = 0;
    for (long r = 0; r < n; r++, w++) {
        if (msk) msk[w] = msk[r];
        if (r + 1 < n && tok[r] == a && tok[r + 1] == b) { tok[w] = m; r++; }
        else tok[w] = tok[r];
    }
    return w;
}

static long bpe_encode(int* tok, int* msk, long n) {
    for (int m = 0; m < V; m++)
        if (g_left[m] >= 0) n = merge_pass(tok, msk, n, g_left[m], g_right[m], m);
    return n;
}

static int g_paircnt[MAXV * MAXV];

static long bpe_learn(int* tok, int* msk, long n) {
    while (V < MAXV) {
        memset(g_paircnt, 0, sizeof(int) * V * MAXV);
        for (long i = 0; i + 1 < n; i++)
            if (mergeable(tok[i], tok[i + 1])) g_paircnt[tok[i] * MAXV + tok[i + 1]]++;
        int best = -1, bestc = 1;
        for (int i = 0; i < V * MAXV; i++)
            if (g_paircnt[i] > bestc) { bestc = g_paircnt[i]; best = i; }
        if (best < 0) break;
        int a = best / MAXV, b = best % MAXV, m = V++;
        g_left[m] = a; g_right[m] = b; g_itos[m] = 0; g_first[m] = g_first[a];
        n = merge_pass(tok, msk, n, a, b, m);
        if (V % 128 == 0) { printf("bpe: vocab %d, stream %ld tokens\n", V, n); fflush(stdout); }
    }
    return n;
}

static int tok_text(int t, char* out, int cap) {
    if (cap <= 0) return 0;
    if (g_left[t] < 0) {
        if (t < N_SPECIAL) return 0;
        out[0] = g_itos[t];
        return 1;
    }
    int k = tok_text(g_left[t], out, cap);
    return k + tok_text(g_right[t], out + k, cap - k);
}

static void data_begin() {
    if (!g_data) {
        g_data = (int*)malloc(sizeof(int) * CORPUS_MAX);
        g_mask = (int*)malloc(sizeof(int) * CORPUS_MAX);
    }
    g_datalen = 0;
}
static void data_push(int tok, int mask) {
    if (g_datalen >= CORPUS_MAX) return;
    g_data[g_datalen] = tok;
    g_mask[g_datalen] = mask;
    g_datalen++;
}
static void data_push_text(const char* text, long n, int mask) {
    for (long i = 0; i < n; i++) {
        unsigned char ch = (unsigned char)text[i];
        if (ch == '\n') ch = ' ';
        int t = char_to_tok(ch);
        if (t >= 0) data_push(t, mask);
    }
}
static void data_ensure(long minlen) {
    long orig = g_datalen;
    if (orig <= 0) return;
    while (g_datalen < minlen && g_datalen + orig <= CORPUS_MAX)
        for (long i = 0; i < orig; i++) data_push(g_data[i], g_mask[i]);
}

// ---------------------------------------------------------------------------
// parameters: one flat block carved into tensors (same order as tinyformer.cpp,
// which is also the checkpoint order). host and device copies share the layout.
// ---------------------------------------------------------------------------
struct Params {
    float *wte, *wpe;
    float *ln1w[L], *ln1b[L];
    float *qkvw[L], *qkvb[L];            // [3C, C], [3C]
    float *apw[L],  *apb[L];             // [C, C], [C]
    float *ln2w[L], *ln2b[L];
    float *f1w[L],  *f1b[L];             // [F, C], [F]
    float *f2w[L],  *f2b[L];             // [C, F], [C]
    float *lnfw, *lnfb;
    float *hw, *hb;                      // [V, C], [V]
};

static long param_count() {
    long per_layer = 4L*C + 3L*C*C + 3L*C + (long)C*C + C + (long)F*C + F + (long)C*F + C;
    return 2L*V*C + (long)T*C + L*per_layer + 2L*C + V;
}

static void layout_params(Params& P, float* base) {
    float* p = base;
    auto take = [&](long n) { float* r = p; p += n; return r; };
    P.wte = take((long)V * C);
    P.wpe = take((long)T * C);
    for (int l = 0; l < L; l++) {
        P.ln1w[l] = take(C);           P.ln1b[l] = take(C);
        P.qkvw[l] = take(3L * C * C);  P.qkvb[l] = take(3 * C);
        P.apw[l]  = take((long)C * C); P.apb[l]  = take(C);
        P.ln2w[l] = take(C);           P.ln2b[l] = take(C);
        P.f1w[l]  = take((long)F * C); P.f1b[l]  = take(F);
        P.f2w[l]  = take((long)C * F); P.f2b[l]  = take(C);
    }
    P.lnfw = take(C); P.lnfb = take(C);
    P.hw = take((long)V * C); P.hb = take(V);
    if (p - base != param_count()) { printf("FATAL: param layout mismatch\n"); exit(1); }
}

static void init_params(Params& P) {
    for (long i = 0; i < (long)V * C; i++) P.wte[i] = rnd_normal() * 0.02f;
    for (long i = 0; i < (long)T * C; i++) P.wpe[i] = rnd_normal() * 0.02f;
    for (int l = 0; l < L; l++) {
        for (int i = 0; i < C; i++) { P.ln1w[l][i] = 1.0f; P.ln2w[l][i] = 1.0f; }
        for (long i = 0; i < 3L*C*C;    i++) P.qkvw[l][i] = rnd_normal() * 0.02f;
        for (long i = 0; i < (long)C*C; i++) P.apw[l][i]  = rnd_normal() * 0.02f;
        for (long i = 0; i < (long)F*C; i++) P.f1w[l][i]  = rnd_normal() * 0.02f;
        for (long i = 0; i < (long)C*F; i++) P.f2w[l][i]  = rnd_normal() * 0.02f;
    }
    for (int i = 0; i < C; i++) P.lnfw[i] = 1.0f;
    for (long i = 0; i < (long)V * C; i++) P.hw[i] = rnd_normal() * 0.02f;
}

static long   NP = 0;
static float* h_par;                     // host params (inference, checkpoints)
static Params hP;

// ---------------------------------------------------------------------------
// shared math (host inference and device kernels)
// ---------------------------------------------------------------------------
HD static inline float gelu_f(float x) {
    float u = 0.7978845608f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(u));
}
HD static inline float gelu_df(float x) {
    float u  = 0.7978845608f * (x + 0.044715f * x * x * x);
    float th = tanhf(u);
    float s2 = 1.0f - th * th;
    return 0.5f*(1.0f + th) + 0.5f*x*s2*0.7978845608f*(1.0f + 0.134145f*x*x);
}

// ---------------------------------------------------------------------------
// kernels. a batch is N = B*T rows; row r is position r % T of sequence r / T.
// ---------------------------------------------------------------------------
KERNEL k_embed(float* out, const int* tok, const float* wte, const float* wpe, long n) {
    long i = KID; if (i >= n) return;
    long r = i / C, c = i % C;
    out[i] = wte[(long)tok[r]*C + c] + wpe[(r % T)*C + c];
}
KERNEL k_embed_bwd(float* dwte, float* dwpe, const float* dx, const int* tok, long n) {
    long i = KID; if (i >= n) return;
    long r = i / C, c = i % C;
    atomicAdd(&dwte[(long)tok[r]*C + c], dx[i]);
    atomicAdd(&dwpe[(r % T)*C + c], dx[i]);
}

KERNEL k_add_bias(float* out, const float* b, int cols, long n) {
    long i = KID; if (i >= n) return;
    out[i] += b[i % cols];
}
KERNEL k_add(float* out, const float* a, long n) {
    long i = KID; if (i >= n) return;
    out[i] += a[i];
}
KERNEL k_colsum(float* db, const float* d, int rows, int cols) {   // db += sum over rows
    long c = KID; if (c >= cols) return;
    float s = 0.0f;
    for (long r = 0; r < rows; r++) s += d[r*cols + c];
    db[c] += s;
}

KERNEL k_gelu(float* out, const float* in, long n) {
    long i = KID; if (i >= n) return;
    out[i] = gelu_f(in[i]);
}
KERNEL k_gelu_bwd(float* d, const float* in, long n) {
    long i = KID; if (i >= n) return;
    d[i] *= gelu_df(in[i]);
}

KERNEL k_ln_fwd(float* out, float* mean, float* rstd, const float* in,
                const float* w, const float* b, int rows) {
    long t = KID; if (t >= rows) return;
    const float* x = in + t * C;
    float m = 0.0f;
    for (int i = 0; i < C; i++) m += x[i];
    m /= C;
    float v = 0.0f;
    for (int i = 0; i < C; i++) { float d = x[i] - m; v += d * d; }
    v /= C;
    float r = 1.0f / sqrtf(v + 1e-5f);
    mean[t] = m; rstd[t] = r;
    float* o = out + t * C;
    for (int i = 0; i < C; i++) o[i] = (x[i] - m) * r * w[i] + b[i];
}
// din += d(layernorm input), one thread per row
KERNEL k_ln_bwd_x(float* din, const float* dout, const float* in, const float* w,
                  const float* mean, const float* rstd, int rows) {
    long t = KID; if (t >= rows) return;
    const float* x  = in   + t * C;
    const float* dy = dout + t * C;
    float m = mean[t], r = rstd[t];
    float dnm = 0.0f, dnnm = 0.0f;
    for (int i = 0; i < C; i++) {
        float norm  = (x[i] - m) * r;
        float dnorm = w[i] * dy[i];
        dnm  += dnorm;
        dnnm += dnorm * norm;
    }
    dnm /= C; dnnm /= C;
    float* dx = din + t * C;
    for (int i = 0; i < C; i++) {
        float norm  = (x[i] - m) * r;
        dx[i] += (w[i] * dy[i] - dnm - norm * dnnm) * r;
    }
}
// dw, db += layernorm weight grads, one thread per channel
KERNEL k_ln_bwd_w(float* dw, float* db, const float* dout, const float* in,
                  const float* mean, const float* rstd, int rows) {
    long i = KID; if (i >= C) return;
    float sw = 0.0f, sb = 0.0f;
    for (long t = 0; t < rows; t++) {
        float dy = dout[t*C + i];
        sw += dy * (in[t*C + i] - mean[t]) * rstd[t];
        sb += dy;
    }
    dw[i] += sw; db[i] += sb;
}

// causal attention, one thread per (sequence, head, position): i = (b*H + h)*T + t
KERNEL k_attn_fwd(float* atty, float* att, const float* qkv, long n) {
    long i = KID; if (i >= n) return;
    long t = i % T, h = (i / T) % H, b = i / (T * H);
    const float scale = 1.0f / sqrtf((float)HS);
    const float* base = qkv + 3L*C*(b*T);
    const float* q = base + 3L*C*t + h*HS;
    float* arow = att + i*T;
    float mx = -1e30f;
    for (int t2 = 0; t2 <= t; t2++) {
        const float* k = base + 3L*C*t2 + C + h*HS;
        float s = 0.0f;
        for (int j = 0; j < HS; j++) s += q[j] * k[j];
        s *= scale;
        arow[t2] = s;
        if (s > mx) mx = s;
    }
    float sum = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) { float e = expf(arow[t2] - mx); arow[t2] = e; sum += e; }
    float inv = 1.0f / sum;
    for (int t2 = 0;     t2 <= t; t2++) arow[t2] *= inv;
    for (int t2 = t + 1; t2 <  T; t2++) arow[t2]  = 0.0f;
    float* o = atty + (b*T + t)*C + h*HS;
    for (int j = 0; j < HS; j++) o[j] = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
        const float* v = base + 3L*C*t2 + 2L*C + h*HS;
        float a = arow[t2];
        for (int j = 0; j < HS; j++) o[j] += a * v[j];
    }
}
// backward part 1, per query row: dpre = d(pre-softmax scores) * scale, and dq
KERNEL k_attn_bwd_q(float* dqkv, float* dpre, const float* datty,
                    const float* att, const float* qkv, long n) {
    long i = KID; if (i >= n) return;
    long t = i % T, h = (i / T) % H, b = i / (T * H);
    const float scale = 1.0f / sqrtf((float)HS);
    const float* base = qkv + 3L*C*(b*T);
    const float* arow = att  + i*T;
    float*       drow = dpre + i*T;
    const float* do_  = datty + (b*T + t)*C + h*HS;
    float dsum = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
        const float* v = base + 3L*C*t2 + 2L*C + h*HS;
        float d = 0.0f;
        for (int j = 0; j < HS; j++) d += do_[j] * v[j];
        drow[t2] = d;
        dsum += arow[t2] * d;
    }
    for (int t2 = 0;     t2 <= t; t2++) drow[t2] = arow[t2] * (drow[t2] - dsum) * scale;
    for (int t2 = t + 1; t2 <  T; t2++) drow[t2] = 0.0f;
    float* dq = dqkv + 3L*C*(b*T + t) + h*HS;
    for (int j = 0; j < HS; j++) {
        float s = 0.0f;
        for (int t2 = 0; t2 <= t; t2++) s += drow[t2] * base[3L*C*t2 + C + h*HS + j];
        dq[j] = s;
    }
}
// backward part 2, per key row t2: dk and dv gather over the queries t >= t2
KERNEL k_attn_bwd_kv(float* dqkv, const float* dpre, const float* datty,
                     const float* att, const float* qkv, long n) {
    long i = KID; if (i >= n) return;
    long t2 = i % T, h = (i / T) % H, b = i / (T * H);
    const float* base = qkv + 3L*C*(b*T);
    const float* arows = att  + (i - t2)*T;            // (b, h, t=0) row
    const float* drows = dpre + (i - t2)*T;
    float* dk = dqkv + 3L*C*(b*T + t2) + C + h*HS;
    float* dv = dk + C;
    for (int j = 0; j < HS; j++) { dk[j] = 0.0f; dv[j] = 0.0f; }
    for (long t = t2; t < T; t++) {
        float dp = drows[t*T + t2], a = arows[t*T + t2];
        const float* q  = base + 3L*C*t + h*HS;
        const float* do_ = datty + (b*T + t)*C + h*HS;
        for (int j = 0; j < HS; j++) { dk[j] += dp * q[j]; dv[j] += a * do_[j]; }
    }
}

// softmax + cross-entropy per row. logits are overwritten with d(logits),
// pre-scaled by rowscale (0 on masked rows, 1/(count*B) on counted ones)
KERNEL k_softmax_ce(float* logits, float* rowloss, const int* tgt,
                    const float* rowscale, int rows, int nv) {
    long r = KID; if (r >= rows) return;
    float* lg = logits + r * nv;
    float mx = -1e30f;
    for (int v = 0; v < nv; v++) if (lg[v] > mx) mx = lg[v];
    float sum = 0.0f;
    for (int v = 0; v < nv; v++) sum += expf(lg[v] - mx);
    rowloss[r] = logf(sum) + mx - lg[tgt[r]];
    float inv = 1.0f / sum, s = rowscale[r];
    for (int v = 0; v < nv; v++)
        lg[v] = (expf(lg[v] - mx) * inv - (v == tgt[r] ? 1.0f : 0.0f)) * s;
}

KERNEL k_adam(float* p, const float* g, float* m, float* v,
              float lr, float c1, float c2, long n) {
    long i = KID; if (i >= n) return;
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    m[i] = b1*m[i] + (1.0f - b1)*g[i];
    v[i] = b2*v[i] + (1.0f - b2)*g[i]*g[i];
    p[i] -= lr * (m[i] * c1) / (sqrtf(v[i] * c2) + eps);
}

// ---------------------------------------------------------------------------
// matmuls. tensors are row-major; w is [Cout, Cin] as in tinyformer.cpp
// ---------------------------------------------------------------------------
// out[N,Cout] = in[N,Cin] . w^T + b
static void mm_fwd(float* out, const float* in, const float* w, const float* b,
                   int N, int Cin, int Cout) {
#ifdef __CUDACC__
    const float one = 1.0f, zero = 0.0f;
    CKB(cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, Cout, N, Cin,
                    &one, w, Cin, in, Cin, &zero, out, Cout));
#else
    for (long t = 0; t < N; t++) {
        const float* ir = in + t * Cin;
        float* orow = out + t * Cout;
        for (int o = 0; o < Cout; o++) {
            const float* wr = w + (long)o * Cin;
            float s = 0.0f;
            for (int i = 0; i < Cin; i++) s += ir[i] * wr[i];
            orow[o] = s;
        }
    }
#endif
    LAUNCH(k_add_bias, (long)N * Cout, out, b, Cout, (long)N * Cout);
}

// din = dout . w (overwritten), dw += dout^T . in, db += column sums of dout
static void mm_bwd(float* din, float* dw, float* db, const float* dout,
                   const float* in, const float* w, int N, int Cin, int Cout) {
#ifdef __CUDACC__
    const float one = 1.0f, zero = 0.0f;
    CKB(cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_N, Cin, N, Cout,
                    &one, w, Cin, dout, Cout, &zero, din, Cin));
    CKB(cublasSgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_T, Cin, Cout, N,
                    &one, in, Cin, dout, Cout, &one, dw, Cin));
#else
    for (long t = 0; t < N; t++) {
        const float* dor = dout + t * Cout;
        const float* ir  = in   + t * Cin;
        float* dir = din + t * Cin;
        for (int i = 0; i < Cin; i++) dir[i] = 0.0f;
        for (int o = 0; o < Cout; o++) {
            const float* wr = w + (long)o * Cin;
            float* dwr = dw + (long)o * Cin;
            float d = dor[o];
            for (int i = 0; i < Cin; i++) { dir[i] += d * wr[i]; dwr[i] += d * ir[i]; }
        }
    }
#endif
    LAUNCH(k_colsum, Cout, db, dout, N, Cout);
}

// ---------------------------------------------------------------------------
// device training state. only what backward reads is kept per layer; gradient
// scratch is shared across layers (dres carries the residual-stream gradient)
// ---------------------------------------------------------------------------
struct Acts {
    float *x0;                                    // [N, C]
    float *ln1o[L], *ln1m[L], *ln1r[L];           // [N,C], [N], [N]
    float *qkv[L];                                // [N, 3C]
    float *att[L];                                // [B, H, T, T]
    float *atty[L], *res1[L];                     // [N, C]
    float *ln2o[L], *ln2m[L], *ln2r[L];
    float *fch[L], *fchg[L];                      // [N, F] pre/post GELU
    float *res2[L];                               // [N, C]
    float *lnfo, *lnfm, *lnfr;
    float *logits;                                // [N, V], becomes d(logits)
    float *rowloss;                               // [N]
};

static Acts   A;
static Params dP, dG;                             // device params and grads
static float *d_par, *d_grad, *d_m, *d_v;
static float *d_res, *d_ln, *d_atty, *d_qkv, *d_ff, *d_pre;   // grad scratch
static int   *d_tok, *d_tgt;
static float *d_rowscale;

static void setup_device() {
#ifdef __CUDACC__
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("gpu: %s, %.1f GB\n", prop.name, (double)prop.totalGlobalMem / 1e9);
    CKB(cublasCreate(&g_blas));
    CKB(cublasSetMathMode(g_blas, CUBLAS_TF32_TENSOR_OP_MATH));   // tensor cores on ampere+
#else
    printf("cpu build: kernels run serially (reference speed)\n");
#endif
    long N = (long)B * T;
    d_par = dalloc<float>(NP); d_grad = dalloc<float>(NP);
    d_m   = dalloc<float>(NP); d_v    = dalloc<float>(NP);
    layout_params(dP, d_par);
    layout_params(dG, d_grad);
    A.x0 = dalloc<float>(N * C);
    for (int l = 0; l < L; l++) {
        A.ln1o[l] = dalloc<float>(N*C); A.ln1m[l] = dalloc<float>(N); A.ln1r[l] = dalloc<float>(N);
        A.qkv[l]  = dalloc<float>(3*N*C);
        A.att[l]  = dalloc<float>((long)B*H*T*T);
        A.atty[l] = dalloc<float>(N*C);
        A.res1[l] = dalloc<float>(N*C);
        A.ln2o[l] = dalloc<float>(N*C); A.ln2m[l] = dalloc<float>(N); A.ln2r[l] = dalloc<float>(N);
        A.fch[l]  = dalloc<float>(N*F); A.fchg[l] = dalloc<float>(N*F);
        A.res2[l] = dalloc<float>(N*C);
    }
    A.lnfo = dalloc<float>(N*C); A.lnfm = dalloc<float>(N); A.lnfr = dalloc<float>(N);
    A.logits = dalloc<float>(N*V); A.rowloss = dalloc<float>(N);
    d_res = dalloc<float>(N*C); d_ln = dalloc<float>(N*C); d_atty = dalloc<float>(N*C);
    d_qkv = dalloc<float>(3*N*C); d_ff = dalloc<float>(N*F);
    d_pre = dalloc<float>((long)B*H*T*T);
    d_tok = dalloc<int>(N); d_tgt = dalloc<int>(N); d_rowscale = dalloc<float>(N);
}

static void params_to_dev()  { to_dev(d_par, h_par, NP * sizeof(float)); }
static void params_to_host() { to_host(h_par, d_par, NP * sizeof(float)); }

// forward + backward for the batch already uploaded to d_tok / d_tgt / d_rowscale
static void forward_backward() {
    const int N = B * T;
    const long NC = (long)N * C, NF = (long)N * F, BHT = (long)B * H * T;

    LAUNCH(k_embed, NC, A.x0, d_tok, dP.wte, dP.wpe, NC);
    const float* x = A.x0;
    for (int l = 0; l < L; l++) {
        LAUNCH(k_ln_fwd, N, A.ln1o[l], A.ln1m[l], A.ln1r[l], x, dP.ln1w[l], dP.ln1b[l], N);
        mm_fwd(A.qkv[l], A.ln1o[l], dP.qkvw[l], dP.qkvb[l], N, C, 3*C);
        LAUNCH(k_attn_fwd, BHT, A.atty[l], A.att[l], A.qkv[l], BHT);
        mm_fwd(A.res1[l], A.atty[l], dP.apw[l], dP.apb[l], N, C, C);
        LAUNCH(k_add, NC, A.res1[l], x, NC);
        LAUNCH(k_ln_fwd, N, A.ln2o[l], A.ln2m[l], A.ln2r[l], A.res1[l], dP.ln2w[l], dP.ln2b[l], N);
        mm_fwd(A.fch[l], A.ln2o[l], dP.f1w[l], dP.f1b[l], N, C, F);
        LAUNCH(k_gelu, NF, A.fchg[l], A.fch[l], NF);
        mm_fwd(A.res2[l], A.fchg[l], dP.f2w[l], dP.f2b[l], N, F, C);
        LAUNCH(k_add, NC, A.res2[l], A.res1[l], NC);
        x = A.res2[l];
    }
    LAUNCH(k_ln_fwd, N, A.lnfo, A.lnfm, A.lnfr, x, dP.lnfw, dP.lnfb, N);
    mm_fwd(A.logits, A.lnfo, dP.hw, dP.hb, N, C, V);
    LAUNCH(k_softmax_ce, N, A.logits, A.rowloss, d_tgt, d_rowscale, N, V);

    // backward: A.logits now holds d(logits)
    mm_bwd(d_ln, dG.hw, dG.hb, A.logits, A.lnfo, dP.hw, N, C, V);
    dzero(d_res, NC * sizeof(float));
    LAUNCH(k_ln_bwd_x, N, d_res, d_ln, A.res2[L-1], dP.lnfw, A.lnfm, A.lnfr, N);
    LAUNCH(k_ln_bwd_w, C, dG.lnfw, dG.lnfb, d_ln, A.res2[L-1], A.lnfm, A.lnfr, N);

    for (int l = L - 1; l >= 0; l--) {
        const float* xin = (l == 0) ? A.x0 : A.res2[l-1];
        // res2 = res1 + mlp(ln2(res1)); d_res is d(res2) and becomes d(res1)
        mm_bwd(d_ff, dG.f2w[l], dG.f2b[l], d_res, A.fchg[l], dP.f2w[l], N, F, C);
        LAUNCH(k_gelu_bwd, NF, d_ff, A.fch[l], NF);
        mm_bwd(d_ln, dG.f1w[l], dG.f1b[l], d_ff, A.ln2o[l], dP.f1w[l], N, C, F);
        LAUNCH(k_ln_bwd_x, N, d_res, d_ln, A.res1[l], dP.ln2w[l], A.ln2m[l], A.ln2r[l], N);
        LAUNCH(k_ln_bwd_w, C, dG.ln2w[l], dG.ln2b[l], d_ln, A.res1[l], A.ln2m[l], A.ln2r[l], N);
        // res1 = x + attn(ln1(x)); d_res is d(res1) and becomes d(x)
        mm_bwd(d_atty, dG.apw[l], dG.apb[l], d_res, A.atty[l], dP.apw[l], N, C, C);
        LAUNCH(k_attn_bwd_q,  BHT, d_qkv, d_pre, d_atty, A.att[l], A.qkv[l], BHT);
        LAUNCH(k_attn_bwd_kv, BHT, d_qkv, d_pre, d_atty, A.att[l], A.qkv[l], BHT);
        mm_bwd(d_ln, dG.qkvw[l], dG.qkvb[l], d_qkv, A.ln1o[l], dP.qkvw[l], N, C, 3*C);
        LAUNCH(k_ln_bwd_x, N, d_res, d_ln, xin, dP.ln1w[l], A.ln1m[l], A.ln1r[l], N);
        LAUNCH(k_ln_bwd_w, C, dG.ln1w[l], dG.ln1b[l], d_ln, xin, A.ln1m[l], A.ln1r[l], N);
    }
    LAUNCH(k_embed_bwd, NC, dG.wte, dG.wpe, d_res, d_tok, NC);
}

// ---------------------------------------------------------------------------
// checkpoint: [magic][V T C L H][itos bytes][left ints][right ints][params]
//   identical to tinyformer.cpp; char-level checkpoints have no left/right
// ---------------------------------------------------------------------------
static const char* CKPT = "tinyformer.ckpt";
enum { CKPT_CHAR = 0x54465632, CKPT_BPE = 0x54465633 };

static bool save_ckpt() {
    FILE* f = fopen(CKPT, "wb");
    if (!f) return false;
    int hdr[6] = { CKPT_BPE, V, T, C, L, H };
    fwrite(hdr, sizeof(hdr), 1, f);
    fwrite(g_itos, 1, V, f);
    fwrite(g_left, sizeof(int), V, f);
    fwrite(g_right, sizeof(int), V, f);
    fwrite(h_par, sizeof(float), NP, f);
    fclose(f);
    return true;
}

// read the header and vocab; leaves f at the params (null on error)
static FILE* open_ckpt_vocab() {
    FILE* f = fopen(CKPT, "rb");
    if (!f) return 0;
    int hdr[6];
    if (fread(hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return 0; }
    if ((hdr[0] != CKPT_CHAR && hdr[0] != CKPT_BPE) || hdr[2] != T || hdr[3] != C
        || hdr[4] != L || hdr[5] != H || hdr[1] > MAXV) {
        printf("checkpoint does not match compiled config\n");
        fclose(f); return 0;
    }
    V = hdr[1];
    fread(g_itos, 1, V, f);
    for (int i = 0; i < V; i++) g_left[i] = g_right[i] = -1;
    if (hdr[0] == CKPT_BPE) {
        fread(g_left, sizeof(int), V, f);
        fread(g_right, sizeof(int), V, f);
    }
    for (int i = 0; i < 256; i++) g_stoi[i] = -1;
    for (int i = 0; i < V; i++) {
        if (g_left[i] < 0) { g_first[i] = g_itos[i]; g_stoi[(unsigned char)g_itos[i]] = i; }
        else g_first[i] = g_first[g_left[i]];
    }
    return f;
}

static void setup_host_params() {        // call once V is known
    NP = param_count();
    h_par = (float*)calloc(NP, sizeof(float));
    layout_params(hP, h_par);
}

static bool load_ckpt() {
    FILE* f = open_ckpt_vocab();
    if (!f) return false;
    setup_host_params();
    long got = (long)fread(h_par, sizeof(float), NP, f);
    fclose(f);
    if (got != NP) { printf("checkpoint read failed\n"); return false; }
    return true;
}

// ---------------------------------------------------------------------------
// KV-cached incremental inference on the host (sample and chat)
// ---------------------------------------------------------------------------
static float kcache[L][(long)T * C];
static float vcache[L][(long)T * C];

static void ln_row(float* out, const float* x, const float* w, const float* b) {
    float m = 0.0f;
    for (int i = 0; i < C; i++) m += x[i];
    m /= C;
    float v = 0.0f;
    for (int i = 0; i < C; i++) { float d = x[i] - m; v += d * d; }
    float r = 1.0f / sqrtf(v / C + 1e-5f);
    for (int i = 0; i < C; i++) out[i] = (x[i] - m) * r * w[i] + b[i];
}
static void mat_row(float* out, const float* in, const float* w,
                    const float* b, int Cin, int Cout) {
    for (int o = 0; o < Cout; o++) {
        const float* wr = w + (long)o * Cin;
        float s = b ? b[o] : 0.0f;
        for (int i = 0; i < Cin; i++) s += in[i] * wr[i];
        out[o] = s;
    }
}

// process one token at position pos (0-based); returns logits row [V]
static const float* forward_one(Params& P, int token, int pos) {
    static float xb[C], l1[C], qkvr[3*C], aty[C], apo[C], l2[C],
                 fh[F], fhg[F], fpo[C], lnf[C], lg[MAXV], sc_[T];
    for (int i = 0; i < C; i++)
        xb[i] = P.wte[(long)token*C + i] + P.wpe[(long)pos*C + i];

    const float scale = 1.0f / sqrtf((float)HS);
    for (int l = 0; l < L; l++) {
        ln_row(l1, xb, P.ln1w[l], P.ln1b[l]);
        mat_row(qkvr, l1, P.qkvw[l], P.qkvb[l], C, 3*C);
        for (int i = 0; i < C; i++) {
            kcache[l][(long)pos*C + i] = qkvr[C + i];
            vcache[l][(long)pos*C + i] = qkvr[2*C + i];
        }
        for (int h = 0; h < H; h++) {
            const float* q = qkvr + h*HS;
            float mx = -1e30f;
            for (int t2 = 0; t2 <= pos; t2++) {
                const float* k = kcache[l] + (long)t2*C + (long)h*HS;
                float s = 0.0f;
                for (int i = 0; i < HS; i++) s += q[i] * k[i];
                s *= scale;
                sc_[t2] = s;
                if (s > mx) mx = s;
            }
            float sum = 0.0f;
            for (int t2 = 0; t2 <= pos; t2++) { float e = expf(sc_[t2] - mx); sc_[t2] = e; sum += e; }
            float inv = 1.0f / sum;
            float* o = aty + h*HS;
            for (int i = 0; i < HS; i++) o[i] = 0.0f;
            for (int t2 = 0; t2 <= pos; t2++) {
                const float* v = vcache[l] + (long)t2*C + (long)h*HS;
                float a = sc_[t2] * inv;
                for (int i = 0; i < HS; i++) o[i] += a * v[i];
            }
        }
        mat_row(apo, aty, P.apw[l], P.apb[l], C, C);
        for (int i = 0; i < C; i++) xb[i] += apo[i];
        ln_row(l2, xb, P.ln2w[l], P.ln2b[l]);
        mat_row(fh, l2, P.f1w[l], P.f1b[l], C, F);
        for (int i = 0; i < F; i++) fhg[i] = gelu_f(fh[i]);
        mat_row(fpo, fhg, P.f2w[l], P.f2b[l], F, C);
        for (int i = 0; i < C; i++) xb[i] += fpo[i];
    }
    ln_row(lnf, xb, P.lnfw, P.lnfb);
    mat_row(lg, lnf, P.hw, P.hb, C, V);
    return lg;
}

static int sample_logits(const float* lg, float temp) {
    float mx = -1e30f;
    for (int v = 0; v < V; v++) if (lg[v] > mx) mx = lg[v];
    float pr[MAXV], sum = 0.0f;
    for (int v = 0; v < V; v++) { pr[v] = expf((lg[v] - mx) / temp); sum += pr[v]; }
    float r = rnd_uniform() * sum, acc = 0.0f;
    for (int v = 0; v < V; v++) { acc += pr[v]; if (r <= acc) return v; }
    return V - 1;
}

// ---------------------------------------------------------------------------
// training loop (shared by pretrain, resume and finetune)
// ---------------------------------------------------------------------------
static void train(int steps, float lr, int log_every) {
    const int N = B * T;
    std::vector<int> tok(N), tgt(N);
    std::vector<float> rowscale(N), rowloss(N);
    float b1t = 1.0f, b2t = 1.0f;
    params_to_dev();
    auto t0 = std::chrono::steady_clock::now();
    for (int step = 1; step <= steps; step++) {
        float prog = (float)(step - 1) / (float)steps;
        float lr_t = 0.1f*lr + 0.45f*lr*(1.0f + cosf(3.14159265f * prog));
        // sample B windows exactly as tinyformer.cpp does, one after another
        for (int b = 0; b < B; b++) {
            int pos = (int)(rnd_u32() % (u32)(g_datalen - T - 1));
            int have = 0;
            for (int tries = 0; tries < 20 && !have; tries++) {
                for (int t = 0; t < T; t++) have |= g_mask[pos + t + 1];
                if (!have) pos = (int)(rnd_u32() % (u32)(g_datalen - T - 1));
            }
            int cnt = 0;
            for (int t = 0; t < T; t++) cnt += g_mask[pos + t + 1] ? 1 : 0;
            for (int t = 0; t < T; t++) {
                tok[b*T + t] = g_data[pos + t];
                tgt[b*T + t] = g_data[pos + t + 1];
                rowscale[b*T + t] = g_mask[pos + t + 1] ? 1.0f / ((float)cnt * (float)B) : 0.0f;
            }
        }
        to_dev(d_tok, tok.data(), N * sizeof(int));
        to_dev(d_tgt, tgt.data(), N * sizeof(int));
        to_dev(d_rowscale, rowscale.data(), N * sizeof(float));

        dzero(d_grad, NP * sizeof(float));
        forward_backward();
        b1t *= 0.9f; b2t *= 0.999f;
        LAUNCH(k_adam, NP, d_par, d_grad, d_m, d_v, lr_t,
               1.0f / (1.0f - b1t), 1.0f / (1.0f - b2t), NP);

        if (step == 1 || step % log_every == 0) {
            // reported loss: mean over sequences of each one's mean masked loss
            to_host(rowloss.data(), A.rowloss, N * sizeof(float));
            double loss = 0.0;
            for (int r = 0; r < N; r++) loss += (double)rowloss[r] * rowscale[r];
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            int done = step == 1 ? 1 : log_every;
            printf("step %5d | loss %.4f | %.1f ms/step\n", step, loss, ms / done);
            fflush(stdout);
            t0 = t1;
        }
        if (step % 2000 == 0 && step < steps) { params_to_host(); save_ckpt(); }
    }
#ifdef __CUDACC__
    CK(cudaGetLastError());
#endif
    params_to_host();
}

// free-running generation; when the context fills, re-prefill with its last half
static void generate(int seed_tok, int nchars, float temp) {
    std::vector<int> hist(1, seed_tok);
    char txt[256];
    int shown = tok_text(seed_tok, txt, 256);
    printf("%.*s", shown, txt);
    const float* lg = forward_one(hP, seed_tok, 0);
    int pos = 1;
    while (shown < nchars) {
        int pick = sample_logits(lg, temp);
        int k = tok_text(pick, txt, 256);
        printf("%.*s", k, txt);
        fflush(stdout);
        shown += k > 0 ? k : 1;
        hist.push_back(pick);
        if (pos < T) lg = forward_one(hP, pick, pos++);
        else {
            pos = 0;
            for (size_t i = hist.size() - T / 2; i < hist.size(); i++)
                lg = forward_one(hP, hist[i], pos++);
        }
    }
    printf("\n");
}

// ---------------------------------------------------------------------------
// chat REPL
// ---------------------------------------------------------------------------
enum { CHAT_KEEP = 64, CHAT_MAXGEN = 200 };

static void chat() {
    std::vector<int> hist;
    char line[512];
    printf("tinyformer chat. empty line or ctrl-d to quit.\n");
    for (;;) {
        printf("you: ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;
        long n = (long)strlen(line);
        while (n > 0 && (line[n-1] == '\n' || line[n-1] == '\r')) line[--n] = 0;
        if (n <= 0 || !strcmp(line, "quit") || !strcmp(line, "exit")) break;
        for (long i = 0; i < n; i++) g_rng ^= (u64)(unsigned char)line[i] << ((i & 7) * 8);

        // dialog training data is single-turn, so start each turn fresh
        hist.clear();
        hist.push_back(TOK_USER);
        for (long i = 0; i < n; i++) {
            int t = char_to_tok((unsigned char)line[i]);
            if (t >= 0) hist.push_back(t);
        }
        hist.resize(1 + bpe_encode(hist.data() + 1, 0, (long)hist.size() - 1));
        hist.push_back(TOK_END);
        hist.push_back(TOK_BOT);

        // prefill the KV cache with the prompt, leaving room to respond
        int len = (int)hist.size();
        int win = len < CHAT_KEEP ? len : CHAT_KEEP;
        int off = len - win;
        const float* lg = 0;
        for (int t = 0; t < win; t++) lg = forward_one(hP, hist[off + t], t);

        printf("bot: ");
        int pos = win, made = 0;
        while (pos < T && made < CHAT_MAXGEN) {
            int pick = sample_logits(lg, 0.4f);
            if (pick == TOK_END) break;
            char txt[256];
            printf("%.*s", tok_text(pick, txt, 256), txt);
            fflush(stdout);
            lg = forward_one(hP, pick, pos++);
            made++;
        }
        printf("\n");
    }
    printf("bye\n");
}

// ---------------------------------------------------------------------------
// dialog file -> masked token stream (see tinyformer.cpp)
// ---------------------------------------------------------------------------
static void build_dialog_data(const char* text, long n) {
    data_begin();
    int role = 0;
    long i = 0;
    while (i < n) {
        long j = i;
        while (j < n && text[j] != '\n') j++;
        long linelen = j - i;
        bool blank = true;
        for (long k = i; k < j; k++) if (text[k] != ' ' && text[k] != '\r') blank = false;
        if (linelen > 0 && !blank) {
            if (role == 0) {
                data_push(TOK_USER, 0);
                data_push_text(text + i, linelen, 0);
                data_push(TOK_END, 0);
                role = 1;
            } else {
                data_push(TOK_BOT, 0);
                data_push_text(text + i, linelen, 1);
                data_push(TOK_END, 1);
                role = 0;
            }
        }
        i = j + 1;
    }
    g_datalen = (int)bpe_encode(g_data, g_mask, g_datalen);
    data_ensure(8L * T);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
static int parse_int(const char* s, int fallback) {
    int v = 0;
    for (const char* p = s; *p; p++) {
        if (*p < '0' || *p > '9') return fallback;
        v = v * 10 + (*p - '0');
    }
    return v > 0 ? v : fallback;
}

int main(int argc, char** argv) {
    // pull "-b N" out of argv so the positional arguments match tinyformer.cpp
    int ac = 0;
    for (int i = 0; i < argc; i++) {
        if ((!strcmp(argv[i], "-b") || !strcmp(argv[i], "--batch")) && i + 1 < argc)
            B = parse_int(argv[++i], B);
        else argv[ac++] = argv[i];
    }
    argc = ac;
    const char* mode = argc > 1 ? argv[1] : "help";

    if (!strcmp(mode, "train")) {
        int steps = argc > 2 ? parse_int(argv[2], 2000) : 2000;
        const char* src = g_builtin;
        long n = (long)strlen(g_builtin);
        if (argc > 3) {
            n = read_file(argv[3], g_text, CORPUS_MAX);
            if (n <= 0) { printf("cannot read %s\n", argv[3]); return 1; }
            src = g_text;
            printf("loaded %ld chars from %s\n", n, argv[3]);
        }
        vocab_reset();
        vocab_add_text(src, n);
        data_begin();
        data_push_text(src, n, 1);
        g_datalen = (int)bpe_learn(g_data, g_mask, g_datalen);
        data_ensure(8L * T);
        setup_host_params();
        init_params(hP);
        setup_device();
        printf("pretrain: %d tokens | vocab %d | context %d | dim %d | layers %d | batch %d | params %ld\n\n",
               g_datalen, V, T, C, L, B, NP);
        train(steps, 0.001f, 250);
        if (save_ckpt()) printf("\nsaved %s\n", CKPT);
        printf("\n--- sample ---\n");
        generate(N_SPECIAL, 300, 0.8f);
    }
    else if (!strcmp(mode, "resume")) {
        if (argc < 4) { printf("usage: tinyformer_gpu resume [steps] corpus.txt\n"); return 1; }
        int steps = parse_int(argv[2], 2000);
        if (!load_ckpt()) { printf("no checkpoint. run train first.\n"); return 1; }
        long n = read_file(argv[3], g_text, CORPUS_MAX);
        if (n <= 0) { printf("cannot read %s\n", argv[3]); return 1; }
        data_begin();
        data_push_text(g_text, n, 1);
        g_datalen = (int)bpe_encode(g_data, g_mask, g_datalen);
        data_ensure(8L * T);
        setup_device();
        printf("resume: %d tokens | vocab %d | batch %d | params %ld\n\n", g_datalen, V, B, NP);
        train(steps, 0.001f, 250);
        if (save_ckpt()) printf("\nsaved %s\n", CKPT);
    }
    else if (!strcmp(mode, "finetune")) {
        if (argc < 4) { printf("usage: tinyformer_gpu finetune [steps] dialog.txt\n"); return 1; }
        int steps = parse_int(argv[2], 1500);
        if (!load_ckpt()) { printf("no checkpoint. run train first.\n"); return 1; }
        long n = read_file(argv[3], g_text, CORPUS_MAX);
        if (n <= 0) { printf("cannot read %s\n", argv[3]); return 1; }
        build_dialog_data(g_text, n);
        setup_device();
        printf("finetune: %d tokens (loss-masked) | vocab %d | batch %d | params %ld\n\n",
               g_datalen, V, B, NP);
        train(steps, 0.0003f, 100);
        if (save_ckpt()) printf("\nsaved %s\n", CKPT);
    }
    else if (!strcmp(mode, "sample")) {
        int nchars = argc > 2 ? parse_int(argv[2], 400) : 400;
        if (!load_ckpt()) { printf("no checkpoint. run train first.\n"); return 1; }
        generate(N_SPECIAL, nchars, 0.8f);
    }
    else if (!strcmp(mode, "chat")) {
        if (!load_ckpt()) { printf("no checkpoint. run train first.\n"); return 1; }
        chat();
    }
    else {
        printf("tinyformer_gpu: tinyformer with a CUDA training path\n");
        printf("usage:\n");
        printf("  tinyformer_gpu train    [steps] [corpus.txt] [-b batch]   pretrain + save ckpt\n");
        printf("  tinyformer_gpu resume   [steps] corpus.txt   [-b batch]   continue pretraining ckpt\n");
        printf("  tinyformer_gpu finetune [steps] dialog.txt   [-b batch]   chat-tune the ckpt\n");
        printf("  tinyformer_gpu sample   [nchars]                          generate from ckpt\n");
        printf("  tinyformer_gpu chat                                       interactive REPL\n");
    }
    return 0;
}
