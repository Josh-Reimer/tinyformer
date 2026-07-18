// ============================================================================
// tinyformer.cpp — a GPT-style transformer language model in pure C++
//
// ZERO #include directives. No <cmath>, no <cstdio>, no <vector>, nothing.
//   - exp / log / sqrt / tanh implemented from scratch (bit tricks + series)
//   - xorshift RNG, Irwin-Hall approximate gaussian for init
//   - bump allocator over one static float pool (no malloc)
//   - full manual backprop: layernorm, causal multi-head attention, GELU MLP
//   - Adam optimizer, char-level tokenizer, autoregressive sampling
//   - checkpoint save/load, corpus loading from disk (raw syscall wrappers)
//   - chat: special turn tokens, loss-masked finetuning, KV-cached REPL
//
// The only symbols borrowed from the outside world are printf and the raw
// file-descriptor calls open/read/write/close, forward-declared by hand.
//
// build:  g++ -O3 -march=native -ffast-math -o tinyformer tinyformer.cpp
//
// usage:
//   ./tinyformer train    [steps] [corpus.txt]   pretrain, save tinyformer.ckpt
//   ./tinyformer finetune [steps] dialog.txt     chat-tune an existing ckpt
//   ./tinyformer sample   [nchars]               free-run generation from ckpt
//   ./tinyformer chat                            interactive REPL from ckpt
//
// dialog.txt format: plain text, non-empty lines alternate
//   user line
//   bot reply line
//   user line
//   bot reply line ...
// ============================================================================

extern "C" {
    int  printf(const char* fmt, ...);
    int  open(const char* path, int flags, ...);
    long read(int fd, void* buf, unsigned long n);
    long write(int fd, const void* buf, unsigned long n);
    int  close(int fd);
}

typedef unsigned int       u32;
typedef unsigned long long u64;

// ---------------------------------------------------------------------------
// math library, hand rolled
// ---------------------------------------------------------------------------
union FB { float f; u32 u; };

// e^x via range reduction: x = n*ln2 + r, e^x = 2^n * e^r (Taylor on r)
static float f_exp(float x) {
    if (x >  88.0f) x =  88.0f;
    if (x < -88.0f) return 0.0f;
    int n = (int)(x * 1.4426950408889634f + (x >= 0.0f ? 0.5f : -0.5f));
    float r = x - (float)n * 0.6931471805599453f;
    float p = 1.0f + r*(1.0f + r*(0.5f + r*(0.16666667f + r*(0.041666668f
                  + r*(0.008333334f + r*0.001388889f)))));
    FB b; b.u = (u32)(n + 127) << 23;   // constructs the float 2^n directly
    return p * b.f;
}

// sqrt via exponent-halving bit hack + 3 Newton iterations
static float f_sqrt(float x) {
    if (x <= 0.0f) return 0.0f;
    FB b; b.f = x;
    b.u = (b.u >> 1) + 0x1fbd1df5u;
    float y = b.f;
    y = 0.5f * (y + x / y);
    y = 0.5f * (y + x / y);
    y = 0.5f * (y + x / y);
    return y;
}

// ln(x): peel exponent from the float bits, atanh series on the mantissa
static float f_log(float x) {
    if (x <= 0.0f) return -87.0f;
    FB b; b.f = x;
    int e = (int)((b.u >> 23) & 255u) - 127;
    b.u = (b.u & 0x007fffffu) | 0x3f800000u;   // mantissa remapped to [1,2)
    float m  = b.f;
    float y  = (m - 1.0f) / (m + 1.0f);
    float y2 = y * y;
    float l  = 2.0f*y*(1.0f + y2*(0.33333334f + y2*(0.2f
                   + y2*(0.14285715f + y2*0.11111111f))));
    return l + (float)e * 0.6931471805599453f;
}

static float f_tanh(float x) {
    if (x >  9.0f) return  1.0f;
    if (x < -9.0f) return -1.0f;
    float e = f_exp(2.0f * x);
    return (e - 1.0f) / (e + 1.0f);
}

// ---------------------------------------------------------------------------
// tiny libc replacements
// ---------------------------------------------------------------------------
static long s_len(const char* s) { long n = 0; while (s[n]) n++; return n; }
static bool s_eq(const char* a, const char* b) {
    while (*a && *b) { if (*a != *b) return false; a++; b++; }
    return *a == *b;
}
static void wstr(const char* s) { write(1, s, (unsigned long)s_len(s)); }
static void wch(char c)         { write(1, &c, 1); }

// ---------------------------------------------------------------------------
// random numbers
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
// memory: one static pool, bump allocator, no free (we never need one)
// ---------------------------------------------------------------------------
enum { POOL_FLOATS = 12000000 };
static float g_pool[POOL_FLOATS];
static long  g_used = 0;

static float* falloc(long n) {
    if (g_used + n > POOL_FLOATS) { printf("FATAL: pool exhausted\n"); *(volatile int*)0 = 0; }
    float* p = g_pool + g_used;
    g_used += n;
    return p;                            // static storage => already zeroed
}
static void fzero(float* p, long n) { for (long i = 0; i < n; i++) p[i] = 0.0f; }

// ---------------------------------------------------------------------------
// model configuration
// ---------------------------------------------------------------------------
enum {
    T  = 128,       // context length
    C  = 128,       // embedding width
    L  = 4,         // transformer layers
    H  = 8,         // attention heads
    HS = C / H,     // head size
    F  = 4 * C,     // MLP hidden width
    B  = 4,         // sequences per optimizer step (grad accumulation)
    MAXV = 128,
    CORPUS_MAX = 524288
};

// special tokens: fixed vocab slots, never produced by plain text
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

// ---------------------------------------------------------------------------
// file helpers (raw fd syscalls; O_RDONLY=0, O_WRONLY|O_CREAT|O_TRUNC=577)
// ---------------------------------------------------------------------------
static long read_file(const char* path, char* buf, long maxn) {
    int fd = open(path, 0);
    if (fd < 0) return -1;
    long total = 0;
    while (total < maxn) {
        long r = read(fd, buf + total, (unsigned long)(maxn - total));
        if (r <= 0) break;
        total += r;
    }
    close(fd);
    return total;
}

// ---------------------------------------------------------------------------
// tokenizer: character level + special turn tokens
// ---------------------------------------------------------------------------
static int  g_stoi[256];
static char g_itos[MAXV];
static int  V = 0;

static int* g_data = 0;                  // tokenized training stream
static int* g_mask = 0;                  // 1 = compute loss on this target
static int  g_datalen = 0;

static char g_text[CORPUS_MAX];          // file loading scratch

static void vocab_reset() {
    for (int i = 0; i < 256; i++) g_stoi[i] = -1;
    V = 0;
    // specials occupy vocab ids 0..2; give them control-char byte identities
    for (int s = 0; s < N_SPECIAL; s++) { g_itos[V] = (char)(s + 1); g_stoi[s + 1] = V; V++; }
}

static int char_to_tok(unsigned char ch) {          // -1 if unknown
    if (ch >= 'A' && ch <= 'Z') ch = (unsigned char)(ch - 'A' + 'a');
    return g_stoi[ch];
}

static void vocab_add_text(const char* text, long n) {
    for (long i = 0; i < n; i++) {
        unsigned char ch = (unsigned char)text[i];
        if (ch >= 'A' && ch <= 'Z') ch = (unsigned char)(ch - 'A' + 'a');
        if (ch < 32 && ch != '\n') continue;        // control chars stay out
        if (ch == '\n') ch = ' ';                   // newlines become spaces
        if (g_stoi[ch] < 0) {
            if (V >= MAXV) continue;
            g_itos[V] = (char)ch;
            g_stoi[ch] = V;
            V++;
        }
    }
}

static void data_begin() {
    if (!g_data) {
        g_data = (int*)(void*)falloc(CORPUS_MAX);
        g_mask = (int*)(void*)falloc(CORPUS_MAX);
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
// tiny finetune files: repeat the stream until there is room to sample windows
static void data_ensure(long minlen) {
    long orig = g_datalen;
    if (orig <= 0) return;
    while (g_datalen < minlen && g_datalen + orig <= CORPUS_MAX)
        for (long i = 0; i < orig; i++) data_push(g_data[i], g_mask[i]);
}

// ---------------------------------------------------------------------------
// parameters
// ---------------------------------------------------------------------------
struct Params {
    float *wte, *wpe;                    // token + position embeddings
    float *ln1w[L], *ln1b[L];
    float *qkvw[L], *qkvb[L];            // [3C, C], [3C]
    float *apw[L],  *apb[L];             // attention output proj [C, C], [C]
    float *ln2w[L], *ln2b[L];
    float *f1w[L],  *f1b[L];             // MLP up   [F, C], [F]
    float *f2w[L],  *f2b[L];             // MLP down [C, F], [C]
    float *lnfw, *lnfb;
    float *hw, *hb;                      // lm head [V, C], [V]
};

static long alloc_params(Params& P) {
    long start = g_used;
    P.wte = falloc((long)V * C);
    P.wpe = falloc((long)T * C);
    for (int l = 0; l < L; l++) {
        P.ln1w[l] = falloc(C);           P.ln1b[l] = falloc(C);
        P.qkvw[l] = falloc(3L * C * C);  P.qkvb[l] = falloc(3 * C);
        P.apw[l]  = falloc((long)C * C); P.apb[l]  = falloc(C);
        P.ln2w[l] = falloc(C);           P.ln2b[l] = falloc(C);
        P.f1w[l]  = falloc((long)F * C); P.f1b[l]  = falloc(F);
        P.f2w[l]  = falloc((long)C * F); P.f2b[l]  = falloc(C);
    }
    P.lnfw = falloc(C); P.lnfb = falloc(C);
    P.hw = falloc((long)V * C); P.hb = falloc(V);
    return g_used - start;               // total parameter count
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

// ---------------------------------------------------------------------------
// activations (single sequence of T tokens; reused across the batch)
// ---------------------------------------------------------------------------
struct Acts {
    float *x0;                                    // [T, C] emb sum
    float *ln1o[L], *ln1m[L], *ln1r[L];           // [T,C], [T], [T]
    float *qkv[L];                                // [T, 3C]
    float *att[L];                                // [H, T, T] softmax probs
    float *atty[L];                               // [T, C] heads concatenated
    float *attp[L];                               // [T, C] after out-proj
    float *res1[L];                               // [T, C]
    float *ln2o[L], *ln2m[L], *ln2r[L];
    float *fch[L], *fchg[L];                      // [T, F] pre/post GELU
    float *fcp[L];                                // [T, C]
    float *res2[L];                               // [T, C]
    float *lnfo, *lnfm, *lnfr;
    float *logits, *probs;                        // [T, V]
};

static long alloc_acts(Acts& A) {
    long start = g_used;
    A.x0 = falloc((long)T * C);
    for (int l = 0; l < L; l++) {
        A.ln1o[l] = falloc((long)T*C); A.ln1m[l] = falloc(T); A.ln1r[l] = falloc(T);
        A.qkv[l]  = falloc(3L*T*C);
        A.att[l]  = falloc((long)H*T*T);
        A.atty[l] = falloc((long)T*C);
        A.attp[l] = falloc((long)T*C);
        A.res1[l] = falloc((long)T*C);
        A.ln2o[l] = falloc((long)T*C); A.ln2m[l] = falloc(T); A.ln2r[l] = falloc(T);
        A.fch[l]  = falloc((long)T*F); A.fchg[l] = falloc((long)T*F);
        A.fcp[l]  = falloc((long)T*C);
        A.res2[l] = falloc((long)T*C);
    }
    A.lnfo = falloc((long)T*C); A.lnfm = falloc(T); A.lnfr = falloc(T);
    A.logits = falloc((long)T*V); A.probs = falloc((long)T*V);
    return g_used - start;
}

// ---------------------------------------------------------------------------
// layers: forward
// ---------------------------------------------------------------------------
// out[t,o] = in[t,:] . w[o,:] + b[o]     w layout: [Cout, Cin]
static void matmul_fwd(float* out, const float* in, const float* w,
                       const float* b, int Cin, int Cout) {
    for (int t = 0; t < T; t++) {
        const float* ir = in + (long)t * Cin;
        float* orow = out + (long)t * Cout;
        for (int o = 0; o < Cout; o++) {
            const float* wr = w + (long)o * Cin;
            float s = b ? b[o] : 0.0f;
            for (int i = 0; i < Cin; i++) s += ir[i] * wr[i];
            orow[o] = s;
        }
    }
}

static void ln_fwd(float* out, float* mean, float* rstd, const float* in,
                   const float* w, const float* b) {
    for (int t = 0; t < T; t++) {
        const float* x = in + (long)t * C;
        float m = 0.0f;
        for (int i = 0; i < C; i++) m += x[i];
        m /= C;
        float v = 0.0f;
        for (int i = 0; i < C; i++) { float d = x[i] - m; v += d * d; }
        v /= C;
        float r = 1.0f / f_sqrt(v + 1e-5f);
        mean[t] = m; rstd[t] = r;
        float* o = out + (long)t * C;
        for (int i = 0; i < C; i++) o[i] = (x[i] - m) * r * w[i] + b[i];
    }
}

// causal multi-head self-attention
static void attn_fwd(float* atty, float* att, const float* qkv) {
    const float scale = 1.0f / f_sqrt((float)HS);
    for (int h = 0; h < H; h++) {
        for (int t = 0; t < T; t++) {
            const float* q = qkv + 3L*C*t + (long)h*HS;
            float* arow = att + (long)h*T*T + (long)t*T;
            float mx = -1e30f;
            for (int t2 = 0; t2 <= t; t2++) {
                const float* k = qkv + 3L*C*t2 + C + (long)h*HS;
                float s = 0.0f;
                for (int i = 0; i < HS; i++) s += q[i] * k[i];
                s *= scale;
                arow[t2] = s;
                if (s > mx) mx = s;
            }
            float sum = 0.0f;
            for (int t2 = 0; t2 <= t; t2++) { float e = f_exp(arow[t2] - mx); arow[t2] = e; sum += e; }
            float inv = 1.0f / sum;
            for (int t2 = 0;     t2 <= t; t2++) arow[t2] *= inv;
            for (int t2 = t + 1; t2 <  T; t2++) arow[t2]  = 0.0f;   // causal mask
            float* o = atty + (long)t*C + (long)h*HS;
            for (int i = 0; i < HS; i++) o[i] = 0.0f;
            for (int t2 = 0; t2 <= t; t2++) {
                const float* v = qkv + 3L*C*t2 + 2L*C + (long)h*HS;
                float a = arow[t2];
                for (int i = 0; i < HS; i++) o[i] += a * v[i];
            }
        }
    }
}

static float gelu_f(float x) {
    float u = 0.7978845608f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + f_tanh(u));
}
static float gelu_df(float x) {
    float u  = 0.7978845608f * (x + 0.044715f * x * x * x);
    float th = f_tanh(u);
    float s2 = 1.0f - th * th;
    return 0.5f*(1.0f + th) + 0.5f*x*s2*0.7978845608f*(1.0f + 0.134145f*x*x);
}

// full forward over T tokens. tgt may be null (inference).
// msk aligns with tgt: msk[t]==1 means the loss counts target t.
static float forward(Params& P, Acts& A, const int* tok,
                     const int* tgt, const int* msk) {
    for (int t = 0; t < T; t++)
        for (int i = 0; i < C; i++)
            A.x0[(long)t*C + i] = P.wte[(long)tok[t]*C + i] + P.wpe[(long)t*C + i];

    const float* x = A.x0;
    for (int l = 0; l < L; l++) {
        ln_fwd(A.ln1o[l], A.ln1m[l], A.ln1r[l], x, P.ln1w[l], P.ln1b[l]);
        matmul_fwd(A.qkv[l], A.ln1o[l], P.qkvw[l], P.qkvb[l], C, 3*C);
        attn_fwd(A.atty[l], A.att[l], A.qkv[l]);
        matmul_fwd(A.attp[l], A.atty[l], P.apw[l], P.apb[l], C, C);
        for (long i = 0; i < (long)T*C; i++) A.res1[l][i] = x[i] + A.attp[l][i];

        ln_fwd(A.ln2o[l], A.ln2m[l], A.ln2r[l], A.res1[l], P.ln2w[l], P.ln2b[l]);
        matmul_fwd(A.fch[l], A.ln2o[l], P.f1w[l], P.f1b[l], C, F);
        for (long i = 0; i < (long)T*F; i++) A.fchg[l][i] = gelu_f(A.fch[l][i]);
        matmul_fwd(A.fcp[l], A.fchg[l], P.f2w[l], P.f2b[l], F, C);
        for (long i = 0; i < (long)T*C; i++) A.res2[l][i] = A.res1[l][i] + A.fcp[l][i];
        x = A.res2[l];
    }
    ln_fwd(A.lnfo, A.lnfm, A.lnfr, x, P.lnfw, P.lnfb);
    matmul_fwd(A.logits, A.lnfo, P.hw, P.hb, C, V);

    float loss = 0.0f;
    int cnt = 0;
    for (int t = 0; t < T; t++) {
        const float* lg = A.logits + (long)t*V;
        float* pr = A.probs + (long)t*V;
        float mx = -1e30f;
        for (int v = 0; v < V; v++) if (lg[v] > mx) mx = lg[v];
        float sum = 0.0f;
        for (int v = 0; v < V; v++) { float e = f_exp(lg[v] - mx); pr[v] = e; sum += e; }
        float inv = 1.0f / sum;
        for (int v = 0; v < V; v++) pr[v] *= inv;
        if (tgt && (!msk || msk[t])) { loss += -f_log(pr[tgt[t]]); cnt++; }
    }
    return cnt ? loss / cnt : 0.0f;
}

// ---------------------------------------------------------------------------
// layers: backward
// ---------------------------------------------------------------------------
static void matmul_bwd(float* din, float* dw, float* db, const float* dout,
                       const float* in, const float* w, int Cin, int Cout) {
    for (int t = 0; t < T; t++) {
        const float* dor = dout + (long)t * Cout;
        const float* ir  = in   + (long)t * Cin;
        if (din) {
            float* dir = din + (long)t * Cin;
            for (int o = 0; o < Cout; o++) {
                const float* wr = w + (long)o * Cin;
                float d = dor[o];
                for (int i = 0; i < Cin; i++) dir[i] += d * wr[i];
            }
        }
        for (int o = 0; o < Cout; o++) {
            float d = dor[o];
            if (db) db[o] += d;
            float* dwr = dw + (long)o * Cin;
            for (int i = 0; i < Cin; i++) dwr[i] += d * ir[i];
        }
    }
}

static void ln_bwd(float* din, float* dw, float* db, const float* dout,
                   const float* in, const float* w,
                   const float* mean, const float* rstd) {
    for (int t = 0; t < T; t++) {
        const float* x  = in   + (long)t * C;
        const float* dy = dout + (long)t * C;
        float m = mean[t], r = rstd[t];
        float dnm = 0.0f, dnnm = 0.0f;
        for (int i = 0; i < C; i++) {
            float norm  = (x[i] - m) * r;
            float dnorm = w[i] * dy[i];
            dnm  += dnorm;
            dnnm += dnorm * norm;
        }
        dnm /= C; dnnm /= C;
        float* dx = din + (long)t * C;
        for (int i = 0; i < C; i++) {
            float norm  = (x[i] - m) * r;
            float dnorm = w[i] * dy[i];
            dw[i] += dy[i] * norm;
            db[i] += dy[i];
            dx[i] += (dnorm - dnm - norm * dnnm) * r;
        }
    }
}

static void attn_bwd(float* dqkv, const float* datty,
                     const float* att, const float* qkv) {
    const float scale = 1.0f / f_sqrt((float)HS);
    float datt[T], dpre[T];
    for (int h = 0; h < H; h++) {
        for (int t = 0; t < T; t++) {
            const float* arow = att   + (long)h*T*T + (long)t*T;
            const float* do_  = datty + (long)t*C   + (long)h*HS;
            // through the value-weighted sum: d(att row), d(v)
            for (int t2 = 0; t2 <= t; t2++) {
                const float* v  = qkv  + 3L*C*t2 + 2L*C + (long)h*HS;
                float*       dv = dqkv + 3L*C*t2 + 2L*C + (long)h*HS;
                float a = arow[t2], d = 0.0f;
                for (int i = 0; i < HS; i++) { d += do_[i]*v[i]; dv[i] += a*do_[i]; }
                datt[t2] = d;
            }
            // through the softmax
            float dsum = 0.0f;
            for (int t2 = 0; t2 <= t; t2++) dsum += arow[t2] * datt[t2];
            for (int t2 = 0; t2 <= t; t2++) dpre[t2] = arow[t2] * (datt[t2] - dsum);
            // through the scaled dot product: d(q), d(k)
            const float* q  = qkv  + 3L*C*t + (long)h*HS;
            float*       dq = dqkv + 3L*C*t + (long)h*HS;
            for (int t2 = 0; t2 <= t; t2++) {
                const float* k  = qkv  + 3L*C*t2 + C + (long)h*HS;
                float*       dk = dqkv + 3L*C*t2 + C + (long)h*HS;
                float dp = dpre[t2] * scale;
                for (int i = 0; i < HS; i++) { dq[i] += dp*k[i]; dk[i] += dp*q[i]; }
            }
        }
    }
}

// grad buffers: params mirror + activation mirror (layout identical)
static Params G;                          // parameter grads
static Acts   D;                          // activation grads
static long   NP = 0;                     // param count
static float *p_base, *g_base, *m_base, *v_base;   // adam state
static long   act_count = 0;
static float* dact_base;

static void backward(Params& P, Acts& A, const int* tok,
                     const int* tgt, const int* msk) {
    fzero(dact_base, act_count);          // fresh activation grads per sequence

    int cnt = 0;
    for (int t = 0; t < T; t++) if (!msk || msk[t]) cnt++;
    if (cnt == 0) return;

    // d(logits): softmax + cross-entropy, mean over counted targets and batch
    const float sc = 1.0f / ((float)cnt * (float)B);
    for (int t = 0; t < T; t++) {
        if (msk && !msk[t]) continue;     // masked-out rows stay zero
        for (int v = 0; v < V; v++)
            D.logits[(long)t*V + v] = (A.probs[(long)t*V + v]
                                       - (v == tgt[t] ? 1.0f : 0.0f)) * sc;
    }

    matmul_bwd(D.lnfo, G.hw, G.hb, D.logits, A.lnfo, P.hw, C, V);
    ln_bwd(D.res2[L-1], G.lnfw, G.lnfb, D.lnfo, A.res2[L-1], P.lnfw, A.lnfm, A.lnfr);

    for (int l = L - 1; l >= 0; l--) {
        const float* x  = (l == 0) ? A.x0 : A.res2[l-1];
        float*       dx = (l == 0) ? D.x0 : D.res2[l-1];

        // res2 = res1 + fcp
        for (long i = 0; i < (long)T*C; i++) { D.fcp[l][i] = D.res2[l][i]; D.res1[l][i] += D.res2[l][i]; }
        matmul_bwd(D.fchg[l], G.f2w[l], G.f2b[l], D.fcp[l], A.fchg[l], P.f2w[l], F, C);
        for (long i = 0; i < (long)T*F; i++) D.fch[l][i] = D.fchg[l][i] * gelu_df(A.fch[l][i]);
        matmul_bwd(D.ln2o[l], G.f1w[l], G.f1b[l], D.fch[l], A.ln2o[l], P.f1w[l], C, F);
        ln_bwd(D.res1[l], G.ln2w[l], G.ln2b[l], D.ln2o[l], A.res1[l], P.ln2w[l], A.ln2m[l], A.ln2r[l]);

        // res1 = x + attp
        for (long i = 0; i < (long)T*C; i++) { D.attp[l][i] = D.res1[l][i]; dx[i] += D.res1[l][i]; }
        matmul_bwd(D.atty[l], G.apw[l], G.apb[l], D.attp[l], A.atty[l], P.apw[l], C, C);
        attn_bwd(D.qkv[l], D.atty[l], A.att[l], A.qkv[l]);
        matmul_bwd(D.ln1o[l], G.qkvw[l], G.qkvb[l], D.qkv[l], A.ln1o[l], P.qkvw[l], C, 3*C);
        ln_bwd(dx, G.ln1w[l], G.ln1b[l], D.ln1o[l], x, P.ln1w[l], A.ln1m[l], A.ln1r[l]);
    }

    // embeddings
    for (int t = 0; t < T; t++)
        for (int i = 0; i < C; i++) {
            float d = D.x0[(long)t*C + i];
            G.wte[(long)tok[t]*C + i] += d;
            G.wpe[(long)t*C + i]      += d;
        }
}

// ---------------------------------------------------------------------------
// Adam
// ---------------------------------------------------------------------------
static float g_b1t = 1.0f, g_b2t = 1.0f;

static void adam_step(float lr) {
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    g_b1t *= b1; g_b2t *= b2;
    float c1 = 1.0f / (1.0f - g_b1t);
    float c2 = 1.0f / (1.0f - g_b2t);
    for (long i = 0; i < NP; i++) {
        float g = g_base[i];
        m_base[i] = b1*m_base[i] + (1.0f - b1)*g;
        v_base[i] = b2*v_base[i] + (1.0f - b2)*g*g;
        float mh = m_base[i] * c1;
        float vh = v_base[i] * c2;
        p_base[i] -= lr * mh / (f_sqrt(vh) + eps);
    }
}

// ---------------------------------------------------------------------------
// checkpoint: [magic][V T C L H][itos bytes][params]
// ---------------------------------------------------------------------------
static const char* CKPT = "tinyformer.ckpt";

static bool save_ckpt(Params& P) {
    int fd = open(CKPT, 577, 420);       // O_WRONLY|O_CREAT|O_TRUNC, 0644
    if (fd < 0) return false;
    int hdr[6] = { 0x54465632, V, T, C, L, H };
    write(fd, hdr, sizeof(hdr));
    write(fd, g_itos, (unsigned long)V);
    write(fd, P.wte, (unsigned long)NP * sizeof(float));   // contiguous block
    close(fd);
    return true;
}

static bool load_ckpt_header() {          // fills V + vocab; params come later
    int fd = open(CKPT, 0);
    if (fd < 0) return false;
    int hdr[6];
    if (read(fd, hdr, sizeof(hdr)) != sizeof(hdr)) { close(fd); return false; }
    if (hdr[0] != 0x54465632 || hdr[2] != T || hdr[3] != C
        || hdr[4] != L || hdr[5] != H || hdr[1] > MAXV) {
        printf("checkpoint does not match compiled config\n");
        close(fd); return false;
    }
    for (int i = 0; i < 256; i++) g_stoi[i] = -1;
    V = hdr[1];
    read(fd, g_itos, (unsigned long)V);
    for (int i = 0; i < V; i++) g_stoi[(unsigned char)g_itos[i]] = i;
    close(fd);
    return true;
}

static bool load_ckpt_params(Params& P) {
    int fd = open(CKPT, 0);
    if (fd < 0) return false;
    int hdr[6];
    read(fd, hdr, sizeof(hdr));
    read(fd, g_itos, (unsigned long)V);
    long want = NP * (long)sizeof(float), got = 0;
    char* dst = (char*)(void*)P.wte;
    while (got < want) {
        long r = read(fd, dst + got, (unsigned long)(want - got));
        if (r <= 0) break;
        got += r;
    }
    close(fd);
    return got == want;
}

// ---------------------------------------------------------------------------
// KV-cached incremental inference (used by chat)
// ---------------------------------------------------------------------------
static float kcache[L][(long)T * C];
static float vcache[L][(long)T * C];

static void ln_row(float* out, const float* x, const float* w, const float* b) {
    float m = 0.0f;
    for (int i = 0; i < C; i++) m += x[i];
    m /= C;
    float v = 0.0f;
    for (int i = 0; i < C; i++) { float d = x[i] - m; v += d * d; }
    float r = 1.0f / f_sqrt(v / C + 1e-5f);
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

    const float scale = 1.0f / f_sqrt((float)HS);
    for (int l = 0; l < L; l++) {
        ln_row(l1, xb, P.ln1w[l], P.ln1b[l]);
        mat_row(qkvr, l1, P.qkvw[l], P.qkvb[l], C, 3*C);
        for (int i = 0; i < C; i++) {                 // cache this position's k,v
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
            for (int t2 = 0; t2 <= pos; t2++) { float e = f_exp(sc_[t2] - mx); sc_[t2] = e; sum += e; }
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
    for (int v = 0; v < V; v++) { pr[v] = f_exp((lg[v] - mx) / temp); sum += pr[v]; }
    float r = rnd_uniform() * sum, acc = 0.0f;
    for (int v = 0; v < V; v++) { acc += pr[v]; if (r <= acc) return v; }
    return V - 1;
}

// ---------------------------------------------------------------------------
// training loop (shared by pretrain and finetune)
// ---------------------------------------------------------------------------
static void train(Params& P, Acts& A, int steps, float lr, int log_every) {
    int tok[T], tgt[T], msk[T];
    for (int step = 1; step <= steps; step++) {
        fzero(g_base, NP);
        float loss = 0.0f;
        for (int b = 0; b < B; b++) {
            int pos = (int)(rnd_u32() % (u32)(g_datalen - T - 1));
            int have = 0;
            for (int tries = 0; tries < 20 && !have; tries++) {
                for (int t = 0; t < T; t++) have |= g_mask[pos + t + 1];
                if (!have) pos = (int)(rnd_u32() % (u32)(g_datalen - T - 1));
            }
            for (int t = 0; t < T; t++) {
                tok[t] = g_data[pos + t];
                tgt[t] = g_data[pos + t + 1];
                msk[t] = g_mask[pos + t + 1];
            }
            loss += forward(P, A, tok, tgt, msk);
            backward(P, A, tok, tgt, msk);
        }
        loss /= B;
        adam_step(lr);
        if (step == 1 || step % log_every == 0)
            printf("step %5d | loss %.4f\n", step, (double)loss);
    }
}

// free-running generation (batch forward path; fine for one-shot sampling)
static void generate(Params& P, Acts& A, int seed_tok, int nchars, float temp) {
    int hist[4096];
    int len = 0;
    hist[len++] = seed_tok;
    if (seed_tok >= N_SPECIAL) printf("%c", g_itos[seed_tok]);
    int tok[T];
    for (int n = 0; n < nchars; n++) {
        int win = len < T ? len : T;
        int off = len - win;
        for (int t = 0; t < win; t++) tok[t] = hist[off + t];
        for (int t = win; t < T; t++) tok[t] = seed_tok;  // causal mask: harmless pad
        forward(P, A, tok, 0, 0);
        int pick = sample_logits(A.logits + (long)(win - 1) * V, temp);
        if (pick >= N_SPECIAL) printf("%c", g_itos[pick]);
        if (len < 4096) hist[len++] = pick;
        else { for (int i = 1; i < 4096; i++) hist[i-1] = hist[i]; hist[4095] = pick; }
    }
    printf("\n");
}

// ---------------------------------------------------------------------------
// chat REPL
// ---------------------------------------------------------------------------
static long read_line(char* buf, long maxn) {       // returns -1 on EOF
    long n = 0;
    while (n < maxn - 1) {
        char c;
        long r = read(0, &c, 1);
        if (r <= 0) return n > 0 ? n : -1;
        if (c == '\n') break;
        buf[n++] = c;
    }
    buf[n] = 0;
    return n;
}

enum { CHAT_HIST = 8192, CHAT_KEEP = 64, CHAT_MAXGEN = 200 };

static void chat(Params& P) {
    static int hist[CHAT_HIST];
    int len = 0;
    char line[512];
    wstr("tinyformer chat. empty line or ctrl-d to quit.\n");
    for (;;) {
        wstr("you: ");
        long n = read_line(line, sizeof(line));
        if (n <= 0 || s_eq(line, "quit") || s_eq(line, "exit")) break;
        for (long i = 0; i < n; i++) g_rng ^= (u64)(unsigned char)line[i] << ((i & 7) * 8);

        // dialog training data is single-turn, so start each turn fresh:
        // the full context minus the prompt is available for the reply
        len = 0;
        if (len < CHAT_HIST - 600) {
            hist[len++] = TOK_USER;
            for (long i = 0; i < n; i++) {
                int t = char_to_tok((unsigned char)line[i]);
                if (t >= 0) hist[len++] = t;
            }
            hist[len++] = TOK_END;
            hist[len++] = TOK_BOT;
        }

        // prefill the KV cache with the prompt, leaving room to respond
        int win = len < CHAT_KEEP ? len : CHAT_KEEP;
        int off = len - win;
        const float* lg = 0;
        for (int t = 0; t < win; t++) lg = forward_one(P, hist[off + t], t);

        wstr("bot: ");
        int pos = win, made = 0;
        while (pos < T && made < CHAT_MAXGEN) {
            int pick = sample_logits(lg, 0.2f);   // low temp: tiny models ramble
            if (pick == TOK_END) break;
            if (pick >= N_SPECIAL) wch(g_itos[pick]);
            if (len < CHAT_HIST) hist[len++] = pick;
            lg = forward_one(P, pick, pos++);
            made++;
        }
        if (len < CHAT_HIST) hist[len++] = TOK_END;
        wch('\n');
    }
    wstr("bye\n");
}

// ---------------------------------------------------------------------------
// dialog file -> masked token stream
//   non-empty lines alternate: user, bot, user, bot ...
//   loss is computed only on bot text and the bot's end-of-turn token
// ---------------------------------------------------------------------------
static void build_dialog_data(const char* text, long n) {
    data_begin();
    int role = 0;                        // 0 = expecting user line
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
                data_push(TOK_BOT, 0);   // we insert <bot> by hand in the REPL
                data_push_text(text + i, linelen, 1);
                data_push(TOK_END, 1);
                role = 0;
            }
        }
        i = j + 1;
    }
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

static void setup_model(Params& P, Acts& A) {       // call after V is known
    NP = alloc_params(P);  p_base = P.wte;
    alloc_params(G);       g_base = G.wte;
    m_base = falloc(NP);
    v_base = falloc(NP);
    alloc_acts(A);
    act_count = alloc_acts(D);  dact_base = D.x0;
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "help";
    Params P; Acts A;

    if (s_eq(mode, "train")) {
        int steps = argc > 2 ? parse_int(argv[2], 2000) : 2000;
        const char* src = g_builtin;
        long n = s_len(g_builtin);
        if (argc > 3) {
            n = read_file(argv[3], g_text, CORPUS_MAX);
            if (n <= 0) { printf("cannot read %s\n", argv[3]); return 1; }
            src = g_text;
            printf("loaded %ld chars from %s\n", n, argv[3]);
        }
        vocab_reset();
        vocab_add_text(src, n);
        data_begin();
        data_push_text(src, n, 1);       // pretraining: loss on everything
        data_ensure(8L * T);
        setup_model(P, A);
        init_params(P);
        printf("pretrain: %d tokens | vocab %d | context %d | dim %d | layers %d | params %ld\n\n",
               g_datalen, V, T, C, L, NP);
        train(P, A, steps, 0.001f, 250);
        if (save_ckpt(P)) printf("\nsaved %s\n", CKPT);
        printf("\n--- sample ---\n");
        generate(P, A, N_SPECIAL, 300, 0.8f);
    }
    else if (s_eq(mode, "finetune")) {
        if (argc < 4) { printf("usage: tinyformer finetune [steps] dialog.txt\n"); return 1; }
        int steps = parse_int(argv[2], 1500);
        if (!load_ckpt_header()) { printf("no checkpoint. run train first.\n"); return 1; }
        setup_model(P, A);
        if (!load_ckpt_params(P)) { printf("checkpoint read failed\n"); return 1; }
        long n = read_file(argv[3], g_text, CORPUS_MAX);
        if (n <= 0) { printf("cannot read %s\n", argv[3]); return 1; }
        build_dialog_data(g_text, n);
        printf("finetune: %d tokens (loss-masked) | vocab %d | params %ld\n\n",
               g_datalen, V, NP);
        train(P, A, steps, 0.0003f, 100);
        if (save_ckpt(P)) printf("\nsaved %s\n", CKPT);
    }
    else if (s_eq(mode, "sample")) {
        int nchars = argc > 2 ? parse_int(argv[2], 400) : 400;
        if (!load_ckpt_header()) { printf("no checkpoint. run train first.\n"); return 1; }
        setup_model(P, A);
        if (!load_ckpt_params(P)) { printf("checkpoint read failed\n"); return 1; }
        generate(P, A, N_SPECIAL, nchars, 0.8f);
    }
    else if (s_eq(mode, "chat")) {
        if (!load_ckpt_header()) { printf("no checkpoint. run train first.\n"); return 1; }
        setup_model(P, A);
        if (!load_ckpt_params(P)) { printf("checkpoint read failed\n"); return 1; }
        chat(P);
    }
    else {
        printf("tinyformer: pure C++ transformer, zero includes\n");
        printf("usage:\n");
        printf("  tinyformer train    [steps] [corpus.txt]   pretrain + save ckpt\n");
        printf("  tinyformer finetune [steps] dialog.txt     chat-tune the ckpt\n");
        printf("  tinyformer sample   [nchars]               generate from ckpt\n");
        printf("  tinyformer chat                            interactive REPL\n");
    }
    return 0;
}
