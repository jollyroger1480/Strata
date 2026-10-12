// metal/src/engine/window.cpp - the decode window on Metal: Verifier::init/run/commit's Metal side (the one backend
// hook, src/core/verify.cpp under STRATA_GPU_METAL).  Everything above it - the decode loop, MTP control, the serve
// protocol, the PLE table reads, the file tier - is the shared host code.
//
// The window is mac-kern's chain (metal/kernels, tests/token_chain.cpp) on the model GGUF itself: shard 1 is mmapped
// and wrapped no-copy as one MTLBuffer, every weight is read where it lies (unified memory: no upload, no expert
// cache), and one window is encoded as a dependency graph in one command buffer.  Its own state lives here (the GDN
// recurrence and conv history, the int8 K/V and the indexer per QSA layer, the PLE history): the engine's
// SessionState buffers are not used by it, except the PLE token window (ss.ple_prev) and table.
//
// First version (MAC-PLAN Phase 3 start): windows of one token (no MTP drafts yet: --mtp-max-t 1), greedy.  The PLE
// block's projections run on the GPU; its norms, gate and conv (10240 values) on the CPU between two command buffers
// (layer 0 | layers 1..47 + head).
#include "gguf_lite.h"
#include "mtl_rt.h"
#include "ref_quant.h"

#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/kernels/ngram.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <vector>

namespace strata::metal_engine {

namespace {

constexpr int N = 2560, HC = 4, LR = 320, NFF = 640, TOPK = 10, NEXP = 512, MAXT = 8, NVOCAB_MAX = 248320;
constexpr int P_MAXE = 128, P_GEXP = 16, P_GSTART = P_GEXP + P_MAXE, P_ETOK = P_GSTART + P_MAXE + 4,
              P_EW = P_ETOK + P_MAXE, P_TOKENT = P_EW + P_MAXE, P_INTS = P_TOKENT + P_MAXE, P_IGU = 4, P_IDN = 8;
struct GemvArgs { uint32_t n_in, n_out, n_tok, row_bytes, x_stride, y_stride; };
struct GrArgs { uint32_t n_embd, hc, hc_lr, n_tok, apply; float eps; };
struct MoeArgs {
    uint64_t gate_off, up_off, down_off, gu_stride, d_stride;
    uint32_t gu_row, d_row, n_embd, n_ff, n_tok, topk, n_expert, gx_gu, gx_dn, pad;
};
struct GdnArgs { uint32_t h_k, h_v, channels, commit; float eps; uint32_t plane; };
struct QsaArgs { uint32_t pos0, pos_base, max_cells, n_tok; float eps; };
struct GlueArgs { uint32_t n, n_tok, n_embd, topk; };
struct ArgArgs { uint32_t n_vocab, parts; };

int xkind(int t) {
    switch (t) {
        case ref::Q2_0: return 1;
        case ref::Q4_K: case ref::Q5_K: return 2;
        case ref::Q6_K: case ref::Q3_K: return 3;
        default: return 0;
    }
}

// a prepared activation: 4 planes of MAXT rows of n, then the per-16 sums
struct Act {
    mtl::Buffer b;
    int n = 0;
    size_t plane_bytes() const { return (size_t) 4 * MAXT * n; }
};

double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

}  // namespace

class Window {
public:
    bool init(const strata::core::ModelGeometry& g, strata::core::SessionState& ss, const std::string& shard1, int max_t,
              int64_t max_cells, std::string& err);
    bool run(int T, const int32_t* tokens, int64_t pos0, bool head, int32_t* out, std::string& err);
    bool commit(int n_keep, std::string& err);
    bool finish(int Tn, const int32_t* tokens, int64_t pos0, bool head, int32_t* out, double t0);
    bool copy_logits(int t, float* host) const;
    int64_t vocab() const { return n_vocab_; }
    bool ready() const { return cx_ != nullptr; }
    void set_max_t(int t) { max_t_ = std::min(t, MAXT); }
    // ---- the MTP draft layer (mtp.hpp's math on mac-kern's kernels; the MTP GGUF from tools/mtp_pack.py)
    bool mtp_load(const std::string& gguf, int max_t, std::string& err);
    bool mtp_prefill(const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err);
    bool mtp_draft(int T, const int32_t* tokens, int64_t p, int a, int max_steps, int32_t* drafts, float* probs,
                   float min_p, int* n_drafts, std::string& err);
    bool mtp_loaded() const { return mtp_on_; }

private:
    const gguf::Tensor& T(const std::string& name) const {
        const gguf::Tensor* t = f_.find(name);
        if (!t) { std::fprintf(stderr, "strata metal: the model has no tensor %s\n", name.c_str()); std::exit(2); }
        return *t;
    }
    const mtl::Pipeline* P(int lib, const std::string& fn, const mtl::Constants& c, const std::string& key) {
        auto it = pcache_.find(key);
        if (it == pcache_.end()) it = pcache_.emplace(key, cx_->pipeline(lib, fn, c)).first;
        return &it->second;
    }
    void push(std::vector<mtl::Dispatch>& ds, mtl::Dispatch d) {
        d.barrier = !nb_;
        nb_ = false;
        ds.push_back(std::move(d));
    }
    void gemv(std::vector<mtl::Dispatch>& ds, const gguf::Tensor& t, const Act& x, const mtl::Buffer& y, size_t y_stride,
              int ntok, size_t y_off = 0, const mtl::Buffer* wb = nullptr);
    void xprep(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& x, const Act& out, int n);
    void gr_read(std::vector<mtl::Dispatch>& ds, const std::string& pre, bool apply, const Act& out, bool final_, int ntok);
    void layer(std::vector<mtl::Dispatch>& ds, int l, int ntok, int64_t pos0);
    void ple_cpu(int T);
    // the MTP layer: the input branches + attention HC read + K/V append for `n` rows (srcR rows, host tokens) at
    // cells cell0..; with `full` (n == 1) the rest of the layer too and the draft head into mlm_/tokout_.  The
    // embedding rows go to memb_ rows erow..; tokens == nullptr: memb_ row erow already holds the embedding (put
    // there by mtp_embed on the GPU).  `slot`: the dense attention's cell list (one per draft step of a round).
    void mtp_encode(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& srcR, size_t src_off, int n, const int32_t* tokens,
                    int64_t cell0, bool full, int erow = 0, int slot = 0);
    // a draft round's step j on the GPU: the pick and its probability (dout_/dprob_ [j]), and with `embed` the picked
    // token's embedding into memb_ row 0 (the next step's input)
    void mtp_pick(std::vector<mtl::Dispatch>& ds, int j, bool embed);
    void gr_read_m(std::vector<mtl::Dispatch>& ds, const std::string& pre, mtl::Buffer& Rb, const mtl::Buffer& bo,
                   const mtl::Buffer& inj_in, const mtl::Buffer& inj_out, bool apply, const Act& out, bool final_, int ntok);
    const gguf::Tensor& TM(const std::string& name) const {
        const gguf::Tensor* x = fm_.find(name);
        if (!x) { std::fprintf(stderr, "strata metal: the MTP model has no tensor %s\n", name.c_str()); std::exit(2); }
        return *x;
    }
    size_t NM(const std::string& name) const {   // the folded (1 + w) copy of an F32 norm of the MTP model
        auto it = mnorm_off_.find(name);
        if (it == mnorm_off_.end()) { std::fprintf(stderr, "strata metal: no MTP norm %s\n", name.c_str()); std::exit(2); }
        return it->second;
    }

    strata::core::SessionState* ss_ = nullptr;
    gguf::File f_;
    std::unique_ptr<mtl::Context> cx_;
    mtl::Buffer W_;
    int lib_ = -1, lgdn_ = -1, lqsa_ = -1, lsmp_ = -1, lple_ = -1;
    std::map<std::string, mtl::Pipeline> pcache_;
    bool nb_ = false;
    int max_t_ = 1;
    int64_t max_cells_ = 0;
    int64_t n_vocab_ = 0;
    float eps_ = 1e-6f;
    // buffers
    mtl::Buffer R_, bo_, inj_[2], xn_, lo_, qkv_, z_, al_, be_, yg_, logits_, ids_, wts_, plan_, h_, ye_, gsh_, ush_,
        ysh_, sg_, qfull_, kcur_, vcur_, iqr_, ikr_, qcur_, qidx_, scores_, qids_, part_, cs_, lm_, apv_, api_, tokout_,
        emb_, pkey_, pval_, qkvL_, alL_, beL_, zL_, ydump_, keyn_, qn_, gated_, normW_, pleHist_;
    Act XA_, XF_, XY_, XQ_, XH_, XE_;
    int cur_inj_ = 0;
    // state
    std::vector<mtl::Buffer> gst_, ghist_[2], kq_, vq_, ksc_, vsc_, tail_, dead_, pooled_;
    int hp_ = 0;                     // which of the two history buffers holds the committed conv history (all layers)
    // commit(n) of a window of several tokens: the GDN fold of the kept prefix waits for the next window and runs at
    // the start of its command buffer (one submit and wait less a round).  Nothing else reads the GDN state between
    // the two (the drafter reads the window's rows and its own caches).  STRATA_METAL_LAZY_FOLD=0: at commit.
    int pend_fold_ = 0;
    void fold(std::vector<mtl::Dispatch>& ds, int n_keep, int hp);
    // the encoded window per (T, head, hp_): built once, then only the positions are patched (the host's encode cost)
    struct Cached {
        std::vector<mtl::Dispatch> ds;
        std::vector<std::pair<int, int>> qsa;   // (dispatch, kind): 0 QsaArgs n_tok = T, 1 the merge (MAXT), 2 the scores
    };
    std::map<int, Cached> wcache_;
    std::vector<std::pair<int, int>>* rec_ = nullptr;
    std::vector<float> ple_hist_;    // NG_HIST x hc_dim normalized rows, row-fastest
    bool ple_on_ = false;
    int last_T_ = 0;
    double last_gpu_ms_ = 0;
    int last_n_disp_ = 0;
    int32_t last_tokens_[MAXT] = {};
    std::vector<float> last_norm_;   // the window's normalized PLE rows (T x hc_dim), for commit
    // STRATA_METAL_LOGPOS=<path> (test tooling, one-token windows): per decode position the log-probability of the
    // token that came next (the next window's input: with --spec-follow, the forced text) and the top 64, in the
    // KL harness's row format (phaseA-tests cuda_p0/cuda_kl.py load_lp): pos, target, lp, argmax, 4 x 0, id:lp ...
    std::FILE* logpos_ = nullptr;
    bool lp_pend_ = false;
    int64_t lp_pos_ = 0, lp_first_ = -1;
    int32_t lp_arg_ = 0;
    double lp_lse_ = 0.0;
    std::vector<float> lp_row_;
    std::vector<std::pair<float, int32_t>> lp_top_;
    void logpos(int Tn, const int32_t* tokens, int64_t pos0, bool head);
    // STRATA_METAL_PROFILE=1: where a window's time goes, per window size (printed when the window goes away)
    struct Prof { int64_t n = 0; double pre = 0, rg = 0, gpu = 0, fin = 0, all = 0; };
    std::map<int, Prof> prof_;
    Prof cprof_, dprof_;   // commit (folds), the draft rounds
    double tp_pre_ = 0, tp_rg_ = 0;
public:
    ~Window();
    void prof_add(bool draft, double ms) { Prof& x = draft ? dprof_ : cprof_; ++x.n; x.all += ms; }
private:
    // the Q8_0 copies of the hyper-connection matrices (the default; STRATA_HC_Q8=0: the BF16 originals)
    bool hc_q8_ = false;
    mtl::Buffer Q8_;
    std::map<std::string, size_t> q8off_;
    // MTP
    bool mtp_on_ = false;
    int lmtp_ = -1;
    gguf::File fm_;
    mtl::Buffer Wm_, mnorm_, mR_, memb_, men_, me2_, mhn_, mh2_, midx_, mbo_, minj_[2], mqids_, mlm_;
    std::map<std::string, size_t> mnorm_off_;
    std::vector<mtl::Buffer> mkq_, mvq_, mksc_, mvsc_, mtail_, mdead_, mpooled_;
    // the draft head over a token subset (data/draft_vocab.bin, as the CUDA drafter): its rows gathered from the head
    mtl::Buffer mhead_;
    gguf::Tensor mhead_t_;
    std::vector<int32_t> dvocab_;
    // the draft layer's BF16 matrices in Q8_0 (half the bytes a draft step reads; drafts only: the verify window
    // decides every token).  STRATA_METAL_MTP_Q8=0 keeps BF16.
    mtl::Buffer mq8_;
    std::map<std::string, gguf::Tensor> mq8t_;
    const mtl::Buffer& MB(const std::string& name) const { return mq8t_.count(name) ? mq8_ : Wm_; }
    const gguf::Tensor& TQ(const std::string& name) const {
        auto it = mq8t_.find(name);
        return it != mq8t_.end() ? it->second : TM(name);
    }
    Act XEn_, XHn_;
    // the draft round in one command buffer (msl/mtp_chain.metal): STRATA_METAL_MTP_CHAIN=0 runs a command buffer
    // per step with the host's pick and embedding (as before)
    int lchain_ = -1;
    bool chain_ = false;
    mtl::Buffer dout_, dprob_, posv_, dmap_;
};

bool Window::init(const strata::core::ModelGeometry& g, strata::core::SessionState& ss, const std::string& shard1,
                  int max_t, int64_t max_cells, std::string& err) {
    ss_ = &ss;
    max_t_ = std::min(max_t, MAXT);
    max_cells_ = max_cells;
    if (g.n_embd != N || g.hc != HC || g.hc_lr != LR || g.n_ff != NFF || g.n_expert != NEXP || g.n_layers != 48) {
        err = "metal window: the geometry differs from Qwen3.8-Flash-Next's";
        return false;
    }
    eps_ = 1e-6f;
    const double t0 = now_ms();
    if (!f_.open(shard1, err)) return false;
    cx_ = std::make_unique<mtl::Context>();
    madvise(f_.map, f_.map_bytes, MADV_WILLNEED);
    W_ = cx_->wrap(f_.map, f_.map_bytes);
    if (W_.handle == nullptr) { err = "metal window: cannot wrap " + shard1 + " as a Metal buffer"; return false; }
    n_vocab_ = (int64_t) T("output.weight").shape[1];
    lib_ = cx_->compile({"quant.metal", "units.metal", "moe.metal", "gemv.metal", "gr.metal", "gr2.metal", "glue.metal"});
    lgdn_ = cx_->compile({"gdn.metal"}, "#define NO_GGML_COMMON 1\n");
    lqsa_ = cx_->compile({"qsa.metal", "qsa2.metal"}, "#define NO_GGML_COMMON 1\n");
    lsmp_ = cx_->compile({"sample.metal"}, "#define NO_GGML_COMMON 1\n");
    lple_ = cx_->compile({std::string(STRATA_METAL_ENGINE_KERNELS_DEFAULT) + "/ple.metal"}, "#define NO_GGML_COMMON 1\n");

    const int MAXC = (int) std::max<int64_t>(4096, (max_cells + 64) / 4 * 4 + 64), MB = MAXC / 4 + 2;
    max_cells_ = MAXC;
    auto A = [&](size_t bytes) { return cx_->alloc(bytes); };
    R_ = A(4 * MAXT * HC * N); bo_ = A(4 * MAXT * N); inj_[0] = A(4 * MAXT * HC); inj_[1] = A(4 * MAXT * HC);
    xn_ = A(4 * MAXT * HC * N); lo_ = A(4 * MAXT * LR); qkv_ = A(4 * MAXT * 10240); z_ = A(4 * MAXT * 6144);
    al_ = A(4 * MAXT * 48); be_ = A(4 * MAXT * 48); yg_ = A(4 * MAXT * 6144); logits_ = A(4 * MAXT * 512);
    ids_ = A(4 * MAXT * TOPK); wts_ = A(4 * MAXT * TOPK); plan_ = A(4 * P_INTS); h_ = A(4 * MAXT * TOPK * NFF);
    ye_ = A(4 * MAXT * TOPK * N); gsh_ = A(4 * MAXT * NFF); ush_ = A(4 * MAXT * NFF); ysh_ = A(4 * MAXT * N);
    sg_ = A(4 * MAXT * 4); qfull_ = A(4 * MAXT * 12288); kcur_ = A(4 * MAXT * 512); vcur_ = A(4 * MAXT * 512);
    iqr_ = A(4 * MAXT * 512); ikr_ = A(4 * MAXT * 128); qcur_ = A(4 * MAXT * 24 * 256); qidx_ = A(4 * MAXT * 4 * 128);
    scores_ = A(4 * (size_t) MAXT * MB); qids_ = A(4 * MAXT * 2051); part_ = A(4 * (size_t) MAXT * 65 * 24 * 258);
    cs_ = A(8 * (size_t) (MAXC + 16) * 32); lm_ = A(4 * (size_t) MAXT * n_vocab_); apv_ = A(4 * MAXT * 120);
    api_ = A(4 * MAXT * 120); tokout_ = A(4 * MAXT); emb_ = A(4 * MAXT * N); pkey_ = A(4 * MAXT * 10240);
    pval_ = A(4 * MAXT * N);
    // the GDN layers' window inputs, kept per layer so commit() can fold the accepted prefix into the state
    qkvL_ = A(4 * (size_t) 36 * MAXT * 10240); alL_ = A(4 * (size_t) 36 * MAXT * 48); beL_ = A(4 * (size_t) 36 * MAXT * 48);
    zL_ = A(4 * (size_t) 36 * MAXT * 6144); ydump_ = A(4 * MAXT * 6144);
    keyn_ = A(4 * MAXT * 10240); qn_ = A(4 * MAXT * 10240); gated_ = A(4 * MAXT * 10240); normW_ = A(4 * MAXT * 10240);
    pleHist_ = A(4 * 9 * 10240);
    auto act = [&](int n) { Act a; a.n = n; a.b = A(4 * ((size_t) 4 * MAXT * n + (size_t) MAXT * n / 16)); return a; };
    XA_ = act(N); XF_ = act(N); XY_ = act(6144); XQ_ = act(6144); XH_ = act(NFF); XE_ = act(N);
    // RoPE: cos/sin [pos][32] for the first 64 dims, NEOX pairs (i, i + 32), in double as build_rope_table
    const double base = f_.meta_num.count("qwen4exp.rope.freq_base") ? f_.meta_num["qwen4exp.rope.freq_base"] : 1e7;
    for (int p = 0; p < MAXC + 16; ++p)
        for (int i = 0; i < 32; ++i) {
            const double th = (double) p * std::pow(base, -2.0 * i / 64.0);
            cs_.as<float>()[(p * 32 + i) * 2] = (float) std::cos(th);
            cs_.as<float>()[(p * 32 + i) * 2 + 1] = (float) std::sin(th);
        }
    for (int t = 0; t < MAXT; ++t)
        for (int c = 0; c < 2051; ++c) qids_.as<int>()[t * 2051 + c] = c;
    for (int l = 0; l < 48; ++l) {
        if (l % 4 != 3) {
            gst_.push_back(A(4 * 128 * 48 * 128));
            ghist_[0].push_back(A(4 * 10240 * 3));
            ghist_[1].push_back(A(4 * 10240 * 3));
        } else {
            kq_.push_back(A((size_t) MAXC * 2 * 256));
            vq_.push_back(A((size_t) MAXC * 2 * 256));
            ksc_.push_back(A(2 * (size_t) MAXC * 2 * 4));
            vsc_.push_back(A(2 * (size_t) MAXC * 2 * 4));
            tail_.push_back(A(4 * 4 * 128));
            dead_.push_back(A(4 * 128));
            pooled_.push_back(A(4 * (size_t) MB * 128));
        }
    }
    // ---- the HC read in Q8_0 (the owner's default on Metal, as STRATA_HC_Q8 on gfx12): the BF16 down / up / inject
    // matrices quantized once at load (ggml's q8_0 rule, ref_quant), read by gr_down_q8 / gr_up_q8
    {
        const char* e = std::getenv("STRATA_HC_Q8");
        hc_q8_ = !(e && std::string(e) == "0");
    }
    if (hc_q8_) {
        const double tq = now_ms();
        size_t tot = 0;
        std::vector<std::string> names;
        for (const auto& kv : f_.tensors) {
            const gguf::Tensor& x = kv.second;
            if (x.type != ref::BF16 || x.name.find("hc_") == std::string::npos || x.name.find("_norm") != std::string::npos) continue;
            uint64_t ne = 1;
            for (uint64_t d : x.shape) ne *= d;
            q8off_[x.name] = tot;
            names.push_back(x.name);
            tot += ne / 32 * 34;
        }
        Q8_ = cx_->alloc((tot + 4095) / 4096 * 4096);
        std::vector<std::thread> th;
        const unsigned nth = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
        for (unsigned k = 0; k < nth; ++k)
            th.emplace_back([&, k] {
                std::vector<float> tmp;
                for (size_t i = k; i < names.size(); i += nth) {
                    const gguf::Tensor& x = *f_.find(names[i]);
                    uint64_t ne = 1;
                    for (uint64_t d : x.shape) ne *= d;
                    tmp.resize(ne);
                    ref::dequant(ref::BF16, (const uint8_t*) f_.map + x.offset, tmp.data(), ne);
                    ref::quantize_q8_0(tmp.data(), ne, Q8_.as<uint8_t>() + q8off_[names[i]]);
                }
            });
        for (auto& x : th) x.join();
        std::fprintf(stderr, "strata metal: HC read in Q8_0 (STRATA_HC_Q8=0: BF16): %zu matrices, %.0f MB, made in %.0f ms\n",
                     names.size(), tot / 1e6, now_ms() - tq);
    }
    ple_on_ = ss.ple.ready() && f_.find("blk.1.ple_key.weight") != nullptr;
    ple_hist_.assign((size_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM, 0.0f);
    last_norm_.assign((size_t) MAXT * strata::kernels::NG_HC_DIM, 0.0f);
    std::fprintf(stderr, "strata metal: decode window on %s: %s wrapped (%.1f GB), %d cells, PLE %s, ready in %.0f ms\n",
                 cx_->device_name().c_str(), shard1.c_str(), f_.file_bytes / 1e9, MAXC, ple_on_ ? "on" : "OFF",
                 now_ms() - t0);
    return true;
}

void Window::gemv(std::vector<mtl::Dispatch>& ds, const gguf::Tensor& t, const Act& x, const mtl::Buffer& y,
                  size_t y_stride, int ntok, size_t y_off, const mtl::Buffer* wb) {
    const int n_in = (int) t.shape[0], n_out = t.shape.size() > 1 ? (int) t.shape[1] : 1;
    const int ty = t.type;
    int L = 16, Rr = 1, S = 4, K = 1;
    if (n_in >= 6144) { L = 32; Rr = 1; S = 8; K = 2; }
    else if (n_in == 640) { L = 8; Rr = 1; S = 4; K = 1; }
    else if (n_out <= 640 || ty == ref::Q2_0 || ty == ref::BF16) { L = 32; Rr = 1; S = 4; K = 1; }
    const std::string fn = "gemv_" + std::to_string(ty) + "_" + std::to_string(L) + "_" + std::to_string(Rr) + "_" +
                           std::to_string(S) + "_" + std::to_string(K);
    const mtl::Pipeline* p = P(lib_, fn, mtl::Constants().i(0, ntok), fn + "#" + std::to_string(ntok));
    GemvArgs a{(uint32_t) n_in, (uint32_t) n_out, (uint32_t) ntok, (uint32_t) ref::row_bytes(t.type, n_in), (uint32_t) n_in,
               (uint32_t) y_stride};
    mtl::Dispatch d;
    d.p = p;
    d.args = {mtl::Arg::B(wb ? *wb : W_, t.offset), mtl::Arg::B(x.b, x.plane_bytes() * xkind(ty)),
              mtl::Arg::B(x.b, x.plane_bytes() * 4), mtl::Arg::B(y, y_off), mtl::Arg::V(a)};
    const int rpt = (S / K) * (32 / L) * Rr;
    d.grid[0] = (n_out + rpt - 1) / rpt;
    d.tg[0] = S * 32;
    push(ds, d);
}

void Window::xprep(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& x, const Act& out, int n) {
    mtl::Dispatch d;
    d.p = P(lib_, "xprep", {}, "xprep");
    const uint32_t nn = n, nt = MAXT;
    d.args = {mtl::Arg::B(x), mtl::Arg::B(out.b), mtl::Arg::V(nn), mtl::Arg::V(nt)};
    d.tg[0] = 256;
    d.grid[0] = (n / 16 * MAXT + 255) / 256;
    push(ds, d);
}

// the hyper-connection read (gr.metal, 3 dispatches): the previous sublayer's write folded in when `apply`
void Window::gr_read(std::vector<mtl::Dispatch>& ds, const std::string& pre, bool apply, const Act& out, bool final_,
                     int ntok) {
    const gguf::Tensor& tn = T(pre + "_norm.weight");
    const gguf::Tensor& tdn = T(pre + "_down.weight");
    const gguf::Tensor& tup = T(pre + "_up.weight");
    const gguf::Tensor& tin = final_ ? tdn : T(pre + "_inject.weight");
    GrArgs a{N, HC, LR, (uint32_t) ntok, apply ? 1u : 0u, eps_};
    mtl::Dispatch d1;
    d1.p = P(lib_, "gr_norm", {}, "gr_norm");
    d1.args = {mtl::Arg::B(R_), mtl::Arg::B(bo_), mtl::Arg::B(inj_[cur_inj_]), mtl::Arg::B(W_, tn.offset), mtl::Arg::B(xn_),
               mtl::Arg::V(a)};
    d1.grid[0] = HC;
    d1.grid[1] = ntok;
    d1.tg[0] = 256;
    push(ds, d1);
    const mtl::Buffer& WB = hc_q8_ ? Q8_ : W_;
    const size_t od = hc_q8_ ? q8off_.at(tdn.name) : tdn.offset, oi = hc_q8_ ? q8off_.at(tin.name) : tin.offset,
                 ou = hc_q8_ ? q8off_.at(tup.name) : tup.offset;
    mtl::Dispatch d2;
    const char* dn = hc_q8_ ? "gr_down_q8_8_8" : "gr_down_8_8";
    d2.p = P(lib_, dn, mtl::Constants().i(1, ntok), std::string(dn) + "#" + std::to_string(ntok));
    d2.args = {mtl::Arg::B(WB, od), mtl::Arg::B(WB, oi), mtl::Arg::B(xn_), mtl::Arg::B(lo_),
               mtl::Arg::B(inj_[1 - cur_inj_]), mtl::Arg::V(a)};
    d2.grid[0] = LR + HC;
    d2.tg[0] = 256;
    push(ds, d2);
    mtl::Dispatch d3;
    const char* up = hc_q8_ ? "gr_up_q8" : "gr_up";
    d3.p = P(lib_, up, mtl::Constants().i(1, ntok), std::string(up) + "#" + std::to_string(ntok));
    GrArgs au = a;
    au.n_tok = MAXT;   // the plane stride of the prepared activation
    d3.args = {mtl::Arg::B(WB, ou), mtl::Arg::B(lo_), mtl::Arg::B(xn_), mtl::Arg::B(out.b), mtl::Arg::V(au)};
    d3.grid[0] = N / 16;
    d3.tg[0] = 256;
    push(ds, d3);
    cur_inj_ = 1 - cur_inj_;
}

void Window::layer(std::vector<mtl::Dispatch>& ds, int l, int ntok, int64_t pos0) {
    const std::string b = "blk." + std::to_string(l) + ".";
    // ---- the mixer: layer 1's read after the PLE is not applied again (its pending write was applied before it)
    gr_read(ds, b + "hc_attn", l > 0 && !(l == 1 && ple_on_), XA_, false, ntok);
    if (l % 4 != 3) {
        const int gi = l - l / 4;
        // a one-token window commits in the step; a longer one leaves the state and commit() folds the kept prefix
        // from the inputs kept per layer (qkvL_ ...)
        const bool self_commit = ntok == 1;
        const size_t o_qkv = (size_t) gi * MAXT * 10240 * 4, o_ab = (size_t) gi * MAXT * 48 * 4, o_z = (size_t) gi * MAXT * 6144 * 4;
        gemv(ds, T(b + "attn_qkv.weight"), XA_, qkvL_, 10240, ntok, o_qkv);
        nb_ = true;
        gemv(ds, T(b + "attn_gate.weight"), XA_, zL_, 6144, ntok, o_z);
        nb_ = true;
        gemv(ds, T(b + "ssm_alpha.weight"), XA_, alL_, 48, ntok, o_ab);
        nb_ = true;
        gemv(ds, T(b + "ssm_beta.weight"), XA_, beL_, 48, ntok, o_ab);
        GdnArgs a{16, 48, 10240, self_commit ? 1u : 0u, eps_, 0};
        mtl::Dispatch d;
        d.p = P(lgdn_, "gdn_step", mtl::Constants().i(2, ntok), "gdn_step#" + std::to_string(ntok));
        const int hp = hp_;
        d.args = {mtl::Arg::B(gst_[(size_t) gi]), mtl::Arg::B(ghist_[hp][(size_t) gi]), mtl::Arg::B(qkvL_, o_qkv),
                  mtl::Arg::B(alL_, o_ab), mtl::Arg::B(beL_, o_ab), mtl::Arg::B(zL_, o_z),
                  mtl::Arg::B(W_, T(b + "ssm_conv1d.weight").offset),
                  mtl::Arg::B(W_, T(b + "ssm_dt.bias").offset), mtl::Arg::B(W_, T(b + "ssm_a").offset),
                  mtl::Arg::B(W_, T(b + "ssm_norm.weight").offset), mtl::Arg::B(yg_), mtl::Arg::V(a),
                  mtl::Arg::B(ghist_[1 - hp][(size_t) gi])};
        d.grid[0] = 48;
        d.tg[0] = 512;
        push(ds, d);
        (void) self_commit;   // the history ping-pong turns after the window (run) or at commit
        xprep(ds, yg_, XY_, 6144);
        gemv(ds, T(b + "ssm_out.weight"), XY_, bo_, N, ntok);
    } else {
        const int qi = l / 4;
        gemv(ds, T(b + "attn_q.weight"), XA_, qfull_, 12288, ntok);
        nb_ = true;
        gemv(ds, T(b + "attn_k.weight"), XA_, kcur_, 512, ntok);
        nb_ = true;
        gemv(ds, T(b + "attn_v.weight"), XA_, vcur_, 512, ntok);
        nb_ = true;
        gemv(ds, T(b + "indexer.q_proj.weight"), XA_, iqr_, 512, ntok);
        nb_ = true;
        gemv(ds, T(b + "indexer.k_proj.weight"), XA_, ikr_, 128, ntok);
        QsaArgs a{(uint32_t) pos0, 0, (uint32_t) max_cells_, (uint32_t) ntok, eps_};
        mtl::Dispatch d;
        d.p = P(lqsa_, "qsa_prep", {}, "qsa_prep");
        d.args = {mtl::Arg::B(qfull_), mtl::Arg::B(kcur_), mtl::Arg::B(vcur_), mtl::Arg::B(iqr_), mtl::Arg::B(ikr_),
                  mtl::Arg::B(W_, T(b + "attn_q_norm.weight").offset), mtl::Arg::B(W_, T(b + "attn_k_norm.weight").offset),
                  mtl::Arg::B(W_, T(b + "indexer.q_norm.weight").offset), mtl::Arg::B(W_, T(b + "indexer.k_norm.weight").offset),
                  mtl::Arg::B(cs_), mtl::Arg::B(qcur_), mtl::Arg::B(qidx_), mtl::Arg::B(kq_[(size_t) qi]),
                  mtl::Arg::B(vq_[(size_t) qi]), mtl::Arg::B(ksc_[(size_t) qi]), mtl::Arg::B(vsc_[(size_t) qi]),
                  mtl::Arg::B(tail_[(size_t) qi]), mtl::Arg::B(dead_[(size_t) qi]), mtl::Arg::B(pooled_[(size_t) qi]),
                  mtl::Arg::V(a)};
        d.grid[0] = 33;
        d.grid[1] = ntok;
        d.tg[0] = 256;
        push(ds, d);
        if (rec_) rec_->push_back({(int) ds.size() - 1, 0});
        mtl::Dispatch s;
        s.p = P(lqsa_, "qsa_score", {}, "qsa_score");
        s.args = {mtl::Arg::B(qidx_), mtl::Arg::B(pooled_[(size_t) qi]), mtl::Arg::B(dead_[(size_t) qi]), mtl::Arg::B(scores_),
                  mtl::Arg::V(a)};
        s.grid[0] = (uint32_t) (((pos0 + ntok) / 4 + 1 + 7) / 8);
        s.grid[1] = ntok;
        s.tg[0] = 256;
        push(ds, s);
        if (rec_) rec_->push_back({(int) ds.size() - 1, 2});
        mtl::Dispatch k;
        k.p = P(lqsa_, "qsa_topk2", {}, "qsa_topk");
        k.args = {mtl::Arg::B(scores_), mtl::Arg::B(qids_), mtl::Arg::V(a)};
        k.grid[0] = ntok;
        k.tg[0] = 1024;
        push(ds, k);
        if (rec_) rec_->push_back({(int) ds.size() - 1, 0});
        mtl::Dispatch at;
        at.p = P(lqsa_, "qsa_attn", {}, "qsa_attn");
        at.args = {mtl::Arg::B(qcur_), mtl::Arg::B(kq_[(size_t) qi]), mtl::Arg::B(vq_[(size_t) qi]), mtl::Arg::B(ksc_[(size_t) qi]),
                   mtl::Arg::B(vsc_[(size_t) qi]), mtl::Arg::B(qids_), mtl::Arg::B(part_), mtl::Arg::V(a)};
        at.grid[0] = 33;
        at.grid[1] = 2;
        at.grid[2] = ntok;
        at.tg[0] = 256;
        push(ds, at);
        if (rec_) rec_->push_back({(int) ds.size() - 1, 0});
        mtl::Dispatch m;
        m.p = P(lqsa_, "qsa_merge", {}, "qsa_merge");
        QsaArgs am = a;
        am.n_tok = MAXT;   // the plane stride
        m.args = {mtl::Arg::B(part_), mtl::Arg::B(qfull_), mtl::Arg::B(XQ_.b), mtl::Arg::V(am)};
        m.grid[0] = 24;
        m.grid[1] = ntok;
        m.tg[0] = 256;
        push(ds, m);
        if (rec_) rec_->push_back({(int) ds.size() - 1, 1});
        gemv(ds, T(b + "attn_output.weight"), XQ_, bo_, N, ntok);
    }
    // ---- the MoE
    gr_read(ds, b + "hc_ffn", true, XF_, false, ntok);
    gemv(ds, T(b + "ffn_gate_inp.weight"), XF_, logits_, 512, ntok);
    nb_ = true;
    gemv(ds, T(b + "ffn_gate_shexp.weight"), XF_, gsh_, NFF, ntok);
    nb_ = true;
    gemv(ds, T(b + "ffn_up_shexp.weight"), XF_, ush_, NFF, ntok);
    nb_ = true;
    gemv(ds, T(b + "ffn_gate_inp_shexp.weight"), XF_, sg_, 1, ntok);
    {
        MoeArgs a{};
        const gguf::Tensor &tg_ = T(b + "ffn_gate_exps.weight"), &tu = T(b + "ffn_up_exps.weight"),
                           &td = T(b + "ffn_down_exps.weight");
        a.gate_off = tg_.offset;
        a.up_off = tu.offset;
        a.down_off = td.offset;
        a.gu_row = (uint32_t) ref::row_bytes(tg_.type, N);
        a.d_row = (uint32_t) ref::row_bytes(td.type, NFF);
        a.gu_stride = (uint64_t) a.gu_row * NFF;
        a.d_stride = (uint64_t) a.d_row * N;
        a.n_embd = N;
        a.n_ff = NFF;
        a.n_tok = ntok;
        a.topk = TOPK;
        a.n_expert = NEXP;
        const bool dq2 = td.type == ref::Q2_0;
        const int gl = 32, gr = 1, gs = 4, dl = dq2 ? 8 : 4, dr = dq2 ? 1 : 2, ds_ = 4;
        a.gx_gu = NFF / (gs * (32 / gl) * gr);
        a.gx_dn = N / (ds_ * (32 / dl) * dr);
        mtl::Dispatch r;
        r.p = P(lib_, "moe_route", {}, "moe_route");
        r.args = {mtl::Arg::B(logits_), mtl::Arg::B(ids_), mtl::Arg::B(wts_), mtl::Arg::B(plan_), mtl::Arg::V(a)};
        r.tg[0] = 512;
        push(ds, r);
        {
            GlueArgs ga{NFF, MAXT, N, TOPK};
            mtl::Dispatch d;
            d.p = P(lib_, "swiglu_prep", {}, "swiglu_prep");
            d.args = {mtl::Arg::B(gsh_), mtl::Arg::B(ush_), mtl::Arg::B(XH_.b), mtl::Arg::V(ga)};
            d.grid[0] = NFF / 16;
            d.grid[1] = ntok;
            d.tg[0] = 16;
            nb_ = true;
            push(ds, d);
        }
        const std::string gun = "moe_gu_" + std::to_string(tg_.type) + "_" + std::to_string(gl) + "_" + std::to_string(gr) +
                                "_" + std::to_string(gs);
        const std::string dnn = "moe_down_" + std::to_string(dq2 ? 142 : td.type) + "_" + std::to_string(dl) + "_" +
                                std::to_string(dr) + "_" + std::to_string(ds_);
        mtl::Dispatch g;
        g.p = P(lib_, gun, {}, gun);
        g.args = {mtl::Arg::B(W_), mtl::Arg::B(XF_.b, XF_.plane_bytes() * xkind(tg_.type)), mtl::Arg::B(h_), mtl::Arg::B(plan_),
                  mtl::Arg::V(a), mtl::Arg::B(XF_.b, XF_.plane_bytes() * 4)};
        g.tg[0] = 32 * gs;
        g.indirect = &plan_;
        g.ind_off = P_IGU * 4;
        push(ds, g);
        nb_ = true;
        gemv(ds, T(b + "ffn_down_shexp.weight"), XH_, ysh_, N, ntok);
        mtl::Dispatch dd;
        dd.p = P(lib_, dnn, {}, dnn);
        dd.args = {mtl::Arg::B(W_), mtl::Arg::B(h_), mtl::Arg::B(ye_), mtl::Arg::B(plan_), mtl::Arg::V(a), mtl::Arg::B(h_)};
        dd.tg[0] = 32 * ds_;
        dd.indirect = &plan_;
        dd.ind_off = P_IDN * 4;
        push(ds, dd);
    }
    {
        GlueArgs a{N, (uint32_t) ntok, N, TOPK};
        mtl::Dispatch d;
        d.p = P(lib_, "moe_combine_sh", {}, "moe_combine_sh");
        d.args = {mtl::Arg::B(ye_), mtl::Arg::B(plan_), mtl::Arg::B(ysh_), mtl::Arg::B(sg_), mtl::Arg::B(bo_), mtl::Arg::V(a)};
        d.grid[0] = (ntok * N + 255) / 256;
        d.tg[0] = 256;
        push(ds, d);
    }
}

namespace {
float bf16f(uint16_t h) { uint32_t i = (uint32_t) h << 16; float f; std::memcpy(&f, &i, 4); return f; }
// ple.cu's gnorm_kernel: per stream, the squares rounded in f32, summed in double
void gnorm(const float* x, const float* w, float* y, int n_embd, int hc, float eps) {
    for (int c = 0; c < hc; ++c) {
        const float* xc = x + (size_t) c * n_embd;
        double acc = 0.0;
        for (int d = 0; d < n_embd; ++d) { const float sq = xc[d] * xc[d]; acc += (double) sq; }
        const float mean = (float) (acc / (double) n_embd);
        const float scale = 1.0f / std::sqrt(mean + eps);
        for (int d = 0; d < n_embd; ++d) y[(size_t) c * n_embd + d] = xc[d] * scale * w[(size_t) c * n_embd + d];
    }
}
}  // namespace

// The PLE block (ple.hpp) for the window's T rows on the CPU, after layer 0 (its pending write applied to R): R +=
// gated + silu(conv); the normalized rows go to last_norm_ (the history advances at commit).
void Window::ple_cpu(int Tn) {
    using namespace strata::kernels;
    const int hc_dim = NG_HC_DIM;
    const float* nkey = (const float*) ((const uint8_t*) f_.map + T("blk.1.ple_norm_key.weight").offset);
    const float* nqry = (const float*) ((const uint8_t*) f_.map + T("blk.1.ple_norm_query.weight").offset);
    const float* nconv = (const float*) ((const uint8_t*) f_.map + T("blk.1.ple_norm_conv.weight").offset);
    const uint16_t* kw = (const uint16_t*) ((const uint8_t*) f_.map + T("blk.1.ple_conv1d.weight").offset);
    std::vector<float> key(hc_dim), query(hc_dim), gated(hc_dim), norm(hc_dim), hist = ple_hist_;
    for (int t = 0; t < Tn; ++t) {
        float* hidden = R_.as<float>() + (size_t) t * HC * N;
        gnorm(pkey_.as<float>() + (size_t) t * hc_dim, nkey, key.data(), N, HC, NG_RMS_EPS);
        gnorm(hidden, nqry, query.data(), N, HC, NG_RMS_EPS);
        const float* value = pval_.as<float>() + (size_t) t * N;
        float gate[HC];
        const float inv_sqrt_n = 1.0f / std::sqrt((float) N);
        for (int c = 0; c < HC; ++c) {
            double acc = 0.0;
            for (int d = 0; d < N; ++d) acc += (double) (key[(size_t) c * N + d] * query[(size_t) c * N + d]);
            const float s = (float) acc * inv_sqrt_n;
            const float mag = std::sqrt(std::max(std::fabs(s), 1e-6f));
            const float sgn = s > 0.0f ? 1.0f : (s < 0.0f ? -1.0f : 0.0f);
            gate[c] = 1.0f / (1.0f + std::exp(-(sgn * mag)));
        }
        for (int i = 0; i < hc_dim; ++i) gated[(size_t) i] = value[i % N] * gate[i / N];
        gnorm(gated.data(), nconv, norm.data(), N, HC, NG_RMS_EPS);
        for (int c = 0; c < hc_dim; ++c) {
            float acc = 0.0f;
            for (int k = 0; k < PLE_CONV_KERNEL; ++k) {
                const int row = NG_HIST - (PLE_CONV_KERNEL - 1 - k) * NGRAM_SIZE;
                const float v = row == NG_HIST ? norm[(size_t) c] : hist[(size_t) row + (size_t) NG_HIST * c];
                acc += ref::f16_to_f32(kw[k + PLE_CONV_KERNEL * c]) * v;
            }
            const float conv = acc / (1.0f + std::exp(-acc));
            hidden[c] = hidden[c] + gated[(size_t) c] + conv;
        }
        std::memcpy(last_norm_.data() + (size_t) t * hc_dim, norm.data(), (size_t) hc_dim * 4);
        // the next row of this window sees this one in its history
        for (int c = 0; c < hc_dim; ++c) {
            float* col = hist.data() + (size_t) c * NG_HIST;
            for (int r = 0; r + 1 < NG_HIST; ++r) col[r] = col[r + 1];
            col[NG_HIST - 1] = norm[(size_t) c];
        }
    }
}

bool Window::run(int Tn, const int32_t* tokens, int64_t pos0, bool head, int32_t* out, std::string& err) {
    if (Tn < 1 || Tn > MAXT || (head && Tn > max_t_)) { err = "metal window: window size out of range"; return false; }
    if (pos0 + Tn + 1 >= max_cells_) { err = "metal window: the window runs past the context"; return false; }
    const double t0 = now_ms();
    // ---- the embeddings (CPU dequant of the token rows straight into the shared buffer), broadcast to the streams
    const gguf::Tensor& te = T("token_embd.weight");
    const size_t rb = ref::row_bytes(te.type, N);
    for (int t = 0; t < Tn; ++t) {
        float* r = R_.as<float>() + (size_t) t * HC * N;
        ref::dequant(te.type, (const uint8_t*) f_.map + te.offset + (size_t) tokens[t] * rb, r, N);
        for (int c = 1; c < HC; ++c) std::memcpy(r + (size_t) c * N, r, (size_t) N * 4);
    }
    // ---- the PLE rows (the table read is the shared host code's), projected on the GPU
    if (ple_on_) {
        strata::core::SessionState& ss = *ss_;
        uint32_t rows[MAXT * strata::kernels::PLE_N_HEADS];
        int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
        for (int t = 0; t < Tn; ++t) {
            strata::kernels::ngram_rows(&tokens[t], prev, 1, ss.ple.consts, rows + t * strata::kernels::PLE_N_HEADS);
            prev[0] = prev[1];
            prev[1] = tokens[t];
        }
        if (!ss.ple.table->gather_batch(rows, (size_t) Tn, emb_.as<float>(), err)) return false;
    }
    tp_pre_ = now_ms();
    const int key = (((Tn * 2 + (head ? 1 : 0)) * 2 + hp_) * (MAXT + 1)) + pend_fold_;
    const int fold_n = pend_fold_;
    pend_fold_ = 0;
    auto hit = wcache_.find(key);
    if (hit != wcache_.end()) {
        // the same window shape: only the positions change (QSA's cells, the score grid)
        Cached& c = hit->second;
        for (const auto& [i, kind] : c.qsa) {
            mtl::Dispatch& d = c.ds[(size_t) i];
            QsaArgs a{(uint32_t) pos0, 0, (uint32_t) max_cells_, (uint32_t) (kind == 1 ? MAXT : Tn), eps_};
            std::memcpy(d.args.back().bytes.data(), &a, sizeof a);
            if (kind == 2) d.grid[0] = (uint32_t) (((pos0 + Tn) / 4 + 1 + 7) / 8);
        }
        const double r0 = now_ms();
        last_gpu_ms_ = 1000.0 * cx_->run_graph(c.ds);
        tp_rg_ = now_ms() - r0;
        last_n_disp_ = (int) c.ds.size();
        return finish(Tn, tokens, pos0, head, out, t0);
    }
    Cached& cw = wcache_[key];
    rec_ = &cw.qsa;
    std::vector<mtl::Dispatch>& ds = cw.ds;
    nb_ = false;
    if (fold_n > 0) fold(ds, fold_n, 1 - hp_);   // the last window's commit (hp_ already turned)
    if (ple_on_) {
        xprep(ds, emb_, XE_, N);
        gemv(ds, T("blk.1.ple_key.weight"), XE_, pkey_, 10240, Tn);
        nb_ = true;
        gemv(ds, T("blk.1.ple_value.weight"), XE_, pval_, N, Tn);
    }
    cur_inj_ = 0;
    layer(ds, 0, Tn, pos0);
    if (ple_on_) {
        // layer 0's FFN write applied to R (gr_norm with apply writes R back; its xn is not used)
        const gguf::Tensor& tn = T("blk.1.hc_attn_norm.weight");
        GrArgs a{N, HC, LR, (uint32_t) Tn, 1u, eps_};
        mtl::Dispatch d1;
        d1.p = P(lib_, "gr_norm", {}, "gr_norm");
        d1.args = {mtl::Arg::B(R_), mtl::Arg::B(bo_), mtl::Arg::B(inj_[cur_inj_]), mtl::Arg::B(W_, tn.offset), mtl::Arg::B(xn_),
                   mtl::Arg::V(a)};
        d1.grid[0] = HC;
        d1.grid[1] = Tn;
        d1.tg[0] = 256;
        push(ds, d1);
        // the PLE block on the GPU (msl/ple.metal): norms, gate, conv, the residual update
        struct PleArgs { uint32_t n_embd, hc; float eps; } pa{N, HC, strata::kernels::NG_RMS_EPS};
        auto off = [&](const char* n) { return T(n).offset; };
        mtl::Dispatch g1, g2, g3, g4;
        g1.p = P(lple_, "ple_gnorm", {}, "ple_gnorm");
        g1.args = {mtl::Arg::B(pkey_), mtl::Arg::B(W_, off("blk.1.ple_norm_key.weight")), mtl::Arg::B(keyn_), mtl::Arg::V(pa)};
        g1.grid[0] = HC; g1.grid[1] = Tn; g1.tg[0] = 256;
        push(ds, g1);
        g2.p = g1.p;
        g2.args = {mtl::Arg::B(R_), mtl::Arg::B(W_, off("blk.1.ple_norm_query.weight")), mtl::Arg::B(qn_), mtl::Arg::V(pa)};
        g2.grid[0] = HC; g2.grid[1] = Tn; g2.tg[0] = 256;
        nb_ = true;   // beside the key norm
        push(ds, g2);
        g3.p = P(lple_, "ple_gate_norm", {}, "ple_gate_norm");
        g3.args = {mtl::Arg::B(keyn_), mtl::Arg::B(qn_), mtl::Arg::B(pval_), mtl::Arg::B(W_, off("blk.1.ple_norm_conv.weight")),
                   mtl::Arg::B(gated_), mtl::Arg::B(normW_), mtl::Arg::V(pa)};
        g3.grid[0] = HC; g3.grid[1] = Tn; g3.tg[0] = 256;
        push(ds, g3);
        g4.p = P(lple_, "ple_conv_add", {}, "ple_conv_add");
        g4.args = {mtl::Arg::B(pleHist_), mtl::Arg::B(normW_), mtl::Arg::B(W_, off("blk.1.ple_conv1d.weight")),
                   mtl::Arg::B(gated_), mtl::Arg::B(R_), mtl::Arg::V(pa)};
        g4.grid[0] = HC * N / 256; g4.grid[1] = Tn; g4.tg[0] = 256;
        push(ds, g4);
    }
    for (int l = 1; l < 48; ++l) layer(ds, l, Tn, pos0);
    // ---- the final mixer and the head
    gr_read(ds, "output_hc", true, XA_, true, Tn);
    if (head) gemv(ds, T("output.weight"), XA_, lm_, (size_t) n_vocab_, Tn);
    if (head) {
        ArgArgs a{(uint32_t) n_vocab_, 120};
        mtl::Dispatch d1, d2;
        d1.p = P(lsmp_, "argmax_part", {}, "argmax_part");
        d1.args = {mtl::Arg::B(lm_), mtl::Arg::B(apv_), mtl::Arg::B(api_), mtl::Arg::V(a)};
        d1.grid[0] = 120;
        d1.grid[1] = Tn;
        d1.tg[0] = 256;
        push(ds, d1);
        d2.p = P(lsmp_, "argmax_final", {}, "argmax_final");
        d2.args = {mtl::Arg::B(apv_), mtl::Arg::B(api_), mtl::Arg::B(tokout_), mtl::Arg::V(a)};
        d2.grid[0] = Tn;
        d2.tg[0] = 256;
        push(ds, d2);
    }
    rec_ = nullptr;
    const double r0 = now_ms();
    last_gpu_ms_ = 1000.0 * cx_->run_graph(ds);
    tp_rg_ = now_ms() - r0;
    last_n_disp_ = (int) ds.size();
    return finish(Tn, tokens, pos0, head, out, t0);
}

void Window::logpos(int Tn, const int32_t* tokens, int64_t pos0, bool head) {
    static const char* path = std::getenv("STRATA_METAL_LOGPOS");
    if (path == nullptr || !head) return;
    if (logpos_ == nullptr) logpos_ = std::fopen(path, "ab");
    if (logpos_ == nullptr) return;
    if (lp_pend_ && Tn >= 1 && pos0 == lp_pos_ + 1) {
        const int32_t tgt = tokens[0];
        std::fprintf(logpos_, "%lld\t%d\t%.6f\t%d\t0\t0\t0\t0", (long long) (lp_pos_ - lp_first_), tgt,
                     (double) lp_row_[(size_t) tgt] - lp_lse_, lp_arg_);
        for (const auto& [v, id] : lp_top_) std::fprintf(logpos_, "\t%d:%.6f", id, (double) v - lp_lse_);
        std::fputc('\n', logpos_);
        std::fflush(logpos_);
    }
    lp_pend_ = false;
    if (Tn != 1) return;
    if (lp_first_ < 0) lp_first_ = pos0;
    const float* lg = lm_.as<float>();
    lp_row_.assign(lg, lg + n_vocab_);
    float mx = lg[0];
    int32_t am = 0;
    for (int64_t i = 1; i < n_vocab_; ++i) if (lg[i] > mx) { mx = lg[i]; am = (int32_t) i; }
    double s = 0.0;
    for (int64_t i = 0; i < n_vocab_; ++i) s += std::exp((double) (lg[i] - mx));
    lp_lse_ = (double) mx + std::log(s);
    lp_arg_ = am;
    lp_top_.clear();
    for (int64_t i = 0; i < n_vocab_; ++i) lp_top_.push_back({lg[i], (int32_t) i});
    std::partial_sort(lp_top_.begin(), lp_top_.begin() + 64, lp_top_.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    lp_top_.resize(64);
    lp_pos_ = pos0;
    lp_pend_ = true;
}

Window::~Window() {
    if (logpos_) std::fclose(logpos_);
    if (std::getenv("STRATA_METAL_PROFILE") == nullptr) return;
    auto line = [](const char* what, const Prof& x) {
        if (x.n == 0) return;
        std::fprintf(stderr, "strata metal profile: %-10s %6lld x  %.3f ms: host before %.3f, run_graph %.3f (GPU %.3f, so "
                             "encode/submit/wait %.3f), patch+finish %.3f\n", what, (long long) x.n, x.all / x.n, x.pre / x.n,
                     x.rg / x.n, x.gpu / x.n, (x.rg - x.gpu) / x.n, x.fin / x.n);
    };
    for (const auto& [k, x] : prof_) {
        const std::string w = "window T" + std::to_string(k / 2) + (k % 2 ? "" : " nohead");
        line(w.c_str(), x);
    }
    line("commit", cprof_);
    line("draft", dprof_);
}

bool Window::finish(int Tn, const int32_t* tokens, int64_t pos0, bool head, int32_t* out, double t0) {
    static const bool prof = std::getenv("STRATA_METAL_PROFILE") != nullptr;
    if (prof) {
        Prof& x = prof_[Tn * 2 + (head ? 1 : 0)];
        const double t1 = now_ms();
        ++x.n; x.pre += tp_pre_ - t0; x.rg += tp_rg_; x.gpu += last_gpu_ms_; x.fin += t1 - tp_pre_ - tp_rg_;
        x.all += t1 - t0;
    }
    logpos(Tn, tokens, pos0, head);
    if (Tn == 1) hp_ = 1 - hp_;   // a one-token window committed its GDN steps
    for (int t = 0; t < Tn; ++t) out[t] = head ? tokout_.as<int32_t>()[t] : -1;
    last_T_ = Tn;
    for (int t = 0; t < Tn; ++t) last_tokens_[t] = tokens[t];
    static const bool trace = std::getenv("STRATA_METAL_TRACE") != nullptr;
    if (trace && head) {
        // the pick and its margin over the runner-up (a near tie says a backend difference may flip it)
        const float* lg = lm_.as<float>();
        int i1 = -1, i2 = -1;
        for (int i = 0; i < n_vocab_; ++i) {
            if (i1 < 0 || lg[i] > lg[i1]) { i2 = i1; i1 = i; }
            else if (i2 < 0 || lg[i] > lg[i2]) i2 = i;
        }
        std::fprintf(stderr, "strata metal: window pos %lld token %d -> %d (top %d %.4f, 2nd %d %.4f, margin %.4f) %.2f ms (GPU %.2f ms, %d dispatches)\n",
                     (long long) pos0, tokens[0], out[0], i1, lg[i1], i2, lg[i2], lg[i1] - lg[i2], now_ms() - t0, last_gpu_ms_, last_n_disp_);
    }
    return true;
}

void Window::fold(std::vector<mtl::Dispatch>& ds, int n_keep, int hp) {
    for (int l = 0; l < 48; ++l) {
        if (l % 4 == 3) continue;
        const int gi = l - l / 4;
        const std::string b = "blk." + std::to_string(l) + ".";
        const size_t o_qkv = (size_t) gi * MAXT * 10240 * 4, o_ab = (size_t) gi * MAXT * 48 * 4, o_z = (size_t) gi * MAXT * 6144 * 4;
        GdnArgs a{16, 48, 10240, 1u, eps_, 0};
        mtl::Dispatch d;
        d.p = P(lgdn_, "gdn_step", mtl::Constants().i(2, n_keep), "gdn_step#" + std::to_string(n_keep));
        d.args = {mtl::Arg::B(gst_[(size_t) gi]), mtl::Arg::B(ghist_[hp][(size_t) gi]), mtl::Arg::B(qkvL_, o_qkv),
                  mtl::Arg::B(alL_, o_ab), mtl::Arg::B(beL_, o_ab), mtl::Arg::B(zL_, o_z),
                  mtl::Arg::B(W_, T(b + "ssm_conv1d.weight").offset), mtl::Arg::B(W_, T(b + "ssm_dt.bias").offset),
                  mtl::Arg::B(W_, T(b + "ssm_a").offset), mtl::Arg::B(W_, T(b + "ssm_norm.weight").offset),
                  mtl::Arg::B(ydump_), mtl::Arg::V(a), mtl::Arg::B(ghist_[1 - hp][(size_t) gi])};
        d.grid[0] = 48;
        d.tg[0] = 512;
        nb_ = true;   // the layers' folds are independent of each other
        push(ds, d);
    }
    nb_ = false;   // what follows reads the folded state: a barrier
}

bool Window::commit(int n_keep, std::string& err) {
    if (n_keep < 1 || n_keep > last_T_) { err = "metal window: commit count out of range"; return false; }
    using namespace strata::kernels;
    if (last_T_ > 1) {
        // the GDN layers: the kept prefix folded into the state (gdn_step over n_keep tokens from the window's own
        // inputs, with commit; bitwise the window's trajectory).  K/V and indexer cells past the prefix are rewritten
        // when those positions run again, before anything reads them.
        static const bool lazy = !(std::getenv("STRATA_METAL_LAZY_FOLD") && std::string(std::getenv("STRATA_METAL_LAZY_FOLD")) == "0");
        if (lazy) {
            pend_fold_ = n_keep;   // folded at the start of the next window (from the history parity before the turn)
        } else {
            std::vector<mtl::Dispatch> ds;
            nb_ = false;
            fold(ds, n_keep, hp_);
            cx_->run_graph(ds);
        }
        hp_ = 1 - hp_;
    }
    strata::core::SessionState& ss = *ss_;
    for (int t = 0; t < n_keep; ++t) {
        if (ple_on_) {
            const float* norm = normW_.as<float>() + (size_t) t * NG_HC_DIM;
            for (int c = 0; c < NG_HC_DIM; ++c) {
                float* col = pleHist_.as<float>() + (size_t) c * NG_HIST;
                for (int r = 0; r + 1 < NG_HIST; ++r) col[r] = col[r + 1];
                col[NG_HIST - 1] = norm[c];
            }
        }
        ss.ple_prev[0] = ss.ple_prev[1];
        ss.ple_prev[1] = last_tokens_[t];
    }
    return true;
}


// ================================================================================================ the MTP draft layer
bool Window::mtp_load(const std::string& gguf_path, int max_t, std::string& err) {
    (void) max_t;
    if (!fm_.open(gguf_path, err)) return false;
    madvise(fm_.map, fm_.map_bytes, MADV_WILLNEED);
    Wm_ = cx_->wrap(fm_.map, fm_.map_bytes);
    if (Wm_.handle == nullptr) { err = "metal mtp: cannot wrap " + gguf_path; return false; }
    // vLLM's GemmaRMSNorm scales by (1 + w): the F32 norms folded once (tools/mtp_rt.py does the same for CUDA)
    size_t tot = 0;
    for (const auto& kv : fm_.tensors)
        if (kv.second.type == 0) { mnorm_off_[kv.first] = tot; tot += (kv.second.bytes + 255) / 256 * 256; }
    mnorm_ = cx_->alloc(tot + 256);
    for (const auto& kv : mnorm_off_) {
        const gguf::Tensor& x = fm_.tensors.at(kv.first);
        const float* src = (const float*) ((const uint8_t*) fm_.map + x.offset);
        float* dst = (float*) (mnorm_.as<uint8_t>() + kv.second);
        for (uint64_t i = 0; i < x.bytes / 4; ++i) dst[i] = src[i] + 1.0f;
    }
    lmtp_ = cx_->compile({std::string(STRATA_METAL_ENGINE_KERNELS_DEFAULT) + "/mtp.metal"}, "#define NO_GGML_COMMON 1\n");
    {
        const char* e = std::getenv("STRATA_METAL_MTP_Q8");
        if (!(e && std::string(e) == "0")) {
            size_t tot = 0;
            std::vector<std::string> names;
            for (const auto& kv : fm_.tensors) {
                const gguf::Tensor& x = kv.second;
                uint64_t ne = 1;
                for (uint64_t d : x.shape) ne *= d;
                if (x.type != ref::BF16 || x.shape.size() < 2 || x.shape[0] % 32 != 0) continue;
                gguf::Tensor q = x;
                q.type = ref::Q8_0;
                q.offset = tot;
                q.bytes = ne / 32 * 34;
                mq8t_[x.name] = q;
                names.push_back(x.name);
                tot += (q.bytes + 255) / 256 * 256;
            }
            mq8_ = cx_->alloc(tot + 4096);
            std::vector<std::thread> th;
            const unsigned nth = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
            for (unsigned k = 0; k < nth; ++k)
                th.emplace_back([&, k] {
                    std::vector<float> tmp;
                    for (size_t i = k; i < names.size(); i += nth) {
                        const gguf::Tensor& x = fm_.tensors.at(names[i]);
                        const uint64_t ne = x.bytes / 2;
                        tmp.resize(ne);
                        ref::dequant(ref::BF16, (const uint8_t*) fm_.map + x.offset, tmp.data(), ne);
                        ref::quantize_q8_0(tmp.data(), ne, mq8_.as<uint8_t>() + mq8t_[names[i]].offset);
                    }
                });
            for (auto& x : th) x.join();
        }
    }
    auto A = [&](size_t bytes) { return cx_->alloc(bytes); };
    const int MAXC = (int) max_cells_, MB = MAXC / 4 + 2;
    mR_ = A(4 * MAXT * HC * N); memb_ = A(4 * MAXT * N); men_ = A(4 * MAXT * N); me2_ = A(4 * MAXT * N);
    mhn_ = A(4 * MAXT * HC * N); mh2_ = A(4 * HC * MAXT * N); midx_ = A(4 * MAXT * 640); mbo_ = A(4 * MAXT * N);
    minj_[0] = A(4 * MAXT * HC); minj_[1] = A(4 * MAXT * HC); mqids_ = A(4 * MAXT * 2051); mlm_ = A(4 * (size_t) n_vocab_);
    auto act = [&](int n) { Act x; x.n = n; x.b = A(4 * ((size_t) 4 * MAXT * n + (size_t) MAXT * n / 16)); return x; };
    XEn_ = act(N); XHn_ = act(N);
    mkq_.push_back(A((size_t) MAXC * 2 * 256)); mvq_.push_back(A((size_t) MAXC * 2 * 256));
    mksc_.push_back(A(2 * (size_t) MAXC * 2 * 4)); mvsc_.push_back(A(2 * (size_t) MAXC * 2 * 4));
    mtail_.push_back(A(4 * 4 * 128)); mdead_.push_back(A(4 * 128)); mpooled_.push_back(A(4 * (size_t) MB * 128));
    {
        std::string vp;
        if (const char* e = std::getenv("STRATA_METAL_DRAFT_VOCAB")) vp = e;
        else vp = std::string(STRATA_METAL_DATA_DEFAULT) + "/draft_vocab.bin";
        FILE* f = vp == "full" ? nullptr : std::fopen(vp.c_str(), "rb");
        if (f) {
            int32_t id;
            while (std::fread(&id, 4, 1, f) == 1) if (id >= 0 && id < n_vocab_) dvocab_.push_back(id);
            std::fclose(f);
        }
        if (!dvocab_.empty()) {
            const gguf::Tensor& h = T("output.weight");
            const size_t rb = ref::row_bytes(h.type, N);
            mhead_ = A(dvocab_.size() * rb + 4096);
            for (size_t i = 0; i < dvocab_.size(); ++i)
                std::memcpy(mhead_.as<uint8_t>() + i * rb, (const uint8_t*) f_.map + h.offset + (size_t) dvocab_[i] * rb, rb);
            mhead_t_ = h;
            mhead_t_.offset = 0;
            mhead_t_.shape = {(uint64_t) N, (uint64_t) dvocab_.size()};
            mhead_t_.bytes = dvocab_.size() * rb;
        }
    }
    mtp_on_ = true;
    {
        const int ty = T("token_embd.weight").type;
        const int ok_types[] = {30, 8, 2, 6, 20, 23, 42, 21, 11, 12, 13, 14, 18};
        const char* e = std::getenv("STRATA_METAL_MTP_CHAIN");
        chain_ = !(e && std::string(e) == "0") && std::find(std::begin(ok_types), std::end(ok_types), ty) != std::end(ok_types);
        if (chain_) {
            lchain_ = cx_->compile({"quant.metal", "units.metal", std::string(STRATA_METAL_ENGINE_KERNELS_DEFAULT) + "/mtp_chain.metal"});
            dout_ = A(4 * MAXT); dprob_ = A(4 * MAXT); posv_ = A(4 * N);
            for (int i = 0; i < N; ++i) posv_.as<float>()[i] = (float) i;
            dmap_ = A(4 * std::max<size_t>(1, dvocab_.size()));
            if (!dvocab_.empty()) std::memcpy(dmap_.as<int32_t>(), dvocab_.data(), 4 * dvocab_.size());
            // parity: a few tokens' embeddings, GPU vs the host's dequant (the per-step path's)
            const gguf::Tensor& te = T("token_embd.weight");
            const size_t rb = ref::row_bytes(te.type, N);
            const int32_t probe[MAXT] = {0, 1, 271, 760, 6511, 11751, 100000, (int32_t) n_vocab_ - 1};
            double worst = 0.0;
            std::vector<float> ref_row(N);
            for (int k = 0; k < MAXT; ++k) {
                dout_.as<int32_t>()[0] = probe[k];
                std::vector<mtl::Dispatch> ds;
                nb_ = false;
                mtl::Dispatch d;
                d.p = P(lchain_, "mtp_embed_" + std::to_string(ty), {}, "mtp_embed_" + std::to_string(ty));
                struct EmbArgs { uint32_t n, row_bytes, j; } ea{(uint32_t) N, (uint32_t) rb, 0};
                d.args = {mtl::Arg::B(W_, te.offset), mtl::Arg::B(dout_), mtl::Arg::B(posv_), mtl::Arg::B(memb_), mtl::Arg::V(ea)};
                d.tg[0] = 256;
                push(ds, d);
                cx_->run_graph(ds);
                ref::dequant(te.type, (const uint8_t*) f_.map + te.offset + (size_t) probe[k] * rb, ref_row.data(), N);
                for (int i = 0; i < N; ++i) worst = std::max(worst, (double) std::fabs(ref_row[i] - memb_.as<float>()[i]));
            }
            std::fprintf(stderr, "strata metal: MTP draft round in one command buffer (embedding type %d on the GPU: max "
                                 "|GPU - host| %.3g over %d rows)\n", ty, worst, MAXT);
            if (worst > 1e-6) {
                std::fprintf(stderr, "strata metal: the GPU embedding differs from the host's: a command buffer per step\n");
                chain_ = false;
            }
        }
    }
    std::fprintf(stderr, "strata metal: MTP draft layer from %s (%.2f GB, %zu norms folded), draft head over %s tokens\n",
                 gguf_path.c_str(), fm_.file_bytes / 1e9, mnorm_off_.size(),
                 dvocab_.empty() ? "all" : std::to_string(dvocab_.size()).c_str());
    return true;
}

void Window::gr_read_m(std::vector<mtl::Dispatch>& ds, const std::string& pre, mtl::Buffer& Rb, const mtl::Buffer& bo,
                       const mtl::Buffer& inj_in, const mtl::Buffer& inj_out, bool apply, const Act& out, bool final_,
                       int ntok) {
    const gguf::Tensor& tdn = TQ(pre + ".input_mix_weight_down.weight");
    const gguf::Tensor& tup = TQ(pre + ".input_mix_weight_up.weight");
    const gguf::Tensor& tin = final_ ? tdn : TQ(pre + ".block_inject_weight.weight");
    const bool q8 = tdn.type == ref::Q8_0 && tup.type == ref::Q8_0 && tin.type == ref::Q8_0;
    const mtl::Buffer& WB = q8 ? mq8_ : Wm_;
    GrArgs a{N, HC, LR, (uint32_t) ntok, apply ? 1u : 0u, eps_};
    mtl::Dispatch d1;
    d1.p = P(lib_, "gr_norm", {}, "gr_norm");
    d1.args = {mtl::Arg::B(Rb), mtl::Arg::B(bo), mtl::Arg::B(inj_in), mtl::Arg::B(mnorm_, NM(pre + ".hc_norm.weight")),
               mtl::Arg::B(xn_), mtl::Arg::V(a)};
    d1.grid[0] = HC; d1.grid[1] = ntok; d1.tg[0] = 256;
    push(ds, d1);
    mtl::Dispatch d2;
    const char* dn = q8 ? "gr_down_q8_8_8" : "gr_down_8_8";
    d2.p = P(lib_, dn, mtl::Constants().i(1, ntok), std::string(dn) + "#" + std::to_string(ntok));
    d2.args = {mtl::Arg::B(WB, tdn.offset), mtl::Arg::B(WB, tin.offset), mtl::Arg::B(xn_), mtl::Arg::B(lo_),
               mtl::Arg::B(inj_out), mtl::Arg::V(a)};
    d2.grid[0] = LR + HC; d2.tg[0] = 256;
    push(ds, d2);
    mtl::Dispatch d3;
    const char* up = q8 ? "gr_up_q8" : "gr_up";
    d3.p = P(lib_, up, mtl::Constants().i(1, ntok), std::string(up) + "#" + std::to_string(ntok));
    GrArgs au = a;
    au.n_tok = MAXT;
    d3.args = {mtl::Arg::B(WB, tup.offset), mtl::Arg::B(lo_), mtl::Arg::B(xn_), mtl::Arg::B(out.b), mtl::Arg::V(au)};
    d3.grid[0] = N / 16; d3.tg[0] = 256;
    push(ds, d3);
}

void Window::mtp_encode(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& srcR, size_t src_off, int n, const int32_t* tokens,
                        int64_t cell0, bool full, int erow, int slot) {
    struct RmsArgs { uint32_t n; float eps; };
    struct XpArgs { uint32_t n, n_tok, plane_tok, stride; };
    struct CombArgs { uint32_t n, hc, n_tok, h_stride; };
    // ---- the embedding rows (host tokens: dequantized on the CPU into the shared buffer)
    const gguf::Tensor& te = T("token_embd.weight");
    const size_t rb = ref::row_bytes(te.type, N);
    if (tokens)
        for (int i = 0; i < n; ++i)
            ref::dequant(te.type, (const uint8_t*) f_.map + te.offset + (size_t) tokens[i] * rb,
                         memb_.as<float>() + (size_t) (erow + i) * N, N);
    const std::string L = "mtp.layers.0.";
    auto rms = [&](const mtl::Buffer& x, size_t xo, const std::string& w, mtl::Buffer& y, int width, int rows) {
        mtl::Dispatch d;
        d.p = P(lmtp_, "mtp_rms", {}, "mtp_rms");
        RmsArgs a{(uint32_t) width, eps_};
        d.args = {mtl::Arg::B(x, xo), mtl::Arg::B(mnorm_, NM(w)), mtl::Arg::B(y), mtl::Arg::V(a)};
        d.grid[0] = rows; d.tg[0] = 256;
        push(ds, d);
    };
    auto xp = [&](const mtl::Buffer& x, size_t xo, const Act& out, int width, int rows, int stride) {
        mtl::Dispatch d;
        d.p = P(lmtp_, "mtp_xprep", {}, "mtp_xprep");
        XpArgs a{(uint32_t) width, (uint32_t) rows, MAXT, (uint32_t) stride};
        d.args = {mtl::Arg::B(x, xo), mtl::Arg::B(out.b), mtl::Arg::V(a)};
        d.grid[0] = (width / 16 * rows + 255) / 256; d.tg[0] = 256;
        push(ds, d);
    };
    auto gm = [&](const std::string& name, const Act& x, const mtl::Buffer& y, size_t y_stride, int rows, size_t y_off = 0) {
        // a GEMV on the MTP model's tensor (the window's gemv() reads the main model's buffer)
        const gguf::Tensor& tt = TQ(name);
        const int n_in = (int) tt.shape[0], n_out = tt.shape.size() > 1 ? (int) tt.shape[1] : 1, ty = tt.type;
        int Lq = 16, Rr = 1, S = 4, K = 1;
        if (n_in >= 6144) { Lq = 32; Rr = 1; S = 8; K = 2; }
        else if (n_in == 640) { Lq = 8; Rr = 1; S = 4; K = 1; }
        else if (n_out <= 640 || ty == ref::Q2_0 || ty == ref::BF16) { Lq = 32; Rr = 1; S = 4; K = 1; }
        const std::string fn = "gemv_" + std::to_string(ty) + "_" + std::to_string(Lq) + "_" + std::to_string(Rr) + "_" +
                               std::to_string(S) + "_" + std::to_string(K);
        GemvArgs a{(uint32_t) n_in, (uint32_t) n_out, (uint32_t) rows, (uint32_t) ref::row_bytes(ty, n_in), (uint32_t) n_in,
                   (uint32_t) y_stride};
        mtl::Dispatch d;
        d.p = P(lib_, fn, mtl::Constants().i(0, rows), fn + "#" + std::to_string(rows));
        d.args = {mtl::Arg::B(MB(name), tt.offset), mtl::Arg::B(x.b, x.plane_bytes() * xkind(ty)),
                  mtl::Arg::B(x.b, x.plane_bytes() * 4), mtl::Arg::B(y, y_off), mtl::Arg::V(a)};
        const int rpt = (S / K) * (32 / Lq) * Rr;
        d.grid[0] = (n_out + rpt - 1) / rpt; d.tg[0] = S * 32;
        push(ds, d);
    };
    // ---- e = fc_embedding(rms(embed(tok)) * (1 + w)),  h = fc_hidden per stream (rms_10240(R) * (1 + w)),  R = h + e
    rms(memb_, (size_t) erow * N * 4, "mtp.pre_fc_norm_embedding.weight", men_, N, n);
    xp(men_, 0, XEn_, N, n, N);
    gm("mtp.fc_embedding.weight", XEn_, me2_, N, n);
    rms(srcR, src_off, "mtp.pre_fc_norm_hidden.weight", mhn_, HC * N, n);
    for (int c = 0; c < HC; ++c) {
        xp(mhn_, (size_t) c * N * 4, XHn_, N, n, HC * N);
        gm("mtp.fc_hidden.weight", XHn_, mh2_, N, n, (size_t) c * MAXT * N * 4);
    }
    {
        mtl::Dispatch d;
        d.p = P(lmtp_, "mtp_combine", {}, "mtp_combine");
        CombArgs a{N, HC, (uint32_t) n, (uint32_t) (MAXT * N)};
        d.args = {mtl::Arg::B(mh2_), mtl::Arg::B(me2_), mtl::Arg::B(mR_), mtl::Arg::V(a)};
        d.grid[0] = (n * HC * N + 255) / 256; d.tg[0] = 256;
        push(ds, d);
    }
    // ---- the attention hyper-connection, K/V (and the indexer keys) into the layer's own cache
    gr_read_m(ds, L + "attn_hyper_connection", mR_, mbo_, minj_[1], minj_[0], false, XA_, false, n);
    gm(L + "self_attn.k_proj.weight", XA_, kcur_, 512, n);
    nb_ = true;
    gm(L + "self_attn.v_proj.weight", XA_, vcur_, 512, n);
    nb_ = true;
    gm(L + "self_attn.indexer.index_qk_proj.weight", XA_, midx_, 640, n);
    if (full) {
        nb_ = true;
        gm(L + "self_attn.q_proj.weight", XA_, qfull_, 12288, n);
    }
    QsaArgs qa{(uint32_t) cell0, 0, (uint32_t) max_cells_, (uint32_t) n, eps_};
    {
        mtl::Dispatch d;
        d.p = P(lqsa_, "qsa_prep", {}, "qsa_prep");
        d.args = {mtl::Arg::B(qfull_), mtl::Arg::B(kcur_), mtl::Arg::B(vcur_), mtl::Arg::B(midx_), mtl::Arg::B(midx_, 512 * 4),
                  mtl::Arg::B(mnorm_, NM(L + "self_attn.q_norm.weight")), mtl::Arg::B(mnorm_, NM(L + "self_attn.k_norm.weight")),
                  mtl::Arg::B(mnorm_, NM(L + "self_attn.indexer.q_layernorm.weight")),
                  mtl::Arg::B(mnorm_, NM(L + "self_attn.indexer.k_layernorm.weight")),
                  mtl::Arg::B(cs_), mtl::Arg::B(qcur_), mtl::Arg::B(qidx_), mtl::Arg::B(mkq_[0]), mtl::Arg::B(mvq_[0]),
                  mtl::Arg::B(mksc_[0]), mtl::Arg::B(mvsc_[0]), mtl::Arg::B(mtail_[0]), mtl::Arg::B(mdead_[0]),
                  mtl::Arg::B(mpooled_[0]), mtl::Arg::V(qa)};
        d.grid[0] = 33; d.grid[1] = n; d.tg[0] = 256;
        push(ds, d);
    }
    if (!full) return;
    // ---- dense attention over the window of cells up to this one (the draft layer's rule), the output gate, o_proj
    {
        int* ids = mqids_.as<int>() + (size_t) slot * 2051;
        const int64_t first = std::max<int64_t>(0, cell0 - 2050);
        for (int64_t c = first; c <= cell0; ++c) ids[c - first] = (int) c;
        mtl::Dispatch at;
        at.p = P(lqsa_, "qsa_attn", {}, "qsa_attn");
        at.args = {mtl::Arg::B(qcur_), mtl::Arg::B(mkq_[0]), mtl::Arg::B(mvq_[0]), mtl::Arg::B(mksc_[0]), mtl::Arg::B(mvsc_[0]),
                   mtl::Arg::B(mqids_, (size_t) slot * 2051 * 4), mtl::Arg::B(part_), mtl::Arg::V(qa)};
        at.grid[0] = 33; at.grid[1] = 2; at.grid[2] = 1; at.tg[0] = 256;
        push(ds, at);
        mtl::Dispatch m;
        m.p = P(lqsa_, "qsa_merge", {}, "qsa_merge");
        QsaArgs am = qa;
        am.n_tok = MAXT;
        m.args = {mtl::Arg::B(part_), mtl::Arg::B(qfull_), mtl::Arg::B(XQ_.b), mtl::Arg::V(am)};
        m.grid[0] = 24; m.grid[1] = 1; m.tg[0] = 256;
        push(ds, m);
        gm(L + "self_attn.o_proj.weight", XQ_, mbo_, N, 1);
    }
    // ---- the MLP hyper-connection (the attention write folded in), the MoE, the write + final mixer, the head
    gr_read_m(ds, L + "mlp_hyper_connection", mR_, mbo_, minj_[0], minj_[1], true, XF_, false, 1);
    gm(L + "mlp.gate.weight", XF_, logits_, 512, 1);
    nb_ = true;
    gm(L + "mlp.shared_expert.gate_proj.weight", XF_, gsh_, NFF, 1);
    nb_ = true;
    gm(L + "mlp.shared_expert.up_proj.weight", XF_, ush_, NFF, 1);
    nb_ = true;
    gm(L + "mlp.shared_expert_gate.weight", XF_, sg_, 1, 1);
    {
        MoeArgs a{};
        const gguf::Tensor &tgu = TM(L + "mlp.experts.gate_up_proj"), &td = TM(L + "mlp.experts.down_proj");
        a.gu_row = (uint32_t) ref::row_bytes(tgu.type, N);
        a.d_row = (uint32_t) ref::row_bytes(td.type, NFF);
        a.gate_off = tgu.offset;
        a.up_off = tgu.offset + (uint64_t) NFF * a.gu_row;   // [gate rows | up rows] per expert
        a.down_off = td.offset;
        a.gu_stride = (uint64_t) a.gu_row * 2 * NFF;
        a.d_stride = (uint64_t) a.d_row * N;
        a.n_embd = N; a.n_ff = NFF; a.n_tok = 1; a.topk = TOPK; a.n_expert = NEXP;
        const bool dq2 = td.type == ref::Q2_0;
        const int gl = 32, gr = 1, gs = 4, dl = dq2 ? 8 : 4, dr = dq2 ? 1 : 2, ds_ = 4;
        a.gx_gu = NFF / (gs * (32 / gl) * gr);
        a.gx_dn = N / (ds_ * (32 / dl) * dr);
        mtl::Dispatch r;
        r.p = P(lib_, "moe_route", {}, "moe_route");
        r.args = {mtl::Arg::B(logits_), mtl::Arg::B(ids_), mtl::Arg::B(wts_), mtl::Arg::B(plan_), mtl::Arg::V(a)};
        r.tg[0] = 512;
        push(ds, r);
        {
            GlueArgs ga{NFF, MAXT, N, TOPK};
            mtl::Dispatch d;
            d.p = P(lib_, "swiglu_prep", {}, "swiglu_prep");
            d.args = {mtl::Arg::B(gsh_), mtl::Arg::B(ush_), mtl::Arg::B(XH_.b), mtl::Arg::V(ga)};
            d.grid[0] = NFF / 16; d.grid[1] = 1; d.tg[0] = 16;
            nb_ = true;
            push(ds, d);
        }
        const std::string gun = "moe_gu_" + std::to_string(tgu.type) + "_" + std::to_string(gl) + "_" + std::to_string(gr) +
                                "_" + std::to_string(gs);
        const std::string dnn = "moe_down_" + std::to_string(dq2 ? 142 : td.type) + "_" + std::to_string(dl) + "_" +
                                std::to_string(dr) + "_" + std::to_string(ds_);
        mtl::Dispatch g;
        g.p = P(lib_, gun, {}, gun);
        g.args = {mtl::Arg::B(Wm_), mtl::Arg::B(XF_.b, XF_.plane_bytes() * xkind(tgu.type)), mtl::Arg::B(h_), mtl::Arg::B(plan_),
                  mtl::Arg::V(a), mtl::Arg::B(XF_.b, XF_.plane_bytes() * 4)};
        g.tg[0] = 32 * gs;
        g.indirect = &plan_;
        g.ind_off = P_IGU * 4;
        push(ds, g);
        nb_ = true;
        gm(L + "mlp.shared_expert.down_proj.weight", XH_, ysh_, N, 1);
        mtl::Dispatch dd;
        dd.p = P(lib_, dnn, {}, dnn);
        dd.args = {mtl::Arg::B(Wm_), mtl::Arg::B(h_), mtl::Arg::B(ye_), mtl::Arg::B(plan_), mtl::Arg::V(a), mtl::Arg::B(h_)};
        dd.tg[0] = 32 * ds_;
        dd.indirect = &plan_;
        dd.ind_off = P_IDN * 4;
        push(ds, dd);
        GlueArgs ca{N, 1, N, TOPK};
        mtl::Dispatch cb;
        cb.p = P(lib_, "moe_combine_sh", {}, "moe_combine_sh");
        cb.args = {mtl::Arg::B(ye_), mtl::Arg::B(plan_), mtl::Arg::B(ysh_), mtl::Arg::B(sg_), mtl::Arg::B(mbo_), mtl::Arg::V(ca)};
        cb.grid[0] = (N + 255) / 256; cb.tg[0] = 256;
        push(ds, cb);
    }
    // the layer's write (R updated in place: the next draft step's input residual) and the final mixer
    gr_read_m(ds, "mtp.hyper_connection_mixer", mR_, mbo_, minj_[1], minj_[0], true, XA_, true, 1);
    const int64_t nv = dvocab_.empty() ? n_vocab_ : (int64_t) dvocab_.size();
    if (dvocab_.empty()) gemv(ds, T("output.weight"), XA_, mlm_, (size_t) n_vocab_, 1);
    else gemv(ds, mhead_t_, XA_, mlm_, (size_t) nv, 1, 0, &mhead_);
    {
        ArgArgs a{(uint32_t) nv, 120};
        mtl::Dispatch d1, d2;
        d1.p = P(lsmp_, "argmax_part", {}, "argmax_part");
        d1.args = {mtl::Arg::B(mlm_), mtl::Arg::B(apv_), mtl::Arg::B(api_), mtl::Arg::V(a)};
        d1.grid[0] = 120; d1.grid[1] = 1; d1.tg[0] = 256;
        push(ds, d1);
        d2.p = P(lsmp_, "argmax_final", {}, "argmax_final");
        d2.args = {mtl::Arg::B(apv_), mtl::Arg::B(api_), mtl::Arg::B(tokout_), mtl::Arg::V(a)};
        d2.grid[0] = 1; d2.tg[0] = 256;
        push(ds, d2);
    }
}

// the prompt's K/V: cell cell0 + i pairs the main window's final residual row i with the token at cell0 + i + 1
bool Window::mtp_prefill(const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err) {
    if (!mtp_on_) { err = "metal mtp: not loaded"; return false; }
    for (int64_t c = 0; c < n; c += MAXT) {
        const int k = (int) std::min<int64_t>(MAXT, n - c);
        std::vector<mtl::Dispatch> ds;
        nb_ = false;
        mtp_encode(ds, R_, (size_t) c * HC * N * 4, k, next_tokens + c, cell0 + c, false);
        cx_->run_graph(ds);
    }
    return true;
}

void Window::mtp_pick(std::vector<mtl::Dispatch>& ds, int j, bool embed) {
    struct PickArgs { uint32_t nv, j, use_map; };
    const int64_t nv = dvocab_.empty() ? n_vocab_ : (int64_t) dvocab_.size();
    mtl::Dispatch d;
    d.p = P(lchain_, "mtp_pick", {}, "mtp_pick");
    PickArgs pa{(uint32_t) nv, (uint32_t) j, dvocab_.empty() ? 0u : 1u};
    d.args = {mtl::Arg::B(mlm_), mtl::Arg::B(tokout_), mtl::Arg::B(dmap_), mtl::Arg::B(dout_), mtl::Arg::B(dprob_),
              mtl::Arg::V(pa)};
    d.tg[0] = 1024;
    push(ds, d);
    if (!embed) return;
    const gguf::Tensor& te = T("token_embd.weight");
    struct EmbArgs { uint32_t n, row_bytes, j; } ea{(uint32_t) N, (uint32_t) ref::row_bytes(te.type, N), (uint32_t) j};
    mtl::Dispatch e;
    const std::string fn = "mtp_embed_" + std::to_string(te.type);
    e.p = P(lchain_, fn, {}, fn);
    e.args = {mtl::Arg::B(W_, te.offset), mtl::Arg::B(dout_), mtl::Arg::B(posv_), mtl::Arg::B(memb_), mtl::Arg::V(ea)};
    e.tg[0] = 256;
    push(ds, e);
}

// One round (mtp.hpp's draft()): the window's rows 0..a-1 caught up (K/V), then the full layer on row a (draft 0) and
// on its own output for the next drafts.  By default the whole round is one command buffer (the pick, its
// probability and the next step's embedding on the GPU); STRATA_METAL_MTP_CHAIN=0: a command buffer per step, the
// host picking and dequantizing.  The embedding rows: the catch-up's at memb_ rows 0..a-1, draft 0's at row a.
bool Window::mtp_draft(int Tn, const int32_t* tokens, int64_t p, int a, int max_steps, int32_t* drafts, float* probs,
                       float min_p, int* n_drafts, std::string& err) {
    if (!mtp_on_) { err = "metal mtp: not loaded"; return false; }
    (void) Tn;
    int n = 0;
    int32_t tok = tokens[a];
    if (chain_ && max_steps > 0) {
        std::vector<mtl::Dispatch> ds;
        nb_ = false;
        if (a > 0) mtp_encode(ds, R_, 0, a, tokens, p, false);
        mtp_encode(ds, R_, (size_t) a * HC * N * 4, 1, &tok, p + a, true, a, 0);
        mtp_pick(ds, 0, max_steps > 1);
        for (int j = 1; j < max_steps; ++j) {
            mtp_encode(ds, mR_, 0, 1, nullptr, p + a + j, true, 0, j);
            mtp_pick(ds, j, j + 1 < max_steps);
        }
        cx_->run_graph(ds);
        for (int j = 0; j < max_steps; ++j) {
            const float pj = dprob_.as<float>()[j];
            drafts[j] = dout_.as<int32_t>()[j];
            if (probs) probs[j] = pj;
            ++n;
            if (min_p > 0.0f && pj < min_p) break;
        }
        if (n_drafts) *n_drafts = n;
        return true;
    }
    for (int j = 0; j < max_steps; ++j) {
        std::vector<mtl::Dispatch> ds;
        nb_ = false;
        if (j == 0) {
            if (a > 0) mtp_encode(ds, R_, 0, a, tokens, p, false);
            mtp_encode(ds, R_, (size_t) a * HC * N * 4, 1, &tok, p + a, true, a, 0);
        } else {
            mtp_encode(ds, mR_, 0, 1, &tok, p + a + j, true);
        }
        cx_->run_graph(ds);
        const int32_t di = tokout_.as<int32_t>()[0];
        const int32_t d = dvocab_.empty() ? di : dvocab_[(size_t) di];
        // the draft's probability: softmax of the draft head's logits at the pick
        const float* lg = mlm_.as<float>();
        const float mx = lg[di];
        const int64_t nv = dvocab_.empty() ? n_vocab_ : (int64_t) dvocab_.size();
        double s = 0.0;
        for (int64_t i = 0; i < nv; ++i) s += std::exp((double) (lg[i] - mx));
        const float pj = (float) (1.0 / s);
        drafts[j] = d;
        if (probs) probs[j] = pj;
        tok = d;
        ++n;
        if (min_p > 0.0f && pj < min_p) break;
    }
    if (n_drafts) *n_drafts = n;
    return true;
}

bool Window::copy_logits(int t, float* host) const {
    if (t < 0 || t >= last_T_) return false;
    std::memcpy(host, lm_.as<float>() + (size_t) t * n_vocab_, (size_t) n_vocab_ * 4);
    return true;
}

// ---------------------------------------------------------------------------------------------- the hook's entry points
namespace {
std::string g_shard1;
}
void window_register(void* w);
bool mtp_load_pending(Window* w, std::string& err);
void set_model_shard(const char* shard1) { g_shard1 = shard1 ? shard1 : ""; }

// One decode window per process: the prompt path (Prefill::init) may bring it up before the Verifier does.
namespace {
Window* g_the_window = nullptr;
}
void* window_create() {
    if (g_the_window == nullptr) g_the_window = new Window();
    return g_the_window;
}
void window_destroy(void* w) {
    if (w != nullptr && w == g_the_window) {
        delete g_the_window;
        g_the_window = nullptr;
        window_register(nullptr);
    }
}
bool window_init(void* w, const strata::core::ModelGeometry& g, strata::core::SessionState& ss, int max_t, int64_t max_cells,
                 int64_t* n_vocab, std::string& err) {
    Window* x = (Window*) w;
    if (!x->ready()) {
        if (g_shard1.empty()) { err = "metal window: the model shard is not set (--native SHARD1)"; return false; }
        if (!x->init(g, ss, g_shard1, MAXT, max_cells, err)) return false;
        window_register(w);
        if (!mtp_load_pending(x, err)) return false;
    }
    x->set_max_t(max_t);
    *n_vocab = x->vocab();
    return true;
}
bool window_ensure(const strata::core::ModelGeometry& g, strata::core::SessionState& ss, int64_t max_cells, std::string& err) {
    int64_t nv = 0;
    return window_init(window_create(), g, ss, MAXT, max_cells, &nv, err);
}
bool window_run(void* w, int T, const int32_t* tokens, int64_t pos0, bool head, int32_t* out, std::string& err) {
    return ((Window*) w)->run(T, tokens, pos0, head, out, err);
}
bool window_commit(void* w, int n_keep, std::string& err) {
    const double t0 = now_ms();
    const bool ok = ((Window*) w)->commit(n_keep, err);
    ((Window*) w)->prof_add(false, now_ms() - t0);
    return ok;
}
bool window_copy_logits(void* w, int t, float* host) { return ((Window*) w)->copy_logits(t, host); }

// the MTP hook (src/core/mtp.cpp under STRATA_GPU_METAL): the drafter lives beside the decode window it reads
namespace {
Window* g_window = nullptr;
}
void window_register(void* w) { g_window = (Window*) w; }
// the engine loads the drafter before the decode window exists: the path is kept and the window loads it at its init
std::string g_mtp_path;
int g_mtp_max_t = 0;
bool mtp_load(const std::string& rt_dir, int max_t, std::string& err) {
    (void) err;
    // the MTP model in GGUF form (tools/mtp_pack.py) beside the rt directory: STRATA_METAL_MTP_GGUF, else
    // <rt>/../mtp-q2_0.gguf
    std::string path;
    if (const char* e = std::getenv("STRATA_METAL_MTP_GGUF")) path = e;
    else {
        std::string d = rt_dir;
        while (!d.empty() && d.back() == '/') d.pop_back();
        const size_t s = d.rfind('/');
        path = (s == std::string::npos ? std::string(".") : d.substr(0, s)) + "/mtp-q2_0.gguf";
    }
    g_mtp_path = path;
    g_mtp_max_t = max_t;
    return true;
}
bool mtp_load_pending(Window* w, std::string& err) {
    return g_mtp_path.empty() || w->mtp_load(g_mtp_path, g_mtp_max_t, err);
}
// Prefill::run's Metal side: the prompt read through the window, MAXT rows at a time (no head), each window
// committed; on_chunk (the MTP's K/V) after each, with the window's final residual rows in place.
bool prompt_run(const int64_t* tokens, int64_t n, int64_t pos0,
                const std::function<bool(const float*, int64_t, int64_t, std::string&)>& on_chunk, std::string& err) {
    if (g_window == nullptr) { err = "metal prompt: the decode window is not up yet"; return false; }
    int32_t tk[MAXT], out[MAXT];
    for (int64_t c = 0; c < n; c += MAXT) {
        const int k = (int) std::min<int64_t>(MAXT, n - c);
        for (int i = 0; i < k; ++i) tk[i] = (int32_t) tokens[c + i];
        if (!g_window->run(k, tk, pos0 + c, false, out, err) || !g_window->commit(k, err)) return false;
        if (on_chunk && !on_chunk(nullptr, k, pos0 + c, err)) return false;
    }
    return true;
}
bool mtp_prefill(const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err) {
    if (g_window == nullptr) { err = "metal mtp: no decode window"; return false; }
    return g_window->mtp_prefill(next_tokens, n, cell0, err);
}
bool mtp_draft(int T, const int32_t* tokens, int64_t p, int a, int max_steps, int32_t* drafts, float* probs, float min_p,
               int* n_drafts, std::string& err) {
    if (g_window == nullptr) { err = "metal mtp: no decode window"; return false; }
    const double t0 = now_ms();
    const bool ok = g_window->mtp_draft(T, tokens, p, a, max_steps, drafts, probs, min_p, n_drafts, err);
    g_window->prof_add(true, now_ms() - t0);
    return ok;
}

}  // namespace strata::metal_engine
