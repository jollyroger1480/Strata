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
#include <map>
#include <memory>
#include <string>
#include <sys/mman.h>
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
    bool copy_logits(int t, float* host) const;
    int64_t vocab() const { return n_vocab_; }

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
              int ntok, size_t y_off = 0);
    void xprep(std::vector<mtl::Dispatch>& ds, const mtl::Buffer& x, const Act& out, int n);
    void gr_read(std::vector<mtl::Dispatch>& ds, const std::string& pre, bool apply, const Act& out, bool final_, int ntok);
    void layer(std::vector<mtl::Dispatch>& ds, int l, int ntok, int64_t pos0);
    void ple_cpu(int T);

    strata::core::SessionState* ss_ = nullptr;
    gguf::File f_;
    std::unique_ptr<mtl::Context> cx_;
    mtl::Buffer W_;
    int lib_ = -1, lgdn_ = -1, lqsa_ = -1, lsmp_ = -1;
    std::map<std::string, mtl::Pipeline> pcache_;
    bool nb_ = false;
    int max_t_ = 1;
    int64_t max_cells_ = 0;
    int64_t n_vocab_ = 0;
    float eps_ = 1e-6f;
    // buffers
    mtl::Buffer R_, bo_, inj_[2], xn_, lo_, qkv_, z_, al_, be_, yg_, logits_, ids_, wts_, plan_, h_, ye_, gsh_, ush_,
        ysh_, sg_, qfull_, kcur_, vcur_, iqr_, ikr_, qcur_, qidx_, scores_, qids_, part_, cs_, lm_, apv_, api_, tokout_,
        emb_, pkey_, pval_, qkvL_, alL_, beL_, zL_, ydump_;
    Act XA_, XF_, XY_, XQ_, XH_, XE_;
    int cur_inj_ = 0;
    // state
    std::vector<mtl::Buffer> gst_, ghist_[2], kq_, vq_, ksc_, vsc_, tail_, dead_, pooled_;
    std::vector<int> hpar_;          // per GDN layer: which history buffer holds the committed history
    std::vector<float> ple_hist_;    // NG_HIST x hc_dim normalized rows, row-fastest
    bool ple_on_ = false;
    int last_T_ = 0;
    int32_t last_tokens_[MAXT] = {};
    std::vector<float> last_norm_;   // the window's normalized PLE rows (T x hc_dim), for commit
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
            hpar_.push_back(0);
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
    ple_on_ = ss.ple.ready() && f_.find("blk.1.ple_key.weight") != nullptr;
    ple_hist_.assign((size_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM, 0.0f);
    last_norm_.assign((size_t) MAXT * strata::kernels::NG_HC_DIM, 0.0f);
    std::fprintf(stderr, "strata metal: decode window on %s: %s wrapped (%.1f GB), %d cells, PLE %s, ready in %.0f ms\n",
                 cx_->device_name().c_str(), shard1.c_str(), f_.file_bytes / 1e9, MAXC, ple_on_ ? "on" : "OFF",
                 now_ms() - t0);
    return true;
}

void Window::gemv(std::vector<mtl::Dispatch>& ds, const gguf::Tensor& t, const Act& x, const mtl::Buffer& y,
                  size_t y_stride, int ntok, size_t y_off) {
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
    d.args = {mtl::Arg::B(W_, t.offset), mtl::Arg::B(x.b, x.plane_bytes() * xkind(ty)), mtl::Arg::B(x.b, x.plane_bytes() * 4),
              mtl::Arg::B(y, y_off), mtl::Arg::V(a)};
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
    mtl::Dispatch d2;
    d2.p = P(lib_, "gr_down_8_8", mtl::Constants().i(1, ntok), "gr_down_8_8#" + std::to_string(ntok));
    d2.args = {mtl::Arg::B(W_, tdn.offset), mtl::Arg::B(W_, tin.offset), mtl::Arg::B(xn_), mtl::Arg::B(lo_),
               mtl::Arg::B(inj_[1 - cur_inj_]), mtl::Arg::V(a)};
    d2.grid[0] = LR + HC;
    d2.tg[0] = 256;
    push(ds, d2);
    mtl::Dispatch d3;
    d3.p = P(lib_, "gr_up", mtl::Constants().i(1, ntok), "gr_up#" + std::to_string(ntok));
    GrArgs au = a;
    au.n_tok = MAXT;   // the plane stride of the prepared activation
    d3.args = {mtl::Arg::B(W_, tup.offset), mtl::Arg::B(lo_), mtl::Arg::B(xn_), mtl::Arg::B(out.b), mtl::Arg::V(au)};
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
        const int hp = hpar_[(size_t) gi];
        d.args = {mtl::Arg::B(gst_[(size_t) gi]), mtl::Arg::B(ghist_[hp][(size_t) gi]), mtl::Arg::B(qkvL_, o_qkv),
                  mtl::Arg::B(alL_, o_ab), mtl::Arg::B(beL_, o_ab), mtl::Arg::B(zL_, o_z),
                  mtl::Arg::B(W_, T(b + "ssm_conv1d.weight").offset),
                  mtl::Arg::B(W_, T(b + "ssm_dt.bias").offset), mtl::Arg::B(W_, T(b + "ssm_a").offset),
                  mtl::Arg::B(W_, T(b + "ssm_norm.weight").offset), mtl::Arg::B(yg_), mtl::Arg::V(a),
                  mtl::Arg::B(ghist_[1 - hp][(size_t) gi])};
        d.grid[0] = 48;
        d.tg[0] = 512;
        push(ds, d);
        if (self_commit) hpar_[(size_t) gi] = 1 - hp;   // the history ping-pong turns with every commit
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
        mtl::Dispatch s;
        s.p = P(lqsa_, "qsa_score", {}, "qsa_score");
        s.args = {mtl::Arg::B(qidx_), mtl::Arg::B(pooled_[(size_t) qi]), mtl::Arg::B(dead_[(size_t) qi]), mtl::Arg::B(scores_),
                  mtl::Arg::V(a)};
        s.grid[0] = (uint32_t) (((pos0 + ntok) / 4 + 1 + 7) / 8);
        s.grid[1] = ntok;
        s.tg[0] = 256;
        push(ds, s);
        mtl::Dispatch k;
        k.p = P(lqsa_, "qsa_topk2", {}, "qsa_topk");
        k.args = {mtl::Arg::B(scores_), mtl::Arg::B(qids_), mtl::Arg::V(a)};
        k.grid[0] = ntok;
        k.tg[0] = 1024;
        push(ds, k);
        mtl::Dispatch at;
        at.p = P(lqsa_, "qsa_attn", {}, "qsa_attn");
        at.args = {mtl::Arg::B(qcur_), mtl::Arg::B(kq_[(size_t) qi]), mtl::Arg::B(vq_[(size_t) qi]), mtl::Arg::B(ksc_[(size_t) qi]),
                   mtl::Arg::B(vsc_[(size_t) qi]), mtl::Arg::B(qids_), mtl::Arg::B(part_), mtl::Arg::V(a)};
        at.grid[0] = 33;
        at.grid[1] = 2;
        at.grid[2] = ntok;
        at.tg[0] = 256;
        push(ds, at);
        mtl::Dispatch m;
        m.p = P(lqsa_, "qsa_merge", {}, "qsa_merge");
        QsaArgs am = a;
        am.n_tok = MAXT;   // the plane stride
        m.args = {mtl::Arg::B(part_), mtl::Arg::B(qfull_), mtl::Arg::B(XQ_.b), mtl::Arg::V(am)};
        m.grid[0] = 24;
        m.grid[1] = ntok;
        m.tg[0] = 256;
        push(ds, m);
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
    (void) head;
    if (Tn < 1 || Tn > max_t_) { err = "metal window: window size out of range"; return false; }
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
    // ---- the PLE rows (the table read is the shared host code's), projected on the GPU in the first command buffer
    std::vector<mtl::Dispatch> ds;
    nb_ = false;
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
        cx_->run_graph(ds);
        ple_cpu(Tn);
        ds.clear();
        nb_ = false;
    }
    for (int l = 1; l < 48; ++l) layer(ds, l, Tn, pos0);
    // ---- the final mixer and the head
    gr_read(ds, "output_hc", true, XA_, true, Tn);
    gemv(ds, T("output.weight"), XA_, lm_, (size_t) n_vocab_, Tn);
    {
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
    cx_->run_graph(ds);
    for (int t = 0; t < Tn; ++t) out[t] = tokout_.as<int32_t>()[t];
    last_T_ = Tn;
    for (int t = 0; t < Tn; ++t) last_tokens_[t] = tokens[t];
    static const bool trace = std::getenv("STRATA_METAL_TRACE") != nullptr;
    if (trace) {
        // the pick and its margin over the runner-up (a near tie says a backend difference may flip it)
        const float* lg = lm_.as<float>();
        int i1 = -1, i2 = -1;
        for (int i = 0; i < n_vocab_; ++i) {
            if (i1 < 0 || lg[i] > lg[i1]) { i2 = i1; i1 = i; }
            else if (i2 < 0 || lg[i] > lg[i2]) i2 = i;
        }
        std::fprintf(stderr, "strata metal: window pos %lld token %d -> %d (top %d %.4f, 2nd %d %.4f, margin %.4f) %.2f ms\n",
                     (long long) pos0, tokens[0], out[0], i1, lg[i1], i2, lg[i2], lg[i1] - lg[i2], now_ms() - t0);
    }
    return true;
}

bool Window::commit(int n_keep, std::string& err) {
    if (n_keep < 1 || n_keep > last_T_) { err = "metal window: commit count out of range"; return false; }
    using namespace strata::kernels;
    if (last_T_ > 1) {
        // the GDN layers: the kept prefix folded into the state (gdn_step over n_keep tokens from the window's own
        // inputs, with commit; bitwise the window's trajectory).  K/V and indexer cells past the prefix are rewritten
        // when those positions run again, before anything reads them.
        std::vector<mtl::Dispatch> ds;
        nb_ = false;
        for (int l = 0; l < 48; ++l) {
            if (l % 4 == 3) continue;
            const int gi = l - l / 4;
            const std::string b = "blk." + std::to_string(l) + ".";
            const size_t o_qkv = (size_t) gi * MAXT * 10240 * 4, o_ab = (size_t) gi * MAXT * 48 * 4, o_z = (size_t) gi * MAXT * 6144 * 4;
            GdnArgs a{16, 48, 10240, 1u, eps_, 0};
            mtl::Dispatch d;
            d.p = P(lgdn_, "gdn_step", mtl::Constants().i(2, n_keep), "gdn_step#" + std::to_string(n_keep));
            const int hp = hpar_[(size_t) gi];
            d.args = {mtl::Arg::B(gst_[(size_t) gi]), mtl::Arg::B(ghist_[hp][(size_t) gi]), mtl::Arg::B(qkvL_, o_qkv),
                      mtl::Arg::B(alL_, o_ab), mtl::Arg::B(beL_, o_ab), mtl::Arg::B(zL_, o_z),
                      mtl::Arg::B(W_, T(b + "ssm_conv1d.weight").offset), mtl::Arg::B(W_, T(b + "ssm_dt.bias").offset),
                      mtl::Arg::B(W_, T(b + "ssm_a").offset), mtl::Arg::B(W_, T(b + "ssm_norm.weight").offset),
                      mtl::Arg::B(ydump_), mtl::Arg::V(a), mtl::Arg::B(ghist_[1 - hp][(size_t) gi])};
            d.grid[0] = 48;
            d.tg[0] = 512;
            nb_ = true;   // the layers' folds are independent of each other
            push(ds, d);
            hpar_[(size_t) gi] = 1 - hp;
        }
        cx_->run_graph(ds);
    }
    strata::core::SessionState& ss = *ss_;
    for (int t = 0; t < n_keep; ++t) {
        if (ple_on_) {
            const float* norm = last_norm_.data() + (size_t) t * NG_HC_DIM;
            for (int c = 0; c < NG_HC_DIM; ++c) {
                float* col = ple_hist_.data() + (size_t) c * NG_HIST;
                for (int r = 0; r + 1 < NG_HIST; ++r) col[r] = col[r + 1];
                col[NG_HIST - 1] = norm[c];
            }
        }
        ss.ple_prev[0] = ss.ple_prev[1];
        ss.ple_prev[1] = last_tokens_[t];
    }
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
void set_model_shard(const char* shard1) { g_shard1 = shard1 ? shard1 : ""; }

void* window_create() { return new Window(); }
void window_destroy(void* w) { delete (Window*) w; }
bool window_init(void* w, const strata::core::ModelGeometry& g, strata::core::SessionState& ss, int max_t, int64_t max_cells,
                 int64_t* n_vocab, std::string& err) {
    if (g_shard1.empty()) { err = "metal window: the model shard is not set (--native SHARD1)"; return false; }
    if (!((Window*) w)->init(g, ss, g_shard1, max_t, max_cells, err)) return false;
    *n_vocab = ((Window*) w)->vocab();
    return true;
}
bool window_run(void* w, int T, const int32_t* tokens, int64_t pos0, bool head, int32_t* out, std::string& err) {
    return ((Window*) w)->run(T, tokens, pos0, head, out, err);
}
bool window_commit(void* w, int n_keep, std::string& err) { return ((Window*) w)->commit(n_keep, err); }
bool window_copy_logits(void* w, int t, float* host) { return ((Window*) w)->copy_logits(t, host); }

}  // namespace strata::metal_engine
