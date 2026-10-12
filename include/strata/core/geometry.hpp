// include/strata/core/geometry.hpp - the model's geometry (moved out of layout.hpp so that tools and tests can use
// it without the weight table).  The defaults are Qwen3.8-Flash-Next's: they are the shapes this engine's kernels
// are built for.  core::geometry_from_gguf (family_layout.hpp) fills one from a model file's metadata.
#pragma once

#include "strata/artifact/model_family.hpp"

#include <cstdint>

namespace strata::core {

/// The model's geometry, taken from `docs/semantics.md` and the artifact's own metadata.  Every field here
/// is a number a kernel depends on, so a change is a change to a kernel contract and not a tuning knob.
struct ModelGeometry {
    int64_t n_embd = 2560;
    int64_t n_layers = 48;         ///< the trunk's layers (an in-file MTP block is counted in n_mtp_layers)
    int64_t qsa_interval = 4;      ///< every 4th layer is full attention: layers 3, 7, ... 47

    // GDN (36 layers)
    int64_t ssm_state_size = 128;
    int64_t ssm_k_heads = 16;
    int64_t ssm_v_heads = 48;
    int64_t ssm_d_conv = 4;
    int64_t ssm_conv_channels = 10240;   ///< 2*128*16 + 128*48
    int64_t ssm_value_dim = 6144;        ///< 128 * 48

    // QSA (12 layers)
    int64_t n_head = 24;
    int64_t n_head_kv = 2;
    int64_t head_dim = 256;
    int64_t idx_q_heads = 4;       ///< 0: no sparse indexer (dense attention only: qwen35moe)
    int64_t idx_key_dim = 128;

    // gated residual, on every layer
    int64_t hc = 4;                ///< 0: no hyper-connections (a plain pre-norm residual: qwen35moe)
    int64_t hc_lr = 320;

    // MoE, on every layer
    int64_t n_expert = 512;
    int64_t n_ff = 640;
    int64_t n_ff_shexp = 640;      ///< the shared expert's FFN width

    // appended for the model families (NEW-MODELS-PLAN.md T0); the defaults keep Flash-Next's values
    ModelFamily family = ModelFamily::Qwen4Exp;
    int64_t n_mtp_layers = 0;      ///< MTP blocks inside the model file, after the trunk (qwen35moe: blk.40)
    int64_t n_vocab = 248320;

    int64_t hc_dim() const { return hc * n_embd; }
    /// `layer % qsa_interval == qsa_interval - 1` is full attention.  Derived, not a second list.
    int64_t n_qsa_layers() const { return n_layers / qsa_interval; }
    int64_t n_gdn_layers() const { return n_layers - n_qsa_layers(); }
    bool has_indexer() const { return idx_q_heads > 0; }
    bool has_hyper_connections() const { return hc > 0; }
};

/// True for the full-attention layers.  `docs/semantics.md` gives this twice over - `full_attention_interval
/// = 4` and an explicit `attention.compress_ratios` array - and this is the first of the two.
inline bool is_qsa_layer(const ModelGeometry& g, int64_t layer) {
    return layer % g.qsa_interval == g.qsa_interval - 1;
}

}  // namespace strata::core
