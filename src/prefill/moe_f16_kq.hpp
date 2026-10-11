// src/prefill/moe_f16_kq.hpp - gfx11: the K-quant prompt experts' gate/up on the F16 matrix cores (moe_f16_kq.cu).
// Internal to moe_fused_iq.cu, which owns the native experts' tables, launch and down projection.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "strata/prefill/moe_fused.hpp"
#include "strata/prefill/moe_fused_iq.hpp"

namespace strata::prefill::fused::f16kq {

/// STRATA_PF_F16_KQ=1 (on top of STRATA_PF_FUSED=1 and STRATA_PF_FUSED_KQ=1): Q4_K / Q5_K gate/up take this path.
bool requested();
/// The gate/up type has a kernel here (and this build has gfx11 device code).
bool covers(int gu_type);

/// Gate/up, SwiGLU and H for experts [b.e0, b.e1): the same tiles, activations and H as native_w11_kernel<WT, true>.
/// `ts`, `tiles`, `off` are fused::group's tables (at most `work_bound` tiles); `n` the layer's routed rows over
/// `n_expert` experts; `act` the int8 activations (quantize_act_native); `out` H.
void gate_up(const Batch& b, const NativeGeom& g, const int32_t* ts, const int2* tiles, const int32_t* off, int64_t n,
             int n_expert, int64_t work_bound, const uint8_t* act, const int32_t* src, uint8_t* out, cudaStream_t s);

}  // namespace strata::prefill::fused::f16kq
