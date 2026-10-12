// metal/kernels/qsa_prompt.metal - QSA attention for a PROMPT chunk on the matrix units (after qsa.metal: QH, QKV, QD,
// QBUDGET, QCH, QsaArgs).  The same split-K partials as qsa_attn (part [t][chunk][QH][QD + 2]: acc, then m, l), so
// qsa_merge is unchanged; per (chunk of QCH = 64 selected cells, kv head g, token t) one simdgroup:
//   S[12 x 64] = Qh[12 x 256] . K^T / 16      single-simdgroup matmul2d, K (int8 * fp16 scale per 64) decoded into
//                                             the right-input cooperative tensor, 32 dims a step
//   per head: m = max, p = exp(s - m), l = sum p (fp32, through threadgroup memory), P -> fp16
//   acc[12 x 256] = P[12 x 64] . V[64 x 256]  four 64-dim matmuls, V decoded into the right input
// The operands are fp16 (q rounded from fp32; k, v = int8 * scale rounded): to rounding of qsa_attn's fp32.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// grid (chunks, QKV, n_tok), 32 threads.  qh: the q heads as fp16 [t][QH][QD] (qcur rounded).
kernel void qsa_attn_mm(device const half* qh [[buffer(0)]], device const char* kq [[buffer(1)]],
                        device const char* vq [[buffer(2)]], device const half* ksc [[buffer(3)]],
                        device const half* vsc [[buffer(4)]], device const int* ids [[buffer(5)]],
                        device float* part [[buffer(6)]], constant QsaArgs& a [[buffer(7)]],
                        uint3 tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int HPG = QH / QKV, NCH = (QBUDGET + QCH - 1) / QCH;
    threadgroup float s[16 * QCH];
    threadgroup half p[16 * QCH];
    threadgroup int cid[QCH];
    threadgroup half vs[QCH][4];
    const int chunk = int(tg.x), g = int(tg.y), t = int(tg.z);
    const uint n_kv = a.pos0 + uint(t) + 1, width = min(n_kv, uint(QBUDGET));
    const int c0 = chunk * QCH;
    device float* pp = part + ((uint(t) * NCH + chunk) * QH + g * HPG) * (QD + 2);
    if (uint(c0) >= width) {
        if (lane < HPG) { pp[lane * (QD + 2) + QD] = -INFINITY; pp[lane * (QD + 2) + QD + 1] = 0.0f; }
        return;
    }
    const int nc = min(QCH, int(width) - c0);
    for (int c = lane; c < QCH; c += 32) {
        const int id = c < nc ? ids[t * QBUDGET + c0 + c] : -1;
        cid[c] = id;
        for (int j = 0; j < 4; ++j) vs[c][j] = id >= 0 ? vsc[(uint(id) * QKV + g) * 4 + j] : half(0.0f);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    // ---- S = Q K^T
    {
        constexpr auto desc = matmul2d_descriptor(16, QCH, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc, execution_simdgroup> op;
        tensor<device half, dextents<int32_t, 2>, tensor_inline> tQ((device half*)qh + (uint(t) * QH + g * HPG) * QD,
                                                                   dextents<int32_t, 2>(QD, HPG));
        auto rk = op.template get_right_input_cooperative_tensor<half, half, float>();
        auto tA = tQ.template slice<32, 16>(0, 0);
        auto acc = op.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tA)>, __tensor_ops_detail::__remove_addrspace_t<decltype(rk)>, float>();
        for (uint16_t i = 0; i < acc.get_capacity(); ++i)
            if (acc.is_valid_element(i)) acc[i] = 0.0f;
        for (int k0 = 0; k0 < QD; k0 += 32) {
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < rk.get_capacity(); i += 4) {
                const auto ix = rk.get_multidimensional_index(i);
                const int k = k0 + ix[0], c = ix[1], id = cid[c];
                half4 w = half4(0.0h);
                if (id >= 0) {
                    const uint row = uint(id) * QKV + g;
                    const char4 q4 = *(device const char4*)(kq + row * QD + k);
                    w = half4(float4(q4) * float(ksc[row * 4 + (k >> 6)]));
                }
                rk[i] = w.x; rk[i + 1] = w.y; rk[i + 2] = w.z; rk[i + 3] = w.w;
            }
            tA = tQ.template slice<32, 16>(k0, 0);
            op.run(tA, rk, acc);
        }
        tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tS(s, dextents<int32_t, 2>(QCH, 16));
        acc.store(tS);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    // ---- softmax pieces per head (lane = head for the reductions; 12 heads)
    float mh = -INFINITY, lh = 0.0f;
    if (lane < HPG) {
        for (int c = 0; c < nc; ++c) mh = max(mh, s[lane * QCH + c] * 0.0625f);
        for (int c = 0; c < QCH; ++c) {
            const float e = c < nc ? exp(s[lane * QCH + c] * 0.0625f - mh) : 0.0f;
            lh += e;
            p[lane * QCH + c] = half(e);
        }
    } else if (lane < 16) {
        for (int c = 0; c < QCH; ++c) p[lane * QCH + c] = half(0.0f);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    // ---- acc = P V, 64 dims at a time
    {
        constexpr auto desc = matmul2d_descriptor(16, 64, QCH, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<desc, execution_simdgroup> op;
        tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tP(p, dextents<int32_t, 2>(QCH, 16));
        tensor<device float, dextents<int32_t, 2>, tensor_inline> tO(pp, dextents<int32_t, 2>(QD + 2, HPG));
        for (int d0 = 0; d0 < QD; d0 += 64) {
            auto rv = op.template get_right_input_cooperative_tensor<half, half, float>();
            auto acc = op.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tP)>, __tensor_ops_detail::__remove_addrspace_t<decltype(rv)>, float>();
            for (uint16_t i = 0; i < acc.get_capacity(); ++i)
                if (acc.is_valid_element(i)) acc[i] = 0.0f;
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < rv.get_capacity(); ++i) {
                const auto ix = rv.get_multidimensional_index(i);   // (cell, dim)
                const int c = ix[0], d = d0 + ix[1], id = cid[c];
                rv[i] = id >= 0 ? half(float(vq[(uint(id) * QKV + g) * QD + d]) * float(vs[c][d >> 6])) : half(0.0f);
            }
            op.run(tP, rv, acc);
            auto mO = tO.slice(d0, 0);
            acc.store(mO);
        }
    }
    if (lane < HPG) { pp[lane * (QD + 2) + QD] = mh; pp[lane * (QD + 2) + QD + 1] = lh; }
}

// qcur (fp32) -> fp16 for qsa_attn_mm
kernel void qsa_q_f16(device const float* q [[buffer(0)]], device half* qh [[buffer(1)]], constant uint& n [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid < n) qh[gid] = half(q[gid]);
}
