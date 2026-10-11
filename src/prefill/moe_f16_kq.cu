// src/prefill/moe_f16_kq.cu - see moe_f16_kq.hpp.  gfx11 only (opt-in STRATA_PF_F16_KQ=1).
//
// The paired gate/up kernel of gufo's routed experts, on Strata's tables, int8 activations and H:
//   - one work item is 64 features of one expert (its 64 gate and 64 up rows, BM = 128) against one 64-row tile of
//     fused::group, K = 2560 in 40 stages of 64 values;
//   - the weight codes stay packed in LDS (one 32-byte Q4_K nibble group per row and stage) and each wave turns its
//     own 16 rows into F16 right after the LDS read: w = q * (d sc) - dmin m as one packed FMA, so the F16 matrix
//     core accumulates the whole K in F32 with no per-block epilogue;
//   - the activations are native_w11_kernel's int8 blocks, staged in LDS as F16 (a * d_x, one product each), shared
//     by every wave;
//   - the epilogue applies SwiGLU and writes H as int8 per 32 features (with the codes' sum), the form the existing
//     down kernels read.
// The output differs from native_w11_kernel's (F16 weights instead of int8 sub-blocks), like STRATA_PF_FUSED_KQ's
// does from the MMQ path.
//
// Adapted from gufo 97b8645f: src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp
// (RoutedF16GEMMKernel and its helpers).
// MIT License
// Copyright (c) 2026 gufo contributors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "moe_f16_kq.hpp"

#include <cuda_fp16.h>

#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::prefill::fused::f16kq {
namespace {

constexpr int T_Q4_K = GGML_TYPE_Q4_K, T_Q5_K = GGML_TYPE_Q5_K;
constexpr int AB = 80;              // activation / H bytes per 64 values: 64 codes, {d0, d1}, {sum0, sum1}
constexpr int K_GU = 2560;          // n_embd
constexpr int NS = K_GU / 64;       // stages
constexpr int ACT_LD = NS * AB;     // bytes per activation row
constexpr int FEAT = 64;            // features (gate rows = up rows) per work item
constexpr int NFB = 640 / FEAT;     // work items per tile
constexpr int BM = 2 * FEAT;        // code rows in LDS: gate 0..63, up 64..127
constexpr int THREADS = 256;

#if defined(__HIPCC__) && (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
                           defined(__gfx1150__) || defined(__gfx1151__))
#define STRATA_F16KQ_DEVICE 1
#else
#define STRATA_F16KQ_DEVICE 0
#endif

using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f wmma_f16(v16h a, v16h b, v8f c) {
#if STRATA_F16KQ_DEVICE
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    __builtin_trap();
    return c;
#endif
}

// Byte i of the 16-byte K-quant block header (d, dmin, scales[12]).
__device__ __forceinline__ uint32_t header_byte(const uint4& h, uint32_t i) {
    const uint32_t word = i < 4 ? h.x : i < 8 ? h.y : i < 12 ? h.z : h.w;
    return (word >> (8U * (i & 3U))) & 0xFFU;
}
// Bit s of each of the four bytes of w, packed into bits 0-3.
__device__ __forceinline__ uint32_t gather_bit(uint32_t w, int s) { return (((w >> s) & 0x01010101U) * 0x01020408U) >> 24U; }
// Bits 0-3 of `bits` to bit 4 of each byte.
__device__ __forceinline__ uint32_t spread_high_bits(uint32_t bits) { return (__umul24(bits, 0x00204081U) & 0x01010101U) << 4U; }

constexpr uint32_t HALF_MAGIC = 0x64646464U;   // the high bytes of 1024.0 as F16
// Four code bytes (< 32) to four halves: 1024 + q exactly, minus 1024, then q * scale + bias.
__device__ __forceinline__ void codes_to_halves(uint32_t codes, __half2 magic, __half2 scale2, __half2 bias2, __half2& lo,
                                                __half2& hi) {
#if STRATA_F16KQ_DEVICE
    const uint32_t p0 = __builtin_amdgcn_perm(codes, HALF_MAGIC, 0x01050004U);
    const uint32_t p1 = __builtin_amdgcn_perm(codes, HALF_MAGIC, 0x03070206U);
    lo = __hfma2(__hadd2(__builtin_bit_cast(__half2, p0), magic), scale2, bias2);
    hi = __hfma2(__hadd2(__builtin_bit_cast(__half2, p1), magic), scale2, bias2);
#else
    (void) codes; (void) magic; (void) scale2; (void) bias2; lo = hi = __half2{};
#endif
}

// BN: tokens per work item, fused::group's 64-row tile or two consecutive tiles of one expert (128)
template <int WT, int BN>
__global__ void __launch_bounds__(THREADS) gate_up_kernel(const Batch b, const NativeGeom geo, const int32_t* __restrict__ ts,
                                                          const int2* __restrict__ tiles, const int32_t* __restrict__ off,
                                                          const uint8_t* __restrict__ act, const int32_t* __restrict__ src,
                                                          uint8_t* __restrict__ out) {
#if STRATA_F16KQ_DEVICE
    static_assert(BN == kTileRows || BN == 2 * kTileRows, "one tile or a pair");
    constexpr bool Q5K = WT == T_Q5_K;
    constexpr int TOK_TILES = BN / 16;
    constexpr int BLOCK_CHUNKS = Q5K ? 11 : 9;   // 16-byte chunks per 256-value block
    constexpr int CODE_CHUNK = Q5K ? 3 : 1;      // the first chunk of qs
    // LDS (bytes): codes BM rows x 2 chunks (swizzled); Q5_K's high bits and the (scale, bias) half2 per row and K
    // block; the activations [kb][quarter][token][16 B], one slot of padding per quarter plane.  The epilogue reuses
    // the front for eight 16 x 18 float planes.
    constexpr int CODE_BYTES = BM * 2 * 16;
    constexpr int HIGH_BYTES = Q5K ? 2 * BM * 4 : 0;
    constexpr int SCALE_BYTES = 2 * BM * 4;
    constexpr int ACT_STRIDE = BN + 1;
    constexpr int ACT_BYTES = 2 * 4 * ACT_STRIDE * 16;
    constexpr int STAGE_BYTES = CODE_BYTES + HIGH_BYTES + SCALE_BYTES + ACT_BYTES;
    constexpr int EPI_STRIDE = 18, EPI_PLANE = 16 * EPI_STRIDE;
    static_assert(8 * EPI_PLANE * 4 <= STAGE_BYTES, "the epilogue planes fit the stage");
    __shared__ __attribute__((aligned(16))) uint8_t lds[STAGE_BYTES];
    __shared__ int srow[BN];
    auto* s_codes = reinterpret_cast<uint4*>(lds);
    auto* s_high = reinterpret_cast<uint32_t*>(lds + CODE_BYTES);
    auto* s_scale = reinterpret_cast<uint32_t*>(lds + CODE_BYTES + HIGH_BYTES);
    auto* s_act = reinterpret_cast<uint4*>(lds + CODE_BYTES + HIGH_BYTES + SCALE_BYTES);

    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, sub_lane = lane & 15, half_id = lane >> 4;
    const int f_c = tid & 1;   // this thread's chunk (and K block) of its weight row
    const auto swizzle = [](int row, int c) { return row * 2 + (c ^ ((row >> 2) & 1)); };
    const __half2 magic = __floats2half2_rn(-1024.0f, -1024.0f);

    // One work item per block, the grid the batch's bound (gufo's launch: the hardware balances the blocks; the
    // NFB items of a tile are consecutive, so they meet its activations in L2)
    const int t0 = ts[b.e0], w = blockIdx.x;
    if (w >= (ts[b.e1] - t0) * NFB) return;
    {
        const int2 tl = tiles[t0 + w / NFB];
        const int fb = w % NFB, e = tl.x, row0 = tl.y;
        if (BN > kTileRows && ((t0 + w / NFB - ts[e]) & 1)) return;   // the second tile of a pair: its first takes it
        const int nrows = min(BN, off[e + 1] - row0);
        const int live_tok_tiles = (nrows + 15) / 16;
        const uint8_t* blob = b.blob[e - b.e0];
        if (tid < BN) srow[tid] = tid < nrows ? src[row0 + tid] : -1;
        __syncthreads();

        // Weight fetch: row tid / 2 of the code plane (gate rows, then up rows), chunk f_c.
        const uint8_t* f_ptr = blob + ((tid >> 1) >= FEAT ? geo.up_off : 0) +
                               (size_t) (fb * FEAT + ((tid >> 1) & (FEAT - 1))) * geo.gu_row;
        uint4 f_header = make_uint4(0u, 0u, 0u, 0u), f_codes, f_qh[2] = {};
        uint4 code_cache[4];   // the block's four nibble groups, this thread's chunk of each
        uint32_t f_high = 0;
        int f_sb32 = 0;
        // Activation fetch: BN tokens x 8 chunks of 8 values per stage; chunk sub = kb * 4 + quarter.
        constexpr int ACT_FETCH = BN * 8 / THREADS;
        const uint8_t* a_src[ACT_FETCH];
        int a_slot[ACT_FETCH], a_sub[ACT_FETCH];
        uint2 a_codes[ACT_FETCH];
        float a_d[ACT_FETCH];
#pragma unroll
        for (int i = 0; i < ACT_FETCH; ++i) {
            const int chunk = tid + i * THREADS, t = chunk >> 3, sub = chunk & 7;
            const int tok = srow[t];
            a_src[i] = tok >= 0 ? act + (size_t) tok * ACT_LD : nullptr;
            a_slot[i] = sub * ACT_STRIDE + t;
            a_sub[i] = sub;
        }

        const auto fetch_stage = [&](int s) {
            const int kb0 = 2 * s;
            const uint4* blk = reinterpret_cast<const uint4*>(f_ptr) + (kb0 / 8) * BLOCK_CHUNKS;
            if (kb0 % 8 == 0) {   // a new 256-value block every four stages
                f_header = blk[0];
#pragma unroll
                for (int group = 0; group < 4; ++group) code_cache[group] = blk[CODE_CHUNK + group * 2 + f_c];
                if constexpr (Q5K) {
                    f_qh[0] = blk[1];
                    f_qh[1] = blk[2];
                }
            }
            const int sb32 = (kb0 % 8) + f_c;
            {
                const int group = sb32 / 2;   // uniform across the wave
                const uint32_t low = 0u - (group & 1u), high = 0u - ((group >> 1u) & 1u);
                const auto select = [](uint32_t a, uint32_t bb, uint32_t mask) {
                    uint32_t value;
                    asm("v_cndmask_b32_e64 %0, %1, %2, %3" : "=v"(value) : "v"(a), "v"(bb), "s"(mask));
                    return value;
                };
                const uint4 &c0 = code_cache[0], &c1 = code_cache[1], &c2 = code_cache[2], &c3 = code_cache[3];
                f_codes = make_uint4(select(select(c0.x, c1.x, low), select(c2.x, c3.x, low), high),
                                     select(select(c0.y, c1.y, low), select(c2.y, c3.y, low), high),
                                     select(select(c0.z, c1.z, low), select(c2.z, c3.z, low), high),
                                     select(select(c0.w, c1.w, low), select(c2.w, c3.w, low), high));
            }
            f_sb32 = sb32;
            if constexpr (Q5K) {
                const uint32_t qh[8] = {f_qh[0].x, f_qh[0].y, f_qh[0].z, f_qh[0].w, f_qh[1].x, f_qh[1].y, f_qh[1].z, f_qh[1].w};
                uint32_t hb = 0;
#pragma unroll
                for (int i = 0; i < 8; ++i) hb |= gather_bit(qh[i], sb32) << (4 * i);
                f_high = hb;
            }
#pragma unroll
            for (int i = 0; i < ACT_FETCH; ++i) {
                if (a_src[i] != nullptr) {
                    const uint8_t* p = a_src[i] + (size_t) s * AB;
                    a_codes[i] = *reinterpret_cast<const uint2*>(p + 8 * a_sub[i]);
                    a_d[i] = *reinterpret_cast<const float*>(p + 64 + 4 * (a_sub[i] >> 2));
                } else {
                    a_codes[i] = make_uint2(0u, 0u);
                    a_d[i] = 0.0f;
                }
            }
        };

        const auto commit_stage = [&]() {
            const int row = tid >> 1;
            s_codes[swizzle(row, f_c)] = f_codes;
            if constexpr (Q5K) s_high[f_c * BM + row] = f_high;
            const int sb32 = f_sb32;
            uint32_t sc, mn;
            if (sb32 < 4) {
                sc = header_byte(f_header, 4 + sb32) & 0x3FU;
                mn = header_byte(f_header, 8 + sb32) & 0x3FU;
            } else {
                sc = (header_byte(f_header, 8 + sb32) & 0x0FU) | ((header_byte(f_header, sb32) >> 6U) << 4U);
                mn = (header_byte(f_header, 8 + sb32) >> 4U) | ((header_byte(f_header, 4 + sb32) >> 6U) << 4U);
            }
            const __half2 dm = __builtin_bit_cast(__half2, f_header.x);
            const float scale = __low2float(dm) * (float) sc, offset = __high2float(dm) * (float) mn;
            s_scale[f_c * BM + row] = __builtin_bit_cast(uint32_t, __floats2half2_rn(scale, -offset));
#pragma unroll
            for (int i = 0; i < ACT_FETCH; ++i) {
                const uint32_t ws[2] = {a_codes[i].x, a_codes[i].y};
                __half2 h[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t wd = ws[j >> 1] >> (16 * (j & 1));
                    const float lo = (float) (int8_t) (wd & 0xFFu), hi = (float) (int8_t) ((wd >> 8) & 0xFFu);
                    h[j] = __floats2half2_rn(lo * a_d[i], hi * a_d[i]);
                }
                s_act[a_slot[i]] = make_uint4(__builtin_bit_cast(uint32_t, h[0]), __builtin_bit_cast(uint32_t, h[1]),
                                              __builtin_bit_cast(uint32_t, h[2]), __builtin_bit_cast(uint32_t, h[3]));
            }
        };

        v8f acc[TOK_TILES];
#pragma unroll
        for (int j = 0; j < TOK_TILES; ++j) acc[j] = v8f{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        const auto compute_stage = [&]() {
            const int row = wave * 16 + sub_lane;   // waves 0-3: gate rows, 4-7: the matching up rows
            const uint4 raw0 = s_codes[swizzle(row, 0)], raw1 = s_codes[swizzle(row, 1)];
#pragma unroll
            for (int kb = 0; kb < 2; ++kb) {
                const __half2 sb = __builtin_bit_cast(__half2, s_scale[kb * BM + row]);
                const __half2 scale2 = __low2half2(sb), bias2 = __high2half2(sb);
                // elements 0-15 of K block 2s + kb are the low (kb = 0) or high (kb = 1) nibbles of chunk 0, 16-31 of chunk 1
                const unsigned shift = 4U * (unsigned) kb;
                const uint32_t words[8] = {raw0.x, raw0.y, raw0.z, raw0.w, raw1.x, raw1.y, raw1.z, raw1.w};
                const uint32_t hb = Q5K ? s_high[kb * BM + row] : 0U;
                __half2 h[16];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    uint32_t nib = (words[i] >> shift) & 0x0F0F0F0FU;
                    if constexpr (Q5K) nib |= spread_high_bits((hb >> (4 * i)) & 0xFU);
                    codes_to_halves(nib, magic, scale2, bias2, h[2 * i], h[2 * i + 1]);
                }
                v16h a_lo, a_hi;
                __builtin_memcpy(&a_lo, &h[0], 32);
                __builtin_memcpy(&a_hi, &h[8], 32);
#pragma unroll
                for (int j = 0; j < TOK_TILES; ++j) {
                    asm volatile("" ::: "memory");   // one token tile's fragments live at a time (gufo: hoisting spills)
                    if (j >= live_tok_tiles) continue;
                    const uint4* frag = s_act + (kb * 4) * ACT_STRIDE + j * 16 + sub_lane;
                    uint4 bq[4];
#pragma unroll
                    for (int q = 0; q < 4; ++q) bq[q] = frag[q * ACT_STRIDE];
                    v16h b_lo, b_hi;
                    __builtin_memcpy(&b_lo, &bq[0], 32);
                    __builtin_memcpy(&b_hi, &bq[2], 32);
                    acc[j] = wmma_f16(a_lo, b_lo, acc[j]);
                    acc[j] = wmma_f16(a_hi, b_hi, acc[j]);
                }
            }
        };

        fetch_stage(0);
        for (int s = 0; s < NS; ++s) {
            commit_stage();
            __syncthreads();
            if (s + 1 < NS) fetch_stage(s + 1);
            compute_stage();
            __syncthreads();
        }

        // Epilogue per token tile: each wave's 16 x 16 accumulators as plane [token][row] (gate planes 0-3, up 4-7),
        // then a thread per (token, half of the 64 features): SwiGLU, int8 per 32 with the codes' sum.
        float* base = reinterpret_cast<float*>(lds);
        float* scratch = base + wave * EPI_PLANE;
#pragma unroll
        for (int j = 0; j < TOK_TILES; ++j) {
            if (j >= live_tok_tiles) break;   // uniform
#pragma unroll
            for (int l = 0; l < 8; ++l) scratch[sub_lane * EPI_STRIDE + 2 * l + half_id] = acc[j][l];
            __syncthreads();
            if (tid < 32) {
                const int t = tid >> 1, hh = tid & 1, r = j * 16 + t;
                if (r < nrows) {
                    float hv[32];
                    float am = 0.0f;
#pragma unroll
                    for (int f = 0; f < 32; ++f) {
                        const int ft = 32 * hh + f;
                        const int idx = (ft >> 4) * EPI_PLANE + t * EPI_STRIDE + (ft & 15);
                        const float gt = base[idx], up = base[idx + 4 * EPI_PLANE];
                        hv[f] = gt / (1.0f + __expf(-gt)) * up;
                        am = fmaxf(am, fabsf(hv[f]));
                    }
                    const float inv = am > 0.0f ? 127.0f / am : 0.0f;
                    uint8_t* o = out + (size_t) (row0 + r) * (10 * AB) + (size_t) fb * AB;
                    uint32_t wd[8] = {};
                    int csum = 0;
#pragma unroll
                    for (int f = 0; f < 32; ++f) {
                        const int c = __float2int_rn(hv[f] * inv);
                        csum += c;
                        wd[f >> 2] |= (uint32_t) (uint8_t) (int8_t) c << (8 * (f & 3));
                    }
                    uint4* o4 = reinterpret_cast<uint4*>(o + 32 * hh);
                    o4[0] = make_uint4(wd[0], wd[1], wd[2], wd[3]);
                    o4[1] = make_uint4(wd[4], wd[5], wd[6], wd[7]);
                    *reinterpret_cast<float*>(o + 64 + 4 * hh) = am / 127.0f;
                    *reinterpret_cast<float*>(o + 72 + 4 * hh) = (float) csum;
                }
            }
            __syncthreads();
        }
    }
#else
    (void) b; (void) geo; (void) ts; (void) tiles; (void) off; (void) act; (void) src; (void) out;
    __builtin_trap();
#endif
}

// Two tiles a work item when an expert has a tile's rows or more on average.  gfx1151, Q4_K, top 10 over 512 experts,
// one 128-expert launch, ms (64 / 128): 1024 tokens 1.22 / 1.28, 2048 1.51 / 1.64, 4096 2.73 / 2.48, 8192 4.76 / 4.59,
// 16384 9.19 / 8.46.  STRATA_PF_F16_KQ_BN=64|128 forces one.
int pick_bn(int64_t n, int n_expert) {
    static const int forced = [] {
        const char* v = std::getenv("STRATA_PF_F16_KQ_BN");
        const int t = v ? std::atoi(v) : 0;
        return t == 64 || t == 128 ? t : 0;
    }();
    if (forced) return forced;
    return (double) n / std::max(n_expert, 1) >= kTileRows ? 2 * kTileRows : kTileRows;
}

template <int BN>
void launch(const Batch& b, const NativeGeom& g, const int32_t* ts, const int2* tiles, const int32_t* off, int64_t work_bound,
            const uint8_t* act, const int32_t* src, uint8_t* out, cudaStream_t s) {
    const unsigned grid = (unsigned) std::max<int64_t>(1, work_bound * NFB);
    if (g.gu_type == T_Q5_K) gate_up_kernel<T_Q5_K, BN><<<grid, THREADS, 0, s>>>(b, g, ts, tiles, off, act, src, out);
    else gate_up_kernel<T_Q4_K, BN><<<grid, THREADS, 0, s>>>(b, g, ts, tiles, off, act, src, out);
}

}  // namespace

bool requested() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_PF_F16_KQ");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}

bool covers(int gu_type) { return gu_type == T_Q4_K || gu_type == T_Q5_K; }

void gate_up(const Batch& b, const NativeGeom& g, const int32_t* ts, const int2* tiles, const int32_t* off, int64_t n,
             int n_expert, int64_t work_bound, const uint8_t* act, const int32_t* src, uint8_t* out, cudaStream_t s) {
    if (pick_bn(n, n_expert) == 2 * kTileRows) launch<2 * kTileRows>(b, g, ts, tiles, off, work_bound, act, src, out, s);
    else launch<kTileRows>(b, g, ts, tiles, off, work_bound, act, src, out, s);
}

}  // namespace strata::prefill::fused::f16kq
