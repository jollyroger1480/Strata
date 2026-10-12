// qi21-native.cu - Strata's native kernels for the Qwen-Image-2.1 DiT inside stable-diffusion.cpp (CUDA and HIP).
//
// Copied into ggml/src/ggml-cuda/ by tools/image/sdcpp_native/apply.py, so ggml-cuda and ggml-hip compile it with
// their own toolchain and vendor headers.  The host side (qwen_image_2_1_native.hpp) runs the GEMMs as ggml graphs
// on the same backend and calls these launchers on the backend's own stream, so everything stays stream-ordered.
//
//   qi21_res_ln_mod   x += h * tanh(g) (optional), then out = LayerNorm(x) * (1 + m)     (no affine, eps)
//   qi21_qkv_post     per head: RMSNorm(q|k) * w, 3-axis RoPE from sd.cpp's pe table, K/V to fp16, head-major
//   qi21_swiglu       out = up * silu(gate)  (gate = first half of the fused gate_up output)
//   qi21_attn_causal  plain fp32 attention for the text prefix (causal; once per prompt)
//   qi21_attn_wmma    gfx11 WMMA flash attention (fp16 in, fp32 softmax and accumulation) for the image tokens
#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstdint>

namespace {

constexpr int QI_D = 128;

__device__ __forceinline__ float qi_block_sum(float v, float* red) {
    // blockDim.x is a multiple of 32 and <= 1024
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off, 32);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) red[w] = v;
    __syncthreads();
    const int nw = blockDim.x >> 5;
    float s = threadIdx.x < nw ? red[threadIdx.x] : 0.f;
    if (w == 0) {
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off, 32);
        if (l == 0) red[0] = s;
    }
    __syncthreads();
    return red[0];
}

template <bool RES, bool LN>
__global__ void __launch_bounds__(256) k_res_ln_mod(float* __restrict__ x, const float* __restrict__ h,
                                                   const float* __restrict__ g, float* __restrict__ out,
                                                   const float* __restrict__ m, int C, float eps) {
    __shared__ float red[32];
    const size_t row = blockIdx.x;
    float* xr = x + row * C;
    float v[32];
    int n = 0;
    float s = 0.f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float a = xr[c];
        if (RES) {
            a += h[row * C + c] * tanhf(g[c]);
            xr[c] = a;
        }
        v[n++] = a;
        s += a;
    }
    if (!LN) return;
    const float mean = qi_block_sum(s, red) / C;
    float q = 0.f;
    for (int i = 0; i < n; ++i) q += (v[i] - mean) * (v[i] - mean);
    const float inv = rsqrtf(qi_block_sum(q, red) / C + eps);
    n = 0;
    for (int c = threadIdx.x; c < C; c += blockDim.x) out[row * C + c] = (v[n++] - mean) * inv * (1.f + m[c]);
}

// grid (T, H), block 128: one head of one token.  src rows have stride ld (floats).
template <typename QT>
__global__ void __launch_bounds__(QI_D) k_qkv_post(const float* __restrict__ q, const float* __restrict__ k,
                                                  const float* __restrict__ v, int ld, const float* __restrict__ wq,
                                                  const float* __restrict__ wk, const float* __restrict__ pe,
                                                  QT* __restrict__ qo, half* __restrict__ qlo, int q_tstride,
                                                  half* __restrict__ ko,
                                                  half* __restrict__ vo, int kv_tstride, int kv_off, float eps) {
    __shared__ float red[32];
    __shared__ float buf[QI_D];
    const int t = blockIdx.x, hh = blockIdx.y, d = threadIdx.x;
    const size_t src = (size_t) t * ld + hh * QI_D + d;
    const float* p = pe + ((size_t) t * (QI_D / 2) + (d >> 1)) * 4;
    for (int which = 0; which < 2; ++which) {
        const float x = which == 0 ? q[src] : k[src];
        const float ss = qi_block_sum(x * x, red);
        const float y = x * rsqrtf(ss / QI_D + eps) * (which == 0 ? wq[d] : wk[d]);
        buf[d] = y;
        __syncthreads();
        const float x0 = buf[d & ~1], x1 = buf[d | 1];
        const float r = (d & 1) ? p[2] * x0 + p[3] * x1 : p[0] * x0 + p[1] * x1;
        __syncthreads();
        if (which == 0) {
            const size_t qi = ((size_t) hh * q_tstride + t) * QI_D + d;
            qo[qi] = (QT) r;
            if (qlo != nullptr) qlo[qi] = __float2half(r - __half2float(__float2half(r)));
        } else {
            ko[((size_t) hh * kv_tstride + kv_off + t) * QI_D + d] = __float2half(r);
        }
    }
    vo[((size_t) hh * kv_tstride + kv_off + t) * QI_D + d] = __float2half(v[src]);
}

__global__ void k_swiglu(const float* __restrict__ gu, float* __restrict__ out, int F, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t t = i / F, c = i % F;
    const float g = gu[t * 2 * F + c], u = gu[t * 2 * F + F + c];
    out[i] = u * (g / (1.f + expf(-g)));
}

// grid (Lq, H), block 128, dynamic smem Lk floats.  q fp32 head-major [H][q_tstride][128], k/v fp16 [H][kv_tstride][128],
// out token-major [Lq][H*128].  causal: query i sees keys 0..i.
__global__ void __launch_bounds__(QI_D) k_attn_causal(const float* __restrict__ q, int q_tstride,
                                                     const half* __restrict__ k, const half* __restrict__ v,
                                                     int kv_tstride, float* __restrict__ out, int H, int Lk,
                                                     int causal, float scale) {
    extern __shared__ float s[];
    __shared__ float red[32];
    __shared__ float qs[QI_D];
    const int i = blockIdx.x, hh = blockIdx.y, d = threadIdx.x;
    qs[d] = q[((size_t) hh * q_tstride + i) * QI_D + d];
    __syncthreads();
    const int n = causal ? i + 1 : Lk;
    float mx = -INFINITY;
    for (int j = d; j < n; j += QI_D) {
        const half* kr = k + ((size_t) hh * kv_tstride + j) * QI_D;
        float dot = 0.f;
        for (int c = 0; c < QI_D; ++c) dot += qs[c] * __half2float(kr[c]);
        s[j] = dot * scale;
        mx = fmaxf(mx, s[j]);
    }
    // block max
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off, 32));
    __syncthreads();
    if ((d & 31) == 0) red[d >> 5] = mx;
    __syncthreads();
    mx = red[0];
    for (int w = 1; w < QI_D / 32; ++w) mx = fmaxf(mx, red[w]);
    __syncthreads();
    float sum = 0.f;
    for (int j = d; j < n; j += QI_D) {
        s[j] = expf(s[j] - mx);
        sum += s[j];
    }
    sum = qi_block_sum(sum, red);
    float acc = 0.f;
    for (int j = 0; j < n; ++j) acc += s[j] * __half2float(v[((size_t) hh * kv_tstride + j) * QI_D + d]);
    out[(size_t) i * H * QI_D + hh * QI_D + d] = acc / sum;
}

#if defined(GGML_USE_HIP)
typedef _Float16 qi_h16 __attribute__((ext_vector_type(16)));
typedef float qi_f8 __attribute__((ext_vector_type(8)));
constexpr int FA_BQ = 128, FA_BKV = 64, FA_NW = 8, FA_KLD = QI_D + 8, FA_VLD = FA_BKV + 8, FA_PLD = FA_BKV + 8;

__device__ __forceinline__ qi_h16 qi_ld16(const _Float16* p) {
    const uint4 a = *reinterpret_cast<const uint4*>(p), b = *reinterpret_cast<const uint4*>(p + 8);
    const uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    return __builtin_bit_cast(qi_h16, w);
}

// gfx11 WMMA flash attention: q fp16 head-major [H][q_tstride][128], k/v fp16 head-major [H][kv_tstride][128]
// (no mask), out fp32 token-major [nq][H*128].  8 waves x 16 query rows; 64-key tiles through LDS (V transposed).
template <bool LO>
__global__ void __launch_bounds__(FA_NW * 32) k_attn_wmma(const _Float16* __restrict__ Q, const _Float16* __restrict__ Qlo,
                                                         int q_tstride,
                                                         const _Float16* __restrict__ K, const _Float16* __restrict__ V,
                                                         int kv_tstride, float* __restrict__ O, int H, int nq, int nkv,
                                                         float scale_log2) {
#if defined(RDNA3)
    __shared__ __align__(16) _Float16 sK[FA_BKV][FA_KLD];
    __shared__ __align__(16) _Float16 sVt[QI_D][FA_VLD];
    __shared__ __align__(16) _Float16 sP[FA_NW][16][FA_PLD];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, l16 = lane & 15, hi = lane >> 4;
    const int head = blockIdx.y, q0 = blockIdx.x * FA_BQ + wave * 16;
    const _Float16* Qh = Q + (size_t) head * q_tstride * QI_D;
    const _Float16* Kh = K + (size_t) head * kv_tstride * QI_D;
    const _Float16* Vh = V + (size_t) head * kv_tstride * QI_D;
    qi_h16 qa[QI_D / 16], ql[LO ? QI_D / 16 : 1];
    {
        const int qr = min(q0 + l16, nq - 1);
        for (int c = 0; c < QI_D / 16; ++c) qa[c] = qi_ld16(Qh + (size_t) qr * QI_D + c * 16);
        if (LO) {
            const _Float16* Ql = Qlo + (size_t) head * q_tstride * QI_D;
            for (int c = 0; c < QI_D / 16; ++c) ql[c] = qi_ld16(Ql + (size_t) qr * QI_D + c * 16);
        }
    }
    qi_f8 o[QI_D / 16];
    for (int c = 0; c < QI_D / 16; ++c) o[c] = qi_f8{};
    float m[8], l[8];
    for (int i = 0; i < 8; ++i) { m[i] = -INFINITY; l[i] = 0.f; }
    for (int k0 = 0; k0 < nkv; k0 += FA_BKV) {
        __syncthreads();
        for (int e = tid * 8; e < FA_BKV * QI_D; e += FA_NW * 32 * 8) {
            const int r = e / QI_D, c = e % QI_D, key = k0 + r;
            uint4 kv = make_uint4(0, 0, 0, 0);
            if (key < nkv) kv = *reinterpret_cast<const uint4*>(Kh + (size_t) key * QI_D + c);
            *reinterpret_cast<uint4*>(&sK[r][c]) = kv;
        }
        for (int e = tid; e < (FA_BKV / 4) * (QI_D / 8); e += FA_NW * 32) {
            const int r4 = (e / (QI_D / 8)) * 4, c = (e % (QI_D / 8)) * 8;
            _Float16 vv[4][8];
            for (int j = 0; j < 4; ++j) {
                uint4 x = make_uint4(0, 0, 0, 0);
                if (k0 + r4 + j < nkv) x = *reinterpret_cast<const uint4*>(Vh + (size_t) (k0 + r4 + j) * QI_D + c);
                *reinterpret_cast<uint4*>(vv[j]) = x;
            }
            for (int dd = 0; dd < 8; ++dd) {
                _Float16 w4[4] = {vv[0][dd], vv[1][dd], vv[2][dd], vv[3][dd]};
                *reinterpret_cast<uint2*>(&sVt[c + dd][r4]) = *reinterpret_cast<uint2*>(w4);
            }
        }
        __syncthreads();
        qi_f8 s[FA_BKV / 16];
        for (int t = 0; t < FA_BKV / 16; ++t) {
            s[t] = qi_f8{};
            for (int c = 0; c < QI_D / 16; ++c) {
                const qi_h16 kb = qi_ld16(&sK[t * 16 + l16][c * 16]);
                s[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(qa[c], kb, s[t]);
                if (LO) s[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ql[LO ? c : 0], kb, s[t]);
            }
        }
        float alpha[8];
        for (int i = 0; i < 8; ++i) {
            float mx = -INFINITY;
            for (int t = 0; t < FA_BKV / 16; ++t) {
                const float val = k0 + t * 16 + l16 < nkv ? s[t][i] * scale_log2 : -INFINITY;
                s[t][i] = val;
                mx = fmaxf(mx, val);
            }
            for (int off = 1; off < 16; off <<= 1) mx = fmaxf(mx, __shfl_xor(mx, off, 32));
            const float mnew = fmaxf(m[i], mx);
            alpha[i] = exp2f(m[i] - mnew);
            float sum = 0.f;
            for (int t = 0; t < FA_BKV / 16; ++t) {
                const float pv = exp2f(s[t][i] - mnew);
                s[t][i] = pv;
                sum += pv;
            }
            for (int off = 1; off < 16; off <<= 1) sum += __shfl_xor(sum, off, 32);
            l[i] = l[i] * alpha[i] + sum;
            m[i] = mnew;
        }
        for (int c = 0; c < QI_D / 16; ++c)
            for (int i = 0; i < 8; ++i) o[c][i] *= alpha[i];
        for (int t = 0; t < FA_BKV / 16; ++t)
            for (int i = 0; i < 8; ++i) sP[wave][2 * i + hi][t * 16 + l16] = (_Float16) s[t][i];
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_wave_barrier();
        for (int kc = 0; kc < FA_BKV / 16; ++kc) {
            const qi_h16 pa = qi_ld16(&sP[wave][l16][kc * 16]);
            for (int c = 0; c < QI_D / 16; ++c)
                o[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, qi_ld16(&sVt[c * 16 + l16][kc * 16]), o[c]);
        }
    }
    for (int c = 0; c < QI_D / 16; ++c)
        for (int i = 0; i < 8; ++i) {
            const int row = q0 + 2 * i + hi;
            if (row < nq) O[(size_t) row * H * QI_D + head * QI_D + c * 16 + l16] = o[c][i] / l[i];
        }
#else
    __builtin_trap();
#endif
}
#endif  // GGML_USE_HIP

inline cudaStream_t qi_s(void* s) { return static_cast<cudaStream_t>(s); }

}  // namespace

extern "C" {

void* qi21_stream(ggml_backend_t backend) {
    return static_cast<ggml_backend_cuda_context*>(backend->context)->stream();
}

// 1 when this backend's GPU runs qi21_attn_wmma (an RDNA3 / gfx11 card in a HIP build)
int qi21_has_wmma_attn(ggml_backend_t backend) {
#if defined(GGML_USE_HIP)
    const int dev = static_cast<ggml_backend_cuda_context*>(backend->context)->device;
    return GGML_CUDA_CC_IS_RDNA3(ggml_cuda_info().devices[dev].cc) ? 1 : 0;
#else
    (void) backend;
    return 0;
#endif
}

int qi21_res_ln_mod(void* stream, float* x, const float* h, const float* g, float* out, const float* m, int T, int C,
                    float eps) {
    if (C > 32 * 256) return 1;
    if (h != nullptr && out != nullptr)
        k_res_ln_mod<true, true><<<T, 256, 0, qi_s(stream)>>>(x, h, g, out, m, C, eps);
    else if (h != nullptr)
        k_res_ln_mod<true, false><<<T, 256, 0, qi_s(stream)>>>(x, h, g, out, m, C, eps);
    else
        k_res_ln_mod<false, true><<<T, 256, 0, qi_s(stream)>>>(x, h, g, out, m, C, eps);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

int qi21_qkv_post(void* stream, const float* q, const float* k, const float* v, int ld, const float* wq, const float* wk,
                  const float* pe, void* qo, void* qlo, int q_f16, int q_tstride, void* ko, void* vo, int kv_tstride,
                  int kv_off, int T, int H, float eps) {
    const dim3 grid(T, H);
    if (q_f16)
        k_qkv_post<half><<<grid, QI_D, 0, qi_s(stream)>>>(q, k, v, ld, wq, wk, pe, static_cast<half*>(qo),
                                                          static_cast<half*>(qlo), q_tstride,
                                                          static_cast<half*>(ko), static_cast<half*>(vo), kv_tstride,
                                                          kv_off, eps);
    else
        k_qkv_post<float><<<grid, QI_D, 0, qi_s(stream)>>>(q, k, v, ld, wq, wk, pe, static_cast<float*>(qo), nullptr,
                                                           q_tstride,
                                                           static_cast<half*>(ko), static_cast<half*>(vo), kv_tstride,
                                                           kv_off, eps);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

int qi21_swiglu(void* stream, const float* gu, float* out, int T, int F) {
    const int64_t n = (int64_t) T * F;
    k_swiglu<<<(unsigned) ((n + 255) / 256), 256, 0, qi_s(stream)>>>(gu, out, F, n);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

int qi21_attn_causal(void* stream, const float* q, int q_tstride, const void* k, const void* v, int kv_tstride,
                     float* out, int H, int Lq, int Lk, int causal) {
    const size_t smem = (size_t) Lk * sizeof(float);
    if (smem > 48 * 1024) return 1;
    k_attn_causal<<<dim3(Lq, H), QI_D, smem, qi_s(stream)>>>(q, q_tstride, static_cast<const half*>(k),
                                                             static_cast<const half*>(v), kv_tstride, out, H, Lk,
                                                             causal, 1.f / sqrtf((float) QI_D));
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

int qi21_attn_wmma(void* stream, const void* q, const void* qlo, int q_tstride, const void* k, const void* v,
                   int kv_tstride, float* out, int H, int nq, int nkv) {
#if defined(GGML_USE_HIP)
    const dim3 grid((nq + FA_BQ - 1) / FA_BQ, H);
    const float sl = (float) (1.0 / sqrt((double) QI_D) * 1.4426950408889634);
    if (qlo != nullptr)
        k_attn_wmma<true><<<grid, FA_NW * 32, 0, qi_s(stream)>>>(static_cast<const _Float16*>(q),
                                                                 static_cast<const _Float16*>(qlo), q_tstride,
                                                                 static_cast<const _Float16*>(k),
                                                                 static_cast<const _Float16*>(v), kv_tstride, out, H,
                                                                 nq, nkv, sl);
    else
        k_attn_wmma<false><<<grid, FA_NW * 32, 0, qi_s(stream)>>>(static_cast<const _Float16*>(q), nullptr, q_tstride,
                                                                  static_cast<const _Float16*>(k),
                                                                  static_cast<const _Float16*>(v), kv_tstride, out, H,
                                                                  nq, nkv, sl);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
#else
    (void) stream; (void) q; (void) qlo; (void) q_tstride; (void) k; (void) v; (void) kv_tstride; (void) out; (void) H; (void) nq;
    (void) nkv;
    return 1;
#endif
}

int qi21_memcpy_d2d(void* stream, void* dst, const void* src, size_t n) {
    return cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, qi_s(stream)) == cudaSuccess ? 0 : 2;
}

}  // extern "C"
