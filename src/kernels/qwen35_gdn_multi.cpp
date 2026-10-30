// qwen35 GDN closing norm (SiLU, not sigmoid) and a value-head split across HIP devices.
// No model weights. Exits non-zero when the GPU disagrees with the CPU reference or the
// second device's slice disagrees with a one-device run.
#include "strata/kernels/gdn.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

float* up(const std::vector<float>& h) {
    float* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(float)), "malloc");
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice), "upload");
    return d;
}

double rel_l1(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += std::fabs((double) a[i] - (double) b[i]);
        den += std::fabs((double) a[i]);
    }
    return den == 0 ? num : num / den;
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

}  // namespace

int main() {
    int n = 0;
    ck(cudaGetDeviceCount(&n), "cudaGetDeviceCount");
    if (n < 1) {
        std::fprintf(stderr, "no HIP device\n");
        return 1;
    }
    std::printf("devices %d\n", n);
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp p{};
        ck(cudaGetDeviceProperties(&p, i), "prop");
        std::printf("  %d %s %.1f GiB\n", i, p.name, (double) p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }

    constexpr int S = 128, hv = 8;
    const float eps = 1e-6f;
    std::mt19937 rng(35);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> o((size_t) hv * S), z((size_t) hv * S), sn(S), y_silu((size_t) hv * S), y_sig((size_t) hv * S);
    for (auto& v : o) v = nd(rng);
    for (auto& v : z) v = nd(rng);
    for (auto& v : sn) v = 1.0f + 0.1f * nd(rng);
    for (int h = 0; h < hv; ++h) {
        double acc = 0;
        for (int i = 0; i < S; ++i) acc += (double) o[(size_t) h * S + i] * (double) o[(size_t) h * S + i];
        const float inv = (float) (1.0 / std::sqrt(acc / S + (double) eps));
        for (int i = 0; i < S; ++i) {
            const size_t ix = (size_t) h * S + i;
            y_silu[ix] = o[ix] * inv * sn[(size_t) i] * silu(z[ix]);
            y_sig[ix] = o[ix] * inv * sn[(size_t) i] * sigmoid(z[ix]);
        }
    }
    const double apart = rel_l1(y_silu, y_sig);
    std::printf("sigmoid vs silu rel %.4f\n", apart);
    if (!(apart > 0.05)) {
        std::fprintf(stderr, "sigmoid and silu are not observably different\n");
        return 1;
    }

    ck(cudaSetDevice(0), "set0");
    float* d_o = up(o);
    float* d_z = up(z);
    float* d_sn = up(sn);
    float* d_y = nullptr;
    ck(cudaMalloc(&d_y, y_silu.size() * sizeof(float)), "y");
    strata::kernels::gdn_out_norm_silu(d_o, d_z, d_sn, d_y, hv, S, eps, nullptr);
    std::vector<float> got(y_silu.size());
    ck(cudaMemcpy(got.data(), d_y, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "download");
    const double rel = rel_l1(y_silu, got);
    std::printf("device0 silu vs cpu rel %.3e\n", rel);
    if (!(rel <= 1e-5)) {
        std::fprintf(stderr, "silu kernel missed the cpu reference\n");
        return 1;
    }
    strata::kernels::gdn_out_norm(d_o, d_z, d_sn, d_y, hv, S, eps, nullptr);
    ck(cudaMemcpy(got.data(), d_y, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "download sig");
    const double rel_sig = rel_l1(y_silu, got);
    std::printf("device0 sigmoid kernel vs silu ref rel %.4f\n", rel_sig);
    if (!(rel_sig > 0.05)) {
        std::fprintf(stderr, "sigmoid kernel matched the silu reference\n");
        return 1;
    }
    ck(cudaFree(d_o), "free o");
    ck(cudaFree(d_z), "free z");
    ck(cudaFree(d_sn), "free sn");
    ck(cudaFree(d_y), "free y");

    if (n < 2) {
        std::fprintf(stderr, "need two HIP devices for the head split\n");
        return 1;
    }
    const int devices[2] = {0, 1};
    std::vector<float> split(y_silu.size());
    strata::kernels::gdn_out_norm_silu_split(o.data(), z.data(), sn.data(), split.data(), hv, S, eps, devices, 2);
    const double rel_split = rel_l1(y_silu, split);
    std::printf("two-device silu vs cpu rel %.3e\n", rel_split);
    if (!(rel_split <= 1e-5)) {
        std::fprintf(stderr, "head split missed the cpu reference\n");
        return 1;
    }

    constexpr int hk = 4;
    strata::kernels::GdnShapes shapes{S, hk, hv};
    std::vector<float> state((size_t) S * hv * S), q((size_t) hk * S), kq((size_t) hk * S), vv((size_t) hv * S);
    std::vector<float> gate(hv), beta(hv), out_one((size_t) hv * S), out_two((size_t) hv * S);
    for (auto& v : state) v = 0.01f * nd(rng);
    for (auto& v : q) v = nd(rng);
    for (auto& v : kq) v = nd(rng);
    for (auto& v : vv) v = nd(rng);
    for (auto& v : gate) v = -0.1f * std::fabs(nd(rng));
    for (auto& v : beta) v = sigmoid(nd(rng));
    std::vector<float> state_one = state, state_two = state;
    ck(cudaSetDevice(0), "set0b");
    float* d_st = up(state_one);
    float* d_q = up(q);
    float* d_k = up(kq);
    float* d_v = up(vv);
    float* d_g = up(gate);
    float* d_b = up(beta);
    float* d_out = nullptr;
    ck(cudaMalloc(&d_out, out_one.size() * sizeof(float)), "out");
    strata::kernels::gdn_step(d_st, d_q, d_k, d_v, d_g, d_b, d_out, shapes, nullptr);
    ck(cudaMemcpy(state_one.data(), d_st, state_one.size() * sizeof(float), cudaMemcpyDeviceToHost), "st");
    ck(cudaMemcpy(out_one.data(), d_out, out_one.size() * sizeof(float), cudaMemcpyDeviceToHost), "out");
    strata::kernels::gdn_step_split(state_two.data(), q.data(), kq.data(), vv.data(), gate.data(), beta.data(),
                                    out_two.data(), shapes, devices, 2);
    const double rel_o = rel_l1(out_one, out_two);
    const double rel_st = rel_l1(state_one, state_two);
    std::printf("step split vs device0  o rel %.3e  state rel %.3e\n", rel_o, rel_st);
    if (!(rel_o <= 1e-6 && rel_st <= 1e-6)) {
        std::fprintf(stderr, "gdn_step head split disagrees with one device\n");
        return 1;
    }
    std::printf("qwen35_gdn_multi OK\n");
    return 0;
}
