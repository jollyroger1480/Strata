// esimd_moe_parity (151-b70): the fused int8 DPAS expert products (prefill/esimd_moe.dp.cpp) against
//   (a) the engine's dequantize + oneMKL path (FP16 weights, FP16 activations), the path they replace,
//   (b) an FP64 host reference from the blobs (IQ3_S gate/up, IQ4_NL down) with the unquantized activations,
//   (c) an exact host reference on the SAME int8 activations the device produced (isolates the kernel arithmetic from the
//       activation quantization: only the float accumulation order differs).
// and times both paths per expert.   esimd_moe_parity [mean_rows=80] [experts=16] [reps=10] [seed=1]
// Exit status 1 when (c) is off by more than 1e-5 of the row's scale or (a)/(b) exceed the int8 activation error budget.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/iq_dequant_expert.hpp"
#include "strata/prefill/esimd_moe.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/sycl_queue.hpp"
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <dpct/blas_utils.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define GGML_TABLE_BEGIN(type, name, size) static const type name[size] = {
#define GGML_TABLE_END() };
#include "../prefill/iq_tables.inc"

namespace i8 = strata::prefill::i8;
static uint16_t f2h(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }
static float h2f(uint16_t h) { return (float) sycl::bit_cast<sycl::half>(h); }

// host decode of one IQ3_S row (K values) -> double
static void dec_iq3s(const uint8_t* row, int K, double* out) {
    for (int sb = 0; sb < K / 256; ++sb) {
        const uint8_t* blk = row + sb * 110;
        uint16_t dh; std::memcpy(&dh, blk, 2);
        const uint8_t *qs = blk + 2, *qh = blk + 66, *sgn = blk + 74, *scl = blk + 106;
        for (int ib = 0; ib < 8; ++ib) {
            const double d = (double) h2f(dh) * (1 + 2 * ((scl[ib / 2] >> (4 * (ib % 2))) & 0xf));
            for (int r = 0; r < 8; ++r) {
                const int idx = qs[8 * ib + r] | (((qh[ib] >> r) & 1) << 8);
                const uint8_t* g = (const uint8_t*) &iq3s_grid[idx];
                const uint8_t sbyte = sgn[4 * ib + r / 2];
                for (int j = 0; j < 4; ++j) {
                    const int kk = 4 * (r & 1) + j;
                    out[sb * 256 + ib * 32 + 4 * r + j] = d * ((sbyte >> kk & 1) ? -(int) g[j] : (int) g[j]);
                }
            }
        }
    }
}
static void dec_iq4nl(const uint8_t* row, int K, double* out) {
    for (int b = 0; b < K / 32; ++b) {
        const uint8_t* blk = row + b * 18;
        uint16_t dh; std::memcpy(&dh, blk, 2);
        for (int j = 0; j < 16; ++j) {
            out[b * 32 + j] = (double) h2f(dh) * kvalues_iq4nl[blk[2 + j] & 15];
            out[b * 32 + 16 + j] = (double) h2f(dh) * kvalues_iq4nl[blk[2 + j] >> 4];
        }
    }
}

int main(int argc, char** argv) {
    const int mean = argc > 1 ? std::atoi(argv[1]) : 80;
    const int n = argc > 2 ? std::atoi(argv[2]) : 16;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 10;
    const int seed = argc > 4 ? std::atoi(argv[4]) : 1;
    sycl::queue* q = &dpct::get_in_order_queue();
    if (!i8::available(q)) { std::printf("esimd_moe_parity: SKIP (no sub-group 16 / s8 DPAS device)\n"); return 0; }
    dpct::blas::descriptor_ptr hd = new dpct::blas::descriptor();
    hd->set_queue(q);
    const int K = 2560, NFF = 640, NE = 2560;
    std::mt19937 rng(seed);
    std::vector<int32_t> cnt(n);
    std::gamma_distribution<double> gd(1.5, mean / 1.5);
    int tot = 0;
    for (auto& c : cnt) { c = std::max(1, (int) gd(rng)); tot += c; }
    if (n >= 3) { cnt[0] = 1; cnt[1] = 7; cnt[2] = 9; tot = 0; for (int c : cnt) tot += c; }   // pad edge cases
    i8::Layout lo;
    i8::make_layout(lo, cnt.data(), n);

    const size_t gate_b = (size_t) NFF * (K / 256) * 110, down_b = (size_t) NE * (NFF / 32) * 18;
    const size_t blob_b = 2 * gate_b + down_b;
    std::vector<std::vector<uint8_t>> hb(n);
    std::vector<uint8_t*> db(n);
    for (int i = 0; i < n; ++i) {
        hb[i].resize(blob_b);
        for (auto& x : hb[i]) x = (uint8_t) rng();
        for (size_t o = 0; o < 2 * gate_b; o += 110) { const uint16_t d = f2h(0.0004f + 0.00002f * (rng() % 50)); std::memcpy(&hb[i][o], &d, 2); }
        for (size_t o = 2 * gate_b; o < blob_b; o += 18) { const uint16_t d = f2h(0.0004f + 0.00002f * (rng() % 50)); std::memcpy(&hb[i][o], &d, 2); }
        db[i] = sycl::malloc_device<uint8_t>(blob_b + 256, *q);
        q->memcpy(db[i], hb[i].data(), blob_b);
    }
    std::vector<uint16_t> hx((size_t) tot * K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& x : hx) { float v = nd(rng) * 0.5f; if (rng() % 97 == 0) v *= 12.f; x = f2h(v); }   // a few outliers
    uint16_t* dxh = sycl::malloc_device<uint16_t>(hx.size(), *q);
    q->memcpy(dxh, hx.data(), hx.size() * 2);
    int8_t* Xq = sycl::malloc_device<int8_t>(i8::xq_bytes(lo, K), *q);
    float* dX = sycl::malloc_device<float>(i8::dx_bytes(lo, K) / 4, *q);
    int8_t* Hq = sycl::malloc_device<int8_t>(i8::xq_bytes(lo, NFF), *q);
    float* dH = sycl::malloc_device<float>(i8::dx_bytes(lo, NFF) / 4, *q);
    float *GU = sycl::malloc_device<float>((size_t) tot * 2 * NFF, *q), *Dm = sycl::malloc_device<float>((size_t) tot * NE, *q);
    float *GUr = sycl::malloc_device<float>((size_t) tot * 2 * NFF, *q), *Dmr = sycl::malloc_device<float>((size_t) tot * NE, *q);
    uint16_t *Hh = sycl::malloc_device<uint16_t>((size_t) tot * NFF, *q);
    uint16_t* dwgu = sycl::malloc_device<uint16_t>((size_t) 2 * NFF * K, *q);
    uint16_t* dwd = sycl::malloc_device<uint16_t>((size_t) NE * NFF, *q);
    q->wait();
    std::vector<const uint8_t*> pb(n);
    if (std::getenv("I8_STAGE")) { for (int i = 0; i < n && i < 4; ++i) std::printf("blob %d at %p size %zu\n", i, (void*) db[i], blob_b); }
    for (int i = 0; i < n; ++i) pb[i] = db[i];

    auto fused = [&](bool all = true) {
        i8::quantize_rows(dxh, lo, K, Xq, dX, q);
        if (!i8::gate_up(21, pb.data(), 2 * 0 + gate_b, NFF, lo, K, Xq, dX, GU, q)) { std::printf("gate_up refused\n"); std::exit(2); }
        i8::swiglu_quantize(GU, lo, NFF, Hq, dH, q);
        if (!i8::down(20, pb.data(), 2 * gate_b, NE, lo, NFF, Hq, dH, Dm, q)) { std::printf("down refused\n"); std::exit(2); }
        (void) all;
    };
    auto ref = [&]() {
        int64_t o = 0;
        for (int i = 0; i < n; ++i) {
            strata::kernels::iq_dequant_expert_f16(21, 20, db[i], db[i] + gate_b, db[i] + 2 * gate_b, NFF, K, dwgu, dwd, q);
            const float a = 1.f, b = 0.f;
            dpct::blas::gemm(hd, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, 2 * NFF, cnt[i], K, &a, dwgu,
                             dpct::library_data_t::real_half, K, dxh + o * K, dpct::library_data_t::real_half, K, &b,
                             GUr + o * 2 * NFF, dpct::library_data_t::real_float, 2 * NFF, dpct::compute_type::f32);
            strata::prefill::swiglu_interleaved(GUr + o * 2 * NFF, Hh + o * NFF, cnt[i], q);
            dpct::blas::gemm(hd, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, NE, cnt[i], NFF, &a, dwd,
                             dpct::library_data_t::real_half, NFF, Hh + o * NFF, dpct::library_data_t::real_half, NFF, &b,
                             Dmr + o * NE, dpct::library_data_t::real_float, NE, dpct::compute_type::f32);
            o += cnt[i];
        }
    };
    auto tm = [&](auto f) {
        f(); q->wait();
        const auto t0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.5) { f(); q->wait(); }   // GPU clocks up
        const auto a = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) f();
        q->wait();
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - a).count() / reps;
    };

    if (const char* st = std::getenv("I8_STAGE")) {   // bring-up: run the first N stages, synchronising after each
        const int ns = std::atoi(st);
        i8::quantize_rows(dxh, lo, K, Xq, dX, q); q->wait(); std::printf("stage 1 quantize ok%c", 10);
        if (ns >= 2) { i8::gate_up(21, pb.data(), gate_b, NFF, lo, K, Xq, dX, GU, q); q->wait(); std::printf("stage 2 gate_up ok%c", 10); }
        if (ns >= 3) { i8::swiglu_quantize(GU, lo, NFF, Hq, dH, q); q->wait(); std::printf("stage 3 swiglu ok%c", 10); }
        if (ns >= 4) { i8::down(20, pb.data(), 2 * gate_b, NE, lo, NFF, Hq, dH, Dm, q); q->wait(); std::printf("stage 4 down ok%c", 10); }
        if (const char* dd = std::getenv("STRATA_I8_DBG"); dd && (std::atoi(dd) & 128)) {   // the down kernel stored rowoff + P per thread
            std::vector<uint32_t> o((size_t) 480 * 16);
            q->memcpy(o.data(), Dm, o.size() * 4).wait();
            uint32_t mx = 0, mn = ~0u;
            for (int th = 0; th < 480; ++th) for (int l = 0; l < 16; ++l) { mx = std::max(mx, o[th * 16 + l]); mn = std::min(mn, o[th * 16 + l]); }
            std::printf("rowoff+P: min %u max %u%c", mn, mx, 10); for (int th : {0, 1, 158, 159, 160, 161, 319, 320, 321, 479}) std::printf("  t%d lane0 %u row %u%c", th, o[th * 16], (o[th * 16] - 1408000) / 360, 10);
        }
        return 0;
    }
    fused(); q->wait();
    // ---- (c) exact reference on the device's int8 activations
    std::vector<int8_t> hXq(i8::xq_bytes(lo, K));
    std::vector<float> hdX(i8::dx_bytes(lo, K) / 4), hGU((size_t) tot * 2 * NFF);
    q->memcpy(hXq.data(), Xq, hXq.size());
    q->memcpy(hdX.data(), dX, hdX.size() * 4);
    q->memcpy(hGU.data(), GU, hGU.size() * 4).wait();
    const int nkb = K / 32;
    auto xq_at = [&](int tile, int r, int k) { return (int) hXq[(((size_t) tile * nkb + k / 32) * 8 + r) * 32 + (k % 32)]; };
    auto dx_at = [&](int tile, int r, int kb) { return (double) hdX[((size_t) tile * nkb + kb) * 8 + r]; };
    int fails = 0;
    double err_c = 0, mx_c = 0, err_b = 0, rms_b = 0, rms_bref = 0, mx_b = 0;
    std::vector<double> wrow(K);
    {
        int sampled = 0;
        for (int e = 0; e < n; ++e) {
            for (int rl = 0; rl < cnt[e]; rl += (cnt[e] > 12 ? 5 : 1)) {
                const int tile = lo.tile0[e] + rl / 8, r = rl % 8, row = lo.row0[e] + rl;
                for (int v = 0; v < 2 * NFF; v += 37) {
                    dec_iq3s(&hb[e][(size_t) (v & 1 ? gate_b : 0) + (size_t) (v >> 1) * (K / 256) * 110], K, wrow.data());
                    double refc = 0, refb = 0;
                    for (int kb = 0; kb < nkb; ++kb) {
                        double s = 0;
                        for (int k = 0; k < 32; ++k) s += (double) xq_at(tile, r, kb * 32 + k) * wrow[kb * 32 + k];
                        refc += s * dx_at(tile, r, kb);
                    }
                    for (int k = 0; k < K; ++k) refb += (double) h2f(hx[(size_t) row * K + k]) * wrow[k];
                    const double got = hGU[(size_t) row * 2 * NFF + v];
                    err_c = std::max(err_c, std::fabs(got - refc)); mx_c = std::max(mx_c, std::fabs(refc));
                    err_b += (got - refb) * (got - refb); rms_bref += refb * refb; mx_b = std::max(mx_b, std::fabs(refb));
                    ++sampled;
                }
            }
        }
        const bool ok = err_c <= 2e-5 * mx_c && std::sqrt(err_b / rms_bref) < 0.02;
        fails += !ok;
        std::printf("gate/up IQ3_S  : exact-int8 reference max err %.2e of %.2e (%.1e)   vs FP64 true: rms rel err %.2e (%d samples)  %s\n",
                    err_c, mx_c, err_c / mx_c, std::sqrt(err_b / rms_bref), sampled, ok ? "OK" : "FAIL");
    }
    {   // down: needs the device's Hq / dH and the true H from the device GU (float math on the host)
        std::vector<int8_t> hHq(i8::xq_bytes(lo, NFF));
        std::vector<float> hdH(i8::dx_bytes(lo, NFF) / 4), hDm((size_t) tot * NE);
        q->memcpy(hHq.data(), Hq, hHq.size());
        q->memcpy(hdH.data(), dH, hdH.size() * 4);
        q->memcpy(hDm.data(), Dm, hDm.size() * 4).wait();
        const int nkh = NFF / 32;
        double ec = 0, mc = 0, eb = 0, rb = 0, hq_bad = 0;
        std::vector<double> drow(NFF);
        int sampled = 0;
        for (int e = 0; e < n; ++e)
            for (int rl = 0; rl < cnt[e]; rl += (cnt[e] > 12 ? 5 : 1)) {
                const int tile = lo.tile0[e] + rl / 8, r = rl % 8, row = lo.row0[e] + rl;
                // the true H (double) of this row, and a check of the device's swiglu quantization
                std::vector<double> H(NFF);
                for (int j = 0; j < NFF; ++j) {
                    const double g = hGU[(size_t) row * 2 * NFF + 2 * j], u = hGU[(size_t) row * 2 * NFF + 2 * j + 1];
                    H[j] = g / (1.0 + std::exp(-g)) * u;
                }
                for (int kb = 0; kb < nkh; ++kb) {
                    double amax = 0; for (int k = 0; k < 32; ++k) amax = std::max(amax, std::fabs(H[kb * 32 + k]));
                    const double dd = hdH[((size_t) tile * nkh + kb) * 8 + r];
                    if (std::fabs(dd - amax / 127.0) > 1e-5 * amax + 1e-30) hq_bad += 1;
                }
                for (int v = 0; v < NE; v += 41) {
                    dec_iq4nl(&hb[e][2 * gate_b + (size_t) v * nkh * 18], NFF, drow.data());
                    double refc = 0, refb = 0;
                    for (int kb = 0; kb < nkh; ++kb) {
                        double s = 0;
                        for (int k = 0; k < 32; ++k) s += (double) hHq[(((size_t) tile * nkh + kb) * 8 + r) * 32 + k] * drow[kb * 32 + k];
                        refc += s * hdH[((size_t) tile * nkh + kb) * 8 + r];
                    }
                    for (int k = 0; k < NFF; ++k) refb += H[k] * drow[k];
                    const double got = hDm[(size_t) row * NE + v];
                    ec = std::max(ec, std::fabs(got - refc)); mc = std::max(mc, std::fabs(refc));
                    eb += (got - refb) * (got - refb); rb += refb * refb;
                    ++sampled;
                }
            }
        const bool ok = ec <= 2e-5 * mc && std::sqrt(eb / rb) < 0.03 && hq_bad == 0;
        fails += !ok;
        std::printf("down IQ4_NL    : exact-int8 reference max err %.2e of %.2e (%.1e)   vs FP64 true: rms rel err %.2e, swiglu scale mismatches %.0f (%d samples)  %s\n",
                    ec, mc, ec / mc, std::sqrt(eb / rb), hq_bad, sampled, ok ? "OK" : "FAIL");
    }
    // ---- (a) against the dequantize + oneMKL path, whole chain
    ref(); q->wait();
    {
        std::vector<float> a((size_t) tot * NE), b((size_t) tot * NE);
        q->memcpy(a.data(), Dmr, a.size() * 4);
        q->memcpy(b.data(), Dm, b.size() * 4).wait();
        double e2 = 0, r2 = 0;
        for (size_t i = 0; i < a.size(); ++i) { e2 += (double) (a[i] - b[i]) * (a[i] - b[i]); r2 += (double) a[i] * a[i]; }
        std::printf("chain vs dequant + oneMKL (FP16 path): rms rel diff %.2e (rows %d)\n", std::sqrt(e2 / r2), tot);
    }
    const double t_ref = tm(ref), t_f = tm([&] { fused(); });
    const double t_q = tm([&] { i8::quantize_rows(dxh, lo, K, Xq, dX, q); });
    const double t_gu = tm([&] { i8::gate_up(21, pb.data(), gate_b, NFF, lo, K, Xq, dX, GU, q); });
    const double t_sw = tm([&] { i8::swiglu_quantize(GU, lo, NFF, Hq, dH, q); });
    const double t_dn = tm([&] { i8::down(20, pb.data(), 2 * gate_b, NE, lo, NFF, Hq, dH, Dm, q); });
    std::printf("rows %d (%d experts, mean %.0f): dequant+oneMKL chain %.1f us/expert | fused int8 chain %.1f us/expert (%.2fx)  [quantize %.1f, gate/up %.1f, swiglu+q %.1f, down %.1f]\n",
                tot, n, (double) tot / n, t_ref / n, t_f / n, t_ref / t_f, t_q / n, t_gu / n, t_sw / n, t_dn / n);
    std::printf("%s\n", fails ? "FAILED" : "all OK");
    return fails ? 1 : 0;
}
