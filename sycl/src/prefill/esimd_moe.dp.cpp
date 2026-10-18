// 151-b70: fused int8 DPAS expert products (ESIMD, Battlemage) - see include/strata/prefill/esimd_moe.hpp.
//
// Behaviour note (docs/INTEL.md, "int8 expert path"):
//  * Tile.  One hardware thread owns 16 output rows (a DPAS N tile; for gate/up the 16 virtual rows alternate gate/up) and up to
//    MT x 8 activation rows (MT = 2, 4, 8 or 12, picked per group from the experts' mean row count).  It walks K in steps of 32:
//    decode the 16 weight rows' 32 values into the B operand (registers only), then for each of its row tiles one s8 DPAS 8x16x32
//    into a zeroed int32 tile, converted to float and scaled by (weight scale of the row and 32-block) x (activation scale of the
//    row and 32-block) into a float accumulator.  A 128-wide work-group tile (the owner's 16/32/64 rows x 128 outputs) is eight of
//    these threads side by side; there is no shared memory and no barrier.
//  * Decode once per tile: the B operand is decoded once per thread and applied to all its MT row tiles.  An expert with more rows
//    than 8 x MT is split over row-block threads, each decoding again (the redundancy is ceil(rows / (8 MT)); MT 12 keeps it 1 up
//    to 96 rows).
//  * Scales.  Activations: q8_1 as llama.cpp (d = amax / 127, q = round half away from zero), one float per row and 32 of K.
//    Weights: IQ3_S d * (1 + 2 s) per 32 (exact in float), IQ4_NL d per 32; the int8 values are the codebook entries with their
//    signs, never rounded.  The int32 dot product of a 32-block is at most 32 * 127 * 15 (IQ3_S) so it is exact, and int32 -> float
//    is exact below 2^24.  The float accumulation order is fixed (K ascending).
//  * Edge cases.  Rows are padded to tiles of 8 with zeros (stores are masked to the real rows); a group of up to 16 experts per
//    launch; K a multiple of 256 (IQ3_S) / 32 (IQ4_NL); output rows a multiple of 16.  Blob reads never go past a matrix end
//    (the last dword of a block is read as two dwords only when the block is not dword aligned).
//  * Fallback.  available() is false without sub-group 16 DPAS (a non-Battlemage device) and the caller keeps the dequantize +
//    oneMKL path; gate_up()/down() return false for a type / shape they do not handle.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <type_traits>
#include <string>
#include <unordered_map>
#include "strata/sycl_queue.hpp"
#include "strata/prefill/esimd_moe.hpp"

namespace strata::prefill::i8 {
namespace {
#define GGML_TABLE_BEGIN(type, name, size) static const type name[size] = {
#define GGML_TABLE_END() };
#include "iq_tables.inc"
#undef GGML_TABLE_BEGIN
#undef GGML_TABLE_END

using namespace sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
using half = sycl::half;

constexpr int BLK_IQ3S = 110;   // sizeof(block_iq3_s)
constexpr int BLK_IQ4NL = 18;   // sizeof(block_iq4_nl)

struct Tables {
    uint32_t* grid3s = nullptr;   // iq3s_grid[512]
    uint32_t* iq4 = nullptr;      // [256]: byte b -> kvalues[b & 15] in byte 0, kvalues[b >> 4] in byte 1
};
Tables& tables(sycl::queue& q) {
    static std::mutex mu;
    static Tables t;   // one device per process in the port
    std::lock_guard<std::mutex> lk(mu);
    if (!t.grid3s) {
        t.grid3s = sycl::malloc_device<uint32_t>(512, q);
        t.iq4 = sycl::malloc_device<uint32_t>(256, q);
        uint32_t h4[256];
        for (int b = 0; b < 256; ++b)
            h4[b] = (uint32_t) (uint8_t) kvalues_iq4nl[b & 15] | ((uint32_t) (uint8_t) kvalues_iq4nl[b >> 4] << 8);
        q.memcpy(t.grid3s, iq3s_grid, 512 * 4);
        q.memcpy(t.iq4, h4, 256 * 4);
        q.wait();
    }
    return t;
}

// unaligned 32-bit read at a 2-byte aligned byte offset (per lane) through two dword gathers
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> ld32(const uint8_t* base, simd<uint32_t, 16> off) {
    simd<uint32_t, 16> a = off & ~3u;
    simd<uint32_t, 16> lo = gather<uint32_t, 16>(reinterpret_cast<const uint32_t*>(base), a);
    simd<uint32_t, 16> hi = gather<uint32_t, 16>(reinterpret_cast<const uint32_t*>(base), a + 4u);
    simd<uint32_t, 16> sh = (off & 2u) << 3;   // 0 or 16
    return (lo >> sh) | ((hi << 16) << (16u - sh));
}
SYCL_ESIMD_FUNCTION inline simd<uint32_t, 16> ld16(const uint8_t* base, simd<uint32_t, 16> off) {
    simd<uint32_t, 16> a = off & ~3u;
    simd<uint32_t, 16> lo = gather<uint32_t, 16>(reinterpret_cast<const uint32_t*>(base), a);
    return (lo >> ((off & 2u) << 3)) & 0xffffu;
}
SYCL_ESIMD_FUNCTION inline simd<float, 16> half_of(simd<uint32_t, 16> h16) {
    simd<uint16_t, 16> lo = h16.template bit_cast_view<uint16_t>().template select<16, 2>(0);
    simd<half, 16> hh = lo.template bit_cast_view<half>();
    simd<float, 16> f = hh;
    return f;
}
SYCL_ESIMD_FUNCTION inline simd<int, 128> dpas_s8(simd<uint32_t, 128> b, simd<int, 64> a) {
    simd<int, 128> bi = b.template bit_cast_view<int>();
    return xmx::dpas<8, 8, int, int, int, int, xmx::dpas_argument_type::s8, xmx::dpas_argument_type::s8>(simd<int, 128>(0), bi, a);
}

struct GArgs {
    const uint8_t* blob[kMaxGroup];
    uint32_t sec_off;                 // gate/up: byte offset of the up rows; down: byte offset of the down rows
    int n;
    int row0[kMaxGroup + 1];
    int tile0[kMaxGroup + 1];
    int rb0[kMaxGroup + 1];           // first row block of expert i
    const int8_t* Xq;
    const float* dx;
    float* Y;
    const uint32_t* tab;
    int ntiles;                       // output tiles of 16
    int nkb;                          // K / 32
    int rowb;                         // bytes per weight row
    int ldy;
    int dbg;                          // bring-up switches (STRATA_I8_DBG): 1 skip the decode, 2 skip the activation loads, 4 skip the stores
};


// The expert a work item belongs to, with constant-index reads of the kernel argument arrays (a dynamic index into the argument
// struct gave a garbage pointer in an ESIMD kernel on the B70: the first weight read faulted the GPU).
struct Pick { int e, row0, cnt, tile0, rb0; const uint8_t* blob; };
SYCL_ESIMD_FUNCTION inline Pick pick_expert(const GArgs& a, int rbg) {
    Pick p{0, a.row0[0], a.row0[1] - a.row0[0], a.tile0[0], a.rb0[0], a.blob[0]};
#pragma unroll
    for (int i = 1; i < kMaxGroup; ++i)
        if (i < a.n && rbg >= a.rb0[i]) p = Pick{i, a.row0[i], a.row0[i + 1] - a.row0[i], a.tile0[i], a.rb0[i], a.blob[i]};
    return p;
}

struct LaunchInfo { int nthreads; };
template <int MT>
void fill_args(GArgs& a, const Layout& lo, int ntiles_out, int& nthreads) {
    int rb = 0;
    for (int i = 0; i < lo.n; ++i) {
        a.row0[i] = lo.row0[i];
        a.tile0[i] = lo.tile0[i];
        a.rb0[i] = rb;
        const int ntl = (lo.cnt[i] + 7) / 8;
        rb += (ntl + MT - 1) / MT;
    }
    a.row0[lo.n] = lo.row0[lo.n];
    a.tile0[lo.n] = lo.tile0[lo.n];
    a.rb0[lo.n] = rb;
    a.n = lo.n;
    nthreads = rb * ntiles_out;
}

template <int MT> struct TagGuS3;
template <int MT> struct TagDnNL;

// ------------------------------------------------------------------------------------------------ gate/up, IQ3_S
template <int MT>
void launch_gu_iq3s(sycl::queue& q, const GArgs a, int nthreads, const uint32_t* grid) {
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagGuS3<MT>>(sycl::range<2>((size_t) (nthreads / a.ntiles), (size_t) a.ntiles), props, [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {
            // a 2-D grid: this device's ESIMD compiler gave `t % n` = t for a run-time n (integer / and % by a non-constant are not usable here)
            const int rbg = (int) id[0], nt = (int) id[1];
            const int t = rbg * a.ntiles + nt;
            const Pick pk = pick_expert(a, rbg);
            const int e = pk.e;
            const int cnt = pk.cnt;
            const int ntl = (cnt + 7) >> 3;
            const int tl0 = (rbg - pk.rb0) * MT;
            const int nvalid = (ntl - tl0) < MT ? (ntl - tl0) : MT;
            const int tbase = pk.tile0 + tl0;
            const uint8_t* We = pk.blob;
            const int nsb = a.nkb >> 3;
            simd<uint32_t, 16> lane(0, 1);
            simd<uint32_t, 16> rowoff = (lane & 1u) * a.sec_off + ((lane >> 1) + (uint32_t) (8 * nt)) * (uint32_t) a.rowb;
            simd<float, 128 * MT> acc = 0.f;
            const int* Xi = reinterpret_cast<const int*>(a.Xq);
            for (int sb = 0; sb < ((a.dbg & 8) ? 0 : nsb); ++sb) {
                const uint32_t P = (uint32_t) sb * BLK_IQ3S;
                simd<float, 16> d = half_of(ld16(We, rowoff + P));
                simd<uint32_t, 16> sc4 = ld32(We, rowoff + (P + 106));
                simd<uint32_t, 16> qh4 = ld32(We, rowoff + (P + 66));
                simd<uint32_t, 16> qh8 = ld32(We, rowoff + (P + 70));
#pragma unroll
                for (int ib = 0; ib < 8; ++ib) {
                    simd<uint32_t, 128> frag;
                    simd<float, 16> swv = d;
                    if (a.dbg & 1) {
                        frag = 0x01010101u;
                    } else {
                    simd<uint32_t, 16> q0 = ld32(We, rowoff + (P + 2 + 8 * ib));
                    simd<uint32_t, 16> q1 = ld32(We, rowoff + (P + 6 + 8 * ib));
                    simd<uint32_t, 16> sg = ld32(We, rowoff + (P + 74 + 4 * ib));
                    simd<uint32_t, 16> qhv = ((ib < 4 ? qh4 : qh8) >> (8 * (ib & 3))) & 0xffu;
                    simd<uint32_t, 16> scn = ((sc4 >> (8 * (ib >> 1))) >> (4 * (ib & 1))) & 0xfu;
                    swv = d * simd<float, 16>(simd<int, 16>(scn * 2u + 1u));
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        simd<uint32_t, 16> qsw = r < 4 ? q0 : q1;
                        simd<uint32_t, 16> idx = ((qsw >> (8 * (r & 3))) & 0xffu) | (((qhv >> r) & 1u) << 8);
                        simd<uint32_t, 16> g = gather<uint32_t, 16>(grid, idx << 2);
                        simd<uint32_t, 16> nib = ((sg >> (8 * (r >> 1))) >> (4 * (r & 1))) & 0xfu;
                        simd<uint32_t, 16> m = ((nib * 0x00204081u) & 0x01010101u);
                        simd<uint32_t, 16> mask = m * 255u;
                        frag.template select<16, 1>(16 * r) = (g ^ mask) + m;
                    }
                    }
                    const int kb = sb * 8 + ib;
#pragma unroll
                    for (int i = 0; i < MT; ++i) {
                        if (i < nvalid) {
                            simd<int, 64> av = 0x01010101;
                            simd<float, 8> dxv = 1.f;
                            if (!(a.dbg & 2)) {
                                av = block_load<int, 64>(Xi + ((size_t) (tbase + i) * a.nkb + kb) * 64);
                                dxv = block_load<float, 8>(a.dx + ((size_t) (tbase + i) * a.nkb + kb) * 8);
                            }
                            simd<float, 128> tf = dpas_s8(frag, av);
#pragma unroll
                            for (int r = 0; r < 8; ++r) {
                                const float dxr = dxv[r];
                                acc.template select<16, 1>(128 * i + 16 * r) += tf.template select<16, 1>(16 * r) * (swv * dxr);
                            }
                        }
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                if (i < nvalid) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int rl = (tl0 + i) * 8 + r;
                        if (rl < cnt && !(a.dbg & 4))
                            block_store<float, 16>(a.Y + ((size_t) (pk.row0 + rl)) * a.ldy + 16 * nt,
                                                   acc.template select<16, 1>(128 * i + 16 * r));
                    }
                }
            }
        });
    });
}

// ------------------------------------------------------------------------------------------------ down, IQ4_NL
template <int MT>
void launch_dn_iq4nl(sycl::queue& q, const GArgs a, int nthreads, const uint32_t* tab) {
    auto props = sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>};
    q.submit([&](sycl::handler& h) {
        h.parallel_for<TagDnNL<MT>>(sycl::range<2>((size_t) (nthreads / a.ntiles), (size_t) a.ntiles), props, [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {
            const int rbg = (int) id[0], nt = (int) id[1];
            const int t = rbg * a.ntiles + nt;
            const Pick pk = (a.dbg & 32) ? Pick{0, a.row0[0], a.row0[1] - a.row0[0], a.tile0[0], a.rb0[0], a.blob[0]} : pick_expert(a, rbg);
            const int e = pk.e;
            const int cnt = pk.cnt;
            const int ntl = (cnt + 7) >> 3;
            const int tl0 = (rbg - pk.rb0) * MT;
            const int nvalid = (ntl - tl0) < MT ? (ntl - tl0) : MT;
            const int tbase = pk.tile0 + tl0;
            const uint8_t* We = pk.blob;
            simd<uint32_t, 16> lane(0, 1);
            simd<uint32_t, 16> rowoff = (lane + (uint32_t) (16 * nt)) * (uint32_t) a.rowb + a.sec_off;
            simd<float, 128 * MT> acc = 0.f;
            const int* Xi = reinterpret_cast<const int*>(a.Xq);
            for (int kb = 0; kb < ((a.dbg & 8) ? 0 : a.nkb); ++kb) {
                const uint32_t P = (uint32_t) kb * BLK_IQ4NL;
                simd<float, 16> swv = 1.f;
                if (a.dbg & 128) { if (kb == 0) block_store<uint32_t, 16>(reinterpret_cast<uint32_t*>(a.Y) + (size_t) t * 16, rowoff + P); }
                else if (!(a.dbg & 16)) swv = half_of(ld16(We, rowoff + P));
                simd<uint32_t, 128> frag = 0x01010101u;
#pragma unroll
                for (int j = 0; j < ((a.dbg & 1) ? 0 : 4); ++j) {
                    simd<uint32_t, 16> qd = ld32(We, rowoff + (P + 2 + 4 * j));
                    // four byte-pair lookups (lo value in byte 0, hi value in byte 1), repacked into the two dwords
                    simd<uint32_t, 16> t0 = gather<uint32_t, 16>(tab, (qd & 0xffu) << 2);
                    simd<uint32_t, 16> t1 = gather<uint32_t, 16>(tab, ((qd >> 8) & 0xffu) << 2);
                    simd<uint32_t, 16> t2 = gather<uint32_t, 16>(tab, ((qd >> 16) & 0xffu) << 2);
                    simd<uint32_t, 16> t3 = gather<uint32_t, 16>(tab, (qd >> 24) << 2);
                    frag.template select<16, 1>(16 * j) = (t0 & 0xffu) | ((t1 & 0xffu) << 8) | ((t2 & 0xffu) << 16) | (t3 << 24);
                    frag.template select<16, 1>(16 * (j + 4)) = ((t0 >> 8) & 0xffu) | (t1 & 0xff00u) | ((t2 & 0xff00u) << 8) | ((t3 & 0xff00u) << 16);
                }
#pragma unroll
                for (int i = 0; i < MT; ++i) {
                    if (i < nvalid) {
                        simd<int, 64> av = 0x01010101;
                        simd<float, 8> dxv = 1.f;
                        if (!(a.dbg & 2)) {
                            av = block_load<int, 64>(Xi + ((size_t) (tbase + i) * a.nkb + kb) * 64);
                            dxv = block_load<float, 8>(a.dx + ((size_t) (tbase + i) * a.nkb + kb) * 8);
                        }
                        simd<float, 128> tf = dpas_s8(frag, av);
#pragma unroll
                        for (int r = 0; r < 8; ++r) {
                            const float dxr = dxv[r];
                            acc.template select<16, 1>(128 * i + 16 * r) += tf.template select<16, 1>(16 * r) * (swv * dxr);
                        }
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                if (i < nvalid) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const int rl = (tl0 + i) * 8 + r;
                        if (rl < cnt && !(a.dbg & 4))
                            block_store<float, 16>(a.Y + ((size_t) (pk.row0 + rl)) * a.ldy + 16 * nt,
                                                   acc.template select<16, 1>(128 * i + 16 * r));
                    }
                }
            }
        });
    });
}

int pick_mt(const Layout& lo) {
    int tiles = lo.tile0[lo.n];
    const double mean = lo.n ? (double) tiles / lo.n : 0;
    if (mean <= 2.0) return 2;
    if (mean <= 4.0) return 4;
    if (mean <= 8.0) return 8;
    return 12;
}
int dbg_flags() {
    static const int v = [] { const char* s = std::getenv("STRATA_I8_DBG"); return s ? std::atoi(s) : 0; }();
    return v;
}
int mt_override() {
    static const int v = [] { const char* s = std::getenv("STRATA_I8_MT"); return s ? std::atoi(s) : 0; }();
    return v;
}

template <class F>
void dispatch_mt(int mt, F&& f) {
    switch (mt) {
        case 2: f(std::integral_constant<int, 2>{}); break;
        case 4: f(std::integral_constant<int, 4>{}); break;
        case 8: f(std::integral_constant<int, 8>{}); break;
        default: f(std::integral_constant<int, 12>{}); break;
    }
}

// ------------------------------------------------------------------------------------------------ quantize / swiglu
struct QArgs {
    int n;
    int row0[kMaxGroup + 1];
    int tile0[kMaxGroup + 1];
    int nkb;
    int ld;   // row length in elements of the source
};
class QuantTag;
class SwigluTag;

}  // namespace

void make_layout(Layout& lo, const int32_t* cnt, int n) {
    lo.n = n;
    int r = 0, t = 0;
    for (int i = 0; i < n; ++i) {
        lo.cnt[i] = cnt[i];
        lo.row0[i] = r;
        lo.tile0[i] = t;
        r += cnt[i];
        t += (cnt[i] + 7) / 8;
    }
    lo.row0[n] = r;
    lo.tile0[n] = t;
}

bool available(void* stream) {
    sycl::queue& q = *strata::q_of(stream);
    static const bool ok = [&] {
        const char* v = std::getenv("STRATA_I8_FORCE");
        if (v && v[0] == '0') return false;
        sycl::device d = q.get_device();
        if (!d.has(sycl::aspect::ext_intel_esimd)) return false;
        namespace mx = sycl::ext::oneapi::experimental::matrix;
        bool s8 = false;
        for (const auto& c : d.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>())
            if (c.atype == mx::matrix_type::sint8 && c.btype == mx::matrix_type::sint8 && c.nsize == 16 && c.ksize == 32) s8 = true;
        return s8;
    }();
    return ok;
}

size_t xq_bytes(const Layout& lo, int K) { return (size_t) lo.tile0[lo.n] * 8 * K; }
size_t dx_bytes(const Layout& lo, int K) { return (size_t) lo.tile0[lo.n] * (K / 32) * 8 * sizeof(float); }

bool workspace(const Layout& lo, int K, int n_ff, Workspace& ws, void* stream) {
    sycl::queue& q = *strata::q_of(stream);
    static std::mutex mu;
    static Workspace cur;
    static size_t cap_x = 0, cap_h = 0;
    std::lock_guard<std::mutex> lk(mu);
    const size_t nx = xq_bytes(lo, K), nh = xq_bytes(lo, n_ff);
    if (nx > cap_x || nh > cap_h) {
        q.wait();
        if (cur.xq) { sycl::free(cur.xq, q); sycl::free(cur.dx, q); sycl::free(cur.hq, q); sycl::free(cur.dh, q); }
        cap_x = std::max(nx, cap_x + cap_x / 4);
        cap_h = std::max(nh, cap_h + cap_h / 4);
        cur.xq = sycl::malloc_device<int8_t>(cap_x, q);
        cur.dx = sycl::malloc_device<float>(cap_x / 32 / 8 * 8 + 64, q);   // rows * K/32 floats = bytes / 32
        cur.hq = sycl::malloc_device<int8_t>(cap_h, q);
        cur.dh = sycl::malloc_device<float>(cap_h / 32 + 64, q);
        if (!cur.xq || !cur.dx || !cur.hq || !cur.dh) { cap_x = cap_h = 0; cur = Workspace{}; ws = Workspace{}; return false; }
    }
    ws = cur;
    return true;
}

void quantize_rows(const uint16_t* X, const Layout& lo, int K, int8_t* Xq, float* dx, void* stream) {
    if (lo.tile0[lo.n] == 0) return;
    sycl::queue& q = *strata::q_of(stream);
    QArgs a{};
    a.n = lo.n;
    for (int i = 0; i <= lo.n; ++i) { a.row0[i] = lo.row0[i]; a.tile0[i] = lo.tile0[i]; }
    a.nkb = K / 32;
    a.ld = K;
    const size_t nthreads = (size_t) lo.tile0[lo.n] * a.nkb;
    const half* Xh = reinterpret_cast<const half*>(X);
    uint32_t* Xd = reinterpret_cast<uint32_t*>(Xq);
    q.submit([&](sycl::handler& h) {
        h.parallel_for<QuantTag>(sycl::range<2>((size_t) lo.tile0[lo.n], (size_t) a.nkb), [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {
            const int T = (int) id[0], kb = (int) id[1];
            int e = 0;
            while (e + 1 < a.n && T >= a.tile0[e + 1]) ++e;
            const int cnt = a.row0[e + 1] - a.row0[e];
            simd<float, 8> dxv = 0.f;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int rl = (T - a.tile0[e]) * 8 + r;
                simd<uint32_t, 8> qd = 0u;
                if (rl < cnt) {
                    simd<half, 32> xh = block_load<half, 32>(Xh + ((size_t) (a.row0[e] + rl)) * a.ld + kb * 32);
                    simd<float, 32> x = xh;
                    float amax = hmax<float>(abs(x));
                    const float dd = amax / 127.f;
                    const float id127 = amax > 0.f ? 127.f / amax : 0.f;
                    simd<float, 32> y = x * id127;
                    simd<float, 32> adj = 0.5f;
                    adj.merge(simd<float, 32>(-0.5f), y < 0.f);
                    simd<int, 32> qi = convert<int>(y + adj);
                    simd<int8_t, 32> q8 = convert<int8_t>(qi);
                    qd = q8.template bit_cast_view<uint32_t>();
                    dxv[r] = dd;
                }
                block_store<uint32_t, 8>(Xd + ((size_t) T * a.nkb + kb) * 64 + r * 8, qd);
            }
            block_store<float, 8>(dx + ((size_t) T * a.nkb + kb) * 8, dxv);
        });
    });
}

void swiglu_quantize(const float* GU, const Layout& lo, int n_ff, int8_t* Hq, float* dh, void* stream) {
    if (lo.tile0[lo.n] == 0) return;
    sycl::queue& q = *strata::q_of(stream);
    QArgs a{};
    a.n = lo.n;
    for (int i = 0; i <= lo.n; ++i) { a.row0[i] = lo.row0[i]; a.tile0[i] = lo.tile0[i]; }
    a.nkb = n_ff / 32;
    a.ld = 2 * n_ff;
    const size_t nthreads = (size_t) lo.tile0[lo.n] * a.nkb;
    uint32_t* Hd = reinterpret_cast<uint32_t*>(Hq);
    q.submit([&](sycl::handler& h) {
        h.parallel_for<SwigluTag>(sycl::range<2>((size_t) lo.tile0[lo.n], (size_t) a.nkb), [=](sycl::id<2> id) SYCL_ESIMD_KERNEL {
            const int T = (int) id[0], kb = (int) id[1];
            int e = 0;
            while (e + 1 < a.n && T >= a.tile0[e + 1]) ++e;
            const int cnt = a.row0[e + 1] - a.row0[e];
            simd<float, 8> dxv = 0.f;
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int rl = (T - a.tile0[e]) * 8 + r;
                simd<uint32_t, 8> qd = 0u;
                if (rl < cnt) {
                    simd<float, 64> v = block_load<float, 64>(GU + ((size_t) (a.row0[e] + rl)) * a.ld + kb * 64);
                    simd<float, 32> g = v.template select<32, 2>(0);
                    simd<float, 32> u = v.template select<32, 2>(1);
                    simd<float, 32> x = g * u / (1.f + exp(-g));
                    float amax = hmax<float>(abs(x));
                    const float dd = amax / 127.f;
                    const float id127 = amax > 0.f ? 127.f / amax : 0.f;
                    simd<float, 32> y = x * id127;
                    simd<float, 32> adj = 0.5f;
                    adj.merge(simd<float, 32>(-0.5f), y < 0.f);
                    simd<int, 32> qi = convert<int>(y + adj);
                    simd<int8_t, 32> q8 = convert<int8_t>(qi);
                    qd = q8.template bit_cast_view<uint32_t>();
                    dxv[r] = dd;
                }
                block_store<uint32_t, 8>(Hd + ((size_t) T * a.nkb + kb) * 64 + r * 8, qd);
            }
            block_store<float, 8>(dh + ((size_t) T * a.nkb + kb) * 8, dxv);
        });
    });
}

bool gate_up(int gu_type, const uint8_t* const* blob, size_t up_off, int n_ff, const Layout& lo, int K, const int8_t* Xq,
             const float* dx, float* GU, void* stream) {
    if (lo.n == 0 || lo.tile0[lo.n] == 0) return true;
    if (gu_type != 21 || K % 256 != 0 || (2 * n_ff) % 16 != 0 || (up_off & 3) != 0) return false;
    sycl::queue& q = *strata::q_of(stream);
    Tables& tb = tables(q);
    GArgs a{};
    for (int i = 0; i < lo.n; ++i) a.blob[i] = blob[i];
    a.sec_off = (uint32_t) up_off;
    a.Xq = Xq; a.dx = dx; a.Y = GU;
    a.ntiles = 2 * n_ff / 16;
    a.nkb = K / 32;
    a.rowb = (K / 256) * BLK_IQ3S;
    a.ldy = 2 * n_ff;
    a.dbg = dbg_flags();
    const int mt = mt_override() ? mt_override() : pick_mt(lo);
    dispatch_mt(mt, [&](auto mtc) {
        constexpr int MT = decltype(mtc)::value;
        int nthreads = 0;
        fill_args<MT>(a, lo, a.ntiles, nthreads);
        launch_gu_iq3s<MT>(q, a, nthreads, tb.grid3s);
    });
    return true;
}

bool down(int d_type, const uint8_t* const* blob, size_t down_off, int n_embd, const Layout& lo, int n_ff, const int8_t* Hq,
          const float* dh, float* Dm, void* stream) {
    if (lo.n == 0 || lo.tile0[lo.n] == 0) return true;
    if (d_type != 20 || n_ff % 32 != 0 || n_embd % 16 != 0 || (down_off & 3) != 0) return false;
    sycl::queue& q = *strata::q_of(stream);
    Tables& tb = tables(q);
    GArgs a{};
    for (int i = 0; i < lo.n; ++i) a.blob[i] = blob[i];
    a.sec_off = (uint32_t) down_off;
    a.Xq = Hq; a.dx = dh; a.Y = Dm;
    a.ntiles = n_embd / 16;
    a.nkb = n_ff / 32;
    a.rowb = (n_ff / 32) * BLK_IQ4NL;
    a.ldy = n_embd;
    a.dbg = dbg_flags();
    const int mt = mt_override() ? mt_override() : pick_mt(lo);
    dispatch_mt(mt, [&](auto mtc) {
        constexpr int MT = decltype(mtc)::value;
        int nthreads = 0;
        fill_args<MT>(a, lo, a.ntiles, nthreads);
        if (a.dbg & 64) {
            std::fprintf(stderr, "i8 down: MT %d n %d threads %d ntiles %d nkb %d rowb %d ldy %d sec_off %u blob0 %p Xq %p dx %p Y %p tab %p\n",
                         MT, a.n, nthreads, a.ntiles, a.nkb, a.rowb, a.ldy, a.sec_off, (const void*) a.blob[0], (const void*) a.Xq,
                         (const void*) a.dx, (void*) a.Y, (const void*) tb.iq4);
            for (int i = 0; i <= a.n; ++i) std::fprintf(stderr, "  e%d row0 %d tile0 %d rb0 %d blob %p\n", i, a.row0[i], a.tile0[i], a.rb0[i], i < a.n ? (const void*) a.blob[i] : nullptr);
        }
        launch_dn_iq4nl<MT>(q, a, nthreads, tb.iq4);
    });
    return true;
}

}  // namespace strata::prefill::i8
