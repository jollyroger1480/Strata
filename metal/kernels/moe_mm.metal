// metal/kernels/moe_mm.metal - routed experts for a PROMPT chunk (after quant/units/mma.metal): a grouped GEMM on
// the matrix units through Metal 4's MetalPerformancePrimitives matmul2d (on the M5 its neural accelerators).
//
// Plan (deterministic, three passes):  the chunk's T x topk (token, expert) entries are sorted by expert, then by
// entry index; each expert's run is padded to a multiple of TM rows; tile i = (expert, first padded row).
//   mm_hist   per block of 1024 entries: a histogram of experts      -> bcnt[block][expert]
//   mm_scan   one threadgroup (thread = expert): block offsets, the padded expert offsets, the tile list
//   mm_place  per block: an entry's padded row = expert offset + earlier blocks' count + rank in its block;
//             srt_tok[row] = token (-1 on padding), srt_w[row] = weight, pos[entry] = row
// GEMMs: a threadgroup takes a tile (TM rows of one expert) and TN output rows; per K chunk of TK it converts the
// tile's activations to fp16 and dequantizes TN x TK weights (ref_quant's arithmetic, then rounded to fp16) into
// threadgroup memory, and matmul2d multiplies them, fp32 accumulation.
//   mm_gu     H[row][f] = silu(x . gate_f) * (x . up_f), fp32, for the n_ff rows
//   mm_down   YE[row][d] = h . down_d, fp32
//   mm_combine y[t][d] = sum_k w(t, k) * YE[pos(t, k)][d], k in order
//
// The idea of feeding matmul2d tiles of dequantized weights from threadgroup memory follows yibie/strata-mlx's
// tile_matmul (MIT); the code is our own.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

struct MmPlanArgs {
    uint n_ent;      // T * topk
    uint topk;
    uint n_expert;
    uint tm;         // the tile height (rows of one expert)
    uint n_blocks;   // ceil(n_ent / 1024)
};

// counts per block: bcnt[block * n_expert + e]
kernel void mm_hist(device const int* ids [[buffer(0)]], device uint* bcnt [[buffer(1)]],
                    constant MmPlanArgs& a [[buffer(2)]],
                    uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint h[1024];
    for (uint e = tid; e < a.n_expert; e += 1024) atomic_store_explicit(&h[e], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint i = tg.x * 1024 + tid;
    if (i < a.n_ent) atomic_fetch_add_explicit(&h[ids[i]], 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < a.n_expert; e += 1024)
        bcnt[tg.x * a.n_expert + e] = atomic_load_explicit(&h[e], memory_order_relaxed);
}

// one threadgroup of n_expert (<= 1024) threads.  boff[block][e] = expert e's rows before this block (within e);
// eoff[e] = expert e's first padded row; tiles[i] = (expert, row0); meta[0] = n_tiles, meta[1] = padded rows,
// meta[2..4] = an indirect grid (gx, n_tiles, 1) for the GEMMs, gx given in meta[2] by the host.
kernel void mm_scan(device const uint* bcnt [[buffer(0)]], device uint* boff [[buffer(1)]],
                    device uint* eoff [[buffer(2)]], device int2* tiles [[buffer(3)]], device uint* meta [[buffer(4)]],
                    constant MmPlanArgs& a [[buffer(5)]],
                    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup uint s_rows[32], s_tiles[32];
    const uint e = tid;
    uint tot = 0;
    if (e < a.n_expert)
        for (uint b = 0; b < a.n_blocks; ++b) {
            boff[b * a.n_expert + e] = tot;
            tot += bcnt[b * a.n_expert + e];
        }
    const uint nt = (tot + a.tm - 1) / a.tm, rows = nt * a.tm;
    const uint pr = simd_prefix_exclusive_sum(rows), pt = simd_prefix_exclusive_sum(nt);
    if (lane == 31) { s_rows[sg] = pr + rows; s_tiles[sg] = pt + nt; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint orow = 0, otile = 0, all_rows = 0, all_tiles = 0;
    for (int i = 0; i < 32; ++i) {
        if (i < int(sg)) { orow += s_rows[i]; otile += s_tiles[i]; }
        all_rows += s_rows[i];
        all_tiles += s_tiles[i];
    }
    if (e < a.n_expert) {
        const uint r0 = orow + pr, t0 = otile + pt;
        eoff[e] = r0;
        for (uint k = 0; k < nt; ++k) tiles[t0 + k] = int2(int(e), int(r0 + k * a.tm));
    }
    if (tid == 0) {
        meta[0] = all_tiles;
        meta[1] = all_rows;
        meta[3] = all_tiles;
        meta[4] = 1;
    }
}

// grid (n_blocks), 1024 threads.  srt_* must be cleared to the padding value (-1 / 0) beforehand by mm_clear.
kernel void mm_place(device const int* ids [[buffer(0)]], device const float* w [[buffer(1)]],
                     device const uint* boff [[buffer(2)]], device const uint* eoff [[buffer(3)]],
                     device int* srt_tok [[buffer(4)]], device float* srt_w [[buffer(5)]], device int* pos [[buffer(6)]],
                     constant MmPlanArgs& a [[buffer(7)]],
                     uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup int s_id[1024];
    const uint i = tg.x * 1024 + tid;
    const int e = i < a.n_ent ? ids[i] : -1;
    s_id[tid] = e;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i >= a.n_ent) return;
    uint rank = 0;
    for (uint j = 0; j < tid; ++j) rank += s_id[j] == e;
    const uint row = eoff[e] + boff[tg.x * a.n_expert + e] + rank;
    srt_tok[row] = int(i / a.topk);
    srt_w[row] = w[i];
    pos[i] = int(row);
}

kernel void mm_clear(device int* srt_tok [[buffer(0)]], constant uint& n [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) srt_tok[gid] = -1;
}

// ---- the GEMMs -----------------------------------------------------------------------------------------------------
// weights of one expert: gate at W + gate_off + e * gu_stride (+ up_off for up), down at W + down_off + e * d_stride
struct MmArgs {
    ulong gate_off;
    ulong up_off;
    ulong down_off;
    ulong gu_stride;
    ulong d_stride;
    uint gu_row;     // bytes per gate/up row (n_embd values)
    uint d_row;      // bytes per down row (n_ff values)
    uint n_embd;
    uint n_ff;
};

// TN x TK weights of `nrow`-strided rows into threadgroup half: thread-strided units of the format
template <int F, int TN, int TK>
static inline void stage_w(device const uchar* w0, uint row_bytes, int k0, threadgroup half* dst,
                           threadgroup const uint* tab, ushort tid, ushort nth) {
    constexpr int UNIT = fmt<F>::UNIT, UPC = TK / UNIT, H = UNIT / 2;
    static_assert(TK % UNIT == 0, "the K chunk must hold whole units of the format");
    for (int it = tid; it < TN * UPC; it += nth) {
        const int r = it / UPC, uu = it - r * UPC;
        float wv[UNIT];
        int p0, p1;
        DW<F>::run(w0 + ulong(r) * row_bytes, k0 / UNIT + uu, k0, wv, p0, p1, tab);
        for (int j = 0; j < H; ++j) {
            dst[r * TK + p0 + j] = half(wv[j]);
            dst[r * TK + p1 + j] = half(wv[H + j]);
        }
    }
}

template <int F, int TM, int TN, int TK, int SG>
kernel void mm_gu(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]],
                  device float* H [[buffer(2)]], device const int* srt_tok [[buffer(3)]],
                  device const int2* tiles [[buffer(4)]], constant MmArgs& a [[buffer(5)]],
                  uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NTH = SG * 32;
    threadgroup half xa[TM * TK];
    threadgroup half wg[TN * TK];
    threadgroup half wu[TN * TK];
    threadgroup uint tab[fmt<F>::TAB];
    threadgroup int tok[TM];
    load_tab<F>(tab, tid, NTH);
    const int2 tl = tiles[tg.y];
    const int e = tl.x, row0 = tl.y, f0 = int(tg.x) * TN;
    for (int m = tid; m < TM; m += NTH) tok[m] = srt_tok[row0 + m];
    device const uchar* g0 = W + a.gate_off + ulong(e) * a.gu_stride + ulong(f0) * a.gu_row;
    device const uchar* u0 = W + a.up_off + ulong(e) * a.gu_stride + ulong(f0) * a.gu_row;
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tA(xa, dextents<int32_t, 2>(TK, TM));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tG(wg, dextents<int32_t, 2>(TK, TN));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tU(wu, dextents<int32_t, 2>(TK, TN));
    auto accg = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tG), float>();
    auto accu = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tG), float>();
    for (uint16_t i = 0; i < accg.get_capacity(); ++i)
        if (accg.is_valid_element(i)) { accg[i] = 0.0f; accu[i] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k0 = 0; k0 < int(a.n_embd); k0 += TK) {
        for (int it = tid; it < TM * TK; it += NTH) {
            const int m = it / TK, k = it - m * TK;
            xa[it] = tok[m] >= 0 ? half(x[ulong(tok[m]) * a.n_embd + k0 + k]) : half(0.0f);
        }
        stage_w<F, TN, TK>(g0, a.gu_row, k0, wg, tab, tid, NTH);
        stage_w<F, TN, TK>(u0, a.gu_row, k0, wu, tab, tid, NTH);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto pg = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tG), float>();
        auto pu = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tG), float>();
        op.run(tA, tG, pg);
        op.run(tA, tU, pu);
        for (uint16_t i = 0; i < pg.get_capacity(); ++i)
            if (pg.is_valid_element(i)) { accg[i] += pg[i]; accu[i] += pu[i]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t i = 0; i < accg.get_capacity(); ++i)
        if (accg.is_valid_element(i)) {
            const float g = accg[i];
            accg[i] = g / (1.0f + exp(-g)) * accu[i];
        }
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tH(H, dextents<int32_t, 2>(a.n_ff, 1 << 30));
    accg.store(tH.slice(f0, row0));
}

template <int F, int TM, int TN, int TK, int SG>
kernel void mm_down(device const uchar* W [[buffer(0)]], device const float* H [[buffer(1)]],
                    device float* YE [[buffer(2)]], device const int2* tiles [[buffer(3)]],
                    constant MmArgs& a [[buffer(4)]],
                    uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NTH = SG * 32;
    threadgroup half ha[TM * TK];
    threadgroup half wd[TN * TK];
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, NTH);
    const int2 tl = tiles[tg.y];
    const int e = tl.x, row0 = tl.y, d0 = int(tg.x) * TN;
    device const uchar* w0 = W + a.down_off + ulong(e) * a.d_stride + ulong(d0) * a.d_row;
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tA(ha, dextents<int32_t, 2>(TK, TM));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tW(wd, dextents<int32_t, 2>(TK, TN));
    auto acc = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tW), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k0 = 0; k0 < int(a.n_ff); k0 += TK) {
        for (int it = tid; it < TM * TK; it += NTH) {
            const int m = it / TK, k = it - m * TK;
            ha[it] = half(H[ulong(row0 + m) * a.n_ff + k0 + k]);
        }
        stage_w<F, TN, TK>(w0, a.d_row, k0, wd, tab, tid, NTH);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto p = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tW), float>();
        op.run(tA, tW, p);
        for (uint16_t i = 0; i < p.get_capacity(); ++i)
            if (p.is_valid_element(i)) acc[i] += p[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tY(YE, dextents<int32_t, 2>(a.n_embd, 1 << 30));
    acc.store(tY.slice(d0, row0));
}

// grid (T * n_embd / 256), 256 threads
kernel void mm_combine(device const float* YE [[buffer(0)]], device const int* pos [[buffer(1)]],
                       device const float* srt_w [[buffer(2)]], device float* y [[buffer(3)]],
                       constant uint4& a [[buffer(4)]],   // (T, n_embd, topk, -)
                       uint gid [[thread_position_in_grid]]) {
    const uint t = gid / a.y, d = gid % a.y;
    if (t >= a.x) return;
    float s = 0.0f;
    for (uint k = 0; k < a.z; ++k) {
        const int r = pos[t * a.z + k];
        s = fma(srt_w[r], YE[ulong(r) * a.y + d], s);
    }
    y[gid] = s;
}

#define MMGU_(F, M, N, K, S) template [[host_name("mm_gu_" #F "_" #M "_" #N "_" #K "_" #S)]] kernel decltype(mm_gu<F, M, N, K, S>) mm_gu<F, M, N, K, S>;
#define MMDN_(F, M, N, K, S) template [[host_name("mm_down_" #F "_" #M "_" #N "_" #K "_" #S)]] kernel decltype(mm_down<F, M, N, K, S>) mm_down<F, M, N, K, S>;
#define MMGU(F, M, N, K, S) MMGU_(F, M, N, K, S)
#define MMDN(F, M, N, K, S) MMDN_(F, M, N, K, S)
#define MM_GU_SET(F) MMGU(F, 32, 64, 64, 4) MMGU(F, 64, 64, 64, 4) MMGU(F, 16, 64, 64, 2) MMGU(F, 32, 32, 64, 2)
#define MM_DN_SET(F) MMDN(F, 32, 64, 64, 4) MMDN(F, 64, 64, 64, 4) MMDN(F, 16, 64, 64, 2) MMDN(F, 32, 32, 64, 2)
MM_GU_SET(T_Q2_0) MM_GU_SET(T_IQ3_XXS) MM_GU_SET(T_IQ2_XXS) MM_GU_SET(T_IQ2_XS) MM_GU_SET(T_IQ2_S) MM_GU_SET(T_IQ3_S)
MM_DN_SET(T_Q2_0) MM_DN_SET(T_IQ4_NL)

// ---- v2: activations gathered once per layer as fp16 rows, read by matmul2d from device memory ---------------------
// mm_gather: XS[row][k] = half(x[srt_tok[row]][k]) (0 on padding rows).  grid (rows), 256 threads.
kernel void mm_gather(device const float* x [[buffer(0)]], device const int* srt_tok [[buffer(1)]],
                      device half* XS [[buffer(2)]], constant uint& n [[buffer(3)]],
                      uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const int t = srt_tok[tg.x];
    device half* o = XS + ulong(tg.x) * n;
    for (uint k = tid; k < n; k += 256) o[k] = t >= 0 ? half(x[ulong(t) * n + k]) : half(0.0f);
}

template <int F, int TM, int TN, int TK, int SG>
kernel void mm_gu2(device const uchar* W [[buffer(0)]], device const half* XS [[buffer(1)]],
                   device half* HS [[buffer(2)]], device const int2* tiles [[buffer(3)]], constant MmArgs& a [[buffer(4)]],
                   uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NTH = SG * 32;
    threadgroup half wg[TN * TK];
    threadgroup half wu[TN * TK];
    threadgroup float ot[TM * TN];
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, NTH);
    const int2 tl = tiles[tg.y];
    const int e = tl.x, row0 = tl.y, f0 = int(tg.x) * TN;
    device const uchar* g0 = W + a.gate_off + ulong(e) * a.gu_stride + ulong(f0) * a.gu_row;
    device const uchar* u0 = W + a.up_off + ulong(e) * a.gu_stride + ulong(f0) * a.gu_row;
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tX((device half*)XS, dextents<int32_t, 2>(a.n_embd, 1 << 30));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tG(wg, dextents<int32_t, 2>(TK, TN));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tU(wu, dextents<int32_t, 2>(TK, TN));
    auto tA = tX.template slice<TK, TM>(0, row0);
    auto accg = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tG), float>();
    auto accu = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tG), float>();
    for (uint16_t i = 0; i < accg.get_capacity(); ++i)
        if (accg.is_valid_element(i)) { accg[i] = 0.0f; accu[i] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k0 = 0; k0 < int(a.n_embd); k0 += TK) {
        stage_w<F, TN, TK>(g0, a.gu_row, k0, wg, tab, tid, NTH);
        stage_w<F, TN, TK>(u0, a.gu_row, k0, wu, tab, tid, NTH);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tA = tX.template slice<TK, TM>(k0, row0);
        op.run(tA, tG, accg);
        op.run(tA, tU, accu);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t i = 0; i < accg.get_capacity(); ++i)
        if (accg.is_valid_element(i)) {
            const float g = accg[i];
            accg[i] = g / (1.0f + exp(-g)) * accu[i];
        }
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tO(ot, dextents<int32_t, 2>(TN, TM));
    accg.store(tO);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int it = tid; it < TM * TN; it += NTH) {
        const int m = it / TN, f = it - m * TN;
        HS[ulong(row0 + m) * a.n_ff + f0 + f] = half(ot[it]);
    }
}

template <int F, int TM, int TN, int TK, int SG>
kernel void mm_down2(device const uchar* W [[buffer(0)]], device const half* HS [[buffer(1)]],
                     device float* YE [[buffer(2)]], device const int2* tiles [[buffer(3)]],
                     constant MmArgs& a [[buffer(4)]],
                     uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NTH = SG * 32;
    threadgroup half wd[TN * TK];
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, NTH);
    const int2 tl = tiles[tg.y];
    const int e = tl.x, row0 = tl.y, d0 = int(tg.x) * TN;
    device const uchar* w0 = W + a.down_off + ulong(e) * a.d_stride + ulong(d0) * a.d_row;
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tH((device half*)HS, dextents<int32_t, 2>(a.n_ff, 1 << 30));
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tW(wd, dextents<int32_t, 2>(TK, TN));
    auto tA = tH.template slice<TK, TM>(0, row0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(tA), decltype(tW), float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k0 = 0; k0 < int(a.n_ff); k0 += TK) {
        stage_w<F, TN, TK>(w0, a.d_row, k0, wd, tab, tid, NTH);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tA = tH.template slice<TK, TM>(k0, row0);
        op.run(tA, tW, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tY(YE, dextents<int32_t, 2>(a.n_embd, 1 << 30));
    acc.store(tY.slice(d0, row0));
}

#define MMGU2_(F, M, N, K, S) template [[host_name("mm_gu2_" #F "_" #M "_" #N "_" #K "_" #S)]] kernel decltype(mm_gu2<F, M, N, K, S>) mm_gu2<F, M, N, K, S>;
#define MMDN2_(F, M, N, K, S) template [[host_name("mm_down2_" #F "_" #M "_" #N "_" #K "_" #S)]] kernel decltype(mm_down2<F, M, N, K, S>) mm_down2<F, M, N, K, S>;
#define MMGU2(F, M, N, K, S) MMGU2_(F, M, N, K, S)
#define MMDN2(F, M, N, K, S) MMDN2_(F, M, N, K, S)
#define MM_GU2_SET(F) MMGU2(F, 32, 64, 32, 2) MMGU2(F, 32, 32, 32, 2) MMGU2(F, 32, 64, 16, 2) MMGU2(F, 64, 32, 16, 2) MMGU2(F, 16, 32, 32, 2) MMGU2(F, 16, 64, 16, 2) MMGU2(F, 16, 128, 32, 4) MMGU2(F, 32, 64, 16, 4) MMGU2(F, 16, 32, 32, 1) MMGU2(F, 16, 64, 32, 4) MMGU2(F, 16, 64, 64, 2) MMGU2(F, 32, 64, 64, 4) MMGU2(F, 64, 64, 64, 4) MMGU2(F, 16, 32, 64, 2) MMGU2(F, 32, 64, 32, 4) MMGU2(F, 64, 64, 32, 4) MMGU2(F, 16, 64, 32, 2)
#define MM_DN2_SET(F) MMDN2(F, 32, 64, 32, 2) MMDN2(F, 32, 32, 32, 2) MMDN2(F, 32, 64, 16, 2) MMDN2(F, 64, 32, 16, 2) MMDN2(F, 16, 32, 32, 2) MMDN2(F, 16, 64, 16, 2) MMDN2(F, 16, 128, 32, 4) MMDN2(F, 32, 64, 16, 4) MMDN2(F, 16, 32, 32, 1) MMDN2(F, 16, 64, 32, 4) MMDN2(F, 16, 64, 64, 2) MMDN2(F, 32, 64, 64, 4) MMDN2(F, 64, 64, 64, 4) MMDN2(F, 16, 32, 64, 2) MMDN2(F, 32, 64, 32, 4) MMDN2(F, 64, 64, 32, 4) MMDN2(F, 16, 64, 32, 2)
MM_GU2_SET(T_Q2_0) MM_GU2_SET(T_IQ3_XXS) MM_GU2_SET(T_IQ2_XXS) MM_GU2_SET(T_IQ2_XS) MM_GU2_SET(T_IQ2_S) MM_GU2_SET(T_IQ3_S)
MM_DN2_SET(T_Q2_0)
MMDN2(T_IQ4_NL, 16, 64, 32, 2) MMDN2(T_IQ4_NL, 16, 64, 64, 2) MMDN2(T_IQ4_NL, 16, 32, 32, 2) MMDN2(T_IQ4_NL, 32, 32, 32, 2) MMDN2(T_IQ4_NL, 16, 128, 32, 4)
