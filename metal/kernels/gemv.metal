// metal/kernels/gemv.metal - y[t][r] = W[r] . x[t] for a window of 1..8 tokens (dense projections, the shared
// expert, the LM head), any format of quant.metal/units.metal.
//
// Shape of a threadgroup: SG simdgroups; KS of them split the row's units (KS = 1: no split), the other SG/KS cover
// different rows.  In a simdgroup, LPR lanes share a row and 32/LPR rows run side by side; each lane holds NR rows.
// The weights of a unit are decoded once and applied to every token.  Partial sums of a K split are added in a
// fixed order (deterministic).

struct GemvArgs {
    uint n_in;
    uint n_out;
    uint n_tok;
    uint row_bytes;
    uint x_stride;   // floats between tokens' x rows
    uint y_stride;   // floats between tokens' y rows
};

constant int NTOK [[function_constant(0)]];   // tokens of the window (1..8): the token loops are specialised

template <int F, int LPR, int NR, int SG, int KS>
kernel void gemv(device const uchar* W [[buffer(0)]], device const float* xp [[buffer(1)]],
                 device const float* s16 [[buffer(2)]], device float* y [[buffer(3)]], constant GemvArgs& a [[buffer(4)]],
                 uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                 ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int UNIT = fmt<F>::UNIT, RS = SG / KS, RG = 32 / LPR;
    threadgroup uint tab[fmt<F>::TAB];
    threadgroup float red[KS > 1 ? KS * RS * RG * NR * 8 : 1];
    load_tab<F>(tab, tid, SG * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int ks = sg / RS, rs = sg % RS;
    const int li = lane % LPR;
    const int lrow0 = (rs * RG + lane / LPR) * NR;
    const int row0 = int(tg.x) * RS * RG * NR + lrow0;
    const int U = int(a.n_in) / UNIT;
    const int u0 = ks * U / KS, u1 = (ks + 1) * U / KS;
    float acc[NR][8];
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < NTOK; ++t) acc[r][t] = 0.0f;
    device const uchar* rp[NR];
    for (int r = 0; r < NR; ++r) rp[r] = W + (ulong)min(row0 + r, int(a.n_out) - 1) * a.row_bytes;
    const int s16n = int(a.n_in) / 16;
    for (int u = u0 + li; u < u1; u += LPR) {
        float q[NR][UNIT];
        float4 sc[NR];
        for (int r = 0; r < NR; ++r) Dec<F>::run(rp[r], u, q[r], sc[r], tab);
        for (int t = 0; t < NTOK; ++t) {
            float xs[UNIT], xsum[2];
            X<F>::load(xp + t * int(a.x_stride), s16 + t * s16n, u, xs, xsum);
            for (int r = 0; r < NR; ++r) acc[r][t] += unit_combine<UNIT>(q[r], sc[r], xs, xsum);
        }
    }
    constexpr int RPT = RS * RG * NR;
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < NTOK; ++t) {
            const float v = lane_sum<LPR>(acc[r][t]);
            if (KS == 1) {
                if (li == 0 && row0 + r < int(a.n_out)) y[t * int(a.y_stride) + row0 + r] = v;
            } else if (li == 0) {
                red[(ks * RPT + lrow0 + r) * 8 + t] = v;
            }
        }
    if (KS > 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ks == 0 && li == 0)
            for (int r = 0; r < NR; ++r)
                for (int t = 0; t < NTOK; ++t) {
                    float v = 0.0f;
                    for (int k = 0; k < KS; ++k) v += red[(k * RPT + lrow0 + r) * 8 + t];
                    if (row0 + r < int(a.n_out)) y[t * int(a.y_stride) + row0 + r] = v;
                }
    }
}

#define GV(F, L, R, S, K) template [[host_name("gemv_" #F "_" #L "_" #R "_" #S "_" #K)]] kernel decltype(gemv<F, L, R, S, K>) gemv<F, L, R, S, K>;
#define GV_SET(F) GV(F, 32, 1, 4, 1) GV(F, 32, 2, 4, 1) GV(F, 16, 1, 4, 1) GV(F, 16, 2, 4, 1) GV(F, 8, 1, 4, 1) \
                  GV(F, 8, 2, 4, 1) GV(F, 4, 2, 4, 1) GV(F, 4, 1, 4, 1) GV(F, 32, 1, 4, 4) GV(F, 32, 2, 4, 2) \
                  GV(F, 32, 1, 8, 8) GV(F, 16, 1, 8, 4) GV(F, 32, 1, 8, 2) GV(F, 8, 1, 8, 1)                   GV(F, 1, 1, 4, 1) GV(F, 1, 2, 4, 1) GV(F, 2, 1, 4, 1) GV(F, 2, 2, 4, 1) GV(F, 1, 1, 2, 1) GV(F, 4, 4, 4, 1)
#ifndef GEMV_TYPES
#define GEMV_TYPES GV_SET(T_BF16) GV_SET(T_Q8_0) GV_SET(T_Q4_0) GV_SET(T_Q5_0) GV_SET(T_IQ4_NL) GV_SET(T_IQ4_XS) \
                   GV_SET(T_Q2_0) GV_SET(T_IQ3_S) GV_SET(T_Q3_K) GV_SET(T_Q4_K) GV_SET(T_Q5_K) GV_SET(T_Q6_K)
#endif
GEMV_TYPES
