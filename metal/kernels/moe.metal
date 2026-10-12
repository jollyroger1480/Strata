// metal/kernels/moe.metal - routed experts of one layer for a verify window of 1..8 tokens, grouped by expert:
//
//   moe_plan    one threadgroup: the window's top-k ids -> groups (distinct experts, by id) with their entries
//               (token, weight), each (token, k)'s entry index, and the indirect grid sizes of the next two kernels.
//               Deterministic (no atomics).
//   moe_gu      per group, a block of FFN rows: gate and up rows read once, applied to every entry of the group
//               (the shared input x is read from cache), h = silu(gate) * up per entry.
//   moe_down    per group, a block of output rows: ye[entry] = weight * (down . h[entry]).
//   moe_combine y[token] = sum over k of ye[entry(token, k)], in k order.
//
// The plan is one int buffer (see P_* below).  The arena is one layer's experts, blob e at e * blob_bytes:
// [gate rows | up rows | down rows] in GGUF blocks (tools/iq_pack.py's experts.bin layout).

#define P_NG 0
#define P_IGU 4
#define P_IDN 8
#define P_MAXE 128
#define P_GEXP 16
#define P_GSTART (P_GEXP + P_MAXE)
#define P_ETOK (P_GSTART + P_MAXE + 4)
#define P_EW (P_ETOK + P_MAXE)
#define P_TOKENT (P_EW + P_MAXE)
#define P_INTS (P_TOKENT + P_MAXE)

struct MoeArgs {
    ulong blob_bytes;
    ulong up_off;
    ulong down_off;
    uint gu_row;     // bytes per gate/up row
    uint d_row;      // bytes per down row
    uint n_embd;
    uint n_ff;
    uint n_tok;
    uint topk;
    uint n_expert;
    uint gx_gu;      // threadgroups along the rows for moe_gu (n_ff / rows per threadgroup)
    uint gx_dn;      // ... for moe_down (n_embd / rows per threadgroup)
    uint pad;
};

kernel void moe_plan(device const int* ids [[buffer(0)]], device const float* w [[buffer(1)]],
                     device int* plan [[buffer(2)]], constant MoeArgs& a [[buffer(3)]],
                     ushort tid [[thread_position_in_threadgroup]], ushort nth [[threads_per_threadgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup int s_grp[32], s_ent[32];
    const int ne = int(a.n_tok * a.topk);
    // one thread per expert id (nth == n_expert, at most 1024)
    const int e = tid;
    int cnt = 0;
    if (e < int(a.n_expert))
        for (int i = 0; i < ne; ++i) cnt += ids[i] == e;
    const int has = cnt > 0;
    const int pg = simd_prefix_exclusive_sum(has), pe = simd_prefix_exclusive_sum(cnt);
    if (lane == 31) { s_grp[sg] = pg + has; s_ent[sg] = pe + cnt; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        int og = 0, oe = 0;
        const int nsg = (nth + 31) / 32;
        for (int i = 0; i < nsg; ++i) {
            const int g = s_grp[i], n = s_ent[i];
            s_grp[i] = og;
            s_ent[i] = oe;
            og += g;
            oe += n;
        }
        plan[P_NG] = og;
        plan[P_IGU] = int(a.gx_gu); plan[P_IGU + 1] = og; plan[P_IGU + 2] = 1;
        plan[P_IDN] = int(a.gx_dn); plan[P_IDN + 1] = og; plan[P_IDN + 2] = 1;
        plan[P_GSTART + og] = oe;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (has) {
        const int gi = s_grp[sg] + pg, off = s_ent[sg] + pe;
        plan[P_GEXP + gi] = e;
        plan[P_GSTART + gi] = off;
        int j = 0;
        for (int i = 0; i < ne; ++i)
            if (ids[i] == e) {
                plan[P_ETOK + off + j] = i / int(a.topk);
                plan[P_EW + off + j] = as_type<int>(w[i]);
                plan[P_TOKENT + i] = off + j;
                ++j;
            }
    }
}

// ---- gate + up + SwiGLU -------------------------------------------------------------------------------------------
// One pass over a group's rows for all its entries (at most 8: an expert appears once per token).
template <int F, int LPR, int NR, int SG>
kernel void moe_gu(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]],
                   device float* h [[buffer(2)]], device const int* plan [[buffer(3)]],
                   constant MoeArgs& a [[buffer(4)]], device const float* xs16 [[buffer(5)]],
                   uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                   ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int UNIT = fmt<F>::UNIT, RG = 32 / LPR;
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, SG * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int g = int(tg.y);
    if (g >= plan[P_NG]) return;
    const int li = lane % LPR;
    const int row0 = ((int(tg.x) * SG + sg) * RG + lane / LPR) * NR;
    device const uchar* blob = W + (ulong)plan[P_GEXP + g] * a.blob_bytes;
    const int c = plan[P_GSTART + g], ne = plan[P_GSTART + g + 1] - c;
    device const float* xt[8];
    device const float* st[8];
    for (int t = 0; t < 8; ++t) {
        const int tok = plan[P_ETOK + c + min(t, ne - 1)];
        xt[t] = x + tok * int(a.n_embd);
        st[t] = xs16 + tok * int(a.n_embd / 16);
    }
    float ag[NR][8], au[NR][8];
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < 8; ++t) ag[r][t] = au[r][t] = 0.0f;
    device const uchar* g0 = blob + (ulong)row0 * a.gu_row;
    device const uchar* u0 = blob + a.up_off + (ulong)row0 * a.gu_row;
    const int U = int(a.n_embd) / UNIT;
    for (int u = li; u < U; u += LPR) {
        float qg[NR][UNIT], qu[NR][UNIT];
        float4 cg[NR], cu[NR];
        for (int r = 0; r < NR; ++r) {
            Dec<F>::run(g0 + r * a.gu_row, u, qg[r], cg[r], tab);
            Dec<F>::run(u0 + r * a.gu_row, u, qu[r], cu[r], tab);
        }
        for (int t = 0; t < 8; ++t) {
            if (t < ne) {
                float xs[UNIT], xsum[2];
                X<F>::load(xt[t], st[t], u, xs, xsum);
                for (int r = 0; r < NR; ++r) {
                    ag[r][t] += unit_combine<UNIT>(qg[r], cg[r], xs, xsum);
                    au[r][t] += unit_combine<UNIT>(qu[r], cu[r], xs, xsum);
                }
            }
        }
    }
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < 8; ++t) {
            if (t < ne) {
                const float gv = lane_sum<LPR>(ag[r][t]), uv = lane_sum<LPR>(au[r][t]);
                if (li == 0) h[(c + t) * int(a.n_ff) + row0 + r] = gv / (1.0f + exp(-gv)) * uv;
            }
        }
}

// ---- down ---------------------------------------------------------------------------------------------------------
template <int F, int LPR, int NR, int SG>
kernel void moe_down(device const uchar* W [[buffer(0)]], device const float* h [[buffer(1)]],
                     device float* ye [[buffer(2)]], device const int* plan [[buffer(3)]],
                     constant MoeArgs& a [[buffer(4)]], device const float* hs16 [[buffer(5)]],
                     uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                     ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int UNIT = fmt<F>::UNIT, RG = 32 / LPR;
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, SG * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int g = int(tg.y);
    if (g >= plan[P_NG]) return;
    const int li = lane % LPR;
    const int row0 = ((int(tg.x) * SG + sg) * RG + lane / LPR) * NR;
    device const uchar* blob = W + (ulong)plan[P_GEXP + g] * a.blob_bytes;
    const int c = plan[P_GSTART + g], ne = plan[P_GSTART + g + 1] - c;
    float ac[NR][8];
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < 8; ++t) ac[r][t] = 0.0f;
    device const uchar* d0 = blob + a.down_off + (ulong)row0 * a.d_row;
    const int U = int(a.n_ff) / UNIT;
    for (int u = li; u < U; u += LPR) {
        float q[NR][UNIT];
        float4 sc[NR];
        for (int r = 0; r < NR; ++r) Dec<F>::run(d0 + r * a.d_row, u, q[r], sc[r], tab);
        for (int t = 0; t < 8; ++t) {
            if (t < ne) {
                float xs[UNIT], xsum[2];
                X<F>::load(h + (c + t) * int(a.n_ff), hs16 + (c + t) * int(a.n_ff / 16), u, xs, xsum);
                for (int r = 0; r < NR; ++r) ac[r][t] += unit_combine<UNIT>(q[r], sc[r], xs, xsum);
            }
        }
    }
    for (int r = 0; r < NR; ++r)
        for (int t = 0; t < 8; ++t) {
            if (t < ne) {
                const float v = lane_sum<LPR>(ac[r][t]);
                if (li == 0) ye[(c + t) * int(a.n_embd) + row0 + r] = as_type<float>(plan[P_EW + c + t]) * v;
            }
        }
}

kernel void moe_combine(device const float* ye [[buffer(0)]], device float* y [[buffer(1)]],
                        device const int* plan [[buffer(2)]], constant MoeArgs& a [[buffer(3)]],
                        uint gid [[thread_position_in_grid]]) {
    const uint t = gid / a.n_embd, k = gid % a.n_embd;
    if (t >= a.n_tok) return;
    float s = 0.0f;
    for (uint j = 0; j < a.topk; ++j) s += ye[plan[P_TOKENT + t * a.topk + j] * int(a.n_embd) + k];
    y[gid] = s;
}

// ---- instantiations: moe_gu_<type>_<LPR>_<NR>_<SG>, moe_down_<type>_<LPR>_<NR>_<SG> -------------------------------
#define GU(F, L, R, S) template [[host_name("moe_gu_" #F "_" #L "_" #R "_" #S)]] kernel decltype(moe_gu<F, L, R, S>) moe_gu<F, L, R, S>;
#define DN(F, L, R, S) template [[host_name("moe_down_" #F "_" #L "_" #R "_" #S)]] kernel decltype(moe_down<F, L, R, S>) moe_down<F, L, R, S>;
#define GU_SET(F) GU(F, 32, 1, 4) GU(F, 32, 2, 4) GU(F, 32, 4, 4) GU(F, 32, 2, 2) GU(F, 32, 2, 8) GU(F, 32, 1, 8) \
                  GU(F, 16, 1, 4) GU(F, 16, 2, 4) GU(F, 8, 1, 4) GU(F, 8, 2, 4) GU(F, 8, 1, 2)
#define DN_SET(F) DN(F, 8, 1, 4) DN(F, 8, 2, 4) DN(F, 8, 4, 4) DN(F, 8, 1, 8) DN(F, 8, 2, 8) DN(F, 4, 1, 4) DN(F, 4, 2, 4) \
                  DN(F, 4, 4, 4) DN(F, 16, 1, 4) DN(F, 16, 2, 4) DN(F, 32, 1, 4) DN(F, 2, 2, 4)
GU_SET(T_Q2_0)
GU_SET(T_IQ3_XXS)
GU_SET(T_IQ2_XXS)
GU_SET(T_IQ2_XS)
GU_SET(T_IQ2_S)
GU_SET(T_IQ3_S)
DN_SET(T_Q2_0)
DN_SET(T_IQ4_NL)
