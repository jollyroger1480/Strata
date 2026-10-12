// metal/kernels/quant.metal - the weight formats as "units": a unit is UNIT consecutive weights of one row that one
// SIMD lane decodes at a time into raw values q[0..UNIT) and one scale s, so that the unit's dot with x is
// s * sum(q[j] * x[j]).  Every GEMV kernel (moe.metal, gemv.metal) is written once over these.
//
// Decode rules transcribed from ggml-quants.c dequantize_row_* (3cf03257) and Strata's Q2_0 contract
// (docs/q2_0-contract.md); the grids and sign tables are ggml-common.h's (prepended by the runtime).  The tables a
// format needs are copied to threadgroup memory once per threadgroup (constant-memory lookups with divergent indices
// serialise on Apple GPUs).
//
// Blocks start at 2-byte aligned addresses (18, 66, 74, 82, 98, 110 bytes), so every multi-byte field is read as
// ushorts.

#define T_Q4_0 2
#define T_Q5_0 6
#define T_Q8_0 8
#define T_Q4_K 12
#define T_Q5_K 13
#define T_Q6_K 14
#define T_IQ2_XXS 16
#define T_IQ2_XS 17
#define T_IQ3_XXS 18
#define T_IQ4_NL 20
#define T_IQ3_S 21
#define T_IQ2_S 22
#define T_IQ4_XS 23
#define T_Q2_0 42

template <int F> struct fmt;
// QK block elements, BB block bytes, UNIT weights per unit, TAB words of threadgroup table
template <> struct fmt<T_Q2_0>    { enum { QK = 64,  BB = 18,  UNIT = 16, TAB = 1 }; };
template <> struct fmt<T_IQ3_XXS> { enum { QK = 256, BB = 98,  UNIT = 16, TAB = 256 + 32 }; };
template <> struct fmt<T_IQ2_XXS> { enum { QK = 256, BB = 66,  UNIT = 16, TAB = 512 + 32 }; };
template <> struct fmt<T_IQ2_XS>  { enum { QK = 256, BB = 74,  UNIT = 16, TAB = 1024 + 32 }; };
template <> struct fmt<T_IQ2_S>   { enum { QK = 256, BB = 82,  UNIT = 16, TAB = 2048 + 32 }; };
template <> struct fmt<T_IQ3_S>   { enum { QK = 256, BB = 110, UNIT = 16, TAB = 512 + 32 }; };
template <> struct fmt<T_IQ4_NL>  { enum { QK = 32,  BB = 18,  UNIT = 32, TAB = 16 }; };
template <> struct fmt<T_IQ4_XS>  { enum { QK = 256, BB = 136, UNIT = 32, TAB = 16 }; };
template <> struct fmt<T_Q8_0>    { enum { QK = 32,  BB = 34,  UNIT = 32, TAB = 1 }; };
template <> struct fmt<T_Q4_0>    { enum { QK = 32,  BB = 18,  UNIT = 32, TAB = 1 }; };
template <> struct fmt<T_Q5_0>    { enum { QK = 32,  BB = 22,  UNIT = 32, TAB = 1 }; };

static inline uint rd32(device const uchar* p) {
    device const ushort* s = (device const ushort*)p;
    return uint(s[0]) | (uint(s[1]) << 16);
}
static inline float rdh(device const uchar* p) { return float(*(device const half*)p); }

// ---- threadgroup tables -------------------------------------------------------------------------------------------
// layout in `tab` (uint words): [0, G) the grid, [G, G + 32) ksigns_iq2xs as bytes; IQ4_NL/XS: 16 floats.
template <int F> static inline void load_tab(threadgroup uint* tab, ushort tid, ushort nth);

template <> inline void load_tab<T_Q2_0>(threadgroup uint*, ushort, ushort) {}
template <> inline void load_tab<T_Q8_0>(threadgroup uint*, ushort, ushort) {}
template <> inline void load_tab<T_Q4_0>(threadgroup uint*, ushort, ushort) {}
template <> inline void load_tab<T_Q5_0>(threadgroup uint*, ushort, ushort) {}

static inline void load_signs(threadgroup uint* tab, int at, ushort tid, ushort nth) {
    threadgroup uchar* s = (threadgroup uchar*)(tab + at);
    for (int i = tid; i < 128; i += nth) s[i] = ksigns_iq2xs[i];
}
template <> inline void load_tab<T_IQ3_XXS>(threadgroup uint* tab, ushort tid, ushort nth) {
    for (int i = tid; i < 256; i += nth) tab[i] = iq3xxs_grid[i];
    load_signs(tab, 256, tid, nth);
}
template <> inline void load_tab<T_IQ3_S>(threadgroup uint* tab, ushort tid, ushort nth) {
    for (int i = tid; i < 512; i += nth) tab[i] = iq3s_grid[i];
    load_signs(tab, 512, tid, nth);
}
template <> inline void load_tab<T_IQ2_XXS>(threadgroup uint* tab, ushort tid, ushort nth) {
    for (int i = tid; i < 256; i += nth) { tab[2 * i] = uint(iq2xxs_grid[i]); tab[2 * i + 1] = uint(iq2xxs_grid[i] >> 32); }
    load_signs(tab, 512, tid, nth);
}
template <> inline void load_tab<T_IQ2_XS>(threadgroup uint* tab, ushort tid, ushort nth) {
    for (int i = tid; i < 512; i += nth) { tab[2 * i] = uint(iq2xs_grid[i]); tab[2 * i + 1] = uint(iq2xs_grid[i] >> 32); }
    load_signs(tab, 1024, tid, nth);
}
template <> inline void load_tab<T_IQ2_S>(threadgroup uint* tab, ushort tid, ushort nth) {
    for (int i = tid; i < 1024; i += nth) { tab[2 * i] = uint(iq2s_grid[i]); tab[2 * i + 1] = uint(iq2s_grid[i] >> 32); }
    load_signs(tab, 2048, tid, nth);
}
template <> inline void load_tab<T_IQ4_NL>(threadgroup uint* tab, ushort tid, ushort nth) {
    threadgroup float* kv = (threadgroup float*)tab;
    for (int i = tid; i < 16; i += nth) kv[i] = float(kvalues_iq4nl[i]);
}
template <> inline void load_tab<T_IQ4_XS>(threadgroup uint* tab, ushort tid, ushort nth) {
    load_tab<T_IQ4_NL>(tab, tid, nth);
}

// ---- unit decoders: return s, fill q[UNIT] ------------------------------------------------------------------------
// ui = unit index inside the block (0 .. QK/UNIT - 1).
template <int F> static inline float deq(device const uchar* b, int ui, thread float* q, threadgroup const uint* tab);

// Q2_0: 64 = 4 units of 16; code c (2 bits, LSB first) -> c - 1; scale d.
template <> inline float deq<T_Q2_0>(device const uchar* b, int ui, thread float* q, threadgroup const uint*) {
    device const ushort* p = (device const ushort*)(b + 2 + 4 * ui);
    const ushort a = p[0], c = p[1];
    for (int j = 0; j < 8; ++j) {
        q[j] = float((a >> (2 * j)) & 3) - 1.0f;
        q[8 + j] = float((c >> (2 * j)) & 3) - 1.0f;
    }
    return rdh(b);
}

// 8 signed grid bytes from a 4-byte grid word pair helper
static inline void put_signed4(thread float* q, uint g, uint s) {
    const uchar4 v = as_type<uchar4>(g);
    q[0] = (s & 1) ? -float(v[0]) : float(v[0]);
    q[1] = (s & 2) ? -float(v[1]) : float(v[1]);
    q[2] = (s & 4) ? -float(v[2]) : float(v[2]);
    q[3] = (s & 8) ? -float(v[3]) : float(v[3]);
}

// IQ3_XXS: 256 = 8 x 32; unit = half of a 32 (two grid pairs).
template <> inline float deq<T_IQ3_XXS>(device const uchar* b, int ui, thread float* q, threadgroup const uint* tab) {
    const int ib = ui >> 1, h = ui & 1;
    const uint aux = rd32(b + 2 + 64 + 4 * ib);
    threadgroup const uchar* ks = (threadgroup const uchar*)(tab + 256);
    device const uchar* qs = b + 2 + 8 * ib + 4 * h;
    for (int l2 = 0; l2 < 2; ++l2) {
        const int l = 2 * h + l2;
        const uint s = ks[(aux >> (7 * l)) & 127];
        put_signed4(q + 8 * l2, tab[qs[2 * l2]], s);
        put_signed4(q + 8 * l2 + 4, tab[qs[2 * l2 + 1]], s >> 4);
    }
    return rdh(b) * (0.5f + float(aux >> 28)) * 0.5f;
}

// IQ2_XXS: per 32: two u32 (4 grid indices, then 4 x 7 sign bits + 4-bit scale).
template <> inline float deq<T_IQ2_XXS>(device const uchar* b, int ui, thread float* q, threadgroup const uint* tab) {
    const int ib = ui >> 1, h = ui & 1;
    const uint a0 = rd32(b + 2 + 8 * ib), a1 = rd32(b + 2 + 8 * ib + 4);
    threadgroup const uchar* ks = (threadgroup const uchar*)(tab + 512);
    for (int l2 = 0; l2 < 2; ++l2) {
        const int l = 2 * h + l2;
        const uint idx = (a0 >> (8 * l)) & 255;
        const uint s = ks[(a1 >> (7 * l)) & 127];
        put_signed4(q + 8 * l2, tab[2 * idx], s);
        put_signed4(q + 8 * l2 + 4, tab[2 * idx + 1], s >> 4);
    }
    return rdh(b) * (0.5f + float(a1 >> 28)) * 0.25f;
}

// IQ2_XS: uint16 qs[32] (9-bit grid index + 7-bit sign index), scales[8] nibbles.
template <> inline float deq<T_IQ2_XS>(device const uchar* b, int ui, thread float* q, threadgroup const uint* tab) {
    const int ib = ui >> 1, h = ui & 1;
    device const ushort* qs = (device const ushort*)(b + 2);
    threadgroup const uchar* ks = (threadgroup const uchar*)(tab + 1024);
    for (int l2 = 0; l2 < 2; ++l2) {
        const uint v = qs[4 * ib + 2 * h + l2];
        const uint idx = v & 511, s = ks[v >> 9];
        put_signed4(q + 8 * l2, tab[2 * idx], s);
        put_signed4(q + 8 * l2 + 4, tab[2 * idx + 1], s >> 4);
    }
    const uint sc = b[66 + ib];
    return rdh(b) * (0.5f + float((sc >> (4 * h)) & 15)) * 0.25f;
}

// IQ2_S: qs[32] low 8 bits of the 10-bit index, signs[32], qh[8] (2 high bits per group of 8), scales[8].
template <> inline float deq<T_IQ2_S>(device const uchar* b, int ui, thread float* q, threadgroup const uint* tab) {
    const int ib = ui >> 1, h = ui & 1;
    const uint qh = b[66 + ib];
    for (int l2 = 0; l2 < 2; ++l2) {
        const int l = 2 * h + l2;
        const uint idx = uint(b[2 + 4 * ib + l]) | ((qh << (8 - 2 * l)) & 0x300);
        const uint s = b[2 + 32 + 4 * ib + l];
        put_signed4(q + 8 * l2, tab[2 * idx], s);
        put_signed4(q + 8 * l2 + 4, tab[2 * idx + 1], s >> 4);
    }
    const uint sc = b[74 + ib];
    return rdh(b) * (0.5f + float((sc >> (4 * h)) & 15)) * 0.25f;
}

// IQ3_S: qs[64] low 8 bits of 9-bit indices, qh[8], signs[32], scales[4] (nibble per 32).
template <> inline float deq<T_IQ3_S>(device const uchar* b, int ui, thread float* q, threadgroup const uint* tab) {
    const int ib = ui >> 1, h = ui & 1;
    const uint qh = b[66 + ib];
    for (int l2 = 0; l2 < 2; ++l2) {
        const int l = 2 * h + l2;
        const uint i1 = uint(b[2 + 8 * ib + 2 * l]) | ((qh << (8 - 2 * l)) & 256);
        const uint i2 = uint(b[2 + 8 * ib + 2 * l + 1]) | ((qh << (7 - 2 * l)) & 256);
        const uint s = b[74 + 4 * ib + l];
        put_signed4(q + 8 * l2, tab[i1], s);
        put_signed4(q + 8 * l2 + 4, tab[i2], s >> 4);
    }
    const uint sc = b[106 + (ib >> 1)];
    return rdh(b) * float(1 + 2 * ((sc >> (4 * (ib & 1))) & 15));
}

// IQ4_NL: 32 = 1 unit; low nibbles are elements 0..15, high 16..31.
template <> inline float deq<T_IQ4_NL>(device const uchar* b, int, thread float* q, threadgroup const uint* tab) {
    threadgroup const float* kv = (threadgroup const float*)tab;
    device const ushort* p = (device const ushort*)(b + 2);
    for (int j = 0; j < 8; ++j) {
        const ushort v = p[j];
        q[2 * j] = kv[v & 15];
        q[2 * j + 16] = kv[(v >> 4) & 15];
        q[2 * j + 1] = kv[(v >> 8) & 15];
        q[2 * j + 17] = kv[v >> 12];
    }
    return rdh(b);
}

// IQ4_XS: 256 = 8 units of 32; 6-bit scale from scales_l nibble + 2 bits of scales_h, minus 32.
template <> inline float deq<T_IQ4_XS>(device const uchar* b, int ui, thread float* q, threadgroup const uint* tab) {
    threadgroup const float* kv = (threadgroup const float*)tab;
    device const ushort* p = (device const ushort*)(b + 8 + 16 * ui);
    for (int j = 0; j < 8; ++j) {
        const ushort v = p[j];
        q[2 * j] = kv[v & 15];
        q[2 * j + 16] = kv[(v >> 4) & 15];
        q[2 * j + 1] = kv[(v >> 8) & 15];
        q[2 * j + 17] = kv[v >> 12];
    }
    const uint sh = *(device const ushort*)(b + 2);
    const int ls = int((b[4 + (ui >> 1)] >> (4 * (ui & 1))) & 15) | int(((sh >> (2 * ui)) & 3) << 4);
    return rdh(b) * float(ls - 32);
}

template <> inline float deq<T_Q8_0>(device const uchar* b, int, thread float* q, threadgroup const uint*) {
    device const ushort* p = (device const ushort*)(b + 2);
    for (int j = 0; j < 16; ++j) {
        const ushort v = p[j];
        q[2 * j] = float(as_type<char>(uchar(v & 255)));
        q[2 * j + 1] = float(as_type<char>(uchar(v >> 8)));
    }
    return rdh(b);
}

template <> inline float deq<T_Q4_0>(device const uchar* b, int, thread float* q, threadgroup const uint*) {
    device const ushort* p = (device const ushort*)(b + 2);
    for (int j = 0; j < 8; ++j) {
        const ushort v = p[j];
        q[2 * j] = float(v & 15) - 8.0f;
        q[2 * j + 16] = float((v >> 4) & 15) - 8.0f;
        q[2 * j + 1] = float((v >> 8) & 15) - 8.0f;
        q[2 * j + 17] = float(v >> 12) - 8.0f;
    }
    return rdh(b);
}

template <> inline float deq<T_Q5_0>(device const uchar* b, int, thread float* q, threadgroup const uint*) {
    const uint hb = rd32(b + 2);
    device const ushort* p = (device const ushort*)(b + 6);
    for (int j = 0; j < 8; ++j) {
        const ushort v = p[j];
        const int j0 = 2 * j, j1 = 2 * j + 1;
        q[j0] = float(int(v & 15) | int(((hb >> j0) & 1) << 4)) - 16.0f;
        q[j0 + 16] = float(int((v >> 4) & 15) | int(((hb >> (j0 + 16)) & 1) << 4)) - 16.0f;
        q[j1] = float(int((v >> 8) & 15) | int(((hb >> j1) & 1) << 4)) - 16.0f;
        q[j1 + 16] = float(int(v >> 12) | int(((hb >> (j1 + 16)) & 1) << 4)) - 16.0f;
    }
    return rdh(b);
}

// ---- the dot of one unit with NT token slices ----------------------------------------------------------------------
// The weights of unit `u` of `row` (u counts units along the row) dotted with x[t][u*UNIT ..], accumulated into acc[t].
template <int F, int NT>
static inline void unit_dot(device const uchar* row, int u, thread const float (*xs)[fmt<F>::UNIT], thread float* acc,
                            threadgroup const uint* tab) {
    constexpr int UNIT = fmt<F>::UNIT, UPB = fmt<F>::QK / UNIT;
    float q[UNIT];
    const float s = deq<F>(row + (u / UPB) * fmt<F>::BB, u % UPB, q, tab);
    for (int t = 0; t < NT; ++t) {
        float d = 0.0f;
        for (int j = 0; j < UNIT; ++j) d = fma(q[j], xs[t][j], d);
        acc[t] = fma(s, d, acc[t]);
    }
}

template <int UNIT>
static inline void load_x(device const float* x, thread float* xs) {
    device const float4* p = (device const float4*)x;
    for (int j = 0; j < UNIT / 4; ++j) {
        const float4 v = p[j];
        xs[4 * j] = v.x;
        xs[4 * j + 1] = v.y;
        xs[4 * j + 2] = v.z;
        xs[4 * j + 3] = v.w;
    }
}

// sum over groups of LPR consecutive lanes (LPR = 1, 2, 4, 8, 16 or 32); the result is valid in every lane
template <int LPR>
static inline float lane_sum(float v) {
    if (LPR == 32) return simd_sum(v);
    for (ushort o = LPR / 2; o > 0; o >>= 1) v += simd_shuffle_xor(v, o);
    return v;
}
