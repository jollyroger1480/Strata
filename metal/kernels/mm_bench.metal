// metal/kernels/mm_bench.metal - matrix throughput probes (mm_bench.cpp): C[M][N] = A[M][K] . B[N][K]^T, fp32 out.
//   mpp_mm<T, TM, TN, SG>   Metal 4 MetalPerformancePrimitives matmul2d (the M5's neural accelerators), a TM x TN
//                           tile of C per threadgroup, K dynamic, A and B straight from device memory
//   sg_mm<T>                simdgroup_matrix 8x8, a 32 x 32 tile per simdgroup (4 x 4 accumulators), 4 simdgroups
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

struct MmArgs { int M, N, K; };

template <typename T, int TM, int TN, int SG>
kernel void mpp_mm(device T* A [[buffer(0)]], device T* B [[buffer(1)]], device float* C [[buffer(2)]],
                   constant MmArgs& a [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]]) {
    // extents are (inner, outer): A is M rows of K, B is N rows of K, C is M rows of N
    tensor<device T, dextents<int32_t, 2>, tensor_inline> tA(A, dextents<int32_t, 2>(a.K, a.M));
    tensor<device T, dextents<int32_t, 2>, tensor_inline> tB(B, dextents<int32_t, 2>(a.K, a.N));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tC(C, dextents<int32_t, 2>(a.N, a.M));
    constexpr auto desc = matmul2d_descriptor(TM, TN, dynamic_length_v<int>, false, true, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto mA = tA.slice(0, int(tg.y) * TM);
    auto mB = tB.slice(0, int(tg.x) * TN);
    auto mC = tC.slice(int(tg.x) * TN, int(tg.y) * TM);
    op.run(mA, mB, mC);
}

#define MPP(T, M, N, S) template [[host_name("mpp_mm_" #T "_" #M "_" #N "_" #S)]] kernel decltype(mpp_mm<T, M, N, S>) mpp_mm<T, M, N, S>;
MPP(half, 64, 32, 4) MPP(half, 64, 64, 4) MPP(half, 128, 64, 4) MPP(half, 32, 64, 4) MPP(half, 64, 128, 4)
MPP(half, 16, 64, 2) MPP(half, 128, 128, 4)
MPP(float, 64, 32, 4) MPP(float, 64, 64, 4) MPP(float, 128, 64, 4)
MPP(bfloat, 64, 64, 4) MPP(bfloat, 128, 64, 4)

// simdgroup_matrix: threadgroup tile 64 x 64 (4 simdgroups x 32 x 32), K in steps of 8, operands from device memory
template <typename T>
kernel void sg_mm(device const T* A [[buffer(0)]], device const T* B [[buffer(1)]], device float* C [[buffer(2)]],
                  constant MmArgs& a [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]],
                  ushort sg [[simdgroup_index_in_threadgroup]]) {
    const int m0 = int(tg.y) * 64 + (sg >> 1) * 32, n0 = int(tg.x) * 64 + (sg & 1) * 32;
    simdgroup_matrix<float, 8, 8> acc[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k = 0; k < a.K; k += 8) {
        simdgroup_matrix<T, 8, 8> ma[4], mb[4];
        for (int i = 0; i < 4; ++i) simdgroup_load(ma[i], A + (m0 + 8 * i) * a.K + k, a.K);
        for (int j = 0; j < 4; ++j) simdgroup_load(mb[j], B + (n0 + 8 * j) * a.K + k, a.K, ulong2(0, 0), true);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], ma[i], mb[j], acc[i][j]);
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) simdgroup_store(acc[i][j], C + (m0 + 8 * i) * a.N + n0 + 8 * j, a.N);
}
template [[host_name("sg_mm_half")]] kernel decltype(sg_mm<half>) sg_mm<half>;
template [[host_name("sg_mm_float")]] kernel decltype(sg_mm<float>) sg_mm<float>;
