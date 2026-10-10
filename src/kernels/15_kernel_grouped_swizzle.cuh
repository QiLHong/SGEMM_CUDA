#pragma once

// K15 只改变 thread block 到输出 tile 的映射；计算主体直接复用 K10。
#ifndef SGEMM_K15_GROUP_M
#define SGEMM_K15_GROUP_M 1
#endif

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS,
          const int AS_PAD, const int GROUP_M>
#if SGEMM_K10_MIN_BLOCKS > 0
__global__ void __launch_bounds__(NUM_THREADS, SGEMM_K10_MIN_BLOCKS)
#else
__global__ void __launch_bounds__(NUM_THREADS)
#endif
    sgemmWarptilingGrouped(int M, int N, int K, float alpha,
                           const float *__restrict__ A,
                           const float *__restrict__ B, float beta,
                           float *__restrict__ C) {
  static_assert(GROUP_M > 0);
  static_assert(AS_PAD >= 0);

  const uint pid = blockIdx.x;
  const uint gridM = CEIL_DIV(M, BM);
  const uint gridN = CEIL_DIV(N, BN);
  const uint groupSize = GROUP_M * gridN;
  const uint groupId = pid / groupSize;
  const uint firstM = groupId * GROUP_M;
  const uint remainingM = gridM - firstM;
  const uint actualGroupM = remainingM < GROUP_M ? remainingM : GROUP_M;
  const uint localPid = pid % groupSize;
  const uint cRow = firstM + localPid % actualGroupM;
  const uint cCol = localPid / actualGroupM;

  constexpr uint AS_STRIDE = BM + AS_PAD;
  __shared__ float As[AS_STRIDE * BK];
  __shared__ float Bs[BK * BN];
  wt::runWarptilingCore<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS,
                        AS_STRIDE>(M, N, K, alpha, A, B, beta, C, cRow, cCol,
                                   As, Bs);
}
