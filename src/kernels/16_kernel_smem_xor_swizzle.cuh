#pragma once

// K16 只改变 K10 的 A shared-memory 物理布局；数学、线程映射和累加顺序不变。
#ifndef SGEMM_K16_XOR_MASK
#define SGEMM_K16_XOR_MASK 1
#endif
#ifndef SGEMM_K16_XOR_SHIFT
#define SGEMM_K16_XOR_SHIFT 1
#endif

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS,
          const int AS_PAD, const int XOR_MASK, const int XOR_SHIFT>
#if SGEMM_K10_MIN_BLOCKS > 0
__global__ void __launch_bounds__(NUM_THREADS, SGEMM_K10_MIN_BLOCKS)
#else
__global__ void __launch_bounds__(NUM_THREADS)
#endif
    sgemmWarptilingASwizzle(int M, int N, int K, float alpha,
                            const float *__restrict__ A,
                            const float *__restrict__ B, float beta,
                            float *__restrict__ C) {
  static_assert(XOR_MASK == 1 || XOR_MASK == 3 || XOR_MASK == 7);
  static_assert(XOR_SHIFT >= 1 && XOR_SHIFT <= 3);
  static_assert(AS_PAD >= 0);

  constexpr uint AS_STRIDE = BM + AS_PAD;
  __shared__ float As[AS_STRIDE * BK];
  __shared__ float Bs[BK * BN];
  wt::runWarptilingCore<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS,
                        AS_STRIDE, XOR_MASK, XOR_SHIFT>(
      M, N, K, alpha, A, B, beta, C, blockIdx.y, blockIdx.x, As, Bs);
}
