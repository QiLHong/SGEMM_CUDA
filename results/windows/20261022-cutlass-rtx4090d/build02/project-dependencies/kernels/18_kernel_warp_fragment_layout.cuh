// 【文件导读】K18：实验：重新给 lane 分配输出片段
// 给初学者：这是回答某个性能问题的实验，不是编号越大越快；先了解对照版本与唯一改动，再看复杂细节。
// 先知道：kernel 是在 GPU 上执行的函数；K18 是本工程实验编号，不代表性能排名。
// 这一版在做什么：fragment 指一个线程拿到的一小组输入或结果，不是新矩阵；改 ownership 后，shared 读取与 C 写回必须对应。
// 谁调用它：src/runner.cu 的编号 18 入口选择参数；头文件不能直接用 bash 或 python 运行。
// 输入/输出：行主序 FP32 矩阵 A[M,K]、B[K,N]、C[M,N]，计算 C=alpha*AB+beta*C。
// 边界：这是特定分块/对齐条件下的实验核，不是任意尺寸 GEMM；先读调用方约束再改 M/N/K。
// 阅读顺序：看线程如何分工 → 数据怎么搬 → 何时同步 → 怎么乘加 → 如何写回 C。
// 评价方法：正确性 PASS 只说明该次测试满足误差条件；是否更快还要看同口径多轮计时和 profiler。
// 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

#pragma once

// K18 freezes K10's block/warp tiles, global-memory loads, shared-memory
// layout, barriers and pipeline.  The only experimental variables are the
// lane-to-fragment mapping and the per-thread M/N fragment shape.
namespace wt_layout {

template <const int VARIANT, const int ROW_GROUPS, const int COL_GROUPS>
__device__ __forceinline__ void mapLane(const uint lane, uint &threadRow,
                                        uint &threadCol) {
  static_assert(ROW_GROUPS * COL_GROUPS == WARPSIZE);
  if constexpr (VARIANT == 1) {
    // V1: transpose the 8x4 lane lattice. Consecutive lanes move along M.
    threadRow = lane % ROW_GROUPS;
    threadCol = lane / ROW_GROUPS;
  } else {
    threadRow = lane / COL_GROUPS;
    threadCol = lane % COL_GROUPS;
    if constexpr (VARIANT == 3) {
      // V3: permute the four N groups inside each M group. This preserves the
      // exact per-warp output set while changing which physical lane owns it.
      threadCol ^= threadRow & (COL_GROUPS - 1);
    }
  }
}

template <const int VARIANT, const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER, const int TM,
          const int TN, const int NUM_THREADS, const int AS_STRIDE>
__device__ __forceinline__ void
run(int M, int N, int K, float alpha, const float *__restrict__ A,
    const float *__restrict__ B, float beta, float *__restrict__ C,
    const uint cRow, const uint cCol, float *__restrict__ As,
    float *__restrict__ Bs) {
  constexpr uint WMITER =
      (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER;
  constexpr uint WSUBN = WN / WNITER;
  constexpr uint ROW_GROUPS = WSUBM / TM;
  constexpr uint COL_GROUPS = WSUBN / TN;
  static_assert(VARIANT >= 1 && VARIANT <= 5);
  static_assert(NUM_THREADS == 128 && BM == 128 && BN == 128 && BK == 16);
  static_assert(WM == 64 && WN == 64 && AS_STRIDE == BM + 4);
  static_assert(WMITER == 1);
  static_assert(ROW_GROUPS * COL_GROUPS == WARPSIZE);
  static_assert(TN % 4 == 0);
  static_assert(WMITER * TM * WNITER * TN == 128,
                "all K18 variants keep 128 accumulators per thread");
  static_assert(VARIANT != 3 || (COL_GROUPS & (COL_GROUPS - 1)) == 0,
                "XOR lane swizzle requires a power-of-two N group count");

  const uint warpIdx = threadIdx.x / WARPSIZE;
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);
  const uint lane = threadIdx.x % WARPSIZE;
  uint threadRowInWarp;
  uint threadColInWarp;
  mapLane<VARIANT, ROW_GROUPS, COL_GROUPS>(lane, threadRowInWarp,
                                           threadColInWarp);

  A += cRow * BM * K;
  B += cCol * BN;
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
  float regM[WMITER * TM] = {0.0f};
  float regN[WNITER * TN] = {0.0f};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // The exact K10 global->shared helper and A/B shared layouts are reused.
    wt::loadFromGmem<BM, BN, BK, AS_STRIDE, rowStrideA, rowStrideB>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads();
    wt::processFromSmem<BM, BN, BK, AS_STRIDE, WM, WN, WMITER, WNITER,
                        WSUBM, WSUBN, TM, TN>(
        regM, regN, threadResults, As, Bs, warpRow, warpCol, threadRowInWarp,
        threadColInWarp);
    A += BK;
    B += BK * N;
    __syncthreads();
  }

  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      float *CInterim =
          C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          float4 value = reinterpret_cast<float4 *>(
              &CInterim[(threadRowInWarp * TM + resIdxM) * N +
                        threadColInWarp * TN + resIdxN])[0];
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          value.x = alpha * threadResults[i + 0] + beta * value.x;
          value.y = alpha * threadResults[i + 1] + beta * value.y;
          value.z = alpha * threadResults[i + 2] + beta * value.z;
          value.w = alpha * threadResults[i + 3] + beta * value.w;
          reinterpret_cast<float4 *>(
              &CInterim[(threadRowInWarp * TM + resIdxM) * N +
                        threadColInWarp * TN + resIdxN])[0] = value;
        }
      }
    }
  }
}

} // namespace wt_layout

template <const int VARIANT, const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER, const int TM,
          const int TN, const int NUM_THREADS, const int AS_PAD>
#if SGEMM_K10_MIN_BLOCKS > 0
__global__ void __launch_bounds__(NUM_THREADS, SGEMM_K10_MIN_BLOCKS)
#else
__global__ void __launch_bounds__(NUM_THREADS)
#endif
    sgemmWarptilingLayout(int M, int N, int K, float alpha,
                          const float *__restrict__ A,
                          const float *__restrict__ B, float beta,
                          float *__restrict__ C) {
  static_assert(BM == 128 && BN == 128 && BK == 16);
  static_assert(WM == 64 && WN == 64 && NUM_THREADS == 128 && AS_PAD == 4);
  constexpr uint AS_STRIDE = BM + AS_PAD;
  __shared__ float As[AS_STRIDE * BK];
  __shared__ float Bs[BK * BN];
  if constexpr (VARIANT == 0) {
    // V0 deliberately calls the unchanged K10 core; this is the control used
    // for the SASS identity check.
    wt::runWarptilingCore<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS,
                          AS_STRIDE>(M, N, K, alpha, A, B, beta, C,
                                     blockIdx.y, blockIdx.x, As, Bs);
  } else {
    wt_layout::run<VARIANT, BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS,
                   AS_STRIDE>(M, N, K, alpha, A, B, beta, C, blockIdx.y,
                              blockIdx.x, As, Bs);
  }
}
