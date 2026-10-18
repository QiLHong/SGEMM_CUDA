// 【文件导读】K15：实验：只重排 block 负责输出块的先后编号
// 给初学者：这是回答某个性能问题的实验，不是编号越大越快；先了解对照版本与唯一改动，再看复杂细节。
// 先知道：kernel 是在 GPU 上执行的函数；K15 是本工程实验编号，不代表性能排名。
// 这一版在做什么：grouped grid 像重排任务单，不改一个任务内部的乘加；最后一组可能不足 GROUP_M，必须用实际组高映射。
// 谁调用它：src/runner.cu 的编号 15 入口选择参数；头文件不能直接用 bash 或 python 运行。
// 输入/输出：行主序 FP32 矩阵 A[M,K]、B[K,N]、C[M,N]，计算 C=alpha*AB+beta*C。
// 边界：这是特定分块/对齐条件下的实验核，不是任意尺寸 GEMM；先读调用方约束再改 M/N/K。
// 阅读顺序：看线程如何分工 → 数据怎么搬 → 何时同步 → 怎么乘加 → 如何写回 C。
// 评价方法：正确性 PASS 只说明该次测试满足误差条件；是否更快还要看同口径多轮计时和 profiler。
// 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

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
