// 【文件导读】K16：实验：只置换 A 的共享内存列号
// 给初学者：这是回答某个性能问题的实验，不是编号越大越快；先了解对照版本与唯一改动，再看复杂细节。
// 先知道：kernel 是在 GPU 上执行的函数；K16 是本工程实验编号，不代表性能排名。
// 这一版在做什么：m XOR ((k & mask)<<shift) 改的是存放位置，不是矩阵元素；读写两边都用同一函数，数学含义才能不变。
// 谁调用它：src/runner.cu 的编号 16 入口选择参数；头文件不能直接用 bash 或 python 运行。
// 输入/输出：行主序 FP32 矩阵 A[M,K]、B[K,N]、C[M,N]，计算 C=alpha*AB+beta*C。
// 边界：这是特定分块/对齐条件下的实验核，不是任意尺寸 GEMM；先读调用方约束再改 M/N/K。
// 阅读顺序：看线程如何分工 → 数据怎么搬 → 何时同步 → 怎么乘加 → 如何写回 C。
// 评价方法：正确性 PASS 只说明该次测试满足误差条件；是否更快还要看同口径多轮计时和 profiler。
// 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

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
