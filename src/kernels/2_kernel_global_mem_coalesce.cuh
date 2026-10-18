// 【文件导读】K2：只改变相邻线程的读写位置
// 给初学者：先看本文件的分工与下标，不必一次读懂全部模板；BM/BN 是输出块高宽，BK 是一次处理的 K 长度。
// 先知道：kernel 是在 GPU 上执行的函数；K2 是本工程实验编号，不代表性能排名。
// 这一版在做什么：把连续线程放到输出的连续列，使 B 的读取与 C 的写回连续；coalescing 是合并内存请求，不是合并计算结果。
// 谁调用它：src/runner.cu 的编号 2 入口选择参数；头文件不能直接用 bash 或 python 运行。
// 输入/输出：行主序 FP32 矩阵 A[M,K]、B[K,N]、C[M,N]，计算 C=alpha*AB+beta*C。
// 边界：代码检查输出行列是否越界；更多高性能版本不一定有这种保护，不可类推。
// 阅读顺序：看线程如何分工 → 数据怎么搬 → 何时同步 → 怎么乘加 → 如何写回 C。
// 评价方法：正确性 PASS 只说明该次测试满足误差条件；是否更快还要看同口径多轮计时和 profiler。
// 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// 【学习补充】Kernel 2：保持一个线程计算一个输出，只改变线程到行/列的映射。
// runner 使用 BLOCKSIZE=32、一个 block 含 32*32 个线程。
// threadIdx.x 是一维线性编号：除法得到行，取余得到列。
template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
  const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  // 网格按整块向上取整；屏蔽边缘块中超出 M/N 的线程。
  if (cRow < M && cCol < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      // BLOCKSIZE=32 时，一个 warp 对应同一行的连续 32 列：
      // A[cRow*K+i] 在这些线程间相同，B[i*N+cCol] 的地址连续。
      // 相比 Kernel 1，这有利于全局内存合并访问（coalescing）。
      // 注意：此时仍未显式使用 shared memory，K 次点积循环也没有减少。
      tmp += A[cRow * K + i] * B[i * N + cCol];
    }
    // 同一 warp 写回的 C 元素也连续；数学公式与 Kernel 1 完全相同。
    C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
  }
}
