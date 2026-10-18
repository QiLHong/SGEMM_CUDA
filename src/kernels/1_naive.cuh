// 【文件导读】K1：从一个线程计算一个输出开始
// 给初学者：先看本文件的分工与下标，不必一次读懂全部模板；BM/BN 是输出块高宽，BK 是一次处理的 K 长度。
// 先知道：kernel 是在 GPU 上执行的函数；K1 是本工程实验编号，不代表性能排名。
// 这一版在做什么：不显式缓存 A/B；每个线程沿 K 做一遍点积。x 是行、y 是列，同一 warp 的 A 地址可能跨很大步长。
// 谁调用它：src/runner.cu 的编号 1 入口选择参数；头文件不能直接用 bash 或 python 运行。
// 输入/输出：行主序 FP32 矩阵 A[M,K]、B[K,N]、C[M,N]，计算 C=alpha*AB+beta*C。
// 边界：代码检查输出行列是否越界；更多高性能版本不一定有这种保护，不可类推。
// 阅读顺序：看线程如何分工 → 数据怎么搬 → 何时同步 → 怎么乘加 → 如何写回 C。
// 评价方法：正确性 PASS 只说明该次测试满足误差条件；是否更快还要看同口径多轮计时和 profiler。
// 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*
矩阵尺寸（以下数组均按行主序存储）：
A[M, K] × B[K, N] = C[M, N]
实际计算：C = alpha * A * B + beta * C。

【学习补充】Kernel 1：一个线程负责一个输出元素。
行主序地址 = 行号 * 该矩阵的列数 + 列号；因此 A 的行跨度为 K，B/C 为 N。
先理解“一个输出 = A 的一行与 B 的一列做点积”，再观察线程如何对应输出。
*/

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  // x 在这里是输出的行号，y 是列号；不要把变量名 x 自动理解成列。
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // 网格按整块向上取整，边缘线程可能超出矩阵，因此必须检查边界。
  // 原文 tile quantization 指分块取整，不是 INT8/INT4 数值量化。
  if (x < M && y < N) {
    // tmp 是该线程独有的累加值；沿 K 维依次累加乘积。
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      // 【学习补充】同一 warp 的 threadIdx.x 连续，x 随之变化。
      // 固定 i 时，相邻线程读 A 的地址相差 K 个 float，不是连续元素。
      // 这是与 Kernel 2 对比时应重点观察的访问方式。
      tmp += A[x * K + i] * B[i * N + y];
    }
    // 融合 alpha/beta 缩放：不能丢掉旧 C，除非实际输入 beta 为 0。
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}
