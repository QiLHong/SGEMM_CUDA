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
