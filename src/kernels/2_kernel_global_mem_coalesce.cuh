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
