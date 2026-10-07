#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// 【学习补充】Kernel 3：先合作搬运 A/B 小块到 shared memory，再重复使用。
// 每个 block 计算 BLOCKSIZE×BLOCKSIZE 个 C 元素，每线程仍只计算一个。
// As 的一个元素被同一输出行的多个线程复用，Bs 的一个元素被多行复用。
// 本 kernel 没有边界屏蔽；M/N/K 必须满足当前分块覆盖要求。
// runner 的向上取整并不自动让这个 kernel 支持任意尺寸。
template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
  // 当前线程块负责的输出块坐标（不是完整矩阵中的元素坐标）。
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // 在较快的共享内存中为当前 A/B 子块分配缓存。
  // 同一 block 内的线程共用这两块数组，不同 block 不共用。
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // 当前线程在输出块内部负责的行、列。
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // 把指针移到该 block 的起点；后面的索引相对于这些新起点计算。
  A += cRow * BLOCKSIZE * K;                    // A 的行块起点，列为 0。
  B += cCol * BLOCKSIZE;                        // B 的列块起点，行为 0。
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // C 的二维输出块起点。

  float tmp = 0.0;
  // 沿 K 维分段：每次处理 BLOCKSIZE 个乘加项。
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // 每个线程分别为 As/Bs 搬运一个元素。
    // threadCol 在相邻线程间连续，以便全局内存合并访问。
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // 第一道屏障：所有线程都完成写入后，才能读取其他线程搬来的数据。
    __syncthreads();
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // 当前子块的点积；从 shared memory 而非反复从 global memory 取数。
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // 第二道屏障：等待所有线程用完当前子块，避免快线程提前覆盖缓存。
    // 两次 __syncthreads() 解决不同问题，不能任意删掉一个。
    __syncthreads();
  }
  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
}
