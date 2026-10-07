#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// 【学习补充】Kernel 5：每线程计算 TM×TN 的二维寄存器小块。
// 每个 dotIdx 读取 TM 个 A 和 TN 个 B，组合出 TM*TN 次乘加。
// 复用更高，但 threadResults 越大，寄存器压力也越高；不是 tile 越大越好。
// __launch_bounds__ 向编译器提供线程数/驻留目标信息，不等于保证实际 occupancy。
// 本版本仍要求尺寸、分块与搬运步长匹配；没有通用边界处理。
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint totalResultsBlocktile = BM * BN;
  // 一个线程负责输出块中的 TM*TN 个元素。
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  // 每块输出数 / 每线程输出数 = 每块线程数。
  assert(numThreadsBlocktile == blockDim.x);

  // BN/TN 是沿 N 方向铺满输出块所需的线程小块数。
  // 【注释辨析】原文称 span a column；以下取余实际对应横向的列小块编号。
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // 当前 A/B 子块占用的共享内存。
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // 移到 A 行块、B 列块和 C 输出块的起点。
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // 搬运任务与计算任务分开映射：这些坐标用于合作填充 As/Bs。
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  // 全块线程每轮可覆盖 As 的多少行；循环让每线程搬运多个元素。
  const uint strideA = numThreadsBlocktile / BK;
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  // 对 As/Bs，先让线程覆盖连续列，再按行步长推进，以改善全局合并访问。
  const uint strideB = numThreadsBlocktile / BN;

  // 线程私有结果与 A/B 寄存器缓存。
  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // 最外层遍历 K 方向的子块。
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // 多轮合作加载，使较少的计算线程仍能填满整个共享缓存。
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // 推进到下一段 K。
    A += BK;     // A 向右移动 BK 列。
    B += BK * N; // B 向下移动 BK 行。

    // 每个 dotIdx 做一次 TM×TN 外积，累加到线程的二维结果块。
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // 共享内存 → 线程私有寄存器缓存。
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // 寄存器结果映射回 C；这里 N 是完整 C 的行跨度，不是 BN。
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          alpha * threadResults[resIdxM * TN + resIdxN] +
          beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
    }
  }
}
