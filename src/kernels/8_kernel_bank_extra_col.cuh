#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// 【学习补充】Kernel 8：用行尾填充改变 Bs 的物理行跨度。
// extraCols 只改变 shared memory 布局，不增加数学上的 B 列数。
// 对不同访问模式，padding 的收益不同；5 是此实现的选择，不是通用最佳值。
// 仍需遵守整除、float4 对齐和子块加载覆盖条件。
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmResolveBankExtraCol(int M, int N, int K, float alpha,
                                         float *A, float *B, float beta,
                                         float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // 每线程计算 TM×TN；取余得到 N 方向的线程小块编号。
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // 为当前 A/B 子块分配共享内存。Bs 每行多留 extraCols 个位置。
  __shared__ float As[BM * BK];
  const int extraCols = 5;
  __shared__ float Bs[BK * (BN + extraCols)];

  // A/B/C 仍然使用原来的全局矩阵布局。
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // 每步每线程搬运 128 bit，即四个 FP32 元素。
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  // 线程私有结果与寄存器缓存。
  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // 沿 K 方向遍历子块。
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // A 在写入 shared memory 时转置。
    float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    // B 从 global memory 向量读取，随后按带 padding 的行跨度逐元素存储。
    // 不直接把 Bs 行首强转 float4，因为 BN+5 不保持每行的 16 字节对齐。
    tmp = reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 0] = tmp.x;
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 1] = tmp.y;
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 2] = tmp.z;
    Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 3] = tmp.w;
    __syncthreads();

    // 推进全局指针；padding 不参与这个全局地址计算。
    A += BK;     // 向右 BK 列。
    B += BK * N; // 向下 BK 行。

    // 把本线程所需 A/B 值读入寄存器并累加。
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      }
      for (uint i = 0; i < TN; ++i) {
        // 读取也必须使用 BN+extraCols，而非 BN。
        regN[i] = Bs[dotIdx * (BN + extraCols) + threadCol * TN + i];
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

  // 输出矩阵 C 没有 padding，写回仍使用完整矩阵的 N。
  for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      // 读取旧 C 向量到寄存器。
      float4 tmp = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      // 保留 alpha/beta 的 GEMM 更新。
      tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
      tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
      tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
      tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
      // 向量写回。
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
          tmp;
    }
  }
}
