// 【文件导读】K6：一次搬四个 float，计算仍是 FP32
// 给初学者：先看本文件的分工与下标，不必一次读懂全部模板；BM/BN 是输出块高宽，BK 是一次处理的 K 长度。
// 先知道：kernel 是在 GPU 上执行的函数；K6 是本工程实验编号，不代表性能排名。
// 这一版在做什么：float4 是 16 字节向量，不是四倍精度；A 只在 shared 中转置，原始全局矩阵仍按行存储。
// 谁调用它：src/runner.cu 的编号 6 入口选择参数；头文件不能直接用 bash 或 python 运行。
// 输入/输出：行主序 FP32 矩阵 A[M,K]、B[K,N]、C[M,N]，计算 C=alpha*AB+beta*C。
// 边界：这是特定分块/对齐条件下的实验核，不是任意尺寸 GEMM；先读调用方约束再改 M/N/K。
// 阅读顺序：看线程如何分工 → 数据怎么搬 → 何时同步 → 怎么乘加 → 如何写回 C。
// 评价方法：正确性 PASS 只说明该次测试满足误差条件；是否更快还要看同口径多轮计时和 profiler。
// 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// 【学习补充】Kernel 6：在二维分块上增加 float4 向量化搬运。
// float4 一次表达 4 个 FP32 元素（16 字节），并没有把精度改成 FP16。
// 向量化要求地址对齐、连续元素可访问以及相关维度可整除；本代码不处理通用尾块。
// A 在 global memory 中仍是 [M,K]，只在 shared memory 中转成 [BK,BM]。
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmVectorize(int M, int N, int K, float alpha, float *A,
                               float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // 每线程负责 TM×TN；BN/TN 是 N 方向线程小块数。
  // 原注释的 span a column 应结合这里的行主序索引理解。
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // 元素数不变；As 使用转置布局，Bs 仍是 [BK,BN]。
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // 当前 block 所负责的矩阵起点。
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // 一次搬运 128 bit / 32 bit = 4 个元素，所以列索引以 float4 为单位。
  // 此版本每轮每线程各搬一个 A/B 向量，线程总数必须与子块容量匹配。
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  // 线程私有的结果与 A/B 寄存器缓存。
  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // 沿 K 维遍历子块。
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // 从 A 连续读取四个数，写入 As 时交换子块行列。
    // 这样计算阶段同一 dotIdx 的多个 A 行在 shared memory 中连续。
    float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    // B 无需在 shared memory 中转置，直接按 float4 搬运。
    reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    __syncthreads();

    // 推进全局指针到下一段 K。
    A += BK;     // 向右 BK 列。
    B += BK * N; // 向下 BK 行。

    // 计算每线程负责的二维结果块。
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // 与 Kernel 5 对比：As 索引已经从 [row,k] 变为 [k,row]。
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
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

  // 写回也以连续四列为一组；TN 及相应地址须满足向量访问条件。
  for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      // 先读旧 C，因为 GEMM 包含 beta*C。
      float4 tmp = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      // 在寄存器中完成四个输出的 alpha/beta 更新。
      tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
      tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
      tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
      tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
      // 按 float4 写回同一位置。
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
          tmp;
    }
  }
}
