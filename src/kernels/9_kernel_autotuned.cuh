// 【文件导读】K9：由外部脚本挑选分块参数的版本
// 给初学者：先看本文件的分工与下标，不必一次读懂全部模板；BM/BN 是输出块高宽，BK 是一次处理的 K 长度。
// 先知道：kernel 是在 GPU 上执行的函数；K9 是本工程实验编号，不代表性能排名。
// 这一版在做什么：autotuned 不表示 GPU 会自己搜索；脚本改参数、编译、计时。此处 WM/WN 不是 K10 那种严格的单 warp tile。
// 谁调用它：src/runner.cu 的编号 9 入口选择参数；头文件不能直接用 bash 或 python 运行。
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
const int K9_NUM_THREADS = 256;

// 【学习补充】Kernel 9：让分块参数参与外部自动调优。
// “autotuned” 不是 kernel 自己在运行时搜索；搜索脚本改参数、重编译并测量。
// 参数影响寄存器数量、共享内存、搬运覆盖和可用并行度，须同时满足 runner 断言。
// CEIL_DIV 不代表本 kernel 有尾块屏蔽；默认 benchmark 使用匹配尺寸。
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(K9_NUM_THREADS)
    sgemmAutotuned(int M, int N, int K, float alpha, float *A, float *B,
                   float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // 原文称 warptile 尺寸。
  // 【学习补充】这里 WM/TM = WN/TN = 16，配合 256 个线程铺成 16×16。
  // 因而这个区域不是“单个 32-thread warp”的输出块；真正 warp 分工见 Kernel 10。
  constexpr int WM = TM * 16;
  constexpr int WN = TN * 16;
  // 这类输出区域在 blocktile 的 M/N 方向上各迭代多少次。
  constexpr int WMITER = CEIL_DIV(BM, WM);
  constexpr int WNITER = CEIL_DIV(BN, WN);

  // 线程在当前区域内的位置。
  const int threadCol = threadIdx.x % (WN / TN);
  const int threadRow = threadIdx.x / (WN / TN);

  // A/B 子块的共享缓存，As 采用转置布局。
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // 移到当前 A/B/C 子块起点。
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // 向量加载：每线程每次四个 FP32；rowStride 是全块一轮搬运覆盖的行数。
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (K9_NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = K9_NUM_THREADS / (BN / 4);

  // 线程可能负责多个区域，因此结果数组乘以 WMITER*WNITER。
  float threadResults[WMITER * WNITER * TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // K 方向的子块循环。
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // 合作填充 As；循环只覆盖完整 rowStrideA，所以参数须满足整除约束。
    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
      float4 tmp = reinterpret_cast<float4 *>(
          &A[(innerRowA + offset) * K + innerColA * 4])[0];
      // 写入时转置 A 子块。
      As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
      As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
      As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
      As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }

    // B 按行主序向量搬入 Bs。
    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
      reinterpret_cast<float4 *>(
          &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
          reinterpret_cast<float4 *>(
              &B[(innerRowB + offset) * N + innerColB * 4])[0];
    }
    __syncthreads();

    // 依次遍历本线程负责的区域；每个区域内部仍是 TM×TN 外积累加。
    for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
      for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
          // 当前区域的共享内存 → 寄存器缓存。
          for (uint i = 0; i < TM; ++i) {
            regM[i] = As[dotIdx * BM + (wmIdx * WM) + threadRow * TM + i];
          }
          for (uint i = 0; i < TN; ++i) {
            regN[i] = Bs[dotIdx * BN + (wnIdx * WN) + threadCol * TN + i];
          }
          for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
              // threadResults 按 [WMITER*TM, WNITER*TN] 展平。
              threadResults[(wmIdx * TM + resIdxM) * (WNITER * TN) +
                            wnIdx * TN + resIdxN] +=
                  regM[resIdxM] * regN[resIdxN];
            }
          }
        }
      }
    }
    __syncthreads();
    // 所有线程用完缓存，再推进到下一段 K。
    A += BK;     // A 向右 BK 列。
    B += BK * N; // B 向下 BK 行。
  }

  // 把每个区域的结果映射回全局 C。
  for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
    for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
      float *C_interim = C + (wmIdx * WM * N) + (wnIdx * WN);
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // 先读旧 C 的连续四列。
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRow * TM + resIdxM) * N + threadCol * TN +
                         resIdxN])[0];
          // 寄存器中完成四个 GEMM 更新。
          const int i =
              (wmIdx * TM + resIdxM) * (WNITER * TN) + wnIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          // 写回，不改变 C 的行主序布局。
          reinterpret_cast<float4 *>(&C_interim[(threadRow * TM + resIdxM) * N +
                                                threadCol * TN + resIdxN])[0] =
              tmp;
        }
      }
    }
  }
}
