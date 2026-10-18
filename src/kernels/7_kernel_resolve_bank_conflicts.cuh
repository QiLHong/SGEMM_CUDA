// 【文件导读】K7：实验：调整 B 在共享内存中的摆放顺序
// 给初学者：先看本文件的分工与下标，不必一次读懂全部模板；BM/BN 是输出块高宽，BK 是一次处理的 K 长度。
// 先知道：kernel 是在 GPU 上执行的函数；K7 是本工程实验编号，不代表性能排名。
// 这一版在做什么：写入与读取必须使用配套映射；降低 bank 冲突的目标不等于已经测得提速，额外地址运算也有成本。
// 谁调用它：src/runner.cu 的编号 7 入口选择参数；头文件不能直接用 bash 或 python 运行。
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

// 【学习补充】Kernel 7：改变 Bs 的共享内存布局，尝试减少 bank 冲突。
// 对 FP32 地址，可先用“字节地址/4 再模 32”推导 bank（本教程目标架构）。
// 同一 warp 访问同 bank 的不同字可能冲突；读取同一字的广播不能等同于冲突。
// 本版本的 8、16 是布局常量，匹配 runner 的 TN=8、BN=128；不是通用模板。
// 降低 bank 冲突不保证更快：这里重排时还增加了标量写入和地址计算。
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmResolveBankConflicts(int M, int N, int K, float alpha,
                                          float *A, float *B, float beta,
                                          float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // BN/TN 是 N 方向线程小块数；原注释称 span a column，按下式理解。
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // As 转置存储；Bs 的元素数不变，但物理排列与 Kernel 6 不同。
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // 移到当前 block 的 A/B/C 子块起点。
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // 合作搬运坐标；每步每线程从全局内存读取四个 FP32 元素。
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  // 线程私有的结果和 A/B 寄存器缓存。
  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // 沿 K 维遍历子块；与此前优化 kernel 一样，未实现通用尾块屏蔽。
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // A 在加载到共享内存时转置。
    float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    // 原文称把 Bs “线性化”：把连续 8 列分成两个 float4，
    // 改存为 [k*8 + 组内列, 8列组编号]，最内层有 16 个组。
    // 写入与下面 regN 的读取必须采用同一映射，不能只改其中一侧。
    tmp = reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 0) * 16 + innerColB / 2] = tmp.x;
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 1) * 16 + innerColB / 2] = tmp.y;
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 2) * 16 + innerColB / 2] = tmp.z;
    Bs[((innerColB % 2) * 4 + innerRowB * 8 + 3) * 16 + innerColB / 2] = tmp.w;
    __syncthreads();

    // 推进到下一段 K。
    A += BK;     // A 向右 BK 列。
    B += BK * N; // B 向下 BK 行。

    // 读取共享缓存到寄存器，再执行 TM×TN 外积。
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[(dotIdx * 8 + i) * 16 + threadCol];
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

  // 沿 C 的列方向每次写回四个结果。
  for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      // 读取旧 C 的向量。
      float4 tmp = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      // 在寄存器中进行 GEMM 的 alpha/beta 更新。
      tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
      tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
      tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
      tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
      // 写回。
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
          tmp;
    }
  }
}
