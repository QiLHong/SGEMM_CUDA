// 【文件导读】K4：一个线程负责一小列输出
// 给初学者：先看本文件的分工与下标，不必一次读懂全部模板；BM/BN 是输出块高宽，BK 是一次处理的 K 长度。
// 先知道：kernel 是在 GPU 上执行的函数；K4 是本工程实验编号，不代表性能排名。
// 这一版在做什么：threadResults[TM] 保存该线程的 TM 个部分和；读一个 B 值可配多个 A 值，减少反复从 shared 取数。
// 谁调用它：src/runner.cu 的编号 4 入口选择参数；头文件不能直接用 bash 或 python 运行。
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

// 【学习补充】BM/BN：输出块的行/列数；BK：每次遍历的 K 长度。
// TM：一个线程沿 M 方向计算 TM 个结果，而不是像 Kernel 3 只算一个。
// 每次读出的一个 B 值可复用于 TM 次乘加，减少 shared memory 读取。
// 本教学实现没有通用尾块屏蔽；使用 runner 已匹配的参数与整除尺寸。
template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  // 原作者观察：交换这里的 x/y 后，大矩阵性能约下降 30%。
  // 当前排列让连续 block ID 对应 B 的连续列块，并复用 A 的同一行块。
  // 原作者将差异归因于更好的空间局部性与 L2 命中率；这不是所有 GPU 的保证。
  // 注意：这里与 Kernel 1–3 的网格轴含义不同，须同时检查 runner 的 gridDim。
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // 一个 warp 计算 32*TM 个元素，32 对应列方向（runner 参数下）。
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  // 当前 A/B 子块的共享内存缓存，分别为 [BM,BK] 与 [BK,BN]。
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // 移到 A 行块、B 列块及 C 输出块的起点。
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // 原作者待办：让每个线程搬运多个元素，以更好地利用缓存容量。
  // 当前每线程恰好加载一个 A、一个 B；下面两个断言检查这种映射的前提。
  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);
  const uint innerColA = threadIdx.x % BK; // A 搬运时的块内列，组织合并访问。
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN; // B 搬运时的块内列，组织合并访问。
  const uint innerRowB = threadIdx.x / BN;

  // 线程私有的 TM 个累加值，目标是由编译器放入寄存器。
  float threadResults[TM] = {0.0};

  // 沿 K 维遍历所有子块。
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // 合作填满共享内存缓存，然后等待所有线程完成搬运。
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    // 为下一段 K 子块推进全局指针，不影响当前 As/Bs。
    A += BK;
    B += BK * N;

    // dotIdx 放在外层，以便先把一个 B 元素读入 tmpB 并重复使用。
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        // 当前线程负责同一列上的 TM 行；threadRow 是“线程行组”编号。
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    // 所有线程用完当前共享缓存后，才允许下一轮覆盖它。
    __syncthreads();
  }

  // 将本线程 TM 个结果写回 C，并保留完整的 alpha/beta 更新。
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta * C[(threadRow * TM + resIdx) * N + threadCol];
  }
}
