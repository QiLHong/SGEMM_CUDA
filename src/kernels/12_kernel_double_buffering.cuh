// 【文件导读】K12：原作者的 CUDA 异步复制与 barrier 实验
// 给初学者：先看本文件的分工与下标，不必一次读懂全部模板；BM/BN 是输出块高宽，BK 是一次处理的 K 长度。
// 先知道：kernel 是在 GPU 上执行的函数；K12 是本工程实验编号，不代表性能排名。
// 这一版在做什么：cuda::memcpy_async 先提交搬运、barrier 等待完成；发出复制不代表已可读取，也不自动保证速度更快。
// 谁调用它：src/runner.cu 的编号 12 入口选择参数；头文件不能直接用 bash 或 python 运行。
// 输入/输出：行主序 FP32 矩阵 A[M,K]、B[K,N]、C[M,N]，计算 C=alpha*AB+beta*C。
// 边界：这是特定分块/对齐条件下的实验核，不是任意尺寸 GEMM；先读调用方约束再改 M/N/K。
// 阅读顺序：看线程如何分工 → 数据怎么搬 → 何时同步 → 怎么乘加 → 如何写回 C。
// 评价方法：正确性 PASS 只说明该次测试满足误差条件；是否更快还要看同口径多轮计时和 profiler。
// 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

#pragma once

#include <algorithm>
#include <cassert>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda/barrier>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// 【学习补充】Kernel 12：用 cuda::memcpy_async 与两份 barrier 管理双缓冲。
// 异步提交不等于数据已经可读；计算前必须等待当前槽的复制完成。
// 这是 CUDA C++ 的异步复制接口，不是 PyTorch CUDA Graph，也不是自动保证提速。
namespace {
// A 需要转置写入 shared memory，所以拆成四次单 float 异步复制；
// B 的四个元素连续，使用一次 float4 大小的复制。
template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB, typename T>
__device__ void loadFromGmem(int N, int K, float *A, float *B, float *As,
                             float *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB, T &barrier) {

  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    // aligned_size_t 表达调用者提供的对齐条件，不会替你修正未对齐地址。
    cuda::memcpy_async(&As[(innerColA * 4 + 0) * BM + innerRowA + offset],
                       &A[(innerRowA + offset) * K + innerColA * 4],
                       cuda::aligned_size_t<sizeof(float)>(sizeof(float)),
                       barrier);
    cuda::memcpy_async(&As[(innerColA * 4 + 1) * BM + innerRowA + offset],
                       &A[(innerRowA + offset) * K + innerColA * 4 + 1],
                       cuda::aligned_size_t<sizeof(float)>(sizeof(float)),
                       barrier);
    cuda::memcpy_async(&As[(innerColA * 4 + 2) * BM + innerRowA + offset],
                       &A[(innerRowA + offset) * K + innerColA * 4 + 2],
                       cuda::aligned_size_t<sizeof(float)>(sizeof(float)),
                       barrier);
    cuda::memcpy_async(&As[(innerColA * 4 + 3) * BM + innerRowA + offset],
                       &A[(innerRowA + offset) * K + innerColA * 4 + 3],
                       cuda::aligned_size_t<sizeof(float)>(sizeof(float)),
                       barrier);
  }

  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    cuda::memcpy_async(&Bs[(innerRowB + offset) * BN + innerColB * 4],
                       &B[(innerRowB + offset) * N + innerColB * 4],
                       cuda::aligned_size_t<sizeof(float4)>(sizeof(float4)),
                       barrier);
  }
}

// 数学计算保持 warp tiling 的外积累加结构；这里只读取已就绪的共享槽。
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // 填充本线程在整个 warp tile 中所需的寄存器。
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] =
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }

    // 各 warp 子块组合的矩阵乘。
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        // 当前线程的 TM×TN 结果。
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          (wSubColIdx * TN) + resIdxN] +=
                regM[wSubRowIdx * TM + resIdxM] *
                regN[wSubColIdx * TN + resIdxN];
          }
        }
      }
    }
  }
}

} // 匿名命名空间结束。

/*
 * @tparam BM block 输出区域在 M 方向的大小。
 * @tparam BN block 输出区域在 N 方向的大小。
 * @tparam BK 每次 shared memory 缓存的 K 长度。
 * @tparam WM 每 warp 连续输出区域的 M 大小。
 * @tparam WN 每 warp 连续输出区域的 N 大小。
 * @tparam WNITER warp 区域沿 N 的子块数。
 * @tparam TM 每线程在子块内计算的 M 大小。
 * @tparam TN 每线程在子块内计算的 N 大小。
 * @tparam NUM_THREADS block 线程数。
 * 【学习补充】WMITER 在函数体内推导，并非外部模板参数。
 * 本实现要求 K 至少包含一个完整 BK 子块，且尺寸整除/向量对齐条件成立。
 * 不能通过改循环边界就把它当成支持任意形状的 GEMM。
 */
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    runSgemmDoubleBuffering2(int M, int N, int K, float alpha, float *A,
                             float *B, float beta, float *C) {
  // 初始化两份 block 作用域的 barrier；初始化之后要让全部线程可见。
  auto block = cooperative_groups::this_thread_block();
  __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> frontBarrier;
  __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> backBarrier;
  auto frontBarrierPtr = &frontBarrier;
  auto backBarrierPtr = &backBarrier;
  if (block.thread_rank() == 0) {
    init(&frontBarrier, block.size());
    init(&backBarrier, block.size());
  }
  __syncthreads();

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // warp 在 block 输出区域内的位置。
  const uint warpIdx = threadIdx.x / WARPSIZE; // 当前线程所属的 warp。
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  // warp 子块大小。
  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER; // 原例 64/2=32，实际由参数决定。
  constexpr uint WSUBN = WN / WNITER; // 原例 32/2=16。

  // lane 映射到 warp 子块中的线程小块。
  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0,31]。
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // 原例 i%(16/4)。
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // 原例 i/4。

  // 两份 A/B 共享槽，offset 在 0/1 之间交替。
  __shared__ float As[2 * BM * BK];
  __shared__ float Bs[2 * BK * BN];

  // A/B 定位到 block 起点。
  A += cRow * BM * K;
  B += cCol * BN;
  // C 定位到 warp 的输出起点。
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // 全 block 合作加载，每线程每步负责四个 FP32 元素。
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  // 本线程的累加结果与 warp tile 范围的 A/B 缓存。
  float threadResults[WMITER * TM * WNITER * TN] = {0.0};
  float regM[WMITER * TM] = {0.0};
  float regN[WNITER * TN] = {0.0};

  int As_offset = 0;
  int Bs_offset = 0;

  // 流水预热：提交第一个子块的异步复制到 front 槽。
  loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
      N, K, A, B, As + As_offset * BM * BK, Bs + Bs_offset * BK * BN, innerRowA,
      innerColA, innerRowB, innerColB, (*frontBarrierPtr));

  // 遍历除最后一个之外的所有 K 子块。
  for (uint bkIdx = 0; bkIdx < K - BK; bkIdx += BK) {
    // 先把下一块复制提交到 back 槽，尝试与当前块计算重叠。
    loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A + BK, B + BK * N, As + (1 - As_offset) * BM * BK,
        Bs + (1 - Bs_offset) * BK * BN, innerRowA, innerColA, innerRowB,
        innerColB, (*backBarrierPtr));

    // 等待 front 的复制和线程到达条件满足，才能读取当前块。
    (*frontBarrierPtr).arrive_and_wait();
    processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
        regM, regN, threadResults, As + As_offset * BM * BK,
        Bs + Bs_offset * BK * BN, warpRow, warpCol, threadRowInWarp,
        threadColInWarp);
    A += BK;     // A 向右 BK 列。
    B += BK * N; // B 向下 BK 行。

    // 数据槽和它们对应的 barrier 必须一起交换，保持等待对象匹配。
    As_offset = 1 - As_offset;
    Bs_offset = 1 - Bs_offset;
    auto tmp = frontBarrierPtr;
    frontBarrierPtr = backBarrierPtr;
    backBarrierPtr = tmp;

    // 等所有线程用完旧槽，下一轮才可以复用它作为 back。
    __syncthreads();
  }

  // 流水排空：最后一块没有后继块，等待并计算即可。
  (*frontBarrierPtr).arrive_and_wait();
  processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
      regM, regN, threadResults, As + As_offset * BM * BK,
      Bs + Bs_offset * BK * BN, warpRow, warpCol, threadRowInWarp,
      threadColInWarp);

  // 写回所有 warp 子块结果。
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // 移到当前 warp 子块。
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // 读取旧 C 向量。
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0];
          // 寄存器中完成四个 GEMM 更新。
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          // 写回。
          reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0] = tmp;
        }
      }
    }
  }
}
