#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int WARPSIZE = 32; // 内建 warpSize 不是 constexpr，这里供编译期公式使用。

// 【学习补充】Kernel 10 的三层分工：block 的 BM×BN → warp 的 WM×WN
// → 线程在各个 warp 子块中的 TM×TN。不是增加了数学运算，而是重排所有权与复用。
// 搬运由整个 block 合作，计算则由各 warp 负责自己的输出区域。
namespace wt {
// 全局内存 → 共享内存：A 子块转置成 [BK,BM]，B 子块保留 [BK,BN]。
template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                             float *As, float *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB) {
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    const float4 tmp = reinterpret_cast<const float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    // 原作者保留的 PTX 替代写法，仍然注释掉；不会参与实际编译。
    // float4 tmp;
    // asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
    //     : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
    //     : "l"(&A[(innerRowA + offset) * K + innerColA * 4]));
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }

  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
    // 原作者的 B 加载 PTX 实验写法，仅作为注释保留。
    // asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
    //     : "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 0]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 1]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 2]),
    //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 3])
    //     : "l"(&B[(innerRowB + offset) * N + innerColB * 4]));
  }
}

// 共享内存 → 寄存器 → 外积累加；索引同时包含 warp、warp 子块与线程的位置。
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // 为本线程在整个 warp tile 中负责的所有子块填充寄存器。
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

    // 遍历 warp 子块组合：每份 regM 可与多个 regN 子块复用，反之亦然。
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        // 每个组合中，本线程计算 TM×TN 个结果。
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

} // wt 命名空间结束。

/*
 * @tparam BM block 在 M 方向缓存/计算的输出块大小。
 * @tparam BN block 在 N 方向缓存/计算的输出块大小。
 * @tparam BK 每次共享缓存覆盖的 K 长度。
 * @tparam WM 每个 warp 负责的连续输出区域的 M 大小。
 * @tparam WN 每个 warp 负责的连续输出区域的 N 大小。
 * @tparam WNITER warp 区域沿 N 方向分成的子块数。
 * @tparam TM 每线程在一个子块中的 M 大小。
 * @tparam TN 每线程在一个子块中的 N 大小。
 * @tparam NUM_THREADS 每个 block 的线程总数。
 * 【学习补充】原注释还列了 WMITER；它实际在函数内推导，并非调用方模板参数。
 * 必须满足 32 * TM * TN * WMITER * WNITER = WM * WN。
 * “warp 子块”是输出区域的分组，不是改变硬件 warp 的 32 线程大小。
 * 本实现假定分块、向量对齐和尺寸整除条件成立；约束还要查看 runner。
 */
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B,
                    float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // 当前 warp 在 block 输出块中的位置。
  const uint warpIdx = threadIdx.x / WARPSIZE; // 线程所属的 warp 编号。
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  // 由覆盖元素数推导 M 方向子块数，再算每个子块的 M/N 尺寸。
  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER; // 原例：64/2=32；实际值取决于模板参数。
  constexpr uint WSUBN = WN / WNITER; // 原例：32/2=16；不是固定常量。

  // lane 编号映射到 warp 子块中的线程小块坐标。
  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // 范围 [0,31]。
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // 原例 i%(16/4)。
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // 原例 i/4。

  // 全 block 共用的 A/B 子块缓存。
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // A/B 定位到 block 起点。
  A += cRow * BM * K;
  B += cCol * BN;
  // C 则直接定位到当前 warp 的输出起点；写回时不应再重复添加 warp 偏移。
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // 合作搬运使用全 block 的线程编号，每步每线程加载四个 FP32 元素。
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  // 本线程跨 warp 子块的所有结果。
  float threadResults[WMITER * TM * WNITER * TN] = {0.0};
  // 在 warp tile 范围内缓存本线程需要的 A/B 值。
  float regM[WMITER * TM] = {0.0};
  float regN[WNITER * TN] = {0.0};

  // 每段 K：合作加载 → 屏障 → 计算 → 屏障，仍是单份共享缓存。
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads();
    wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                        TN>(regM, regN, threadResults, As, Bs, warpRow, warpCol,
                            threadRowInWarp, threadColInWarp);
    A += BK;     // A 向右 BK 列。
    B += BK * N; // B 向下 BK 行。
    __syncthreads();
  }

  // 遍历 warp 子块，将寄存器结果写回 C。
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // 相对于 warp 起点，移动到当前子块。
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // 读取旧 C 的连续四列。
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0];
          // 对应 threadResults 中的展平索引。
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
