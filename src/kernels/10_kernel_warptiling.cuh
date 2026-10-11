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
// A 在 shared memory 中的逻辑坐标仍为 [k,m]。MASK=0 时保持 K10 的原始布局；
// K16 只把 m 的若干低位与 k 混合，写入和读取必须调用同一个函数。
template <const int BM, const int BK, const int AS_STRIDE,
          const int A_XOR_MASK, const int A_XOR_SHIFT>
__device__ __forceinline__ constexpr uint aSmemIndex(const uint k,
                                                      const uint m) {
  static_assert(A_XOR_MASK >= 0 && A_XOR_SHIFT >= 0);
  static_assert((BM & (BM - 1)) == 0,
                "A XOR swizzle requires power-of-two BM");
  static_assert(((((BK - 1) & A_XOR_MASK) << A_XOR_SHIFT)) < BM,
                "A XOR swizzle must stay inside the logical BM columns");
  if constexpr (A_XOR_MASK == 0) {
    return k * AS_STRIDE + m;
  } else {
    return k * AS_STRIDE +
           (m ^ ((k & A_XOR_MASK) << A_XOR_SHIFT));
  }
}

// 全局内存 → 共享内存：A 子块转置成 [BK,BM]，B 子块保留 [BK,BN]。
template <const int BM, const int BN, const int BK, const int AS_STRIDE,
          const int rowStrideA, const int rowStrideB,
          const int A_XOR_MASK = 0, const int A_XOR_SHIFT = 0>
__device__ void loadFromGmem(int N, int K, const float *__restrict__ A,
                             const float *__restrict__ B,
                             float *__restrict__ As, float *__restrict__ Bs,
                             int innerRowA, int innerColA, int innerRowB,
                             int innerColB) {
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    const float4 tmp = reinterpret_cast<const float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    // 原作者保留的 PTX 替代写法，仍然注释掉；不会参与实际编译。
    // float4 tmp;
    // asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
    //     : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
    //     : "l"(&A[(innerRowA + offset) * K + innerColA * 4]));
    As[aSmemIndex<BM, BK, AS_STRIDE, A_XOR_MASK, A_XOR_SHIFT>(
        innerColA * 4 + 0, innerRowA + offset)] = tmp.x;
    As[aSmemIndex<BM, BK, AS_STRIDE, A_XOR_MASK, A_XOR_SHIFT>(
        innerColA * 4 + 1, innerRowA + offset)] = tmp.y;
    As[aSmemIndex<BM, BK, AS_STRIDE, A_XOR_MASK, A_XOR_SHIFT>(
        innerColA * 4 + 2, innerRowA + offset)] = tmp.z;
    As[aSmemIndex<BM, BK, AS_STRIDE, A_XOR_MASK, A_XOR_SHIFT>(
        innerColA * 4 + 3, innerRowA + offset)] = tmp.w;
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
template <const int BM, const int BN, const int BK, const int AS_STRIDE,
          const int WM, const int WN, const int WMITER, const int WNITER,
          const int WSUBM, const int WSUBN, const int TM, const int TN,
          const int A_XOR_MASK = 0, const int A_XOR_SHIFT = 0>
__device__ void
processFromSmem(float *__restrict__ regM, float *__restrict__ regN,
                float *__restrict__ threadResults,
                const float *__restrict__ As, const float *__restrict__ Bs,
                const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // 为本线程在整个 warp tile 中负责的所有子块填充寄存器。
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < TM; ++i) {
        const uint logicalM = warpRow * WM + wSubRowIdx * WSUBM +
                              threadRowInWarp * TM + i;
        regM[wSubRowIdx * TM + i] =
            As[aSmemIndex<BM, BK, AS_STRIDE, A_XOR_MASK, A_XOR_SHIFT>(
                dotIdx, logicalM)];
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

// K10/K15 共用的数学主体。调用方只决定输出 tile 的 (cRow,cCol) 映射；
// shared-memory 布局、寄存器累加和浮点运算顺序均由这里唯一维护。
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS,
          const int AS_STRIDE, const int A_XOR_MASK = 0,
          const int A_XOR_SHIFT = 0>
__device__ __forceinline__ void
runWarptilingCore(int M, int N, int K, float alpha,
                  const float *__restrict__ A,
                  const float *__restrict__ B, float beta,
                  float *__restrict__ C, const uint cRow, const uint cCol,
                  float *__restrict__ As, float *__restrict__ Bs) {
  // 当前 warp 在 block 输出块中的位置。
  const uint warpIdx = threadIdx.x / WARPSIZE;
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER;
  constexpr uint WSUBN = WN / WNITER;
  static_assert(NUM_THREADS > 0 && NUM_THREADS % WARPSIZE == 0);
  static_assert(BK > 0 && BK % 4 == 0);
  static_assert(BN > 0 && BN % 4 == 0);
  static_assert(TN > 0 && TN % 4 == 0);
  static_assert(AS_STRIDE >= BM);
  static_assert(WSUBM >= TM && WSUBM % TM == 0);
  static_assert(WSUBN >= TN && WSUBN % TN == 0);

  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

  A += cRow * BM * K;
  B += cCol * BN;
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  float threadResults[WMITER * TM * WNITER * TN] = {0.0};
  float regM[WMITER * TM] = {0.0};
  float regN[WNITER * TN] = {0.0};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    loadFromGmem<BM, BN, BK, AS_STRIDE, rowStrideA, rowStrideB, A_XOR_MASK,
                 A_XOR_SHIFT>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads();
    processFromSmem<BM, BN, BK, AS_STRIDE, WM, WN, WMITER, WNITER, WSUBM,
                    WSUBN, TM, TN, A_XOR_MASK, A_XOR_SHIFT>(
        regM, regN, threadResults, As, Bs, warpRow, warpCol, threadRowInWarp,
        threadColInWarp);
    A += BK;
    B += BK * N;
    __syncthreads();
  }

  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      float *C_interim =
          C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0];
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0] = tmp;
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
#ifndef SGEMM_K10_MIN_BLOCKS
// SM86 实测：把两个 block/SM 作为编译器资源目标时，ptxas 可使用更多寄存器
// 且不 spill。第二参数不是“运行时始终驻留两个 block”的保证；实际驻留还受
// 寄存器、shared memory、线程数和调度状态共同限制。0 恢复单参数版本。
#define SGEMM_K10_MIN_BLOCKS 2
#endif

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS,
          const int AS_PAD = 0>
#if SGEMM_K10_MIN_BLOCKS > 0
__global__ void __launch_bounds__(NUM_THREADS, SGEMM_K10_MIN_BLOCKS)
#else
__global__ void __launch_bounds__(NUM_THREADS)
#endif
    sgemmWarptiling(int M, int N, int K, float alpha,
                    const float *__restrict__ A,
                    const float *__restrict__ B, float beta,
                    float *__restrict__ C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;
  static_assert(AS_PAD >= 0);
  constexpr uint AS_STRIDE = BM + AS_PAD;
  __shared__ float As[AS_STRIDE * BK];
  __shared__ float Bs[BK * BN];
  wt::runWarptilingCore<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS,
                        AS_STRIDE>(M, N, K, alpha, A, B, beta, C, cRow, cCol,
                                   As, Bs);
}
