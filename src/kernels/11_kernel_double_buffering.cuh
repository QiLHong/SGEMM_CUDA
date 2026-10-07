#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// 【学习补充】Kernel 11：两份共享缓存 + 两组线程交错搬运和计算。
// B0/B1 在这里表示缓冲槽 0/1，每槽都包含一份 As 和 Bs，不是只有矩阵 B。
// 两组线程最终都计算自己 warp 的输出；不是一组永远搬运、另一组永远计算。
// 这是原作者的实验性流水实现，不保证比 Kernel 10 快。
// 阅读时按两组线程画出时间线，尤其检查分支中的屏障配对；本中文分支不改同步逻辑。
namespace db {

// 半个 block 的线程合作加载一个完整 A/B 子块。
template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(const int N, const int K, float *A, float *B,
                             float *As, float *Bs, const int innerRowA,
                             const int innerColA, const int innerRowB,
                             const int innerColB) {
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    float4 tmp = reinterpret_cast<float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    // 写入共享内存时转置 A 子块。
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
  }

  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
  }
}

// 与 warp tiling 相同的计算核心；传入指针决定当前读取哪个共享缓冲槽。
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // 为本线程在 warp tile 中负责的所有区域加载寄存器。
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

    // warp 子块组合的矩阵乘累加。
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        // 每个组合中，本线程处理 TM×TN 个元素。
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

} // db 命名空间结束。

// WARPSIZE 由此前包含的 Kernel 10 头文件提供（见 src/kernels.cuh 的顺序）。
// 参数须满足 runner 的分块约束、半块线程搬运覆盖以及向量对齐条件。
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemmDoubleBuffering(const int M, const int N, const int K,
                         const float alpha, float *A, float *B, float beta,
                         float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // warp 在 block 输出块中的位置。
  const uint warpIdx = threadIdx.x / WARPSIZE; // 当前线程所属 warp。
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  // 根据线程覆盖面积，推导 warp 子块尺寸。
  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER; // 原例 64/2=32，实际由参数决定。
  constexpr uint WSUBN = WN / WNITER; // 原例 32/2=16。

  // lane 映射到 warp 子块中的线程小块坐标。
  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0,31]。
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // 原例 i%(16/4)。
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // 原例 i/4。

  // 两倍共享内存：槽 0 放当前块，槽 1 放下一块。
  __shared__ float As[2 * BM * BK];
  __shared__ float Bs[2 * BK * BN];

  // 按线程编号分为前半组/后半组；不要把它误认成全 block 共用的单一布尔值。
  bool doubleBufferIdx = threadIdx.x >= (NUM_THREADS / 2);

  // 定位 A/B 的 block 起点。
  A += cRow * BM * K;
  B += cCol * BN;
  // C 定位到当前 warp 的输出区域。
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // 搬运时按“只有半个 block 的线程”分配任务。
  // 对 NUM_THREADS/2 取余，使两组各自拥有从 0 开始的搬运编号。
  const uint innerRowA = (threadIdx.x % (NUM_THREADS / 2)) / (BK / 4);
  const uint innerColA = (threadIdx.x % (NUM_THREADS / 2)) % (BK / 4);
  constexpr uint rowStrideA = ((NUM_THREADS / 2) * 4) / BK;
  const uint innerRowB = (threadIdx.x % (NUM_THREADS / 2)) / (BN / 4);
  const uint innerColB = (threadIdx.x % (NUM_THREADS / 2)) % (BN / 4);
  constexpr uint rowStrideB = (NUM_THREADS / 2) / (BN / 4);

  // 每线程自己的累加结果；两个槽的贡献都累加到这份数组。
  float threadResults[WMITER * TM * WNITER * TN] = {0.0};
  // warp tile 范围内所需的寄存器缓存。
  float regM[WMITER * TM] = {0.0};
  float regN[WNITER * TN] = {0.0};

  if (doubleBufferIdx == 0) {
    // 流水预热：前半组先把第一个子块装入槽 B0。
    db::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
  }
  __syncthreads();

  // 每轮处理两段 BK。下面保留原实现的分支与同步结构。
  for (uint bkIdx = 0; bkIdx < K; bkIdx += 2 * BK) {
    if (doubleBufferIdx == 0) {
      // 前半组计算当前块 B0；后半组此时尝试加载 B1。
      db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                          TN>(regM, regN, threadResults, As, Bs, warpRow,
                              warpCol, threadRowInWarp, threadColInWarp);
      __syncthreads();

      // 前半组计算下一块 B1。
      if (bkIdx + BK < K) {
        db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN,
                            TM, TN>(regM, regN, threadResults, As + (BM * BK),
                                    Bs + (BK * BN), warpRow, warpCol,
                                    threadRowInWarp, threadColInWarp);
      }
      __syncthreads();

      // 前半组把再下一块写入已消费的 B0，供下一轮使用。
      if (bkIdx + 2 * BK < K) {
        db::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
            N, K, A + 2 * BK, B + 2 * BK * N, As, Bs, innerRowA, innerColA,
            innerRowB, innerColB);
      }
    } else {
      // 后半组先加载下一块 B1。
      if (bkIdx + BK < K) {
        db::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
            N, K, A + BK, B + BK * N, As + (BM * BK), Bs + (BK * BN), innerRowA,
            innerColA, innerRowB, innerColB);
      }
      __syncthreads();

      // 后半组计算自己 warp 在当前 B0 上的贡献。
      db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                          TN>(regM, regN, threadResults, As, Bs, warpRow,
                              warpCol, threadRowInWarp, threadColInWarp);
      __syncthreads();

      // 后半组再计算 B1；前半组此时尝试加载下一轮 B0。
      if (bkIdx + BK < K) {
        db::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN,
                            TM, TN>(regM, regN, threadResults, As + (BM * BK),
                                    Bs + (BK * BN), warpRow, warpCol,
                                    threadRowInWarp, threadColInWarp);
      }
    }

    // 【注释辨析】原注释写 BK；实际表达式一次推进 2*BK。
    A += 2 * BK;     // 向右移动 2*BK 列。
    B += 2 * BK * N; // 向下移动 2*BK 行。
    __syncthreads();
  }

  // 所有 K 子块完成后再写回；输出映射与 Kernel 10 一致。
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // 移动到当前 warp 子块。
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          // 读取旧 C 向量。
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0];
          // 寄存器中执行 GEMM 更新。
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
