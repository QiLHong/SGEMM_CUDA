// 【文件导读】K19：实验：把下一步 B 片段提前读入寄存器
// 给初学者：这是回答某个性能问题的实验，不是编号越大越快；先了解对照版本与唯一改动，再看复杂细节。
// 先知道：kernel 是在 GPU 上执行的函数；K19 是本工程实验编号，不代表性能排名。
// 这一版在做什么：Even/Odd 两组寄存器轮流保存 B(k) 与 B(k+1)，不是两份 shared tile，也没有改变全局内存加载。
// 谁调用它：src/runner.cu 的编号 19 入口选择参数；头文件不能直接用 bash 或 python 运行。
// 输入/输出：行主序 FP32 矩阵 A[M,K]、B[K,N]、C[M,N]，计算 C=alpha*AB+beta*C。
// 变体：V1 普通 B 预取；V2 四个指针提前推进；V3 每轮显式处理两个 k。源代码不同不保证 SASS 不同。
// 边界：这是特定分块/对齐条件下的实验核，不是任意尺寸 GEMM；先读调用方约束再改 M/N/K。
// 阅读顺序：看线程如何分工 → 数据怎么搬 → 何时同步 → 怎么乘加 → 如何写回 C。
// 评价方法：正确性 PASS 只说明该次测试满足误差条件；是否更快还要看同口径多轮计时和 profiler。
// 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

#pragma once

// K19 freezes all K10 tiling, global-memory loading and shared-memory layout.
// Only the order of B shared-memory loads relative to the current dot-product
// FFMA group changes. A remains single-buffered in registers.
namespace wt_prefetch {

template <const int BM, const int BK, const int AS_STRIDE, const int WM,
          const int WMITER, const int WSUBM, const int TM>
__device__ __forceinline__ void
loadAFragment(const float *__restrict__ As, float *__restrict__ regM,
              const uint dotIdx, const uint warpRow,
              const uint threadRowInWarp) {
#pragma unroll
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
#pragma unroll
    for (uint i = 0; i < TM; ++i) {
      regM[wSubRowIdx * TM + i] =
          As[dotIdx * AS_STRIDE + warpRow * WM + wSubRowIdx * WSUBM +
             threadRowInWarp * TM + i];
    }
  }
}

template <const int BN, const int WN, const int WNITER, const int WSUBN,
          const int TN>
__device__ __forceinline__ void
loadBFragment(const float *__restrict__ Bs, float *__restrict__ regN,
              const uint dotIdx, const uint warpCol,
              const uint threadColInWarp) {
#pragma unroll
  for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
#pragma unroll
    for (uint i = 0; i < TN; ++i) {
      regN[wSubColIdx * TN + i] =
          Bs[dotIdx * BN + warpCol * WN + wSubColIdx * WSUBN +
             threadColInWarp * TN + i];
    }
  }
}

// V2 only: four B fragment addresses are advanced before the current FFMA
// group. The fixed K19 shape is asserted by the caller.
__device__ __forceinline__ void
// 四个指针分别指向当前 k 的四组连续 B 元素，每组 float4。
// V2 想把地址推进提前；有四个指针也可能延长地址寄存器的存活时间，所以必须同时看编译资源。
loadBFragmentFromPointers(const float *__restrict__ b0,
                          const float *__restrict__ b1,
                          const float *__restrict__ b2,
                          const float *__restrict__ b3,
                          float *__restrict__ regN) {
  const float4 n0 = reinterpret_cast<const float4 *>(b0)[0];
  const float4 n1 = reinterpret_cast<const float4 *>(b1)[0];
  const float4 n2 = reinterpret_cast<const float4 *>(b2)[0];
  const float4 n3 = reinterpret_cast<const float4 *>(b3)[0];
  regN[0] = n0.x;
  regN[1] = n0.y;
  regN[2] = n0.z;
  regN[3] = n0.w;
  regN[4] = n1.x;
  regN[5] = n1.y;
  regN[6] = n1.z;
  regN[7] = n1.w;
  regN[8] = n2.x;
  regN[9] = n2.y;
  regN[10] = n2.z;
  regN[11] = n2.w;
  regN[12] = n3.x;
  regN[13] = n3.y;
  regN[14] = n3.z;
  regN[15] = n3.w;
}

template <const int WMITER, const int WNITER, const int TM, const int TN>
__device__ __forceinline__ void
computeFragment(const float *__restrict__ regM,
                const float *__restrict__ regN,
                float *__restrict__ threadResults) {
#pragma unroll
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
#pragma unroll
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
#pragma unroll
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
#pragma unroll
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN] +=
              regM[wSubRowIdx * TM + resIdxM] *
              regN[wSubColIdx * TN + resIdxN];
        }
      }
    }
  }
}

template <const int VARIANT, const int BM, const int BN, const int BK,
          const int AS_STRIDE, const int WM, const int WN, const int WMITER,
          const int WNITER, const int WSUBM, const int WSUBN, const int TM,
          const int TN>
__device__ __forceinline__ void
// Even/Odd 像两只轮流使用的托盘：读下一份 B 时，不覆盖当前乘加还要用的那份。
// 每个 k 的 A 仍现读现用；这里只改变 shared→register 的取数次序，不改变每个输出的 k 累加顺序。
process(float *__restrict__ regM, float *__restrict__ regNEven,
        float *__restrict__ regNOdd, float *__restrict__ threadResults,
        const float *__restrict__ As, const float *__restrict__ Bs,
        const uint warpRow, const uint warpCol, const uint threadRowInWarp,
        const uint threadColInWarp) {
  static_assert(VARIANT >= 1 && VARIANT <= 3);
  static_assert(BK == 16 && BN == 128 && WM == 64 && WN == 64);
  static_assert(WMITER == 1 && WNITER == 4 && TM == 8 && TN == 4);

  if constexpr (VARIANT == 2) {
    const uint laneBase = warpCol * WN + threadColInWarp * TN;
    const float *b0 = Bs + laneBase;
    const float *b1 = b0 + WSUBN;
    const float *b2 = b1 + WSUBN;
    const float *b3 = b2 + WSUBN;
    loadBFragmentFromPointers(b0, b1, b2, b3, regNEven);
#pragma unroll
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      loadAFragment<BM, BK, AS_STRIDE, WM, WMITER, WSUBM, TM>(
          As, regM, dotIdx, warpRow, threadRowInWarp);
      if (dotIdx + 1 < BK) {
        b0 += BN;
        b1 += BN;
        b2 += BN;
        b3 += BN;
        if ((dotIdx & 1) == 0) {
          loadBFragmentFromPointers(b0, b1, b2, b3, regNOdd);
        } else {
          loadBFragmentFromPointers(b0, b1, b2, b3, regNEven);
        }
      }
      if ((dotIdx & 1) == 0) {
        computeFragment<WMITER, WNITER, TM, TN>(regM, regNEven,
                                                threadResults);
      } else {
        computeFragment<WMITER, WNITER, TM, TN>(regM, regNOdd,
                                                threadResults);
      }
    }
  // 把两个 k 步骤写在同一循环里只是软件展开；ptxas 本来就可能做相同展开，最终 SASS 可能与 V1 完全一致。
  } else if constexpr (VARIANT == 3) {
    // Explicit two-dot software schedule: B(k+1) is loaded before compute(k),
    // then the consumed register bank is reused for B(k+2).
    loadBFragment<BN, WN, WNITER, WSUBN, TN>(
        Bs, regNEven, 0, warpCol, threadColInWarp);
#pragma unroll
    for (uint dotPair = 0; dotPair < BK; dotPair += 2) {
      loadAFragment<BM, BK, AS_STRIDE, WM, WMITER, WSUBM, TM>(
          As, regM, dotPair, warpRow, threadRowInWarp);
      loadBFragment<BN, WN, WNITER, WSUBN, TN>(
          Bs, regNOdd, dotPair + 1, warpCol, threadColInWarp);
      computeFragment<WMITER, WNITER, TM, TN>(regM, regNEven,
                                              threadResults);
      loadAFragment<BM, BK, AS_STRIDE, WM, WMITER, WSUBM, TM>(
          As, regM, dotPair + 1, warpRow, threadRowInWarp);
      if (dotPair + 2 < BK) {
        loadBFragment<BN, WN, WNITER, WSUBN, TN>(
            Bs, regNEven, dotPair + 2, warpCol, threadColInWarp);
      }
      computeFragment<WMITER, WNITER, TM, TN>(regM, regNOdd,
                                              threadResults);
    }
  } else {
    loadBFragment<BN, WN, WNITER, WSUBN, TN>(
        Bs, regNEven, 0, warpCol, threadColInWarp);
#pragma unroll
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      loadAFragment<BM, BK, AS_STRIDE, WM, WMITER, WSUBM, TM>(
          As, regM, dotIdx, warpRow, threadRowInWarp);
      if (dotIdx + 1 < BK) {
        if ((dotIdx & 1) == 0) {
          loadBFragment<BN, WN, WNITER, WSUBN, TN>(
              Bs, regNOdd, dotIdx + 1, warpCol, threadColInWarp);
        } else {
          loadBFragment<BN, WN, WNITER, WSUBN, TN>(
              Bs, regNEven, dotIdx + 1, warpCol, threadColInWarp);
        }
      }
      if ((dotIdx & 1) == 0) {
        computeFragment<WMITER, WNITER, TM, TN>(regM, regNEven,
                                                threadResults);
      } else {
        computeFragment<WMITER, WNITER, TM, TN>(regM, regNOdd,
                                                threadResults);
      }
    }
  }
}

template <const int VARIANT, const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER, const int TM,
          const int TN, const int NUM_THREADS, const int AS_STRIDE>
__device__ __forceinline__ void
run(int M, int N, int K, float alpha, const float *__restrict__ A,
    const float *__restrict__ B, float beta, float *__restrict__ C,
    const uint cRow, const uint cCol, float *__restrict__ As,
    float *__restrict__ Bs) {
  constexpr uint WMITER =
      (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER;
  constexpr uint WSUBN = WN / WNITER;
  static_assert(NUM_THREADS == 128 && BM == 128 && BN == 128 && BK == 16);
  static_assert(WM == 64 && WN == 64 && WNITER == 4 && TM == 8 && TN == 4);
  static_assert(AS_STRIDE == BM + 4 && WMITER == 1);

  const uint warpIdx = threadIdx.x / WARPSIZE;
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);
  const uint lane = threadIdx.x % WARPSIZE;
  const uint threadColInWarp = lane % (WSUBN / TN);
  const uint threadRowInWarp = lane / (WSUBN / TN);

  A += cRow * BM * K;
  B += cCol * BN;
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
  float regM[WMITER * TM] = {0.0f};
  float regNEven[WNITER * TN] = {0.0f};
  float regNOdd[WNITER * TN] = {0.0f};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    wt::loadFromGmem<BM, BN, BK, AS_STRIDE, rowStrideA, rowStrideB>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads();
    process<VARIANT, BM, BN, BK, AS_STRIDE, WM, WN, WMITER, WNITER, WSUBM,
            WSUBN, TM, TN>(regM, regNEven, regNOdd, threadResults, As, Bs,
                           warpRow, warpCol, threadRowInWarp,
                           threadColInWarp);
    A += BK;
    B += BK * N;
    __syncthreads();
  }

  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      float *CInterim =
          C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          float4 value = reinterpret_cast<float4 *>(
              &CInterim[(threadRowInWarp * TM + resIdxM) * N +
                        threadColInWarp * TN + resIdxN])[0];
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          value.x = alpha * threadResults[i + 0] + beta * value.x;
          value.y = alpha * threadResults[i + 1] + beta * value.y;
          value.z = alpha * threadResults[i + 2] + beta * value.z;
          value.w = alpha * threadResults[i + 3] + beta * value.w;
          reinterpret_cast<float4 *>(
              &CInterim[(threadRowInWarp * TM + resIdxM) * N +
                        threadColInWarp * TN + resIdxN])[0] = value;
        }
      }
    }
  }
}

} // namespace wt_prefetch

template <const int VARIANT, const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER, const int TM,
          const int TN, const int NUM_THREADS, const int AS_PAD>
#if SGEMM_K10_MIN_BLOCKS > 0
__global__ void __launch_bounds__(NUM_THREADS, SGEMM_K10_MIN_BLOCKS)
#else
__global__ void __launch_bounds__(NUM_THREADS)
#endif
    sgemmWarptilingFragmentPrefetch(int M, int N, int K, float alpha,
                                    const float *__restrict__ A,
                                    const float *__restrict__ B, float beta,
                                    float *__restrict__ C) {
  static_assert(VARIANT >= 1 && VARIANT <= 3);
  constexpr uint AS_STRIDE = BM + AS_PAD;
  __shared__ float As[AS_STRIDE * BK];
  __shared__ float Bs[BK * BN];
  wt_prefetch::run<VARIANT, BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS,
                   AS_STRIDE>(M, N, K, alpha, A, B, beta, C, blockIdx.y,
                              blockIdx.x, As, Bs);
}
