// 【文件导读】K17：实验：只提前异步搬运下一块 B
// 给初学者：这是回答某个性能问题的实验，不是编号越大越快；先了解对照版本与唯一改动，再看复杂细节。
// 先知道：kernel 是在 GPU 上执行的函数；K17 是本工程实验编号，不代表性能排名。
// 这一版在做什么：A 只有一个 shared 槽，B 有两个；wait_group 等各线程复制完成，block 屏障让大家能安全读别人搬来的数据。
// 谁调用它：src/runner.cu 的编号 17 入口选择参数；头文件不能直接用 bash 或 python 运行。
// 输入/输出：行主序 FP32 矩阵 A[M,K]、B[K,N]、C[M,N]，计算 C=alpha*AB+beta*C。
// 边界：这是特定分块/对齐条件下的实验核，不是任意尺寸 GEMM；先读调用方约束再改 M/N/K。
// 阅读顺序：看线程如何分工 → 数据怎么搬 → 何时同步 → 怎么乘加 → 如何写回 C。
// 评价方法：正确性 PASS 只说明该次测试满足误差条件；是否更快还要看同口径多轮计时和 profiler。
// 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

#pragma once

// K17 只把 K10 的 B global->shared 搬运改成双 shared buffer 的 cp.async
// 流水。A 仍同步加载；寄存器 tile、warp/thread 映射和 FFMA 顺序均保持不变。
namespace wt_b_async {

__device__ __forceinline__ void copy16Async(void *sharedDst,
                                             const void *globalSrc) {
  const uint sharedAddress =
      static_cast<uint>(__cvta_generic_to_shared(sharedDst));
  // .ca 保留 K10 默认 global-load 的 L1+L2 缓存语义；.cg 的受控失败
  // 实验会绕过 L1，在本机使 L2 hit 下降且 DRAM 读取增加。
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::
                   "r"(sharedAddress),
               "l"(globalSrc));
}

__device__ __forceinline__ void commitAsyncGroup() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void waitAllAsyncGroups() {
  asm volatile("cp.async.wait_group 0;\n" ::);
}

template <const int BM, const int BK, const int AS_STRIDE,
          const int ROW_STRIDE_A>
__device__ __forceinline__ void
loadA(int K, const float *__restrict__ A, float *__restrict__ As,
      const uint innerRowA, const uint innerColA) {
  for (uint offset = 0; offset + ROW_STRIDE_A <= BM;
       offset += ROW_STRIDE_A) {
    const float4 value = reinterpret_cast<const float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * AS_STRIDE + innerRowA + offset] = value.x;
    As[(innerColA * 4 + 1) * AS_STRIDE + innerRowA + offset] = value.y;
    As[(innerColA * 4 + 2) * AS_STRIDE + innerRowA + offset] = value.z;
    As[(innerColA * 4 + 3) * AS_STRIDE + innerRowA + offset] = value.w;
  }
}

template <const int BN, const int BK, const int ROW_STRIDE_B>
__device__ __forceinline__ void
loadBAsync(int N, const float *__restrict__ B, float *__restrict__ Bs,
           const uint innerRowB, const uint innerColB) {
  for (uint offset = 0; offset + ROW_STRIDE_B <= BK;
       offset += ROW_STRIDE_B) {
    copy16Async(&Bs[(innerRowB + offset) * BN + innerColB * 4],
                &B[(innerRowB + offset) * N + innerColB * 4]);
  }
}

// K17 的运行时 ping-pong buffer 选择会让 ptxas 放弃 K10 原本的 BK 完全
// 展开。这里仅对 K17 明确恢复展开；K10 源码和 SASS 不受影响。
template <const int BM, const int BN, const int BK, const int AS_STRIDE,
          const int WM, const int WN, const int WMITER, const int WNITER,
          const int WSUBM, const int WSUBN, const int TM, const int TN>
__device__ __forceinline__ void
processFromSmemUnrolled(float *__restrict__ regM, float *__restrict__ regN,
                        float *__restrict__ threadResults,
                        const float *__restrict__ As,
                        const float *__restrict__ Bs, const uint warpRow,
                        const uint warpCol, const uint threadRowInWarp,
                        const uint threadColInWarp) {
#pragma unroll
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] =
            As[dotIdx * AS_STRIDE + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i];
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[dotIdx * BN + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i];
      }
    }
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
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
}

} // namespace wt_b_async

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS,
          const int AS_PAD = 0>
#if SGEMM_K10_MIN_BLOCKS > 0
__global__ void __launch_bounds__(NUM_THREADS, SGEMM_K10_MIN_BLOCKS)
#else
__global__ void __launch_bounds__(NUM_THREADS)
#endif
    sgemmWarptilingBAsync(int M, int N, int K, float alpha,
                          const float *__restrict__ A,
                          const float *__restrict__ B, float beta,
                          float *__restrict__ C) {
  constexpr uint AS_STRIDE = BM + AS_PAD;
  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER;
  constexpr uint WSUBN = WN / WNITER;
  constexpr uint ROW_STRIDE_A = (NUM_THREADS * 4) / BK;
  constexpr uint ROW_STRIDE_B = NUM_THREADS / (BN / 4);
  static_assert(NUM_THREADS > 0 && NUM_THREADS % WARPSIZE == 0);
  static_assert(BK > 0 && BK % 4 == 0);
  static_assert(BN > 0 && BN % 4 == 0);
  static_assert(TN > 0 && TN % 4 == 0);

  __shared__ __align__(16) float As[AS_STRIDE * BK];
  __shared__ __align__(16) float Bs[2][BK * BN];

  const uint warpIdx = threadIdx.x / WARPSIZE;
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);
  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += (blockIdx.y * BM + warpRow * WM) * N + blockIdx.x * BN +
       warpCol * WN;

  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  float threadResults[WMITER * TM * WNITER * TN] = {0.0};
  float regM[WMITER * TM] = {0.0};
  float regN[WNITER * TN] = {0.0};

  // 首个 K tile 必须就绪后才能计算。
  wt_b_async::loadA<BM, BK, AS_STRIDE, ROW_STRIDE_A>(
      K, A, As, innerRowA, innerColA);
  wt_b_async::loadBAsync<BN, BK, ROW_STRIDE_B>(
      N, B, Bs[0], innerRowB, innerColB);
  wt_b_async::commitAsyncGroup();
  wt_b_async::waitAllAsyncGroups();
  __syncthreads();

  uint currentBuffer = 0;
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    const bool hasNext = bkIdx + BK < K;
    if (hasNext) {
      // B(next) 的异步搬运与当前 tile 的 FFMA 重叠；不预取 A，也不增加
      // register fragment buffer。
      wt_b_async::loadBAsync<BN, BK, ROW_STRIDE_B>(
          N, B + BK * N, Bs[currentBuffer ^ 1], innerRowB, innerColB);
      wt_b_async::commitAsyncGroup();
    }

    wt_b_async::processFromSmemUnrolled<
        BM, BN, BK, AS_STRIDE, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
        regM, regN, threadResults, As, Bs[currentBuffer], warpRow, warpCol,
        threadRowInWarp, threadColInWarp);

    if (!hasNext) {
      break;
    }

    // 所有线程读完当前 A 后才能覆盖单缓冲 As。每线程 wait 后再经过
    // block barrier，才允许任意线程读取下一块 Bs。
    __syncthreads();
    A += BK;
    B += BK * N;
    wt_b_async::loadA<BM, BK, AS_STRIDE, ROW_STRIDE_A>(
        K, A, As, innerRowA, innerColA);
    wt_b_async::waitAllAsyncGroups();
    __syncthreads();
    currentBuffer ^= 1;
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
