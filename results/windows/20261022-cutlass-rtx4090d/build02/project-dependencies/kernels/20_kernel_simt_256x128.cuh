// 【文件导读】K20：实验：放大 CTA，并允许分批计算输出
// 给初学者：这是回答某个性能问题的实验，不是编号越大越快；先了解对照版本与唯一改动，再看复杂细节。
// 先知道：kernel 是在 GPU 上执行的函数；K20 是本工程实验编号，不代表性能排名。
// 这一版在做什么：CTA 就是 thread block；这里 256 线程覆盖 256×128。M_PHASES=2 每次只保留 64 个部分和，但要重走 K。
// 谁调用它：src/runner.cu 的编号 20 入口选择参数；头文件不能直接用 bash 或 python 运行。
// 输入/输出：行主序 FP32 矩阵 A[M,K]、B[K,N]、C[M,N]，计算 C=alpha*AB+beta*C。
// 边界：这是特定分块/对齐条件下的实验核，不是任意尺寸 GEMM；先读调用方约束再改 M/N/K。
// 阅读顺序：看线程如何分工 → 数据怎么搬 → 何时同步 → 怎么乘加 → 如何写回 C。
// 评价方法：正确性 PASS 只说明该次测试满足误差条件；是否更快还要看同口径多轮计时和 profiler。
// 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

#pragma once

// K20：用 256 个线程覆盖 256x128 CTA tile 的纯 FP32 SIMT 基线。
// 这个实验只放大 CTA/warp 资源规模；不引入 Tensor Core、cp.async 或双缓冲。
namespace simt256 {

template <bool EDGE_M, const int BM, const int BN, const int BK,
          const int AS_STRIDE, const int NUM_THREADS>
__device__ __forceinline__ void
loadTile(int M, int N, int K, const float *__restrict__ A,
         const float *__restrict__ B, float *__restrict__ As,
         float *__restrict__ Bs, const uint blockRow) {
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  static_assert((BM * BK) % (4 * NUM_THREADS) == 0);
  static_assert((BN * BK) % (4 * NUM_THREADS) == 0);

  for (uint offset = 0; offset < BM; offset += rowStrideA) {
    const uint localRow = innerRowA + offset;
    float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if constexpr (EDGE_M) {
      if (blockRow + localRow < static_cast<uint>(M)) {
        value = reinterpret_cast<const float4 *>(
            &A[localRow * K + innerColA * 4])[0];
      }
    } else {
      value = reinterpret_cast<const float4 *>(
          &A[localRow * K + innerColA * 4])[0];
    }
    As[(innerColA * 4 + 0) * AS_STRIDE + localRow] = value.x;
    As[(innerColA * 4 + 1) * AS_STRIDE + localRow] = value.y;
    As[(innerColA * 4 + 2) * AS_STRIDE + localRow] = value.z;
    As[(innerColA * 4 + 3) * AS_STRIDE + localRow] = value.w;
  }

  for (uint offset = 0; offset < BK; offset += rowStrideB) {
    reinterpret_cast<float4 *>(
        &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(
            &B[(innerRowB + offset) * N + innerColB * 4])[0];
  }
}

template <bool EDGE_M, const int M_PHASES, const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER, const int TM,
          const int TN, const int NUM_THREADS, const int AS_STRIDE>
__device__ __forceinline__ void
run(int M, int N, int K, float alpha, const float *__restrict__ A,
    const float *__restrict__ B, float beta, float *__restrict__ C,
    float *__restrict__ As, float *__restrict__ Bs) {
  constexpr uint NUM_WARPS = NUM_THREADS / WARPSIZE;
  constexpr uint WMITER =
      (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER;
  constexpr uint WSUBN = WN / WNITER;
  constexpr uint ACCUMULATORS = WMITER * TM * WNITER * TN;

  static_assert(NUM_THREADS == 256);
  static_assert(BM == 256 && BN == 128 && BK == 16);
  static_assert(NUM_THREADS % WARPSIZE == 0);
  static_assert(M_PHASES == 1 || M_PHASES == 2);
  static_assert((BM / (WM * M_PHASES)) * (BN / WN) == NUM_WARPS);
  static_assert(BM % (WM * M_PHASES) == 0 && BN % WN == 0);
  static_assert(WM % WMITER == 0 && WN % WNITER == 0);
  static_assert(WSUBM % TM == 0 && WSUBN % TN == 0);
  static_assert(ACCUMULATORS == 128 / M_PHASES);

  const uint warpIdx = threadIdx.x / WARPSIZE;
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);
  const uint laneId = threadIdx.x % WARPSIZE;
  const uint threadColInWarp = laneId % (WSUBN / TN);
  const uint threadRowInWarp = laneId / (WSUBN / TN);
  const uint blockRow = blockIdx.y * BM;

  const float *const blockA = A + blockRow * K;
  const float *const blockB = B + blockIdx.x * BN;

  // M_PHASES=2 是一个真实的 64-accumulator 方案：每阶段完整遍历 K，计算并
  // 写回互不重叠的 64x32 warp tile。它保持每个 C 元素的 K 累加顺序与 beta
  // 语义，但代价是 CTA 的 global/shared 搬运和 barrier 次数翻倍。
  for (uint phase = 0; phase < M_PHASES; ++phase) {
    const uint logicalWarpRow = warpRow * M_PHASES + phase;
    const uint outputRow = blockRow + logicalWarpRow * WM;
    float threadResults[ACCUMULATORS] = {0.0f};
    float regM[WMITER * TM] = {0.0f};
    float regN[WNITER * TN] = {0.0f};
    const float *tileA = blockA;
    const float *tileB = blockB;

    for (uint bkIdx = 0; bkIdx < static_cast<uint>(K); bkIdx += BK) {
      loadTile<EDGE_M, BM, BN, BK, AS_STRIDE, NUM_THREADS>(
          M, N, K, tileA, tileB, As, Bs, blockRow);
      __syncthreads();
      wt::processFromSmem<BM, BN, BK, AS_STRIDE, WM, WN, WMITER, WNITER,
                          WSUBM, WSUBN, TM, TN>(
          regM, regN, threadResults, As, Bs, logicalWarpRow, warpCol,
          threadRowInWarp, threadColInWarp);
      tileA += BK;
      tileB += BK * N;
      __syncthreads();
    }

    if constexpr (EDGE_M) {
      // 不能在 barrier 前退出；尾块中的无效 warp 必须完成所有协作加载。
      if (outputRow >= static_cast<uint>(M))
        continue;
    }
    float *warpC = C + outputRow * N + blockIdx.x * BN + warpCol * WN;
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        float *C_interim =
            warpC + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
            const uint row = threadRowInWarp * TM + resIdxM;
            const uint col = threadColInWarp * TN + resIdxN;
            float4 tmp =
                reinterpret_cast<float4 *>(&C_interim[row * N + col])[0];
            const uint resultIdx =
                (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                wSubColIdx * TN + resIdxN;
            tmp.x = alpha * threadResults[resultIdx + 0] + beta * tmp.x;
            tmp.y = alpha * threadResults[resultIdx + 1] + beta * tmp.y;
            tmp.z = alpha * threadResults[resultIdx + 2] + beta * tmp.z;
            tmp.w = alpha * threadResults[resultIdx + 3] + beta * tmp.w;
            reinterpret_cast<float4 *>(&C_interim[row * N + col])[0] = tmp;
          }
        }
      }
    }
  }
}

} // namespace simt256

template <bool EDGE_M, const int M_PHASES, const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER, const int TM,
          const int TN, const int NUM_THREADS, const int AS_PAD>
__global__ void __launch_bounds__(NUM_THREADS, 1)
    sgemmSimt256x128(int M, int N, int K, float alpha,
                     const float *__restrict__ A,
                     const float *__restrict__ B, float beta,
                     float *__restrict__ C) {
  constexpr uint AS_STRIDE = BM + AS_PAD;
  __shared__ float As[AS_STRIDE * BK];
  __shared__ float Bs[BK * BN];
  simt256::run<EDGE_M, M_PHASES, BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS,
               AS_STRIDE>(M, N, K, alpha, A, B, beta, C, As, Bs);
}
