// 【文件导读】K22：实验：BK=8 的两槽和四槽流水
// 给初学者：这是回答某个性能问题的实验，不是编号越大越快；先了解对照版本与唯一改动，再看复杂细节。
// 先知道：kernel 是在 GPU 上执行的函数；K22 是本工程实验编号，不代表性能排名。
// 这一版在做什么：stage 是共享缓存槽，不是 GPU 线程；环形下标重复使用槽，必须保证旧数据读完、新数据搬完后再交接。
// 谁调用它：src/runner.cu 的编号 22 入口选择参数；头文件不能直接用 bash 或 python 运行。
// 输入/输出：行主序 FP32 矩阵 A[M,K]、B[K,N]、C[M,N]，计算 C=alpha*AB+beta*C。
// V1 同步 load/barrier/compute/barrier；V2/V3 用 cp.async 的 2/4-stage 环，要求 K/BK 至少覆盖预填充深度。
// 边界：这是特定分块/对齐条件下的实验核，不是任意尺寸 GEMM；先读调用方约束再改 M/N/K。
// 阅读顺序：看线程如何分工 → 数据怎么搬 → 何时同步 → 怎么乘加 → 如何写回 C。
// 评价方法：正确性 PASS 只说明该次测试满足误差条件；是否更快还要看同口径多轮计时和 profiler。
// 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

#pragma once

// K22 冻结 K10 的 128x128 CTA、4-warp 映射、线程 fragment 和 accumulator
// ownership，只把 BK 从 16 改为 8，并逐步增加 shared-memory pipeline 深度。
namespace k22 {

__device__ __forceinline__ void copy4Async(void *sharedDst,
                                            const void *globalSrc) {
  const uint sharedAddress =
      static_cast<uint>(__cvta_generic_to_shared(sharedDst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::
                   "r"(sharedAddress),
               "l"(globalSrc));
}

__device__ __forceinline__ void copy16Async(void *sharedDst,
                                             const void *globalSrc) {
  const uint sharedAddress =
      static_cast<uint>(__cvta_generic_to_shared(sharedDst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::
                   "r"(sharedAddress),
               "l"(globalSrc));
}

__device__ __forceinline__ void commitAsyncGroup() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int PendingGroups>
__device__ __forceinline__ void waitAsyncGroups() {
  static_assert(PendingGroups >= 0 && PendingGroups <= 7);
  asm volatile("cp.async.wait_group %0;\n" :: "n"(PendingGroups));
}

__device__ __forceinline__ void waitAllAsyncGroups() {
  asm volatile("cp.async.wait_group 0;\n" ::);
}

// A 从 global row-major 以 4-byte 粒度直接写入 K10 的转置 shared layout
// [k][m+pad]。B 保持 [k][n]，可使用 16-byte copy。这样 async copy 不会迫使
// shared->register 路径改成 K21 V2 中产生 bank conflict 的 row-major A。
template <const int BM, const int BN, const int BK, const int AS_STRIDE>
__device__ __forceinline__ void
issueAsyncTile(int N, int K, const float *__restrict__ A,
               const float *__restrict__ B, float *__restrict__ As,
               float *__restrict__ Bs) {
  static_assert(BM * BK % 128 == 0);
  static_assert((BK * BN / 4) % 128 == 0);

#pragma unroll
  for (uint linear = threadIdx.x; linear < BM * BK; linear += 128) {
    const uint row = linear / BK;
    const uint k = linear % BK;
    copy4Async(&As[k * AS_STRIDE + row], &A[row * K + k]);
  }

#pragma unroll
  for (uint linear = threadIdx.x; linear < BK * BN / 4; linear += 128) {
    const uint k = linear / (BN / 4);
    const uint vectorCol = linear % (BN / 4);
    copy16Async(&Bs[k * BN + vectorCol * 4],
                &B[k * N + vectorCol * 4]);
  }
  commitAsyncGroup();
}

template <const int BM, const int BN, const int BK, const int AS_STRIDE,
          const int WM, const int WN, const int WNITER, const int TM,
          const int TN>
__device__ __forceinline__ void
computeStage(float *__restrict__ regM, float *__restrict__ regN,
             float *__restrict__ accumulators,
             const float *__restrict__ As, const float *__restrict__ Bs,
             const uint warpRow, const uint warpCol, const uint threadRow,
             const uint threadCol) {
  constexpr uint WMITER =
      (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER;
  constexpr uint WSUBN = WN / WNITER;
  wt::processFromSmem<BM, BN, BK, AS_STRIDE, WM, WN, WMITER, WNITER,
                      WSUBM, WSUBN, TM, TN>(
      regM, regN, accumulators, As, Bs, warpRow, warpCol, threadRow,
      threadCol);
}

template <const int WMITER, const int WNITER, const int WSUBM,
          const int WSUBN, const int TM, const int TN>
__device__ __forceinline__ void
storeTile(int N, float alpha, float beta, float *__restrict__ warpC,
          const float *__restrict__ accumulators, const uint threadRow,
          const uint threadCol) {
#pragma unroll
  for (uint wSubRow = 0; wSubRow < WMITER; ++wSubRow) {
#pragma unroll
    for (uint wSubCol = 0; wSubCol < WNITER; ++wSubCol) {
      float *subtile = warpC + wSubRow * WSUBM * N + wSubCol * WSUBN;
#pragma unroll
      for (uint row = 0; row < TM; ++row) {
#pragma unroll
        for (uint col = 0; col < TN; col += 4) {
          float4 value = reinterpret_cast<float4 *>(
              &subtile[(threadRow * TM + row) * N + threadCol * TN + col])[0];
          const uint result =
              (wSubRow * TM + row) * (WNITER * TN) + wSubCol * TN + col;
          value.x = alpha * accumulators[result + 0] + beta * value.x;
          value.y = alpha * accumulators[result + 1] + beta * value.y;
          value.z = alpha * accumulators[result + 2] + beta * value.z;
          value.w = alpha * accumulators[result + 3] + beta * value.w;
          reinterpret_cast<float4 *>(
              &subtile[(threadRow * TM + row) * N + threadCol * TN + col])[0] =
              value;
        }
      }
    }
  }
}

} // namespace k22

// V1：BK=8，两个 shared slot 轮换，但仍严格执行 load/barrier/compute/barrier。
// 它没有 load/compute overlap；两个 BK=8 slot 的总容量与 K10 的 BK=16 单槽相同。
__global__ void __launch_bounds__(128, 2)
    sgemmK22SyncPingPong(int M, int N, int K, float alpha,
                         const float *__restrict__ A,
                         const float *__restrict__ B, float beta,
                         float *__restrict__ C) {
  constexpr uint BM = 128, BN = 128, BK = 8, AS_STRIDE = BM + 4;
  constexpr uint WM = 64, WN = 64, WNITER = 4, TM = 8, TN = 4;
  constexpr uint WMITER = 1, WSUBM = 64, WSUBN = 16;
  constexpr uint ROW_STRIDE_A = (128 * 4) / BK;
  constexpr uint ROW_STRIDE_B = 128 / (BN / 4);

  __shared__ __align__(16) float As[2][AS_STRIDE * BK];
  __shared__ __align__(16) float Bs[2][BK * BN];

  const uint warp = threadIdx.x / WARPSIZE;
  const uint warpRow = warp / 2;
  const uint warpCol = warp % 2;
  const uint lane = threadIdx.x % WARPSIZE;
  const uint threadCol = lane % 4;
  const uint threadRow = lane / 4;
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  const float *tileA = A + blockIdx.y * BM * K;
  const float *tileB = B + blockIdx.x * BN;
  float *warpC = C + (blockIdx.y * BM + warpRow * WM) * N +
                 blockIdx.x * BN + warpCol * WN;

  float accumulators[TM * WNITER * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[WNITER * TN] = {0.0f};
  uint stage = 0;

  for (uint bk = 0; bk < static_cast<uint>(K); bk += BK) {
    wt::loadFromGmem<BM, BN, BK, AS_STRIDE, ROW_STRIDE_A, ROW_STRIDE_B>(
        N, K, tileA, tileB, As[stage], Bs[stage], innerRowA, innerColA,
        innerRowB, innerColB);
    __syncthreads();
    k22::computeStage<BM, BN, BK, AS_STRIDE, WM, WN, WNITER, TM, TN>(
        regM, regN, accumulators, As[stage], Bs[stage], warpRow, warpCol,
        threadRow, threadCol);
    __syncthreads();
    tileA += BK;
    tileB += BK * N;
    stage ^= 1;
  }

  k22::storeTile<WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
      N, alpha, beta, warpC, accumulators, threadRow, threadCol);
}

// V2/V3：相同 BK=8 和相同 K10 fragment ownership，仅 stage ring 深度不同。
// steady state 先等待最旧 group，再计算当前 stage；计算后复用该 slot 搬入下一 tile。
template <int STAGES>
__global__ void __launch_bounds__(128, 2)
    sgemmK22AsyncPipeline(int M, int N, int K, float alpha,
                          const float *__restrict__ A,
                          const float *__restrict__ B, float beta,
                          float *__restrict__ C) {
  static_assert(STAGES == 2 || STAGES == 4);
  constexpr uint BM = 128, BN = 128, BK = 8, AS_STRIDE = BM + 4;
  constexpr uint WM = 64, WN = 64, WNITER = 4, TM = 8, TN = 4;
  constexpr uint WMITER = 1, WSUBM = 64, WSUBN = 16;

  __shared__ __align__(16) float As[STAGES][AS_STRIDE * BK];
  __shared__ __align__(16) float Bs[STAGES][BK * BN];

  const uint warp = threadIdx.x / WARPSIZE;
  const uint warpRow = warp / 2;
  const uint warpCol = warp % 2;
  const uint lane = threadIdx.x % WARPSIZE;
  const uint threadCol = lane % 4;
  const uint threadRow = lane / 4;
  const uint blockRow = blockIdx.y * BM;
  const float *tileA = A + blockRow * K;
  const float *tileB = B + blockIdx.x * BN;
  float *warpC = C + (blockRow + warpRow * WM) * N + blockIdx.x * BN +
                 warpCol * WN;

  float accumulators[TM * WNITER * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[WNITER * TN] = {0.0f};
  const uint tileCount = static_cast<uint>(K) / BK;

#pragma unroll
  for (uint prefill = 0; prefill < STAGES; ++prefill) {
    k22::issueAsyncTile<BM, BN, BK, AS_STRIDE>(
        N, K, tileA + prefill * BK, tileB + prefill * BK * N, As[prefill],
        Bs[prefill]);
  }

  uint stage = 0;
  const uint steadyTiles = tileCount - STAGES;
  for (uint tile = 0; tile < steadyTiles; ++tile) {
    k22::waitAsyncGroups<STAGES - 1>();
    __syncthreads();
    k22::computeStage<BM, BN, BK, AS_STRIDE, WM, WN, WNITER, TM, TN>(
        regM, regN, accumulators, As[stage], Bs[stage], warpRow, warpCol,
        threadRow, threadCol);
    __syncthreads();
    k22::issueAsyncTile<BM, BN, BK, AS_STRIDE>(
        N, K, tileA + (tile + STAGES) * BK,
        tileB + (tile + STAGES) * BK * N, As[stage], Bs[stage]);
    stage = (stage + 1) % STAGES;
  }

  // 尾部不再提交新 group；一次 wait_all 后按 ring 顺序消费剩余 stage。
  k22::waitAllAsyncGroups();
  __syncthreads();
#pragma unroll
  for (uint tail = 0; tail < STAGES; ++tail) {
    k22::computeStage<BM, BN, BK, AS_STRIDE, WM, WN, WNITER, TM, TN>(
        regM, regN, accumulators, As[stage], Bs[stage], warpRow, warpCol,
        threadRow, threadCol);
    stage = (stage + 1) % STAGES;
  }

  k22::storeTile<WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
      N, alpha, beta, warpC, accumulators, threadRow, threadCol);
}
