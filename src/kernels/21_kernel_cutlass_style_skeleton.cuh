#pragma once

// K21 把“iterator 抽象”和“256-thread 双阶段流水”分成两个独立版本。
// V1 只替换 K10 的 shared->register 寻址方式；V2 才改变 CTA 与 staging。
namespace k21 {

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

__device__ __forceinline__ void waitAllAsyncGroups() {
  asm volatile("cp.async.wait_group 0;\n" ::);
}

template <const int AS_STRIDE, const int WMITER, const int WSUBM,
          const int TM>
struct SharedIteratorATransposed {
  const float *ptr;

  __device__ __forceinline__ explicit SharedIteratorATransposed(
      const float *base)
      : ptr(base) {}

  __device__ __forceinline__ void load(float *fragment) const {
#pragma unroll
    for (uint wSubRow = 0; wSubRow < WMITER; ++wSubRow) {
#pragma unroll
      for (uint i = 0; i < TM; ++i) {
        fragment[wSubRow * TM + i] = ptr[wSubRow * WSUBM + i];
      }
    }
  }

  __device__ __forceinline__ void advance() { ptr += AS_STRIDE; }
};

template <const int BN, const int WNITER, const int WSUBN, const int TN>
struct SharedIteratorB {
  const float *ptr;

  __device__ __forceinline__ explicit SharedIteratorB(const float *base)
      : ptr(base) {}

  __device__ __forceinline__ void load(float *fragment) const {
#pragma unroll
    for (uint wSubCol = 0; wSubCol < WNITER; ++wSubCol) {
#pragma unroll
      for (uint i = 0; i < TN; ++i) {
        fragment[wSubCol * TN + i] = ptr[wSubCol * WSUBN + i];
      }
    }
  }

  __device__ __forceinline__ void advance() { ptr += BN; }
};

// V2 的 A 直接按 global row-major 形状异步搬入 shared。这个 iterator
// 明确暴露 shared layout 与 fragment load 的关系；它不是 cuBLAS 私有布局的复刻。
template <const int BK, const int TM>
struct SharedIteratorARowMajor {
  const float *ptr;

  __device__ __forceinline__ explicit SharedIteratorARowMajor(
      const float *base)
      : ptr(base) {}

  __device__ __forceinline__ void load(float *fragment) const {
#pragma unroll
    for (uint i = 0; i < TM; ++i) {
      fragment[i] = ptr[i * BK];
    }
  }

  __device__ __forceinline__ void advance() { ++ptr; }
};

template <const int BK, const int WMITER, const int WNITER, const int TM,
          const int TN, class IteratorA, class IteratorB>
__device__ __forceinline__ void
computeTile(IteratorA iteratorA, IteratorB iteratorB, float *regM,
            float *regN, float *accumulators) {
#pragma unroll
  for (uint dot = 0; dot < BK; ++dot) {
    iteratorA.load(regM);
    iteratorB.load(regN);
    iteratorA.advance();
    iteratorB.advance();

#pragma unroll
    for (uint wSubRow = 0; wSubRow < WMITER; ++wSubRow) {
#pragma unroll
      for (uint wSubCol = 0; wSubCol < WNITER; ++wSubCol) {
#pragma unroll
        for (uint row = 0; row < TM; ++row) {
#pragma unroll
          for (uint col = 0; col < TN; ++col) {
            const uint result =
                (wSubRow * TM + row) * (WNITER * TN) + wSubCol * TN + col;
            accumulators[result] += regM[wSubRow * TM + row] *
                                    regN[wSubCol * TN + col];
          }
        }
      }
    }
  }
}

template <const int WMITER, const int WNITER, const int WSUBM,
          const int WSUBN, const int TM, const int TN>
__device__ __forceinline__ void
storeTile(int N, float alpha, float beta, float *warpC,
          const float *accumulators, const uint threadRow,
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

template <bool EDGE_M, const int BM, const int BN, const int BK>
__device__ __forceinline__ void
issueAsyncTile(int M, int N, int K, const float *__restrict__ A,
               const float *__restrict__ B, float *__restrict__ As,
               float *__restrict__ Bs, const uint blockRow) {
  constexpr uint A_VECTORS = (BM * BK) / 4;
  constexpr uint B_VECTORS = (BK * BN) / 4;
  static_assert(A_VECTORS % 256 == 0 && B_VECTORS % 256 == 0);

#pragma unroll
  for (uint linear = threadIdx.x; linear < A_VECTORS; linear += 256) {
    const uint row = linear / (BK / 4);
    const uint vectorCol = linear % (BK / 4);
    float4 *dst = reinterpret_cast<float4 *>(&As[row * BK + vectorCol * 4]);
    if constexpr (EDGE_M) {
      if (blockRow + row < static_cast<uint>(M)) {
        copy16Async(dst, &A[row * K + vectorCol * 4]);
      } else {
        dst[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
      }
    } else {
      copy16Async(dst, &A[row * K + vectorCol * 4]);
    }
  }

#pragma unroll
  for (uint linear = threadIdx.x; linear < B_VECTORS; linear += 256) {
    const uint row = linear / (BN / 4);
    const uint vectorCol = linear % (BN / 4);
    copy16Async(&Bs[row * BN + vectorCol * 4],
                &B[row * N + vectorCol * 4]);
  }
  commitAsyncGroup();
}

} // namespace k21

// K21 V1：K10 的所有 shape、ownership、浮点计算顺序与同步方式均不变。
// 唯一变量是 shared fragment 使用显式 iterator 的 load/advance 接口。
__global__ void __launch_bounds__(128, 2)
    sgemmK21Iterator(int M, int N, int K, float alpha,
                     const float *__restrict__ A,
                     const float *__restrict__ B, float beta,
                     float *__restrict__ C) {
  constexpr uint BM = 128, BN = 128, BK = 16;
  constexpr uint WM = 64, WN = 64, WNITER = 4;
  constexpr uint TM = 8, TN = 4, NUM_THREADS = 128;
  constexpr uint WMITER = 1, WSUBM = 64, WSUBN = 16;
  constexpr uint AS_STRIDE = BM + 4;
  constexpr uint ROW_STRIDE_A = (NUM_THREADS * 4) / BK;
  constexpr uint ROW_STRIDE_B = NUM_THREADS / (BN / 4);

  __shared__ float As[AS_STRIDE * BK];
  __shared__ float Bs[BK * BN];

  const uint warp = threadIdx.x / WARPSIZE;
  const uint warpRow = warp / 2;
  const uint warpCol = warp % 2;
  const uint lane = threadIdx.x % WARPSIZE;
  const uint threadCol = lane % 4;
  const uint threadRow = lane / 4;

  const float *tileA = A + blockIdx.y * BM * K;
  const float *tileB = B + blockIdx.x * BN;
  float *warpC = C + (blockIdx.y * BM + warpRow * WM) * N +
                 blockIdx.x * BN + warpCol * WN;

  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  float accumulators[TM * WNITER * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[WNITER * TN] = {0.0f};

  for (uint bk = 0; bk < static_cast<uint>(K); bk += BK) {
    wt::loadFromGmem<BM, BN, BK, AS_STRIDE, ROW_STRIDE_A, ROW_STRIDE_B>(
        N, K, tileA, tileB, As, Bs, innerRowA, innerColA, innerRowB,
        innerColB);
    __syncthreads();

    k21::SharedIteratorATransposed<AS_STRIDE, WMITER, WSUBM, TM> iteratorA(
        As + warpRow * WM + threadRow * TM);
    k21::SharedIteratorB<BN, WNITER, WSUBN, TN> iteratorB(
        Bs + warpCol * WN + threadCol * TN);
    k21::computeTile<BK, WMITER, WNITER, TM, TN>(
        iteratorA, iteratorB, regM, regN, accumulators);

    tileA += BK;
    tileB += BK * N;
    __syncthreads();
  }

  k21::storeTile<WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
      N, alpha, beta, warpC, accumulators, threadRow, threadCol);
}

// K21 V2：256x128 CTA、8 warp、两份 24 KiB shared stage。A/B 都通过
// cp.async 搬运，下一 K tile 与当前 shared->register->FFMA 主体重叠。
template <bool EDGE_M>
__global__ void __launch_bounds__(256, 1)
    sgemmK21Pipeline(int M, int N, int K, float alpha,
                     const float *__restrict__ A,
                     const float *__restrict__ B, float beta,
                     float *__restrict__ C) {
  constexpr uint BM = 256, BN = 128, BK = 16;
  constexpr uint WM = 64, WN = 64, WNITER = 4;
  constexpr uint TM = 8, TN = 4;
  constexpr uint WMITER = 1, WSUBM = 64, WSUBN = 16;

  __shared__ __align__(16) float As[2][BM * BK];
  __shared__ __align__(16) float Bs[2][BK * BN];

  const uint warp = threadIdx.x / WARPSIZE;
  const uint warpRow = warp / 2;
  const uint warpCol = warp % 2;
  const uint lane = threadIdx.x % WARPSIZE;
  const uint threadCol = lane % 4;
  const uint threadRow = lane / 4;
  const uint blockRow = blockIdx.y * BM;
  const uint outputRow = blockRow + warpRow * WM;

  const float *tileA = A + blockRow * K;
  const float *tileB = B + blockIdx.x * BN;
  float *warpC = C + outputRow * N + blockIdx.x * BN + warpCol * WN;

  float accumulators[TM * WNITER * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[WNITER * TN] = {0.0f};

  k21::issueAsyncTile<EDGE_M, BM, BN, BK>(M, N, K, tileA, tileB, As[0],
                                          Bs[0], blockRow);
  k21::waitAllAsyncGroups();
  __syncthreads();

  uint stage = 0;
  for (uint bk = 0; bk < static_cast<uint>(K); bk += BK) {
    const bool hasNext = bk + BK < static_cast<uint>(K);
    if (hasNext) {
      k21::issueAsyncTile<EDGE_M, BM, BN, BK>(
          M, N, K, tileA + BK, tileB + BK * N, As[stage ^ 1], Bs[stage ^ 1],
          blockRow);
    }

    k21::SharedIteratorARowMajor<BK, TM> iteratorA(
        As[stage] + (warpRow * WM + threadRow * TM) * BK);
    k21::SharedIteratorB<BN, WNITER, WSUBN, TN> iteratorB(
        Bs[stage] + warpCol * WN + threadCol * TN);
    k21::computeTile<BK, WMITER, WNITER, TM, TN>(
        iteratorA, iteratorB, regM, regN, accumulators);

    if (!hasNext)
      break;
    k21::waitAllAsyncGroups();
    __syncthreads();
    tileA += BK;
    tileB += BK * N;
    stage ^= 1;
  }

  if constexpr (EDGE_M) {
    if (outputRow >= static_cast<uint>(M))
      return;
  }
  k21::storeTile<WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
      N, alpha, beta, warpC, accumulators, threadRow, threadCol);
}
