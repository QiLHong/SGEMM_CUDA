#pragma once

// K23 retains the K10 output ownership and arithmetic. The two layouts below
// permute different operands and are deliberately not combined.
namespace k23 {
constexpr uint BM = 128, BN = 128, BK = 16, WM = 64, WN = 64;
constexpr uint TM = 8, TN = 4, WNITER = 4, AS_STRIDE = BM + 4;

// V2: [k][n] -> [n/16][k][n%16]. Each 16-column fragment stays contiguous.
__device__ __forceinline__ constexpr uint bFragmentOffset(uint k, uint n) {
  return (n / 16) * (BK * 16) + k * 16 + n % 16;
}

// V3: [k][m+pad] -> [m/64][k][m%64]. One warp row occupies one slab.
__device__ __forceinline__ constexpr uint aWarpOffset(uint k, uint m) {
  return (m / WM) * (BK * WM) + k * WM + m % WM;
}

template <int VARIANT>
struct AIterator {
  const float *ptr;
  __device__ __forceinline__ AIterator(const float *base, uint warpRow,
                                      uint threadRow) {
    if constexpr (VARIANT == 3)
      ptr = base + warpRow * BK * WM + threadRow * TM;
    else
      ptr = base + warpRow * WM + threadRow * TM;
  }
  __device__ __forceinline__ void load(float *fragment) const {
#pragma unroll
    for (uint row = 0; row < TM; ++row)
      fragment[row] = ptr[row];
  }
  __device__ __forceinline__ void advance() {
    ptr += (VARIANT == 3 ? WM : AS_STRIDE);
  }
};

template <int VARIANT>
struct BIterator {
  const float *ptr;
  __device__ __forceinline__ BIterator(const float *base, uint warpCol,
                                      uint threadCol) {
    if constexpr (VARIANT == 2)
      ptr = base + warpCol * (WNITER * BK * 16) + threadCol * TN;
    else
      ptr = base + warpCol * WN + threadCol * TN;
  }
  __device__ __forceinline__ void load(float *fragment) const {
#pragma unroll
    for (uint group = 0; group < WNITER; ++group) {
#pragma unroll
      for (uint col = 0; col < TN; ++col) {
        const uint groupStride = VARIANT == 2 ? BK * 16 : 16;
        fragment[group * TN + col] = ptr[group * groupStride + col];
      }
    }
  }
  __device__ __forceinline__ void advance() {
    ptr += (VARIANT == 2 ? 16 : BN);
  }
};

template <int VARIANT>
__device__ __forceinline__ void loadTile(int N, int K,
                                        const float *__restrict__ A,
                                        const float *__restrict__ B,
                                        float *__restrict__ As,
                                        float *__restrict__ Bs) {
  static_assert(VARIANT == 2 || VARIANT == 3);
  constexpr uint ROW_STRIDE_A = (128 * 4) / BK;
  constexpr uint ROW_STRIDE_B = 128 / (BN / 4);
  const uint aRow = threadIdx.x / (BK / 4);
  const uint aCol4 = threadIdx.x % (BK / 4);
  const uint bRow = threadIdx.x / (BN / 4);
  const uint bCol4 = threadIdx.x % (BN / 4);

  // The source addresses, float4 load/store widths and per-thread ownership
  // match K10. Only the shared destination offset differs.
  for (uint offset = 0; offset + ROW_STRIDE_A <= BM;
       offset += ROW_STRIDE_A) {
    const float4 value = reinterpret_cast<const float4 *>(
        &A[(aRow + offset) * K + aCol4 * 4])[0];
    const uint m = aRow + offset;
    const uint k = aCol4 * 4;
    if constexpr (VARIANT == 3) {
      As[aWarpOffset(k + 0, m)] = value.x;
      As[aWarpOffset(k + 1, m)] = value.y;
      As[aWarpOffset(k + 2, m)] = value.z;
      As[aWarpOffset(k + 3, m)] = value.w;
    } else {
      As[(k + 0) * AS_STRIDE + m] = value.x;
      As[(k + 1) * AS_STRIDE + m] = value.y;
      As[(k + 2) * AS_STRIDE + m] = value.z;
      As[(k + 3) * AS_STRIDE + m] = value.w;
    }
  }

  for (uint offset = 0; offset + ROW_STRIDE_B <= BK;
       offset += ROW_STRIDE_B) {
    const uint k = bRow + offset;
    const uint n = bCol4 * 4;
    const float4 value =
        reinterpret_cast<const float4 *>(&B[k * N + n])[0];
    if constexpr (VARIANT == 2)
      reinterpret_cast<float4 *>(&Bs[bFragmentOffset(k, n)])[0] = value;
    else
      reinterpret_cast<float4 *>(&Bs[k * BN + n])[0] = value;
  }
}
} // namespace k23

// V1 is exactly sgemmK21Iterator, selected by the K23 host dispatcher.
template <int VARIANT>
__global__ void __launch_bounds__(128, 2)
    sgemmK23SharedFragmentIterator(int M, int N, int K, float alpha,
                                   const float *__restrict__ A,
                                   const float *__restrict__ B, float beta,
                                   float *__restrict__ C) {
  static_assert(VARIANT == 2 || VARIANT == 3);
  constexpr uint BM = k23::BM, BN = k23::BN, BK = k23::BK;
  constexpr uint WM = k23::WM, WN = k23::WN, WNITER = k23::WNITER;
  constexpr uint TM = k23::TM, TN = k23::TN;
  constexpr uint WMITER = 1, WSUBM = 64, WSUBN = 16;

  __shared__ float As[k23::AS_STRIDE * BK];
  __shared__ float Bs[BK * BN];
  const uint warp = threadIdx.x / WARPSIZE;
  const uint warpRow = warp / 2;
  const uint warpCol = warp % 2;
  const uint lane = threadIdx.x % WARPSIZE;
  const uint threadRow = lane / 4;
  const uint threadCol = lane % 4;
  const float *tileA = A + blockIdx.y * BM * K;
  const float *tileB = B + blockIdx.x * BN;
  float *warpC = C + (blockIdx.y * BM + warpRow * WM) * N +
                 blockIdx.x * BN + warpCol * WN;
  float accumulators[TM * WNITER * TN] = {0.0f};
  float regM[TM] = {0.0f};
  float regN[WNITER * TN] = {0.0f};

  for (uint bk = 0; bk < static_cast<uint>(K); bk += BK) {
    k23::loadTile<VARIANT>(N, K, tileA, tileB, As, Bs);
    __syncthreads();
    k23::AIterator<VARIANT> iteratorA(As, warpRow, threadRow);
    k23::BIterator<VARIANT> iteratorB(Bs, warpCol, threadCol);
    k21::computeTile<BK, WMITER, WNITER, TM, TN>(
        iteratorA, iteratorB, regM, regN, accumulators);
    tileA += BK;
    tileB += BK * N;
    __syncthreads();
  }

  k21::storeTile<WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
      N, alpha, beta, warpC, accumulators, threadRow, threadCol);
}
