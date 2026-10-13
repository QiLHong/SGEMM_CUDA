#pragma once
#include "30_kernel_mini_simt.cuh"

// K32: one producer warp + the unchanged four K30 consumer warps.
// Two controlled protocols, not a Hopper/Tensor Core implementation.
namespace k32 {
using S = k30::Shape<128,128,16,64,64,4,4,128>;
constexpr int Threads=160;
constexpr int ASize=k30::SharedLayoutA<S>::Size;
constexpr int StageSize=ASize+k30::SharedLayoutB<S>::Size;
constexpr int SharedBytes=2*StageSize*sizeof(float);

template<int Id> __device__ __forceinline__ void rendezvous() {
  // Non-.aligned PTX: different warp-uniform branches reach the same barrier.
  // Every phase has exactly 160 arrivals, in order 1 then 2. No early exits.
  asm volatile("barrier.sync %0, 160;" :: "n"(Id) : "memory");
}

__device__ __forceinline__ void copyTile(const float *a,const float *b,
    int N,int K,int lane,float *as,float *bs) {
  // 32 lanes cover the same float4 coordinates that K30 assigns to 128 lanes.
  // No register-staged full tile: each source vector is stored before reuse.
#pragma unroll
  for(int t=0;t<16;++t) {
    int m=lane/4+t*8, k=(lane%4)*4;
    float4 v=*reinterpret_cast<const float4*>(a+m*K+k);
    as[k30::SharedLayoutA<S>::offset(m,k+0)]=v.x;
    as[k30::SharedLayoutA<S>::offset(m,k+1)]=v.y;
    as[k30::SharedLayoutA<S>::offset(m,k+2)]=v.z;
    as[k30::SharedLayoutA<S>::offset(m,k+3)]=v.w;
  }
#pragma unroll
  for(int k=0;k<16;++k)
    *reinterpret_cast<float4*>(bs+k*128+lane*4)=
        *reinterpret_cast<const float4*>(b+k*N+lane*4);
}
} // namespace k32

// Overlap=false: copy(next) -> barrier 1 -> compute(current) -> barrier 2.
// Overlap=true:  copy(next) || compute(current) -> barrier 1 -> barrier 2.
// The intentionally redundant second barrier in Overlap=true keeps the dynamic
// barrier count matched. This is an attribution experiment, not a best kernel.
template<bool Overlap, bool SingleBarrier=false>
__global__ __launch_bounds__(k32::Threads)
void sgemmK32(int M,int N,int K,float alpha,const float *__restrict__ A,
    const float *__restrict__ B,float beta,float *__restrict__ C) {
  static_assert(!SingleBarrier || Overlap);
  using S=k32::S;
  extern __shared__ __align__(16) float storage[];
  int tid=threadIdx.x, blockM=blockIdx.y*128, blockN=blockIdx.x*128;
  bool producer=tid<32;
  const float *a=A+blockM*K, *b=B+blockN;
  if(producer) k32::copyTile(a,b,N,K,tid,storage,storage+k32::ASize);
  __syncthreads(); // publish tile 0 once, using barrier 0
  if(producer) {
#pragma unroll 1
    for(int tile=0;tile<K/16;++tile) {
      int future=tile+1;
      if(future<K/16) {
        float *dst=storage+(future%2)*k32::StageSize;
        k32::copyTile(a+future*16,b+future*16*N,N,K,tid,dst,dst+k32::ASize);
      }
      k32::rendezvous<1>();
      if constexpr(!SingleBarrier) k32::rendezvous<2>();
    }
  } else {
    k30::FragmentIterator<S> map(tid-32);
    float acc[S::Rows*S::Cols]={};
#pragma unroll 1
    for(int tile=0;tile<K/16;++tile) {
      if constexpr(!Overlap) k32::rendezvous<1>();
      float *current=storage+(tile%2)*k32::StageSize;
      k30::compute<S>(current,current+k32::ASize,map,acc);
      if constexpr(Overlap) k32::rendezvous<1>();
      if constexpr(!SingleBarrier) k32::rendezvous<2>();
    }
    k30::epilogue<S>(M,N,blockM,blockN,alpha,beta,C,map,acc);
  }
}
