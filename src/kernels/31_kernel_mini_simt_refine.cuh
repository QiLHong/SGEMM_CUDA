#pragma once
#include "30_kernel_mini_simt.cuh"

// K31 reuses K30's shape, ownership, shared layouts, arithmetic and epilogue.
// Each experiment changes one of: temporary lifetime, stage count, CTA size,
// or compiler register-budget annotation. K30 itself remains the saved control.
namespace k31 {
using Small = k30::Shape<128,128,16,64,64,4,4,128>;
using Large = k30::Shape<256,128,16,64,64,4,4,256>;

template<int Variant> struct Config {
  static_assert(Variant>=0 && Variant<=5);
  using S = Small;
  static constexpr int Stages = Variant==3 ? 3 : Variant==4 ? 4 : 2;
  static constexpr int Policy = Variant==1 ? 1 : Variant==2 ? 2 : 0;
};
template<> struct Config<5> {
  using S = Large;
  static constexpr int Stages=2, Policy=0;
};

// Policy 2: consume each float4 before loading the next vector. The physical
// global/shared coordinates and transfer count are identical to GlobalIterator.
// ptxas may still reschedule these operations; lexical scope is not a guarantee.
template<class S>
__device__ __forceinline__ void copyVectors(const k30::GlobalIterator<S> &g,
                                           float *as,float *bs) {
#pragma unroll
  for(int t=0;t<S::ALoads;++t) {
    int r=g.ar+t*(S::Threads*4/S::K);
    float4 value=r<g.mRemaining ? *reinterpret_cast<const float4*>(g.a+r*g.strideA)
                               : make_float4(0,0,0,0);
    as[k30::SharedLayoutA<S>::offset(r,g.ak+0)]=value.x;
    as[k30::SharedLayoutA<S>::offset(r,g.ak+1)]=value.y;
    as[k30::SharedLayoutA<S>::offset(r,g.ak+2)]=value.z;
    as[k30::SharedLayoutA<S>::offset(r,g.ak+3)]=value.w;
  }
#pragma unroll
  for(int t=0;t<S::BLoads;++t) {
    float4 value=*reinterpret_cast<const float4*>(g.b+t*(S::Threads*4/S::N)*g.strideB);
    *reinterpret_cast<float4*>(bs+k30::SharedLayoutB<S>::offset(
        g.bk+t*(S::Threads*4/S::N),g.bn))=value;
  }
}

template<class S,int Stages,int Policy>
__device__ __forceinline__ void run(int M,int N,int K,float alpha,
    const float *__restrict__ A,const float *__restrict__ B,float beta,
    float *__restrict__ C,float *storage) {
  static_assert(Stages>=2 && Stages<=4 && Policy>=0 && Policy<=2);
  constexpr int ASize=k30::SharedLayoutA<S>::Size;
  constexpr int StageSize=ASize+k30::SharedLayoutB<S>::Size;
  int blockM=blockIdx.y*S::M, blockN=blockIdx.x*S::N;
  k30::GlobalIterator<S> global(A,B,M,N,K,blockM,blockN);
  k30::FragmentIterator<S> map(threadIdx.x);
  float acc[S::Rows*S::Cols]={};
#pragma unroll
  for(int stage=0;stage<Stages-1;++stage) {
    if(stage<K/S::K) {
      typename k30::GlobalIterator<S>::Fragment fragment;
      global.load(fragment); global.advance();
      float *dst=storage+stage*StageSize;
      global.store(fragment,dst,dst+ASize);
    }
  }
  __syncthreads();
  for(int tile=0;tile<K/S::K;++tile) {
    int future=tile+Stages-1;
    if constexpr(Policy==0) {
      typename k30::GlobalIterator<S>::Fragment next;
      if(future<K/S::K) { global.load(next); global.advance(); }
      float *current=storage+(tile%Stages)*StageSize;
      k30::compute<S>(current,current+ASize,map,acc);
      if(future<K/S::K) {
        float *dst=storage+(future%Stages)*StageSize;
        global.store(next,dst,dst+ASize);
      }
    } else {
      float *current=storage+(tile%Stages)*StageSize;
      k30::compute<S>(current,current+ASize,map,acc);
      if(future<K/S::K) {
        float *dst=storage+(future%Stages)*StageSize;
        if constexpr(Policy==1) {
          // Late-load control: shorten next's source-level lifetime only.
          typename k30::GlobalIterator<S>::Fragment next;
          global.load(next); global.advance();
          global.store(next,dst,dst+ASize);
        } else {
          copyVectors<S>(global,dst,dst+ASize);
          global.advance();
        }
      }
    }
    // Same retirement/publication protocol for every variant, including 3-stage.
    __syncthreads();
  }
  k30::epilogue<S>(M,N,blockM,blockN,alpha,beta,C,map,acc);
}
} // namespace k31

template<int Variant>
__global__ __launch_bounds__(k31::Config<Variant>::S::Threads,
                            (k31::Config<Variant>::S::Threads==128 ? 2 : 1))
void sgemmK31(int M,int N,int K,float alpha,const float *__restrict__ A,
    const float *__restrict__ B,float beta,float *__restrict__ C) {
  extern __shared__ __align__(16) float storage[];
  k31::run<typename k31::Config<Variant>::S,k31::Config<Variant>::Stages,
           k31::Config<Variant>::Policy>(M,N,K,alpha,A,B,beta,C,storage);
}

// __maxnreg__ and __launch_bounds__ cannot be combined. These are separate
// annotation controls, not an assertion that ptxas will allocate exactly Limit.
// All five budgets still permit at most two 128-thread CTAs by registers.
template<int Limit>
__global__ __maxnreg__(Limit)
void sgemmK31Budget(int M,int N,int K,float alpha,const float *__restrict__ A,
    const float *__restrict__ B,float beta,float *__restrict__ C) {
  static_assert(Limit==220 || Limit==225 || Limit==230 || Limit==235 || Limit==240);
  extern __shared__ __align__(16) float storage[];
  k31::run<k31::Small,2,0>(M,N,K,alpha,A,B,beta,C,storage);
}
