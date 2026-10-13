#pragma once
#include <cuda_runtime.h>

// 独立的 SIMT 教学微内核；不依赖 CUTLASS，不改变 K10。
namespace k30 {
template<int BM, int BN, int BK, int WM, int WN, int TM, int TN, int THREADS>
struct Shape {
  static constexpr int M=BM, N=BN, K=BK, Threads=THREADS;
  static constexpr int WarpM=WM, WarpN=WN, ThreadM=TM, ThreadN=TN;
  // 4x8 lane lattice, interleaved along M. Each lane owns 16 A x 8 B.
  static constexpr int Rows=WM/4, Cols=WN/8;
  static constexpr int ALoads=BM*BK/(4*THREADS), BLoads=BK*BN/(4*THREADS);
  static_assert(BM%WM==0 && BN%WN==0 && (BM/WM)*(BN/WN)*32==THREADS);
  static_assert(WM%(4*TM)==0 && WN%(8*TN)==0 && TM==4 && TN==4);
  static_assert(BK%4==0 && BM*BK%(4*THREADS)==0 && BK*BN%(4*THREADS)==0);
};

template<class S> struct SharedLayoutA {
  static constexpr int Stride=S::M+4, Size=S::K*Stride;
  __device__ static int offset(int m,int k) { return k*Stride+m; }
};
template<class S> struct SharedLayoutB {
  static constexpr int Stride=S::N, Size=S::K*Stride;
  __device__ static int offset(int k,int n) { return k*Stride+n; }
};

template<class S> struct FragmentIterator {
  int row, col;
  __device__ explicit FragmentIterator(int tid) {
    int warp=tid/32, lane=tid%32;
    row=(warp/(S::N/S::WarpN))*S::WarpM + (2*(lane/16)+lane%2)*4;
    col=(warp%(S::N/S::WarpN))*S::WarpN + ((lane%16)/2)*4;
  }
  __device__ int m(int i) const { return row+(i/4)*16+i%4; }
  __device__ int n(int j) const { return col+(j/4)*32+j%4; }
  __device__ static int reg(int i,int j) { return i*S::Cols+j; }
};

// Float4 source coordinates are established once. advance() only moves pointers.
// Missing M rows (128-size test with a 256-row CTA) are explicitly zero-filled.
template<class S> struct GlobalIterator {
  const float *a, *b;
  int strideA, strideB, mRemaining, ar, ak, bk, bn;
  struct Fragment { float4 a[S::ALoads], b[S::BLoads]; };
  __device__ GlobalIterator(const float *A,const float *B,int M,int N,int K,int blockM,int blockN)
      : strideA(K),strideB(N),mRemaining(M-blockM) {
    ar=threadIdx.x/(S::K/4); ak=(threadIdx.x%(S::K/4))*4;
    bk=threadIdx.x/(S::N/4); bn=(threadIdx.x%(S::N/4))*4;
    a=A+blockM*K+ak; b=B+blockN+bk*N+bn;
  }
  __device__ void load(Fragment &f) const {
#pragma unroll
    for(int t=0;t<S::ALoads;++t) {
      int r=ar+t*(S::Threads*4/S::K);
      f.a[t]=r<mRemaining ? *reinterpret_cast<const float4*>(a+r*strideA)
                         : make_float4(0,0,0,0);
    }
#pragma unroll
    for(int t=0;t<S::BLoads;++t)
      f.b[t]=*reinterpret_cast<const float4*>(b+t*(S::Threads*4/S::N)*strideB);
  }
  __device__ void store(const Fragment &f,float *as,float *bs) const {
#pragma unroll
    for(int t=0;t<S::ALoads;++t) {
      int r=ar+t*(S::Threads*4/S::K);
      as[SharedLayoutA<S>::offset(r,ak+0)]=f.a[t].x;
      as[SharedLayoutA<S>::offset(r,ak+1)]=f.a[t].y;
      as[SharedLayoutA<S>::offset(r,ak+2)]=f.a[t].z;
      as[SharedLayoutA<S>::offset(r,ak+3)]=f.a[t].w;
    }
#pragma unroll
    for(int t=0;t<S::BLoads;++t)
      *reinterpret_cast<float4*>(bs+SharedLayoutB<S>::offset(bk+t*(S::Threads*4/S::N),bn))=f.b[t];
  }
  __device__ void advance() { a+=S::K; b+=S::K*strideB; }
};

template<class S> struct WarpIteratorA {
  const float *ptr;
  __device__ WarpIteratorA(const float *base,const FragmentIterator<S> &f)
      : ptr(base+SharedLayoutA<S>::offset(f.row,0)) {}
  __device__ void load(float *fragment) const {
#pragma unroll
    for(int i=0;i<S::Rows;++i) fragment[i]=ptr[(i/4)*16+i%4];
  }
  __device__ void advance() { ptr+=SharedLayoutA<S>::Stride; }
};
template<class S> struct WarpIteratorB {
  const float *ptr;
  __device__ WarpIteratorB(const float *base,const FragmentIterator<S> &f)
      : ptr(base+SharedLayoutB<S>::offset(0,f.col)) {}
  __device__ void load(float *fragment) const {
#pragma unroll
    for(int j=0;j<S::Cols;++j) fragment[j]=ptr[(j/4)*32+j%4];
  }
  __device__ void advance() { ptr+=SharedLayoutB<S>::Stride; }
};

template<class S>
__device__ __forceinline__ void compute(const float *as,const float *bs,
    const FragmentIterator<S> &map,float *acc) {
  WarpIteratorA<S> a(as,map); WarpIteratorB<S> b(bs,map);
  float fa[S::Rows],fb[S::Cols];
#pragma unroll
  for(int k=0;k<S::K;++k) {
    a.load(fa); b.load(fb); a.advance(); b.advance();
#pragma unroll
    for(int i=0;i<S::Rows;++i)
#pragma unroll
      for(int j=0;j<S::Cols;++j)
        acc[FragmentIterator<S>::reg(i,j)]+=fa[i]*fb[j];
  }
}

template<class S>
__device__ void epilogue(int M,int N,int blockM,int blockN,float alpha,float beta,
    float *C,const FragmentIterator<S> &map,const float *acc) {
#pragma unroll
  for(int i=0;i<S::Rows;++i) {
    int row=blockM+map.m(i);
    if(row<M) {
#pragma unroll
      for(int j=0;j<S::Cols;j+=4) {
        float *dst=C+row*N+blockN+map.n(j);
        float4 old=beta==0 ? make_float4(0,0,0,0) : *reinterpret_cast<float4*>(dst);
        int r=FragmentIterator<S>::reg(i,j);
        float4 out=make_float4(alpha*acc[r]+beta*old.x,alpha*acc[r+1]+beta*old.y,
                              alpha*acc[r+2]+beta*old.z,alpha*acc[r+3]+beta*old.w);
        *reinterpret_cast<float4*>(dst)=out;
      }
    }
  }
}
} // namespace k30

// Host contract: M multiple of 128; N multiple of BN; K multiple of BK;
// aligned row-major FP32 matrices without aliasing. Not a general tail kernel.
template<class S, int STAGES>
__global__ __launch_bounds__(S::Threads, (S::Threads==128 ? 2 : 1))
void sgemmK30(int M,int N,int K,float alpha,const float *__restrict__ A,
    const float *__restrict__ B,float beta,float *__restrict__ C) {
  static_assert(STAGES==1 || STAGES==2 || STAGES==4);
  extern __shared__ __align__(16) float storage[];
  constexpr int ASize=k30::SharedLayoutA<S>::Size;
  constexpr int StageSize=ASize+k30::SharedLayoutB<S>::Size;
  int blockM=blockIdx.y*S::M, blockN=blockIdx.x*S::N;
  k30::GlobalIterator<S> global(A,B,M,N,K,blockM,blockN);
  k30::FragmentIterator<S> map(threadIdx.x);
  float acc[S::Rows*S::Cols]={};
  if constexpr(STAGES==1) {
    for(int tile=0;tile<K/S::K;++tile) {
      typename k30::GlobalIterator<S>::Fragment fragment;
      global.load(fragment); global.advance();
      global.store(fragment,storage,storage+ASize);
      __syncthreads();
      k30::compute<S>(storage,storage+ASize,map,acc);
      __syncthreads();
    }
  } else {
    // S-1 ready shared tiles + one register-staged incoming tile.
    // Software LDG->register lookahead, NOT cp.async or guaranteed overlap.
    // All threads participate; no producer/consumer warp specialization.
#pragma unroll
    for(int stage=0;stage<STAGES-1;++stage) {
      if(stage<K/S::K) {
        typename k30::GlobalIterator<S>::Fragment fragment;
        global.load(fragment); global.advance();
        float *dst=storage+stage*StageSize;
        global.store(fragment,dst,dst+ASize);
      }
    }
    __syncthreads();
    for(int tile=0;tile<K/S::K;++tile) {
      typename k30::GlobalIterator<S>::Fragment next;
      int future=tile+STAGES-1;
      if(future<K/S::K) { global.load(next); global.advance(); }
      float *current=storage+(tile%STAGES)*StageSize;
      k30::compute<S>(current,current+ASize,map,acc);
      if(future<K/S::K) {
        float *dst=storage+(future%STAGES)*StageSize;
        global.store(next,dst,dst+ASize);
      }
      // Publish future stage and retire current stage. No thread can reuse
      // a previous stage before all readers have passed this block barrier.
      __syncthreads();
    }
  }
  k30::epilogue<S>(M,N,blockM,blockN,alpha,beta,C,map,acc);
}
