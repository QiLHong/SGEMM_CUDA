// 【文件导读】K30：当前研究基线：重新组织 fragment 的小型 SIMT 核
// 给初学者：这是回答某个性能问题的实验，不是编号越大越快；先了解对照版本与唯一改动，再看复杂细节。
// 先知道：kernel 是在 GPU 上执行的函数；K30 是本工程实验编号，不代表性能排名。
// 这一版在做什么：关键不是换寄存器名字，而是改变每个 lane 负责哪些行列，并让 shared 读取、累加和写回使用同一张映射表。
// 谁调用它：scripts/k30_audit.cu 独立测试程序；主程序 ./build/sgemm 不支持编号 30。
// 输入/输出：行主序 FP32 矩阵 A[M,K]、B[K,N]、C[M,N]，计算 C=alpha*AB+beta*C。
// 常用 Shape 下每线程 16 行×8 列=128 个累加值；V2 为两个 shared 槽，普通 LDG→寄存器→shared，不是 cp.async。
// 边界：这是特定分块/对齐条件下的实验核，不是任意尺寸 GEMM；先读调用方约束再改 M/N/K。
// 阅读顺序：看线程如何分工 → 数据怎么搬 → 何时同步 → 怎么乘加 → 如何写回 C。
// 评价方法：正确性 PASS 只说明该次测试满足误差条件；是否更快还要看同口径多轮计时和 profiler。
// 中文总索引：docs/zh/23-代码与脚本使用索引.md；脚本命令默认在仓库根目录执行。

#pragma once
#include <cuda_runtime.h>

// 独立的 SIMT 教学微内核；不依赖 CUTLASS，不改变 K10。
namespace k30 {
template<int BM, int BN, int BK, int WM, int WN, int TM, int TN, int THREADS>
// Shape 是编译时的尺寸说明书，不会在运行时分配一个矩阵。
// 例如 BM=BN=128、BK=16：一个 block 算 128×128 输出，每次沿 K 只推进 16。
// WM=WN=64：4 个 warp 按 2×2 排列；TM=TN=4 是组成 fragment 的小组宽度，
// 不是说每线程最终只有 4×4 个结果。Rows=16、Cols=8 才是完整的每线程结果范围。
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

// layout 只回答“逻辑元素放在 shared 的哪个下标”，不负责算乘法。
// A 的 offset(m,k)=k*(BM+4)+m，把同一 k 的多个 m 放连续；+4 是空位，不参与数学。
// B 的 offset(k,n)=k*BN+n 仍按行摆放。Size 的单位是 float 个数，不是字节。
template<class S> struct SharedLayoutA {
  static constexpr int Stride=S::M+4, Size=S::K*Stride;
  __device__ static int offset(int m,int k) { return k*Stride+m; }
};
template<class S> struct SharedLayoutB {
  static constexpr int Stride=S::N, Size=S::K*Stride;
  __device__ static int offset(int k,int n) { return k*Stride+n; }
};

// 这是分工表：给一个线程号 tid，返回它负责的输出坐标和本地累加器下标。
// warp=tid/32 是小组号，lane=tid%32 是组内号。四组沿 CTA 的 M/N 方向各铺两组。
// 以 warp 0 为例：lane 0 行={0..3,16..19,32..35,48..51}，列={0..3,32..35}；
// lane 1 取相同行组结构但起点改为 4；lane 2 行同 lane 0，列起点改为 4。
// 所以“重排 fragment”是改谁算哪些格子，不是在运行中把所有寄存器搬家。
template<class S> struct FragmentIterator {
  int row, col;
  __device__ explicit FragmentIterator(int tid) {
    int warp=tid/32, lane=tid%32;
    row=(warp/(S::N/S::WarpN))*S::WarpM + (2*(lane/16)+lane%2)*4;
    col=(warp%(S::N/S::WarpN))*S::WarpN + ((lane%16)/2)*4;
  }
  // i=0..15 分成四组，每组 4 行，组间隔 16；j=0..7 分成两组，每组 4 列，组间隔 32。
  // 举例 i=5 得 row+17；j=4 得 col+32；acc[5*8+4] 就存这两个坐标对应的部分和。
  // load 与 epilogue 都用这张表，保证“取到的 A/B”与“最终写回的 C”是同一组格子。
  __device__ int m(int i) const { return row+(i/4)*16+i%4; }
  __device__ int n(int j) const { return col+(j/4)*32+j%4; }
  __device__ static int reg(int i,int j) { return i*S::Cols+j; }
};

// Float4 source coordinates are established once. advance() only moves pointers.
// Missing M rows (128-size test with a 256-row CTA) are explicitly zero-filled.
// GlobalIterator 管搬运，不管输出归谁；搬运任务与计算任务可以是两张不同的分工表。
// 它记住全局地址与行跨度，避免在源代码中每轮从矩阵原点重新写完整公式。
// Fragment 中的 float4 a/b 是运输途中的临时包裹，不是最终 C 的 128 个累加器。
// 每个 float4 装连续 4 个 FP32，地址须满足 16 字节对齐；这里只屏蔽 M 尾部，没实现 N/K 通用尾块。
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
  // 第一步：大家各读一部分全局 A/B 到自己的临时寄存器。
  // Small 配置下 ar=tid/4、ak=(tid%4)*4；t=0..3 再覆盖相距 32 的 A 行。
  // 缺失的 M 行填 0，让 256 行 CTA 在 128 尺寸测试中也能完成协作，而不是让线程提前退出。
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
  // 第二步：把运输包裹放进共享仓库。A 的四个连续 k 分量拆开转置存放；B 保持 float4 连续写入。
  // 这里没有同步；调用者必须保证全块搬完以后，才让计算线程读取别人写下的数据。
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

// WarpIterator 像一个读数游标：起点由线程分工表决定，load 取当前 k 所需的片段。
// A 一次取 16 个数、B 一次取 8 个数；每四个连续，再跳到下一个行/列组。
// advance 推进到下一个 k，不是移动输出块；A/B 各自的 stride 与 shared 布局必须匹配。
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
// 数学主体：对当前 k，把 16 个 A 与 8 个 B 两两相乘，更新 128 个相互独立的输出部分和。
// 一个输出的累加顺序仍沿 k 前进；只是很多输出交错计算，不改变 C=alpha*AB+beta*C。
// #pragma unroll 请求编译器展开固定循环；最终生成几条 LDS/FFMA、用多少寄存器要看 SASS/ptxas。
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
// epilogue 是收尾写回：前面 acc 只积累 AB，这里才乘 alpha、加 beta*旧C。
// beta=0 时不用读旧 C；每次 j+=4 把同一行的四个连续结果一起写回。
// row<M 保护 M 尾部；列和 K 仍依赖调用方整除约束，不能据此认为支持任意矩形。
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
  // extern shared 的大小由发射 <<<grid,block,字节数>>> 的第三项提供，不在这里写死。
  // 一个 stage 放 ASize 个 A 再放 BSize 个 B；Small 的两槽总用户容量为 33280 字节。
  // 这是程序声明的容量，profiler 的 shared_mem_per_block 还可能包含工具/运行时统计的额外占用。
  extern __shared__ __align__(16) float storage[];
  constexpr int ASize=k30::SharedLayoutA<S>::Size;
  constexpr int StageSize=ASize+k30::SharedLayoutB<S>::Size;
  int blockM=blockIdx.y*S::M, blockN=blockIdx.x*S::N;
  k30::GlobalIterator<S> global(A,B,M,N,K,blockM,blockN);
  k30::FragmentIterator<S> map(threadIdx.x);
  // acc 是每线程的 128 个输出部分和，跨全部 K tile 保留；不是整个 CTA 的 C 矩阵。
  // 数组声明不保证一定全在物理寄存器里：编译器资源报告和 spill 检查才是依据。
  float acc[S::Rows*S::Cols]={};
  // 单槽对照：先搬完并同步，再计算，再同步等所有人读完，最后才能覆盖同一槽。
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
    // 多槽准备阶段：先填好 STAGES-1 份 shared 数据，留一槽给后续 tile。
    // STAGES=2 时只先填 tile 0；不要把源代码里的“预取”理解成硬件保证同时执行。
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
      // 两槽例子：算 tile 0 前先把 tile 1 读到 next；算完 0，再把 next 放进槽 1。
      // 下一轮算 1 时读 2，随后写回槽 0。源码给了重排空间，实际 LDG/FFMA 距离由编译器决定。
      int future=tile+STAGES-1;
      if(future<K/S::K) { global.load(next); global.advance(); }
      float *current=storage+(tile%STAGES)*StageSize;
      k30::compute<S>(current,current+ASize,map,acc);
      if(future<K/S::K) {
        float *dst=storage+(future%STAGES)*StageSize;
        global.store(next,dst,dst+ASize);
      }
      // 轮尾屏障有两层作用：发布刚写好的 future，同时确认 current 已被所有线程读完。
      // 即使本线程已经算完，也不能在大家越过屏障前抢着复用别人还在读取的旧槽。
      // Publish future stage and retire current stage. No thread can reuse
      // a previous stage before all readers have passed this block barrier.
      __syncthreads();
    }
  }
  k30::epilogue<S>(M,N,blockM,blockN,alpha,beta,C,map,acc);
}
